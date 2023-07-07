# Copyright 2022 The Nerfstudio Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Field for compound nerf model, adds scene contraction and image embeddings to instant ngp
"""

import math
from dataclasses import dataclass, field
from typing import Optional, Type, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.parameter import Parameter
from torchtyping import TensorType
from typing_extensions import Literal

from nerfstudio.cameras.rays import RaySamples
from nerfstudio.field_components.embedding import Embedding
from nerfstudio.field_components.encodings import (
    NeRFEncoding,
    PeriodicVolumeEncoding,
    TensorVMEncoding,
    SHEncoding,
    HashEncoding,
)
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.spatial_distortions import SpatialDistortion
from nerfstudio.fields.base_field import Field, FieldConfig

try:
    import tinycudann as tcnn
except ImportError:
    # tinycudann module doesn't exist
    pass


class LaplaceDensity(nn.Module):  # alpha * Laplace(loc=0, scale=beta).cdf(-sdf)
    """Laplace density from VolSDF"""

    def __init__(self, init_val, beta_min=0.0001):
        super().__init__()
        self.register_parameter("beta_min", nn.Parameter(beta_min * torch.ones(1), requires_grad=False))
        self.register_parameter("beta", nn.Parameter(init_val * torch.ones(1), requires_grad=True))

    def forward(
        self, sdf: TensorType["bs":...], beta: Union[TensorType["bs":...], None] = None
    ) -> TensorType["bs":...]:
        """convert sdf value to density value with beta, if beta is missing, then use learable beta"""

        if beta is None:
            beta = self.get_beta()

        alpha = 1.0 / beta
        return alpha * (0.5 + 0.5 * sdf.sign() * torch.expm1(-sdf.abs() / beta))

    def get_beta(self):
        """return current beta value"""
        beta = self.beta.abs() + self.beta_min
        return beta


class SigmoidDensity(nn.Module):  # alpha * Laplace(loc=0, scale=beta).cdf(-sdf)
    """Sigmoid density from VolSDF"""

    def __init__(self, init_val, beta_min=0.0001):
        super().__init__()
        self.register_parameter("beta_min", nn.Parameter(beta_min * torch.ones(1), requires_grad=False))
        self.register_parameter("beta", nn.Parameter(init_val * torch.ones(1), requires_grad=True))

    def forward(
        self, sdf: TensorType["bs":...], beta: Union[TensorType["bs":...], None] = None
    ) -> TensorType["bs":...]:
        """convert sdf value to density value with beta, if beta is missing, then use learable beta"""

        if beta is None:
            beta = self.get_beta()

        alpha = 1.0 / beta

        # negtive sdf will have large density
        return alpha * torch.sigmoid(-sdf * alpha)

    def get_beta(self):
        """return current beta value"""
        beta = self.beta.abs() + self.beta_min
        return beta


class SingleVarianceNetwork(nn.Module):
    """Variance network in NeuS

    Args:
        nn (_type_): init value in NeuS variance network
    """

    def __init__(self, init_val):
        super(SingleVarianceNetwork, self).__init__()
        self.register_parameter("variance", nn.Parameter(init_val * torch.ones(1), requires_grad=True))

    def forward(self, x):
        """Returns current variance value"""
        return torch.ones([len(x), 1], device=x.device) * torch.exp(self.variance * 10.0)

    def get_variance(self):
        """return current variance value"""
        return torch.exp(self.variance * 10.0).clip(1e-6, 1e6)


@dataclass
class SDFFieldConfig(FieldConfig):
    """Nerfacto Model Config"""

    _target: Type = field(default_factory=lambda: SDFField)
    num_layers: int = 8
    """Number of layers for geometric network"""
    hidden_dim: int = 256
    """Number of hidden dimension of geometric network"""
    geo_feat_dim: int = 256
    """Dimension of geometric feature"""
    num_layers_color: int = 4
    """Number of layers for color network"""
    hidden_dim_color: int = 256
    """Number of hidden dimension of color network"""
    appearance_embedding_dim: int = 32
    """Dimension of appearance embedding"""
    use_appearance_embedding: bool = False
    """Dimension of appearance embedding"""
    bias: float = 0.8
    """sphere size of geometric initializaion"""
    geometric_init: bool = True
    """Whether to use geometric initialization"""
    inside_outside: bool = True
    """whether to revert signed distance value, set to True for indoor scene"""
    weight_norm: bool = True
    """Whether to use weight norm for linear laer"""
    use_grid_feature: bool = False
    """Whether to use multi-resolution feature grids"""
    divide_factor: float = 2.0
    """Normalization factor for multi-resolution grids"""
    beta_init: float = 0.1
    """Init learnable beta value for transformation of sdf to density"""
    encoding_type: Literal["hash", "periodic", "tensorf_vm"] = "hash"
    """feature grid encoding type"""
    position_encoding_max_degree: int = 6
    """positional encoding max degree"""
    use_diffuse_color: bool = False
    """whether to use diffuse color as in ref-nerf"""
    use_specular_tint: bool = False
    """whether to use specular tint as in ref-nerf"""
    use_reflections: bool = False
    """whether to use reflections as in ref-nerf"""
    use_n_dot_v: bool = False
    """whether to use n dot v as in ref-nerf"""
    rgb_padding: float = 0.001
    """Padding added to the RGB outputs"""
    off_axis: bool = False
    """whether to use off axis encoding from mipnerf360"""
    use_numerical_gradients: bool = False
    """whether to use numercial gradients"""
    num_levels: int = 16
    """number of levels for multi-resolution hash grids"""
    max_res: int = 2048
    """max resolution for multi-resolution hash grids"""
    base_res: int = 16
    """base resolution for multi-resolution hash grids"""
    log2_hashmap_size: int = 19
    """log2 hash map size for multi-resolution hash grids"""
    hash_features_per_level: int = 2
    """number of features per level for multi-resolution hash grids"""
    hash_smoothstep: bool = True
    """whether to use smoothstep for multi-resolution hash grids"""
    use_position_encoding: bool = True
    """whether to use positional encoding as input for geometric network"""


class SDFField(Field):
    """_summary_

    Args:
        Field (_type_): _description_
    """

    config: SDFFieldConfig

    def __init__(
        self,
        config: SDFFieldConfig,
        aabb,
        num_images: int,
        use_average_appearance_embedding: bool = False,
        spatial_distortion: Optional[SpatialDistortion] = None,
        poses: TensorType = None,
        scale: float = 1.0,
        hashgrid_len: float = 20.0,
    ) -> None:
        super().__init__()
        self.config = config

        # TODO do we need aabb here?
        self.aabb = Parameter(aabb, requires_grad=False)

        self.spatial_distortion = spatial_distortion
        self.num_images = num_images

        self.embedding_appearance = Embedding(self.num_images, self.config.appearance_embedding_dim)
        self.use_average_appearance_embedding = use_average_appearance_embedding
        self.use_grid_feature = self.config.use_grid_feature
        self.divide_factor = self.config.divide_factor

        self.num_levels = self.config.num_levels
        self.max_res = self.config.max_res 
        self.base_res = self.config.base_res 
        self.log2_hashmap_size = self.config.log2_hashmap_size 
        self.features_per_level = self.config.hash_features_per_level

        self.poses = poses
        self.scale = scale
        self.hashgrid_len = hashgrid_len * 1.3
        # self.grid_len = 20
        # self.grid_len /= self.scale
        # self.offset = torch.tensor([0, 5, 0]) / self.scale

        use_hash = True
        smoothstep = self.config.hash_smoothstep
        self.growth_factor = np.exp((np.log(self.max_res) - np.log(self.base_res)) / (self.num_levels - 1))

        if self.config.encoding_type == "hash":
            # Single Feature Encoding
            self.encoding = tcnn.Encoding(
                n_input_dims=3,
                encoding_config={
                    "otype": "HashGrid" if use_hash else "DenseGrid",
                    "n_levels": self.num_levels,
                    "n_features_per_level": self.features_per_level,
                    "log2_hashmap_size": self.log2_hashmap_size,
                    "base_resolution": self.base_res,
                    "per_level_scale": self.growth_factor,
                    "interpolation": "Smoothstep" if smoothstep else "Linear",
                },
            )

            # # MegaNerf Encoding
            # self.encoding = []
            # self.hash_aabb = torch.tensor([])
            # z = -1.3
            # while z < 1:
            #     self.encoding.append(
            #         tcnn.Encoding(
            #             n_input_dims=3,
            #             encoding_config={
            #                 "otype": "HashGrid" if use_hash else "DenseGrid",
            #                 "n_levels": self.num_levels,
            #                 "n_features_per_level": self.features_per_level,
            #                 "log2_hashmap_size": self.log2_hashmap_size,
            #                 "base_resolution": self.base_res,
            #                 "per_level_scale": self.growth_factor,
            #                 "interpolation": "Smoothstep" if smoothstep else "Linear",
            #             },
            #         )
            #     )
            #     new_aabb = torch.tensor([[[-self.hashgrid_len, -self.hashgrid_len, z],
            #                               [self.hashgrid_len, self.hashgrid_len, z+self.hashgrid_len]]])
            #     self.hash_aabb = torch.cat([self.hash_aabb, new_aabb])
            #     z += self.hashgrid_len
            # self.hash_aabb = self.hash_aabb.to('cuda:0')
            # self.offsets = (self.hash_aabb[:, 0, :] + self.hash_aabb[:, 1, :]) / 2
            # # print('aabbs', self.hash_aabb)
            # # test_points = torch.tensor([[0, 0, -1.0], [0, 0, -0.6], [0, 0, 0.7]]).cuda()
            # # mask = self.point_in_aabb(test_points)
            # # print('mask', mask)
            # # grid_idx = torch.nonzero(mask, as_tuple=True)[1]
            # # print('grid idx', grid_idx)
            # # trans = self.offsets[grid_idx]
            # # print('trans', trans)
            # # positions = ((test_points - trans) / self.hashgrid_len * 2 + 1) / 2.01
            # # print('pos', positions)
            # # exit(0)

            # LocalRF Encoding
            # scene AABB will not work, all rays should reset far according to hash grid AABBs.
            # self.encoding = []
            # self.hash_aabb = torch.tensor([])
            # for pose in self.poses:
            #     cam_pos = pose[:3, 3]
            #     if torch.any(self.point_in_aabb(cam_pos)):
            #        continue
            #     self.encoding.append(
            #         tcnn.Encoding(
            #             n_input_dims=3,
            #             encoding_config={
            #                 "otype": "HashGrid" if use_hash else "DenseGrid",
            #                 "n_levels": self.num_levels,
            #                 "n_features_per_level": self.features_per_level,
            #                 "log2_hashmap_size": self.log2_hashmap_size,
            #                 "base_resolution": self.base_res,
            #                 "per_level_scale": self.growth_factor,
            #                 "interpolation": "Smoothstep" if smoothstep else "Linear",
            #             },
            #         )
            #     )
            #     lo = cam_pos + self.offset - (self.grid_len / 2)
            #     hi = cam_pos + self.offset + (self.grid_len / 2)
            #     new_aabb = torch.stack([lo, hi])[None, ...]
            #     self.hash_aabb = torch.cat([self.hash_aabb, new_aabb])
            # self.hash_aabb = self.hash_aabb.to('cuda:0')
            # torch.save(self.hash_aabb, '/data4/hyzhou/exp/LocalRF/hash_aabb.pt')
            # torch.save(self.poses, '/data4/hyzhou/exp/LocalRF/poses.pt')

            self.hash_encoding_mask = torch.ones(
                self.num_levels * self.features_per_level,
                dtype=torch.float32,
            )

        elif self.config.encoding_type == "periodic":
            print("using periodic encoding")
            self.encoding = PeriodicVolumeEncoding(
                num_levels=self.num_levels,
                min_res=self.base_res,
                max_res=self.max_res,
                log2_hashmap_size=18,  # 64 ** 3 = 2^18
                features_per_level=self.features_per_level,
                smoothstep=smoothstep,
            )
        elif self.config.encoding_type == "tensorf_vm":
            print("using tensor vm")
            self.encoding = TensorVMEncoding(128, 24, smoothstep=smoothstep)

        # we concat inputs position ourselves
        self.position_encoding = NeRFEncoding(
            in_dim=3,
            num_frequencies=self.config.position_encoding_max_degree,
            min_freq_exp=0.0,
            max_freq_exp=self.config.position_encoding_max_degree - 1,
            include_input=False,
            off_axis=self.config.off_axis,
        )

        # self.direction_encoding = NeRFEncoding(
        #     in_dim=3, num_frequencies=4, min_freq_exp=0.0, max_freq_exp=3.0, include_input=True
        # )
        self.direction_encoding = SHEncoding()

        # TODO move it to field components
        # MLP with geometric initialization
        dims = [self.config.hidden_dim for _ in range(self.config.num_layers)]
        if not self.config.use_position_encoding:
            if isinstance(self.encoding, list):
                in_dim = self.encoding[0].n_output_dims
            else:
                in_dim = self.encoding.n_output_dims
        else:
            in_dim = 3 + self.position_encoding.get_out_dim() + self.encoding.n_output_dims

        dims = [in_dim] + dims + [1 + self.config.geo_feat_dim]
        self.num_layers = len(dims)
        # TODO check how to merge skip_in to config
        self.skip_in = [4]

        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]

            lin = nn.Linear(dims[l], out_dim)

            if self.config.geometric_init:
                if l == self.num_layers - 2:
                    if not self.config.inside_outside:
                        torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                        torch.nn.init.constant_(lin.bias, -0.01*self.config.bias)
                    else:
                        torch.nn.init.normal_(lin.weight, mean=-np.sqrt(0.05*np.pi) / np.sqrt(dims[l]), std=0.0001)
                        torch.nn.init.constant_(lin.bias, self.config.bias)
                elif l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                elif l in self.skip_in:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, -(dims[0] - 3) :], 0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            if self.config.weight_norm:
                lin = nn.utils.weight_norm(lin)
                # print("=======", lin.weight.shape)
            setattr(self, "glin" + str(l), lin)

        # self.geo_mlp = tcnn.Network(
        #     n_input_dims=32,
        #     n_output_dims=1 + self.config.geo_feat_dim,
        #     network_config={
        #         "otype": "FullyFusedMLP",
        #         "activation": "ReLU",
        #         "output_activation": "None",
        #         "n_neurons": self.config.hidden_dim,
        #         "n_hidden_layers": self.config.num_layers,
        #     },
        # )

        # laplace function for transform sdf to density from VolSDF
        self.laplace_density = LaplaceDensity(init_val=self.config.beta_init)
        # self.laplace_density = SigmoidDensity(init_val=self.config.beta_init)

        # TODO use different name for beta_init for config
        # deviation_network to compute alpha from sdf from NeuS
        self.deviation_network = SingleVarianceNetwork(init_val=self.config.beta_init)

        # diffuse and specular tint layer
        if self.config.use_diffuse_color:
            self.diffuse_color_pred = nn.Linear(self.config.geo_feat_dim, 3)
        if self.config.use_specular_tint:
            self.specular_tint_pred = nn.Linear(self.config.geo_feat_dim, 3)

        # view dependent color network
        dims = [self.config.hidden_dim_color for _ in range(self.config.num_layers_color)]
        if self.config.use_diffuse_color:
            in_dim = (
                self.direction_encoding.get_out_dim()
                + self.config.geo_feat_dim
                + self.embedding_appearance.get_out_dim()
            )
        else:
            # point, view_direction, normal, feature, embedding
            in_dim = (
                3
                + self.direction_encoding.get_out_dim()
                + 3
                + self.config.geo_feat_dim
                + self.embedding_appearance.get_out_dim()
            )
        if self.config.use_n_dot_v:
            in_dim += 1

        dims = [in_dim] + dims + [3]
        self.num_layers_color = len(dims)

        for l in range(0, self.num_layers_color - 1):
            out_dim = dims[l + 1]
            lin = nn.Linear(dims[l], out_dim)
            torch.nn.init.kaiming_uniform_(lin.weight.data)
            torch.nn.init.zeros_(lin.bias.data)

            if self.config.weight_norm:
                lin = nn.utils.weight_norm(lin)
            # print("=======", lin.weight.shape)
            setattr(self, "clin" + str(l), lin)

        self.softplus = nn.Softplus(beta=100)
        self.relu = nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

        self._cos_anneal_ratio = 1.0
        self.numerical_gradients_delta = 0.0001

        self.grads = {}

    def save_grad(self, name):
        def hook(grad):
            self.grads[name] = grad

        return hook

    def point_in_aabb(self, point):
        assert isinstance(self.hash_aabb, TensorType)
        if self.hash_aabb.shape[0] == 0:
            return torch.tensor([False])
        if point.dim() == 1:
            lo_mask = torch.all(point > self.hash_aabb[:, 0, :], dim=-1)
            hi_mask = torch.all(point <= self.hash_aabb[:, 1, :], dim=-1)
        elif point.dim() == 2:
            lo_mask = torch.all(point[:, None, :] > self.hash_aabb[None, :, 0, :], dim=-1)
            hi_mask = torch.all(point[:, None, :] <= self.hash_aabb[None, :, 1, :], dim=-1)
        else:
            print(point.shape, self.hash_aabb.shape)
            raise NotImplementedError
        return lo_mask & hi_mask

    def set_cos_anneal_ratio(self, anneal: float) -> None:
        """Set the anneal value for the proposal network."""
        self._cos_anneal_ratio = anneal

    def update_mask(self, level: int):
        self.hash_encoding_mask[:] = 1.0
        self.hash_encoding_mask[level * self.features_per_level:] = 0
        
    def forward_geonetwork(self, inputs):
        """forward the geonetwork"""
        if self.use_grid_feature:
            #TODO normalize inputs depending on the whether we model the background or not
            # positions = (inputs + 2.0) / 4.0
            if isinstance(self.encoding, list):
                mask = self.point_in_aabb(inputs)
                grid_idx = torch.nonzero(mask, as_tuple=True)[1]
                # if mask.shape[0] != grid_idx.shape[0]:
                #     print(mask.shape, grid_idx.shape)
                #     mask_sum = torch.sum(mask, dim=-1)
                #     abnormal = torch.nonzero(mask_sum == 0, as_tuple=True)[0]
                #
                #     print(inputs[abnormal])
                #     n = torch.linalg.norm(inputs[abnormal], ord=float('inf'), dim=-1)
                #     print(torch.max(n))
                #     exit(0)
                trans = self.offsets[grid_idx]
                positions = ((inputs - trans) / self.hashgrid_len * 2 + 1) / 2.01
                feature = torch.zeros((inputs.shape[0], self.encoding[0].n_output_dims), device=inputs.device)
                for idx, encoder in enumerate(self.encoding):
                    sample_idx = torch.nonzero(grid_idx == idx, as_tuple=True)[0]
                    feature[sample_idx] = encoder(positions[sample_idx]).float()
                # feature.register_hook(self.save_grad('hash_feat_grad'))
            else:
                # assert torch.all((-2 < inputs) & (inputs < 2))
                positions = (inputs + 1.2) / 2.4

                feature = self.encoding(positions)
            # mask feature
            feature = feature * self.hash_encoding_mask.to(feature.device)
        else:
            feature = torch.zeros_like(inputs[:, :1].repeat(1, self.encoding.n_output_dims))


        pe = self.position_encoding(inputs)
        if not self.config.use_position_encoding:
            inputs = feature.float()
        else:
            inputs = torch.cat((inputs, pe, feature), dim=-1)

        x = inputs

        # # pe = self.position_encoding(inputs)
        # #
        # # inputs = torch.cat((inputs, pe, feature), dim=-1)
        # #
        # # x = inputs
        # x = feature.float()
        # # x = torch.cat((inputs, feature), dim=-1)
        # # x = self.geo_mlp(x)

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "glin" + str(l))

            if l in self.skip_in:
                x = torch.cat([x, inputs], 1) / np.sqrt(2)

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.softplus(x)

        # # NeUDF
        # x = self.softplus(x)

        return x

    def get_sdf(self, ray_samples: RaySamples):
        """predict the sdf value for ray samples"""
        positions = ray_samples.frustums.get_start_positions()
        positions_flat = positions.view(-1, 3)
        h = self.forward_geonetwork(positions_flat).view(*ray_samples.frustums.shape, -1)
        sdf, _ = torch.split(h, [1, self.config.geo_feat_dim], dim=-1)
        return sdf

    def set_numerical_gradients_delta(self, delta: float) -> None:
        """Set the delta value for numerical gradient."""
        self.numerical_gradients_delta = delta

    def gradient(self, x, skip_spatial_distortion=False, return_sdf=False):
        """compute the gradient of the ray"""
        if self.spatial_distortion is not None and not skip_spatial_distortion:
            x = self.spatial_distortion(x)

        # compute gradient in contracted space
        if self.config.use_numerical_gradients:
            # https://github.com/bennyguo/instant-nsr-pl/blob/main/models/geometry.py#L173
            delta = self.numerical_gradients_delta
            points = torch.stack(
                [
                    x + torch.as_tensor([delta, 0.0, 0.0]).to(x),
                    x + torch.as_tensor([-delta, 0.0, 0.0]).to(x),
                    x + torch.as_tensor([0.0, delta, 0.0]).to(x),
                    x + torch.as_tensor([0.0, -delta, 0.0]).to(x),
                    x + torch.as_tensor([0.0, 0.0, delta]).to(x),
                    x + torch.as_tensor([0.0, 0.0, -delta]).to(x),
                ],
                dim=0,
            )
            # n = torch.linalg.norm(points, ord=float('inf'), dim=-1)
            # print('ckpt2', torch.max(n))

            points_sdf = self.forward_geonetwork(points.view(-1, 3))[..., 0].view(6, *x.shape[:-1])
            gradients = torch.stack(
                [
                    0.5 * (points_sdf[0] - points_sdf[1]) / delta,
                    0.5 * (points_sdf[2] - points_sdf[3]) / delta,
                    0.5 * (points_sdf[4] - points_sdf[5]) / delta,
                ],
                dim=-1,
            )
        else:
            x.requires_grad_(True)

            y = self.forward_geonetwork(x)[:, :1]
            d_output = torch.ones_like(y, requires_grad=False, device=y.device)
            gradients = torch.autograd.grad(
                outputs=y, inputs=x, grad_outputs=d_output, create_graph=True, retain_graph=True, only_inputs=True
            )[0]
        if not return_sdf:
            return gradients
        else:
            return gradients, points_sdf

    def get_density(self, ray_samples: RaySamples):
        """Computes and returns the densities."""
        positions = ray_samples.frustums.get_start_positions()
        positions_flat = positions.view(-1, 3)
        h = self.forward_geonetwork(positions_flat).view(*ray_samples.frustums.shape, -1)
        sdf, geo_feature = torch.split(h, [1, self.config.geo_feat_dim], dim=-1)
        density = self.laplace_density(sdf)
        return density, geo_feature

    def get_alpha(self, ray_samples: RaySamples, sdf=None, gradients=None):
        """compute alpha from sdf as in NeuS"""
        if sdf is None or gradients is None:
            inputs = ray_samples.frustums.get_start_positions()
            inputs.requires_grad_(True)
            with torch.enable_grad():
                h = self.forward_geonetwork(inputs)
                sdf, _ = torch.split(h, [1, self.config.geo_feat_dim], dim=-1)
            d_output = torch.ones_like(sdf, requires_grad=False, device=sdf.device)
            gradients = torch.autograd.grad(
                outputs=sdf,
                inputs=inputs,
                grad_outputs=d_output,
                create_graph=True,
                retain_graph=True,
                only_inputs=True,
            )[0]

        inv_s = self.deviation_network.get_variance()  # Single parameter

        true_cos = (ray_samples.frustums.directions * gradients).sum(-1, keepdim=True)

        # anneal as NeuS
        cos_anneal_ratio = self._cos_anneal_ratio

        # "cos_anneal_ratio" grows from 0 to 1 in the beginning training iterations. The anneal strategy below makes
        # the cos value "not dead" at the beginning training iterations, for better convergence.
        iter_cos = -(
            F.relu(-true_cos * 0.5 + 0.5) * (1.0 - cos_anneal_ratio) + F.relu(-true_cos) * cos_anneal_ratio
        )  # always non-positive

        # Estimate signed distances at section points
        estimated_next_sdf = sdf + iter_cos * ray_samples.deltas * 0.5
        estimated_prev_sdf = sdf - iter_cos * ray_samples.deltas * 0.5

        # Neus
        prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_s)
        next_cdf = torch.sigmoid(estimated_next_sdf * inv_s)

        # # NeUDF
        # prev_cdf = (estimated_prev_sdf * inv_s / (1 + estimated_prev_sdf * inv_s)).clip(0., 1e6)
        # next_cdf = (estimated_next_sdf * inv_s / (1 + estimated_next_sdf * inv_s)).clip(0., 1e6)

        p = prev_cdf - next_cdf
        c = prev_cdf

        alpha = ((p + 1e-5) / (c + 1e-5)).clip(0.0, 1.0)

        # HF-NeuS
        # # sigma
        # cdf = torch.sigmoid(sdf * inv_s)
        # e = inv_s * (1 - cdf) * (-iter_cos) * ray_samples.deltas
        # alpha = (1 - torch.exp(-e)).clip(0.0, 1.0)

        return alpha

    def get_occupancy(self, sdf):
        """compute occupancy as in UniSurf"""
        occupancy = self.sigmoid(-10.0 * sdf)
        return occupancy

    def get_colors(self, points, directions, gradients, geo_features, camera_indices):
        """compute colors"""

        # diffuse color and specular tint
        if self.config.use_diffuse_color:
            raw_rgb_diffuse = self.diffuse_color_pred(geo_features.view(-1, self.config.geo_feat_dim))
        if self.config.use_specular_tint:
            tint = self.sigmoid(self.specular_tint_pred(geo_features.view(-1, self.config.geo_feat_dim)))

        normals = F.normalize(gradients, p=2, dim=-1)

        if self.config.use_reflections:
            # https://github.com/google-research/multinerf/blob/5d4c82831a9b94a87efada2eee6a993d530c4226/internal/ref_utils.py#L22
            refdirs = 2.0 * torch.sum(normals * -directions, axis=-1, keepdims=True) * normals + directions
            d = self.direction_encoding(refdirs)
        else:
            d = self.direction_encoding(directions)

        # appearance
        if self.training:
            embedded_appearance = self.embedding_appearance(camera_indices)
            # set it to zero if don't use it
            if not self.config.use_appearance_embedding:
                embedded_appearance = torch.zeros_like(embedded_appearance)
        else:
            if self.use_average_appearance_embedding:
                embedded_appearance = torch.ones(
                    (*directions.shape[:-1], self.config.appearance_embedding_dim), device=directions.device
                ) * self.embedding_appearance.mean(dim=0)
            else:
                embedded_appearance = torch.zeros(
                    (*directions.shape[:-1], self.config.appearance_embedding_dim), device=directions.device
                )
        if self.config.use_diffuse_color:
            h = [
                d,
                geo_features.view(-1, self.config.geo_feat_dim),
                embedded_appearance.view(-1, self.config.appearance_embedding_dim),
            ]
        else:
            h = [
                points,
                d,
                gradients,
                geo_features.view(-1, self.config.geo_feat_dim),
                embedded_appearance.view(-1, self.config.appearance_embedding_dim),
            ]

        if self.config.use_n_dot_v:
            n_dot_v = torch.sum(normals * directions, dim=-1, keepdims=True)
            h.append(n_dot_v)

        h = torch.cat(h, dim=-1)

        for l in range(0, self.num_layers_color - 1):
            lin = getattr(self, "clin" + str(l))

            h = lin(h)

            if l < self.num_layers_color - 2:
                h = self.relu(h)

        rgb = self.sigmoid(h)

        if self.config.use_diffuse_color:
            # Initialize linear diffuse color around 0.25, so that the combined
            # linear color is initialized around 0.5.
            diffuse_linear = self.sigmoid(raw_rgb_diffuse - math.log(3.0))
            if self.config.use_specular_tint:
                specular_linear = tint * rgb
            else:
                specular_linear = 0.5 * rgb

            # TODO linear to srgb?
            # Combine specular and diffuse components and tone map to sRGB.
            rgb = torch.clamp(specular_linear + diffuse_linear, 0.0, 1.0)

        # Apply padding, mapping color to [-rgb_padding, 1+rgb_padding].
        rgb = rgb * (1 + 2 * self.config.rgb_padding) - self.config.rgb_padding

        return rgb

    def get_outputs(self, ray_samples: RaySamples, return_alphas=False, return_occupancy=False, sky_mask=None):
        """compute output of ray samples"""
        if ray_samples.camera_indices is None:
            raise AttributeError("Camera indices are not provided.")

        outputs = {}

        camera_indices = ray_samples.camera_indices.squeeze()

        inputs = ray_samples.frustums.get_start_positions()
        inputs = inputs.view(-1, 3)

        directions = ray_samples.frustums.directions
        directions_flat = directions.reshape(-1, 3)

        if self.spatial_distortion is not None:
            inputs = self.spatial_distortion(inputs)
        points_norm = inputs.norm(dim=-1)
        # compute gradient in constracted space
        inputs.requires_grad_(True)
        # n = torch.linalg.norm(inputs, ord=float('inf'), dim=-1)
        # print('ckpt1', torch.max(n))
        with torch.enable_grad():
            h = self.forward_geonetwork(inputs)
            sdf, geo_feature = torch.split(h, [1, self.config.geo_feat_dim], dim=-1)

        if self.config.use_numerical_gradients:
            gradients, sampled_sdf = self.gradient(
                inputs,
                skip_spatial_distortion=True,
                return_sdf=True,
            )
            sampled_sdf = sampled_sdf.view(-1, *ray_samples.frustums.directions.shape[:-1]).permute(1, 2, 0).contiguous()
        else:
            d_output = torch.ones_like(sdf, requires_grad=False, device=sdf.device)
            gradients = torch.autograd.grad(
                outputs=sdf,
                inputs=inputs,
                grad_outputs=d_output,
                create_graph=True,
                retain_graph=True,
                only_inputs=True,
            )[0]
            sampled_sdf = None

        # sky_mask = sky_mask[:, None].repeat(1, ray_samples.shape[1]).view(-1)
        rgb = self.get_colors(inputs, directions_flat, gradients, geo_feature, camera_indices)

        density = self.laplace_density(sdf)

        rgb = rgb.view(*ray_samples.frustums.directions.shape[:-1], -1)
        sdf = sdf.view(*ray_samples.frustums.directions.shape[:-1], -1)
        density = density.view(*ray_samples.frustums.directions.shape[:-1], -1)
        gradients = gradients.view(*ray_samples.frustums.directions.shape[:-1], -1)
        normals = F.normalize(gradients, p=2, dim=-1)
        points_norm = points_norm.view(*ray_samples.frustums.directions.shape[:-1], -1)
        
        outputs.update(
            {
                FieldHeadNames.RGB: rgb,
                FieldHeadNames.DENSITY: density,
                FieldHeadNames.SDF: sdf,
                FieldHeadNames.NORMAL: normals,
                FieldHeadNames.GRADIENT: gradients,
                "points_norm": points_norm,
                "sampled_sdf": sampled_sdf,
            }
        )

        if return_alphas:
            # TODO use mid point sdf for NeuS
            alphas = self.get_alpha(ray_samples, sdf, gradients)
            outputs.update({FieldHeadNames.ALPHA: alphas})

        if return_occupancy:
            occupancy = self.get_occupancy(sdf)
            outputs.update({FieldHeadNames.OCCUPANCY: occupancy})

        return outputs

    def forward(self, ray_samples: RaySamples, return_alphas=False, return_occupancy=False, sky_mask=None):
        """Evaluates the field at points along the ray.

        Args:
            ray_samples: Samples to evaluate field on.
        """
        field_outputs = self.get_outputs(ray_samples,
                                         return_alphas=return_alphas,
                                         return_occupancy=return_occupancy,
                                         sky_mask=sky_mask)
        return field_outputs

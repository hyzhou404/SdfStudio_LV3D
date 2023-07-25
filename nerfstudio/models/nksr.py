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
NeRF implementation that combines many recent advancements.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Type

import numpy as np
import torch
from torch.nn import Parameter
from torchmetrics import PeakSignalNoiseRatio
from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from typing_extensions import Literal

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.engine.callbacks import (
    TrainingCallback,
    TrainingCallbackAttributes,
    TrainingCallbackLocation,
)
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.spatial_distortions import SceneContraction
from nerfstudio.fields.density_fields import HashMLPDensityField
from nerfstudio.fields.nerfacto_field import TCNNNerfactoField
from nerfstudio.model_components.losses import (
    MSELoss,
    distortion_loss,
    interlevel_loss,
    orientation_loss,
    pred_normal_loss,
    monosdf_normal_loss
)
from nerfstudio.model_components.ray_samplers import sphere_tracing
from nerfstudio.model_components.renderers import (
    AccumulationRenderer,
    DepthRenderer,
    NormalsRenderer,
    RGBRenderer,
)
from nerfstudio.model_components.scene_colliders import NearFarCollider
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.utils import colormaps
from nerfstudio.utils.colors import get_color
import random
import open3d as o3d
import nksr
import tinycudann as tcnn
from nerfstudio.field_components.activations import trunc_exp


@dataclass
class NKSRModelConfig(ModelConfig):
    """Nerfacto Model Config"""

    _target: Type = field(default_factory=lambda: NKSRModel)
    use_sensors: bool = False
    """Whether to use sensor or already have normals"""
    near_plane: float = 0.05
    """How far along the ray to start sampling."""
    far_plane: float = 1000.0
    """How far along the ray to stop sampling."""
    background_color: Literal["random", "last_sample", "white", "black"] = "white"
    """Whether to randomize the background color."""
    num_levels: int = 16
    """Number of levels of the hashmap for the base mlp."""
    base_res: int = 16
    """Maximum resolution of the hashmap for the base mlp."""
    log2_hashmap_size: int = 19
    """Size of the hashmap for the base mlp"""
    features_per_level: int = 2
    """feature length"""
    growth_factor: int = 2
    """growth factor"""
    orientation_loss_mult: float = 0.0001
    """Orientation loss multipier on computed noramls."""
    pred_normal_loss_mult: float = 0.001
    """Predicted normal loss multiplier."""
    use_average_appearance_embedding: bool = True
    """Whether to use average appearance embedding or zeros for inference."""
    use_single_jitter: bool = True
    """Whether use single jitter or not for the proposal networks."""
    predict_normals: bool = False
    """Whether to predict normals or not."""
    compute_normals: bool = False
    """Whether to compute normals by density gradients or not"""


class NKSRModel(Model):
    """Nerfacto model

    Args:
        config: Nerfacto configuration to instantiate model
    """

    config: NKSRModelConfig

    def populate_modules(self):
        """Set the fields and modules."""
        super().populate_modules()

        scene_contraction = SceneContraction(order=float("inf"))

        device = torch.device('cuda:0')

        if not self.config.use_sensors:
            points = np.load('/data4/datasets/DTU/snowman/points.npy')
            normals = np.load('/data4/datasets/DTU/snowman/normals.npy')
            downsampling_idx = np.random.choice(points.shape[0], 100000)
            points, normals = points[downsampling_idx], normals[downsampling_idx]
            self.points = torch.from_numpy(points).to(device).float()
            self.normals = torch.from_numpy(normals).to(device).float()

            with torch.no_grad():
                reconstructor = nksr.Reconstructor(device)
                self.nksr_field = reconstructor.reconstruct(self.points, self.normals, detail_level=0.9)
                self.scale = self.nksr_field.scale
                mesh = self.nksr_field.extract_dual_mesh(mise_iter=1)
                from pycg import vis
                vis.to_file(vis.mesh(mesh.v, mesh.f), "/data4/hyzhou/exp/snowman/snowman_test_mesh.ply")

        else:
            points = np.load('/data4/hyzhou/data/kitti_neus/3353_f60_aabb/lidar_point.npy')
            locs = np.load('/data4/hyzhou/data/kitti_neus/3353_f60_aabb/lidar_loc.npy')
            # downsampling_idx = np.random.choice(points.shape[0], 100000)
            # points, normals = points[downsampling_idx], normals[downsampling_idx]
            self.points = torch.from_numpy(points).to(device).float()
            self.locs = torch.from_numpy(locs).to(device).float()

            with torch.no_grad():
                reconstructor = nksr.Reconstructor(device)
                self.nksr_field = reconstructor.reconstruct(
                    self.points, sensor=self.locs, detail_level=0.9,
                    preprocess_fn=nksr.get_estimate_normal_preprocess_fn(64, 85.0))
                self.scale = self.nksr_field.scale
                # mesh = self.nksr_field.extract_dual_mesh(mise_iter=1)
                # from pycg import vis
                # vis.to_file(vis.mesh(mesh.v, mesh.f), "/data4/hyzhou/exp/kitti_nksr/kitti_test_mesh.ply")

        scene = None
        flattened_grids = []
        for d in range(self.nksr_field.svh.depth):
            f_grid = nksr.ext.meshing.build_flattened_grid(
                self.nksr_field.svh.grids[d]._grid,
                self.nksr_field.svh.grids[d - 1]._grid if d > 0 else None,
                d != self.nksr_field.svh.depth - 1
            )
            flattened_grids.append(f_grid)
        dual_grid = nksr.ext.meshing.build_joint_dual_grid(flattened_grids)
        active_ijk = dual_grid.active_grid_coords()
        base_coords = dual_grid.grid_to_world(active_ijk.float())
        base_scale = dual_grid.voxel_size()
        for coord in base_coords:
            box = o3d.geometry.TriangleMesh().create_box(width=base_scale, height=base_scale, depth=base_scale)
            box = box.translate(coord.detach().cpu().numpy())
            if scene:
                scene += box
            else:
                scene = box

        o3d.io.write_triangle_mesh(f'/data4/hyzhou/exp/snowman/cubes_dual.ply', scene)

        # for idx, f_grid in enumerate(flattened_grids):
        #     active_ijk = f_grid.active_grid_coords()
        #     base_coords = f_grid.grid_to_world(active_ijk.float())
        #     base_scale = f_grid.voxel_size()
        #
        #     for coord in base_coords:
        #         box = o3d.geometry.TriangleMesh().create_box(width=base_scale, height=base_scale, depth=base_scale)
        #         box = box.translate(coord.detach().cpu().numpy())
        #         if scene:
        #             scene += box
        #         else:
        #             scene = box
        #
        #     o3d.io.write_triangle_mesh(f'/data4/hyzhou/exp/snowman/cubes_{str(idx)}.ply', scene)

        # active_ijk = self.nksr_field.svh.grids[0].active_grid_coords()
        # scene = None
        # for d in range(1):
        #     base_coords = self.nksr_field.svh.grids[d].grid_to_world(active_ijk.float())
        #     base_scale = self.nksr_field.svh.grids[d].voxel_size()
        #     print(base_coords.shape)
        #     for coord in base_coords:
        #         box = o3d.geometry.TriangleMesh().create_box(width=base_scale, height=base_scale, depth=base_scale)
        #         box = box.translate(coord.detach().cpu().numpy())
        #         if scene:
        #             scene += box
        #         else:
        #             scene = box
        # o3d.io.write_triangle_mesh('/data4/hyzhou/exp/snowman/cubes.ply', scene)

        exit(0)

        # for k, v in nksr_field.features.items():
        #     print(k, v.shape)

        # Encoding
        self.direction_encoding = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "SphericalHarmonics",
                "degree": 4,
            },
        )

        # Fields
        self.feat_field = tcnn.NetworkWithInputEncoding(
            n_input_dims=3,
            n_output_dims=1+64,
            encoding_config={
                "otype": "HashGrid",
                "n_levels": self.config.num_levels,
                "n_features_per_level": self.config.features_per_level,
                "log2_hashmap_size": self.config.log2_hashmap_size,
                "base_resolution": self.config.base_res,
                "per_level_scale": self.config.growth_factor,
                "interpolation": "Smoothstep"
            },
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": 64,
                "n_hidden_layers": 2,
            },
        )

        self.color_field = tcnn.Network(
            n_input_dims=self.direction_encoding.n_output_dims + 64,
            n_output_dims=3,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "Sigmoid",
                "n_neurons": 64,
                "n_hidden_layers": 2,
            },
        )

        # Sampler
        self.sampler = sphere_tracing

        # Collider
        self.collider = NearFarCollider(near_plane=self.config.near_plane, far_plane=self.config.far_plane)

        # renderers
        background_color = (
            get_color(self.config.background_color)
            if self.config.background_color in set(["white", "black"])
            else self.config.background_color
        )
        self.renderer_rgb = RGBRenderer(background_color=background_color)

        # losses
        self.rgb_loss = MSELoss()

        # metrics
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = structural_similarity_index_measure
        self.lpips = LearnedPerceptualImagePatchSimilarity()

        self.acc_points = None

    def sdf_func(self, xyz):
        sdf = -self.nksr_field.evaluate_f(xyz/self.scale).value
        active_mask = self.nksr_field.svh.grids[0].points_in_active_voxel(xyz/self.scale)
        sdf[~active_mask] = 0.01
        return torch.unsqueeze(sdf, 1)

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = {}
        param_groups["fields"] = list(self.feat_field.parameters()) + list(self.color_field.parameters())
        return param_groups

    def get_outputs(self, ray_bundle: RayBundle, lidar_rays=None, sky_mask=None):
        ray_positions, converged = self.sampler(self.sdf_func, ray_bundle.origins, ray_bundle.directions)
        # active_mask = 0
        # for d in range(4):
        #     active_mask += self.nksr_field.svh.grids[d].points_in_active_voxel(ray_positions / self.scale)
        # active_mask = (active_mask / 4) > 0.8
        # active_mask = self.nksr_field.svh.grids[0].points_in_active_voxel(ray_positions / self.scale)
        # udf_mask = self.nksr_field.mask_field.evaluate_f(ray_positions).value > 0.5
        # mask_ray_positions = ray_positions[active_mask]
        # mask_origins = ray_bundle.origins[active_mask]
        # mask_directions = ray_bundle.directions[active_mask]
        # direct_emb = self.direction_encoding(mask_directions).float()
        # h = self.feat_field(mask_ray_positions).float()
        direct_emb = self.direction_encoding(ray_bundle.directions).float()
        h = self.feat_field(ray_positions).float()
        density = trunc_exp(h)[:, 0]
        acc = 1 - torch.exp(-density)
        h = torch.cat([h[:, 1:], direct_emb], dim=-1)
        rgb = self.color_field(h).float()
        rgb = rgb * acc[:, None] + (1.0 - acc[:, None])

        depth = torch.mean((ray_positions - ray_bundle.origins) / ray_bundle.directions, -1)

        outputs = {
            "rgb": rgb,
            "accumulation": acc,
            "depth": depth,
        }

        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(ray_positions.detach().cpu().numpy())
        # o3d.io.write_point_cloud("/data4/hyzhou/exp/snowman/nerf_points.ply", pcd)
        # exit(0)

        return outputs

    def get_metrics_dict(self, outputs, batch):
        metrics_dict = {}
        image = batch["image"].to(self.device)
        metrics_dict["psnr"] = self.psnr(outputs["rgb"], image)
        return metrics_dict

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        loss_dict = {}
        image = batch["image"].to(self.device)
        loss_dict["rgb_loss"] = self.rgb_loss(image, outputs["rgb"])

        return loss_dict

    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:

        image = batch["image"].cpu()
        rgb = outputs["rgb"]
        acc = colormaps.apply_colormap(outputs["accumulation"])
        depth = colormaps.apply_depth_colormap(
            outputs["depth"],
            accumulation=outputs["accumulation"],
        )

        combined_rgb = torch.cat([image, rgb], dim=1)
        combined_acc = torch.cat([acc], dim=1)
        combined_depth = torch.cat([depth], dim=1)

        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        image = torch.moveaxis(image, -1, 0)[None, ...]
        rgb = torch.moveaxis(rgb, -1, 0)[None, ...]

        psnr = self.psnr(image, rgb)
        ssim = self.ssim(image, rgb)
        cpu_lpips = self.lpips.to(image.device)
        lpips = cpu_lpips(image, rgb)

        # all of these metrics will be logged as scalars
        metrics_dict = {"psnr": float(psnr.item()), "ssim": float(ssim)}  # type: ignore
        metrics_dict["lpips"] = float(lpips)

        images_dict = {"img": combined_rgb, "accumulation": combined_acc, "depth": combined_depth}

        # normals to RGB for visualization. TODO: use a colormap
        if "normals" in outputs:
            images_dict["normals"] = (outputs["normals"] + 1.0) / 2.0
        if "pred_normals" in outputs:
            images_dict["pred_normals"] = (outputs["pred_normals"] + 1.0) / 2.0

        return metrics_dict, images_dict

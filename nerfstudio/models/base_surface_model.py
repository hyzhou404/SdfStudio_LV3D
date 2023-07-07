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
Implementation of Base surface model.
"""

from __future__ import annotations

import random
from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Type

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import Parameter
from torchmetrics import PeakSignalNoiseRatio
from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchtyping import TensorType
from typing_extensions import Literal
from nerfstudio.cameras.rays import RayBundle, RaySamples
from nerfstudio.field_components.encodings import NeRFEncoding
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.spatial_distortions import SceneContraction
from nerfstudio.fields.nerfacto_field import TCNNNerfactoField
from nerfstudio.fields.sdf_field import SDFFieldConfig
from nerfstudio.fields.vanilla_nerf_field import NeRFField
from nerfstudio.model_components.losses import (
    L1Loss,
    MSELoss,
    MultiViewLoss,
    ScaleAndShiftInvariantLoss,
    SensorDepthLoss,
    compute_scale_and_shift,
    monosdf_normal_loss,
    monosdf_normal_diff_loss,
)
from nerfstudio.model_components.patch_warping import PatchWarping
from nerfstudio.model_components.ray_samplers import LinearDisparitySampler
from nerfstudio.model_components.renderers import (
    AccumulationRenderer,
    DepthRenderer,
    RGBRenderer,
    SemanticRenderer,
)
from nerfstudio.model_components.scene_colliders import (
    AABBBoxCollider,
    NearFarCollider,
    SphereCollider,
)
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.utils import colormaps
from nerfstudio.utils.colors import get_color
import torch.distributions.normal as normal


@dataclass
class SurfaceModelConfig(ModelConfig):
    """Nerfacto Model Config"""

    _target: Type = field(default_factory=lambda: SurfaceModel)
    near_plane: float = 0.05
    """How far along the ray to start sampling."""
    far_plane: float = 4.0
    """How far along the ray to stop sampling."""
    far_plane_bg: float = 1000.0
    """How far along the ray to stop sampling of the background model."""
    background_color: Literal["random", "last_sample", "white", "black"] = "white"
    """Whether to randomize the background color."""
    use_average_appearance_embedding: bool = False
    """Whether to use average appearance embedding or zeros for inference."""
    eikonal_loss_mult: float = 0.1
    """Eikonal loss multiplier."""
    fg_mask_loss_mult: float = 0.01
    """Foreground mask loss multiplier."""
    mono_normal_loss_mult: float = 0.0
    """Monocular normal consistency loss multiplier."""
    mono_depth_loss_mult: float = 0.0
    """Monocular depth consistency loss multiplier."""
    patch_warp_loss_mult: float = 0.0
    """Multi-view consistency warping loss multiplier."""
    patch_size: int = 11
    """Multi-view consistency warping loss patch size."""
    patch_warp_angle_thres: float = 0.3
    """Threshold for valid homograph of multi-view consistency warping loss"""
    min_patch_variance: float = 0.01
    """Threshold for minimal patch variance"""
    topk: int = 4
    """Number of minimal patch consistency selected for training"""
    sensor_depth_truncation: float = 0.015
    """Sensor depth trunction, default value is 0.015 which means 5cm with a rough scale value 0.3 (0.015 = 0.05 * 0.3)"""
    sensor_depth_l1_loss_mult: float = 0.0
    """Sensor depth L1 loss multiplier."""
    sensor_depth_freespace_loss_mult: float = 0.0
    """Sensor depth free space loss multiplier."""
    sensor_depth_sdf_loss_mult: float = 0.0
    """Sensor depth sdf loss multiplier."""
    sparse_points_sdf_loss_mult: float = 0.0
    """sparse point sdf loss multiplier"""
    sdf_field: SDFFieldConfig = SDFFieldConfig()
    """Config for SDF Field"""
    background_model: Literal["grid", "mlp", "none"] = "mlp"
    """background models"""
    num_samples_outside: int = 32
    """Number of samples outside the bounding sphere for backgound"""
    periodic_tvl_mult: float = 0.0
    """Total variational loss mutliplier"""
    overwrite_near_far_plane: bool = False
    """whether to use near and far collider from command line"""
    scene_contraction_norm: Literal["inf", "l2"] = "inf"
    """Which norm to use for the scene contraction."""


class SurfaceModel(Model):
    """Base surface model

    Args:
        config: Base surface model configuration to instantiate model
    """

    config: SurfaceModelConfig

    def populate_modules(self):
        """Set the fields and modules."""
        super().populate_modules()

        if self.config.scene_contraction_norm == "inf":
            order = float("inf")
        elif self.config.scene_contraction_norm == "l2":
            order = None
        else:
            raise ValueError("Invalid scene contraction norm")

        self.scene_contraction = SceneContraction(order=order)

        # Can we also use contraction for sdf?
        # Fields
        self.field = self.config.sdf_field.setup(
            aabb=self.scene_box.aabb,
            spatial_distortion=None,
            # spatial_distortion=self.scene_contraction,
            num_images=self.num_train_data,
            use_average_appearance_embedding=self.config.use_average_appearance_embedding,
            poses=self.poses,
            scale=self.scale,
            hashgrid_len=self.scene_box.hashgrid_len,
        )
        # Collider
        if self.scene_box.collider_type == "near_far":
            self.collider = NearFarCollider(near_plane=self.scene_box.near, far_plane=self.scene_box.far)
        elif self.scene_box.collider_type == "box":
            # print(self.scene_box.aabb, self.scene_box.near)
            self.collider = AABBBoxCollider(self.scene_box, near_plane=self.scene_box.near)
        elif self.scene_box.collider_type == "sphere":
            # TODO do we also use near if the ray don't intersect with the sphere
            self.collider = SphereCollider(radius=self.scene_box.radius, soft_intersection=True)
        else:
            raise NotImplementedError

        # command line near and far has highest priority
        if self.config.overwrite_near_far_plane:
            self.collider = NearFarCollider(near_plane=self.config.near_plane, far_plane=self.config.far_plane)

        # background model
        if self.config.background_model == "grid":
            self.field_background = TCNNNerfactoField(
                torch.tensor([[-2.0, -2.0, -2.0], [2.0, 2.0, 2.0]]),
                # self.scene_box.aabb,
                spatial_distortion=self.scene_contraction,
                num_images=self.num_train_data,
                use_average_appearance_embedding=self.config.use_average_appearance_embedding,
            )
        elif self.config.background_model == "mlp":
            position_encoding = NeRFEncoding(
                in_dim=3, num_frequencies=10, min_freq_exp=0.0, max_freq_exp=9.0, include_input=True
            )
            direction_encoding = NeRFEncoding(
                in_dim=3, num_frequencies=4, min_freq_exp=0.0, max_freq_exp=3.0, include_input=True
            )

            self.field_background = NeRFField(
                position_encoding=position_encoding,
                direction_encoding=direction_encoding,
                spatial_distortion=self.scene_contraction,
            )
        else:
            # dummy background model
            self.field_background = Parameter(torch.ones(1), requires_grad=False)

        self.sampler_bg = LinearDisparitySampler(num_samples=self.config.num_samples_outside)

        # renderers
        background_color = (
            get_color(self.config.background_color)
            if self.config.background_color in set(["white", "black"])
            else self.config.background_color
        )
        self.renderer_rgb = RGBRenderer(background_color=background_color)
        self.renderer_accumulation = AccumulationRenderer()
        self.renderer_depth = DepthRenderer(method="expected")
        self.renderer_normal = SemanticRenderer()
        # patch warping
        self.patch_warping = PatchWarping(
            patch_size=self.config.patch_size, valid_angle_thres=self.config.patch_warp_angle_thres
        )

        # losses
        self.rgb_loss = MSELoss(reduction='none')
        self.eikonal_loss = MSELoss()
        self.depth_loss = ScaleAndShiftInvariantLoss(alpha=0.5, scales=1)
        self.patch_loss = MultiViewLoss(
            patch_size=self.config.patch_size, topk=self.config.topk, min_patch_variance=self.config.min_patch_variance
        )
        self.sensor_depth_loss = SensorDepthLoss(truncation=self.config.sensor_depth_truncation)

        # metrics
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = structural_similarity_index_measure
        self.lpips = LearnedPerceptualImagePatchSimilarity()

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = {}
        param_groups["fields"] = list(self.field.parameters())
        if self.config.background_model != "none":
            param_groups["field_background"] = list(self.field_background.parameters())
        else:
            param_groups["field_background"] = list(self.field_background)
        return param_groups

    @abstractmethod
    def sample_and_forward_field(self, ray_bundle: RayBundle, sky_mask=None) -> Dict:
        """_summary_

        Args:
            ray_bundle (RayBundle): _description_
            return_samples (bool, optional): _description_. Defaults to False.
        """

    # def get_foreground_mask(self, ray_samples: RaySamples) -> TensorType:
    #     """_summary_
    #
    #     Args:
    #         ray_samples (RaySamples): _description_
    #     """
    #     # TODO support multiple foreground type: box and sphere
    #     inside_sphere_mask = (ray_samples.frustums.get_start_positions().norm(dim=-1, keepdim=True) < 1.0).float()
    #     return inside_sphere_mask
    #
    # def forward_background_field_and_merge(self, ray_samples: RaySamples, field_outputs: Dict) -> Dict:
    #     """_summary_
    #
    #     Args:
    #         ray_samples (RaySamples): _description_
    #         field_outputs (Dict): _description_
    #     """
    #
    #     inside_sphere_mask = self.get_foreground_mask(ray_samples)
    #     # TODO only forward the points that are outside the sphere if there is a background model
    #
    #     field_outputs_bg = self.field_background(ray_samples)
    #     field_outputs_bg[FieldHeadNames.ALPHA] = ray_samples.get_alphas(field_outputs_bg[FieldHeadNames.DENSITY])
    #
    #     field_outputs[FieldHeadNames.ALPHA] = (
    #         field_outputs[FieldHeadNames.ALPHA] * inside_sphere_mask
    #         + (1.0 - inside_sphere_mask) * field_outputs_bg[FieldHeadNames.ALPHA]
    #     )
    #     field_outputs[FieldHeadNames.RGB] = (
    #         field_outputs[FieldHeadNames.RGB] * inside_sphere_mask
    #         + (1.0 - inside_sphere_mask) * field_outputs_bg[FieldHeadNames.RGB]
    #     )
    #
    #     # TODO make everything outside the sphere to be 0
    #     return field_outputs

    def get_outputs(self, ray_bundle: RayBundle, lidar_rays=None, sky_mask=None) -> Dict:
        ray_bundle = self.collider.set_nears_and_fars(ray_bundle)
        ray_bundle.nears *= 0
        # print('nears', ray_bundle.nears)
        # farthest = ray_bundle.origins + ray_bundle.directions * ray_bundle.fars
        # print('farthest', farthest)
        # n = torch.linalg.norm(farthest, ord=float('inf'), dim=-1)
        # print(torch.max(n))
        # TODO make this configurable
        # compute near and far from from sphere with radius 1.0
        # ray_bundle = self.sphere_collider(ray_bundle)
        samples_and_field_outputs = self.sample_and_forward_field(ray_bundle=ray_bundle, sky_mask=sky_mask)
        """60 samples front the lidar point, 1 sample exact the point, 3 sample back the point in eps"""
        if lidar_rays is not None:
            lidar_occupancy = self.lidar_sample_and_sdf_field(lidar_rays)
        else:
            lidar_occupancy = None

        # Shotscuts
        field_outputs = samples_and_field_outputs["field_outputs"]
        ray_samples = samples_and_field_outputs["ray_samples"]
        weights = samples_and_field_outputs["weights"]
        bg_transmittance = samples_and_field_outputs["bg_transmittance"]

        use_background_color = (self.config.background_model == 'none')
        rgb = self.renderer_rgb(rgb=field_outputs[FieldHeadNames.RGB], weights=weights, use_bgc=use_background_color)
        depth = self.renderer_depth(weights=weights, ray_samples=ray_samples)
        # the rendered depth is point-to-point distance and we should convert to depth
        # depth = depth / ray_bundle.directions_norm

        # remove the rays that don't intersect with the surface
        # sdf_out = field_outputs[FieldHeadNames.SDF]
        # hit = (field_outputs[FieldHeadNames.SDF] > 0.0).any(dim=1) & (field_outputs[FieldHeadNames.SDF] < 0).any(dim=1)
        # depth[~hit] = 10.0

        normal = self.renderer_normal(semantics=field_outputs[FieldHeadNames.NORMAL], weights=weights)
        accumulation = self.renderer_accumulation(weights=weights)

        # background model
        if self.config.background_model != "none":
            # TODO remove hard-coded far value
            # sample inversely from far to 1000 and points and forward the bg model
            ray_bundle.nears = ray_bundle.fars
            ray_bundle.fars = torch.ones_like(ray_bundle.fars) * self.config.far_plane_bg

            ray_samples_bg = self.sampler_bg(ray_bundle)
            # use the same background model for both density field and occupancy field
            field_outputs_bg = self.field_background(ray_samples_bg)
            weights_bg = ray_samples_bg.get_weights(field_outputs_bg[FieldHeadNames.DENSITY])

            rgb_bg = self.renderer_rgb(rgb=field_outputs_bg[FieldHeadNames.RGB], weights=weights_bg)
            depth_bg = self.renderer_depth(weights=weights_bg, ray_samples=ray_samples_bg)
            accumulation_bg = self.renderer_accumulation(weights=weights_bg)

            # merge background color to forgound color
            rgb = rgb + bg_transmittance * rgb_bg

            bg_outputs = {
                "bg_rgb": rgb_bg,
                "bg_accumulation": accumulation_bg,
                "bg_depth": depth_bg,
                "bg_weights": weights_bg,
            }
        else:
            bg_outputs = {}


        density = field_outputs[FieldHeadNames.DENSITY]
        ray_points = self.scene_contraction(
            ray_samples.frustums.get_start_positions()
        )
        ray_steps = ray_samples.frustums.starts

        outputs = {
            "rgb": rgb,
            "accumulation": accumulation,
            # "transmit": bg_transmittance,
            "density": density,
            # "sdf_out": sdf_out,
            "depth": depth,
            "normal": normal,
            "weights": weights,
            "ray_points": ray_points,
            "ray_steps": ray_steps,
            # "occupancy": field_outputs[FieldHeadNames.OCCUPANCY],
            "lidar_occupancy": lidar_occupancy
            # "directions_norm": ray_bundle.directions_norm,  # used to scale z_vals for free space and sdf loss
        }

        if self.training:
            grad_points = field_outputs[FieldHeadNames.GRADIENT]
            points_norm = field_outputs["points_norm"]
            outputs.update({"eik_grad": grad_points, "points_norm": points_norm})

            # TODO volsdf use different point set for eikonal loss
            # grad_points = self.field.gradient(eik_points)
            # outputs.update({"eik_grad": grad_points})

            outputs.update(samples_and_field_outputs)

        # TODO how can we move it to neus_facto without out of memory
        if "weights_list" in samples_and_field_outputs:
            weights_list = samples_and_field_outputs["weights_list"]
            ray_samples_list = samples_and_field_outputs["ray_samples_list"]

            for i in range(len(weights_list) - 1):
                outputs[f"prop_depth_{i}"] = self.renderer_depth(
                    weights=weights_list[i], ray_samples=ray_samples_list[i]
                )
        # this is used only in viewer
        outputs["normal_vis"] = (outputs["normal"] + 1.0) / 2.0
        return outputs

    def get_outputs_flexible(self, ray_bundle: RayBundle, additional_inputs: Dict[str, TensorType]) -> Dict:
        """run the model with additional inputs such as warping or rendering from unseen rays
        Args:
            ray_bundle: containing all the information needed to render that ray latents included
            additional_inputs: addtional inputs such as images, src_idx, src_cameras

        Returns:
            dict: information needed for compute gradients
        """
        if self.collider is not None:
            ray_bundle = self.collider(ray_bundle)

        outputs = self.get_outputs(ray_bundle)

        ray_samples = outputs["ray_samples"]
        field_outputs = outputs["field_outputs"]

        if self.config.patch_warp_loss_mult > 0:
            # patch warping
            warped_patches, valid_mask = self.patch_warping(
                ray_samples,
                field_outputs[FieldHeadNames.SDF],
                field_outputs[FieldHeadNames.NORMAL],
                additional_inputs["src_cameras"],
                additional_inputs["src_imgs"],
                pix_indices=additional_inputs["uv"],
            )

            outputs.update({"patches": warped_patches, "patches_valid_mask": valid_mask})

        return outputs

    def get_loss_dict(self, outputs, batch, metrics_dict=None) -> Dict:
        loss_dict = {}
        image = batch["image"].to(self.device)
        loss_dict["rgb_loss"] = self.rgb_loss(image, outputs["rgb"])
        if "sky_mask" in batch:
            loss_dict["sky_accumulation_loss"] = 0.005*torch.nan_to_num(
                torch.mean(outputs["accumulation"][batch["sky_mask"]]), 0
            )

            # loss_dict["sky_transmit_loss"] = torch.nan_to_num(
            #     torch.mean(1 - outputs["transmit"][batch["sky_mask"]]), 0
            # )
            #
            # loss_dict["rgb_loss"][batch["sky_mask"]] = 0
            # loss_dict["sky_density_loss"] = 0.01 * torch.mean(outputs["occupancy"][batch["sky_mask"]])
        loss_dict["rgb_loss"] = torch.mean(loss_dict["rgb_loss"])
        if self.training:
            # eikonal loss
            grad_theta = outputs["eik_grad"]
            loss_dict["eikonal_loss"] = ((grad_theta.norm(2, dim=-1) - 1) ** 2).mean() * self.config.eikonal_loss_mult

            # foreground mask loss
            if "fg_mask" in batch and self.config.fg_mask_loss_mult > 0.0:
                fg_label = batch["fg_mask"].float().to(self.device)
                weights_sum = outputs["weights"].sum(dim=1).clip(1e-3, 1.0 - 1e-3)
                loss_dict["fg_mask_loss"] = (
                    F.binary_cross_entropy(weights_sum, fg_label) * self.config.fg_mask_loss_mult
                )

            # monocular normal loss
            # if "normal" in batch and self.config.mono_normal_loss_mult > 0.0 and batch["step"] > 0:
            #     normal_gt = batch["normal"].to(self.device)
            #     normal_pred = outputs["normal"]
            #     loss_dict["normal_loss"] = (
            #         monosdf_normal_loss(normal_pred, normal_gt, batch["road_mask"]) * self.config.mono_normal_loss_mult
            #     )

            if "normal" in batch and self.config.mono_normal_loss_mult > 0.0:
                # mask = (outputs['accumulation'] > 0.5)[:, 0]
                mask = ~batch['sky_mask']
                if torch.sum(mask) == 0:
                    loss_dict['normal_loss'] = 0
                else:
                    normal_gt = batch["normal"].to(self.device)[mask]
                    normal_pred = outputs["normal"][mask]
                    loss_dict["normal_loss"] = (
                            monosdf_normal_loss(normal_pred, normal_gt) * self.config.mono_normal_loss_mult
                    )

            # if "normal" in batch and self.config.mono_normal_loss_mult > 0.0 \
            #         and batch["step"] > 20000 and batch["step"] % 10 == 0:
            #     normal_gt = batch["normal"].to(self.device)
            #     normal_pred = outputs["normal"]
            #     normal_gt = normal_gt.view(-1, 4, 4, 3)
            #     normal_pred = normal_pred.view(-1, 4, 4, 3)
            #     loss_dict["normal_loss"] = (
            #         monosdf_normal_diff_loss(normal_pred, normal_gt,
            #                                  ~batch["dilate_sky_mask"]) * self.config.mono_normal_loss_mult
            #     )

            if outputs["lidar_occupancy"] is not None and batch["step"] > 1500:
                occupancy = outputs["lidar_occupancy"]

                # occupancy style loss
                # l1 = torch.mean(torch.abs(occupancy[:, :60]))
                # l2 = torch.mean(torch.abs(1 - occupancy[:, 60]))
                # l3 = torch.mean(torch.abs(occupancy[:, 61:]))

                # sdf style loss
                device = occupancy.device
                l1 = torch.mean(
                    torch.maximum(0.1-occupancy[:, :60], torch.zeros_like(occupancy[:, :60], device=device))
                )
                l2 = torch.mean(torch.abs(occupancy[:, 60]))
                l3 = torch.mean(
                    torch.maximum(occupancy[:, 61:], torch.zeros_like(occupancy[:, 61:], device=device))
                )
                # UDF
                # l3 = torch.mean(
                #     torch.maximum(0.01-occupancy[:, 61:], torch.zeros_like(occupancy[:, 61:], device=device))
                # )
                loss_dict["lidar_loss"] = 0.1 * (l1 + l2 + l3)

            # urban radiance field loss
            # if "sky_mask" in batch:
            #
            #     eps = 0.01 - ((0.01 - 0.001) / 20000 * batch['step'])
            #     # eps = 0.1
            #
            #     depth_pred = outputs["depth"]  # mono pred (bs, 1)
            #     sky_mask = batch['sky_mask'].bool()
            #     zs = outputs['ray_steps']  # scaled z near~far (bs, sample, 1)
            #     weights = outputs['weights']  # (bs, sample, 1)
            #
            #     zs_flat = torch.squeeze(zs, -1)  # (2048, 48)
            #     depth_flat = depth_pred
            #     weights_flat = torch.squeeze(weights, -1)  # (2048, 48)
            #     sky_ray_flat = sky_mask.squeeze(-1)  # (2048,)
            #     filter_weights = weights_flat[~sky_ray_flat]
            #
            #     # term1
            #     t1_mask = (zs_flat <= depth_flat - eps)[~sky_ray_flat]
            #     if filter_weights[t1_mask].shape[0] > 0:
            #         urf_l1 = torch.mean(filter_weights[t1_mask] ** 2)
            #     else:
            #         urf_l1 = 0
            #     # term2
            #     gaussian_dist = normal.Normal(depth_flat[:, 0][~sky_ray_flat], eps / 3)
            #     t2_mask = ((depth_flat - eps < zs_flat) & (zs_flat < depth_flat + eps))[~sky_ray_flat]
            #     t2_weights = filter_weights[t2_mask]
            #     t2_gt = torch.exp(gaussian_dist.log_prob(zs_flat[~sky_ray_flat].T)).T[t2_mask]
            #     scale = torch.sum(t2_gt) / torch.sum(t2_weights)
            #     urf_l2 = torch.mean((t2_weights - t2_gt / scale) ** 2)
            #     # term3
            #     t3_mask = (zs_flat >= depth_flat + eps)[~sky_ray_flat]
            #     if filter_weights[t3_mask].shape[0] > 0:
            #         urf_l3 = torch.mean(filter_weights[t3_mask] ** 2)
            #     else:
            #         urf_l3 = 0
            #
            #     loss_dict['urf_loss'] = 0.1 * (urf_l1 + urf_l2 + urf_l3)
            #
            #     # metric but in loss
            #     # loss_dict["inside_count"] = torch.sum(t2_mask)
            #     # loss_dict["outside_count"] = torch.sum(t1_mask) + torch.sum(t3_mask)

            # monocular depth loss
            if "depth" in batch and self.config.mono_depth_loss_mult > 0.0:
                # TODO check it's true that's we sample from only a single image
                # TODO only supervised pixel that hit the surface and remove hard-coded scaling for depth
                depth_gt = batch["depth"].to(self.device)[..., None]
                depth_pred = outputs["depth"]

                # mask = torch.ones_like(depth_gt).reshape(1, 32, -1).bool()
                mask = ~batch['sky_mask'].reshape(1, 32, -1).bool()

                loss_dict["depth_loss"] = (
                    self.depth_loss(depth_pred.reshape(1, 32, -1), (depth_gt * 50 + 0.5).reshape(1, 32, -1), mask)
                    * self.config.mono_depth_loss_mult
                )

            # sensor depth loss
            if "sensor_depth" in batch and (
                self.config.sensor_depth_l1_loss_mult > 0.0
                or self.config.sensor_depth_freespace_loss_mult > 0.0
                or self.config.sensor_depth_sdf_loss_mult > 0.0
            ):
                l1_loss, free_space_loss, sdf_loss = self.sensor_depth_loss(batch, outputs)

                loss_dict["sensor_l1_loss"] = l1_loss * self.config.sensor_depth_l1_loss_mult
                loss_dict["sensor_freespace_loss"] = free_space_loss * self.config.sensor_depth_freespace_loss_mult
                loss_dict["sensor_sdf_loss"] = sdf_loss * self.config.sensor_depth_sdf_loss_mult

            # multi-view photoconsistency loss as Geo-NeuS
            if "patches" in outputs and self.config.patch_warp_loss_mult > 0.0:
                patches = outputs["patches"]
                patches_valid_mask = outputs["patches_valid_mask"]

                loss_dict["patch_loss"] = (
                    self.patch_loss(patches, patches_valid_mask) * self.config.patch_warp_loss_mult
                )

            # sparse points sdf loss
            if "sparse_sfm_points" in batch and self.config.sparse_points_sdf_loss_mult > 0.0:
                sparse_sfm_points = batch["sparse_sfm_points"].to(self.device)
                sparse_sfm_points_sdf = self.field.forward_geonetwork(sparse_sfm_points)[:, 0].contiguous()
                loss_dict["sparse_sfm_points_sdf_loss"] = (
                    torch.mean(torch.abs(sparse_sfm_points_sdf)) * self.config.sparse_points_sdf_loss_mult
                )

            # total variational loss for multi-resolution periodic feature volume
            if self.config.periodic_tvl_mult > 0.0:
                assert self.field.config.encoding_type == "periodic"
                loss_dict["tvl_loss"] = self.field.encoding.get_total_variation_loss() * self.config.periodic_tvl_mult

        return loss_dict

    def get_metrics_dict(self, outputs, batch) -> Dict:
        metrics_dict = {}
        image = batch["image"].to(self.device)
        metrics_dict["psnr"] = self.psnr(outputs["rgb"], image)
        return metrics_dict

    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:

        # image = batch["image"].to(self.device)
        rgb = outputs["rgb"]
        image = batch["image"].cpu()

        # torch.save(rgb, '/data/hyzhou/data/infer_save/rgb.pt')
        # torch.save(image, '/data/hyzhou/data/infer_save/image.pt')
        # exit(0)

        # if "sky_mask" in batch:
        #     rgb[batch["sky_mask"]] = torch.tensor([1.0, 1.0, 1.0]).to(outputs["rgb"].device)
        acc = colormaps.apply_colormap(outputs["accumulation"])

        normal = outputs["normal"]
        # don't need to normalize here
        # normal = torch.nn.functional.normalize(normal, p=2, dim=-1)
        normal = (normal + 1.0) / 2.0

        combined_rgb = torch.cat([image, rgb], dim=1)
        combined_acc = torch.cat([acc], dim=1)
        # if "depth" in batch:
        #     depth_gt = batch["depth"].to(self.device)
        #     depth_pred = outputs["depth"]
        #
        #     # align to predicted depth and normalize
        #     scale, shift = compute_scale_and_shift(
        #         depth_pred[None, ..., 0], depth_gt[None, ...], depth_gt[None, ...] > 0.0
        #     )
        #     depth_pred = depth_pred * scale + shift
        #
        #     combined_depth = torch.cat([depth_gt[..., None], depth_pred], dim=1)
        #     combined_depth = colormaps.apply_depth_colormap(combined_depth)
        # else:
            # depth = colormaps.apply_depth_colormap(
            #     outputs["depth"],
            #     accumulation=outputs["accumulation"],
            # )
            # combined_depth = torch.cat([depth], dim=1)
        depth = colormaps.apply_depth_colormap(
            outputs["depth"],
            accumulation=outputs["accumulation"],
        )
        combined_depth = torch.cat([depth], dim=1)

        # ray weights, depth and sdf debug
        debug = False
        if debug:
            depth = outputs['depth']
            density = outputs["density"]
            weights = outputs['weights']  # (376, 1408, 48)
            points = outputs["ray_points"]  # (376, 1408, 144)
            ray_steps = outputs['ray_steps']

            ray_data = []
            depths = []
            x = np.linspace(0, depth.shape[0]-1, 10, dtype=int)
            y = np.linspace(0, depth.shape[1]-1, 30, dtype=int)
            xx, yy = np.meshgrid(x, y)
            ij = np.stack((xx.ravel(), yy.ravel()), axis=1)
            for i, j in ij:
                ray_density = density[i, j]
                ray_weights = weights[i, j]
                ray_points = points[i, j].view(-1, 3)
                # ray_zs = torch.tensor(
                #             [torch.mean((point - ray.origins) / ray.directions).item() / ray.directions_norm for point in ray_points]
                #         ).to(ray_points.device)
                ray_zs = ray_steps[i, j]
                ray_sdf = self.field.forward_geonetwork(ray_points)[:, 0]
                ray_data.append(torch.stack([ray_zs, ray_density, ray_weights, ray_sdf]).detach().cpu().numpy())

                depths.append(depth[i, j].item())

                # eps = 0.05
                # xyzs = []
                # for k in [-1, 0, 1]:
                #     xyzs.append(ray.origins + (depth[i, j] + k*eps) * ray.directions)
                # xyzs = torch.stack(xyzs)
                # print(self.field.forward_geonetwork(xyzs)[:, 0].detach().cpu())
            ray_data = np.array(ray_data)
            depths = np.array(depths)
            np.save('/data/hyzhou/data/ray_data/img0.npy', batch['image'].detach().cpu().numpy())
            np.save('/data/hyzhou/data/ray_data/img0_ray.npy', ray_data)
            np.save('/data/hyzhou/data/ray_data/img0_depths.npy', depths)
            np.save('/data/hyzhou/data/ray_data/img0_ij.npy', ij)
            exit(0)

        if "normal" in batch:
            normal_gt = (batch["normal"].cpu() + 1.0) / 2.0
            combined_normal = torch.cat([normal_gt, normal], dim=1)
        else:
            combined_normal = torch.cat([normal], dim=1)

        images_dict = {
            "img": combined_rgb,
            "accumulation": combined_acc,
            "depth": combined_depth,
            "normal": combined_normal,
        }

        # if "depth" in batch:
        #     depth_gt = batch["depth"].to(self.device)
        #     depth_gt = colormaps.apply_depth_colormap(
        #         depth_gt,
        #         accumulation=outputs["accumulation"],
        #     )
        #     depth_gt = torch.cat([depth_gt], dim=1)
        #     images_dict['depth_gt'] = depth_gt

        if "sensor_depth" in batch:
            sensor_depth = batch["sensor_depth"]
            depth_pred = outputs["depth"]

            combined_sensor_depth = torch.cat([sensor_depth[..., None], depth_pred], dim=1)
            combined_sensor_depth = colormaps.apply_depth_colormap(combined_sensor_depth)
            images_dict["sensor_depth"] = combined_sensor_depth

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

        return metrics_dict, images_dict

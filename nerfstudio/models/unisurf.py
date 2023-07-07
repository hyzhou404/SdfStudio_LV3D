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
Implementation of VolSDF.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Type

import torch

from nerfstudio.engine.callbacks import (
    TrainingCallback,
    TrainingCallbackAttributes,
    TrainingCallbackLocation,
)
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.model_components.ray_samplers import UniSurfSampler
from nerfstudio.models.base_surface_model import SurfaceModel, SurfaceModelConfig
from nerfstudio.model_components.losses import monosdf_normal_diff_loss


@dataclass
class UniSurfModelConfig(SurfaceModelConfig):
    """UniSurf Model Config"""

    _target: Type = field(default_factory=lambda: UniSurfModel)
    eikonal_loss_mult: float = 0.0
    """overwirte eikonal loss because it's not need for unisurf"""
    smooth_loss_multi: float = 0.005
    """smoothness loss on surface points in unisurf"""
    num_samples_interval: int = 64
    """Number of uniform samples"""
    num_samples_importance: int = 32
    """Number of important samples"""
    num_marching_steps: int = 256
    """number of up sample step, 1 for simple coarse-to-fine sampling"""
    perturb: bool = True
    """use to use perturb for the sampled points"""

    enable_progressive_hash_encoding: bool = False
    """whether to use progressive hash encoding"""
    enable_numerical_gradients_schedule: bool = False
    """whether to use numerical gradients delta schedule"""
    enable_curvature_loss_schedule: bool = False
    """whether to use curvature loss weight schedule"""
    curvature_loss_multi: float = 0.0
    """curvature loss weight"""
    curvature_loss_warmup_steps: int = 5000
    """curvature loss warmup steps"""
    level_init: int = 4
    """initial level of multi-resolution hash encoding"""
    steps_per_level: int = 2000
    """steps per level of multi-resolution hash encoding"""

class UniSurfModel(SurfaceModel):
    """VolSDF model

    Args:
        config: MonoSDF configuration to instantiate model
    """

    config: UniSurfModelConfig

    def populate_modules(self):
        """Set the fields and modules."""
        super().populate_modules()

        # can't use eikonal loss in Unisurf? or we could use learnable paramter to transform sdf to occupancy
        assert self.config.eikonal_loss_mult == 0.0

        self.sampler = UniSurfSampler(
            num_samples_interval=self.config.num_samples_interval,
            num_samples_outside=self.config.num_samples_outside,
            num_samples_importance=self.config.num_samples_importance,
            num_marching_steps=self.config.num_marching_steps,
        )

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        callbacks = []
        callbacks.append(
            TrainingCallback(
                where_to_run=[TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                update_every_num_iters=1,
                func=self.sampler.step_cb,
            )
        )

        level_init = self.config.level_init
        steps_per_level = self.config.steps_per_level
        # schedule the current level of multi-resolution hash encoding
        if self.config.enable_progressive_hash_encoding:
            def set_mask(step):
                # TODO make this consistent with delta schedule
                level = int(step / steps_per_level) + 1
                level = max(level, level_init)
                self.field.update_mask(level)

            callbacks.append(
                TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                    update_every_num_iters=1,
                    func=set_mask,
                )
            )
        return callbacks

    def sample_and_forward_field(self, ray_bundle: RayBundle, sky_mask=None) -> Dict:
        ray_samples, surface_points = self.sampler(
            ray_bundle, occupancy_fn=self.field.get_occupancy, sdf_fn=self.field.get_sdf, return_surface_points=True
        )
        field_outputs = self.field(ray_samples, return_occupancy=True)
        weights, transmittance = ray_samples.get_weights_and_transmittance_from_alphas(
            field_outputs[FieldHeadNames.OCCUPANCY]
        )
        bg_transmittance = transmittance[:, -1, :]

        samples_and_field_outputs = {
            "ray_samples": ray_samples,
            "surface_points": surface_points,
            "field_outputs": field_outputs,
            "weights": weights,
            "bg_transmittance": bg_transmittance,
        }
        return samples_and_field_outputs

    def lidar_sample_and_sdf_field(self, lidar_rays):
        device = lidar_rays.device
        lidar_locs, lidar_points = lidar_rays[:, :3][:, None, :], lidar_rays[:, 3:][:, None, :]
        lidar_range = lidar_points - lidar_locs

        front_samples = lidar_locs.repeat(1, 60, 1) + \
                        lidar_range.repeat(1, 60, 1) * torch.rand((1, 60, 1), device=device)
        point_sample = lidar_points
        back_samples = lidar_points.repeat(1, 3, 1) + torch.rand((1, 3, 1), device=device) * 0.01

        lidar_ray_samples = torch.cat([front_samples, point_sample, back_samples], 1)
        lidar_ray_samples = lidar_ray_samples.view(-1, 3).float()

        sdf = self.field.forward_geonetwork(lidar_ray_samples)[..., 0]
        # TODO: Why using occupancy supervision is unsuitable?
        # occupancy = F.sigmoid(-10.0 * sdf)
        return sdf.view(1024, 64)

    def get_metrics_dict(self, outputs, batch) -> Dict:
        metrics_dict = super().get_metrics_dict(outputs, batch)
        if self.training:
            # training statics
            metrics_dict["delta"] = self.sampler.delta

        return metrics_dict

    def get_loss_dict(self, outputs, batch, metrics_dict=None) -> Dict:
        loss_dict = super().get_loss_dict(outputs, batch, metrics_dict)

        # TODO move to base model as other model could also use it?
        if self.training and self.config.smooth_loss_multi > 0.0:
            # if batch["step"] > 40000 and batch["step"] % 10 == 0:
            #     # normal_gt = batch["normal"].to(self.device)
            #     # normal_gt = normal_gt.view(-1, 4, 4, 3)
            #     normal_pred = outputs["normal"]
            #     normal_pred = normal_pred.view(-1, 4, 4, 3)
            #
            #     normal_pred = torch.nn.functional.normalize(normal_pred, p=2, dim=-1)
            #     diff_x = torch.mean(torch.diff(normal_pred, dim=1))
            #     diff_y = torch.mean(torch.diff(normal_pred, dim=2))
            #     loss_dict["normal_smoothness_loss2"] = (diff_x + diff_y) * self.config.mono_normal_loss_mult
            #     # loss_dict["normal_smooth_loss2"] = (
            #     #     monosdf_normal_diff_loss(normal_pred, normal_gt) * self.config.mono_normal_loss_mult
            #     # )

            surface_points = outputs["surface_points"]

            surface_points_neig = surface_points + (torch.rand_like(surface_points) - 0.5) * 0.01
            pp = torch.cat([surface_points, surface_points_neig], dim=0)
            surface_grad = self.field.gradient(pp)
            surface_points_normal = torch.nn.functional.normalize(surface_grad, p=2, dim=-1)

            N = surface_points_normal.shape[0] // 2

            diff_norm = torch.norm(surface_points_normal[:N] - surface_points_normal[N:], dim=-1)
            loss_dict["normal_smoothness_loss"] = torch.mean(diff_norm) * self.config.smooth_loss_multi

        return loss_dict

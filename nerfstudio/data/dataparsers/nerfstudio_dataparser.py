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
""" Data parser for nerfstudio datasets. """

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from pathlib import Path, PurePath
from typing import Optional, Type

import numpy as np
import torch
from PIL import Image
import cv2 as cv
from rich.console import Console
from typing_extensions import Literal

from nerfstudio.cameras import camera_utils
from nerfstudio.cameras.cameras import CAMERA_MODEL_TO_TYPE, Cameras, CameraType
from nerfstudio.data.dataparsers.base_dataparser import (
    DataParser,
    DataParserConfig,
    DataparserOutputs,
)
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.utils.io import load_from_json
from nerfstudio.utils import plotly_utils as vis

CONSOLE = Console(width=120)
MAX_AUTO_RESOLUTION = 1600
from nerfstudio.data.utils.annotation_3d import Annotation3D, global2local,id2label

@dataclass
class NerfstudioDataParserConfig(DataParserConfig):
    """Nerfstudio dataset config"""

    _target: Type = field(default_factory=lambda: Nerfstudio)
    """target class to instantiate"""
    data: Path = Path("data/nerfstudio/poster")
    """Directory specifying location of data."""
    scale_factor: float = 1.0
    """How much to scale the camera origins by."""
    downscale_factor: Optional[int] = None
    """How much to downscale images. If not set, images are chosen such that the max dimension is <1600px."""
    scene_scale: float = 1.0
    """How much to scale the region of interest by."""
    orientation_method: Literal["pca", "up", "none"] = "none"
    """The method to use for orientation."""
    center_poses: bool = True
    """Whether to center the poses."""
    auto_scale_poses: bool = True
    """Whether to automatically scale the poses to fit in +/- 1 bounding box."""
    train_split_percentage: float = 0.9
    """The percent of images to use for training. The remaining images are for eval."""
    annotation_3d = None


@dataclass
class Nerfstudio(DataParser):
    """Nerfstudio DatasetParser"""

    config: NerfstudioDataParserConfig
    downscale_factor: Optional[int] = None

    def _generate_dataparser_outputs(self, split="train"):
        # pylint: disable=too-many-statements

        meta = load_from_json(self.config.data / "transforms.json")
        image_filenames = []
        mask_filenames = []
        poses = []
        num_skipped_image_filenames = 0
        img_index=[]

        fx_fixed = "fl_x" in meta
        fy_fixed = "fl_y" in meta
        cx_fixed = "cx" in meta
        cy_fixed = "cy" in meta
        height_fixed = "h" in meta
        width_fixed = "w" in meta
        distort_fixed = False
        for distort_key in ["k1", "k2", "k3", "p1", "p2"]:
            if distort_key in meta:
                distort_fixed = True
                break
        fx = []
        fy = []
        cx = []
        cy = []
        height = []
        width = []
        distort = []

        for frame in meta["frames"]:
            filepath = PurePath(frame["file_path"])
            fname = self._get_fname(filepath)
            if not fname.exists():
                num_skipped_image_filenames += 1
                continue

            if not fx_fixed:
                assert "fl_x" in frame, "fx not specified in frame"
                fx.append(float(frame["fl_x"]))
            if not fy_fixed:
                assert "fl_y" in frame, "fy not specified in frame"
                fy.append(float(frame["fl_y"]))
            if not cx_fixed:
                assert "cx" in frame, "cx not specified in frame"
                cx.append(float(frame["cx"]))
            if not cy_fixed:
                assert "cy" in frame, "cy not specified in frame"
                cy.append(float(frame["cy"]))
            if not height_fixed:
                assert "h" in frame, "height not specified in frame"
                height.append(int(frame["h"]))
            if not width_fixed:
                assert "w" in frame, "width not specified in frame"
                width.append(int(frame["w"]))
            if not distort_fixed:
                distort.append(
                    camera_utils.get_distortion_params(
                        k1=float(meta["k1"]) if "k1" in meta else 0.0,
                        k2=float(meta["k2"]) if "k2" in meta else 0.0,
                        k3=float(meta["k3"]) if "k3" in meta else 0.0,
                        k4=float(meta["k4"]) if "k4" in meta else 0.0,
                        p1=float(meta["p1"]) if "p1" in meta else 0.0,
                        p2=float(meta["p2"]) if "p2" in meta else 0.0,
                    )
                )

            image_filenames.append(fname)
            poses.append(np.array(frame["transform_matrix"]))
            if "mask_path" in frame:
                mask_filepath = PurePath(frame["mask_path"])
                mask_fname = self._get_fname(mask_filepath, downsample_folder_prefix="masks_")
                mask_filenames.append(mask_fname)
            if "leader_board" in meta and meta['leader_board']:
                index = str(fname).split('/')[-1]
                img_index.append(index)
        if num_skipped_image_filenames >= 0:
            CONSOLE.log(f"Skipping {num_skipped_image_filenames} files in dataset split {split}.")
        assert (
            len(image_filenames) != 0
        ), """
        No image files found. 
        You should check the file_paths in the transforms.json file to make sure they are correct.
        """
        assert len(mask_filenames) == 0 or (
            len(mask_filenames) == len(image_filenames)
        ), """
        Different number of image and mask filenames.
        You should check that mask_path is specified for every frame (or zero frames) in transforms.json.
        """

        # filter image_filenames and poses based on train/eval split percentage
        num_images = len(image_filenames)
        i_all = np.arange(num_images)
        ## 50% dropout Setting
        if "dropout" in meta and meta['dropout']:
            i_train = []
            for i in range(0, num_images, 2):
                if i % 4 == 0:
                    i_train.extend([i,i+1,i+2])
            i_train = np.array(i_train)
            num_train_images = len(i_train)
            i_eval = np.setdiff1d(i_all, i_train)[:-2]  # Demo kitti360
        elif "leader_board" in meta and meta['leader_board']:
            num_eval_images = int(meta['num_test'])
            i_train = i_all[:(num_images - num_eval_images)]
            i_eval = np.setdiff1d(i_all, i_train)
            # i_train = np.arange(10,79)
            # i_eval = np.arange(83,99)
        else:
            self.config.train_split_percentage = 0.8
            num_train_images = math.ceil(num_images * self.config.train_split_percentage)
            num_eval_images = num_images - num_train_images
            i_train = np.linspace(
                0, num_images - 1, num_train_images, dtype=int
            )  # equally spaced training images starting and ending at 0 and num_images-1
            i_eval = np.setdiff1d(i_all, i_train)  # eval images are the remaining images

        if split == "train":
            indices = i_train
            print(f"Train View:  {indices}" + f"Train View Num{len(i_train)}")
        elif split in ["val", "test"]:
            indices = i_eval
            print(f"Test View: {indices}" + f"Test View Num{len(i_eval)}")
        else:
            raise ValueError(f"Unknown dataparser split {split}")

        if "orientation_override" in meta:
            orientation_method = meta["orientation_override"]
            CONSOLE.log(f"[yellow] Dataset is overriding orientation method to {orientation_method}")
        else:
            orientation_method = self.config.orientation_method

        poses = torch.from_numpy(np.array(poses).astype(np.float32))

        diff_mean_poses = torch.mean(poses[:,:3,-1], dim=0)
        poses, _ = camera_utils.auto_orient_and_center_poses(
            poses,
            method=orientation_method,
            center_poses=self.config.center_poses,
        )

        # Scale poses[translation]
        scale_factor = 1.0
        if self.config.auto_scale_poses:
            scale_factor /= torch.max(torch.abs(poses[:, :3, 3]))

        poses[:, :3, 3] *= scale_factor * self.config.scale_factor

        # Choose image_filenames and poses based on split, but after auto orient and scaling the poses.
        image_filenames = [image_filenames[i] for i in indices]
        mask_filenames = [mask_filenames[i] for i in indices] if len(mask_filenames) > 0 else []
        poses = poses[indices]

        # in x,y,z order
        # assumes that the scene is centered at the origin
        aabb_scale = self.config.scene_scale
        scene_box = SceneBox(
            aabb=torch.tensor(
                [[-aabb_scale, -aabb_scale, -aabb_scale], [aabb_scale, aabb_scale, aabb_scale]], dtype=torch.float32
            )
        )

        if "camera_model" in meta:
            camera_type = CAMERA_MODEL_TO_TYPE[meta["camera_model"]]
        else:
            camera_type = CameraType.PERSPECTIVE

        idx_tensor = torch.tensor(indices, dtype=torch.long)
        fx = float(meta["fl_x"]) if fx_fixed else torch.tensor(fx, dtype=torch.float32)[idx_tensor]
        fy = float(meta["fl_y"]) if fy_fixed else torch.tensor(fy, dtype=torch.float32)[idx_tensor]
        cx = float(meta["cx"]) if cx_fixed else torch.tensor(cx, dtype=torch.float32)[idx_tensor]
        cy = float(meta["cy"]) if cy_fixed else torch.tensor(cy, dtype=torch.float32)[idx_tensor]
        height = int(meta["h"]) if height_fixed else torch.tensor(height, dtype=torch.int32)[idx_tensor]
        width = int(meta["w"]) if width_fixed else torch.tensor(width, dtype=torch.int32)[idx_tensor]
        if distort_fixed:
            distortion_params = camera_utils.get_distortion_params(
                k1=float(meta["k1"]) if "k1" in meta else 0.0,
                k2=float(meta["k2"]) if "k2" in meta else 0.0,
                k3=float(meta["k3"]) if "k3" in meta else 0.0,
                k4=float(meta["k4"]) if "k4" in meta else 0.0,
                p1=float(meta["p1"]) if "p1" in meta else 0.0,
                p2=float(meta["p2"]) if "p2" in meta else 0.0,
            )
        else:
            distortion_params = torch.stack(distort, dim=0)[idx_tensor]

        ## Where Use bounding box ,if Used,need to read instance image and bbx
        if meta['use_bbx']:
            print(f"BBx Abled!")
            bbx2world = np.array(meta["bbx2w"])
            bbox = []
            instance_imgs = []
            data_dir = '/data/datasets/KITTI-360/'
            instance_path = os.path.join(data_dir, 'data_2d_semantics', 'train',
                                         '2013_05_28_drive_0000_sync', 'image_00/instance')
            for idx in range(3353, 3353 + 10, 1):
                img_file = os.path.join(instance_path, "{:010d}.png".format(idx))
                instance_imgs.append(cv.imread(img_file, -1))

            bbx_root = os.path.join(data_dir, 'data_3d_bboxes')
            self.annotation_3d = Annotation3D(os.path.join(bbx_root, 'train'), '2013_05_28_drive_0000_sync')
            all_bbxes = self.load_bbx(instance_imgs=instance_imgs, bbx2w=bbx2world,
                                      scale=scale_factor * self.config.scale_factor,
                                      diff_centor_translation=diff_mean_poses)

            # self.project2Dbbx(bbx=all_bbxes, img_idx=0, f=fx, cx=cx, cy=cy, img_file=image_filenames)
        else:
            print(f"BBx Unabled!")

        cameras = Cameras(
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            distortion_params=distortion_params,
            height=height,
            width=width,
            camera_to_worlds=poses[:, :3, :4],
            camera_type=camera_type,
            bounding_box=all_bbxes,
            test_idx= i_eval,
            train_idx= i_train,
        )

        assert self.downscale_factor is not None
        cameras.rescale_output_resolution(scaling_factor=1.0 / self.downscale_factor)

        dataparser_outputs = DataparserOutputs(
            image_filenames=image_filenames,
            cameras=cameras,
            scene_box=scene_box,
            mask_filenames=mask_filenames if len(mask_filenames) > 0 else None,
        )
        return dataparser_outputs

    def _get_fname(self, filepath: PurePath, downsample_folder_prefix="images_") -> Path:
        """Get the filename of the image file.
        downsample_folder_prefix can be used to point to auxillary image data, e.g. masks
        """

        if self.downscale_factor is None:
            if self.config.downscale_factor is None:
                test_img = Image.open(self.config.data / filepath)
                h, w = test_img.size
                max_res = max(h, w)
                df = 0
                while True:
                    if (max_res / 2 ** (df)) < MAX_AUTO_RESOLUTION:
                        break
                    if not (self.config.data / f"{downsample_folder_prefix}{2**(df+1)}" / filepath.name).exists():
                        break
                    df += 1

                self.downscale_factor = 2**df
                CONSOLE.log(f"Auto image downscale factor of {self.downscale_factor}")
            else:
                self.downscale_factor = self.config.downscale_factor

        if self.downscale_factor > 1:
            return self.config.data / f"{downsample_folder_prefix}{self.downscale_factor}" / filepath.name
        return self.config.data / filepath


    def load_bbx(self,instance_imgs = None,bbx2w =None,scale = 1,diff_centor_translation=0):
        num_bbx = len(instance_imgs)

        w2c = np.linalg.inv(bbx2w)
        R_w2c = w2c[:3, :3]
        t_w2c = w2c[:3, 3:]
        all_bbxes = []
        for img_id in range(num_bbx):
            instance_map = instance_imgs[img_id]
            car_global = instance_map[(instance_map > 26 * 1000) & (instance_map < 27 * 1000)]  ## Car 的global id 在26000-27000之间
            set_idx = np.unique(car_global)

            ## 找出该张图像对应的bbx 的 8个顶点（w系）
            vertices = []
            for idx in set_idx:
                vertice = self.annotation_3d.objects[idx][-1].vertices
                vertices.append(vertice)

            vertices_w = np.array(vertices)[..., None]  ## 世界系下的 vertices
            vertices_c = np.matmul(R_w2c[None, None, ...], vertices_w) + t_w2c[None, None, ...]  ## 当前图像的 相机系下的 vertices
            vertices_c = torch.from_numpy(vertices_c) - diff_centor_translation[...,None]       ## Centor Pose
            all_bbxes.append(vertices_c * scale)
        return all_bbxes

    def project2Dbbx(self, bbx=None, img_idx=-1,f=0, cx=0, cy=0,img_file = None):
        intrinsics_all = np.array([
            [f, 0, cx, 0],
            [0, f, cy, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        vertices = bbx[img_idx]
        bbx_img = cv.imread(str(img_file[img_idx*2]))
        img = np.array(bbx_img).astype(np.float32).copy()
        for idx in range(vertices.shape[0]):
            uv = np.matmul(intrinsics_all[None, :3, :3], vertices[idx, ...]).squeeze(-1)
            uv[:, :2] = (uv[:, :2] / uv[:, -1:])  ## 最后一维度归一化为1

            left = int(min(uv[:, 0]))
            right = int(max(uv[:, 0]))
            top = int(min(uv[:, 1]))
            bottom = int(max(uv[:, 1]))
            img[top:bottom, left:right] = 0
        cv.imwrite('projectbbx.png', np.concatenate((img , bbx_img ), axis=0))
        exit()

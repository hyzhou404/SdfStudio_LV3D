#!/usr/bin/env python
"""
render.py
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import mediapy as media
import numpy as np
import torch
import tyro
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)
from typing_extensions import Literal, assert_never

from nerfstudio.cameras.camera_paths import get_path_from_json, get_spiral_path
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.configs.base_config import Config  # pylint: disable=unused-import
from nerfstudio.pipelines.base_pipeline import Pipeline
from nerfstudio.utils import install_checks
from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.utils.rich_utils import ItersPerSecColumn
from nerfstudio.utils import plotly_utils as vis

CONSOLE = Console(width=120)


def _render_trajectory_video(
    pipeline: Pipeline,
    cameras: Cameras,
    output_filename: Path,
    rendered_output_names: List[str],
    rendered_resolution_scaling_factor: float = 1.0,
    seconds: float = 5.0,
    output_format: Literal["images", "video"] = "video",
) -> None:
    """Helper function to create a video of the spiral trajectory.

    Args:
        pipeline: Pipeline to evaluate with.
        cameras: Cameras to render.
        output_filename: Name of the output file.
        rendered_output_names: List of outputs to visualise.
        rendered_resolution_scaling_factor: Scaling factor to apply to the camera image resolution.
        seconds: Length of output video.
        output_format: How to save output data.
    """
    CONSOLE.print("[bold green]Creating trajectory video")
    images = []
    cameras.rescale_output_resolution(rendered_resolution_scaling_factor)
    cameras = cameras.to(pipeline.device)

    progress = Progress(
        TextColumn(":movie_camera: Rendering :movie_camera:"),
        BarColumn(),
        TaskProgressColumn(show_speed=True),
        ItersPerSecColumn(suffix="fps"),
        TimeRemainingColumn(elapsed_when_finished=True, compact=True),
    )
    output_image_dir = output_filename.parent / output_filename.stem
    if output_format == "images":
        output_image_dir.mkdir(parents=True, exist_ok=True)
    with progress:
        for camera_idx in progress.track(range(cameras.size), description=""):
            camera_ray_bundle = cameras.generate_rays(camera_indices=camera_idx)
            with torch.no_grad():
                ## 这个output 里面包括 rgb map, depth map,nomal map 等
                outputs = pipeline.model.get_outputs_for_camera_ray_bundle(camera_ray_bundle)
            render_image = []
            for rendered_output_name in rendered_output_names:
                if rendered_output_name not in outputs:
                    CONSOLE.rule("Error", style="red")
                    CONSOLE.print(f"Could not find {rendered_output_name} in the model outputs", justify="center")
                    CONSOLE.print(f"Please set --rendered_output_name to one of: {outputs.keys()}", justify="center")
                    sys.exit(1)
                output_image = outputs[rendered_output_name].cpu().numpy()
                render_image.append(output_image)
            render_image = np.concatenate(render_image, axis=1)
            if output_format == "images":
                media.write_image(output_image_dir / f"{camera_idx:05d}.png", render_image)
            else:
                images.append(render_image)

    if output_format == "video":
        fps = len(images) / seconds
        # make the folder if it doesn't exist
        output_filename.parent.mkdir(parents=True, exist_ok=True)
        with CONSOLE.status("[yellow]Saving video", spinner="bouncingBall"):
            media.write_video(output_filename, images, fps=fps)
    CONSOLE.rule("[green] :tada: :tada: :tada: Success :tada: :tada: :tada:")
    CONSOLE.print(f"[green]Saved video to {output_filename}", justify="center")


@dataclass
class RenderTrajectory:
    """Load a checkpoint, render a trajectory, and save to a video file."""

    # Path to config YAML file.
    load_config: Path
    # Name of the renderer outputs to use. rgb, depth, etc. concatenates them along y axis
    rendered_output_names: List[str] = field(default_factory=lambda: ["rgb"])
    #  Trajectory to render.
    traj: Literal["spiral", "filename"] = "spiral"
    # Scaling factor to apply to the camera image resolution.
    downscale_factor: int = 1
    # Filename of the camera path to render.
    camera_path_filename: Path = Path("camera_path.json")
    # Name of the output file.
    output_path: Path = Path("renders/output.mp4")
    # How long the video should be.
    seconds: float = 8.0
    # How to save output data.
    output_format: Literal["images", "video"] = "video"
    # Specifies number of rays per chunk during eval.
    eval_num_rays_per_chunk: Optional[int] = None


    def render_sptial_view(self, camera: Cameras,steps: int = 30 ,rots: int = 2, zrate: float = 1,radius: Optional[float] = None):

        N_per_seg = 10
        new_c2ws = []
        ## 每次间隔4帧，选取首帧和末帧，转动 360 角度
        for i in range(0,len(camera)-1,4):
            start_point = camera[i].camera_to_worlds[:,3].detach().cpu().numpy()
            end_point = camera[i+3].camera_to_worlds[:, 3].detach().cpu().numpy()

            new_z = np.linspace(start_point[2],end_point[2],N_per_seg)
            new_xyz = []
            for theta in np.linspace(0., 2*np.pi, N_per_seg + 1)[:-1]:
                x = radius * np.cos(theta) + (start_point[0] + end_point[0]) * 0.5
                y = radius * np.sin(-theta) + (start_point[1] + end_point[1]) * 0.5
                new_xyz.append(np.array([x,y]))
            new_xyz = np.concatenate([np.array(new_xyz),new_z[...,None]],axis=1)
            camera_pose = np.eye(4)
            camera_pose = camera_pose[None,...].repeat(repeats=N_per_seg,axis = 0)
            camera_pose[:,:3,:3] = camera[i].camera_to_worlds[:3,:3].detach().cpu().numpy()
            camera_pose[:,:3,3] = new_xyz
            new_c2ws.append(camera_pose)

        new_c2ws = torch.tensor(new_c2ws).reshape(-1,4,4) ##[B,N,4,4]---> [B*N,4,4]

        return Cameras(
            fx=camera[0].fx[0],
            fy=camera[0].fy[0],
            cx=camera[0].cx[0],
            cy=camera[0].cy[0],
            height=camera[0].height,
            width=camera[0].width,
            distortion_params=camera[0].distortion_params,
            camera_type=camera[0].camera_type,
            camera_to_worlds=new_c2ws[:,:3,:4],
        )


    def main(self) -> None:
        """Main function."""
        _, pipeline, _ = eval_setup(
            self.load_config,
            eval_num_rays_per_chunk=self.eval_num_rays_per_chunk,
            test_mode="test" if self.traj == "spiral" else "inference",
        )

        install_checks.check_ffmpeg_installed()

        seconds = self.seconds

        # TODO(ethan): use camera information from parsing args
        if self.traj == "spiral":
            camera_start = pipeline.datamanager.eval_dataloader.get_camera(image_idx=0).flatten()
            num_cameras = len(pipeline.datamanager.eval_dataset.filenames)
            cameras = [pipeline.datamanager.eval_dataloader.get_camera(image_idx=i) for i in range(num_cameras)]
            # TODO(ethan): pass in the up direction of the camera
            # camera_path = get_spiral_path(camera_start, steps=30, radius=0.1)
            camera_path = self.render_sptial_view(cameras, steps=30, radius=0.01)
        elif self.traj == "filename":
            with open(self.camera_path_filename, "r", encoding="utf-8") as f:
                camera_path = json.load(f)
            seconds = camera_path["seconds"]
            camera_path = get_path_from_json(camera_path)
        else:
            assert_never(self.traj)

        '将相机的参数信息，送至这个函数进行渲染'
        # self.rendered_output_names = ['depth']
        # _render_trajectory_video(pipeline,camera_path[:1,...],
        #     output_filename=self.output_path,
        #     rendered_output_names=self.rendered_output_names,
        #     rendered_resolution_scaling_factor=1.0 / 2,
        #     seconds=seconds,
        #     output_format= "images",
        # )
        _render_trajectory_video(
            pipeline,
            camera_path,
            output_filename=self.output_path,
            rendered_output_names=self.rendered_output_names,
            rendered_resolution_scaling_factor=1.0 / self.downscale_factor,
            seconds=seconds,
            output_format=self.output_format,
        )


def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(RenderTrajectory).main()


if __name__ == "__main__":
    entrypoint()

# For sphinx docs
get_parser_fn = lambda: tyro.extras.get_parser(RenderTrajectory)  # noqa

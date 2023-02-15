from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional
import matplotlib.pyplot as plt

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
from mpl_toolkits.axes_grid1 import make_axes_locatable
import argparse
import cv2

CONSOLE = Console(width=120)

class RenderDatasets():
    """Load a checkpoint, render the trainset and testset rgb,normal,depth, and save to the picture"""
    def __init__(self,parser_path):
        self.load_config = Path(parser_path.config)
        # self.load_config = Path("outputs/datasets-kitti360_Neus/neus/2023-02-08_112306/config.yml")
        self.rendered_output_names = ['rgb','depth','normal']
       #  self.rendered_output_names = ['rgb_fine', 'depth_fine']
        exp_method = str(self.load_config).split('/')[-3]
        self.root_dir = Path('exp_psnr_' + exp_method)
        self.task = parser_path.task



    def main(self):
        config, pipeline, _ = eval_setup(
            self.load_config,
            test_mode= "test",
        )
        trainDataCache = pipeline.datamanager.train_dataset
        testDatasetCache = pipeline.datamanager.eval_dataset

        if self.task == 'trainset':
            DataCache = trainDataCache
        elif self.task == 'testset':
            DataCache = testDatasetCache
        else:
            raise print("Task Input is trainset or testset")


        'Read the image and save in target directory'
        os.makedirs(self.root_dir,exist_ok=True)
        CONSOLE.print(f"[bold green]Rendering {len(DataCache.image_cache)} Images")
        cameras = DataCache.cameras.to(pipeline.device)

        progress = Progress(
            TextColumn(":movie_camera: Rendering :movie_camera:"),
            BarColumn(),
            TaskProgressColumn(show_speed=True),
            ItersPerSecColumn(suffix="fps"),
            TimeRemainingColumn(elapsed_when_finished=True, compact=True),
        )
        render_image = []
        render_depth = []
        render_normal = []
        with progress:
            for camera_idx in progress.track(range(cameras.size), description=""):
                camera_ray_bundle = cameras.generate_rays(camera_indices=camera_idx)
                with torch.no_grad():
                    outputs = pipeline.model.get_outputs_for_camera_ray_bundle(camera_ray_bundle)

                for rendered_output_name in self.rendered_output_names:
                    if rendered_output_name not in outputs:
                        CONSOLE.rule("Error", style="red")
                        CONSOLE.print(f"Could not find {rendered_output_name} in the model outputs", justify="center")
                        CONSOLE.print(f"Please set --rendered_output_name to one of: {outputs.keys()}",
                                      justify="center")
                        sys.exit(1)
                    output_image = outputs[rendered_output_name].cpu().numpy()
                    if rendered_output_name == 'rgb':
                        render_image.append(output_image)
                    elif rendered_output_name == 'depth':
                        render_depth.append(output_image)
                    elif rendered_output_name == 'normal':
                        render_normal.append(output_image)
        CONSOLE.print("[bold green]Rendering Images Finished")

        ''' Output rgb depth and normal image'''
        sum_psnr = 0
        for i,image in sorted(DataCache.image_cache.items()):
            media.write_image(self.root_dir/f'{self.task}_{i:02d}_rgb.png',np.concatenate((image.detach().cpu().numpy(),render_image[i]),axis=0))
            psnr = -10. * np.log10(np.mean(np.square(image.detach().cpu().numpy() - render_image[i])))
            sum_psnr += psnr
            print("{} Mode image {} PSNR:{}".format(self.task,i,psnr))
        print(f"Average PSNR:{sum_psnr/len(DataCache.image_cache)}")

        for i in range(len(render_depth)):
            pred_depth = render_depth[i].squeeze(2)
            pred_depth = pred_depth.clip(0,20)
            print(f"Predecited Depth Max:{pred_depth.max()}  clip max = 20 ")
            # plt.imsave(str(self.root_dir)+f'/{self.task}_{i:02d}_depth.png', ((pred_depth / pred_depth.max()) * 255).astype(np.uint8), cmap='viridis')
            ax = plt.subplot()
            sc = ax.imshow((pred_depth), cmap='viridis')
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(sc, cax=cax)
            plt.savefig(os.path.join(str(self.root_dir)+f'/{self.task}_{i:02d}_depth.png'))
            plt.close('all')
            # pred_depth = cv2.applyColorMap(cv2.convertScaleAbs(((pred_depth / pred_depth.max()) * 255).astype(np.uint8), alpha=2), cv2.COLORMAP_JET)
            # cv2.imwrite(str(self.root_dir)+f'/{self.task}_{i:02d}_depth.png',pred_depth)
            if "normal" in self.rendered_output_names:
                media.write_image(self.root_dir/f'{self.task}_{i:02d}_normal.png', render_normal[i])
        CONSOLE.print(f"[bold blue] Store image to {self.root_dir}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, help='testset or trainset')
    parser.add_argument('--config',type=str,help='Config Path')
    config = parser.parse_args()

    RenderDatasets(config).main()
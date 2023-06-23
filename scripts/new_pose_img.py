import torch
from pathlib import Path
import open3d as o3d
import torch
from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.cameras.cameras import Cameras, CameraType
from imageio.v2 import imwrite
import json
import numpy as np

load_config = Path('outputs/3353_bgm/nerfacto/2023-06-16_204725/config.yml')
# load_config = Path('outputs/3353_lidar_2/unisurf/2023-06-20_205342/config.yml')
config, pipeline, checkpoint_path = eval_setup(load_config)

with open('/data/hyzhou/data/kitti_neus_v2/frame50_3353_lidar/meta_data.json', 'r') as rf:
    meta_data = json.load(rf)
poses = []
for frame in meta_data['frames']:
    poses.append(frame['camtoworld'])
poses = torch.tensor(poses)
# Important: To NDC
poses[:, 0:3, 1:3] *= -1

camera_idx = 41
origin_pose = poses[camera_idx, :3, :4]
new_pose = torch.clone(origin_pose)
new_pose[:3, 3] += torch.tensor([0.15, 0, 0])
rot = -0 * np.pi
xz_rotation_matrix = torch.tensor([[np.cos(rot), 0, -np.sin(rot)], [0, 1, 0], [np.sin(rot), 0, np.cos(rot)]]).float()
new_pose[:3, :3] = new_pose[:3, :3] @ xz_rotation_matrix

cx = 682.049453
cy = 238.769549
fx = 552.554261
fy = 552.554261
height = 376
width = 1408

output_images = []
for pose in [origin_pose, new_pose]:
    cameras = Cameras(
                fx=fx,
                fy=fy,
                cx=cx,
                cy=cy,
                height=height,
                width=width,
                camera_to_worlds=pose,
                camera_type=CameraType.PERSPECTIVE,
            )
    ray_bundle = cameras.generate_rays(camera_indices=0, keep_shape=True).reshape((height, width)).to(pipeline.model.device)
    outputs = pipeline.model.get_outputs_for_camera_ray_bundle(ray_bundle)
    output_images.append((outputs['rgb'].detach().cpu().numpy()*255).astype(np.uint8))

imwrite('./exports/pose2img/origin.png', output_images[0])
imwrite('./exports/pose2img/new.png', output_images[1])
cameras = o3d.geometry.TriangleMesh()
for p in poses:
    cone = cameras.create_cone(radius=0.01, height=0.02)
    cone.paint_uniform_color([1.0, 0.0, 0.0])
    cone.translate(p[:3, -1])
    cameras += cone
new_pose_cone = cameras.create_cone(radius=0.01, height=0.02)
new_pose_cone.paint_uniform_color([0.0, 0.0, 1.0])
new_pose_cone.translate(new_pose[:3, -1])
cameras += new_pose_cone
o3d.io.write_triangle_mesh("./exports/pose2img/vis_cam.ply", cameras)

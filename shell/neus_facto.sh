CUDA_VISIBLE_DEVICES=4 \
ns-train unisurf --pipeline.model.sdf-field.inside-outside True \
                    --vis tensorboard \
                    --experiment-name 3353_backbone \
                    --discription s21f4_h256_SH_Ref \
                    sdfstudio-data --data /data/hyzhou/data/kitti_neus_v2/frame50_3353_lidar

#CUDA_VISIBLE_DEVICES=3 \
#ns-train nerfacto   --vis tensorboard \
#                    --experiment-name 3353_bgm \
#                    sdfstudio-data --data /data/hyzhou/data/kitti_neus_v2/frame50_3353_lidar

#CUDA_VISIBLE_DEVICES=1 \
#ns-train unisurf --pipeline.model.sdf-field.inside-outside True \
#                    --pipeline.model.mono-depth-loss-mult 0 \
#                    --pipeline.model.mono-normal-loss-mult 0.01 \
#                    --vis tensorboard \
#                    --experiment-name no_lidar \
#                    sdfstudio-data --data /data/hyzhou/data/kitti_neus_v2/frame50_3390_velodyne

#CUDA_VISIBLE_DEVICES=3 \
#ns-train unisurf --pipeline.model.sdf-field.inside-outside True \
#                    --pipeline.model.mono-depth-loss-mult 0 \
#                    --pipeline.model.mono-normal-loss-mult 0.01 \
#                    --trainer.load-dir outputs/lidar_smooth/unisurf/2023-05-23_154714/sdfstudio_models \
#                    --vis tensorboard \
#                    --experiment-name lidar_smooth \
#                    sdfstudio-data --data /data/hyzhou/data/kitti_neus_v2/frame50_3390_lidar

#CUDA_VISIBLE_DEVICES=1 \
#ns-train neus-facto --pipeline.model.sdf-field.inside-outside True \
#                    --vis tensorboard \
#                    --trainer.load-dir outputs/normal/neus-facto/2023-05-15_132629/sdfstudio_models \
#                    --experiment-name new_scene \
#                    sdfstudio-data --data /data/hyzhou/data/kitti_neus/frame20_8020

#CUDA_VISIBLE_DEVICES=1 \
#ns-train neus-facto --pipeline.model.sdf-field.inside-outside True \
#                    --pipeline.model.mono-normal-loss-mult 0 \
#                    --pipeline.model.mono-depth-loss-mult 0.1 \
#                    --vis tensorboard \
#                    --experiment-name dev \
#                    sdfstudio-data --data /data/hyzhou/data/kitti_neus/frame20_3390
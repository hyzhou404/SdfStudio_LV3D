CUDA_VISIBLE_DEVICES=1 \
ns-train neus-facto --pipeline.model.sdf-field.inside-outside True \
                    --pipeline.model.mono-depth-loss-mult 0 \
                    --pipeline.model.mono-normal-loss-mult 0.01 \
                    --vis tensorboard \
                    --experiment-name road \
                    sdfstudio-data --data /data/hyzhou/data/kitti_neus_v2/frame20_3390_v2

#CUDA_VISIBLE_DEVICES=1 \
#ns-train neus-facto --pipeline.model.sdf-field.inside-outside True \
#                    --vis tensorboard \
#                    --experiment-name new_scene \
#                    sdfstudio-data --data /data/hyzhou/data/kitti_neus/frame20_8020

#CUDA_VISIBLE_DEVICES=1 \
#ns-train neus-facto --pipeline.model.sdf-field.inside-outside True \
#                    --pipeline.model.mono-normal-loss-mult 0 \
#                    --pipeline.model.mono-depth-loss-mult 0.1 \
#                    --vis tensorboard \
#                    --experiment-name dev \
#                    sdfstudio-data --data /data/hyzhou/data/kitti_neus/frame20_3390
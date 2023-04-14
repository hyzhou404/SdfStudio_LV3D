#CUDA_VISIBLE_DEVICES=2 \
#ns-train neus-facto --pipeline.model.sdf-field.inside-outside False \
#                    --vis viewer --viewer.websocket-port 8008 \
#                    --experiment-name neus-facto-dtu65 sdfstudio-data \
#                    --data /data/datasets/DTU/scan65/

#CUDA_VISIBLE_DEVICES=2 \
#ns-train neus-facto --pipeline.model.sdf-field.inside-outside False \
#                    --vis tensorboard \
#                    --trainer.max-num-iterations 40000 \
#                    --optimizers.proposal-networks.scheduler.max-steps 40000 \
#                    --optimizers.fields.scheduler.max-steps 40000 \
#                    --optimizers.field-background.scheduler.max-steps 40000 \
#                    --experiment-name kitti360 \
#                    sdfstudio-data --data /data/hyzhou/data/kitti/kitti360_neus

#CUDA_VISIBLE_DEVICES=2 \
#ns-train neus-facto --pipeline.model.sdf-field.inside-outside False \
#                    --vis tensorboard \
#                    --experiment-name kitti360_50frames_whitensky \
#                    sdfstudio-data --data /data/hyzhou/data/kitti_neus/kitti360_50_whitensky

#CUDA_VISIBLE_DEVICES=2 \
#ns-train neus --pipeline.model.sdf-field.inside-outside False \
#                    --pipeline.datamanager.eval-num-rays-per-batch 1024 \
#                    --pipeline.model.eval-num-rays-per-chunk 1024 \
#                    --vis tensorboard \
#                    --experiment-name kitti360_short \
#                    sdfstudio-data --data /data/hyzhou/data/kitti/kitti360_mono_priors

#CUDA_VISIBLE_DEVICES=3 \
#ns-train neus-acc --pipeline.model.sdf-field.inside-outside False \
#                  --pipeline.datamanager.train-num-rays-per-batch 1024 \
#                  --vis tensorboard \
#                  --experiment-name kitti360_short\
#                  sdfstudio-data --data /data/hyzhou/data/kitti/kitti360_mono_priors

#CUDA_VISIBLE_DEVICES=2 \
#ns-train bakedsdf --pipeline.model.sdf-field.inside-outside False \
#                  --pipeline.datamanager.train-num-rays-per-batch 4096 \
#                  --vis tensorboard \
#                  --experiment-name kitti360 sdfstudio-data \
#                  --data /data/hyzhou/data/kitti/kitti360_neus

CUDA_VISIBLE_DEVICES=1 \
ns-train bakedsdf --pipeline.model.sdf-field.inside-outside False \
                  --pipeline.datamanager.train-num-rays-per-batch 4096 \
                  --trainer.max-num-iterations 50000 \
                  --vis tensorboard \
                  --experiment-name kitti360_50frames_whitensky \
                  sdfstudio-data --data /data/hyzhou/data/kitti_neus/kitti360_50_whitensky
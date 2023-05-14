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

#CUDA_VISIBLE_DEVICES=1 \
#ns-train neus-facto --pipeline.model.sdf-field.inside-outside False \
#                    --vis tensorboard \
#                    --experiment-name kitti_10frames_ws \
#                    sdfstudio-data --data /data/hyzhou/data/kitti_neus/frame10_ws

#CUDA_VISIBLE_DEVICES=1 \
#ns-train neus-facto --pipeline.model.sdf-field.inside-outside True \
#                    --pipeline.model.mono-normal-loss-mult 0 \
#                    --pipeline.model.mono-depth-loss-mult 1e-5 \
#                    --vis tensorboard \
#                    --experiment-name dev \
#                    sdfstudio-data --data /data/hyzhou/data/kitti_neus/frame20_mono

CUDA_VISIBLE_DEVICES=1 \
ns-train neus-facto --pipeline.model.sdf-field.inside-outside True \
                    --pipeline.model.mono-normal-loss-mult 0 \
                    --pipeline.model.mono-depth-loss-mult 0.01 \
                    --vis tensorboard \
                    --experiment-name density_url_dp \
                    sdfstudio-data --data /data/hyzhou/data/kitti_neus/frame20_3390

#CUDA_VISIBLE_DEVICES=1 \
#ns-train volsdf --pipeline.model.sdf-field.inside-outside False \
#                --pipeline.model.mono-normal-loss-mult 0 \
#                --pipeline.model.mono-depth-loss-mult 0 \
#                --trainer.max-num-iterations 20000 \
#                --vis tensorboard \
#                --experiment-name dev \
#                sdfstudio-data --data /data/hyzhou/data/kitti_neus/frame20_mono

#CUDA_VISIBLE_DEVICES=2 \
#ns-train neus --pipeline.model.sdf-field.inside-outside False \
#                    --pipeline.datamanager.eval-num-rays-per-batch 1024 \
#                    --pipeline.model.eval-num-rays-per-chunk 1024 \
#                    --vis viewer \
#                    --experiment-name kitti360_40frame_raw \
#                    sdfstudio-data --data /data/hyzhou/data/kitti_neus/frame40_raw

#CUDA_VISIBLE_DEVICES=3 \
#ns-train neus-acc --pipeline.model.sdf-field.inside-outside False \
#                  --pipeline.datamanager.train-num-rays-per-batch 1024 \
#                  --vis tensorboard \
#                  --experiment-name kitti360_short\
#                  sdfstudio-data --data /data/hyzhou/data/kitti/kitti360_mono_priors

#CUDA_VISIBLE_DEVICES=2 \
#ns-train bakedsdf --pipeline.model.sdf-field.inside-outside False \
#                  --pipeline.datamanager.train-num-rays-per-batch 4096 \
#                  --trainer.max-num-iterations 50000 \
#                  --vis tensorboard \
#                  --experiment-name kitti360_40frame_raw \
#                  sdfstudio-data --data /data/hyzhou/data/kitti_neus/frame40_raw

#CUDA_VISIBLE_DEVICES=1 \
#ns-train nerfacto --data /data/hyzhou/data/kitti_nerfacto/frame50 \
#                  --vis tensorboard \
#                  --experiment_name kitti_50frames \
#                  --pipeline.datamanager.camera-optimizer.mode off
#CUDA_VISIBLE_DEVICES=4 \
#ns-train neus-facto --pipeline.model.sdf-field.geometric_init False \
#                    --pipeline.model.mono-depth-loss-mult 0 \
#                    --pipeline.model.mono-normal-loss-mult 0.01 \
#                    --pipeline.datamanager.train-num-images-to-sample-from -1 \
#                    --pipeline.datamanager.train-num-times-to-repeat-images -1 \
#                    --vis tensorboard \
#                    --output-dir /data4/hyzhou/outputs/waymo \
#                    --experiment-name 100613_300f_rear \
#                    --discription fullnormal_llloww \
#                    sdfstudio-data --data /data4/datasets/waymo/neus/100613_120_300frame_rear60


CUDA_VISIBLE_DEVICES=5 \
ns-train nerfacto   --pipeline.datamanager.train-num-images-to-sample-from -1 \
                    --pipeline.datamanager.train-num-times-to-repeat-images -1 \
                    --vis tensorboard \
                    --output-dir /data4/hyzhou/outputs/waymo \
                    --experiment-name 100613_300f_rear30 \
                    --discription full \
                    sdfstudio-data --data /data4/datasets/waymo/neus/100613_120_300frame_rear30


#CUDA_VISIBLE_DEVICES=1 \
#ns-train neus-facto --pipeline.model.sdf-field.geometric_init False \
#                    --pipeline.model.mono-depth-loss-mult 0 \
#                    --pipeline.model.mono-normal-loss-mult 0.05 \
#                    --vis tensorboard \
#                    --output-dir /data4/hyzhou/outputs/waymo \
#                    --experiment-name localrf \
#                    --discription dev \
#                    sdfstudio-data --data /data4/datasets/waymo/neus/100613_120_300frame_rear60

#CUDA_VISIBLE_DEVICES=4 \
#ns-train unisurf --pipeline.model.sdf-field.inside-outside True \
#                    --vis tensorboard \
#                    --experiment-name w100613_300f \
#                    sdfstudio-data --data /data/datasets/waymo/neus/100613_300frame_front

#CUDA_VISIBLE_DEVICES=4 \
#ns-train nerfacto   --vis tensorboard \
#                    --experiment-name w100613_300f \
#                    sdfstudio-data --data /data/datasets/waymo/neus/100613_300frame_front
CUDA_VISIBLE_DEVICES=4 \
ns-train unisurf --pipeline.model.sdf-field.inside-outside True \
                    --vis tensorboard \
                    --experiment-name w100613_150f \
                    sdfstudio-data --data /data/datasets/waymo/neus/100613_150frame_front

#CUDA_VISIBLE_DEVICES=4 \
#ns-train unisurf --pipeline.model.sdf-field.inside-outside True \
#                    --vis tensorboard \
#                    --experiment-name w100613_300f \
#                    sdfstudio-data --data /data/datasets/waymo/neus/100613_300frame_front

#CUDA_VISIBLE_DEVICES=4 \
#ns-train nerfacto   --vis tensorboard \
#                    --experiment-name w100613_300f \
#                    sdfstudio-data --data /data/datasets/waymo/neus/100613_300frame_front
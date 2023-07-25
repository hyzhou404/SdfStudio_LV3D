# CUDA_VISIBLE_DEVICES=2 \
# ns-train neus-facto --pipeline.model.sdf-field.geometric_init False \
#                     --pipeline.model.mono-depth-loss-mult 0 \
#                     --pipeline.model.mono-normal-loss-mult 0.01 \
#                     --pipeline.datamanager.train-num-images-to-sample-from -1 \
#                     --pipeline.datamanager.train-num-times-to-repeat-images -1 \
#                     --vis tensorboard \
#                     --output-dir /data4/hyzhou/outputs/kitti \
#                     --experiment-name 3360_f60_d50 \
#                     --discription udf_noskyacc \
#                     sdfstudio-data --data /data4/hyzhou/data/kitti_neus/3353_f60_aabb

CUDA_VISIBLE_DEVICES=4 \
ns-train nerfacto --pipeline.datamanager.train-num-images-to-sample-from -1 \
                   --pipeline.datamanager.train-num-times-to-repeat-images -1 \
                   --vis tensorboard \
                   --output-dir /data4/hyzhou/outputs/kitti \
                   --experiment-name 3353_f60_d50 \
                   --discription specular2 \
                   sdfstudio-data --data /data4/hyzhou/data/kitti_neus/3353_f60_aabb

#CUDA_VISIBLE_DEVICES=4 \
#ns-train nerfacto --pipeline.datamanager.train-num-images-to-sample-from -1 \
#                    --pipeline.datamanager.train-num-times-to-repeat-images -1 \
#                    --vis tensorboard \
#                    --output-dir /data4/hyzhou/outputs/kitti \
#                    --experiment-name 3360_f60_d50 \
#                    --discription rear \
#                    --trainer.load-dir /data4/hyzhou/outputs/kitti/3360_f60_d50/nerfacto/same_setting/sdfstudio_models \
#                    sdfstudio-data --data /data4/hyzhou/data/kitti_neus/3353_f60_aabb
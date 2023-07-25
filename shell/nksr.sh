CUDA_VISIBLE_DEVICES=4 \
ns-train nksr   --pipeline.datamanager.train-num-images-to-sample-from -1 \
                --pipeline.datamanager.train-num-times-to-repeat-images -1 \
                --vis tensorboard \
                --output-dir /data4/hyzhou/outputs/dtu \
                --experiment-name nksr \
                --discription grid \
                sdfstudio-data --data /data4/datasets/DTU/snowman

#CUDA_VISIBLE_DEVICES=4 \
#ns-train nksr   --pipeline.datamanager.train-num-images-to-sample-from -1 \
#                --pipeline.datamanager.train-num-times-to-repeat-images -1 \
#                --vis tensorboard \
#                --output-dir /data4/hyzhou/outputs/kitti_nksr \
#                --experiment-name nksr \
#                --discription dev \
#                sdfstudio-data --data /data4/hyzhou/data/kitti_neus/3353_f60_aabb

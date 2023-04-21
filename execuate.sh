## Neus Render
# python scripts/only_inference.py --config outputs/datasets-kitti360_Neus/neus/2023-03-31_100645/config.yml --task testset

## Nefacto Train
python scripts/train.py nerfacto --pipeline.model.collider-params near_plane 0.0 far_plane 6.0
        --pipeline.datamanager.camera-optimizer.mode off --vis tensorboard --trainer.max-num-iterations 20000 --data datasets/Car_bbx

## Nerfaco Render (camera pose refinement off, orientened pose = 'none')
python scripts/only_inference.py --config outputs/datasets-kitti360_nerfacto/nerfacto/2023-04-10_100018/config.yml --task testset


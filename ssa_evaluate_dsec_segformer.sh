# python scripts/evaluation.py --gt_path data/cityscapes/gtFine/val/ --result_path output_cityscapes/ --dataset cityscapes
export CUDA_VISIBLE_DEVICES=2,3
scene='night'
# 运行Python脚本

python scripts/evaluation_segformer.py \
    --scene ${scene} \
    --gt_path /home/xiaoshan/work/adap_v/my_proj/data/DSEC/${scene}/gtFine/val/ \
    --result_path dsec_${scene}_val_19/ \
    --dataset dsec
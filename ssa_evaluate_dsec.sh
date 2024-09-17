# python scripts/evaluation.py --gt_path data/cityscapes/gtFine/val/ --result_path output_cityscapes/ --dataset cityscapes
export CUDA_VISIBLE_DEVICES=2,3

# 运行Python脚本

python scripts/evaluation.py \
    --scene night \
    --gt_path /home/xiaoshan/work/adap_v/my_proj/data/DSEC/night/gtFine/val/ \
    --result_path dsec_evlight_night_val_19/ \
    --dataset dsec
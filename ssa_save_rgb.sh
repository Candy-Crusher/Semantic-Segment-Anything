# python scripts/evaluation.py --gt_path data/cityscapes/gtFine/val/ --result_path output_cityscapes/ --dataset cityscapes
export CUDA_VISIBLE_DEVICES=2,3

# 运行Python脚本

python scripts/save_cs_style_seg_output.py \
    --gt_path data/dsec/gtFine/val/ \
    --result_path output_dsec/ \
    --save_path output_dsec_rgb/ \
    --dataset dsec
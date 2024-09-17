# python scripts/main_ssa_engine.py --data_dir=data/<The name of your dataset> --out_dir=output --world_size=8 --save_img --sam --ckpt_path=ckp/sam_vit_h_4b8939.pth
# python scripts/evaluation.py --gt_path data/cityscapes/gtFine/val/ --result_path output_cityscapes/ --dataset cityscapes
# 设置可见的GPU卡为2和3
export CUDA_VISIBLE_DEVICES=2,3

# 运行Python脚本
python scripts/main_ssa_engine.py \
    --data_dir=data/dsec/leftImg8bit/val//zurcih_city_12_a/ \
    --out_dir=output_dsec_engine \
    --world_size=2 \
    --save_img \
    --sam \
    --ckpt_path=ckp/sam_vit_h_4b8939.pth
# python scripts/main_ssa.py --ckpt_path ./ckp/sam_vit_h_4b8939.pth --save_img --world_size 8 --dataset cityscapes --data_dir data/cityscapes/leftImg8bit/val/ --gt_path data/cityscapes/gtFine/val/ --out_dir output_cityscapes# 设置可见的GPU卡为2和3
export CUDA_VISIBLE_DEVICES=0,1

# 运行Python脚本
python scripts/main_ssa.py \
    --ckpt_path ./ckp/sam_vit_h_4b8939.pth \
    --save_img \
    --save_sem_map \
    --world_size 2 \
    --dataset dsec \
    --data_dir /home/xiaoshan/work/adap_v/EvLight/log/release/night_val_sde_dsec_out/epoch-best/ \
    --gt_path /home/xiaoshan/work/adap_v/my_proj/data/DSEC/night/gtFine/val/ \
    --out_dir dsec_evlight_night_val_19 \
    --points_per_side 128 \
    --crop_n_layers 1
    # --data_dir /home/xiaoshan/work/adap_v/my_proj/data/DSEC/night/leftImg8bit/val/ \

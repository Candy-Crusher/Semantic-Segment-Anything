# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from mmseg.core import add_prefix
from mmseg.ops import resize
from mmcv.utils import print_log

import os
import mmcv
import argparse
import numpy as np
from collections import OrderedDict
import pycocotools.mask as maskUtils
from prettytable import PrettyTable
from torchvision.utils import save_image, make_grid
from mmseg.core import eval_metrics, intersect_and_union, pre_eval_to_metrics

from configs.ade20k_id2label import CONFIG as CONFIG_ADE20K_ID2LABEL
from configs.cityscapes_id2label import CONFIG as CONFIG_CITYSCAPES_ID2LABEL

from PIL import Image

def parse_args():
    parser = argparse.ArgumentParser(description='Semantically segment anything.')
    parser.add_argument('--gt_path', help='the directory of gt annotations')
    parser.add_argument('--result_path', help='the directory of semantic predictions')
    parser.add_argument('--save_path', help='the directory to save rgb semantic predictions')
    parser.add_argument('--dataset', type=str, default='cityscapes', choices=['ade20k', 'cityscapes', 'foggy_driving', 'dsec'], help='specify the dataset')
    args = parser.parse_args()
    return args

args = parse_args()
logger = None
if args.dataset == 'cityscapes' or args.dataset == 'foggy_driving' or args.dataset == 'dsec':
    class_names = ('road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle')
elif args.dataset == 'ade20k':
    class_names = ('wall', 'building', 'sky', 'floor', 'tree', 'ceiling', 'road', 'bed ', 'windowpane', 'grass', 'cabinet', 'sidewalk', 'person', 'earth', 'door', 'table', 'mountain', 'plant', 'curtain', 'chair', 'car', 'water', 'painting', 'sofa', 'shelf', 'house', 'sea', 'mirror', 'rug', 'field', 'armchair', 'seat', 'fence', 'desk', 'rock', 'wardrobe', 'lamp', 'bathtub', 'railing', 'cushion', 'base', 'box', 'column', 'signboard', 'chest of drawers', 'counter', 'sand', 'sink', 'skyscraper', 'fireplace', 'refrigerator', 'grandstand', 'path', 'stairs', 'runway', 'case', 'pool table', 'pillow', 'screen door', 'stairway', 'river', 'bridge', 'bookcase', 'blind', 'coffee table', 'toilet', 'flower', 'book', 'hill', 'bench', 'countertop', 'stove', 'palm', 'kitchen island', 'computer', 'swivel chair', 'boat', 'bar', 'arcade machine', 'hovel', 'bus', 'towel', 'light', 'truck', 'tower', 'chandelier', 'awning', 'streetlight', 'booth', 'television receiver', 'airplane', 'dirt track', 'apparel', 'pole', 'land', 'bannister', 'escalator', 'ottoman', 'bottle', 'buffet', 'poster', 'stage', 'van', 'ship', 'fountain', 'conveyer belt', 'canopy', 'washer', 'plaything', 'swimming pool', 'stool', 'barrel', 'basket', 'waterfall', 'tent', 'bag', 'minibike', 'cradle', 'oven', 'ball', 'food', 'step', 'tank', 'trade name', 'microwave', 'pot', 'animal', 'bicycle', 'lake', 'dishwasher', 'screen', 'blanket', 'sculpture', 'hood', 'sconce', 'vase', 'traffic light', 'tray', 'ashcan', 'fan', 'pier', 'crt screen', 'plate', 'monitor', 'bulletin board', 'shower', 'radiator', 'glass', 'clock', 'flag')
file_client = mmcv.FileClient(**{'backend': 'disk'})
pre_eval_results = []
gt_path = args.gt_path
res_path = args.result_path
save_path = args.save_path
if not os.path.exists(save_path):
    os.mkdir(save_path)
if args.dataset == 'cityscapes':
    prefixs = ['frankfurt','lindau','munster']
if args.dataset == 'dsec':
    prefixs = ['zurich_city_12_a']
elif args.dataset == 'foggy_driving':
    prefixs = ['public', 'pedestrian']
elif args.dataset == 'ade20k':
    prefixs = ['']
else:
    raise NotImplementedError

if args.dataset == 'ade20k':
    id2label = CONFIG_ADE20K_ID2LABEL
elif args.dataset == 'cityscapes' or args.dataset == 'foggy_driving' or args.dataset == 'dsec':
    id2label = CONFIG_CITYSCAPES_ID2LABEL
    semseg_dict = {
        "num_classes": 19,
        "ignore_label": 255,
        "class_names": [
            "road", "sidewalk", "building", "wall", "fence",
            "pole", "traffic light", "traffic sign", "vegetation", "terrain",
            "sky", "person", "rider", "car", "truck", "bus",
            "train", "motorcycle", "bicycle"
        ],
        "color_map": np.array([
            [128, 64, 128],      # road
            [244, 35, 232],      # sidewalk
            [70, 70, 70],        # building
            [102, 102, 156],     # wall
            [190, 153, 153],     # fence
            [153, 153, 153],     # pole
            [250, 170, 30],      # traffic light
            [220, 220, 0],       # traffic sign
            [107, 142, 35],      # vegetation
            [152, 251, 152],     # terrain
            [70, 130, 180],      # sky
            [220, 20, 60],       # person
            [255, 0, 0],         # rider
            [0, 0, 142],         # car
            [0, 0, 70],          # truck
            [0, 60, 100],        # bus
            [0, 80, 100],        # train
            [0, 0, 230],         # motorcycle
            [119, 11, 32]        # bicycle
        ], dtype=np.uint8)
    }
else:
    raise NotImplementedError()
    
for split in tqdm(prefixs, desc="Split loop"):
    gt_path_split = os.path.join(gt_path, split)
    res_path_split = os.path.join(res_path, split)
    save_path_split = os.path.join(save_path, split)
    filenames = [fn_ for fn_ in os.listdir(res_path_split) if '.json' in fn_]
    for i, fn_ in enumerate(tqdm(filenames, desc="File loop")):
        pred_fn = os.path.join(res_path_split, fn_)
        result = mmcv.load(pred_fn)
        num_classes = len(class_names)
        init_flag = True
        for id_str, mask in result['semantic_mask'].items():
            mask_ = maskUtils.decode(mask)
            h, w = mask_.shape
            if init_flag:
                seg_mask = torch.zeros((1, 1, h, w))
                init_flag = False
            mask_ = torch.from_numpy(mask_).unsqueeze(0).unsqueeze(0)
            seg_mask[mask_] = int(id_str)
        seg_logit = torch.zeros((1, num_classes, h, w))
        seg_logit.scatter_(1, seg_mask.long(), 1)
        seg_logit = seg_logit.float()
        seg_pred = F.softmax(seg_logit, dim=1).argmax(dim=1).squeeze(0).numpy()

        rgb_image = semseg_dict['color_map'][seg_pred]

        # 将numpy数组转换为PIL图像
        rgb_image = Image.fromarray(rgb_image)

        # 保存图像
        if not os.path.exists(save_path_split):
            os.makedirs(save_path_split)
        rgb_image.save(os.path.join(save_path_split, fn_.replace('_semantic.json','_color.png')))

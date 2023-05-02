import torch
import numpy as np
import os
import time
import torch.nn as nn
from tqdm import tqdm
from utils.util import AverageMeter, ensure_dir, object_based_infer
import shutil
from utils.metrics import Evaluator_tensor
import random
import datetime
import argparse
import torch
import random
import numpy as np
from configs.config import MyConfiguration
from Tester import Tester
from data.dataset_list import MyDataset
from torch.utils.data import DataLoader
from models import FCN_backbone
import cv2
from pathlib import Path
from skimage import measure



def object_based_infer(pre_logit, post_logit):
    loc = (pre_logit > 0.).cpu().squeeze(1).numpy()
    dam = post_logit.argmax(dim=1).cpu().squeeze(1).numpy()

    refined_dam = np.zeros_like(dam)
    for i, (single_loc, single_dam) in enumerate(zip(loc, dam)):
        refined_dam[i, :, :] = _object_vote(single_loc, single_dam)

    return loc, refined_dam


def _object_vote(loc, dam):
    damage_cls_list = [1, 2, 3, 4]
    local_mask = loc
    labeled_local, nums = measure.label(local_mask, connectivity=2, background=0, return_num=True)
    region_idlist = np.unique(labeled_local)
    if len(region_idlist) > 1:
        dam_mask = dam
        new_dam = local_mask.copy()
        for region_id in region_idlist:
            if all(local_mask[local_mask == region_id]) == 0:
                continue
            region_dam_count = [int(np.sum(dam_mask[labeled_local == region_id] == dam_cls_i)) * cls_weight \
                                for dam_cls_i, cls_weight in zip(damage_cls_list, [4.6, 136.6, 353.9, 83.2])]
            dam_index = np.argmax(region_dam_count) + 1
            new_dam = np.where(labeled_local == region_id, dam_index, new_dam)
    else:
        new_dam = local_mask.copy()
    return new_dam


def display_segmentation_mask(mask):
    # Define the color codes for each class
    color_codes = {
        0: [0, 0, 0],     # background (black)
        1: [0, 255, 0],   # no visible damage (green)
        2: [255, 255, 0], # possibly damaged (yellow)
        3: [255, 165, 0], # damaged (orange)
        4: [255, 0, 0]    # destroyed (red)
    }
    
    # Create an empty array for the colored mask
    height, width = mask.shape
    colored_mask = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Loop over each pixel in the mask and set the corresponding color
    for i in range(height):
        for j in range(width):
            pixel_class = mask[i, j]
            colored_mask[i, j] = color_codes[pixel_class]
    
    # Display the colored mask using OpenCV
    return colored_mask



config = MyConfiguration('configs/config.cfg')

pathCkpt = '/home/julian/Documents/MS4D-Net-Building-Damage-Assessment/save_d100/Si_FCN_Dam_vgg19_bn_stack_PDMT/20230430_165559/checkpoint-best.pth'

parser = argparse.ArgumentParser(description="Model Evaluation")

parser.add_argument('-input', metavar='input', type=str, default=config.root_dir,
                    help='root path to directory containing input images, including train & valid & test')
parser.add_argument('-output', metavar='output', type=str, default=config.save_dir,
                    help='root path to directory containing all the output, including predictions, logs and ckpt')
parser.add_argument('-weight', metavar='weight', type=str, default=pathCkpt,
                    help='path to ckpt which will be loaded')
parser.add_argument('-threads', metavar='threads', type=int, default=2,
                    help='number of thread used for DataLoader')
parser.add_argument('-only_prediction', action='store_true', default=True,
                    help='in test mode, only prediciton, no evaluation')
parser.add_argument('-is_test', action='store_true', default=False,
                    help='in test mode, is_test=True')
if config.use_gpu:
    parser.add_argument('-gpu', metavar='gpu', type=int, default=0,
                        help='gpu id to be used for prediction')
else:
    parser.add_argument('-gpu', metavar='gpu', type=int, default=-1,
                        help='gpu id to be used for prediction')

args = parser.parse_args()





if config.use_seed:
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed(config.random_seed)
    torch.manual_seed(config.random_seed)
    random.seed(config.random_seed)
    np.random.seed(config.random_seed)
else:
    torch.backends.cudnn.benchmark = True


out_channel = config.nb_classes
model1 = FCN_backbone.SiameseFCN_damage(config.input_channel, out_channel, backbone='vgg19_bn', pretrained=True,
                                        shared=False, fused_method='stack')
model2 = FCN_backbone.SiameseFCN_damage(config.input_channel, out_channel, backbone='vgg19_bn', pretrained=True,
                                        shared=False, fused_method='stack')

if hasattr(model1, 'name'):
    config.config.set("Directory", "model_name", model1.name+'_PDMT')

test_dataset = MyDataset(config=config, args=args, subset='test')

test_data_loader = DataLoader(dataset=test_dataset,
                                batch_size=1,
                                shuffle=False,
                                pin_memory=True,
                                num_workers=args.threads,
                                drop_last=False)

begin_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
begin_time = 'test-' + begin_time

if config.use_gpu:
    model1 = model1.cuda(device=args.gpu)
    model2 = model2.cuda(device=args.gpu)

model = []
model.append(model1)
model.append(model2)

checkpoint = torch.load(pathCkpt)
model[0].load_state_dict(checkpoint['state_dict1'], strict=True)
model[1].load_state_dict(checkpoint['state_dict2'], strict=True)


pred_dir = os.path.join(args.output, 'predictions_vgg_19_stack')
os.makedirs(pred_dir, exist_ok=True)
device = torch.device('cuda:{}'.format(args.gpu))
with torch.no_grad():
    tic = time.time()
    for i, data in tqdm(enumerate(test_data_loader, start=1)):
        
        filename = Path(data[1][0]).stem
        data_pred = data[0].to(device, non_blocking=True)
        # target = target.to(device, non_blocking=True)
        print(filename)
        logits0= model[0](data_pred)
        logits1= model[1](data_pred)
        logits = {}
        logits['damage'] = (logits0['damage'] + logits1['damage']) / 2
        logits['building'] = (logits0['building'] + logits1['building']) / 2

        # vote by object
        mask_BD, mask_D = object_based_infer(logits['building'], logits['damage'])
        cv2.imwrite(pred_dir + '/' + filename + '.png', mask_D.transpose(1, 2, 0))

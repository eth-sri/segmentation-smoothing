# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

import argparse
import os
import pprint
import shutil
import sys

import logging
import time
import timeit
from pathlib import Path

import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.nn import functional as F

import _init_paths
import models
import datasets
from config import config
from config import update_config
#from core.function import testval, test 
from utils.modelsummary import get_model_summary
from utils.utils import create_logger, FullModel

from tqdm import tqdm
import scipy.stats as sps

from PIL import Image
import itertools
import scipy.stats as sps

sys.path.append('..')
from multiclassify_utils import certify, Log, setup_segmentation_args, str2bool
from tqdm import trange

def attack(args, config, test_dataset, testloader, model):
    model.eval()
    for i, inp in enumerate(testloader):
        print(i)
        image, label, _, name = inp
        _, _, h, w = image.size()
        label = label.view((-1)).long().cuda()
        loss = nn.CrossEntropyLoss(ignore_index=config.TRAIN.IGNORE_LABEL)
        image /= 255.0


        mean = torch.tensor(test_dataset.mean).reshape((1, 3, 1, 1))
        std = torch.tensor(test_dataset.std).reshape((1, 3, 1, 1))


        out = model((image - mean) / std)
        out = F.upsample(out, (h, w), 
                         mode='bilinear')
        out = torch.transpose(out.view((config.DATASET.NUM_CLASSES, -1)), 0, 1)
        incorr_base = (label != out.argmax(dim=1)).sum().item() / np.prod(label.size())

        
        eps = 0.25 * sps.norm.ppf(0.75)
        steps = 100
        step_size = 10.0 * eps / steps

        rp = torch.randn_like(image)
        norm = rp.norm()
        img = torch.clamp(image + eps * rp / (norm + 1e-10), 0, 1)


        path = os.path.join('.', 'attack', args.name)
        os.makedirs(path, exist_ok=True)
        pbar = trange(steps)
        for k in pbar: 
            img = img.clone().detach().requires_grad_(True)
            out = model((img - mean) / std)
            out = F.upsample(out, (h, w), 
                            mode='bilinear')
            out = torch.transpose(out.view((config.DATASET.NUM_CLASSES, -1)), 0, 1)
            with torch.no_grad():
                incorr = (label != out.argmax(dim=1)).sum().item() / np.prod(label.size())


            #untargeted attack
            l = loss(out, label)

            #targeted attack
            #l = -loss(out, torch.zeros_like(label))

            pbar.set_description(f"Loss: {l.item()} Cnt: {incorr/incorr_base}")
            grad, = torch.autograd.grad(l, [img]) 

            with torch.no_grad():
                grad_norm = grad.norm()
                grad_scaled = grad / (grad_norm + 1e-10)
                img = img + grad_scaled * step_size
                diff = img - image
                diff.renorm(p=2, dim=0, maxnorm=eps)
                img = torch.clamp(image + diff, 0, 1)
        with torch.no_grad():
            pred = model((img - mean) / std)
        test_dataset.save_pred(pred, path, [f'attacked{i}'])

        img = (255*img).cpu().numpy()[0].transpose((1, 2, 0)).copy().astype(np.uint8)
        img = Image.fromarray(img)
        img.save(os.path.join(path, f'img{i}.png'))
        label = test_dataset.convert_label(label.view((h,w)).cpu().numpy().astype(np.uint8), inverse=True)
        label = Image.fromarray(label)
        label.putpalette(test_dataset.get_palette(256))
        label.save(os.path.join(path, f'label{i}.png'))

def parse_args():
    parser = argparse.ArgumentParser(description='Train segmentation network')
    
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)

    setup_segmentation_args(parser)
    parser.add_argument('--crop', type=str2bool, default=False)
    parser.add_argument('--name', type=str, default='atk')
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()
    update_config(config, args)

    return args

def main():
    args = parse_args()

    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED

    # build model
    model = eval('models.'+config.MODEL.NAME +
                 '.get_seg_model')(config)

    dump_input = torch.rand(
        (1, 3, config.TRAIN.IMAGE_SIZE[1], config.TRAIN.IMAGE_SIZE[0])
    )

    if config.TEST.MODEL_FILE:
        model_state_file = config.TEST.MODEL_FILE
    else:
        model_state_file = os.path.join(final_output_dir,
                                        'final_state.pth')

        
    pretrained_dict = torch.load(model_state_file)
    model_dict = model.state_dict()
    pretrained_dict = {k[6:]: v for k, v in pretrained_dict.items()
                        if k[6:] in model_dict.keys()}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    gpus = list(config.GPUS)
    model = nn.DataParallel(model, device_ids=gpus).cuda()

    # prepare data
    test_size = (config.TEST.IMAGE_SIZE[1], config.TEST.IMAGE_SIZE[0])
    test_dataset = eval('datasets.'+config.DATASET.DATASET)(
                        root=config.DATASET.ROOT,
                        list_path=config.DATASET.TEST_SET,
                        num_samples=None,
                        num_classes=config.DATASET.NUM_CLASSES,
                        multi_scale=False,
                        flip=False,
                        ignore_label=config.TRAIN.IGNORE_LABEL,
                        base_size=config.TEST.BASE_SIZE,
                        center_crop_test=args.crop,
                        crop_size=test_size,
                        downsample_rate=1,
                        normalize=False)

    testloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=True)
    


    
    attack(args, config, test_dataset, testloader, model)

if __name__ == '__main__':
    main()

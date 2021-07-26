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
from math import floor
from PIL import Image
import itertools
import cv2

sys.path.append('..')
from multiclassify_utils import certify, Log, setup_segmentation_args, str2bool

def get_confusion_matrix(label, seg_pred, size, num_class, ignore=-1):
    """
    Cal cute the confusion matrix by given label and pred
    """
    seg_gt = np.asarray(
    label.cpu().numpy()[:, :size[-2], :size[-1]], dtype=np.int)

    ignore_index = seg_gt != ignore
    seg_gt = seg_gt[ignore_index]
    seg_pred = seg_pred[ignore_index]

    index = (seg_gt * num_class + seg_pred).astype('int32')
    label_count = np.bincount(index)
    confusion_matrix = np.zeros((num_class, num_class))

    for i_label in range(num_class):
        for i_pred in range(num_class):
            cur_index = i_label * num_class + i_pred
            if cur_index < len(label_count):
                confusion_matrix[i_label,
                                 i_pred] = label_count[cur_index]
    return confusion_matrix

def sample(image_np01, size, n, sigma, model, config, test_dataset, do_tqdm=True, unscaled=False):
    BS = config.TEST.BATCH_SIZE_PER_GPU
    out = []
    remaining = n
    if do_tqdm: pbar = tqdm(total=n)
    with torch.no_grad():
        while (remaining) > 0:
            cnt = min(remaining, BS)
            pred = test_dataset.multi_scale_inference_noisybatch(
                        model, 
                        cnt, sigma,
                        image_np01,
                        normalize=True,
                        scales=config.TEST.SCALE_LIST, 
                        flip=config.TEST.FLIP_TEST,
                        unscaled=unscaled)
            if not unscaled and (pred.size()[-2] != size[-2] or pred.size()[-1] != size[-1]):
                pred = F.upsample(pred, (size[-2], size[-1]), 
                                   mode='bilinear')
            pred = pred.argmax(dim=1).cpu().numpy()
            out.append(pred)
            remaining -= cnt
            if do_tqdm: pbar.update(cnt)
    if do_tqdm: pbar.close()
    return np.concatenate(out)

def save_segmentation(seg, test_dataset, palette, path, abstain, abstain_mapping):
    img = np.asarray(seg[0, ...], dtype=np.uint8)
    I = (img == abstain)
    img = test_dataset.convert_label(img, inverse=True)
    img[I] = abstain_mapping
    img = Image.fromarray(img)
    img.putpalette(palette)
    img.save(path)


def testval(args, config, test_dataset, testloader, model):
    model.eval()

    log = Log(config.DATASET.DATASET + args.name)
    log.write(str(args) + "\n")
    with open(log.folder() / "config.cfg", 'w') as f:
        f.write(str(config))
    
    num_classes = config.DATASET.NUM_CLASSES+1 # threat abstain as an additional class
    abstain = num_classes-1
    abstain_mapping = 254 
    
    corr = ['bonferroni', 'holm']
    out_confusion_matrix = {}
    out_confusion_matrix['baseline'] = np.zeros((num_classes, num_classes))
    outname_base = "certify" if not args.unscaled else "certify_unscale"
    for k in range(len(args.n)):
        tau = args.tau[k]
        n = args.n[k]
        for c in corr:
            out_confusion_matrix[f"{outname_base}_{c}_{n}_{tau}"] = np.zeros((num_classes, num_classes))

    out_confusion_matrix['baseline'] = np.zeros((num_classes, num_classes))
    palette = test_dataset.get_palette(256)
    palette[abstain_mapping * 3 + 0] = 255
    palette[abstain_mapping * 3 + 1] = 255
    palette[abstain_mapping * 3 + 2] = 255

    m = floor(len(testloader) / args.N)
    itt = itertools.islice(testloader, 0, None, m)
    itt = enumerate(itertools.islice(itt, args.N))

    with torch.no_grad():
        for index, batch in itt: 
            if index < args.N0: continue
            if len(args.NN) > 0 and index not in args.NN: continue
            print("Image", index, file=log)
            image, label, _, name = batch
            image_np = image.numpy()[0].transpose((1, 2, 0)).astype(np.float32).copy()
            image_np01 = image_np/255.0
            size = label.size()
            
            # save image and label
            img = Image.fromarray(image_np.astype(np.uint8))
            img.save(log.folder() / f"{index}_image.png")
            save_segmentation(label, test_dataset, palette, log.folder() / f"{index}_gt.png", abstain, abstain_mapping )
            del img

            t0 = time.time()
            classes_baseline = sample(image_np01, size, 1, 0, model, config, test_dataset)
            t1 = time.time()
            save_segmentation(classes_baseline, test_dataset, palette, log.folder() / f"{index}_baseline.png", abstain, abstain_mapping)

            confusion_matrix = get_confusion_matrix(label, classes_baseline, size, num_classes, config.TRAIN.IGNORE_LABEL)
            pos = confusion_matrix.sum(1)
            res = confusion_matrix.sum(0)
            tp = np.diag(confusion_matrix)
            pixel_acc = tp.sum()/pos.sum()
            mean_acc = (tp/np.maximum(1.0, pos)).mean()
            IoU_array = (tp / np.maximum(1.0, pos + res - tp))
            mean_IoU = IoU_array.mean()
            print('baseline', file=log)
            print('accuracy', pixel_acc, file=log)
            print('IoU', IoU_array, file=log)
            print('time', t1 - t0, file=log)
            print(file=log)
            out_confusion_matrix["baseline"] += confusion_matrix

            if args.baseline: continue
            
            # # run certification
            for k in range(len(args.n)):
                tau = args.tau[k]
                n = args.n[k]
                t0 = time.time()
                s = sample(image_np01, size, args.n0 + n, args.sigma, model, config, test_dataset, unscaled=args.unscaled)
                s_shape = s.shape
                s = np.reshape(s, (s_shape[0], -1))
                t1 = time.time()
                time_sample = t1 - t0

                for c in corr:
                    classes_certify, radius, timings = certify(num_classes-1, s, args.n0, n, args.sigma, tau, args.alpha, abstain=abstain, parallel=True, correction=c)
                    classes_certify = np.reshape(classes_certify, (1, *s_shape[1:]) )
                    if args.unscaled:
                        classes_certify = cv2.resize(np.transpose(classes_certify, (1, 2, 0)).astype(np.float32), dsize=(size[-2], size[-1]), interpolation=cv2.INTER_NEAREST)
                        classes_certify = np.reshape(classes_certify, (1, size[-2], size[-1])).astype(np.int64)
                    save_segmentation(classes_certify, test_dataset, palette, log.folder() / f"{index}_certify_{c}_{n}_{tau}.png", abstain, abstain_mapping)
                    time_pvals, time_correction = timings

                    confusion_matrix = get_confusion_matrix(label, classes_certify, size, num_classes, config.TRAIN.IGNORE_LABEL)
                    I = (classes_certify != abstain)
                    cnt_nonabstain = np.sum(I)
                    out_confusion_matrix[f"{outname_base}_{c}_{n}_{tau}"] += confusion_matrix

                    print(f"{outname_base}_{c}_{n}_{tau}", file=log)
                    pos = confusion_matrix.sum(1)
                    res = confusion_matrix.sum(0)
                    tp = np.diag(confusion_matrix)
                    pixel_acc = tp.sum()/pos.sum()
                    mean_acc = (tp/np.maximum(1.0, pos)).mean()
                    IoU_array = (tp / np.maximum(1.0, pos + res - tp))
                    mean_IoU = IoU_array.mean()
                    print('accuracy', pixel_acc, file=log)
                    print('non-abstain', cnt_nonabstain/classes_certify.size, file=log)
                    print('IoU', IoU_array, file=log)
                    print('time', time_sample, time_pvals, time_correction, file=log)
                    print(file=log)
            print(file=log)

        print('done', file=log)
        print(file=log)
            
        for key in out_confusion_matrix.keys():
            print(key, file=log)
            pos = out_confusion_matrix[key].sum(1)
            res = out_confusion_matrix[key].sum(0)
            tp = np.diag(out_confusion_matrix[key])
            pixel_acc = tp.sum()/pos.sum()
            mean_acc = (tp/np.maximum(1.0, pos)).mean()
            IoU_array = (tp / np.maximum(1.0, pos + res - tp))
            mean_IoU = IoU_array.mean()
            print('accuracy', pixel_acc, file=log)
            print('mean-accuracy', mean_acc, file=log)
            print('IoU', IoU_array, file=log)
            print('Mean IoU', mean_IoU, file=log)
            print(file=log)


def parse_args():
    parser = argparse.ArgumentParser(description='Train segmentation network')
    
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)

    setup_segmentation_args(parser)
    parser.add_argument('--crop', type=str2bool, default=False)
    parser.add_argument('--unscaled', type=str2bool, default=False)
    parser.add_argument('--name', type=str, default="")
    parser.add_argument('-N0', type=int, default=0)
    parser.add_argument('-NN', type=int, nargs='+', default=[])
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()
    update_config(config, args)

    return args

def main():
    args = parse_args()
    if args.unscaled: assert len(config.TEST.SCALE_LIST) == 1
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
                        crop_size=test_size,
                        center_crop_test=args.crop,
                        downsample_rate=1,
                        normalize=False)

    testloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=True)
    
    testval(args, config, test_dataset, testloader, model)

if __name__ == '__main__':
    main()

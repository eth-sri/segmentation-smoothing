"""
Author: Benny
Date: Nov 2019
"""
import argparse
import os
from data_utils.ShapeNetDataLoader import PartNormalDataset
import torch
import logging
import sys
import importlib
from tqdm  import tqdm
import numpy as np
import scipy.stats as sps
import time
sys.path.append('..')
from multiclassify_utils import certify, Log, setup_segmentation_args
import itertools
import copy
import random

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

seg_classes = {'Earphone': [16, 17, 18], 'Motorbike': [30, 31, 32, 33, 34, 35], 'Rocket': [41, 42, 43], 'Car': [8, 9, 10, 11], 'Laptop': [28, 29], 'Cap': [6, 7], 'Skateboard': [44, 45, 46], 'Mug': [36, 37], 'Guitar': [19, 20, 21], 'Bag': [4, 5], 'Lamp': [24, 25, 26, 27], 'Table': [47, 48, 49], 'Airplane': [0, 1, 2, 3], 'Pistol': [38, 39, 40], 'Chair': [12, 13, 14, 15], 'Knife': [22, 23]}
seg_label_to_cat = {} # {0:Airplane, 1:Airplane, ...49:Table}
for cat in seg_classes.keys():
    for label in seg_classes[cat]:
        seg_label_to_cat[label] = cat

def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    new_y = torch.eye(num_classes)[y.cpu().data.numpy(),]
    if (y.is_cuda):
        return new_y.cuda()
    return new_y


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('PointNet')
    parser.add_argument('--batch_size', type=int, default=24, help='batch size in testing [default: 24]')
    parser.add_argument('--num_point', type=int, default=2048, help='Point Number [default: 2048]')
    parser.add_argument('--log_dir', type=str, default='pointnet2_part_seg_ssg', help='Experiment root')
    parser.add_argument('--normal', action='store_true', default=False, help='Whether to use normal information [default: False]')
    parser.add_argument('--num_votes', type=int, default=3, help='Aggregate segmentation scores with voting [default: 3]')
    setup_segmentation_args(parser)
    return parser.parse_args()


def sample(args, num_part, num_classes, sigma, n, input, label, model, do_tqdm=True):
    remaining = n
    out = []
    if do_tqdm: pbar = tqdm(total=n)
    while remaining > 0:
        size = min(remaining, args.batch_size)
        batch = input.repeat(size, 1, 1)
        batch[:, 0:3, :] += sigma * torch.randn_like(batch[:, 0:3, :])
        vote_pool = torch.zeros(size, batch.size(2), num_part).cuda()
        for _ in range(args.num_votes):
            seg_pred, _ = model(batch, to_categorical(label.repeat(size, 1), num_classes))
            vote_pool += seg_pred
        seg_pred = vote_pool / args.num_votes
        #cur_pred_val = seg_pred.cpu().data.numpy()
        seg_pred = seg_pred.argmax(dim=-1)
        out.append(seg_pred.cpu())
        remaining -= size
        if do_tqdm: pbar.update(size)
    if do_tqdm: pbar.close()
    return torch.cat(out)


class Stats:

    def __init__(self, num_parts, seg_classes):
        self.stats = {}
        self.num_parts = num_parts
        self.seg_label_to_cat = {}  # {0:Airplane, 1:Airplane, ...49:Table}
        for cat in seg_classes.keys():
            for label in seg_classes[cat]:
                self.seg_label_to_cat[label] = cat
        self.seg_classes = seg_classes


    def _init_name(self, name):
        self.stats[name] = {}
        self.stats[name]['cnt'] = 0 
        self.stats[name]['correct'] = 0
        self.stats[name]['seen'] = 0
        self.stats[name]['parts'] = {}
        for p in range(self.num_parts):
            self.stats[name]['parts'][p] = {}
            self.stats[name]['parts'][p]['correct'] = 0
            self.stats[name]['parts'][p]['seen']  = 0
        self.stats[name]['shape_ious'] = {cat: [] for cat in self.seg_classes.keys()}


    def update(self, name, target, predict, abstain=None, times=None, file=None):
        if file is None: file = sys.stdout
        if name not in self.stats: self._init_name(name)
        self.stats[name]['correct'] += np.sum(target == predict)
        self.stats[name]['seen'] += target.size
        for l in range(self.num_parts):
            self.stats[name]['parts'][l]['seen'] += np.sum(target == l)
            self.stats[name]['parts'][l]['correct'] += (np.sum((predict == l) & (target == l)))
        cat = self.seg_label_to_cat[target[0, 0]]
        part_ious = [0.0 for _ in range(len(self.seg_classes[cat]))]
        for l in self.seg_classes[cat]:
            if (np.sum(target == l) == 0) and (
                    np.sum(predict == l) == 0):  # part is not present, no prediction as well
                part_ious[l - self.seg_classes[cat][0]] = 1.0
            else:
                part_ious[l - self.seg_classes[cat][0]] = np.sum((target == l) & (predict == l)) / float(
                    np.sum((target == l) | (predict == l)) + 1e-5)
        self.stats[name]['shape_ious'][cat].append(np.mean(part_ious))

        acc = np.sum(target == predict) / target.size
        out = [name, self.stats[name]['cnt'], acc]
        if abstain is not None: out.append(1 - (np.sum(predict == abstain) / predict.size))
        if times is not None: out.extend([sum(times), *times])

        print(', '.join(map(str, out)), file=file)

        self.stats[name]['cnt'] += 1


        
    def report(self, name, file=None):
        assert name in self.stats
        if file is None: file = sys.stdout
        all_shape_ious = []
        shape_ious = copy.deepcopy(self.stats[name]['shape_ious'])
        for cat in shape_ious.keys():
            for iou in shape_ious[cat]:
                all_shape_ious.append(iou)
            shape_ious[cat] = sum(shape_ious[cat]) / (len(shape_ious[cat]) + 1e-5)
        mean_shape_ious = np.mean(list(shape_ious.values()))
        accuracy = self.stats[name]['correct'] / self.stats[name]['seen']


        class_accuracy = [self.stats[name]['parts'][p]['correct']/(self.stats[name]['parts'][p]['seen'] + 1e-5) for p in range(self.num_parts)]
        class_avg_accuracy = np.mean(class_accuracy)
        

        print(name, file=file)
        print('accuracy', accuracy, file=file)
        print('class_avg_iou', mean_shape_ious, file=file)
        print('class_avg_accuracy', class_avg_accuracy, file=file)
        print('instance_avg_iou', np.mean(all_shape_ious), file=file)
        print(file=file)

        

def main(args):
    args = parse_args()

    assert len(args.n) == len(args.tau)

    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    
    log = Log('partseg_l2')
    log.write(str(args) + "\n")

    root = 'data/shapenetcore_partanno_segmentation_benchmark_v0_normal/'

    TEST_DATASET = PartNormalDataset(root = root, npoints=args.num_point, split='test', normal_channel=args.normal)
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=1, shuffle=False, num_workers=4)
    num_classes = 16
    num_part = 50

    '''MODEL LOADING'''
    experiment_dir = 'log/part_seg/' + args.log_dir
    model_name = os.listdir(experiment_dir+'/logs')[0].split('.')[0]
    MODEL = importlib.import_module(model_name)
    classifier = MODEL.get_model(num_part, normal_channel=args.normal).cuda()
    checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
    classifier.load_state_dict(checkpoint['model_state_dict'])
    classifier = torch.nn.DataParallel(classifier)


    with torch.no_grad():
        test_metrics = {}

        stats = Stats(num_part, seg_classes)
        classifier = classifier.eval()
        
        for batch_id, (points, label, target) in enumerate(itertools.islice(testDataLoader, args.N)):
            batchsize, num_point, _ = points.size()
            assert batchsize == 1

            points, label = points.float().cuda(), label.long().cuda()
            points = points.transpose(2, 1)
            target = target.cpu().data.numpy()
            cat = seg_label_to_cat[target[0, 0]]


            # run baseline
            t0 = time.time()
            classes_baseline = sample(args, num_part, num_classes, 0, 1, points, label, classifier, do_tqdm=False).numpy()
            t1 = time.time()
            classes_baseline = classes_baseline + seg_classes[cat][0]
            stats.update('baseline', target, classes_baseline, file=log)
            print(t1 - t0)
            
            # run certification
            for k in range(len(args.n)):
                tau = args.tau[k]
                n = args.n[k]
                t0 = time.time()
                s = sample(args, num_part, num_classes, args.sigma, args.n0 + n, points, label, classifier)
                t1 = time.time()
                time_sample = t1 - t0
                for c in ['holm', 'bonferroni']:
                    classes_certify, radius, timings = certify(num_classes, s.numpy(), args.n0, n, args.sigma, tau, args.alpha, correction=c)
                    time_pvals, time_correction = timings
                    classes_certify = np.expand_dims(classes_certify, axis=0)

                    I = (classes_certify != -1)
                    cnt_nonabstain = np.sum(I)
                    classes_certify[I] = classes_certify[I] + seg_classes[cat][0]
                    stats.update(f'certify_{c}_{n}_{tau}', target, classes_certify, abstain=-1, times=[time_sample, time_pvals, time_correction], file=log)

        stats.report('baseline', log)

        for k in range(len(args.n)):
            tau = args.tau[k]
            n = args.n[k]
            for c in ['holm', 'bonferroni']:
                stats.report(f'certify_{c}_{n}_{tau}', log)

if __name__ == '__main__':
    args = parse_args()
    main(args)


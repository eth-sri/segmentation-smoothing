#!/usr/bin/env python3

import pathlib
import os
from glob2 import glob
from argparse import Namespace
import pandas as pd
import re
import numpy as np
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', choices=['cityscapes', 'pascal_ctx'], default='cityscapes')
args_ = parser.parse_args()


log_dir = pathlib.Path('../logs/')
dirs = os.listdir(log_dir)
dirs = filter(lambda x: os.path.isdir(log_dir / x), dirs)
dirs = filter(lambda x: x.startswith(f"{args_.dataset}_2021_06"), dirs)

STATS_COLS = ['scale', 'flip', 'batch_size', 'model',
              'n', 'n0', 'tau', 'N', 'alpha', 'sigma', 'baseline',
                'accuracy', 'mean accuracy', 'mean iou', 'correction',   'abstain', 'time',  'fn']
df = pd.DataFrame([], columns=STATS_COLS)

for dir in dirs:
    dir = log_dir / dir
    log = dir / 'log.log'
    img_cnt = len(glob(os.path.join(str(dir), '*_image.png')))


    has_result = False
    
    if log.exists():
        with open(log, 'r') as f:
            lines = f.readlines()
            args = eval(lines[0])
            if args.N == img_cnt:
                has_result = True
            if hasattr(args, 'unscaled') and args.unscaled:
                has_result = False

    if has_result:
        print(dir)
        print(args)



        read_opt = ['TEST.SCALE_LIST', 'TEST.FLIP_TEST', 'TEST.BATCH_SIZE_PER_GPU', 'TEST.MODEL_FILE']
        values_opt = [None for _ in read_opt] 
        opts = vars(args)['opts']
        for i in range(0, len(opts), 2):
            key = opts[i]
            value = opts[i+1]
            for k, r in enumerate(read_opt):
               if key == r:
                   values_opt[k] = value

        n0 = args.n0
        N = args.N
        alpha = args.alpha
        sigma = args.sigma
        baseline = ('baseline' in args) and args.baseline

        if baseline:
            sigma = 0

        for m in range(len(args.n)):
            n = args.n[m]
            tau = args.tau[m]
            if baseline:
                n = 0
                tau = 0


            for i, l in enumerate(lines):
                if l.strip() == 'done': break

            starts = []
            for j in range(i+1, len(lines)-1):
                if  lines[j] == '\n':
                    starts.append(j+1)

            read_stats = ['accuracy', 'mean-accuracy']
            values_stats = [None for _ in read_stats] + [None, None]
            for s,e in zip(starts, starts[1:]+[len(lines)]):
                correction = None
                if len(lines[s].split('_')) > 1:
                    correction = lines[s].split('_')[1]
                print(correction)
                for i in range(s+1, e):
                    for k, r in enumerate(read_stats):
                        if lines[i].startswith(r):
                            if r == 'time': print(lines[i])
                            values_stats[k] = float(lines[i].split(' ')[-1].strip())
                    if lines[i].startswith('IoU'):
                        ll = lines[i]
                        mm = 1
                        while ']' not in  ll:
                            ll += lines[i + mm]
                            mm += 1
                        iou = np.array(eval(re.sub('\s+', ' ', ll.replace('\n', '').replace('IoU ', '')).replace(' ', ',')))
                        values_stats[-2] = (np.mean(iou[:-1]))
                        values_stats[-1] = correction
                time = []
                abstain = []
                for i, l in enumerate(lines):
                    if l.strip() == 'done': break
                    if l.strip() == lines[s].strip():
                        k = i
                        while True:
                            if  lines[k] == '\n': break
                            if not baseline and lines[k].startswith('non-abstain'):
                                abstain.append(float(lines[k].split(' ')[-1].strip()))
                            if lines[k].startswith('time'):
                                tokens = lines[k].split(' ')
                                if len(tokens) == 4:
                                    time.append(sum(map(lambda x: float(x.strip()), tokens[1:])))
                            k+=1
                abstain = 1 - np.mean(abstain)
                if baseline: abstain = 0
                time = np.mean(time)
                values = values_opt + [n, n0, tau, N, alpha, sigma, baseline] + values_stats + [abstain, time, log]
                df = df.append(pd.DataFrame([values], columns=STATS_COLS))
df = df.sort_values(by=['model','correction','scale', 'tau', 'sigma'])
print(df)
print(df[['correction','scale', 'n', 'tau', 'sigma', 'accuracy', 'mean iou', 'abstain', 'time', 'model']].to_latex(index=False, float_format="%.2f")) 

#df = df[df['scale']=='0.25,']
df = df[df['model']=='cityscapes_025_adv_e005_1step.pth']
#df = df[df['correction']=='None']
#cityscapes\_025\_adv\_e005\_1step.pth
#df = df[df['sigma']==0.25]
df = df[df['n']==100]
#print(df.groupby(['scale', 'n', 'n0', 'tau', 'N', 'alpha', 'sigma']))
#df = df[['tau', 'accuracy', 'fn']].sort_values(by=['tau'])
#df = df[['scale', 'tau', 'sigma', 'time']].sort_values(by=['scale', 'tau', 'sigma'])
print(df)


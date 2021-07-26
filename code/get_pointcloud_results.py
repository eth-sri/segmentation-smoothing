#!/usr/bin/env python3

import pathlib
import os
from glob2 import glob
from argparse import Namespace
import argparse
import pandas as pd
import re
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', choices=['partseg_l2', 'partseg_rot'], default='partseg_l2')
cargs = parser.parse_args()


log_dir = pathlib.Path('../logs/')
dirs = os.listdir(log_dir)
dirs = filter(lambda x: os.path.isdir(log_dir / x), dirs)
dirs = filter(lambda x: x.startswith(cargs.dataset), dirs)


STATS_COLS = ['fn', 'name', 'corr', 'baseline', 'n', 'tau', 'N', 'alpha', 'sigma', 'batch_size', 'n0', 'normal', 'num_point', 'num_votes', 'model', 'acc', 'class_avg_iou', 'class_avg_accuracy', 'instance_avg_iou', 'abstain', 'time']
df = pd.DataFrame([], columns=STATS_COLS)
experiments = {}

for dir in dirs:
    dir = log_dir / dir
    log = dir / 'log.log'

    if log.exists():
        with open(log, 'r') as f:
            lines = f.readlines()
            args = eval(lines[0])
        if len(lines) <= 100: continue
    else: continue

    print(dir)
    print(args)
    args = vars(args)
    read_args = ['N', 'alpha', 'sigma', 'batch_size', 'n0', 'normal', 'num_point', 'num_votes', 'log_dir']
    values_args = [None] * len(read_args) 
    for k, v in args.items():
        if k in read_args:
            i = read_args.index(k)
            values_args[i] = v

    ns = args['n']
    taus = args['tau']

    for i, l in enumerate(lines[::-1]):
        if l.startswith('baseline'): break

    summary_lines = lines[-(i+1):]
    print('>')
    print(summary_lines)
    print('<')
    if len(summary_lines) == 0: continue
    starts = [0] + [j+1 for j, l in enumerate(summary_lines) if l.strip() == ''][:-1]

    experiments = {}
    for start in starts:
        if start >= len(summary_lines): continue
        name = summary_lines[start].strip()
        print(summary_lines[start+1])
        acc = summary_lines[start+1].split(' ')[1].strip()
        class_avg_iou = summary_lines[start+2].split(' ')[1].strip()
        class_avg_accuracy = summary_lines[start+3].split(' ')[1].strip()
        instance_avg_iou = summary_lines[start+4].split(' ')[1].strip()
        acc = float(acc)
        class_avg_iou = float(class_avg_iou)
        class_avg_accuracy = float(class_avg_accuracy)
        instance_avg_iou = float(instance_avg_iou)
        experiments[name] = [acc, class_avg_iou, class_avg_accuracy, instance_avg_iou]

    for name in experiments.keys():
        baseline = (name == 'baseline')
        if baseline:
            abstain = 0
            time = None
            n, tau = None, None
            corr = None
        else:
            tkns = name.split('_') 
            if len(tkns) == 3:
                _, n, tau = tkns
                corr = 'Holm'
            else:
                _, corr, n, tau = tkns
            n = int(n.strip())
            tau = float(tau.strip())
            assert n in ns
            assert tau in taus
            time, abstain = 0, 0
            N = 0
            for line in lines[1:]:
                tokens = line.split(', ')
                lt = len(tokens)
                if line.startswith(name) and lt == 8:
                    name, idx, acc, nonabstain, t, ts, tp, tc = tokens
                    t = float(t.strip())
                    nonabstain = float(nonabstain.strip())
                    time += t
                    abstain += nonabstain
                    N += 1
            time = time / N
            abstain = 1 - abstain / N

        data = [dir, name, corr, baseline, n, tau] + values_args
        data += experiments[name] + [abstain, time]
        df = df.append(pd.DataFrame([data], columns=STATS_COLS))
    
    print()

df = df.sort_values(by=['model', 'normal', 'tau', 'sigma'])

print(df)


    
key = ['corr', 'n', 'tau', 'N', 'sigma', 'normal', 'model', 'acc', 'abstain', 'time']
df = df[key].sort_values(by=['model', 'n', 'tau', 'sigma', 'corr'])
with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 120):
    print(df)
#print(df.to_latex(index=False, float_format=lambda x: f"{x:.2f}"))

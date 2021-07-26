# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import time
from pathlib import Path

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


import scipy.stats as sps


def resize(logits, target, config):
  ph, pw = logits.size(2), logits.size(3)
  h, w = target.size(1), target.size(2)
  if ph != h or pw != w:
    logits = F.upsample(input=logits, size=(h, w), mode='bilinear')
  return logits





class FullModel(nn.Module):
  """
  Distribute the loss on multi-gpu to reduce 
  the memory cost in the main gpu.
  You can check the following discussion.
  https://discuss.pytorch.org/t/dataparallel-imbalanced-memory-usage/22551/21
  """
  def __init__(self, model, loss):
    super(FullModel, self).__init__()
    self.model = model
    self.loss = loss

  def forward(self, inputs, labels):
    outputs = self.model(inputs)
    loss = self.loss(outputs, labels)
    return torch.unsqueeze(loss,0), outputs

def get_world_size():
    if not torch.distributed.is_initialized():
        return 1
    return torch.distributed.get_world_size()

def get_rank():
    if not torch.distributed.is_initialized():
        return 0
    return torch.distributed.get_rank()

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = val * weight
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum += val * weight
        self.count += weight
        self.avg = self.sum / self.count

    def value(self):
        return self.val

    def average(self):
        return self.avg

def create_logger(cfg, cfg_name, phase='train'):
    root_output_dir = Path(cfg.OUTPUT_DIR)
    # set up logger
    if not root_output_dir.exists():
        print('=> creating {}'.format(root_output_dir))
        root_output_dir.mkdir()

    dataset = cfg.DATASET.DATASET
    model = cfg.MODEL.NAME
    cfg_name = os.path.basename(cfg_name).split('.')[0]

    final_output_dir = root_output_dir / dataset / cfg_name

    print('=> creating {}'.format(final_output_dir))
    final_output_dir.mkdir(parents=True, exist_ok=True)

    time_str = time.strftime('%Y-%m-%d-%H-%M')
    log_file = '{}_{}_{}.log'.format(cfg_name, time_str, phase)
    final_log_file = final_output_dir / log_file
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=str(final_log_file),
                        format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    tensorboard_log_dir = Path(cfg.LOG_DIR) / dataset / model / \
            (cfg_name + '_' + time_str)
    print('=> creating {}'.format(tensorboard_log_dir))
    tensorboard_log_dir.mkdir(parents=True, exist_ok=True)

    return logger, str(final_output_dir), str(tensorboard_log_dir)

def get_confusion_matrix(label, pred, size, num_class, ignore=-1):
    """
    Calcute the confusion matrix by given label and pred
    """
    output = pred.cpu().numpy().transpose(0, 2, 3, 1)
    seg_pred = np.asarray(np.argmax(output, axis=3), dtype=np.uint8)
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

def adjust_learning_rate(optimizer, base_lr, max_iters, 
        cur_iters, power=0.9):
    lr = base_lr*((1-float(cur_iters)/max_iters)**(power))
    optimizer.param_groups[0]['lr'] = lr
    return lr



  
class GaussianAugModel(FullModel):

  def __init__(self, model, loss, sigma=0, std=1):
    super(GaussianAugModel, self).__init__(model, loss)
    self.sigma = sigma
    self.std = torch.nn.Parameter(torch.tensor(std).reshape(1, 3, 1, 1), requires_grad=False)

  def forward(self, inputs, labels, *args, **kwargs):
    if self.sigma > 0:
      inputs = inputs + self.sigma * torch.randn_like(inputs) / self.std
    outputs = self.model(inputs, *args, **kwargs)
    loss = self.loss(outputs, labels)
    return torch.unsqueeze(loss,0), outputs

def kl_div(input, targets):
  return F.kl_div(F.log_softmax(input, dim=1), targets, reduction='none').sum(1)


def entropy(input):
    logsoftmax = torch.log(input.clamp(min=1e-20))
    xent = (-input * logsoftmax).sum(1)
    return xent


class ConsistencyRegModel(GaussianAugModel):

  def __init__(self, model, loss, config, m=2, ldb=1, eta=0.5, sigma=0, std=1):
    super(ConsistencyRegModel, self).__init__(model, loss, sigma, std)
    self.config = config
    self.m = m
    self.eta = eta
    self.ldb = ldb



  def forward(self, inputs, labels, *args, **kwargs):
    inputs_c = torch.cat([inputs + self.sigma * torch.randn_like(inputs) / self.std for _ in range(self.m)], dim=0)
    logits = self.model(inputs_c, *args, **kwargs)
    rep = [1] * labels.dim()
    rep[0] = self.m
    labels_c = labels.repeat(rep) 
    loss = self.loss(logits, labels_c)
    loss = loss / self.m


    logit_chunks = torch.chunk(resize(logits, labels_c, self.config), self.m, dim=0)
    softmax = [F.softmax(l, dim=1) for l in logit_chunks]
    avg_softmax = sum(softmax) / self.m
    
    loss_kl = [kl_div(l, avg_softmax) for l in logit_chunks]
    loss_kl = sum(loss_kl) / self.m

    loss_ent = entropy(avg_softmax)
    consistency = self.ldb * loss_kl + self.eta * loss_ent
    loss = loss + consistency.mean()

    return torch.unsqueeze(loss,0), logits[:(logits.shape[0]//self.m), ...]


class PGDTrainModel(FullModel):

  def __init__(self, model, loss, config, mean=0, std=1):
    super(PGDTrainModel, self).__init__(model, loss)
    self.config = config
    self.std = torch.nn.Parameter(torch.tensor(std).reshape(1, 3, 1, 1), requires_grad=False)
    self.mean = torch.nn.Parameter(torch.tensor(mean).reshape(1, 3, 1, 1), requires_grad=False)

  def attack(self, inputs, labels, eps, nsteps, step_size, *args, **kwargs):
    training = self.model.training 
    self.model.eval()
    inputs = inputs * self.std + self.mean
    rp = torch.randn_like(inputs)
    norm = rp.norm()
    inp = torch.clamp(inputs + eps * rp / (norm + 1e-10), 0, 1)
    loss_fn = nn.CrossEntropyLoss(ignore_index=self.config.TRAIN.IGNORE_LABEL)

    for k in range(nsteps):
      inp = inp.clone().detach().requires_grad_(True)
      logits = self.model( (inp - self.mean) / self.std, *args, **kwargs)
      loss = loss_fn(resize(logits, labels, self.config), labels)
      grad, = torch.autograd.grad(loss, [inp])
      with torch.no_grad():
        grad_norm = grad.norm()
        grad_scaled = grad / (grad_norm + 1e-10)
        inp = inp + grad_scaled * step_size
        diff = inp - inputs 
        diff.renorm(p=2, dim=0, maxnorm=eps)
        inp = torch.clamp(inp + diff, 0, 1)

    if training:
     self.model.train() 
    return (inp.detach() - self.mean) / self.std


  def forward(self, inputs, labels, *args, **kwargs):
    eps = 0.05 #0.25 * sps.norm.ppf(0.75)
    steps = 1
    if torch.is_grad_enabled():
      inputs_a = self.attack(inputs, labels, eps, steps, eps*1.2/steps)
    else:
      inputs_a = inputs
    return super(PGDTrainModel, self).forward(inputs_a, labels, *args, **kwargs)


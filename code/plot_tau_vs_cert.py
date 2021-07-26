    #!/usr/bin/env python3

import pathlib
import os
from glob2 import glob
from argparse import Namespace
import pandas as pd
import re
import numpy as np
import argparse
import matplotlib.pyplot as plt
from matplotlib import rcParams
import scipy.stats as sps



def add_line(f, ax, sigma, n, tau, values, labels, power=4, label_above=True, legend='', label_h_delta=0, label_v_delta=0, marker='o'):
    for i in range(len(values)):
        values[i] = max(values[i:])
    radius = np.array([sigma * sps.norm.ppf(t) for t in tau]).ravel()
    p = ax.plot(radius, np.array(values)**power, '-', marker=marker, label=legend)
    if labels:
        s = 1 if label_above else -2.8
        for i in range(len(tau)):
            t, r, v = tau[i], radius[i], values[i]
            hoffset = label_h_delta
            if i > 0 and v < values[i-1]/2:
                hoffset += max(tau) * 0.05
            plt.text(r + hoffset,
                     (v + s * (label_v_delta + max(values) * 0.005))**power,
                     f"${t}$",
                     color=p[0].get_color(),
                     horizontalalignment='center')
    return values, radius, p


rcParams['pdf.fonttype'] = 42
rcParams['ps.fonttype'] = 42
rcParams['font.family'] = 'serif'
rcParams['font.sans-serif'] = ['Palatio']
rcParams['text.usetex'] = True
fontsize = 14

f = plt.figure()
ax = plt.gca()
power = 4

#sigma =0.25, n=300
values = [0.8612736031706911, 0.853143863282193, 0.844728079564649, 0.8322315581109954, 0.833350590416804, 0.8321445785003747, 0.0]
tau=[0.5, 0.6, 0.7, 0.8, 0.9, 0.92, 0.925]


values, radius, p = add_line(f, ax, sigma=0.25, n=300,
                             tau=tau,
                             values = values,
                             legend='$\sigma=0.25$',
                             labels=False,
                             power=power,
                             marker='o',
                             label_h_delta=-0.015,
                             label_v_delta=0.0005)

labels = list(map(str, tau)) 
lx = radius
lx[-1] += 0.03
lx[-2] += 0.015
lx[-3] += 0.0
ly = np.array(values) + 0.005
for l, x, y in zip(labels, lx, ly):
    plt.text(x,
             (y)**power,
             l,
             color=p[0].get_color(),
             horizontalalignment='center')


#sigma =0.33, n=300
values = [0.8587352180617177, 0.8490494625192792, 0.8391877819305982, 0.8228229241587158, 0.8231633377112763, 0.8223496164917078, 0.0]
tau=[0.5, 0.6, 0.7, 0.8, 0.9, 0.92, 0.925]

values, radius, p  = add_line(f, ax, sigma=0.33, n=300,
         tau=tau,
         values = values,
         legend='$\sigma=0.33$',
         labels=False,
         power=power,
         marker='^',
         label_h_delta=0.011,
         label_v_delta=0.001)

labels = list(map(str, tau))[1:]
lx = radius[1:]
lx[-1] += 0.03
lx[-3] += 0.0
ly = np.array(values[1:]) + 0.005
lx[0] += 0.01
ly[2] -= 4.5*0.003
lx[2] += 0.03
for k, (l, x, y) in enumerate(zip(labels, lx, ly)):
    plt.text(x,
             (y)**power,
             l,
             color=p[0].get_color(),
             horizontalalignment='center')

#sigma =0.5, n=300
values = [0.8501752720339318, 0.835649166597451, 0.8195368304509334, 0.7877080383860458, 0.7870471084709925, 0.7877164610229985, 0.0]
tau=[0.5, 0.6, 0.7, 0.8, 0.9, 0.91, 0.925]

values, radius, p  = add_line(f, ax, sigma=0.5, n=300,
         tau=tau,
         values = values,
         labels=False, label_above=False,
         legend='$\sigma=0.50$',
         power=power,
         marker='s',
         label_h_delta=-0.011,
         label_v_delta=0.0015)

labels = list(map(str, tau))
lx = radius
#lx[-2] -= 0.025
lx[-1] -= 0.03
lx[-2] += 0.03
ly = np.array(values) - 0.005
ly[-1] += 2* 0.005
ly[-2] += 0.005
for l, x, y in zip(labels, lx, ly):
    plt.text(x,
             (y)**power,
             l,
             color=p[0].get_color(),
             verticalalignment='top',
             horizontalalignment='center')

ticks = np.array([0, 0.5, 0.6, 0.7, 0.8, 0.85])
ax.set_yticks(ticks**(power))
ax.set_yticklabels([f"{t:.2f}" for t in ticks])
plt.legend(loc='upper right')

plt.ylabel('cert. acc.', rotation=0, fontsize=fontsize)
ax.yaxis.set_label_coords(0.01, 1.02)

plt.xlabel('$R$', rotation=0, fontsize=fontsize)
ax.xaxis.set_label_coords(1.03, 0.01)

ax.grid(which='major', color=(1,1,1))
ax.set_facecolor( (0.97, 0.97, 0.97) )
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)
plt.setp(ax.get_xticklabels(), fontsize=fontsize)
plt.setp(ax.get_yticklabels(), fontsize=fontsize)

f.tight_layout()

plt.savefig('rvsacc_semseg.eps')



rcParams['pdf.fonttype'] = 42
rcParams['ps.fonttype'] = 42
rcParams['font.family'] = 'serif'
rcParams['font.sans-serif'] = ['Palatio']
rcParams['text.usetex'] = True
fontsize = 14

f = plt.figure()
ax = plt.gca()

power = 1 

#accuracy
tau=[0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 0.999]
values = [0.87, 0.85, 0.82, 0.78, 0.70, 0.62, 0.39, 0.00]

values, radius, p  = add_line(f, ax, sigma=0.25, n=10000,
         tau=tau,
         power=power,
         legend='$f^n_{\sigma=0.25}, \sigma=0.25, n=10000$',
         values=values,
         marker='o',
         labels=False,
         label_above=False,
         label_h_delta=-0.001,
         label_v_delta=0.001)


labels = list(map(str, tau))
lx = radius + 0.05
#lx[-1] -= 0.03
ly = np.array(values) + 2* 0.005
#ly[-1] += 2* 0.005
ly[-3] += 10 * -0.005
lx[-3] += 0.01
for l, x, y in zip(labels, lx, ly):
    plt.text(x,
             (y)**power,
             l,
             color=p[0].get_color(),
             verticalalignment='bottom',
             horizontalalignment='center')


#0.25, 1000n
tau=[0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.96, 0.97, 0.98, 0.99]
values = [0.8639599609375, 0.8603173828125, 0.8050244140625, 0.7529296875, 0.657705078125, 0.533115234375, 0.482412109375, 0.400224609375, 0.2775, 0.0]
#0.91, 0.92, 
#0.6415185546875, 0.621552734375, 

values, radius, p  = add_line(f, ax, sigma=0.25, n=1000,
         tau=tau,
         power=power,
         legend='$f^n_{\sigma=0.25}, \sigma=0.25, n=1000$',
         values=values,
         marker='^',
         labels=False,
         label_above=False,
         label_h_delta=-0.001,
         label_v_delta=0.001)


labels = list(map(str, tau))
lx = radius - 0.06
ly = np.array(values) - 0.005
for l, x, y in list(zip(labels, lx, ly))[-4:]:
    plt.text(x,
             (y)**power,
             l,
             color=p[0].get_color(),
             verticalalignment='bottom',
             horizontalalignment='center')




tau=[0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 0.999]
values = [0.77, 0.75, 0.71, 0.67, 0.60, 0.53, 0.37, 0.00]


values, radius, p  = add_line(f, ax, sigma=0.5, n=10000,
         tau=tau,
         power=power,
         labels=False,
         marker='s',
         legend='$f^n_{\sigma=0.50}, \sigma=0.50, n=10000$',
         values =values,
         label_h_delta=0.001,
         label_v_delta=0.001)


labels = list(map(str, tau))
lx = radius + 0.05
lx[-1] -= 0.15
ly = np.array(values) + 2 * 0.005
ly[-1] -= 0.005
for l, x, y in list(zip(labels, lx, ly))[-5:]:
    plt.text(x,
             (y)**power,
             l,
             color=p[0].get_color(),
             verticalalignment='bottom',
             horizontalalignment='center')

#0.5, 1000n
tau=[0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.96, 0.97, 0.98, 0.99]
values = [0.7615087890625, 0.72833984375, 0.6928125, 0.6428564453125, 0.560888671875, 0.454951171875, 0.4242822265625, 0.3760791015625, 0.2989794921875, 0.0]
#0.91, 0.92, 
#0.5448583984375, 0.52552734375, 
values, radius, p  = add_line(f, ax, sigma=0.5, n=1000,
         tau=tau,
         power=power,
         labels=False,
         marker='d',
         legend='$f^n_{\sigma=0.50}, \sigma=0.50, n=1000$',
         values =values,
         label_h_delta=0.001,
         label_v_delta=0.001)


labels = list(map(str, tau))
lx = radius - 0.05
#lx[-1] -= 0.03
ly = np.array(values) - 0.005
#ly[-1] += 2* 0.005
for k, (l, x, y) in enumerate(zip(labels, lx, ly)):
    if k == 3: continue
    plt.text(x,
             (y)**power,
             l,
             color=p[0].get_color(),
             verticalalignment='top',
             horizontalalignment='center')


ticks = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
ax.set_yticks(ticks**(power))
ax.set_yticklabels([f"{t:.2f}" for t in ticks])

plt.ylabel('cert. acc.', rotation=0, fontsize=fontsize)
ax.yaxis.set_label_coords(0.01, 1.02)

plt.xlabel('$R$', rotation=0, fontsize=fontsize)
ax.xaxis.set_label_coords(1.03, 0.01)

ax.grid(which='major', color=(1,1,1))
ax.set_facecolor( (0.97, 0.97, 0.97) )
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)
plt.setp(ax.get_xticklabels(), fontsize=fontsize)
plt.setp(ax.get_yticklabels(), fontsize=fontsize)

plt.legend(loc='upper right')

f.tight_layout()
plt.savefig('rvsacc_partseg.eps')

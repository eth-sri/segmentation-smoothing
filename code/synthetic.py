#!/usr/bin/env python3
import matplotlib
matplotlib.use('Agg')
import numpy as np
from multiclassify_utils import *
import matplotlib.pyplot as plt
from matplotlib import rcParams
import os
from matplotlib import rcParams
from tqdm  import tqdm, trange
from scipy.signal import savgol_filter


rcParams['pdf.fonttype'] = 42
rcParams['ps.fonttype'] = 42
rcParams['font.family'] = 'serif'
rcParams['font.sans-serif'] = ['Palatio']
rcParams['text.usetex'] = True


def gen(N, gamma=0.0, k=0):
    sample = np.ones((N,)) # 1 is the true class
    if gamma > 0:
        sample = (np.random.random(N) > gamma)
    if k > 0:
        sample[:k] = (np.random.random(k) > 5*gamma)
    return sample.astype(np.int32)

def sample(n, N, *args):
    return np.stack([gen(N, *args) for _ in range(n)])


def run(N, n, n0, taus, alpha, sigma, gamma=0.0, k=0,
      do_segpred=True,
      do_naivemulti=True,
      do_naivecarthesian=True,
      do_multitau=True,
      do_segpred_bonferroni=True,
      do_kFWER=[]):
    
    s = sample(n+n0, N, gamma, k)

    names, radii, nonabstain = [], [], []

    if do_naivecarthesian:
        names.append('carthesian')
        classes, R  = certify_naiveCarthesian(s, n0, n, sigma, alpha, parallel=True)
        nonabstain.append(1-np.sum(classes==-1)/N)
        radii.append(R)

    if do_naivemulti:
        names.append('multi')
        classes, R, _ = certify_naiveMulti(2, s, n0, n, sigma, None, alpha, parallel=True)
        nonabstain.append(1-np.sum(classes==-1)/N)
        radii.append(R)
        

    for tau in taus:

        if do_multitau:
            names.append(f'multi_tau={tau}')
            classes, R, _ = certify_naiveMulti(2, s, n0, n, sigma, tau, alpha, parallel=True)
            nonabstain.append(1-np.sum(classes==-1)/N)
            radii.append(R)

        if do_segpred:
            names.append(f'segpred_tau={tau}')
            classes, R, _ = certify(2, s, n0, n, sigma, tau, alpha, parallel=True)
            nonabstain.append(1-np.sum(classes==-1)/N)
            radii.append(R)
            for k in do_kFWER:
                names.append(f'segpred_tau={tau},k={k}')
                classes, R, _ = certify(2, s, n0, n, sigma, tau, alpha, parallel=True, correction='kfwer', kfwer_k=k)
                nonabstain.append(1-np.sum(classes==-1)/N)
                radii.append(R)

        if do_segpred_bonferroni:
            names.append(f'segpred_bonferroni_tau={tau}')
            classes, R, _ = certify(2, s, n0, n, sigma, tau, alpha, parallel=True, correction='bonferroni')
            nonabstain.append(1-np.sum(classes==-1)/N)
            radii.append(R)
            
    return names, radii, nonabstain


fontsize = 14

cache_file = './plot1.np' 
rep = 600 #20
N = 100
n0 = 100
n = 100
tau = 0.75
alpha = 0.001
delta = 0.001
gammas = np.arange(0, 0.1, delta)

if os.path.exists(cache_file):
    nonabstain1 = np.loadtxt(cache_file)
else:
    nonabstain1 = []
    for gamma in tqdm(gammas):
        v = []
        for r in trange(rep, leave=False):
            names, _, na = run(N, n0, n, [tau], alpha, sigma=1, gamma=gamma, k=1, do_multitau=False, do_segpred_bonferroni=True)
            v.append(na)
        nonabstain1.append(np.mean(np.array(v), axis=0))
    nonabstain1 = np.stack(nonabstain1)
    print(names)
    print(nonabstain1.T)
    np.savetxt(cache_filie, nonabstain1)

for smoothed in [True, False]:
    f = plt.figure()
    ax = plt.gca()

    colors = []
    for i in range(4):
        if smoothed:
            p = ax.plot(gammas, np.clip(savgol_filter(nonabstain1[:, i], 11, 1), 0, 1))
        else:
            p = ax.plot(gammas, nonabstain1[:, i])
        colors.append(p[0].get_color())



    plt.ylabel('\% certified', rotation=0, fontsize=fontsize)
    ax.yaxis.set_label_coords(0.01, 1.02)

    plt.xlabel('$\gamma$', rotation=0, fontsize=fontsize)
    ax.xaxis.set_label_coords(1.03, 0.075)


    delta = 0.025
    ax.set_xticks(np.arange(0, 0.1 + delta, delta))
    ax.set_yticks(np.arange(0, 1.02, 0.2))

    ax.grid(which='major', color=(1,1,1))
    ax.set_facecolor( (0.97, 0.97, 0.97) )
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    plt.setp(ax.get_xticklabels(), fontsize=fontsize)
    plt.setp(ax.get_yticklabels(), fontsize=fontsize)


    plt.text(0.015, 0.05, r'\textsc{JointClass}', fontsize=fontsize, color=colors[0])
    plt.text(0.032, 0.5, r'\textsc{IndivClass}', fontsize=fontsize, color=colors[1])
    plt.text(0.065, 0.9, r'\textsc{SegCertifyHolm}', fontsize=fontsize, color=colors[2])
    plt.text(0.06875, 0.15, r'\textsc{SegCertifyBon}', fontsize=fontsize, color=colors[3])
    #
    f.tight_layout()
    fn = 'plot_abstain_gamma'
    if smoothed: fn += '_smoothed'
    fn += '.eps'
    plt.savefig(fn)



rep = 1 #20
n0 = 1000
n = 1000
tau = 0.90
alpha = 0.1
gamma = 0.05
Ns = [100, 10**3, 10**4, 10**5, 10**6]

cache_file = './plot2.np' 

if os.path.exists(cache_file):
    nonabstain2 = np.loadtxt(cache_file)
else:
    nonabstain2 = []
    for N in tqdm(Ns):
        v = []
        for r in trange(rep, leave=False):
            names, _, na = run(N, n0, n, [tau], alpha, sigma=1, gamma=gamma, k=1, do_multitau=False, do_segpred_bonferroni=True)
            v.append(na)
        nonabstain2.append(np.mean(np.array(v), axis=0))
            
    nonabstain2 = np.stack(nonabstain2)
    print(names)
    print(nonabstain2.T)
    np.savetxt(cache_file, nonabstain2)

    

f = plt.figure()
ax = plt.gca()

for i in range(2):
    p = ax.semilogx(Ns, nonabstain2[:, i], marker='x', color=colors[2+i])

plt.ylabel('\% certified', rotation=0, fontsize=fontsize)
ax.yaxis.set_label_coords(0.01, 1.02)

plt.xlabel('$N$', rotation=0, fontsize=fontsize)
ax.xaxis.set_label_coords(1.03, 0.05)

    
ax.grid(which='major', color=(1,1,1))
ax.set_facecolor( (0.97, 0.97, 0.97) )
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)
plt.setp(ax.get_xticklabels(), fontsize=fontsize)
plt.setp(ax.get_yticklabels(), fontsize=fontsize)

plt.text(0.7 * 10**5, 0.99, r'\textsc{SegCertifyHolm}', fontsize=fontsize, color=colors[2])
plt.text(10**4 - 6 * 10**3, 0.85, r'\textsc{SegCertifyBon}', fontsize=fontsize, color=colors[3])

    
f.tight_layout()
fn = 'plot_abstain_N.eps'
plt.savefig(fn)







cache_file = './plot3.np' 
rep = 30 #20
N = 100
n0 = 1000
n = 1000
tau = 0.75
alpha = 0.001
delta = 0.001
gammas = np.arange(0, 0.1, delta)


if os.path.exists(cache_file):
    nonabstain3 = np.loadtxt(cache_file)
else:
    nonabstain3 = []
    for gamma in tqdm(gammas):
        v = []
        for r in trange(rep, leave=False): 
            names, _, na = run(N, n0, n, [tau], alpha, sigma=1, gamma=gamma, k=1, do_multitau=False, do_segpred_bonferroni=True)
            v.append(na)
        nonabstain3.append(np.mean(np.array(v), axis=0))
    nonabstain3 = np.stack(nonabstain3)
    print(names)
    print(nonabstain3.T)
    np.savetxt(cache_file, nonabstain3)

for smoothed in [True, False]:
    f = plt.figure()
    ax = plt.gca()

    for i in range(4):
        if smoothed:
            p = ax.plot(gammas, np.clip(savgol_filter(nonabstain3[:, i], 11, 1), 0, 1))
        else:
            p = ax.plot(gammas, nonabstain3[:, i])
        colors.append(p[0].get_color())

    plt.ylabel('\% certified', rotation=0, fontsize=fontsize)
    ax.yaxis.set_label_coords(0.01, 1.02)

    plt.xlabel('$\gamma$', rotation=0, fontsize=fontsize)
    ax.xaxis.set_label_coords(1.03, 0.075)


    delta = 0.025
    ax.set_xticks(np.arange(0, 0.1 + delta, delta))
    ax.set_yticks(np.arange(0, 1.02, 0.2))

    ax.grid(which='major', color=(1,1,1))
    ax.set_facecolor( (0.97, 0.97, 0.97) )
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    plt.setp(ax.get_xticklabels(), fontsize=fontsize)
    plt.setp(ax.get_yticklabels(), fontsize=fontsize)

    f.tight_layout()
    fn = 'plot_abstain_n_gamma'
    if smoothed: fn += '_smoothed'
    fn += '.eps'
    plt.savefig(fn)




rep = 1 #20
n0 = 1000
n = 1000
tau = 0.90
alpha = 0.1
gamma = 0.05
Ns = [100, 10**3, 10**4, 10**5, 10**6]

cache_file = './plot4.np' 

if os.path.exists(cache_file):
    nonabstain2 = np.loadtxt(cache_file)
else:
    nonabstain2 = []
    for N in tqdm(Ns):
        v = []
        for r in trange(rep, leave=False):
            names, _, na = run(N, n0, n, [tau], alpha, sigma=1,
                               gamma=gamma, k=1, do_multitau=False, do_segpred_bonferroni=False, do_kFWER=[2, 3, 5, max(N//1000, 1), max(N//100, 1)],
                               do_naivemulti=False, do_naivecarthesian=False) 
            v.append(na) 
        nonabstain2.append(np.mean(np.array(v), axis=0))
            
    nonabstain2 = np.stack(nonabstain2)
    print(names)
    print(nonabstain2.T)
    np.savetxt(cache_file, nonabstain2)

    

f = plt.figure()
ax = plt.gca()

for i in range(6):
    p = ax.semilogx(Ns, nonabstain2[:, i], marker='x')


plt.ylabel('\% certified', rotation=0, fontsize=fontsize)
ax.yaxis.set_label_coords(0.01, 1.02)

plt.xlabel('$N$', rotation=0, fontsize=fontsize)
ax.xaxis.set_label_coords(1.03, 0.05)


ax.legend([r'\textsc{SegCertifyHolm} ($b=0$)', r'$b=1$', r'$b=2$', r'$b=4$', r'$b=0.01N$', r'$b=0.001N$'])

    
ax.grid(which='major', color=(1,1,1))
ax.set_facecolor( (0.97, 0.97, 0.97) )
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)
plt.setp(ax.get_xticklabels(), fontsize=fontsize)
plt.setp(ax.get_yticklabels(), fontsize=fontsize)

    
f.tight_layout()
fn = 'plot_abstain_N_kfwer.eps'
plt.savefig(fn)




rep = 1 #20
n0 = 1000
n = 1000
tau = 0.90
alpha = 0.001
gamma = 0.05
Ns = [100, 10**3, 10**4, 10**5, 10**6]

cache_file = './plot5.np' 

if os.path.exists(cache_file):
    nonabstain2 = np.loadtxt(cache_file)
else:
    nonabstain2 = []
    for N in tqdm(Ns):
        v = []
        for r in trange(rep, leave=False):
            names, _, na = run(N, n0, n, [tau], alpha, sigma=1,
                               gamma=gamma, k=1, do_multitau=False, do_segpred_bonferroni=True, do_kFWER=[2, 3, 5, max(N//1000, 1), max(N//100, 1)],
                               do_naivemulti=False, do_naivecarthesian=False) 
            v.append(na)
        nonabstain2.append(np.mean(np.array(v), axis=0))
            
    nonabstain2 = np.stack(nonabstain2)
    print(names)
    print(nonabstain2.T)
    np.savetxt(cache_file, nonabstain2)

    

f = plt.figure()
ax = plt.gca()

for i in range(6):
    p = ax.semilogx(Ns, nonabstain2[:, i], marker='x')


plt.ylabel('\% certified', rotation=0, fontsize=fontsize)
ax.yaxis.set_label_coords(0.01, 1.02)

plt.xlabel('$N$', rotation=0, fontsize=fontsize)
ax.xaxis.set_label_coords(1.03, 0.05)

ax.legend([r'\textsc{SegCertifyHolm} ($b=0$)', r'$b=1$', r'$b=2$', r'$b=4$', r'$b=0.01N$', r'$b=0.001N$'])
    
ax.grid(which='major', color=(1,1,1))
ax.set_facecolor( (0.97, 0.97, 0.97) )
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)
plt.setp(ax.get_xticklabels(), fontsize=fontsize)
plt.setp(ax.get_yticklabels(), fontsize=fontsize)

    
f.tight_layout()
fn = 'plot_abstain_N_kfwer0001.eps'
plt.savefig(fn)


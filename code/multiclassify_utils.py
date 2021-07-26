import numpy as np
import statsmodels
import pathlib
import scipy.stats as sps
from statsmodels.stats.proportion import proportion_confint
from statsmodels.stats.multitest import multipletests
import time
from datetime import datetime
import sys
import multiprocessing as mp


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')



def mode(sample, n0, num_classes):
    counts = np.zeros((num_classes))
    for i in range(n0):
        counts[sample[i]] += 1
    c = np.argmax(counts)
    cnt = np.sum(sample[n0:] == c) 
    return c, cnt


def map_test(args):
    sample, n0, n, num_classes, tau = args # arguments a tuple to be compatible with multiprocessing map
    c, cnt = mode(sample, n0, num_classes)
    return c, sps.binom_test(cnt, n, tau, alternative='greater')

def map_testNaiveMulti(args):
    sample, n0, n, num_classes, alpha = args # arguments a tuple to be compatible with multiprocessing map
    c, cnt = mode(sample, n0, num_classes)
    pc = statsmodels.stats.proportion.proportion_confint(cnt, n, alpha=2*alpha, method="beta")[0]
    return c, pc



def certify(num_classes, sample, n0, n, sigma, tau, alpha, abstain=-1, parallel=False, correction='holm', kfwer_k=1):
    assert isinstance(sample, np.ndarray)
    shape = sample.shape
    assert len(shape) == 2
    t0 = time.time()
    n_samples = shape[0] # nr samples
    assert(n_samples == n0 + n)
    K = shape[1] # nr points
    if parallel:
        with mp.Pool() as pool:
            c_ps = pool.map(map_test, [(sample[:, i], n0, n, num_classes, tau) for i in range(K) ])
    else:
        c_ps = map(map_test, [(sample[:, i], n0, n, num_classes, tau) for i in range(K) ])
    classes, ps = zip(*c_ps)
    classes = np.array(list(classes))
    ps = np.array(list(ps))
    t1 = time.time()
    if correction == 'kfwer':
        sortind = np.argsort(ps)
        pvals = np.take(ps, sortind)
        ntests = len(pvals)
        alpha_ = kfwer_k * alpha / np.arange(ntests, 0, -1)
        alpha_[:kfwer_k] = kfwer_k * alpha / ntests
        notreject = pvals > alpha_
        nr_index = np.nonzero(notreject)[0]
        if nr_index.size == 0:
            # nonreject is empty, all rejected
            notrejectmin = len(pvals)
        else:
            notrejectmin = np.min(nr_index)
        notreject[notrejectmin:] = True
        reject_ = ~notreject
        reject = np.empty_like(reject_)
        reject[sortind] = reject_
    else:
        reject, _, _ , _= multipletests(ps, alpha=alpha, method=correction)
    I = np.logical_not(reject)
    classes[I] = abstain
    t2 = time.time()
        #time_total = t2 - t0
    time_pvals = t1 - t0
    time_correction = t2 - t1
    radius = sigma * sps.norm.ppf(tau)
    return classes, radius, (time_pvals, time_correction)


def certify_naiveMulti(num_classes, sample, n0, n, sigma, tau, alpha, abstain=-1, parallel=False):
    assert isinstance(sample, np.ndarray)
    shape = sample.shape
    assert len(shape) == 2
    t0 = time.time()
    n_samples = shape[0] # nr samples
    assert(n_samples == n0 + n)
    K = shape[1] # nr points
    alpha = alpha / K # bonferroni
    if parallel:
        with mp.Pool() as pool:
            c_ps = pool.map(map_testNaiveMulti, [(sample[:, i], n0, n, num_classes, alpha) for i in range(K) ])
    else:
        c_ps = map(map_testNaiveMulti, [(sample[:, i], n0, n, num_classes, alpha) for i in range(K) ])
    classes, ps = zip(*c_ps)
    classes = np.array(list(classes))
    ps = np.array(list(ps))
    t1 = time.time()

    if tau is not None:
        R = sigma * sps.norm.ppf(tau)
        classes[ps < tau] = abstain
    else:
        p = np.min(ps)
        if p >= 0.5:
            R = sigma * sps.norm.ppf(p)
        else:
            classes[:] = abstain
            R = None
    t2 = time.time()
    time_pvals = t1 - t0
    time_correction = t2 - t1
    return classes, R, (time_pvals, time_correction)


def certify_naiveCarthesian(sample, n0, n, sigma, alpha, abstain=-1, parallel=False):
    assert isinstance(sample, np.ndarray)
    shape = sample.shape
    assert len(shape) == 2
    t0 = time.time()
    n_samples = shape[0] # nr samples
    assert(n_samples == n0 + n)
    K = shape[1] # nr points
    patterns, count = np.unique(sample[:n0, :], axis=0, return_counts=True)
    c = np.argmax(count)
    pattern_c = patterns[c]
    nc = np.sum(np.sum(sample[n0:, :] == pattern_c, axis=1) == K)
    pc = statsmodels.stats.proportion.proportion_confint(nc, n, alpha=2*alpha, method="beta")[0]
    if pc >= 0.5:
        R = sigma * sps.norm.ppf(pc)
    else:
        R = None
        pattern_c[:] = abstain
    return pattern_c, R





class Log:

    def __init__(self, name):
        self.log_dir = pathlib.Path(__file__).parent.absolute() / '..' / 'logs'
        now = datetime.now()
        self.folder_path = self.log_dir / (name + "_"  + now.strftime("%Y_%m_%d_%H:%M:%S"))
        self.folder_path.mkdir(parents=True, exist_ok=True)
        self.filename = self.folder_path / "log.log"
        filepath = self.log_dir / self.filename
        self.log = open(filepath, 'a')

    def folder(self):
        return self.folder_path
        
    def write(self, message):
        message = str(message)
        sys.stdout.write(message)
        self.log.write(message)
        self.flush()


    def flush(self):
        sys.stdout.flush()
        self.log.flush()


def setup_segmentation_args(parser):
    parser.add_argument('-N', type=int, default=50, help='number of inputs')
    parser.add_argument('--sigma', type=float, default=0.25, help='sigma for randomized smoothing')
    parser.add_argument('--tau', type=float, default=[0.75], nargs='+', help='tau for randomized smoothing inference')
    parser.add_argument('-n', type=int, default=[1000], nargs='+', help='tau for randomized smoothing inference')
    parser.add_argument('-n0', type=int, default=100, help='')
    parser.add_argument('--alpha', type=float, default=0.0001, help='FWER signifiance')
    parser.add_argument('--baseline', type=str2bool, default=False, help='just run baseline')

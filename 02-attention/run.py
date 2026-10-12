"""
why does attention divide the scores by sqrt(d)?
compares no scaling, / sqrt(d) and / d, at init and in a small training test.

python run.py           full run, ~20 min on a laptop cpu
python run.py --quick   skips the d=512 training runs, ~5 min
writes figures/sqrt_d.png
"""

import sys
import time
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from transformer import swap_last2 # also puts 01-autograd on the path
from autograd import Tensor, softmax

OUT = Path(__file__).parent / 'figures' / 'sqrt_d.png'

# name -> (scale as a function of d, color, line style)
SCALES = {
    '÷√d': (lambda d: 1 / np.sqrt(d), '#2a78d6', '-'),
    'no scaling': (lambda d: 1.0, '#eb6834', '--'),
    '÷d': (lambda d: 1 / d, '#1baf7a', ':'),
}

# control for the training part: no scaling, but Wq/Wk start d**-0.25 smaller, so the scores
# start out exactly like ÷√d (std ~1, not saturated). if saturation at init was why the lr window
# shrinks, this one's window shouldn't shrink. (it does, see the README)
CONTROL = 'no scaling, small init'
CONTROL_COLOR = '#eda100'

def softmax_np(z):
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)

# -----------------------------------------------------------------------------
# part 1: what happens at init, no training

def measure_init(dims, T=32, trials=2000, chunk=250, seed=0):
    # 2000 queries per point. with 300 the median grad at d=1024 moved ~7x between seeds (~2.5x with 2000)
    rng = np.random.default_rng(seed)
    res = {name: {'std': [], 'entropy': [], 'max_w': [], 'grad': []} for name in SCALES}
    for d in dims:
        acc = {name: {'s': [], 'ent': [], 'mw': [], 'g': []} for name in SCALES}
        for _ in range(trials // chunk): # chunks so d=1024 doesn't need GBs of memory
            q = rng.standard_normal((chunk, 1, d))
            k = rng.standard_normal((chunk, T, d))
            g = rng.standard_normal((chunk, T)) # some random gradient coming into the weights
            raw = (q @ k.transpose(0, 2, 1))[:, 0, :]
            for name, (scale, _, _) in SCALES.items():
                s = raw * scale(d)
                p = softmax_np(s)
                plogp = np.where(p > 0, p * np.log(np.where(p > 0, p, 1.0)), 0.0)
                grad_s = p * (g - (p * g).sum(-1, keepdims=True)) # softmax backward
                acc[name]['s'].append(s)
                acc[name]['ent'].append(-plogp.sum(-1) / np.log(T)) # 1 = uniform, 0 = one token
                acc[name]['mw'].append(p.max(-1))
                acc[name]['g'].append(np.linalg.norm(grad_s, axis=-1))
        for name in SCALES:
            a = {key: np.concatenate(v) for key, v in acc[name].items()}
            res[name]['std'].append(a['s'].std())
            res[name]['entropy'].append(a['ent'].mean())
            res[name]['max_w'].append(a['mw'].mean())
            res[name]['grad'].append(np.median(a['g']))
    return res

# -----------------------------------------------------------------------------
# part 2: does it matter for training?
# task: 16 random tokens, the query is an exact copy of one of them, the head has to
# learn Wq/Wk so that the copy gets the attention. plain sgd on purpose

def train_retrieval(d, scale, lr, init_mult=1.0, steps=300, B=32, T=16, seed=0):
    r = np.random.default_rng(seed)
    Wq = Tensor(r.standard_normal((d, d)) / np.sqrt(d) * init_mult)
    Wk = Tensor(r.standard_normal((d, d)) / np.sqrt(d) * init_mult)
    data = np.random.default_rng(seed + 1)
    accs = []
    for _ in range(steps):
        x = data.standard_normal((B, T, d))
        j = data.integers(0, T, B)
        query = x[np.arange(B), j][:, None, :]
        Wq.grad, Wk.grad = np.zeros_like(Wq.data), np.zeros_like(Wk.data)
        p = softmax(((Tensor(query) @ Wq) @ swap_last2(Tensor(x) @ Wk)) * scale, axis=-1)
        loss = ((p @ Tensor(x) - query)**2).mean()
        loss.backward()
        if not np.isfinite(loss.data):
            return float('nan')
        Wq.data -= lr * Wq.grad
        Wk.data -= lr * Wk.grad
        accs.append((p.data[:, 0].argmax(-1) == j).mean())
    return float(np.mean(accs[-50:]))

def lr_window(accs, threshold=0.95):
    # range of learning rates that hit the threshold
    ok = [lr for lr, a in accs.items() if a >= threshold]
    return (min(ok), max(ok)) if ok else None

# -----------------------------------------------------------------------------
# plotting

BG, INK, INK2, MUTED, GRID, AXIS = '#fcfcfb', '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7'

def plot(dims, init, results, train_dims):
    plt.rcParams.update({
        'figure.facecolor': BG, 'axes.facecolor': BG, 'savefig.facecolor': BG,
        'axes.edgecolor': AXIS, 'axes.labelcolor': INK2, 'text.color': INK, 'axes.titlecolor': INK,
        'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.grid': True, 'grid.color': GRID,
        'grid.linewidth': 0.6, 'axes.spines.top': False, 'axes.spines.right': False,
        'lines.linewidth': 2, 'font.size': 10, 'legend.frameon': False,
    })
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4.6))

    for name, (_, color, ls) in SCALES.items():
        ax1.plot(dims, init[name]['entropy'], color=color, ls=ls, marker='o', ms=4, label=name)
        ax2.plot(dims, init[name]['grad'], color=color, ls=ls, marker='o', ms=4, label=name)
    for ax in (ax1, ax2):
        ax.set_xscale('log', base=2)
        ax.set_xticks(dims, [str(d) for d in dims])
        ax.set_xlabel('head size d')
    ax1.set_ylim(-0.03, 1.05)
    ax1.set_ylabel('attention entropy (1 = uniform, 0 = one token)')
    ax1.set_title('(a) At init: how spread out is attention?', loc='left')
    ax1.legend(loc='center right', fontsize=9)
    ax2.set_yscale('log')
    ax2.set_ylabel('gradient reaching the scores (median)')
    ax2.set_title('(b) At init: learning signal through softmax', loc='left')

    # control goes last so it sits next to ÷d, not next to no scaling (orange/yellow are hard to tell apart)
    offsets = {'no scaling': -0.27, '÷√d': -0.09, '÷d': 0.09, CONTROL: 0.27}
    colors = {name: c for name, (_, c, _) in SCALES.items()} | {CONTROL: CONTROL_COLOR}
    for name, color in colors.items():
        labeled = False
        for i, d in enumerate(train_dims):
            w = lr_window(results[d, name]) if (d, name) in results else None
            if w is None:
                continue
            x = i + offsets[name]
            ax3.plot([x, x], [w[0], w[1]], color=color, lw=5, solid_capstyle='round',
                     label=None if labeled else name)
            labeled = True
    ax3.set_yscale('log')
    ax3.set_xticks(range(len(train_dims)), [str(d) for d in train_dims])
    ax3.set_xlim(-0.6, len(train_dims) - 0.4)
    ax3.set_xlabel('head size d')
    ax3.set_ylabel('learning rates that train (≥ 95% accuracy)')
    ax3.set_title('(c) Training: which learning rates work?', loc='left')
    ax3.legend(loc='upper center', bbox_to_anchor=(0.5, -0.17), ncol=2, fontsize=9) # below, 4 entries cover the bars otherwise

    fig.suptitle('Without √d, attention saturates and the usable learning rate shrinks as heads grow',
                 x=0.01, ha='left', fontsize=12)
    plt.tight_layout()
    OUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUT, dpi=150)

# -----------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true', help='skip the slow d=512 training runs')
    args = parser.parse_args()

    dims = [8, 16, 32, 64, 128, 256, 512, 1024]
    init = measure_init(dims)
    print("at init (T=32 keys, 2000 random queries per point)")
    for name in SCALES:
        print(f"  {name}")
        for i, d in enumerate(dims):
            r = init[name]
            print(f"    d={d:5d}  score std {r['std'][i]:6.2f}  entropy {r['entropy'][i]:.2f}"
                  f"  max weight {r['max_w'][i]:.2f}  grad {r['grad'][i]:.1e}")

    lrs = [0.03, 0.1, 0.3, 1, 3, 10, 30, 100]
    runs = [(d, name) for d in [16, 64, 256] for name in list(SCALES) + [CONTROL]]
    if not args.quick:
        runs += [(512, name) for name in list(SCALES) + [CONTROL]]
    train_dims = [16, 64, 256] + ([] if args.quick else [512])

    print("\ntraining sweep (accuracy over the last 50 steps)")
    results = {}
    t0 = time.perf_counter()
    for d, name in runs:
        if name == CONTROL:
            scale, init_mult = 1.0, d**-0.25
        else:
            scale, init_mult = SCALES[name][0](d), 1.0
        results[d, name] = {lr: train_retrieval(d, scale, lr, init_mult) for lr in lrs}
        accs = '  '.join(f"{lr:g}:{a:.2f}" for lr, a in results[d, name].items())
        print(f"  d={d:4d} {name:22s} {accs}   [{time.perf_counter() - t0:.0f}s]", flush=True)

    print("\nlearning rates that reach >= 95%")
    for name in list(SCALES) + [CONTROL]:
        row = []
        for d in train_dims:
            w = lr_window(results[d, name]) if (d, name) in results else None
            row.append('-' if (d, name) not in results else (f"{w[0]:g}-{w[1]:g}" if w else 'none'))
        print(f"  {name:22s} " + "  ".join(f"d={d}: {c:10s}" for d, c in zip(train_dims, row)))

    plot(dims, init, results, train_dims)
    print(f"\nsaved {OUT}")

"""
why does training without warmup get stuck at the unigram loss?
same model and data as train.py, lr 3e-2, no clipping. part 1 compares no warmup / warmup /
biases frozen / qk-layernorm for 200 steps, part 2 is the causal test, part 3 redoes 02's sqrt(d) sweep with adam.

python run.py           full run, ~35 min on a laptop cpu
python run.py --quick   skips the two suspect runs and part 3, ~17 min
writes figures/warmup.png
"""

import gc
import math
import time
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from gpt import CONFIG, load_data, get_batch, baselines, init_gpt, block_params, gpt
from transformer import block, layer_norm, split_heads, merge_heads, attention, mlp, swap_last2
from autograd import Tensor, softmax, cross_entropy, embedding
from train import AdamW

OUT = Path(__file__).parent / 'figures' / 'warmup.png'
LR = 3e-2 # 10x what train.py uses, makes the problem obvious
BIASES = ('ln1_b', 'ln2_b', 'b1', 'b2', 'lnf_b')

tok, train_data, val_data = load_data()
cfg = dict(CONFIG, V=len(tok))
XD, _ = get_batch(val_data, 4, cfg['T'], np.random.default_rng(7)) # the 4 sequences everything is measured on

# -----------------------------------------------------------------------------
# the model, plus a qk-layernorm variant (layernorm on q and k per head, the fix from wortsman et al. 2023)

def init(qk=False):
    p = init_gpt(cfg, np.random.default_rng(0))
    if qk:
        hs = cfg['C'] // cfg['H']
        for i in range(cfg['L']):
            p[f'h{i}.qn_g'], p[f'h{i}.qn_b'] = np.ones(hs), np.zeros(hs)
            p[f'h{i}.kn_g'], p[f'h{i}.kn_b'] = np.ones(hs), np.zeros(hs)
    return p

def block_qk(x, p, H):
    h = layer_norm(x, p['ln1_g'], p['ln1_b'])
    q, k, v = split_heads(h @ p['Wq'], H), split_heads(h @ p['Wk'], H), split_heads(h @ p['Wv'], H)
    q, k = layer_norm(q, p['qn_g'], p['qn_b']), layer_norm(k, p['kn_g'], p['kn_b'])
    x = x + merge_heads(attention(q, k, v)) @ p['Wo']
    h = layer_norm(x, p['ln2_g'], p['ln2_b'])
    return x + mlp(h, p['W1'], p['b1'], p['W2'], p['b2'])

def loss_fn(P, x, y, qk=False):
    T = x.shape[1]
    h = embedding(P['wte'], x) + embedding(P['wpe'], np.arange(T))
    for i in range(cfg['L']):
        h = (block_qk if qk else block)(h, block_params(P, i), cfg['H'])
    logits = layer_norm(h, P['lnf_g'], P['lnf_b']) @ P['wte'].transpose((1, 0))
    return cross_entropy(logits.reshape(-1, cfg['V']), y.reshape(-1))

# -----------------------------------------------------------------------------
# measuring: the same forward in plain numpy (no graph), keeping what we want to look at

def _ln(x, g, b, eps=1e-5):
    xc = x - x.mean(-1, keepdims=True)
    return xc * ((xc**2).mean(-1, keepdims=True) + eps)**-0.5 * g + b

def _softmax(z):
    e = np.exp(z - z.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)

def pos_variation(x):
    # how much positions differ, relative to the part they all share. ~0 = every position is the same
    shared = x.mean(1, keepdims=True)
    return float(np.linalg.norm(x - shared) / np.linalg.norm(np.broadcast_to(shared, x.shape)))

def measure(p, qk=False):
    B, T = XD.shape
    C, H = cfg['C'], cfg['H']
    split = lambda a: a.reshape(B, T, H, C // H).transpose(0, 2, 1, 3)
    mask = np.triu(np.full((T, T), -np.inf), k=1)
    x = p['wte'][XD] + p['wpe'][:T]
    pos_var = [pos_variation(x)]
    for i in range(cfg['L']):
        g = lambda k: p[f'h{i}.{k}']
        h = _ln(x, g('ln1_g'), g('ln1_b'))
        q, k, v = split(h @ g('Wq')), split(h @ g('Wk')), split(h @ g('Wv'))
        if qk:
            q, k = _ln(q, g('qn_g'), g('qn_b')), _ln(k, g('kn_g'), g('kn_b'))
        a = _softmax(q @ k.transpose(0, 1, 3, 2) / math.sqrt(C // H) + mask)
        x = x + (a @ v).transpose(0, 2, 1, 3).reshape(B, T, C) @ g('Wo')
        h = _ln(x, g('ln2_g'), g('ln2_b'))
        u = h @ g('W1') + g('b1')
        x = x + 0.5 * u * (1 + np.tanh(math.sqrt(2 / math.pi) * (u + 0.044715 * u**3))) @ g('W2') + g('b2')
        pos_var.append(pos_variation(x))
    logits = _ln(x, p['lnf_g'], p['lnf_b']) @ p['wte'].T
    probs = _softmax(logits)
    avg = probs.reshape(-1, cfg['V']).mean(0)
    # context measure: how much the prediction changes between positions. 0 = a unigram model
    context = float((probs * (np.log(probs + 1e-30) - np.log(avg + 1e-30))).sum(-1).mean())
    return {'pos_var_block0': pos_var[1], 'context': context}, logits

# -----------------------------------------------------------------------------
# part 1 and 2: training runs

def top_direction(M):
    U, s, Vt = np.linalg.svd(M, full_matrices=False)
    return s[0] * np.outer(U[:, 0], Vt[0])

def edit_update(P, before, how):
    # causal test: after each step, remove the strongest direction from every matrix's update,
    # or (the control) shrink the whole update to the same size and keep every direction
    for k, p in P.items():
        if p.data.ndim != 2:
            continue
        dW = p.data - before[k]
        new = dW - top_direction(dW)
        if how == 'shrink':
            new = dW * np.linalg.norm(new) / np.linalg.norm(dW)
        p.data = before[k] + new

def run(lr=LR, warmup=0, steps=200, freeze_biases=False, qk=False, edit=None):
    P = {k: Tensor(v) for k, v in init(qk).items()}
    opt = AdamW({k: p for k, p in P.items() if not (freeze_biases and k.split('.')[-1] in BIASES)})
    rng = np.random.default_rng(1)
    log = {'loss': [], 'step': [], 'pos_var': [], 'context': []}
    for step in range(steps + 1):
        if step <= 30 or step % 10 == 0:
            m, _ = measure({k: p.data for k, p in P.items()}, qk)
            log['step'].append(step)
            log['pos_var'].append(m['pos_var_block0'])
            log['context'].append(m['context'])
        if step == steps:
            break
        before = {k: p.data.copy() for k, p in P.items()} if edit else None
        for p in P.values():
            p.grad = np.zeros_like(p.data)
        loss = loss_fn(P, *get_batch(train_data, 32, cfg['T'], rng), qk)
        loss.backward()
        log['loss'].append(float(loss.data))
        del loss
        opt.step(lr * min(1, (step + 1) / warmup) if warmup else lr)
        if edit:
            edit_update(P, before, edit)
        gc.collect() # the graph is a reference cycle, see train.py
    return log

# -----------------------------------------------------------------------------
# part 3: 02's retrieval task with adam instead of sgd

def train_retrieval_adam(d, scale, lr, steps=300, B=32, T=16, seed=0, b1=0.9, b2=0.99, eps=1e-8):
    r = np.random.default_rng(seed)
    W = [Tensor(r.standard_normal((d, d)) / np.sqrt(d)) for _ in range(2)]
    m = [np.zeros((d, d)) for _ in W]
    v = [np.zeros((d, d)) for _ in W]
    data = np.random.default_rng(seed + 1)
    accs = []
    for t in range(1, steps + 1):
        x = data.standard_normal((B, T, d))
        j = data.integers(0, T, B)
        query = x[np.arange(B), j][:, None, :]
        for w in W:
            w.grad = np.zeros_like(w.data)
        p = softmax(((Tensor(query) @ W[0]) @ swap_last2(Tensor(x) @ W[1])) * scale, axis=-1)
        loss = ((p @ Tensor(x) - query)**2).mean()
        loss.backward()
        if not np.isfinite(loss.data):
            return float('nan')
        for i, w in enumerate(W): # plain adam, no weight decay, so the only change from 02 is the rescaling
            m[i] = b1 * m[i] + (1 - b1) * w.grad
            v[i] = b2 * v[i] + (1 - b2) * w.grad**2
            w.data -= lr * (m[i] / (1 - b1**t)) / (np.sqrt(v[i] / (1 - b2**t)) + eps)
        accs.append((p.data[:, 0].argmax(-1) == j).mean())
    return float(np.mean(accs[-50:]))

# -----------------------------------------------------------------------------
# plotting

BG, INK, INK2, MUTED, GRID, AXIS = '#fcfcfb', '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7'
STYLE = {'no warmup': ('#eb6834', '--'), 'warmup': ('#2a78d6', '-'), 'biases frozen': ('#1baf7a', '-.'),
         'qk-layernorm': ('#eda100', ':'), 'top removed': ('#1baf7a', '-'), 'shrunk evenly': ('#eda100', ':')}

def plot(runs, base):
    plt.rcParams.update({
        'figure.facecolor': BG, 'axes.facecolor': BG, 'savefig.facecolor': BG,
        'axes.edgecolor': AXIS, 'axes.labelcolor': INK2, 'text.color': INK, 'axes.titlecolor': INK,
        'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.grid': True, 'grid.color': GRID,
        'grid.linewidth': 0.6, 'axes.spines.top': False, 'axes.spines.right': False,
        'lines.linewidth': 2, 'font.size': 10, 'legend.frameon': False,
    })
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4.6))
    smooth = lambda y: np.convolve(y, np.ones(10) / 10, mode='valid')

    for name in ('no warmup', 'biases frozen', 'qk-layernorm', 'warmup'):
        if name in runs:
            c, ls = STYLE[name]
            ax1.plot(np.arange(9, len(runs[name]['loss'])), smooth(runs[name]['loss']), color=c, ls=ls, label=name)
    for y, lab in [(base['unigram'], 'unigram'), (base['bigram'], 'bigram')]:
        ax1.axhline(y, color=MUTED, ls='--', lw=1)
        ax1.text(200, y + 0.03, lab, color=MUTED, fontsize=9, ha='right', va='bottom')
    ax1.set_xlabel('step')
    ax1.set_ylabel('train loss (10-step average)')
    ax1.set_title('(a) Without warmup it sticks at the unigram loss', loc='left')
    ax1.legend(loc='upper right')

    # causal test, first 30 steps. warmup in gray as the reference
    for name in ('no warmup', 'shrunk evenly', 'top removed', 'warmup'):
        log = runs[name]
        k = log['step'].index(30) + 1
        c, ls = (MUTED, '-') if name == 'warmup' else STYLE[name]
        lw = 1.2 if name == 'warmup' else 2
        ax2.plot(log['step'][:k], log['pos_var'][:k], color=c, ls=ls, lw=lw, label=name)
        ax3.plot(log['step'][:k], log['context'][:k], color=c, ls=ls, lw=lw, label=name)
    ax2.set_yscale('log')
    ax2.set_xlabel('step')
    ax2.set_ylabel('how different the positions are after block 0')
    ax2.set_title('(b) Removing one direction per update stops the collapse', loc='left')
    ax2.legend(loc='center right', fontsize=9)
    ax3.set_xlabel('step')
    ax3.set_ylabel('how much the prediction uses the context')
    ax3.set_title('(c) ...and the model uses the context again', loc='left')
    ax3.legend(loc='upper left', fontsize=9)

    fig.suptitle("Without warmup, Adam's first steps add one shared direction that makes every position look the same",
                 x=0.01, ha='left', fontsize=12)
    plt.tight_layout()
    OUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUT, dpi=150)

# -----------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true', help='skip the biases / qk-layernorm runs and the sqrt(d) part')
    args = parser.parse_args()
    base = baselines(train_data, val_data, cfg['V'])
    t0 = time.perf_counter()

    # the measurements are only worth something if measure() is the same model as gpt()
    p = init()
    err = np.abs(measure(p)[1] - gpt({k: Tensor(v) for k, v in p.items()}, XD, cfg).data).max()
    assert err < 1e-10, f"measure() doesn't match the model: {err:.1e}"

    print(f"part 1: lr {LR:g}, 200 steps, no clipping   (unigram {base['unigram']:.2f}, bigram {base['bigram']:.2f})")
    setups = [('no warmup', {}), ('warmup', {'warmup': 50})]
    if not args.quick:
        setups += [('biases frozen', {'freeze_biases': True}), ('qk-layernorm', {'qk': True})]
    runs = {}
    for name, kw in setups:
        runs[name] = run(**kw)
        log = runs[name]
        at = lambda key, s: log[key][log['step'].index(s)]
        print(f"  {name:14s} loss at 200 {log['loss'][-1]:.3f} | context at step 2 / 200: {at('context', 2):.3f} / "
              f"{at('context', 200):.3f} | positions after block 0 at step 2: {at('pos_var', 2):.2f}"
              f"   [{time.perf_counter() - t0:.0f}s]", flush=True)

    print("\npart 2: causal test, every update edited, 30 steps")
    for name, how in [('top removed', 'top'), ('shrunk evenly', 'shrink')]:
        runs[name] = run(steps=30, edit=how)
    for name in ('no warmup', 'shrunk evenly', 'top removed', 'warmup'):
        log = runs[name]
        i = log['step'].index(30)
        print(f"  {name:14s} step 30: positions after block 0 {log['pos_var'][i]:.2f}, context {log['context'][i]:.3f},"
              f" loss {np.mean(log['loss'][25:30]):.2f}", flush=True)
    print(f"  [{time.perf_counter() - t0:.0f}s]")

    if not args.quick:
        print("\npart 3: 02's sqrt(d) sweep with adam, learning rates that reach >= 95%")
        scales = {'÷√d': lambda d: 1 / np.sqrt(d), 'no scaling': lambda d: 1.0, '÷d': lambda d: 1 / d}
        lrs = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0]
        for name, scale in scales.items():
            row = []
            for d in [16, 64, 256]:
                ok = [lr for lr in lrs if train_retrieval_adam(d, scale(d), lr) >= 0.95]
                row.append(f"d={d}: {f'{min(ok):g}-{max(ok):g}' if ok else 'none':12s}")
                gc.collect()
            print(f"  {name:11s} " + "  ".join(row), flush=True)
        print(f"  [{time.perf_counter() - t0:.0f}s]")

    plot(runs, base)
    print(f"\nsaved {OUT}")

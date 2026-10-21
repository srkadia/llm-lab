"""
how much faster does generation get with a kv cache, and how much memory does it cost as the context grows?
the long contexts use the same architecture as 03 with random weights and a longer position table
(the trained model only has 128 positions). part 1 checks that the weights don't change the speed.

python run.py           ~2 min on a laptop cpu
python run.py --quick   stops the cached sweep at 2048 and skips the 512-token run, ~1 min
writes figures/kv_cache.png
"""

import time
import argparse
import tracemalloc
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from kv_cache import CKPT, load_checkpoint, forward, new_cache, forward_cached, prefill
from gpt import init_gpt

OUT = Path(__file__).parent / 'figures' / 'kv_cache.png'
CTX = [16, 32, 64, 128, 256, 512, 1024, 2048] # without the cache it stops at 2048, 4096 needs ~0.5 GB per score matrix
CTX_LONG = [4096, 8192, 16384] # cached only

# -----------------------------------------------------------------------------
# timing and memory

def best_time(f, repeats=5):
    # laptop timings jump around, take the fastest of a few
    best = float('inf')
    for _ in range(repeats):
        t0 = time.perf_counter()
        f()
        best = min(best, time.perf_counter() - t0)
    return best

def time_uncached(p, cfg, ids, t, repeats=5):
    return best_time(lambda: forward(p, ids[:, :t], cfg), repeats)

def time_cached(p, cfg, ids, t, repeats=7):
    # fill t-1 positions once, then time decoding token t
    cache = new_cache(cfg)
    prefill(p, ids[:, :t - 1], cache, cfg)
    def step():
        cache['n'] = t - 1
        forward_cached(p, ids[:, t - 1:t], cache, cfg)
    return best_time(step, repeats)

def peak_mb(f):
    tracemalloc.start()
    f()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak / 1e6

def greedy(p, cfg, n, cached):
    ids = [0]
    if cached:
        cache = new_cache(cfg)
        logits = forward_cached(p, np.array([ids]), cache, cfg)[0, -1]
    for _ in range(n):
        if not cached:
            logits = forward(p, np.array(ids)[None], cfg)[0, -1]
        ids.append(int(np.argmax(logits)))
        if cached:
            logits = forward_cached(p, np.array([[ids[-1]]]), cache, cfg)[0, -1]
    return ids

# -----------------------------------------------------------------------------
# plotting

BG, INK, INK2, MUTED, GRID, AXIS = '#fcfcfb', '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7'
BLUE, ORANGE = '#2a78d6', '#eb6834'

def plot(t_unc, t_cac, mem_unc, mem_cache, model_mb, C):
    plt.rcParams.update({
        'figure.facecolor': BG, 'axes.facecolor': BG, 'savefig.facecolor': BG,
        'axes.edgecolor': AXIS, 'axes.labelcolor': INK2, 'text.color': INK, 'axes.titlecolor': INK,
        'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.grid': True, 'grid.color': GRID,
        'grid.linewidth': 0.6, 'axes.spines.top': False, 'axes.spines.right': False,
        'lines.linewidth': 2, 'font.size': 10, 'legend.frameon': False,
    })
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.4))
    xs_u, xs_c = sorted(t_unc), sorted(t_cac)
    ax1.plot(xs_u, [t_unc[t] * 1e3 for t in xs_u], 'o--', color=ORANGE, ms=5, label='without cache')
    ax1.plot(xs_c, [t_cac[t] * 1e3 for t in xs_c], 'o-', color=BLUE, ms=5, label='with KV cache')
    ax1.axvline(6 * C, color=MUTED, ls=':', lw=1)
    ax1.text(6 * C * 1.1, 0.5, 't = 6C: attention starts\nto dominate a cached step', color=MUTED, fontsize=8, va='bottom')
    ax1.set_title('(a) Time per new token', loc='left')
    ax1.set_ylabel('ms (fastest of several runs)')

    xs = sorted(mem_unc)
    ax2.plot(xs, [mem_unc[t] for t in xs], 'o--', color=ORANGE, ms=5, label='without cache: peak during one step')
    ax2.plot(xs, [mem_cache[t] for t in xs], 'o-', color=BLUE, ms=5, label='the KV cache (kept)')
    ax2.axhline(model_mb, color=MUTED, ls=':', lw=1)
    ax2.text(xs[0], model_mb * 1.15, f"the model's weights, {model_mb:.2f} MB", color=MUTED, fontsize=8)
    ax2.set_title('(b) Memory', loc='left')
    ax2.set_ylabel('MB')
    for ax, ticks in ((ax1, xs_c[::2]), (ax2, xs)):
        ax.set_xscale('log', base=2)
        ax.set_yscale('log')
        ax.set_xticks(ticks, [f'{x:,}' for x in ticks])
        ax.set_xlabel('context length (tokens)')
        ax.legend(loc='upper left')
    fig.suptitle("A KV cache turns each new token from a full pass over the context into one position's work",
                 x=0.01, ha='left', fontsize=12)
    plt.tight_layout()
    OUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUT, dpi=150)

# -----------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true', help='cached sweep stops at 2048, no 512-token run')
    args = parser.parse_args()
    t0 = time.perf_counter()

    trained, cfg, tok, info = load_checkpoint(CKPT)
    C = cfg['C']

    print("part 1: do the weights change the speed? trained vs random, same shapes")
    random128 = init_gpt(cfg, np.random.default_rng(0))
    ids = np.random.default_rng(1).integers(0, cfg['V'], (1, 128))
    for t in (16, 64, 128):
        for name, timer in (('uncached', time_uncached), ('cached', time_cached)):
            a = b = float('inf')
            for _ in range(3): # alternate so neither gets a lucky moment
                a = min(a, timer(trained, cfg, ids, t, 3))
                b = min(b, timer(random128, cfg, ids, t, 3))
            print(f"  context {t:4d} {name:9s} trained {a * 1e3:6.2f} ms  random {b * 1e3:6.2f} ms  ratio {b / a:.2f}")

    print("\npart 2: time per new token")
    long_ctx = CTX if args.quick else CTX + CTX_LONG
    cfg_long = dict(cfg, T=long_ctx[-1])
    model = init_gpt(cfg_long, np.random.default_rng(0))
    ids = np.random.default_rng(2).integers(0, cfg['V'], (1, long_ctx[-1]))
    full = forward(model, ids[:, :1500], cfg_long)[0, -1] # still exact at a long context?
    cache = new_cache(cfg_long)
    prefill(model, ids[:, :1499], cache, cfg_long)
    err = np.abs(full - forward_cached(model, ids[:, 1499:1500], cache, cfg_long)[0, -1]).max()
    print(f"  (cached vs uncached at context 1500: {err:.1e})")
    t_unc = {t: time_uncached(model, cfg_long, ids, t, 3 if t >= 1024 else 5) for t in CTX}
    t_cac = {t: time_cached(model, cfg_long, ids, t) for t in long_ctx}
    for t in long_ctx:
        u = f"{t_unc[t] * 1e3:9.1f} ms" if t in t_unc else f"{'-':>12s}"
        s = f"{t_unc[t] / t_cac[t]:5.0f}x" if t in t_unc else ''
        print(f"  context {t:6d}  without {u}  with {t_cac[t] * 1e3:6.2f} ms  {s}", flush=True)

    if not args.quick:
        print("\npart 3: writing 512 tokens from scratch, greedy")
        cfg2k = dict(cfg, T=2048)
        m2k = init_gpt(cfg2k, np.random.default_rng(0))
        s = time.perf_counter(); a = greedy(m2k, cfg2k, 512, cached=False); tu = time.perf_counter() - s
        s = time.perf_counter(); b = greedy(m2k, cfg2k, 512, cached=True); tc = time.perf_counter() - s
        print(f"  {tu:.1f} s without cache, {tc:.2f} s with ({tu / tc:.0f}x), same tokens: {a == b}")

    print("\npart 4: memory")
    mem_unc, mem_cache = {}, {}
    for t in CTX:
        mem_unc[t] = peak_mb(lambda: forward(model, ids[:, :t], cfg_long))
        mem_cache[t] = 2 * cfg['L'] * t * C * 8 / 1e6
    model_mb = sum(v.size for v in trained.values()) * 8 / 1e6
    for t in CTX:
        print(f"  context {t:5d}  uncached step peak {mem_unc[t]:7.1f} MB   cache {mem_cache[t]:6.2f} MB")
    print(f"  weights {model_mb:.2f} MB, the cache passes that at {model_mb * 1e6 / (2 * cfg['L'] * C * 8):.0f} tokens")

    plot(t_unc, t_cac, mem_unc, mem_cache, model_mb, C)
    print(f"\nsaved {OUT}  [{time.perf_counter() - t0:.0f}s]")

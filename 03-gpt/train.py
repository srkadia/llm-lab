"""
train the gpt from gpt.py on the sherlock holmes book, with a hand-written adamw.

python train.py               full run, 5000 steps, ~3 h on a laptop cpu
python train.py --steps 600   the short run from notebook 2, ~20 min
saves the checkpoint with the best val loss to checkpoints/gpt.npz, and the loss log next to it
"""

import gc
import json
import math
import time
import argparse
from pathlib import Path

import numpy as np

from gpt import CONFIG, Tensor, load_data, get_batch, baselines, init_gpt, loss_fn, save_checkpoint

OUT = Path(__file__).parent / 'checkpoints'


class AdamW:

    def __init__(self, P, betas=(0.9, 0.99), eps=1e-8, weight_decay=0.1):
        # beta2 0.99 not 0.999, small batches like nanogpt's char models
        self.P = P
        self.b1, self.b2 = betas
        self.eps, self.wd = eps, weight_decay
        self.m = {k: np.zeros_like(p.data) for k, p in P.items()}
        self.v = {k: np.zeros_like(p.data) for k, p in P.items()}
        self.t = 0

    def step(self, lr):
        self.t += 1
        for k, p in self.P.items():
            g = p.grad
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * g * g
            # bias correction, m and v start at 0 so they're too small for the first steps
            m_hat = self.m[k] / (1 - self.b1**self.t)
            v_hat = self.v[k] / (1 - self.b2**self.t)
            if p.data.ndim >= 2: # decay the matrices, not biases / layernorm gains
                p.data -= lr * self.wd * p.data # decoupled: straight on the weight, not through the gradient
            p.data -= lr * m_hat / (np.sqrt(v_hat) + self.eps)


def lr_at(step, max_steps, lr_max, warmup, lr_min_frac=0.1):
    # linear warmup, then cosine down to 10% of the peak.
    # the warmup is not optional, see run.py: without it adam's first steps make every position look the same
    if step < warmup:
        return lr_max * (step + 1) / warmup
    progress = (step - warmup) / max(1, max_steps - warmup)
    lr_min = lr_max * lr_min_frac
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * progress))


def clip_grads(P, max_norm):
    # global norm over all params. scales the step, never changes its direction
    norm = math.sqrt(sum((p.grad**2).sum() for p in P.values()))
    if max_norm is not None and norm > max_norm:
        for p in P.values():
            p.grad *= max_norm / norm
    return norm


def train_step(P, opt, data, cfg, rng, lr, B=32, clip=1.0):
    # in its own function so the graph is garbage as soon as it returns.
    # every node's _backward closure points back at the node, so the graph is a reference cycle and only
    # gc frees it. gc triggers on object counts, not bytes, and a step is ~2000 objects holding ~3 GB.
    # without the gc.collect() after this it ran out of memory in a few steps
    for p in P.values():
        p.grad = np.zeros_like(p.data)
    loss = loss_fn(P, *get_batch(data, B, cfg['T'], rng), cfg)
    loss.backward()
    norm = clip_grads(P, clip)
    opt.step(lr)
    return float(loss.data), norm


def evaluate(P, data, cfg, n_batches=10, B=32, seed=123):
    # same 10 batches every time (and the same ones as notebook 2) so the numbers are comparable
    rng = np.random.default_rng(seed)
    losses = []
    for _ in range(n_batches):
        losses.append(float(loss_fn(P, *get_batch(data, B, cfg['T'], rng), cfg).data))
        gc.collect()
    return float(np.mean(losses))


def train(steps, lr_max=3e-3, warmup=100, B=32, eval_every=250, seed=0, out=OUT):
    tok, train_data, val_data = load_data()
    cfg = dict(CONFIG, V=len(tok))
    base = baselines(train_data, val_data, cfg['V'])
    P = {k: Tensor(v) for k, v in init_gpt(cfg, np.random.default_rng(seed)).items()}
    opt = AdamW(P)
    rng = np.random.default_rng(seed + 1)
    n_params = sum(p.data.size for p in P.values())
    print(f"{n_params:,} params, {len(train_data):,} train / {len(val_data):,} val chars")
    print(f"to beat: unigram {base['unigram']:.3f}, bigram {base['bigram']:.3f} (val)")

    log = {'cfg': cfg, 'lr_max': lr_max, 'warmup': warmup, 'B': B, 'baselines': base,
           'loss': [], 'norm': [], 'val_step': [], 'val': []}
    best = float('inf')
    t0 = time.perf_counter()
    for step in range(steps + 1):
        if step % eval_every == 0 or step == steps:
            val = evaluate(P, val_data, cfg)
            log['val_step'].append(step)
            log['val'].append(val)
            if val < best:
                best = val
                save_checkpoint(out / 'gpt.npz', {k: p.data for k, p in P.items()}, cfg, tok, step=step, val=val)
            el = time.perf_counter() - t0
            eta = el / step * (steps - step) if step else 0
            recent = f"{np.mean(log['loss'][-eval_every:]):.3f}" if log['loss'] else '  -  '
            print(f"step {step:5d} | train {recent} | val {val:.3f}{' *' if val == best else '  '} | "
                  f"{el / 60:5.1f} min, ~{eta / 60:.0f} min left", flush=True)
            (out / 'log.json').write_text(json.dumps(log))
        if step == steps:
            break
        loss, norm = train_step(P, opt, train_data, cfg, rng, lr_at(step, steps, lr_max, warmup), B)
        gc.collect()
        log['loss'].append(loss)
        log['norm'].append(norm)
    print(f"best val {best:.3f}, saved to {out / 'gpt.npz'}")
    return log


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=5000)
    parser.add_argument('--lr', type=float, default=3e-3)
    parser.add_argument('--warmup', type=int, default=100)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--eval_every', type=int, default=250)
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    train(args.steps, args.lr, args.warmup, args.batch, args.eval_every)

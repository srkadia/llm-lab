"""
checks for gpt.py, train.py and sample.py
python test_gpt.py   (pytest works too), ~15 s
"""

import sys
import tempfile
from pathlib import Path

import numpy as np

from gpt import (CONFIG, Tensor, load_book, CharTokenizer, load_data, get_batch, baselines, init_gpt, gpt,
                 loss_fn, next_probs, save_checkpoint, load_checkpoint)
from train import AdamW, lr_at, clip_grads
from gradcheck import gradcheck, TOL

tok, train_data, val_data = load_data()
V = len(tok)
TINY = dict(V=V, T=6, C=8, L=2, H=2)


def test_tokenizer_round_trip():
    text = load_book()
    assert V == 88
    assert tok.decode(tok.encode(text)) == text

def test_batches_are_shifted_by_one():
    x, y = get_batch(train_data, 4, 16, np.random.default_rng(0))
    assert x.shape == y.shape == (4, 16)
    assert (x[:, 1:] == y[:, :-1]).all()

def test_loss_at_init_is_close_to_uniform():
    # a model that knows nothing should be near ln(V), much higher means the init is confidently wrong
    cfg = dict(CONFIG, V=V)
    P = {k: Tensor(v) for k, v in init_gpt(cfg, np.random.default_rng(0)).items()}
    loss = loss_fn(P, *get_batch(train_data, 4, cfg['T'], np.random.default_rng(1)), cfg).data
    assert abs(loss - np.log(V)) < 0.15, loss

def test_gpt_grads():
    # the whole model, every parameter, wte with both of its paths (lookup and tied output).
    # embeddings at std 1 so the grads reaching the blocks aren't tiny (with 0.02 rounding noise dominates)
    p = init_gpt(TINY, np.random.default_rng(2))
    p['wte'] /= 0.02
    p['wpe'] /= 0.02
    names = list(p)
    x, y = get_batch(train_data, 2, TINY['T'], np.random.default_rng(3))
    err = gradcheck(lambda ts: loss_fn(dict(zip(names, ts)), x, y, TINY), [p[k] for k in names])
    assert err < TOL, f"{err:.1e}"

def test_no_future_leak():
    cfg = dict(CONFIG, V=V)
    P = {k: Tensor(v) for k, v in init_gpt(cfg, np.random.default_rng(0)).items()}
    x1, _ = get_batch(val_data, 2, 64, np.random.default_rng(4))
    x2 = x1.copy()
    x2[:, -1] = (x2[:, -1] + 1) % V
    l1, l2 = gpt(P, x1, cfg).data, gpt(P, x2, cfg).data
    assert np.abs(l1[:, :-1] - l2[:, :-1]).max() == 0.0 # exactly, not close to
    assert np.abs(l1[:, -1] - l2[:, -1]).max() > 1e-3

def test_overfits_one_batch():
    # if the model can't memorize 128 chars something between the loss and the update is broken
    cfg = dict(V=V, T=32, C=64, L=2, H=4)
    P = {k: Tensor(v) for k, v in init_gpt(cfg, np.random.default_rng(5)).items()}
    x, y = get_batch(train_data, 4, 32, np.random.default_rng(6))
    for _ in range(100):
        for p in P.values():
            p.grad = np.zeros_like(p.data)
        loss = loss_fn(P, x, y, cfg)
        loss.backward()
        for p in P.values():
            p.data -= 0.3 * p.grad
    assert loss.data < 0.1, loss.data

def test_baselines():
    b = baselines(train_data, val_data, V)
    assert abs(b['unigram'] - 3.103) < 1e-3 and abs(b['bigram'] - 2.433) < 1e-3, b

def test_adamw_first_step_is_lr():
    # bias correction makes step 1 exactly lr * sign(g), whatever the size of g
    rng = np.random.default_rng(0)
    P = {'big': Tensor(rng.standard_normal((3, 4))), 'tiny': Tensor(rng.standard_normal(5))}
    P['big'].grad = rng.standard_normal((3, 4)) * 1e3
    P['tiny'].grad = rng.standard_normal(5) * 1e-4
    before = {k: p.data.copy() for k, p in P.items()}
    AdamW(P, weight_decay=0.0).step(0.01)
    for k, p in P.items():
        assert np.allclose(np.abs(p.data - before[k]), 0.01, rtol=1e-3), k

def test_weight_decay_is_decoupled():
    # with no loss grad every weight should shrink by exactly (1 - lr*wd) per step, big or small.
    # matrices only, biases don't decay
    w0 = np.array([[10.0, 1.0, 0.01]])
    P = {'w': Tensor(w0.copy()), 'b': Tensor(np.array([10.0, 1.0]))}
    opt = AdamW(P, weight_decay=0.1)
    for _ in range(100):
        for p in P.values():
            p.grad = np.zeros_like(p.data)
        opt.step(0.01)
    assert np.allclose(P['w'].data / w0, 0.999**100, rtol=1e-12)
    assert np.array_equal(P['b'].data, [10.0, 1.0])

def test_lr_schedule():
    assert np.isclose(lr_at(0, 1000, 1e-3, 100), 1e-5)
    assert np.isclose(lr_at(99, 1000, 1e-3, 100), 1e-3) # end of warmup
    assert np.isclose(lr_at(1000, 1000, 1e-3, 100), 1e-4) # cosine ends at 10%
    lrs = [lr_at(s, 1000, 1e-3, 100) for s in range(100, 1001)]
    assert all(a >= b for a, b in zip(lrs, lrs[1:])) # never goes back up

def test_clipping_keeps_direction():
    rng = np.random.default_rng(0)
    P = {'a': Tensor(np.zeros((4, 5))), 'b': Tensor(np.zeros(7))}
    for p in P.values():
        p.grad = rng.standard_normal(p.shape) * 10
    before = np.concatenate([p.grad.ravel() for p in P.values()])
    norm = clip_grads(P, 1.0)
    after = np.concatenate([p.grad.ravel() for p in P.values()])
    assert norm > 1 and np.isclose(np.linalg.norm(after), 1.0)
    assert np.isclose(before @ after / (np.linalg.norm(before) * np.linalg.norm(after)), 1.0, atol=1e-12)

def test_sampling_filters():
    logits = np.array([2.0, 1.0, 0.5, 0.0, -1.0, -3.0])
    p = next_probs(logits)
    assert np.array_equal(next_probs(logits, temperature=0), np.eye(6)[0])
    k3 = next_probs(logits, top_k=3)
    assert (k3 > 0).sum() == 3 and np.allclose(k3[:3] / k3[0], p[:3] / p[0]) # ratios kept
    # sorted mass 0.561, 0.767, 0.892, 0.968: first to reach 0.9 is the 4th token
    assert (next_probs(logits, top_p=0.9) > 0).sum() == 4
    assert (next_probs(logits, top_p=0.5) > 0).sum() == 1
    assert next_probs(logits, temperature=0.5)[0] > p[0] > next_probs(logits, temperature=2.0)[0]

def test_checkpoint_round_trip():
    cfg = dict(TINY)
    p = init_gpt(cfg, np.random.default_rng(0))
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / 'ck.npz'
        save_checkpoint(path, p, cfg, tok, step=7, val=1.5)
        p2, cfg2, tok2, info = load_checkpoint(path)
    assert cfg2 == cfg and tok2.chars == tok.chars and info['step'] == 7
    x, _ = get_batch(val_data, 2, 6, np.random.default_rng(0))
    a = gpt({k: Tensor(v) for k, v in p.items()}, x, cfg).data
    b = gpt({k: Tensor(v) for k, v in p2.items()}, x, cfg2).data
    assert np.array_equal(a, b)


if __name__ == '__main__':
    tests = [v for k, v in list(globals().items()) if k.startswith('test_')]
    fails = 0
    for t in tests:
        try:
            t()
            print(f"{t.__name__:38s} ok")
        except AssertionError as e:
            fails += 1
            print(f"{t.__name__:38s} FAIL {e}")
    print(f"\n{len(tests) - fails}/{len(tests)} passed")
    sys.exit(1 if fails else 0)

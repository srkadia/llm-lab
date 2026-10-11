"""
checks for transformer.py
python test_transformer.py   (pytest works too)
"""

import sys
import math
import numpy as np

from transformer import (attention, causal_mask, multi_head_attention, layer_norm, gelu, mlp,
                         sinusoidal, rope_tables, rope, init_block, block)
# transformer.py already put 01-autograd on the path
from autograd import Tensor
from gradcheck import gradcheck, TOL

rng = np.random.default_rng(0)

def rand(*shape, scale=1.0):
    return rng.standard_normal(shape) * scale

def softmax_np(z):
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def test_attention_grads():
    q, k, v = rand(2, 5, 4), rand(2, 5, 4), rand(2, 5, 4)
    for causal in [True, False]:
        err = gradcheck(lambda t: attention(t[0], t[1], t[2], causal=causal), [q, k, v])
        assert err < TOL, f"causal={causal}: {err:.1e}"

def test_masked_positions_get_zero_grad():
    # output at position 0 can only depend on k/v at position 0
    q, k, v = Tensor(rand(2, 5, 4)), Tensor(rand(2, 5, 4)), Tensor(rand(2, 5, 4))
    out = attention(q, k, v)
    (out * np.eye(5)[0][None, :, None]).sum().backward()
    assert np.abs(k.grad[:, 1:]).max() == 0.0 and np.abs(v.grad[:, 1:]).max() == 0.0

def test_multi_head_matches_loop():
    # reference: explicit loops over batch and heads, no reshapes
    B, T, C, H = 2, 6, 16, 4
    x = rand(B, T, C)
    W = [rand(C, C, scale=0.3) for _ in range(4)]
    hs = C // H
    slow = np.zeros((B, T, C))
    for b in range(B):
        Q, K, V = x[b] @ W[0], x[b] @ W[1], x[b] @ W[2]
        heads = []
        for h in range(H):
            c = slice(h * hs, (h + 1) * hs)
            w = softmax_np(Q[:, c] @ K[:, c].T / math.sqrt(hs) + causal_mask(T))
            heads.append(w @ V[:, c])
        slow[b] = np.concatenate(heads, axis=1) @ W[3]
    fast = multi_head_attention(Tensor(x), *[Tensor(w) for w in W], n_head=H).data
    assert np.abs(fast - slow).max() < 1e-12

def test_multi_head_grads():
    inputs = [rand(2, 4, 8)] + [rand(8, 8, scale=0.3) for _ in range(4)]
    err = gradcheck(lambda t: multi_head_attention(*t, n_head=2), inputs)
    assert err < TOL, f"{err:.1e}"

def test_multi_head_with_rope_grads():
    tables = rope_tables(16, 4) # head size 4, tables longer than T on purpose
    inputs = [rand(2, 5, 8)] + [rand(8, 8, scale=0.3) for _ in range(4)]
    err = gradcheck(lambda t: multi_head_attention(*t, n_head=2, rope_tables=tables), inputs)
    assert err < TOL, f"{err:.1e}"

def test_layer_norm_stats():
    x = rand(2, 5, 8) * 3 + 2
    out = layer_norm(Tensor(x), np.ones(8), np.zeros(8)).data
    assert np.abs(out.mean(-1)).max() < 1e-12
    assert np.abs(out.var(-1) - 1).max() < 1e-4 # not exactly 1 because of eps

def test_layer_norm_grads():
    err = gradcheck(lambda t: layer_norm(*t), [rand(2, 5, 8) * 3 + 2, rand(8), rand(8)])
    assert err < TOL, f"{err:.1e}"

def test_layer_norm_closed_form():
    # the textbook layernorm backward. the engine never sees it, it only knows mean/sub/pow/mul
    x_np, g_np, b_np, R = rand(2, 5, 8) * 3 + 2, rand(8), rand(8), rand(2, 5, 8)
    x = Tensor(x_np)
    (layer_norm(x, g_np, b_np) * R).sum().backward()
    N = x_np.shape[-1]
    r = 1 / np.sqrt(x_np.var(-1, keepdims=True) + 1e-5)
    xhat = (x_np - x_np.mean(-1, keepdims=True)) * r
    g = R * g_np
    expected = r / N * (N * g - g.sum(-1, keepdims=True) - xhat * (g * xhat).sum(-1, keepdims=True))
    assert np.abs(x.grad - expected).max() < 1e-12

def test_gelu():
    z = np.linspace(-6, 6, 2001)
    exact = z * 0.5 * (1 + np.vectorize(math.erf)(z / math.sqrt(2)))
    assert np.abs(gelu(Tensor(z)).data - exact).max() < 1e-3
    err = gradcheck(lambda t: gelu(t[0]), [rand(4, 6)])
    assert err < TOL, f"{err:.1e}"

def test_mlp_grads():
    inputs = [rand(2, 3, 8), rand(8, 32, scale=0.3), np.zeros(32), rand(32, 8, scale=0.3), np.zeros(8)]
    err = gradcheck(lambda t: mlp(*t), inputs)
    assert err < TOL, f"{err:.1e}"

def test_block_grads():
    # every parameter of a full block at once. nudge the layernorm params off 1/0 first
    p = init_block(8, rng)
    for k in ['ln1_g', 'ln1_b', 'ln2_g', 'ln2_b']:
        p[k] = p[k] + rand(8, scale=0.1)
    names = list(p)
    f = lambda t: block(t[0], dict(zip(names, t[1:])), n_head=2)
    err = gradcheck(f, [rand(1, 4, 8)] + [p[k] for k in names])
    assert err < TOL, f"{err:.1e}"

def test_block_no_future_leak():
    # change token 3, positions 0..2 must not move at all (exactly, not approximately)
    p = {k: Tensor(v) for k, v in init_block(8, rng).items()}
    x = rand(1, 6, 8)
    x2 = x.copy()
    x2[0, 3] += rand(8)
    diff = np.abs(block(Tensor(x), p, 2).data - block(Tensor(x2), p, 2).data).max(-1)[0]
    assert np.all(diff[:3] == 0.0), diff
    assert np.all(diff[3:] > 0)

def test_attention_is_blind_to_order():
    # without a mask or positions, shuffling the tokens just shuffles the output
    W = [Tensor(rand(8, 8, scale=8**-0.5)) for _ in range(4)]
    x = rand(1, 5, 8)
    perm = rng.permutation(5)
    out = multi_head_attention(Tensor(x), *W, n_head=2, causal=False).data
    out_shuf = multi_head_attention(Tensor(x[:, perm]), *W, n_head=2, causal=False).data
    assert np.abs(out_shuf - out[:, perm]).max() < 1e-12
    # with sinusoidal positions added it isn't anymore
    pe = sinusoidal(5, 8)
    out = multi_head_attention(Tensor(x + pe), *W, n_head=2, causal=False).data
    out_shuf = multi_head_attention(Tensor(x[:, perm] + pe), *W, n_head=2, causal=False).data
    assert np.abs(out_shuf - out[:, perm]).max() > 1e-2

def test_rope_depends_only_on_distance():
    cos, sin, S = rope_tables(64, 8)
    q, k = rand(8), rand(8)
    rot = lambda v, pos: rope(Tensor(v[None]), cos[pos:pos + 1], sin[pos:pos + 1], S).data[0]
    scores = [rot(q, m) @ rot(k, m - 3) for m in [3, 10, 25, 60]]
    assert np.ptp(scores) < 1e-12, scores
    assert abs((rot(q, 10) @ rot(k, 10)) - (rot(q, 10) @ rot(k, 5))) > 1e-3 # different distance, different score
    assert abs(np.linalg.norm(rot(q, 40)) - np.linalg.norm(q)) < 1e-12 # it's a rotation


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

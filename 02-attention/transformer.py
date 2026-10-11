"""
attention and the transformer block, built on the autograd engine in 01-autograd.
everything here is made of the engine's existing ops, so there's no new backward code.
"""

import sys
import math
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / '01-autograd'))
from autograd import Tensor, softmax


def swap_last2(x):
    # the engine's transpose wants the full axis order, this only swaps the last two
    n = x.data.ndim
    return x.transpose(tuple(range(n - 2)) + (n - 1, n - 2))


def causal_mask(T):
    # -inf above the diagonal so token i can't look at anything after it.
    # exp(-inf) = 0, so those weights and their grads come out exactly 0
    return np.triu(np.full((T, T), -np.inf), k=1)


def attention(q, k, v, causal=True):
    # q, k, v: (..., T, d)
    T, d = q.shape[-2], q.shape[-1]
    att = (q @ swap_last2(k)) * (1.0 / math.sqrt(d)) # without the sqrt(d) softmax saturates on big heads, see run.py
    if causal:
        att = att + causal_mask(T)
    return softmax(att, axis=-1) @ v


def split_heads(x, n_head):
    B, T, C = x.shape
    return x.reshape(B, T, n_head, C // n_head).transpose((0, 2, 1, 3)) # (B, nh, T, hs)


def merge_heads(x):
    B, nh, T, hs = x.shape
    return x.transpose((0, 2, 1, 3)).reshape(B, T, nh * hs)


def multi_head_attention(x, Wq, Wk, Wv, Wo, n_head, causal=True, rope_tables=None):
    q = split_heads(x @ Wq, n_head)
    k = split_heads(x @ Wk, n_head)
    v = split_heads(x @ Wv, n_head)
    if rope_tables is not None:
        q, k = rope(q, *rope_tables), rope(k, *rope_tables)
    # no loop over heads, batched matmul treats (B, nh) as batch dims
    return merge_heads(attention(q, k, v, causal)) @ Wo


def layer_norm(x, gamma, beta, eps=1e-5):
    # per token, over the last dim. the sqrt is just ** -0.5
    mu = x.mean(axis=-1, keepdims=True)
    xc = x - mu
    var = (xc**2).mean(axis=-1, keepdims=True)
    return xc * (var + eps)**-0.5 * gamma + beta


def gelu(x):
    # tanh approximation, same as gpt-2. within ~5e-4 of the exact erf version
    return 0.5 * x * (1 + (math.sqrt(2 / math.pi) * (x + 0.044715 * x**3)).tanh())


def mlp(x, W1, b1, W2, b2):
    return gelu(x @ W1 + b1) @ W2 + b2


def sinusoidal(T, C):
    # fixed position pattern from the original transformer paper, added to the token vectors
    pos = np.arange(T)[:, None]
    freq = 1.0 / 10000**(np.arange(0, C, 2) / C)
    pe = np.zeros((T, C))
    pe[:, 0::2] = np.sin(pos * freq)
    pe[:, 1::2] = np.cos(pos * freq)
    return pe


def pair_swap(d):
    # x @ S turns every pair (x0, x1) into (-x1, x0). as a matmul it needs no new op.
    # pairs neighbouring dims like roformer / gpt-j. llama's rotate_half pairs i with i + d/2
    # instead, same rotation with the dims laid out differently, so don't mix the two
    S = np.zeros((d, d))
    for i in range(0, d, 2):
        S[i + 1, i] = -1
        S[i, i + 1] = 1
    return S


def rope_tables(T_max, d, base=10000):
    # every pair of dims rotates at its own frequency, angle grows with position
    assert d % 2 == 0, "rope needs an even head size"
    theta = base**(-np.arange(0, d, 2) / d)
    ang = np.arange(T_max)[:, None] * theta[None, :]
    cos = np.repeat(np.cos(ang), 2, axis=1)
    sin = np.repeat(np.sin(ang), 2, axis=1)
    return cos, sin, pair_swap(d)


def rope(x, cos, sin, S):
    # x: (..., T, d). rotating q and k by position makes q.k depend only on the distance
    # between the two tokens. tables can be longer than T, we just take the first T rows
    T = x.shape[-2]
    return x * cos[:T] + (x @ S) * sin[:T]


def init_block(C, rng, mlp_mult=4):
    # plain numpy arrays, wrap them in Tensor to train
    s = C**-0.5
    return {
        'ln1_g': np.ones(C), 'ln1_b': np.zeros(C),
        'Wq': rng.standard_normal((C, C)) * s,
        'Wk': rng.standard_normal((C, C)) * s,
        'Wv': rng.standard_normal((C, C)) * s,
        'Wo': rng.standard_normal((C, C)) * s,
        'ln2_g': np.ones(C), 'ln2_b': np.zeros(C),
        'W1': rng.standard_normal((C, mlp_mult * C)) * s,
        'b1': np.zeros(mlp_mult * C),
        'W2': rng.standard_normal((mlp_mult * C, C)) * (mlp_mult * C)**-0.5,
        'b2': np.zeros(C),
    }


def block(x, p, n_head, rope_tables=None):
    # pre-LN like gpt-2: the residual path itself never goes through a layernorm,
    # so the gradient has a clean path all the way down
    h = layer_norm(x, p['ln1_g'], p['ln1_b'])
    x = x + multi_head_attention(h, p['Wq'], p['Wk'], p['Wv'], p['Wo'], n_head, rope_tables=rope_tables)
    h = layer_norm(x, p['ln2_g'], p['ln2_b'])
    x = x + mlp(h, p['W1'], p['b1'], p['W2'], p['b2'])
    return x

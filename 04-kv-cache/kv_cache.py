"""
kv cache for the gpt from 03. inference only, so it's plain numpy with no autograd graph.
forward() runs the whole sequence every time, forward_cached() only runs the new tokens.
"""

import sys
import math
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / '03-gpt'))
from gpt import load_checkpoint, next_probs # also puts 01 and 02 on the path

CKPT = Path(__file__).resolve().parent.parent / '03-gpt' / 'checkpoints' / 'gpt.npz'


def layer_norm(x, g, b, eps=1e-5):
    xc = x - x.mean(-1, keepdims=True)
    return xc * ((xc**2).mean(-1, keepdims=True) + eps)**-0.5 * g + b


def gelu(x):
    return 0.5 * x * (1 + np.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x**3)))


def softmax(z):
    e = np.exp(z - z.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def heads(a, H):
    # (B, T, C) -> (B, H, T, C/H)
    B, T, C = a.shape
    return a.reshape(B, T, H, C // H).transpose(0, 2, 1, 3)


def mlp_and_rest(x, p, i, att, cfg):
    # everything in block i after attention: att is the weighted values (B, H, T, C/H). output
    # projection, then the mlp. shared by both forwards so they can't drift apart
    B, T, C = x.shape
    g = lambda k: p[f'h{i}.{k}']
    x = x + att.transpose(0, 2, 1, 3).reshape(B, T, C) @ g('Wo')
    h = layer_norm(x, g('ln2_g'), g('ln2_b'))
    return x + gelu(h @ g('W1') + g('b1')) @ g('W2') + g('b2')


def forward(p, idx, cfg, rows=None):
    # idx (B, T) -> logits (B, T, V). the whole sequence again, what 03's sample.py does.
    # rows: do attention that many query rows at a time. same logits, but it never holds all
    # T x T scores at once (581 MB at T = 2048 in float64, see run.py)
    B, T = idx.shape
    H = cfg['H']
    x = p['wte'][idx] + p['wpe'][:T]
    rows = rows or T
    for i in range(cfg['L']):
        g = lambda k: p[f'h{i}.{k}']
        h = layer_norm(x, g('ln1_g'), g('ln1_b'))
        q, k, v = heads(h @ g('Wq'), H), heads(h @ g('Wk'), H), heads(h @ g('Wv'), H)
        att = np.empty_like(q)
        for s in range(0, T, rows):
            e = min(s + rows, T)
            mask = np.triu(np.full((e - s, e), -np.inf), k=s + 1) # query row r sees keys 0..r
            a = softmax(q[:, :, s:e] @ k[:, :, :e].transpose(0, 1, 3, 2) / math.sqrt(cfg['C'] // H) + mask)
            att[:, :, s:e] = a @ v[:, :, :e]
        x = mlp_and_rest(x, p, i, att, cfg)
    return layer_norm(x, p['lnf_g'], p['lnf_b']) @ p['wte'].T


def new_cache(cfg, B=1):
    # keys and values for every layer, allocated once for the whole context, plus how much is filled
    shape = (cfg['L'], B, cfg['H'], cfg['T'], cfg['C'] // cfg['H'])
    return {'k': np.zeros(shape), 'v': np.zeros(shape), 'n': 0}


def forward_cached(p, idx, cache, cfg):
    # idx (B, t) holds only the NEW tokens. their keys/values go into the cache, and they attend
    # over everything cached so far. works for a whole prompt (prefill) or one token (decode)
    B, t = idx.shape
    n = cache['n']
    H = cfg['H']
    assert n + t <= cfg['T'], "cache is full, the model only has T positions (see generate)"
    x = p['wte'][idx] + p['wpe'][n:n + t]
    mask = np.triu(np.full((t, n + t), -np.inf), k=n + 1) # new token j sees positions 0..n+j
    for i in range(cfg['L']):
        g = lambda k: p[f'h{i}.{k}']
        h = layer_norm(x, g('ln1_g'), g('ln1_b'))
        q = heads(h @ g('Wq'), H)
        cache['k'][i, :, :, n:n + t] = heads(h @ g('Wk'), H)
        cache['v'][i, :, :, n:n + t] = heads(h @ g('Wv'), H)
        k, v = cache['k'][i, :, :, :n + t], cache['v'][i, :, :, :n + t]
        a = softmax(q @ k.transpose(0, 1, 3, 2) / math.sqrt(cfg['C'] // H) + mask)
        x = mlp_and_rest(x, p, i, a @ v, cfg)
    cache['n'] = n + t
    return layer_norm(x, p['lnf_g'], p['lnf_b']) @ p['wte'].T


def prefill(p, ids, cache, cfg, chunk=256):
    # in chunks: in one go a 16k prompt builds a 16k x 16k score matrix per head (~8.6 GB in float64)
    for s in range(0, ids.shape[1], chunk):
        logits = forward_cached(p, ids[:, s:s + chunk], cache, cfg)
    return logits


def generate(p, cfg, tok, prompt, n, rng, cached=True, **kw):
    # same loop as 03's generate(), so with the same seed it writes the same text
    T = cfg['T']
    ids = list(tok.encode(prompt))
    if cached:
        cache = new_cache(cfg)
        logits = prefill(p, np.array(ids[-T:])[None], cache, cfg)[0, -1]
    for _ in range(n):
        if not cached:
            logits = forward(p, np.array(ids[-T:])[None], cfg)[0, -1]
        ids.append(rng.choice(len(logits), p=next_probs(logits, **kw)))
        if cached:
            if cache['n'] < T:
                logits = forward_cached(p, np.array([[ids[-1]]]), cache, cfg)[0, -1]
            else:
                # full. with learned positions every cached key knows its own position, so the
                # cache can't slide (notebook 1, step 7). from here every step recomputes the
                # last T tokens, exactly like 03
                cached = False
    return tok.decode(ids)

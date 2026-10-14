"""
a small character-level gpt, built on the engine in 01-autograd and the block in 02-attention.
data, tokenizer, model, sampling and checkpoints. the training loop is in train.py
"""

import gc
import sys
import json
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / '02-attention'))
from transformer import block, init_block, layer_norm # also puts 01-autograd on the path
from autograd import Tensor, cross_entropy, embedding

DATA = Path(__file__).parent / 'data' / 'sherlock_raw.txt'
URL = 'https://www.gutenberg.org/cache/epub/1661/pg1661.txt' # the adventures of sherlock holmes

# V comes from the tokenizer. ~0.82M params with the 88 chars of the book
CONFIG = dict(T=128, C=128, L=4, H=4)

# -----------------------------------------------------------------------------
# data

def load_book(path=DATA):
    if not path.exists():
        path.parent.mkdir(exist_ok=True)
        urllib.request.urlretrieve(URL, path)
    raw = path.read_text(encoding='utf-8-sig').replace('\r\n', '\n')
    # cut gutenberg's licence header and footer
    start = raw.find('*** START OF THE PROJECT GUTENBERG EBOOK')
    end = raw.find('*** END OF THE PROJECT GUTENBERG EBOOK')
    return raw[raw.find('\n', start) + 1:end].strip()


class CharTokenizer:
    # one token per distinct character. bpe is a later module

    def __init__(self, chars):
        self.chars = sorted(chars)
        self.stoi = {c: i for i, c in enumerate(self.chars)}

    def __len__(self):
        return len(self.chars)

    def encode(self, s):
        return np.array([self.stoi[c] for c in s], dtype=np.int64)

    def decode(self, ids):
        return ''.join(self.chars[int(i)] for i in ids)


def load_data(split=0.9):
    # one cut, not random chunks: a random split puts the same page in train and val
    text = load_book()
    tok = CharTokenizer(set(text))
    data = tok.encode(text)
    n = int(split * len(data))
    return tok, data[:n], data[n:]


def get_batch(data, B, T, rng):
    # B random windows, targets are the same windows shifted by one
    ix = rng.integers(0, len(data) - T, B)
    x = np.stack([data[i:i + T] for i in ix])
    y = np.stack([data[i + 1:i + T + 1] for i in ix])
    return x, y


def baselines(train, val, V):
    # val loss of the two simplest models. a gpt that doesn't clearly beat the bigram isn't using its context
    p1 = np.bincount(train, minlength=V) + 1.0
    p1 /= p1.sum()
    counts = np.ones((V, V))
    np.add.at(counts, (train[:-1], train[1:]), 1) # add.at, pairs repeat
    p2 = counts / counts.sum(1, keepdims=True)
    return {'unigram': float(-np.log(p1[val]).mean()), 'bigram': float(-np.log(p2[val[:-1], val[1:]]).mean())}

# -----------------------------------------------------------------------------
# model

def init_gpt(cfg, rng):
    # flat dict of numpy arrays, 'h0.Wq' etc for the blocks. wrap in Tensor to train
    V, T, C = cfg['V'], cfg['T'], cfg['C']
    p = {
        'wte': rng.standard_normal((V, C)) * 0.02, # small, it's also the output layer, so it sets the size of the logits
        'wpe': rng.standard_normal((T, C)) * 0.02,
        'lnf_g': np.ones(C), 'lnf_b': np.zeros(C),
    }
    for i in range(cfg['L']):
        for k, v in init_block(C, rng).items():
            p[f'h{i}.{k}'] = v
    return p


def block_params(P, i):
    pre = f'h{i}.'
    return {k[len(pre):]: v for k, v in P.items() if k.startswith(pre)}


def gpt(P, idx, cfg):
    # idx (B, T) token ids -> logits (B, T, V)
    B, T = idx.shape
    assert T <= cfg['T'], f"sequence of {T} is longer than the position table ({cfg['T']})"
    x = embedding(P['wte'], idx) + embedding(P['wpe'], np.arange(T))
    for i in range(cfg['L']):
        x = block(x, block_params(P, i), cfg['H'])
    x = layer_norm(x, P['lnf_g'], P['lnf_b'])
    return x @ P['wte'].transpose((1, 0)) # tied: the output layer is the token embedding


def loss_fn(P, x, y, cfg):
    return cross_entropy(gpt(P, x, cfg).reshape(-1, cfg['V']), y.reshape(-1))

# -----------------------------------------------------------------------------
# sampling

def next_probs(logits, temperature=1.0, top_k=None, top_p=None):
    if temperature == 0: # greedy
        p = np.zeros_like(logits)
        p[np.argmax(logits)] = 1.0
        return p
    z = logits / temperature
    if top_k is not None:
        z = np.where(z < np.sort(z)[-top_k], -np.inf, z)
    p = np.exp(z - z.max())
    p /= p.sum()
    if top_p is not None:
        # smallest set of tokens that covers top_p of the mass
        order = np.argsort(-p)
        mass_before = np.cumsum(p[order]) - p[order]
        keep = order[mass_before < top_p]
        q = np.zeros_like(p)
        q[keep] = p[keep]
        p = q / q.sum()
    return p


def generate(P, cfg, tok, prompt, n, rng, **kw):
    # re-runs the whole context for every new char. the kv cache (04) fixes that
    ids = list(tok.encode(prompt))
    for i in range(n):
        ctx = np.array(ids[-cfg['T']:])[None]
        logits = gpt(P, ctx, cfg).data[0, -1]
        ids.append(rng.choice(len(logits), p=next_probs(logits, **kw)))
        if i % 50 == 49:
            gc.collect() # every forward leaves a graph behind, see train.py
    return tok.decode(ids)

# -----------------------------------------------------------------------------
# checkpoints

def save_checkpoint(path, params, cfg, tok, **meta):
    path = Path(path)
    path.parent.mkdir(exist_ok=True)
    info = json.dumps({'cfg': cfg, 'chars': tok.chars, **meta})
    np.savez(path, __info__=np.array(info), **params)


def load_checkpoint(path):
    f = np.load(path)
    info = json.loads(str(f['__info__']))
    params = {k: f[k] for k in f.files if k != '__info__'}
    return params, info['cfg'], CharTokenizer(info['chars']), info

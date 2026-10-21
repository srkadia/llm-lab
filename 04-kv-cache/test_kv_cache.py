"""
checks for kv_cache.py
python test_kv_cache.py   (pytest works too), ~20 s
"""

import sys

import numpy as np

from kv_cache import CKPT, load_checkpoint, forward, new_cache, forward_cached, prefill, generate
from gpt import Tensor, gpt, init_gpt, generate as generate_engine

p, cfg, tok, info = load_checkpoint(CKPT)
TEXT = "“I have no data yet. It is a capital mistake to theorise before one has data.”"
IDX = tok.encode(TEXT)[None]


def test_forward_matches_engine():
    # the numpy forward is the same model as 03's, which goes through the autograd engine
    engine = gpt({k: Tensor(v) for k, v in p.items()}, IDX, cfg).data
    assert np.abs(forward(p, IDX, cfg) - engine).max() < 1e-12

def test_blocked_attention_matches():
    # doing attention a few query rows at a time is the same model, just less memory
    full = forward(p, IDX, cfg)
    for rows in (1, 7, 32):
        assert np.abs(forward(p, IDX, cfg, rows=rows) - full).max() < 1e-12, rows

def test_appending_does_not_change_the_past():
    # this is what makes caching legal at all
    a = forward(p, IDX[:, :40], cfg)
    b = forward(p, IDX[:, :41], cfg)
    assert np.abs(a - b[:, :40]).max() < 1e-12

def test_prefill_matches_forward():
    cache = new_cache(cfg)
    out = forward_cached(p, IDX, cache, cfg)
    assert cache['n'] == IDX.shape[1]
    assert np.abs(out - forward(p, IDX, cfg)).max() < 1e-12

def test_one_token_at_a_time_matches_forward():
    cache = new_cache(cfg)
    out = np.concatenate([forward_cached(p, IDX[:, j:j + 1], cache, cfg) for j in range(IDX.shape[1])], axis=1)
    assert np.abs(out - forward(p, IDX, cfg)).max() < 1e-12

def test_chunked_prefill_matches_one_shot():
    a, b = new_cache(cfg), new_cache(cfg)
    la = forward_cached(p, IDX, a, cfg)
    lb = prefill(p, IDX, b, cfg, chunk=7)
    assert np.abs(la[:, -1] - lb[:, -1]).max() < 1e-12
    assert np.abs(a['k'] - b['k']).max() < 1e-12 and np.abs(a['v'] - b['v']).max() < 1e-12

def test_cached_generation_matches_uncached():
    # all the way to the 128 position limit, greedy and sampled
    for kw in ({'temperature': 0}, {'temperature': 0.8}):
        a = generate(p, cfg, tok, 'Holmes', 122, np.random.default_rng(0), cached=True, **kw)
        b = generate(p, cfg, tok, 'Holmes', 122, np.random.default_rng(0), cached=False, **kw)
        assert a == b, kw

def test_same_text_as_03_past_the_limit():
    # 200 chars goes past 128 positions, where the cache is full and generate() falls back to
    # recomputing the window. it should still write exactly what 03's engine-based sampler writes
    ours = generate(p, cfg, tok, 'Holmes', 200, np.random.default_rng(0), temperature=0.8)
    theirs = generate_engine({k: Tensor(v) for k, v in p.items()}, cfg, tok, 'Holmes', 200,
                             np.random.default_rng(0), temperature=0.8)
    assert ours == theirs

def test_long_context_random_weights():
    # the experiment uses random weights past 128, check the cache is still exact there
    cfg2k = dict(cfg, T=2048)
    m = init_gpt(cfg2k, np.random.default_rng(0))
    ids = np.random.default_rng(2).integers(0, cfg['V'], (1, 1500))
    cache = new_cache(cfg2k)
    prefill(m, ids[:, :-1], cache, cfg2k)
    last = forward_cached(m, ids[:, -1:], cache, cfg2k)[0, -1]
    assert np.abs(last - forward(m, ids, cfg2k)[0, -1]).max() < 1e-12

def test_full_cache_refuses():
    cache = new_cache(cfg)
    prefill(p, np.zeros((1, cfg['T']), dtype=np.int64), cache, cfg)
    try:
        forward_cached(p, np.zeros((1, 1), dtype=np.int64), cache, cfg)
    except AssertionError:
        return
    assert False, "wrote past the end of the cache"

def test_cache_size():
    c = new_cache(cfg)
    assert c['k'].nbytes + c['v'].nbytes == 2 * cfg['L'] * cfg['T'] * cfg['C'] * 8


if __name__ == '__main__':
    tests = [v for k, v in list(globals().items()) if k.startswith('test_')]
    fails = 0
    for t in tests:
        try:
            t()
            print(f"{t.__name__:42s} ok")
        except AssertionError as e:
            fails += 1
            print(f"{t.__name__:42s} FAIL {e}")
    print(f"\n{len(tests) - fails}/{len(tests)} passed")
    sys.exit(1 if fails else 0)

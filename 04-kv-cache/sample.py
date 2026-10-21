"""
03's sample.py with the kv cache. same model, same seed, same text, just faster.

python sample.py
python sample.py --no-cache          the old way, for comparison
python sample.py --prompt "Watson" --n 500 --temperature 1.0 --top_p 0.9
"""

import time
import argparse

import numpy as np

from kv_cache import CKPT, load_checkpoint, generate

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt', default='Holmes')
    parser.add_argument('--n', type=int, default=300, help='characters to generate')
    parser.add_argument('--temperature', type=float, default=0.8)
    parser.add_argument('--top_k', type=int, default=None)
    parser.add_argument('--top_p', type=float, default=None)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--no-cache', action='store_true')
    parser.add_argument('--ckpt', default=str(CKPT))
    args = parser.parse_args()

    p, cfg, tok, info = load_checkpoint(args.ckpt)
    print(f"checkpoint from step {info['step']}, val loss {info['val']:.3f}\n")
    t0 = time.perf_counter()
    text = generate(p, cfg, tok, args.prompt, args.n, np.random.default_rng(args.seed), cached=not args.no_cache,
                    temperature=args.temperature, top_k=args.top_k, top_p=args.top_p)
    dt = time.perf_counter() - t0
    print(text)
    print(f"\n{args.n} characters in {dt:.2f} s ({'no cache' if args.no_cache else 'kv cache'})")
    if args.n + len(args.prompt) > cfg['T'] and not args.no_cache:
        print(f"(only the first {cfg['T']} positions use the cache, after that every step recomputes the window)")

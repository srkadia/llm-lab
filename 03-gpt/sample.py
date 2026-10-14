"""
generate text from the checkpoint train.py saved.

python sample.py
python sample.py --prompt "Watson" --n 500 --temperature 1.0 --top_p 0.9
python sample.py --temperature 0      greedy, gets stuck in a loop
"""

import argparse
from pathlib import Path

import numpy as np

from gpt import Tensor, load_checkpoint, generate

CKPT = Path(__file__).parent / 'checkpoints' / 'gpt.npz'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt', default='Holmes')
    parser.add_argument('--n', type=int, default=300, help='characters to generate')
    parser.add_argument('--temperature', type=float, default=0.8)
    parser.add_argument('--top_k', type=int, default=None)
    parser.add_argument('--top_p', type=float, default=None)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--ckpt', default=str(CKPT))
    args = parser.parse_args()

    params, cfg, tok, info = load_checkpoint(args.ckpt)
    print(f"checkpoint from step {info['step']}, val loss {info['val']:.3f}\n")
    P = {k: Tensor(v) for k, v in params.items()}
    print(generate(P, cfg, tok, args.prompt, args.n, np.random.default_rng(args.seed),
                   temperature=args.temperature, top_k=args.top_k, top_p=args.top_p))

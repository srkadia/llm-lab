# llm-lab

Small experiments on how LLMs work, built from scratch with numpy. No PyTorch, TensorFlow or JAX: gradients, attention, optimizers and training loops are all written by hand, and numpy is only there for the array math.

Each module tries to answer one question, has tests showing the code is correct, and ends with one chart. Everything runs on a CPU.

| # | module | question | result |
|---|---|---|---|
| 01 | [autograd](01-autograd/) | How do you know your gradients are right, and how small should eps be in a gradient check? | float64 central differences get the error to ~1e-10 and catch bugs as small as 1e-6. float32 misses bugs of 1e-4 and smaller (caught = error at least 10x above the check's noise floor), and at eps = 1e-6 it calls a correct gradient 25% wrong. |
| 02 | [attention](02-attention/) | Why does attention divide the scores by √d? | Without it, at d = 1024 attention is almost one-hot at init and the gradient through softmax is thousands of times smaller. It still trains, but the largest working learning rate drops 10x from d = 16 to 512, and a control shows that part isn't caused by the saturation. With √d, learning rates from 0.1 to 3 work at every size with SGD. With Adam the whole range moves down ~10x as d grows, so only 0.003-0.01 works at every size (see 03). |
| 03 | [tiny GPT](03-gpt/) | Why does training without warmup get stuck? | A 0.82M char-level GPT on a Sherlock Holmes book. Without warmup it first collapses to the unigram loss, and at lr 3e-2 without clipping it's still there after 200 steps. Adam's first steps add one shared direction to the weights, up to 4.5x as strong as the strongest direction already there, and block 0 makes every position look the same. Removing just that direction from each update prevents it, shrinking the update by the same amount doesn't. |
| 04 | [KV cache](04-kv-cache/) | How much faster does generation get with a KV cache, and how much memory does it cost? | Same logits and same text as without it. At 2,048 tokens a new token takes ~2-3 ms instead of ~1-1.7 s (~430-510x against a forward that does attention in blocks). The cache grows linearly, 8 KB per token here, and passes the model's own size at 800 tokens. For one sequence it's still ~3.6x smaller than what a step without it needs, so the real memory cost comes from serving many sequences at once. |

Next is DPO / GRPO on the GPT from 03. After that, roughly: quantization, then a tokenizer (BPE), MoE, mixed precision, LoRA, distillation, speculative decoding, simulated parallelism, and evals.

Every module has a `notebooks/` folder that builds the same thing step by step, if you'd rather learn it than read the final code.

## Setup

```bash
conda env create -f environment.yml
conda activate llm-lab
python -m ipykernel install --user --name llm-lab   # only needed for the notebooks
```

Then `cd` into a module and run its `test_*.py` and `run.py`.

[MIT licensed](LICENSE).

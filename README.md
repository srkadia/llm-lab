# llm-lab

Small experiments on how LLMs work, built from scratch with numpy. No PyTorch, TensorFlow or JAX: gradients, attention, optimizers and training loops are all written by hand, and numpy is only there for the array math.

Each module tries to answer one question, has tests showing the code is correct, and ends with one chart. Everything runs on a CPU.

| # | module | question | result |
|---|---|---|---|
| 01 | [autograd](01-autograd/) | How do you know your gradients are right, and how small should eps be in a gradient check? | float64 central differences get the error to ~1e-10 and catch bugs as small as 1e-6. float32 misses bugs of 1e-4 and smaller (caught = error at least 10x above the check's noise floor), and at eps = 1e-6 it calls a correct gradient 25% wrong. |
| 02 | [attention](02-attention/) | Why does attention divide the scores by √d? | Without it, at d = 1024 attention is almost one-hot at init and the gradient through softmax is thousands of times smaller. It still trains, but the largest working learning rate drops 10x from d = 16 to 512, and a control shows that part isn't caused by the saturation. With √d the same learning rate works at every size. |

Next is training a tiny GPT on top of these two, with a hand-written AdamW. After that, roughly: KV cache, DPO / GRPO, quantization, then a tokenizer (BPE), MoE, mixed precision, LoRA, distillation, speculative decoding, simulated parallelism, and evals.

Every module has a `notebooks/` folder that builds the same thing step by step, if you'd rather learn it than read the final code.

## Setup

```bash
conda env create -f environment.yml
conda activate llm-lab
python -m ipykernel install --user --name llm-lab   # only needed for the notebooks
```

Then `cd` into a module and run its `test_*.py` and `run.py`.

[MIT licensed](LICENSE).

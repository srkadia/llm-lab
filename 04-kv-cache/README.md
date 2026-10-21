# 04 · KV cache

A KV cache for the GPT from 03, in plain numpy. Generation doesn't need gradients, so it skips the autograd graph too. Same model and same text, but each new token only runs one position through the model.

The question for this one: how much faster does generation get, and how much memory does the cache cost as the context grows?

![time per new token and memory against context length, with and without the cache](figures/kv_cache.png)

To write the next token you normally run the model on everything so far. But earlier positions never change when you append a token (the causal mask from 02), so you can keep their keys and values and only compute the new one. With the trained model from 03, the cached version gives the same logits to 2e-14 at every step up to its 128-character limit, and exactly the same text, greedy and sampled.

The trained model only has 128 positions, so for longer contexts I used the same architecture with random weights. The output is nonsense, but the work per token and the memory don't depend on the weights: trained and random take the same time at the same context, within the noise of the timings.

| context | without cache | with cache | speedup |
|---|---|---|---|
| 16 | ~2-4 ms | ~0.65-1.3 ms | ~3-4x |
| 128 | ~15-24 ms | ~0.7-1.5 ms | ~16-23x |
| 512 | ~125-185 ms | ~0.9-1.7 ms | ~105-140x |
| 2,048 | ~1.4-1.7 s | ~1.9-3.3 ms | ~530-760x |
| 16,384 | - | ~9.5-16 ms | |

(Fastest of several tries per point, and the ranges cover six runs on the same laptop. Single timings move by up to 2x between runs, so read the shape, not the third digit.)

Without the cache each new token redoes every position and attention over every pair, so it gets slower faster and faster (about 3-3.5x per doubling by 2,048 tokens). With the cache it's flat at ~1 ms up to ~512 tokens and then grows linearly. The bend is where attention over the cache gets as expensive as the rest of the step, at t = 6C = 768 for this model. The measured speedups are always smaller than counting multiply-adds says (3-4x against 16x at 16 tokens), because a cached step is mostly a fixed ~1 ms of Python and numpy overhead at this size. Writing 512 tokens from scratch: ~27-34 s without, ~0.45-0.7 s with (~45-70x).

Memory surprised me. The cache is 8 KB per token here, so 16.8 MB at 2,048 tokens, and it passes the size of the model itself (6.55 MB) at 800 tokens. But without the cache, every step builds the attention scores for all pairs, and that peaks at 581 MB at 2,048 tokens, 35x the cache. So for one sequence the cache saves memory. The cost shows up with many sequences at once, because the cache is kept per sequence. Calculated, not measured: Llama-2-7B needs 0.5 MB per token in fp16, 2.1 GB for one 4,096-token sequence, so 8 of them need more memory than the model's weights. That's why Llama-2-70B uses grouped-query attention: 10x the parameters, but a smaller cache per token (0.33 MB) than the 7B.

The limit: with learned position embeddings the cache can't slide past 128. Every cached key was made with its position baked in, and after dropping the oldest token they're all off by one. A naive sliding cache picks the same next character as the real model only 10% of the time. So past 128 `generate()` goes back to recomputing the last 128 characters every step, like 03. For short outputs the cache is much faster (100 characters in ~0.1 s), but for 300 characters it only goes from ~4.2 s to ~3.5 s, because most of them are past the limit.

Caveats: one laptop, float64 numpy on a CPU, not a GPU. The long contexts use random weights. At this size the cached step is overhead, not arithmetic, so a GPU or a bigger model would give a different speedup.

## Files

- `kv_cache.py`: the numpy forward, the cache, chunked prefill, and `generate()` (falls back to recomputing past the context limit)
- `sample.py`: 03's sampler with the cache. `--no-cache` for the old way
- `run.py`: the experiment, makes the chart
- `test_kv_cache.py`: 10 checks. Same logits as 03's engine, prefill and one-token-at-a-time against the full forward, chunked prefill against one shot, cached generation against uncached, the same 200 characters as 03's sampler past the limit, and the cache refusing to overflow
- `notebooks/`: the same thing step by step ([01](notebooks/01_kv_cache.ipynb) building and checking the cache, [02](notebooks/02_experiment.ipynb) the experiment)

## Notes

- The first version prefilled a 16k-token context in one call. That builds a 16k x 16k score matrix per head, about 8.6 GB in float64, and the run took 7.7 min instead of 2. Prefilling 256 tokens at a time gives the same cache. Real inference servers do chunked prefill for the same reason.
- Skipping the autograd graph is a speedup on its own: without any cache, 300 characters take ~4.2 s with the numpy forward against ~10 s through the engine in 03.
- Appending a token changes the earlier logits by exactly 0.0, not just "close to 0". That's the whole reason a cache is allowed.

## Run

```bash
conda activate llm-lab
python test_kv_cache.py
python run.py            # ~2 min
python run.py --quick    # ~30 s
python sample.py
```

# 03 · tiny GPT

A character-level GPT (0.82M parameters, 4 layers) trained on *The Adventures of Sherlock Holmes*, using the engine from 01 and the transformer block from 02. AdamW, the warmup + cosine schedule, gradient clipping and sampling are all hand-written. The only new engine op is the embedding lookup.

The question for this one: why does training without warmup get stuck?

![loss, position variation and context use with and without warmup](figures/warmup.png)

I started out looking for loss spikes. I trained 8 short runs with and without warmup and clipping, at learning rates up to 10x the good one, and none of them spiked (that scout's code isn't in the repo). What did happen: every run without warmup first dropped to 3.1 and sat there, which is exactly the loss of a unigram model (always predict the overall character frequencies). How long it sat there depended on the learning rate and on clipping. At lr 3e-2 without clipping it was still there after 200 steps; with clipping, or at 1e-2, it got out later (2.93 and 2.62 at step 200). With warmup it never got stuck there.

So I measured what the model was doing. At lr 3e-2 without warmup, two steps in, it makes the same prediction at every position, whatever the text says. Block 0 is where it happens: after it, the vectors at different positions are almost identical (about 1% of the size is left that differs between positions, against ~100% at init).

The cause is Adam's first steps. At init every position's gradient says the same thing ("predict the common characters more"), so the gradients of the value, output, MLP and embedding matrices point mostly in one direction (80-99% of their energy, the query and key matrices less, 24-40%). Adam's first update is about ±lr per weight, whatever the gradient's size, and at lr 3e-2 that one direction ends up 0.7x to 4.5x as strong as the strongest direction already in the matrix (6.7x in the embedding). So block 0 adds the same big vector to every position.

To check that it's the direction and not just the step size, I edited every update during training (lr 3e-2, no warmup, first 30 steps):

| | how different positions are after block 0 | uses the context | loss |
|---|---|---|---|
| normal | 0.01 | 0.000 | 3.13 |
| update shrunk evenly | 0.03 | 0.000 | 3.11 |
| strongest direction removed | 1.13 | 0.429 | 2.71 |
| with warmup instead (reference) | 1.20 | 0.350 | 2.83 |

("Uses the context" is how much the predicted distribution changes between positions. 0 is a unigram model, the bigram model is at 0.64.)

Shrinking the update to the same size changes nothing. Removing that one direction stops the collapse, and the model is ahead of the warmup run after 30 steps. So warmup works by keeping that direction small while all the gradients still agree. Even the learning rate I actually train with (3e-3) is still blind to the context at step 10 without warmup (it recovers by step 30).

Two things I suspected first turned out not to be the cause. Attention does collapse to one-hot at high learning rates (Wortsman et al. 2023), but the model is already blind to the context at step 2, when its largest attention score is still ~18. And the run with warmup ends up with much larger scores than the stuck one (~2,600 vs ~400 at step 200) and trains fine. The biases (added equally to every position) aren't it either: with them frozen at 0, or with qk-layernorm, the model still collapses at step 2. Both do let it escape after ~90-120 steps; the normal run doesn't within 200 steps.

Caveats: one model size, one seed, one book, 200-step runs. Removing the top direction is a diagnostic here, not something I'd train with. Runs repeat to a few hundredths of a loss, not exactly, because numpy's multithreaded sums don't add in a fixed order. The 8-run scout above, the learning-rate sweep that picked 3e-3 for notebook 2 and two side remarks in notebook 3 are scout runs whose code isn't in the repo. Everything else comes from the notebooks and `run.py`.

**The √d check promised in 02.** Same retrieval task with Adam instead of SGD. √d still gives a wider range of learning rates that work. But with Adam the whole range moves down about 10x from d = 16 to 256 (0.003-0.1 to 0.0003-0.01 for ÷√d), so only a narrow band, 0.003-0.01, works at every size. With SGD it was 0.1-3. A learning rate picked from the middle of the range at d = 16 fails at d = 256. I only measured the edges of each range, not the best learning rate. This lines up with μP, which shrinks Adam's learning rate with width and scales attention by 1/d.

## The model

| | val loss |
|---|---|
| uniform guess | 4.48 |
| unigram | 3.10 |
| bigram | 2.43 |
| this GPT, 600 steps (notebook 2, ~20 min) | 1.67 |
| this GPT, best of 5000 steps (`train.py`, step 2250) | 1.42 |

The best val loss came at step 2250, about 18 passes over the book. After that it memorises: train loss keeps falling to 0.79 while val climbs back to 1.67 by step 5000. `train.py` keeps the best checkpoint, so that's the one `sample.py` uses. 0.82M parameters, half a million characters and no dropout, so it runs out of new things to learn early. Next time I'd train for ~2500 steps or add dropout.

`python sample.py` (temperature 0.8):

```
Holmes gammed dried
the fearsal wound from the fire. She him brought of him in the writing powerson.”

“What will be an insigation don?”

“Never. Jone was the old, the must rooms from who done upon the Eysander on
presence, which was of his lens? Where you guide’s so.”
```

It gets the look of the book right (dialogue, nested quotes, line breaks, mostly real words), but not the meaning. At 600 steps it was mostly made-up words.

## Files

- `gpt.py`: data, character tokenizer, the model (tied input/output embedding), sampling with temperature / top-k / top-p, checkpoints
- `train.py`: AdamW, warmup + cosine, gradient clipping, the training loop. Saves the best val checkpoint
- `sample.py`: text from the checkpoint
- `run.py`: the warmup experiment and the √d recheck, makes the chart
- `test_gpt.py`: 13 checks. Gradcheck of the whole model, no future leak, overfitting one batch, the AdamW and clipping properties, the sampling filters, checkpoint round trip
- `notebooks/`: the same thing step by step ([01](notebooks/01_data_and_model.ipynb) data and model, [02](notebooks/02_training.ipynb) training, [03](notebooks/03_experiment.ipynb) the experiment)

## Notes

- The embedding backward needs `np.add.at`. The obvious `W.grad[idx] += g` only adds one gradient per token when a token shows up more than once in the batch. It doesn't error, the model just trains worse. The gradient check fails it at 0.39.
- The first training run used up all 16 GB in a few steps. Every node's `_backward` closure points back at the node, so each step's graph is a reference cycle, and Python only frees those when its garbage collector runs. That's triggered by how many objects you create, not by how many bytes they hold, and one step is ~2,000 nodes holding ~3 GB. A `gc.collect()` after each step fixes it.
- The whole-model gradient check gives 2e-7 at the real init but 7e-9 with bigger embeddings. Not a bug: with 0.02 embeddings the gradients reaching the blocks are ~1e-3, and the finite-difference noise (~1e-9) is a bigger fraction of that.
- Training a step went from 10.3 s to 2.4 s after I gave `(B, T, C) @ (C, D)` its own path in 01's matmul.
- Greedy sampling gets stuck on "the state of the state of the ...".

## Run

```bash
conda activate llm-lab
python test_gpt.py
python train.py          # ~2.7 h, saves checkpoints/gpt.npz
python sample.py
python run.py            # ~35 min
python run.py --quick    # ~17 min
```

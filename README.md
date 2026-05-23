# Position Embeddings in a Character-Level Bengio MLP

A small experimental study of whether adding a within-name position embedding helps a Bengio-style MLP at character-level name generation, and *why*.

## Architecture

The baseline — a standard Bengio-style MLP. Three character indices get looked up
in an embedding table, the resulting vectors are concatenated, passed through a
hidden layer, and projected to a probability over the next character.

<img src="images\MLP structure detailed.gif" alt="Bengio MLP architecture" width="700">

The variant adds one branch: the current position within the name gets its own
embedding table and joins the concatenation. Everything from the hidden layer
onward is unchanged.

<img src="images\MLP_PE structure detailed.gif" alt="MLP+PE architecture" width="700">

The short answer: yes, it helps, but for two separable reasons. At small context
lengths the model can't recover position from its input window and the embedding
supplies it directly (a 1.2-perplexity gap at context length 2). At large context
lengths that information is already visible in the dot-padding pattern, and the
embedding's contribution mostly disappears. The full writeup is [here](#).

## What's in the repo

- `PosEmb.ipynb` — the experiment notebook. Self-contained, resumable, runs end-to-end in ~3 hours on a consumer GPU.
- `experiments/logs/` — JSONL training logs from all four phases (created on first run).
- `experiments/plots/` — figures from the writeup.
- `experiments/configs/` — best hyperparameters from random search, generated samples.
- `data/names.txt` — Karpathy's makemore names dataset (32,032 names).

## Experiments

| Phase | Question | Answer |
|-------|----------|--------|
| 1 | Best HPs for each model? | Independent random search, 20 trials each |
| 2 | Does pos-emb help at independently-tuned best? | Yes, 0.15 perplexity (5 seeds, σ ≈ 0.02) |
| 3 | Does the gap shrink as context grows? | Yes, from 1.21 at ctx=2 to ~0 by ctx=6 |
| 4 | Does the gap survive equal total parameters? | Yes, at every scale in the data-safe range |

## Running it yourself

```bash
git clone https://github.com/Offei-op/position-embeddings-bengio-mlp
cd position-embeddings-bengio-mlp
pip install -r requirements.txt
jupyter notebook PosEmb.ipynb
```

The notebook is resumable: training results are appended to JSONL logs, so a kernel restart picks up where you left off. Tested on PyTorch 2.x, Python 3.11, CUDA 12.

## Methodology notes

- Independent hyperparameter tuning per model (avoiding shared-HP signal distortion)
- 5 seeds in Phase 2/3, 3 seeds in Phase 4
- Two parameter-matching protocols: width-matched (same E, H, no pos-emb) and param-matched (different H, same total count)
- 80/10/10 train/val/test split; all reported numbers are on test

## Citation
If you reference this work:

Bekoe,O. O. (2026). Position embeddings in a character-level Bengio MLP:
a mechanism study.[ https://github.com/<your-username>/position-embeddings-bengio-mlp](https://github.com/Offei-op/position-embeddings-bengio-mlp)

## Acknowledgements

The base Bengio MLP architecture and the names dataset are from Andrej Karpathy's
[makemore](https://github.com/karpathy/makemore) series.
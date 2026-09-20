# Semantic-Adaptive, Temporally-Coherent LUT Synthesis for LOG Footage

Final-year AI project: generating interpretable, temporally-stable `.cube` LUTs
for LOG-encoded video, conditioned on scene understanding and semantic masks.

## Architecture

1. **Scene understanding** — frozen [DINOv2](https://github.com/facebookresearch/dinov2)
   embeddings (transfer learning).
2. **Feature fusion** — cross-attention: scene embedding + semantic mask tokens
   as keys/values, a learned "LUT query" set as queries. Built from scratch.
3. **LUT generation** — denoising diffusion over the LUT coefficient grid
   (e.g. 17x17x17x3), not pixel space. Chosen because a single scene can
   support multiple valid gradings — regression alone tends to average
   toward a bland result, whereas diffusion can represent that multi-modality.
4. **Temporal coherence** — attention over a sliding window of frames, rather
   than post-hoc smoothing.

## Constraints

Zero-cost only: Colab/Kaggle free tier, open datasets ([MIT-Adobe FiveK](https://data.csail.mit.edu/graphics/fivek/)),
free pretrained weights (DINOv2, YOLOv8-seg/SAM via Hugging Face), open-source
libraries only.

## Status

- [x] **Phase 1** — sanity-check that frozen DINOv2 embeddings linearly encode
  scene properties (outdoor/indoor, light source type) needed downstream.
  See [`PHASE1_RESULTS.md`](PHASE1_RESULTS.md).
- [ ] Phase 2 — cross-attention feature fusion
- [ ] Phase 3 — diffusion LUT generation
- [ ] Phase 4 — temporal coherence

## Setup

```bash
python -m venv .venv

# Windows
.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
.venv\Scripts\python.exe -m pip install -r requirements.txt

# macOS/Linux (no CUDA build needed if no NVIDIA GPU)
.venv/bin/python -m pip install torch torchvision
.venv/bin/python -m pip install -r requirements.txt
```

Adjust the `cu128` CUDA tag to match your GPU/driver if not using an RTX 50-series
card — see the [PyTorch install matrix](https://pytorch.org/get-started/locally/).

## Data

Images are **not** committed to this repo — MIT-Adobe FiveK photos are
copyrighted and distributed under the dataset's own terms. Reproduce the
Phase 1 sample data with:

```bash
.venv\Scripts\python.exe scripts\fetch_fivek_sample.py
```

This downloads one shard (~1.9GB, one-time) of a Hugging Face mirror of FiveK
([`logasja/mit-adobe-fivek`](https://huggingface.co/datasets/logasja/mit-adobe-fivek))
and extracts 77 labeled sample frames into `data/frames/` plus `data/labels.csv`.

## Phase 1 pipeline

```bash
.venv\Scripts\python.exe scripts\extract_embeddings_all.py       # DINOv2 embeddings for all frames
.venv\Scripts\python.exe scripts\linear_probe_generic.py location  # probe: outdoor/indoor
.venv\Scripts\python.exe scripts\linear_probe_generic.py light     # probe: light source type
```

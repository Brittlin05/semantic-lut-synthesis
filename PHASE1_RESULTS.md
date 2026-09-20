# Phase 1: Does frozen DINOv2 carry the scene information we need?

## Goal

Before building the cross-attention fusion module, sanity-check whether
frozen DINOv2 embeddings actually encode the scene properties a LUT-generation
model would need to condition on — specifically outdoor/indoor and light
source type (a proxy for color temperature). If a *linear* classifier can't
recover these from the embedding, that's a warning sign worth catching early.

## Method

- **Data**: 77 sample frames from MIT-Adobe FiveK (`original`/unretouched
  renditions, used as a flat/LOG-footage stand-in), pulled from a Hugging
  Face mirror that includes human-annotated `location` and `light` labels.
- **Embeddings**: `facebook/dinov2-small` (frozen, no fine-tuning), CLS/pooled
  token, 384 dimensions per image.
- **Probe**: `sklearn.linear_model.LogisticRegression` trained on top of the
  frozen embeddings — a single linear decision boundary, nothing more.
- **Split**: 75/25 train/test, stratified by label to preserve class ratios
  given the small sample size.
- **Baseline**: majority-class accuracy (always predicting the most common
  label) — the probe's accuracy is only meaningful relative to this.

## Results

### Outdoor / indoor (`location`)

| | accuracy |
|---|---|
| Majority-class baseline (always "outdoor") | 72.2% |
| Linear probe | **88.9%** |

Test set: 18 images (13 outdoor, 5 indoor). The baseline gets all 13 outdoor
right and all 5 indoor wrong by construction (13/18 = 72.2%). The probe kept
all 13 outdoor correct **and** got 3 of 5 indoor right — the improvement is
concentrated entirely on the minority class, which is the real evidence that
the embedding encodes indoor-ness, not just noise around the baseline.

**Misclassified**: 2 indoor images predicted as outdoor — both were
close-up/macro shots with no visible scene layout (a chrome lamp reflection,
a statue against a plain gradient background). DINOv2's global embedding
appears to lean on scene-layout cues (walls, furniture arrangement), which
these images simply don't provide.

### Light source type (`light`: sun_sky / artificial / mixed)

| | accuracy |
|---|---|
| Majority-class baseline (always "sun_sky") | 63.2% |
| Linear probe | **84.2%** |

Test set: 19 images (12 sun_sky, 5 artificial, 2 mixed). The probe kept all
12 sun_sky correct and picked up 4/5 artificial — but got **0/2 mixed
correct**, identical to the baseline. It never predicted "mixed" for any
test image.

**Misclassified**: the two mixed-light failures were a flash-lit macro flower
shot (ambiguous even to a human — daylight-accurate color, but a crushed-black
background suggesting flash) and a warmly-lit church interior (visually reads
as strongly artificial; the "mixed" label likely reflects an unseen daylight
source). One of the two location-probe failures (the chrome-lamp macro shot)
also failed here, independently confirming that image is a hard case for
DINOv2's global embedding specifically.

## Interpretation

1. **The embedding is usable.** Both probes beat their baselines by a wide
   margin, concentrated on the actual minority classes rather than free
   riding on the majority class. DINOv2's frozen features linearly encode
   the coarse scene properties this project needs.

2. **Known blind spot — macro/abstract shots.** Frames with no visible scene
   layout (extreme close-ups, objects filling the frame) are a consistent
   failure mode across *both* probes for the same image. Worth remembering
   for later phases — this is part of the motivation for temporal context
   (Phase 4): a single ambiguous frame can be disambiguated by neighboring
   frames.

3. **Known blind spot — blended categories.** "Mixed" lighting was never
   once predicted correctly, and structurally can't be: a linear boundary
   separates *regions* of a space, but "a blend of A and B" sits *between*
   two regions, not inside its own. This is not just a data-quantity problem
   (mixed had only 7 total examples) — it's a shape-of-the-model problem.

4. **This is a specific, testable argument for Phase 2's design**, not just
   post-hoc justification: Phase 1 only used the CLS token — one vector
   summarizing the *entire* image. Mixed lighting is often a *spatial* fact
   (window light on one side of a room, a lamp on the other) that a single
   pooled vector cannot represent, no matter how it's read out. Phase 2's
   cross-attention module uses the full grid of patch tokens (not just CLS)
   as keys/values, which is the first point in the pipeline with access to
   per-region information. Whether that actually resolves the mixed-lighting
   blind spot is an open question for Phase 2's own evaluation — not
   something Phase 1 can prove — but it's a principled hypothesis grounded
   in what changed architecturally, not just "a bigger model should work."

## Reproducing this

```bash
.venv\Scripts\python.exe scripts\fetch_fivek_sample.py
.venv\Scripts\python.exe scripts\extract_embeddings_all.py
.venv\Scripts\python.exe scripts\linear_probe_generic.py location
.venv\Scripts\python.exe scripts\linear_probe_generic.py light
```

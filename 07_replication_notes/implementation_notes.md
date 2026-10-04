# Implementation notes

## 1. Paper and code

- Li et al., "PromptAD: Learning Prompts with only Normal Samples for Few-Shot Anomaly
  Detection", CVPR 2024, arXiv 2404.05231.
- Code: github.com/FuNz-0/PromptAD (commit pinned in `02_src/source_code/commit.txt`). Bundles its
  own CLIP copy (`PromptAD/CLIPAD`) — the installed `open_clip` version does not affect the model.
- Backbone ViT-B/16+ (240px), LAION-400M e32 weights; SGD lr 0.002, momentum 0.9, wd 5e-4,
  100 epochs, cosine schedule; λ = 0.001; E_N = 4, E_A = 1, L = 4.
- The paper also reports MVTec LOCO results (supplementary Table 9, image/pixel AUROC):
  71.2/73.0, 72.6/74.1, 73.5/74.5 at 1/2/4-shot. The released code has no LOCO data loader, so
  this replication does not cover it.

## 2. Compute

- AWS `g5.2xlarge`, 1× A10G 24 GB, us-east-1, Ubuntu 22.04; ~$1.21/h.
- Python 3.10 (`uv` venv), torch 2.5.1 + cu121, numpy 1.26.4.
- Fixed-epoch (epoch-100-only) runs: <1 minute each. With 21 test-set evaluations per run
  (the paper's protocol), ≈1–3 minutes per MVTec class at image level.
- Total workspace time for the full sweep + diagnostics: ≈3 hours (≈$4).

## 3. Headline results — fast mode (epoch 100, seed 111, sorted training order) vs. paper

Mean AUROC (%):

| Metric | Dataset | 1-shot | 2-shot | 4-shot | Paper 1/2/4 |
|---|---|---|---|---|---|
| Image | MVTec AD | 86.12 | 86.63 | 87.75 | 94.6 / 95.7 / 96.6 |
| Image | VisA | 82.21 | 81.08 | 83.38 | 86.9 / 88.3 / 89.1 |
| Pixel | MVTec AD | 92.18 | 93.23 | 93.21 | 95.9 / 96.2 / 96.5 |
| Pixel | VisA | 92.13 | 92.81 | 93.48 | 96.7 / 97.1 / 97.4 |

(162 runs total: 2 tasks × 2 datasets × 3 shots × 15/12 classes.)

## 4. Diagnostics (MVTec AD, 1-shot, image AUROC)

| Run | Setting | Result | Gap to paper (94.6) |
|---|---|---|---|
| Main | epoch 100, sorted file order | 86.12 | −8.48 |
| A | epoch 100, original unsorted order (`SORT_TRAIN=0`) | 85.46 | −9.14 |
| B | evaluated at epochs 1, 5, 10, …, 100 — best on test set | **93.43** | **−1.17** |
| B | same run, epoch 100 value | 86.12 | −8.48 |

Diagnostic B per class (sorted order, 21 evaluations each):

| Class | Best | Epoch 100 | Best epoch | Drop best→100 |
|---|---|---|---|---|
| bottle | 100.0 | 97.62 | 1 | −2.4 |
| cable | 91.32 | 75.11 | 10 | −16.2 |
| capsule | 73.27 | 65.98 | 1 | −7.3 |
| carpet | 100.0 | 99.92 | 10 | −0.1 |
| grid | 99.58 | 92.48 | 1 | −7.1 |
| hazelnut | 99.07 | 80.11 | 1 | −19.0 |
| leather | 100.0 | 100.0 | 20 | 0 |
| metal_nut | 99.22 | 90.13 | 15 | −9.1 |
| pill | 91.65 | 82.98 | 25 | −8.7 |
| screw | 65.63 | 53.62 | 1 | −12.0 |
| tile | 99.96 | 99.86 | 30 | −0.1 |
| toothbrush | 100.0 | 86.11 | 15 | −13.9 |
| transistor | 87.71 | 80.88 | 25 | −6.8 |
| wood | 98.6 | 92.46 | 20 | −6.1 |
| zipper | 95.43 | 94.51 | 30 | −0.9 |
| **Mean** | **93.43** | **86.12** | median 15 | −7.3 |

## 5. Findings

- **Test-set epoch selection inflates reported image AUROC** by ≈7.3 points at MVTec AD 1-shot.
  Textures (carpet, leather, tile) barely change between best epoch and epoch 100; objects
  (hazelnut, cable, toothbrush, screw) drop 12–19 points. This is the released code's own
  protocol (not a bug we introduced) — the paper's Table 1/4 numbers are produced this way.
  A closely related issue was found during the LogiCo replication in this same project
  (+2.4 points from the same effect there, smaller because LogiCo's task is less prone to
  early-epoch overfitting).
- **Prompt learning overfits quickly for several classes**: 5 of 15 MVTec classes peak at epoch 1
  (a single optimisation step); none peak after epoch 30 (median best epoch 15). By epoch 100,
  mean performance (86.12) is close to the paper's own ablation that drops the prompt-guided
  branch entirely and uses vision-guided detection alone (85.1) — i.e. without epoch selection,
  the learned prompts contribute comparatively little beyond what the frozen-CLIP visual branch
  already provides.
- **Training-image file order** (sorted vs. the code's original unsorted glob) changes the result
  by only 0.7 points — a minor factor next to epoch selection.
- **Upstream colour-channel bug**: `train_cls.py` / `train_seg.py` convert training images from
  BGR to RGB (`cv2.cvtColor(..., cv2.COLOR_BGR2RGB)`) before encoding, but feed test images to
  CLIP directly from `cv2.imread` (BGR), with no equivalent conversion. This affects anyone
  running the released code as-is; left unpatched here deliberately, so results stay comparable
  to an out-of-the-box run of the official repository. See `02_src/source_code/README.md`.
- **GitHub issue #27** on the upstream repo: other users report results "slightly" below the
  paper even using the full (best-epoch) protocol; the author's response attributes this to
  hardware and sampled-image variance — consistent with the 1.17-point gap found here.
- **Implication for the thesis**: PromptAD should be reported under both protocols —
  "released protocol (test-set epoch selection)" and "fixed epoch 100" — since the two differ by
  nearly 9 points and the released code defaults to the former without making that explicit in
  the paper's text.

## 6. Scope / honesty notes

- Single seed (111), not the paper's mean of 5 seeds, due to time/compute budget.
- The full best-epoch diagnostic (protocol matching the paper exactly) has only been run for
  MVTec AD, 1-shot, image AUROC; all other reported settings use the fixed epoch-100 protocol.
  See `04_configs/run_configs.md` for the estimated cost of extending it to every setting
  (≈4–5 additional GPU-hours).
- MVTec LOCO is not evaluated (no loader in the released code).

## To do

- [x] Fast-mode sweep (162/162 runs)
- [x] Diagnostics A and B
- [x] Paper-style tables and qualitative figures
- [ ] Extend Diagnostic B (best-epoch protocol) to the remaining 11 settings
- [ ] Multi-seed runs (paper reports mean of 5 seeds)

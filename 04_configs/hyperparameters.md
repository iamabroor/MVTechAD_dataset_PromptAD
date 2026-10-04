# Hyperparameters

All values are the released code's defaults (`FuNz-0/PromptAD`) — nothing here was retuned.

## Model

| Parameter | Value |
|---|---|
| Backbone | CLIP ViT-B/16+ (240px), LAION-400M e32 weights |
| Normal prompt tokens (learnable prefix) | — |
| Explicit anomaly prompts (E_N) | 4 |
| Learnable anomaly prompts (E_A) | 1 |
| Prompt context length (L) | 4 |
| Anomaly margin loss weight (λ) | 0.001 |

## Optimisation

| Parameter | Value |
|---|---|
| Optimiser | SGD |
| Learning rate | 0.002 |
| Momentum | 0.9 |
| Weight decay | 5e-4 |
| Schedule | cosine |
| Epochs | 100 |
| Seed (this replication) | 111 (paper reports mean over 5 seeds) |

## Data

| Parameter | Value |
|---|---|
| k-shot | 1, 2, 4 |
| Datasets | MVTec AD (15 classes), VisA (12 classes, 1-class split) |
| Image size | 240×240 (CLIP ViT-B/16+ input) |

## Run configuration (pipeline environment variables)

| Variable | Default | Meaning |
|---|---|---|
| `SEED` | `111` | torch seed |
| `TASKS` | `"cls seg"` | `cls` = image-level, `seg` = pixel-level |
| `DATASETS` | `"mvtec visa"` | which datasets to run |
| `SHOTS` | `"1 2 4"` | k-shot settings |
| `SORT_TRAIN` | `1` | sort the glob of 'good' training images before applying k-shot indices (reproducibility fix — see `02_src/source_code/README.md`) |
| `EVAL_EPOCHS` | unset | comma-separated epochs to evaluate on the test set; unset = fixed epoch-100 only (fast mode); `1,5,10,...,100` reproduces the paper's best-epoch-on-test-set protocol |
| `VIS` | unset | when set for a `seg` run, also save per-image qualitative output (`*_ori.jpg`, `*_gt.jpg`, `*_PromptAD.jpg`) and a per-image pixel-AUROC CSV, used for the paper-style figures |

See [`04_configs/run_configs.md`](run_configs.md) for the exact settings used to produce every
result in `05_results/`.

# PromptAD Replication

Independent replication of:

> Li, Xiaofan, et al. **"PromptAD: Learning Prompts with only Normal Samples for Few-Shot Anomaly Detection."** CVPR 2024. [arXiv:2404.05231](https://arxiv.org/abs/2404.05231)

Official code: [FuNz-0/PromptAD](https://github.com/FuNz-0/PromptAD) (Unlicense).

## What this is

PromptAD is a one-class prompt-learning method for few-shot anomaly detection. It fine-tunes a
small set of learnable prompt tokens on top of a frozen CLIP (ViT-B/16+, LAION-400M) using only
1/2/4 *normal* example images per class, then detects and localises anomalies at test time by
fusing a prompt-guided branch (PAD) and a vision-guided branch (VAD) via a harmonic mean.

This repository reproduces the paper's MVTec AD and VisA experiments (image- and pixel-level
AUROC, 1/2/4-shot) on a single NVIDIA A10G GPU, using the authors' released code with a small
number of documented patches, and reports:

- **A faithful reproduction of the paper's own evaluation protocol** (best epoch selected on
  the test set, as the released code does), and
- **A fixed-epoch (epoch 100) variant**, which isolates and quantifies how much of the paper's
  reported numbers depend on that test-set epoch selection.

## Headline finding

At MVTec AD, 1-shot, image-level AUROC:

| Protocol | AUROC | Gap to paper (94.6) |
|---|---|---|
| Paper (mean of 5 seeds, best-epoch-on-test-set selection) | 94.6 | — |
| **This replication — same protocol** (seed 111, best of 21 checkpoints) | **93.43** | −1.17 |
| **This replication — fixed epoch 100, no selection** | **86.12** | −8.48 |

Selecting the best of 21 test-set evaluations (epochs 1, 5, 10, …, 100) accounts for
**≈7.3 of the 8.5-point gap** between a naive single-checkpoint run and the paper's number.
Full breakdown, per-class results, and discussion are in
[`07_replication_notes/implementation_notes.md`](07_replication_notes/implementation_notes.md).

## Repository layout

```
PromptAD-Replication/
├── README.md                      # this file
├── 01_methodology/                # method summary + diagram
├── 02_src/source_code/            # pointer to the exact upstream commit + our patch diff
├── 03_scripts/                    # preprocessing / training / inference / evaluation entry points
├── 04_configs/                    # hyperparameters and run configurations
├── 05_results/                    # metrics (CSV), figures (PNG), tables (CSV/PNG)
├── 06_environment/                # environment + install instructions
├── 07_replication_notes/          # implementation notes, issues found, and fixes
└── LICENSE
```

## Compute

Single AWS `g5.2xlarge` instance (1× NVIDIA A10G, 24 GB), Ubuntu 22.04, ~3 hours total wall-clock
for the fast-mode sweep (162 runs across tasks × datasets × shots × classes) plus diagnostics.
See [`06_environment/environment_setup.md`](06_environment/environment_setup.md).

## Status

Results, figures and tables for the fast-mode sweep (fixed epoch 100) and the best-epoch
diagnostic on MVTec AD 1-shot are complete and included. See
[`05_results/`](05_results/) and the to-do list in
[`07_replication_notes/implementation_notes.md`](07_replication_notes/implementation_notes.md)
for what is still pending.

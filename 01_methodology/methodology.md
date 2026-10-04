# Methodology

## 1. Task

Few-shot, one-class anomaly detection: given only *k* ∈ {1, 2, 4} normal (defect-free) example
images of an object/texture class, detect whether a new test image of that class is anomalous
(image-level AUROC) and, if so, localise the anomalous region (pixel-level AUROC).

## 2. Method summary (PromptAD, CVPR 2024)

PromptAD builds on CLIP (ViT-B/16+, LAION-400M weights) with everything frozen except a small set
of learnable prompt tokens.

1. **Semantic Concatenation (SC).** Two kinds of text prompts are built per class:
   - A **normal prompt (NP)**: a learnable normal-prefix concatenated with the class name.
   - **Anomaly prompts**: a **manual anomaly prompt (MAP)** bank (short phrases such as "with a
     crack", "with a scratch") and **learnable anomaly prompts (LAP)**, concatenated with the
     normal prompt's representation so that anomaly prompts are expressed as *deviations from
     normal* rather than being learned from scratch (there are no anomalous training images to
     learn them from).
2. **Explicit Anomaly Margin (EAM).** A margin loss pushes the anomaly prototype away from the
   normal prototype in embedding space by at least a fixed amount, preventing the two from
   collapsing together, which is otherwise easy to do when only normal images are available.
3. **Two inference branches, fused by harmonic mean.**
   - **PAD (prompt-guided):** text–image similarity between the test image's CLS embedding and
     the learned normal/anomaly prototypes.
   - **VAD (vision-guided):** a memory bank of normal-image patch/CLS features (built once from
     the *k* shots) compared against the test image via nearest-neighbour distance, giving a
     complementary, purely visual anomaly signal.
   - Image score and pixel map are each computed as the harmonic mean of the two branches'
     scores, combining the two signals' strengths rather than picking one.

Training optimises a combination of three losses: a CLIP-style contrastive loss between images
and prompts, the explicit anomaly margin loss, and an alignment loss between the two branches.
Only the *k* normal shots are used; no anomalous images are seen during training.

See [`methodology_diagram.png`](methodology_diagram.png) for the full training/inference data
flow (also published as an interactive diagram during this replication).

## 3. Datasets

- **MVTec AD** — 15 classes (10 objects, 5 textures), industrial defect inspection.
- **VisA** — 12 classes, more complex objects and layouts (1-class split, `spot-diff` protocol).

## 4. Evaluation protocol

- Metrics: image-level AUROC (classification) and pixel-level AUROC (segmentation).
- Shots: 1, 2, 4 (k-shot normal examples, fixed index files shipped with the released code).
- The released training code evaluates on the test set **after every one of 100 epochs** and
  keeps the result from whichever epoch scored best — i.e. the epoch is itself selected using
  test-set labels. The paper's reported numbers (Table 1, Table 4) come from this protocol,
  averaged over 5 seeds.
- This replication reports results under **two** protocols for transparency (see
  [`05_results/`](../05_results/) and
  [`07_replication_notes/implementation_notes.md`](../07_replication_notes/implementation_notes.md)):
  1. The released protocol, with checkpoints evaluated at epochs 1, 5, 10, …, 100 and the best
     kept (closest apples-to-apples comparison with the paper).
  2. A fixed epoch-100 checkpoint, with no test-set-based selection — a stricter, standard
     train/test split.

## 5. Scope of this replication

- Single seed (111) rather than the paper's mean of 5 seeds, due to compute/time budget.
- Full diagnostic sweep (protocol 1 above) run for MVTec AD, 1-shot, image AUROC only; all other
  settings (other shots, VisA, pixel-level) use the fixed epoch-100 protocol. See the to-do list
  in [`07_replication_notes/implementation_notes.md`](../07_replication_notes/implementation_notes.md).
- MVTec LOCO (paper's supplementary Table 9) is not evaluated — the released code has no loader
  for it.

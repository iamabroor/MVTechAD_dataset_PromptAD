# Results

- **`metrics/`** — raw CSVs behind every number reported elsewhere in this repository:
  - `fast_mode_vs_paper.csv` — all 12 (metric × dataset × shot) settings, fixed epoch 100, vs. paper.
  - `diagnostics.csv` — the four diagnostic runs (main, sort-order, best-epoch, best-epoch's own epoch-100 value).
  - `diagnostic_b_per_class.csv` — MVTec AD 1-shot image AUROC, all 15 classes, best vs. epoch-100 vs. best epoch index.
- **`tables/`** — paper-style Table 1 (image AUROC) and Table 4 (pixel AUROC), as CSV and PNG,
  reproducing the paper's layout with our rows added.
- **`figures/`** — paper-style qualitative results, produced by
  `03_scripts/paper_style_figures.py` from the actual `VIS=1` run on the training workspace:
  - `fig3_qualitative_best.png` / `fig3_qualitative_random.png` — paper Fig. 3 style: 1-shot,
    MVTec AD block + VisA block, 8 classes each, rows = Image / Ground truth / PromptAD (ours).
  - `fig8_mvtec_best.png` / `fig8_mvtec_random.png` — supplementary Fig. 8 style: all 15 MVTec AD
    classes, 1-shot.
  - `fig8_visa_best.png` / `fig8_visa_random.png` — supplementary Fig. 8 style: all 12 VisA
    classes, 1-shot.
  - `comparison_vs_paper.png` — bar chart of paper vs. our epoch-100 results across all 12
    settings.
  "best" = the anomalous test image with the highest per-image pixel AUROC per class; "random" =
  a random anomalous test image per class (unbiased view).

## Pending

These are still outstanding and will be added once available:

- **Raw per-image files** (`*_ori.jpg`, `*_gt.jpg`, `*_PromptAD.jpg`, `per_image_<class>.csv`)
  behind the figures above — only the rendered composite grids have been pulled in so far, not
  the individual source images from `<out>/vis/<dataset>/k_<shot>/imgs/`.
- **`02_src/source_code/commit.txt`** and **`code_patches.diff`** — still placeholders; need the
  actual contents of `~/promptad/output/promptad_commit.txt` and `~/promptad/output/code_patches.diff`
  from the training workspace.
- **`per_class.csv`** and **`summary.md`** (the full per-class numeric breakdown behind the fast-mode
  sweep) — not yet transferred.
- The best-epoch (released-protocol) diagnostic is only complete for MVTec AD, 1-shot, image
  AUROC — the other 11 settings show "–" in `tables/table1_image_auroc.png` /
  `table4_pixel_auroc.png` until that sweep is run (see `04_configs/run_configs.md`).

None of this blocks review of what's here now — these can be added in a follow-up update once the
AWS workspace file transfer is sorted out.

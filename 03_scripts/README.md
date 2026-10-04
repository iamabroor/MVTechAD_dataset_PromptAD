# Scripts

These wrap the original resumable pipeline (`run_promptad.sh`, included here as
[`run_pipeline.sh`](run_pipeline.sh)) into the four stages requested for this repository layout.
Each stage below is literally one `MODE=` of that pipeline — they are kept as thin wrappers
rather than rewritten, so behaviour exactly matches what was actually run.

| File | Pipeline mode | What it does |
|---|---|---|
| `environment_setup.sh` | `MODE=setup` | venv, clone PromptAD at pinned commit, install deps, apply patches |
| `preprocessing.py` | `MODE=data` | download/prepare MVTec AD + VisA, record exact k-shot training images |
| `training.py` | `MODE=run` | train every (task × dataset × shot × class) combination |
| `inference.py` | n/a (invoked per-run inside `MODE=run`/`MODE=smoke`) | runs the released `train_cls.py` / `train_seg.py` end-to-end, which trains *and* evaluates in one process — PromptAD's released code does not separate training from inference |
| `evaluation.py` | `MODE=summary` | parse logs into `per_class.csv` / `summary.md` (best epoch, epoch-100, gap to paper) |

Additional evaluation helpers (run after `evaluation.py`, not part of `run_pipeline.sh`'s modes):

| File | What it does |
|---|---|
| `compare_to_paper.py` | builds `comparison_vs_paper.csv`, `per_class_comparison.csv`, and a paper-vs-ours bar chart |
| `paper_style_figures.py` | builds Table 1 / Table 4 style CSV+PNG tables and Fig. 3 / Fig. 8 style qualitative grids (image / ground truth / PromptAD heatmap), matching the paper's figure layout |
| `qualitative_figures.py` | simpler per-class overview and per-class detail grids from a `VIS=1` pixel-level run's saved images |

Note on naming: the released PromptAD code does not have a separate "train" vs. "inference"
script — `train_cls.py`/`train_seg.py` train the prompts and then immediately evaluate on the
test set each logged epoch. `training.py` and `inference.py` here are both thin wrappers around
the same call for clarity against the requested folder layout; see comments in each file.

## Usage

```bash
bash environment_setup.sh                 # one-time setup
python preprocessing.py                   # download + prepare data
python training.py                        # run all 162 (task, dataset, shot, class) combinations
python evaluation.py                      # build per_class.csv, summary.md, tables, figures
```

Environment variables (all optional, see `04_configs/`):

```
SEED=111            TASKS="cls seg"      DATASETS="mvtec visa"   SHOTS="1 2 4"
SORT_TRAIN=1         EVAL_EPOCHS=1,5,10,...,100 (unset = fixed epoch-100 only)
```

Run inside `tmux` — training is long enough to outlast a terminal disconnect:

```bash
tmux new -s promptad
```

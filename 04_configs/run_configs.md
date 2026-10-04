# Run configurations used

## Main sweep — fixed epoch 100 ("fast mode")

Produces the headline comparison table in `05_results/tables/` and all VisA / 2-shot / 4-shot /
pixel-level rows.

```bash
SEED=111 TASKS="cls seg" DATASETS="mvtec visa" SHOTS="1 2 4" SORT_TRAIN=1 \
  python training.py
```

No `EVAL_EPOCHS` set → each of the 162 runs trains for 100 epochs and is evaluated once, at the
end, with no test-set-based model selection.

## Diagnostic A — training-image order sensitivity

Same as the main sweep but with the original (unsorted) glob order for the 'good' training image
list, MVTec AD 1-shot image AUROC only:

```bash
SEED=111 TASKS=cls DATASETS=mvtec SHOTS=1 SORT_TRAIN=0 python training.py
```

Result: 85.46 vs. 86.12 with `SORT_TRAIN=1` — a 0.66-point difference, i.e. training-image choice
is a minor factor compared to epoch selection (below).

## Diagnostic B — released paper protocol (best-epoch-on-test-set selection)

MVTec AD, 1-shot, image AUROC, all 15 classes, evaluated at epochs 1, 5, 10, …, 100 and the best
kept — reproducing exactly what the released training scripts do by default:

```bash
SEED=111 TASKS=cls DATASETS=mvtec SHOTS=1 SORT_TRAIN=1 \
  EVAL_EPOCHS=1,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100 \
  python training.py
```

Result: 93.43 mean (best of 21 evaluations per class) vs. the paper's 94.6 — a 1.17-point gap,
consistent with GitHub issue #27 on the upstream repo ("slightly below the paper, varies with
hardware/sampled images"). The same run's epoch-100 value (86.12) matches the main sweep exactly,
confirming the evaluation gate does not affect training (deterministic).

## Not yet run

The full best-epoch protocol (Diagnostic B) has only been run for MVTec AD 1-shot image AUROC.
Running it for the remaining 11 settings (MVTec 2/4-shot, VisA 1/2/4-shot, each × image/pixel)
would take an estimated 4–5 additional GPU-hours with image- and pixel-level runs in parallel.

#!/usr/bin/env python3
"""Inference / single-run entry point: wraps `MODE=smoke run_pipeline.sh` (one
(task, dataset, shot, class) combination), or calls the upstream scripts directly.

PromptAD's released code has no separate "load a trained model, run on new
images" script: `train_cls.py` / `train_seg.py` build the k-shot memory + prompts
and immediately evaluate on the dataset's full test split as part of the same
run. This wrapper is provided for the requested folder layout and for timing a
single run (used here to estimate total sweep time before launching `training.py`):

    python inference.py --task cls --dataset mvtec --shot 1 --class_name bottle

which is equivalent to, from the upstream repo root:

    python train_cls.py --dataset mvtec --class_name bottle --k-shot 1 --seed 111
"""
import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["cls", "seg"], default="cls")
    ap.add_argument("--dataset", choices=["mvtec", "visa"], default="mvtec")
    ap.add_argument("--shot", type=int, default=1)
    ap.add_argument("--class_name", default="bottle")
    ap.add_argument("--seed", type=int, default=111)
    a = ap.parse_args()

    import os
    env = {
        "MODE": "smoke", "SEED": str(a.seed), "TASKS": a.task,
        "DATASETS": a.dataset, "SHOTS": str(a.shot), **os.environ,
    }
    # run_pipeline.sh's `smoke` mode is hard-wired to (cls, mvtec, 1, bottle) for
    # timing purposes; for an arbitrary combination, call the upstream script
    # directly instead (same thing run_one() in run_pipeline.sh does per-run).
    repo = Path.home() / "promptad" / "PromptAD"
    venv_python = Path.home() / "promptad" / ".venv" / "bin" / "python"
    extra = ["--vis", "False"] if a.task == "seg" else []
    subprocess.run(
        [str(venv_python), "-u", f"train_{a.task}.py",
         "--dataset", a.dataset, "--class_name", a.class_name,
         "--k-shot", str(a.shot), "--seed", str(a.seed), *extra],
        check=True, cwd=repo,
    )


if __name__ == "__main__":
    sys.exit(main())

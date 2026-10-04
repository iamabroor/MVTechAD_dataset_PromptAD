#!/usr/bin/env python3
"""Training stage: wraps `MODE=run run_pipeline.sh`.

Runs every (task, dataset, shot, class) combination configured via the TASKS /
DATASETS / SHOTS environment variables (defaults: cls+seg, mvtec+visa, 1/2/4-shot
-> 162 total runs), calling the released `train_cls.py` / `train_seg.py` once per
combination. Each run is skipped if its log already shows completion, so this is
safe to re-run after an interruption.

Note: PromptAD's released training scripts train the prompts AND evaluate on the
test set within the same process/run — there is no separate held-out "inference"
step in the upstream code. `inference.py` in this folder documents that single
entry point explicitly; this script is what actually launches it for the full
sweep.
"""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    subprocess.run(["bash", str(HERE / "run_pipeline.sh")], check=True,
                    env={"MODE": "run", **__import__("os").environ})


if __name__ == "__main__":
    sys.exit(main())

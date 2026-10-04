#!/usr/bin/env python3
"""Preprocessing stage: wraps `MODE=data run_pipeline.sh`.

Downloads MVTec AD and VisA, runs VisA's 1-class preprocessing, and records the
exact k-shot training images selected for every (dataset, class, shot) combination
to `selected_train_images.txt` (for reproducibility / the report).

The released PromptAD code's MVTec AD download link (the common `mydrive.ch` URL
seen in many forks) is dead (404 as of Oct 2026). This pipeline uses the current
working link from the `anomalib` project's `mvtecad.py` data module instead; the
VisA link (Amazon S3) was already correct.
"""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    subprocess.run(["bash", str(HERE / "run_pipeline.sh")], check=True,
                    env={"MODE": "data", **__import__("os").environ})


if __name__ == "__main__":
    sys.exit(main())

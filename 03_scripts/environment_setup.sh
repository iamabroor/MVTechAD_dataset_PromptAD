#!/usr/bin/env bash
# Wrapper: environment + code setup stage of run_pipeline.sh (MODE=setup).
# Creates the uv venv, clones FuNz-0/PromptAD at a pinned commit, installs pinned
# dependencies, and applies the three patches documented in
# ../02_src/source_code/README.md.
set -euo pipefail
cd "$(dirname "$0")"
MODE=setup bash run_pipeline.sh

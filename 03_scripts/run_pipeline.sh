#!/usr/bin/env bash
# PromptAD (CVPR 2024) replication pipeline for the RACE g5.2xlarge workspace (1x A10G).
# Resumable: every step and every training run is skipped if already done.
#
# Usage:
#   MODE=setup   bash ~/run_promptad.sh   # venv + code + patches
#   MODE=data    bash ~/run_promptad.sh   # MVTec AD + VisA download/prep
#   MODE=smoke   bash ~/run_promptad.sh   # 1 run (mvtec bottle, 1-shot, cls) to time it
#   MODE=run     bash ~/run_promptad.sh   # all runs (cls + seg, mvtec + visa, 1/2/4-shot)
#   MODE=summary bash ~/run_promptad.sh   # results table vs paper
#   MODE=all     bash ~/run_promptad.sh   # setup -> data -> run -> summary
#
# Optional env vars:
#   SEED=111            torch seed (code default 111)
#   TASKS="cls seg"     which task(s) to run
#   DATASETS="mvtec visa"
#   SHOTS="1 2 4"
#   SORT_TRAIN=1        sort the 'good' train file list before applying the k-shot indices (see notes)
#
# Run inside tmux so training survives a DCV disconnect:  tmux new -s promptad

set -euo pipefail

MODE="${MODE:-all}"
SEED="${SEED:-111}"
TASKS="${TASKS:-cls seg}"
DATASETS="${DATASETS:-mvtec visa}"
SHOTS="${SHOTS:-1 2 4}"
SORT_TRAIN="${SORT_TRAIN:-1}"

ROOT="$HOME/promptad"
REPO="$ROOT/PromptAD"
VENV="$ROOT/.venv"
DATA="$HOME/data"
MVTEC_DIR="$DATA/mvtec_anomaly_detection"
VISA_RAW="$DATA/VisA_20220922"
VISA_DIR="$DATA/VisA_pytorch/1cls"
OUT="$ROOT/output"
LOGS="$OUT/logs"

MVTEC_URL="https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938113-1629952094/mvtec_anomaly_detection.tar.xz"
VISA_URL="https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar"
VISA_SPLIT_URL="https://raw.githubusercontent.com/amazon-science/spot-diff/main/split_csv/1cls.csv"

MVTEC_CLASSES="carpet grid leather tile wood bottle cable capsule hazelnut metal_nut pill screw toothbrush transistor zipper"
VISA_CLASSES="candle capsules cashew chewinggum fryum macaroni1 macaroni2 pcb1 pcb2 pcb3 pcb4 pipe_fryum"

mkdir -p "$ROOT" "$DATA" "$OUT" "$LOGS"
ts() { TZ=Asia/Kuala_Lumpur date '+%Y-%m-%d %H:%M:%S MYT'; }
say() { echo "[$(ts)] $*" | tee -a "$OUT/pipeline.log"; }

# ---------------------------------------------------------------- setup
do_setup() {
  if [[ -f "$ROOT/.setup_done" ]]; then say "setup: already done"; return; fi
  say "setup: start"
  command -v uv >/dev/null || pip install --user uv
  export PATH="$HOME/.local/bin:$PATH"

  [[ -d "$REPO" ]] || git clone https://github.com/FuNz-0/PromptAD.git "$REPO"
  git -C "$REPO" rev-parse HEAD > "$OUT/promptad_commit.txt"
  say "setup: PromptAD commit $(cat "$OUT/promptad_commit.txt")"

  [[ -d "$VENV" ]] || uv venv --python 3.10 "$VENV"
  # shellcheck disable=SC1091
  source "$VENV/bin/activate"

  # The repo's install.sh has a typo on the torch line ("pip3 pip3 install ..."), so torch is
  # installed explicitly here. Same torch/CUDA as the SALAD run (A10G, driver supports cu121).
  # numpy pinned <2 (older code paths); opencv pinned to a numpy-1.x-compatible wheel.
  cat > "$ROOT/constraints.txt" <<'EOF'
torch==2.5.1
torchvision==0.20.1
torchaudio==2.5.1
numpy==1.26.4
opencv-python==4.10.0.84
EOF
  uv pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
      --index-url https://download.pytorch.org/whl/cu121
  # Remaining packages from install.sh (setuptools==59.5.0 pin dropped: not needed on py3.10 + uv)
  uv pip install -c "$ROOT/constraints.txt" \
      "diffusers[torch]" opencv-python pycocotools matplotlib onnxruntime onnx ipykernel \
      transformers addict yapf timm loguru tqdm scikit-image scikit-learn pandas \
      tensorboard seaborn open_clip_torch SciencePlots numpy==1.26.4

  python - <<'EOF'
import torch, open_clip, numpy
print("torch", torch.__version__, "cuda", torch.version.cuda, "gpu", torch.cuda.get_device_name(0))
print("open_clip", open_clip.__version__, "numpy", numpy.__version__)
EOF

  patch_code
  touch "$ROOT/.setup_done"
  say "setup: done"
}

# ---------------------------------------------------------------- code patches
patch_code() {
  say "patch: dataset paths, train-list sorting (SORT_TRAIN=$SORT_TRAIN), last-epoch logging"
  python - "$REPO" "$MVTEC_DIR" "$VISA_DIR" "$SORT_TRAIN" <<'EOF'
import re, sys, pathlib
repo, mvtec_dir, visa_dir, sort_train = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4] == "1"
repo = pathlib.Path(repo)

def edit(path, fn):
    p = repo / path
    src = p.read_text()
    if not (repo / (path + ".orig")).exists():
        (repo / (path + ".orig")).write_text(src)   # keep pristine copy for the report diff
    else:
        src = (repo / (path + ".orig")).read_text()  # re-patch from pristine (idempotent)
    new = fn(src)
    p.write_text(new)

def ds_patch(var, path_value):
    def fn(src):
        src = re.sub(rf"^{var}\s*=.*$", f"{var} = '{path_value}'", src, flags=re.M)
        if sort_train:
            # 'good' images are taken from an unsorted glob, so the k-shot indices in
            # seeds_*/selected_samples_per_run.txt pick filesystem-dependent files.
            src = re.sub(r"img_paths = glob\.glob\((.*?)\)\n(\s+)img_tot_paths\.extend",
                         r"img_paths = sorted(glob.glob(\1))\n\2img_tot_paths.extend", src)
        return src
    return fn

edit("datasets/mvtec.py", ds_patch("MVTEC2D_DIR", mvtec_dir))
edit("datasets/visa.py", ds_patch("VISA_DIR", visa_dir))

# The code evaluates on the TEST set after every epoch and keeps the best epoch.
# Also print the last-epoch result so both can be reported.
def last_epoch(src):
    return src.replace(
        "\n    return best_result_dict\n",
        "\n    print('LAST_EPOCH_METRICS', {k: float(v) for k, v in result_dict.items()"
        " if isinstance(v, (int, float))}, flush=True)\n    return best_result_dict\n", 1)

edit("train_cls.py", last_epoch)
edit("train_seg.py", last_epoch)
print("patched OK")
EOF
  ( cd "$REPO" && for f in datasets/mvtec.py datasets/visa.py train_cls.py train_seg.py; do
      diff -u "$f.orig" "$f" || true; done ) > "$OUT/code_patches.diff"
}

# ---------------------------------------------------------------- data
do_data() {
  source "$VENV/bin/activate"
  if [[ ! -f "$DATA/.mvtec_done" ]]; then
    say "data: MVTec AD download (~4.9 GB)"
    wget -c -O "$DATA/mvtec_anomaly_detection.tar.xz" "$MVTEC_URL"
    mkdir -p "$MVTEC_DIR"
    tar -xf "$DATA/mvtec_anomaly_detection.tar.xz" -C "$MVTEC_DIR"
    rm -f "$DATA/mvtec_anomaly_detection.tar.xz"
    touch "$DATA/.mvtec_done"; say "data: MVTec AD ready"
  fi
  if [[ ! -f "$DATA/.visa_done" ]]; then
    say "data: VisA download"
    wget -c -O "$DATA/VisA_20220922.tar" "$VISA_URL"
    mkdir -p "$VISA_RAW"
    tar -xf "$DATA/VisA_20220922.tar" -C "$VISA_RAW"
    if [[ ! -f "$VISA_RAW/split_csv/1cls.csv" ]]; then
      mkdir -p "$VISA_RAW/split_csv"; wget -O "$VISA_RAW/split_csv/1cls.csv" "$VISA_SPLIT_URL"
    fi
    say "data: VisA 1cls preprocessing"
    ( cd "$REPO" && python datasets/prepare_visa_public.py --split-type 1cls \
        --data-folder "$VISA_RAW" --save-folder "$DATA/VisA_pytorch" \
        --split-file "$VISA_RAW/split_csv/1cls.csv" )
    rm -f "$DATA/VisA_20220922.tar"
    touch "$DATA/.visa_done"; say "data: VisA ready"
  fi
  # Record exactly which training images the k-shot indices select (for the report)
  ( cd "$REPO" && python - "$OUT/selected_train_images.txt" <<'EOF'
import sys, os
from datasets.mvtec import load_mvtec, mvtec_classes
from datasets.visa import load_visa, visa_classes
with open(sys.argv[1], "w") as f:
    for name, loader, classes in [("mvtec", load_mvtec, mvtec_classes), ("visa", load_visa, visa_classes)]:
        for c in classes:
            for k in (1, 2, 4):
                (paths, *_), _ = loader(c, k)
                f.write(f"{name}\t{c}\t{k}-shot\t" + " ".join(os.path.basename(p) for p in paths) + "\n")
print("wrote", sys.argv[1])
EOF
  )
  say "data: counts  mvtec=$(ls "$MVTEC_DIR" | wc -l) classes, visa=$(ls "$VISA_DIR" | wc -l) classes"
}

# ---------------------------------------------------------------- one training run
run_one() {  # task dataset shot class
  local task=$1 ds=$2 shot=$3 cls=$4
  local log="$LOGS/${task}_${ds}_k${shot}_${cls}_s${SEED}.log"
  if [[ -f "$log" ]] && grep -q "===========================" "$log"; then return; fi
  local extra=""; [[ "$task" == "seg" ]] && extra="--vis False"
  say "run: $task $ds ${shot}-shot $cls (seed $SEED)"
  local t0=$SECONDS
  ( cd "$REPO" && python -u "train_${task}.py" --dataset "$ds" --class_name "$cls" \
      --k-shot "$shot" --seed "$SEED" $extra ) > "$log" 2>&1 || { say "run: FAILED $log"; return 1; }
  say "run: done $task $ds ${shot}-shot $cls in $(( (SECONDS - t0) / 60 )) min — $(grep -o 'AUROC:[0-9.]*' "$log" | tail -1)"
}

do_smoke() { source "$VENV/bin/activate"; run_one cls mvtec 1 bottle; }

do_run() {
  source "$VENV/bin/activate"
  for task in $TASKS; do for shot in $SHOTS; do for ds in $DATASETS; do
    local classes; [[ "$ds" == "mvtec" ]] && classes=$MVTEC_CLASSES || classes=$VISA_CLASSES
    for cls in $classes; do run_one "$task" "$ds" "$shot" "$cls" || true; done
  done; done; done
  say "run: ALL DONE"
}

do_summary() {
  source "$VENV/bin/activate"
  python "$HOME/promptad_summary.py" --logs "$LOGS" --out "$OUT"
}

case "$MODE" in
  setup) do_setup ;;
  data) do_data ;;
  smoke) do_smoke ;;
  run) do_run ;;
  summary) do_summary ;;
  all) do_setup; do_data; do_run; do_summary ;;
  *) echo "unknown MODE=$MODE"; exit 1 ;;
esac

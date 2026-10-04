# Environment setup

## Hardware used

- AWS `g5.2xlarge` — 1× NVIDIA A10G (24 GB), 8 vCPU, Ubuntu 22.04.
- Region: us-east-1. On-demand price at time of use: ~$1.21/h.
- Total wall-clock for the fast-mode sweep (162 runs) + diagnostics: ≈3 hours (~$4).

## Software

- Python 3.10, managed with [`uv`](https://github.com/astral-sh/uv) (`uv venv --python 3.10`).
- PyTorch 2.5.1 + CUDA 12.1 (matches the A10G driver on this instance type):

  ```bash
  uv pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
      --index-url https://download.pytorch.org/whl/cu121
  ```

- Remaining dependencies, pinned against the torch/numpy/opencv versions above (see
  [`requirements.txt`](requirements.txt)):

  ```bash
  uv pip install -c constraints.txt -r requirements.txt
  ```

  `numpy` is pinned below 2.0 and `opencv-python` to a numpy-1.x-compatible build — the released
  code was written against numpy 1.x APIs.

## Known install-script issue

The upstream repo's `install.sh` has a typo on the torch install line
(`pip3 pip3 install ...`), so torch is installed explicitly as shown above rather than by running
that script. The `setuptools==59.5.0` pin in the original script was also dropped — unnecessary
on Python 3.10 with `uv`.

## Data

- **MVTec AD** (~4.9 GB): the commonly used `mydrive.ch` download link
  (`.../420938113-1629952094/mvtec_anomaly_detection.tar.xz`) returns 404 as of October 2026.
  This replication uses the current working link from the
  [anomalib](https://github.com/openvinotoolkit/anomalib) project's
  `src/anomalib/data/datamodules/image/mvtecad.py` data module instead:
  `https://www.mydrive.ch/shares/150996/b52ecdcbf521176e9db9c731f2304b27/download/420938113-1629960298/mvtec_anomaly_detection.tar.xz`
- **VisA**: downloaded directly from Amazon S3
  (`amazon-visual-anomaly.s3.us-west-2.amazonaws.com`), already correct in the released code;
  prepared into the 1-class split with the released `datasets/prepare_visa_public.py` and the
  `spot-diff` split CSV.

## Verifying the install

```bash
python - <<'EOF'
import torch, open_clip, numpy
print("torch", torch.__version__, "cuda", torch.version.cuda, "gpu", torch.cuda.get_device_name(0))
print("open_clip", open_clip.__version__, "numpy", numpy.__version__)
EOF
```

Expected: `torch 2.5.1`, `cuda 12.1`, GPU name `NVIDIA A10G`.

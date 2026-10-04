# Source code

This replication runs the authors' released code unmodified except for the small, documented
patches below — it is not vendored into this repository (to avoid duplicating/relicensing
upstream code and to make the diff against upstream unambiguous).

- **Upstream repository:** https://github.com/FuNz-0/PromptAD
- **Exact commit used:** see [`commit.txt`](commit.txt) — recorded automatically by
  `03_scripts/preprocessing.py` / the setup step at clone time.
  *(Pending: paste the contents of `~/promptad/output/promptad_commit.txt` from the run here.)*
- **License:** Unlicense (public domain), per upstream `LICENSE`.

## Reproducing the checkout

```bash
git clone https://github.com/FuNz-0/PromptAD.git
cd PromptAD
git checkout <commit from commit.txt>
git apply ../code_patches.diff
```

## Patches applied

Full unified diff: [`code_patches.diff`](code_patches.diff) *(pending — copy from
`~/promptad/output/code_patches.diff` on the training workspace)*. Summary:

1. **Dataset paths** — `MVTEC2D_DIR` / `VISA_DIR` in `datasets/mvtec.py` / `datasets/visa.py`
   pointed at local data directories.
2. **Deterministic k-shot file selection** (`SORT_TRAIN=1`) — the 'good' training image list is
   built from an unsorted `glob.glob`, so the shipped k-shot index files
   (`seeds_*/selected_samples_per_run.txt`) pick filesystem-order-dependent images. We sort the
   glob result before indexing so the exact training images are reproducible and recorded (see
   `selected_train_images.txt`). Effect of this choice on results: ≈0.7 AUROC points (see
   `07_replication_notes/implementation_notes.md`, Diagnostic A).
3. **Evaluation gate + extra logging** — an `EVAL_EPOCHS` environment variable controls which
   training epochs trigger a test-set evaluation (the released code otherwise evaluates after
   *every* epoch, which is needed for the paper's best-epoch protocol but wasteful for a
   fixed-epoch run). Per-epoch and last-epoch metrics are also printed to the log for later
   parsing. **Training itself is never modified** — only which epochs get evaluated/logged.
4. A per-image pixel-AUROC CSV is written during pixel-level visualisation runs
   (`utils/visualization.py`), to allow choosing representative qualitative examples
   programmatically instead of by hand.

None of the patches change the model, the loss functions, or the training procedure.

## Known upstream issue (not patched)

`train_cls.py` / `train_seg.py` convert training images from BGR to RGB
(`cv2.cvtColor(..., cv2.COLOR_BGR2RGB)`) before encoding, but feed test images to CLIP directly
from `cv2.imread` (BGR) without the same conversion. This is a pre-existing bug in the released
code, left unpatched here so results stay comparable to what anyone running the released code
out-of-the-box would get. See `07_replication_notes/implementation_notes.md`.

#!/usr/bin/env python3
"""Qualitative figures for the PromptAD replication (paper Fig. 3 / Fig. 8 style).

Input : <vis-root>/<dataset>/k_<shot>/imgs/  written by a VIS=1 pixel-level run:
          <class>-<defect>-<id>_ori.jpg       test image (400x400)
          <class>-<defect>-<id>_gt.jpg        image with ground truth painted red
          <class>-<defect>-<id>_PromptAD.jpg  PromptAD anomaly heatmap overlaid on the image
          per_image_<class>.csv               max score + per-image pixel AUROC
        The heatmaps are normalised per class over the whole test set (released-code behaviour),
        and come from the best pixel-level epoch that was evaluated.
Output: <out>/<dataset>_k<shot>/
          overview_best.png/.pdf    one anomalous example per class (highest per-image pixel AUROC)
          overview_random.png/.pdf  one random anomalous example per class
          per_class/<class>_best.png, <class>_random.png   1 normal + 5 anomalous examples
"""
import argparse, csv, random, re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

ROWS = [("ori", "Image"), ("gt", "Ground truth"), ("PromptAD", "PromptAD")]


def load_scores(img_dir, cls):
    p = img_dir / f"per_image_{cls}.csv"
    out = {}
    if p.exists():
        with open(p) as f:
            for r in csv.DictReader(f):
                a = r.get("pixel_auroc", "")
                out[r["name"]] = float(a) if a not in ("", None) else None
    return out


def examples(img_dir):
    """{class: {"good": [names], "anomalous": [names]}} from the *_ori.jpg files."""
    by_cls = {}
    for p in sorted(img_dir.glob("*_ori.jpg")):
        name = p.name[: -len("_ori.jpg")]
        if not all((img_dir / f"{name}_{k}.jpg").exists() for k, _ in ROWS):
            continue
        cls, defect = name.split("-")[0], name.split("-")[1]
        d = by_cls.setdefault(cls, {"good": [], "anomalous": []})
        d["good" if defect == "good" else "anomalous"].append(name)
    return by_cls


def pick(names, scores, n, mode, rng):
    if not names:
        return []
    if mode == "random":
        return rng.sample(names, min(n, len(names)))
    # best: highest per-image pixel AUROC, preferring different defect types
    ranked = sorted(names, key=lambda x: (scores.get(x) is not None, scores.get(x) or 0), reverse=True)
    chosen, seen = [], set()
    for x in ranked:
        t = x.split("-")[1]
        if t not in seen:
            chosen.append(x); seen.add(t)
        if len(chosen) == n:
            return chosen
    for x in ranked:
        if x not in chosen:
            chosen.append(x)
        if len(chosen) == n:
            break
    return chosen


def grid(img_dir, cols, col_titles, out_base, title):
    n = len(cols)
    fig, axes = plt.subplots(len(ROWS), n, figsize=(1.75 * n + 0.9, 1.75 * len(ROWS) + 0.6), squeeze=False)
    for j, name in enumerate(cols):
        for i, (key, label) in enumerate(ROWS):
            ax = axes[i][j]
            ax.imshow(Image.open(img_dir / f"{name}_{key}.jpg").convert("RGB"))
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if i == 0:
                ax.set_title(col_titles[j], fontsize=8)
            if j == 0:
                ax.set_ylabel(label, fontsize=9)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".png"), dpi=200, bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def label(name, scores):
    cls, defect = name.split("-")[0], name.split("-")[1]
    a = scores.get(name)
    return f"{cls}\n{defect}" + (f"\nAUROC {a * 100:.1f}" if isinstance(a, float) else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vis-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    root, out = Path(a.vis_root), Path(a.out)
    made = 0
    for img_dir in sorted(root.glob("*/k_*/imgs")):
        ds, shot = img_dir.parent.parent.name, img_dir.parent.name.split("_")[1]
        by_cls = examples(img_dir)
        if not by_cls:
            continue
        dest = out / f"{ds}_k{shot}"
        scores = {c: load_scores(img_dir, c) for c in by_cls}
        for mode in ("best", "random"):
            rng = random.Random(a.seed)
            cols = [pick(by_cls[c]["anomalous"], scores[c], 1, mode, rng) for c in sorted(by_cls)]
            cols = [c[0] for c in cols if c]
            grid(img_dir, cols, [label(c, scores[c.split("-")[0]]) for c in cols], dest / f"overview_{mode}",
                 f"PromptAD (replication) – {ds}, {shot}-shot – {'best-localised' if mode == 'best' else 'random'} anomaly per class")
            for c in sorted(by_cls):
                rng = random.Random(f"{a.seed}-{c}")
                good = rng.sample(by_cls[c]["good"], 1) if by_cls[c]["good"] else []
                anom = pick(by_cls[c]["anomalous"], scores[c], 5, mode, rng)
                sel = good + anom
                grid(img_dir, sel, [label(x, scores[c]) for x in sel], dest / "per_class" / f"{c}_{mode}",
                     f"PromptAD – {ds} {c}, {shot}-shot ({mode})")
                made += 1
        print(f"{ds} {shot}-shot: {len(by_cls)} classes -> {dest}")
    print(f"done: {made} per-class figures + overviews in {out}" if made else f"no VIS images found under {root}")


if __name__ == "__main__":
    main()

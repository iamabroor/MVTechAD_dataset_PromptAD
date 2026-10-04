#!/usr/bin/env python3
"""Paper-style results for the PromptAD replication.

Makes, in <results>/paper_style/:
  table1_image_auroc.csv/.png   like paper Table 1 (image AUROC, MVTec / VisA, 1/2/4-shot)
  table4_pixel_auroc.csv/.png   like paper Table 4 (pixel AUROC)
  fig3_qualitative_<best|random>.png/.pdf
        like paper Fig. 3: 1-shot, MVTec block on top, VisA block below;
        rows = Image / Ground truth / PromptAD (ours), one column per class
  fig8_mvtec_<best|random>.png/.pdf, fig8_visa_<best|random>.png/.pdf
        like supplementary Fig. 8: all classes of one dataset, 1-shot
'best'   = per class, the anomalous test image with the highest per-image pixel AUROC
'random' = per class, a random anomalous test image (seed 0) – the unbiased view
Usage: python promptad_paper_figures.py [results_dir]   (default: newest ~/promptad/output/fast_e1-5-*)
"""
import csv, glob, os, random, statistics as st, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

PAPER = {  # Tables 1 and 4 of the paper (mean ± std over 5 seeds)
    "cls": {"mvtec": ["94.6±1.7", "95.7±1.5", "96.6±0.9"], "visa": ["86.9±2.3", "88.3±2.0", "89.1±1.7"]},
    "seg": {"mvtec": ["95.9±0.5", "96.2±0.3", "96.5±0.2"], "visa": ["96.7±0.4", "97.1±0.3", "97.4±0.3"]},
}
WINCLIP = {  # WinCLIP+ row from the same tables, for context
    "cls": {"mvtec": ["93.1±2.0", "94.4±1.3", "95.2±1.3"], "visa": ["83.8±4.0", "84.6±2.4", "87.3±1.8"]},
    "seg": {"mvtec": ["95.2±0.5", "96.0±0.3", "96.2±0.3"], "visa": ["96.4±0.4", "96.8±0.3", "97.2±0.2"]},
}
FIG3 = {"mvtec": ["carpet", "grid", "leather", "bottle", "cable", "hazelnut", "metal_nut", "screw"],
        "visa": ["candle", "capsules", "cashew", "chewinggum", "fryum", "macaroni1", "pcb1", "pipe_fryum"]}
ROWS = [("ori", "Image"), ("gt", "Ground truth"), ("PromptAD", "PromptAD\n(ours)")]
NICE = {"mvtec": "MVTec AD", "visa": "VisA"}


# ---------------------------------------------------------------- tables
def tables(R, out):
    rows = list(csv.DictReader(open(R / "per_class.csv")))
    for task, name, title in (("cls", "table1_image_auroc", "Table 1 – image-level AUROC (%)"),
                              ("seg", "table4_pixel_auroc", "Table 4 – pixel-level AUROC (%)")):
        cols = [f"{NICE[d]} {k}-shot" for d in ("mvtec", "visa") for k in (1, 2, 4)]
        lines = [["WinCLIP+ (paper)"] + [WINCLIP[task][d][i] for d in ("mvtec", "visa") for i in range(3)],
                 ["PromptAD (paper)"] + [PAPER[task][d][i] for d in ("mvtec", "visa") for i in range(3)]]
        for key, lab in (("best", "PromptAD (ours) – best epoch*"),
                         ("last", "PromptAD (ours) – epoch 100")):
            r = [lab]
            for d in ("mvtec", "visa"):
                for k in (1, 2, 4):
                    g = [float(x[key]) for x in rows if x["task"] == task and x["dataset"] == d and int(x["shot"]) == k]
                    r.append(f"{st.mean(g):.1f}" if g else "–")
            lines.append(r)
        with open(out / f"{name}.csv", "w", newline="") as f:
            w = csv.writer(f); w.writerow(["Method"] + cols); w.writerows(lines)
        fig, ax = plt.subplots(figsize=(13, 1.0 + 0.42 * len(lines)))
        ax.axis("off")
        t = ax.table(cellText=lines, colLabels=["Method"] + cols, loc="center", cellLoc="center")
        t.auto_set_font_size(False); t.set_fontsize(8.5); t.scale(1, 1.5)
        for (i, j), c in t.get_celld().items():
            if j == 0:
                c.set_text_props(ha="left"); c._loc = "left"; c.set_width(0.25)
            else:
                c.set_width(0.115)
            if i == 0:
                c.set_text_props(weight="bold"); c.set_facecolor("#eeeeee")
            if i in (3, 4):
                c.set_facecolor("#eef6ee" if i == 3 else "#fbeeee")
        ax.set_title(f"{title} – paper vs this replication", fontsize=10)
        fig.text(0.02, 0.02, "Paper: mean ± std over 5 seeds. Ours: seed 111, 1× A10G. *best epoch chosen on the test set "
                 "(evaluated at epochs 1, 5, …, 100), as the released code does.", fontsize=7)
        fig.savefig(out / f"{name}.png", dpi=200, bbox_inches="tight"); plt.close(fig)
        print("table ->", out / f"{name}.csv")


# ---------------------------------------------------------------- figures
def load(img_dir):
    names, auc = {}, {}
    for p in sorted(img_dir.glob("*_ori.jpg")):
        n = p.name[:-8]
        if all((img_dir / f"{n}_{k}.jpg").exists() for k, _ in ROWS):
            c, d = n.split("-")[0], n.split("-")[1]
            names.setdefault(c, {"good": [], "anom": []})["good" if d == "good" else "anom"].append(n)
    for p in img_dir.glob("per_image_*.csv"):
        for r in csv.DictReader(open(p)):
            if r.get("pixel_auroc"):
                auc[r["name"]] = float(r["pixel_auroc"])
    return names, auc


def choose(names, auc, cls, mode):
    a = names.get(cls, {}).get("anom", [])
    if not a:
        return None
    if mode == "random":
        return random.Random(f"0-{cls}").choice(a)
    return max(a, key=lambda n: auc.get(n, -1))


def block(fig, gs_row, img_dir, picks, title, first_block):
    axes = []
    for j, n in enumerate(picks):
        for i, (key, lab) in enumerate(ROWS):
            ax = fig.add_subplot(gs_row[i, j]); axes.append(ax)
            ax.imshow(Image.open(img_dir / f"{n}_{key}.jpg").convert("RGB"))
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_linewidth(0.4)
            if i == 0:
                ax.set_title(n.split("-")[0].replace("_", " "), fontsize=8)
            if j == 0:
                ax.set_ylabel(lab, fontsize=8.5)
    return axes


def figure(blocks, outbase, suptitle):
    ncol = max(len(p) for _, _, p, _ in blocks)
    nrow = len(ROWS) * len(blocks)
    fig = plt.figure(figsize=(1.45 * ncol + 0.8, 1.45 * nrow + 0.9 * len(blocks)))
    outer = fig.add_gridspec(len(blocks), 1, hspace=0.18)
    for b, (img_dir, ds, picks, _) in enumerate(blocks):
        inner = outer[b].subgridspec(len(ROWS), ncol, wspace=0.03, hspace=0.03)
        block(fig, inner, img_dir, picks, NICE[ds], b == 0)
        fig.text(0.5, outer[b].get_position(fig).y1 + 0.022, f"{NICE[ds]}", ha="center", fontsize=10, weight="bold")
    fig.text(0.5, outer[0].get_position(fig).y1 + 0.06, suptitle, ha="center", fontsize=10.5)
    for ext in ("png", "pdf"):
        fig.savefig(f"{outbase}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("figure ->", f"{outbase}.png")


def figures(R, out):
    data = {}
    for ds in ("mvtec", "visa"):
        d = R / "vis" / ds / "k_1" / "imgs"
        if d.exists():
            data[ds] = (d, *load(d))
    if not data:
        print("no 1-shot VIS images found under", R / "vis"); return
    for mode in ("best", "random"):
        what = "best-localised" if mode == "best" else "random"
        bl = []
        for ds in ("mvtec", "visa"):
            if ds in data:
                d, names, auc = data[ds]
                picks = [p for p in (choose(names, auc, c, mode) for c in FIG3[ds]) if p]
                if picks:
                    bl.append((d, ds, picks, auc))
        figure(bl, out / f"fig3_qualitative_{mode}",
               f"PromptAD replication – 1-shot pixel-level anomaly detection (paper Fig. 3 style; {what} example per class)")
        for ds in data:
            d, names, auc = data[ds]
            picks = [p for p in (choose(names, auc, c, mode) for c in sorted(names)) if p]
            figure([(d, ds, picks, auc)], out / f"fig8_{ds}_{mode}",
                   f"PromptAD replication – {NICE[ds]}, all classes, 1-shot (supplementary Fig. 8 style; {what})")


def main():
    R = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(sorted(glob.glob(os.path.expanduser("~/promptad/output/fast_e1-5-*")))[-1])
    out = R / "paper_style"; out.mkdir(exist_ok=True)
    print("results:", R)
    tables(R, out)
    figures(R, out)
    print("done ->", out)


if __name__ == "__main__":
    main()

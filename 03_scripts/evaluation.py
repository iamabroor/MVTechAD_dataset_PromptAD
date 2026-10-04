#!/usr/bin/env python3
"""Collect PromptAD replication results from run logs and compare with the paper.

Reads logs named  <task>_<dataset>_k<shot>_<class>_s<seed>.log  written by run_promptad.sh.
"best"  = best epoch selected on the test set (what the released code reports)
"last"  = final epoch (epoch 100), no test-set selection
Writes per_class.csv and summary.md to --out.
"""
import argparse, ast, csv, re, statistics
from collections import defaultdict
from pathlib import Path

# Paper, Table 1 (image AUROC) and Table 4 (pixel AUROC): mean over 5 seeds
PAPER = {
    ("cls", "mvtec"): {1: 94.6, 2: 95.7, 4: 96.6},
    ("cls", "visa"):  {1: 86.9, 2: 88.3, 4: 89.1},
    ("seg", "mvtec"): {1: 95.9, 2: 96.2, 4: 96.5},
    ("seg", "visa"):  {1: 96.7, 2: 97.1, 4: 97.4},
}
N_CLASSES = {"mvtec": 15, "visa": 12}
NAME = re.compile(r"(cls|seg)_(mvtec|visa)_k(\d)_(.+)_s(\d+)\.log$")


def pct(x):
    return x * 100 if x is not None and x <= 1.0 else x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    rows = []
    for log in sorted(Path(a.logs).glob("*.log")):
        m = NAME.search(log.name)
        if not m:
            continue
        task, ds, shot, cls, seed = m.group(1), m.group(2), int(m.group(3)), m.group(4), int(m.group(5))
        txt = log.read_text(errors="ignore")
        best = re.findall(r"(?:Image|Pixel)-AUROC:([0-9.]+)", txt)
        if not best:
            continue  # unfinished or failed run
        last = None
        lm = re.findall(r"LAST_EPOCH_METRICS (\{.*\})", txt)
        if lm:
            d = ast.literal_eval(lm[-1])
            last = d.get("i_roc" if task == "cls" else "p_roc")
        key = "i_roc" if task == "cls" else "p_roc"
        curve = [(int(e), ast.literal_eval(d).get(key)) for e, d in re.findall(r"EPOCH_METRICS (\d+) (\{.*\})", txt)]
        curve = [(e, v) for e, v in curve if v is not None]
        best_epoch = max(curve, key=lambda t: t[1])[0] if curve else None
        rows.append(dict(task=task, dataset=ds, shot=shot, cls=cls, seed=seed, best_epoch=best_epoch,
                         n_evals=len(curve),
                         best=round(pct(float(best[-1])), 2),
                         last=None if last is None else round(pct(last), 2)))

    out = Path(a.out)
    with open(out / "per_class.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["task", "dataset", "shot", "cls", "seed", "best", "last", "best_epoch", "n_evals"])
        w.writeheader(); w.writerows(rows)

    groups = defaultdict(list)
    for r in rows:
        groups[(r["task"], r["dataset"], r["shot"])].append(r)

    lines = ["# PromptAD replication – summary", "",
             "best = best epoch chosen on the test set (released-code behaviour); "
             "last = epoch 100. Paper = mean over 5 seeds.", "",
             "| Metric | Dataset | Shot | Classes done | Ours (best) | Ours (last) | Paper | Gap (best) |",
             "|---|---|---|---|---|---|---|---|"]
    for task in ("cls", "seg"):
        for ds in ("mvtec", "visa"):
            for shot in (1, 2, 4):
                g = groups.get((task, ds, shot), [])
                if not g:
                    continue
                b = statistics.mean(r["best"] for r in g)
                lasts = [r["last"] for r in g if r["last"] is not None]
                l = f"{statistics.mean(lasts):.2f}" if len(lasts) == len(g) else "–"
                p = PAPER[(task, ds)][shot]
                full = len(g) == N_CLASSES[ds]
                lines.append(f"| {'Image' if task == 'cls' else 'Pixel'} AUROC | {ds} | {shot} | "
                             f"{len(g)}/{N_CLASSES[ds]} | {b:.2f} | {l} | {p} | "
                             f"{(b - p):+.2f}{'' if full else ' (partial)'} |")
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\n{len(rows)} finished runs -> {out/'per_class.csv'}, {out/'summary.md'}")


if __name__ == "__main__":
    main()

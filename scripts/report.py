"""
Comparison table + business impact + figures from all evaluations
  -> results/summary.md, results/figures/*.png

  python scripts/report.py --order before baseline ours ours_no_fft ours_no_synthetic ours_no_degrade ours_ce_loss

Impact assumptions are arguments (state them when you quote numbers):
  --volume 1000000 checks, --fraud_rate 0.01 (1% are spoof attempts), --peak 50 checks/second
"""
import argparse
import csv
import json
import math
import random
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fas.augment import DEGRADATIONS, SYNTHETIC_ATTACKS, fft_target  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--order", nargs="*", default=["before", "baseline", "ours"])
ap.add_argument("--eval", default="results/eval")
ap.add_argument("--runs", default="results/runs")
ap.add_argument("--train", default="data/train")
ap.add_argument("--test", default="data/test")
ap.add_argument("--volume", type=float, default=1_000_000)
ap.add_argument("--fraud_rate", type=float, default=0.01)
ap.add_argument("--peak", type=float, default=50, help="peak checks per second")
ap.add_argument("--deploy_ms", type=float, help="latency per face for impact (default: ONNX from results/deploy)")
ap.add_argument("--out", default="results")
a = ap.parse_args()

found = {p.parent.name: p.parent for p in Path(a.eval).glob("*/metrics.json")}
tags = [t for t in a.order if t in found] + sorted(set(found) - set(a.order))
if not tags:
    raise SystemExit(f"No evaluations in {a.eval}")
M = {t: json.loads((found[t] / "metrics.json").read_text()) for t in tags}
figs = Path(a.out) / "figures"
figs.mkdir(parents=True, exist_ok=True)
pct = "{:.2%}".format

# ---- comparison table
lines = ["## Results", "", "| Model | Fraud accepted (APCER) | Customers blocked (BPCER) | ACER | AUC | Blocked @ 1% fraud "
         "| Bank threshold: fraud / blocked | ms/face (PyTorch CPU) | Params |", "|" + "---|" * 9]
for t in tags:
    m, b = M[t], M[t].get("bank_threshold")
    lines.append(f"| {t} | {pct(m['default_rule']['APCER'])} | {pct(m['default_rule']['BPCER'])} | "
                 f"{pct(m['default_rule']['ACER'])} | {m['threshold_free']['AUC']:.4f} | "
                 f"{pct(m['threshold_free']['BPCER@APCER=0.01'])} | {pct(b['APCER']) + ' / ' + pct(b['BPCER']) if b else '-'} | "
                 f"{m['speed_cpu']['ms_median']:.1f} | {m['params']:,} |")

# ---- business impact
deploy = Path(a.out) / "deploy" / "ours.json"
ms = a.deploy_ms or (json.loads(deploy.read_text())["onnx_fp32"]["ms_median"] if deploy.exists() else None)
attacks, genuine = a.volume * a.fraud_rate, a.volume * (1 - a.fraud_rate)
lines += ["", f"## Impact per {a.volume:,.0f} checks (assumed: {a.fraud_rate:.1%} fraud attempts, "
          f"threshold 0.5, peak {a.peak:g} checks/s, {'%.1f ms/face' % ms if ms else 'PyTorch CPU speed'})", "",
          "| Model | Frauds let through | Genuine customers blocked | CPU hours | Workers at peak |", "|---|---|---|---|---|"]
base = None
for t in tags:
    m = M[t]
    p = m["at_0.5"]  # fixed threshold: the validation-tuned one did not transfer to unseen people
    fr, bl, lat = p["APCER"] * attacks, p["BPCER"] * genuine, ms or m["speed_cpu"]["ms_median"]
    base = base or (fr, bl)
    lines.append(f"| {t} | {fr:,.0f} ({fr - base[0]:+,.0f}) | {bl:,.0f} ({bl - base[1]:+,.0f}) | "
                 f"{a.volume * lat / 3.6e6:.1f} | {math.ceil(a.peak * lat / 1000)} |")
# ---- deployment (written by scripts/export.py) and early exit
dep = {p.stem: json.loads(p.read_text()) for p in sorted((Path(a.out) / "deploy").glob("*.json"))}
if dep:
    lines += ["", "## Deployment (CPU, both models per face)", "",
              "| Model set | Format | ms/face | p95 ms | Size MB | ACER | AUC |", "|---|---|---|---|---|---|---|"]
    for name, rep in dep.items():
        for kind, r in rep.items():
            acc = f"{pct(r['ACER'])} | {r['AUC']:.4f}" if "AUC" in r else "- | -"
            lines.append(f"| {name} | {kind} | {r['ms_median']:.2f} | {r['ms_p95']:.2f} | {r['size_mb']:.2f} | {acc} |")
ee = [t for t in tags if "0.1" in M[t].get("early_exit", {})]
if ee:
    lines += ["", "## Early exit (2nd model only when the 1st is unsure)", "",
              "| Model | Faces needing 2nd model | ACER with early exit | ACER always both | ms/face (est. vs always both) |",
              "|---|---|---|---|---|"]
    for t in ee:
        e, m = M[t]["early_exit"]["0.1"], M[t]
        lines.append(f"| {t} | {e['second_model_rate']:.1%} | {pct(e['ACER'])} | {pct(m['at_0.5']['ACER'])} | "
                     f"{e['est_ms']:.1f} vs {m['speed_cpu']['ms_median']:.1f} |")
(Path(a.out) / "summary.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))

# ---- figure: fraud vs blocked trade-off
fig, ax = plt.subplots(figsize=(6, 5))
for t in tags:
    rows = list(csv.DictReader(open(found[t] / "scores.csv")))
    fpr, tpr, _ = roc_curve([int(r["label"]) for r in rows], [float(r["real_score"]) for r in rows])
    ax.plot(fpr, 1 - tpr, label=f"{t} (AUC {M[t]['threshold_free']['AUC']:.3f})")
ax.set(xscale="log", xlim=(1e-3, 1), xlabel="Fraud accepted (APCER)", ylabel="Real customers blocked (BPCER)",
       title="Lower-left is better")
ax.grid(True, which="both", alpha=0.3)
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(figs / "tradeoff.png", dpi=150)

# ---- figure: training curves
logs = sorted(Path(a.runs).glob("*/log_*.csv"))
if logs:
    fig, ax = plt.subplots(figsize=(7, 4))
    for p in logs:
        rows = list(csv.DictReader(open(p)))
        ax.plot([int(r["epoch"]) for r in rows], [100 * float(r["val_ACER"]) for r in rows],
                label=f"{p.parent.name} {p.stem[4:]}")
    ax.set(xlabel="epoch", ylabel="validation ACER %")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(figs / "training.png", dpi=150)


# ---- figure: what each fraud transform does (top) and its FFT spectrum (bottom)
def grid(tiles, size=150):
    return np.hstack([cv2.resize(t, (size, size), interpolation=cv2.INTER_NEAREST) for t in tiles])


reals = sorted(Path(a.train).glob("*/1/*.png"))
if reals:
    random.seed(1)
    np.random.seed(1)
    shown = ["jpeg", "low_light", "blur", "low_res"]
    names = ["original", *SYNTHETIC_ATTACKS, *shown]
    rows = []
    for p in random.sample(reals, min(3, len(reals))):
        img = cv2.imread(str(p))
        tiles = [img, *(f(img.copy()) for f in SYNTHETIC_ATTACKS.values()), *(DEGRADATIONS[k](img.copy()) for k in shown)]
        rows += [grid(tiles), grid([cv2.cvtColor((fft_target(t, 80, 80) * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
                                    for t in tiles])]
    head = np.full((28, rows[0].shape[1], 3), 255, np.uint8)
    for i, n in enumerate(names):
        cv2.putText(head, n, (i * 150 + 4, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
    cv2.imwrite(str(figs / "fraud_transforms.png"), np.vstack([head, *rows]))

# ---- figure: hardest cases of the last model (fakes it trusted, real faces it doubted)
patch = next(iter(sorted(Path(a.test).glob("*_80x80"))), None)
if patch:
    t = "ours" if "ours" in tags else tags[-1]
    rows = list(csv.DictReader(open(found[t] / "scores.csv")))
    strips = []
    for title, label, key in (("FAKE faces scored most real", "0", lambda r: -float(r["real_score"])),
                              ("REAL faces scored most fake", "1", lambda r: float(r["real_score"]))):
        tiles = []
        for r in sorted((r for r in rows if r["label"] == label), key=key)[:8]:
            img = cv2.resize(cv2.imread(str(patch / label / r["image"])), (140, 140))
            cv2.putText(img, f"{float(r['real_score']):.2f}", (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
            tiles.append(img)
        if tiles:
            s = np.hstack(tiles)
            head = np.full((26, s.shape[1], 3), 255, np.uint8)
            cv2.putText(head, title, (4, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
            strips.append(np.vstack([head, s]))
    if strips:
        w = max(s.shape[1] for s in strips)
        cv2.imwrite(str(figs / f"hardest_{t}.png"),
                    np.vstack([np.pad(s, ((0, 0), (0, w - s.shape[1]), (0, 0)), constant_values=255) for s in strips]))
print(f"figures -> {figs}")

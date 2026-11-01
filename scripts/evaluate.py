"""
Evaluate a model pair on the test crops -> results/eval/<tag>/metrics.json + scores.csv

  python scripts/evaluate.py --models pretrained/anti_spoof --tag before
  python scripts/evaluate.py --models results/runs/ours/models --tag ours

Reports: default rule (original code: argmax), score >= 0.5, bank threshold (chosen on validation
so <= 1% of fakes pass), AUC/EER, early exit, CPU latency, parameter count.
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fas import REAL  # noqa: E402
from fas.metrics import at_threshold, early_exit, threshold_for_apcer, threshold_free  # noqa: E402
from fas.model import load, to_tensor  # noqa: E402


@torch.no_grad()
def softmax_all(models, root):
    """Class probabilities (N x 3) from each model over root/<patch>/<label>/*.png."""
    first = Path(root) / models[0][1]["patch"]
    items = [(p.name, l) for l in (0, 1) for p in sorted((first / str(l)).glob("*.png"))]
    if not items:
        raise SystemExit(f"No crops under {first}")
    out = []
    for model, info in models:
        d = Path(root) / info["patch"]
        out.append(np.concatenate([
            torch.softmax(model(to_tensor([cv2.imread(str(d / str(l) / n)) for n, l in items[i:i + 256]])), 1).numpy()
            for i in range(0, len(items), 256)]))
    return [n for n, _ in items], np.array([l for _, l in items]), out


@torch.no_grad()
def latency(models, runs=200):
    xs = [torch.rand(1, 3, i["h"], i["w"]) * 255 for _, i in models]
    for _ in range(20):
        [m(x) for (m, _), x in zip(models, xs)]

    def timed(n_models):
        t = []
        for _ in range(runs):
            t0 = time.perf_counter()
            [m(x) for (m, _), x in list(zip(models, xs))[:n_models]]
            t.append((time.perf_counter() - t0) * 1000)
        return t

    first, both = timed(1), timed(len(models))
    return {"first_model_ms": float(np.median(first)), "ms_median": float(np.median(both)),
            "ms_p95": float(np.percentile(both, 95))}


ap = argparse.ArgumentParser()
ap.add_argument("--models", default="pretrained/anti_spoof")
ap.add_argument("--tag", default="before")
ap.add_argument("--test", default="data/test")
ap.add_argument("--val", default="data/val", help="for the bank threshold (skipped if missing)")
ap.add_argument("--target_apcer", type=float, default=0.01)
ap.add_argument("--out", default="results/eval")
a = ap.parse_args()

paths = sorted(Path(a.models).glob("*.pth"))
models = [load(p) for p in paths]
names, y, probs = softmax_all(models, a.test)
real = [p[:, REAL] for p in probs]
score = np.mean(real, axis=0)
spd = latency(models)
m = {
    "tag": a.tag,
    "test_images": {"real": int((y == 1).sum()), "fake": int((y == 0).sum())},
    "default_rule": at_threshold((np.sum(probs, 0).argmax(1) == REAL).astype(float), y),
    "at_0.5": at_threshold(score, y),
    "threshold_free": threshold_free(score, y),
    "speed_cpu": spd,
    "params": int(sum(p.numel() for mdl, _ in models for p in mdl.parameters())),
    "size_mb": round(sum(p.stat().st_size for p in paths) / 1e6, 2),
}
if (Path(a.val) / models[0][1]["patch"]).exists():
    _, vy, vp = softmax_all(models, a.val)
    t = threshold_for_apcer(np.mean([p[:, REAL] for p in vp], 0), vy, a.target_apcer)
    m["bank_threshold"] = at_threshold(score, y, t)
if len(models) > 1:
    m["early_exit"] = {}
    for band in (0.05, 0.1, 0.2):
        e = early_exit(real[0], score, y, band)
        e["est_ms"] = spd["first_model_ms"] + e["second_model_rate"] * (spd["ms_median"] - spd["first_model_ms"])
        m["early_exit"][f"{band:g}"] = e

out = Path(a.out) / a.tag
out.mkdir(parents=True, exist_ok=True)
(out / "metrics.json").write_text(json.dumps(m, indent=2))
with open(out / "scores.csv", "w", newline="") as f:
    csv.writer(f).writerows([["image", "label", "real_score"]] + [[n, l, round(s, 5)] for n, l, s in zip(names, y, score)])

d, tf = m["default_rule"], m["threshold_free"]
print(f"[{a.tag}] {len(y)} faces | fraud accepted {d['APCER']:.2%} | customers blocked {d['BPCER']:.2%} | "
      f"ACER {d['ACER']:.2%} | AUC {tf['AUC']:.4f} | blocked @1% fraud {tf['BPCER@APCER=0.01']:.2%}")
if "bank_threshold" in m:
    b = m["bank_threshold"]
    print(f"[{a.tag}] bank threshold {b['threshold']:.3f}: fraud accepted {b['APCER']:.2%} | blocked {b['BPCER']:.2%}")
e = m.get("early_exit", {}).get("0.1")
if e:
    print(f"[{a.tag}] early exit: 2nd model on {e['second_model_rate']:.1%} of faces | ACER {e['ACER']:.2%} "
          f"(always both: {m['at_0.5']['ACER']:.2%}) | ~{e['est_ms']:.1f} vs {spd['ms_median']:.1f} ms")

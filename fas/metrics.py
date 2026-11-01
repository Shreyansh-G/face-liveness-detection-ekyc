"""
Bank metrics (ISO/IEC 30107-3 names), on real-face scores where label 1 = real, 0 = fake.
  APCER  share of fakes accepted as real  -> fraud that gets through
  BPCER  share of real faces rejected      -> genuine customers blocked
  ACER   (APCER + BPCER) / 2
"""
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def at_threshold(scores, labels, t=0.5):
    s, y = np.asarray(scores), np.asarray(labels)
    apcer = float(np.mean(s[y == 0] >= t)) if (y == 0).any() else float("nan")
    bpcer = float(np.mean(s[y == 1] < t)) if (y == 1).any() else float("nan")
    return {"threshold": float(t), "accuracy": float(np.mean((s >= t) == (y == 1))),
            "APCER": apcer, "BPCER": bpcer, "ACER": (apcer + bpcer) / 2}


def threshold_for_apcer(scores, labels, target=0.01):
    """Lowest threshold that lets at most `target` share of fakes through (pick it on validation data)."""
    fake = np.sort(np.asarray(scores)[np.asarray(labels) == 0])
    if len(fake) == 0:
        return 0.5
    return float(min(fake[len(fake) - int(target * len(fake)) - 1] + 1e-6, 1.0))


def threshold_free(scores, labels):
    """AUC, EER, and customers blocked when fraud is held to 1% / 0.1%."""
    fpr, tpr, _ = roc_curve(labels, scores)  # fpr = APCER, 1 - tpr = BPCER
    fnr = 1 - tpr
    i = int(np.nanargmin(np.abs(fnr - fpr)))
    out = {"AUC": float(roc_auc_score(labels, scores)), "EER": float((fpr[i] + fnr[i]) / 2)}
    for target in (0.01, 0.001):
        ok = np.where(fpr <= target)[0]
        out[f"BPCER@APCER={target:g}"] = float(fnr[ok[-1]]) if len(ok) else 1.0
    return out


def early_exit(first, fused, labels, band, t=0.5):
    """Use the first model alone when it is confident (<= band or >= 1-band), else the average of both."""
    sure = (first <= band) | (first >= 1 - band)
    m = at_threshold(np.where(sure, first, fused), labels, t)
    m["second_model_rate"] = float(1 - sure.mean())
    return m

"""
Fine-tune the pretrained MiniFASNet pair (2.7x and 4x crops).

  baseline:  python scripts/train.py --aug standard --loss ce --out results/runs/baseline
  ours:      python scripts/train.py --aug fraud --loss live_vs_attack --out results/runs/ours
  ablation:  add --no_fft  or  --disable moire glare ...

Saves the best epoch (lowest validation ACER) as the plain backbone with the original
file name, so evaluation, export and the app load it directly.
"""
import argparse
import csv
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fas import PRETRAINED, REAL, parse_name  # noqa: E402
from fas.augment import FraudAwareAugment  # noqa: E402
from fas.data import CropDataset, list_images  # noqa: E402
from fas.metrics import at_threshold, threshold_free  # noqa: E402
from fas.model import FASModel, live_vs_attack_loss, load_weights  # noqa: E402


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    s, y = [], []
    for x, _, lab in loader:
        s.append(F.softmax(model(x.to(device))[0].float(), 1)[:, REAL].cpu().numpy())
        y.append(lab.numpy())
    s, y = np.concatenate(s), np.concatenate(y)
    return {**at_threshold(s, y), "AUC": threshold_free(s, y)["AUC"]}


def train(patch, a, device):
    fname = PRETRAINED[patch]
    info = parse_name(fname)
    tr, va = list_images(Path(a.train_root) / patch, a.limit, a.seed), list_images(Path(a.val_root) / patch)
    if not tr or not va:
        raise SystemExit(f"No crops for {patch}; run scripts/prepare_data.py first")
    fraud = FraudAwareAugment(a.p_synthetic, a.p_degrade, a.disable) if a.aug == "fraud" else None
    kw = dict(num_workers=a.workers, pin_memory=device.type == "cuda", persistent_workers=a.workers > 0)
    # RAM cache: decode crops once. Skipped on Windows with workers, where each worker would get its own copy.
    cache = not a.no_cache and (a.workers == 0 or sys.platform != "win32")
    train_dl = DataLoader(CropDataset(tr, True, fraud, not a.no_fft, cache=cache), a.batch_size, shuffle=True,
                          drop_last=True, **kw)
    val_dl = DataLoader(CropDataset(va), 512, **kw)

    model = FASModel(info["arch"], use_fft=not a.no_fft)
    load_weights(model.backbone, Path(a.pretrained) / fname)
    model.to(device)
    opt = torch.optim.SGD(model.parameters(), lr=a.lr, momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.epochs * len(train_dl), pct_start=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    out = Path(a.out)
    (out / "models").mkdir(parents=True, exist_ok=True)
    log_file = open(out / f"log_{patch}.csv", "w", newline="")
    log = csv.writer(log_file)
    log.writerow(["epoch", "loss", "train_acc", "val_ACER", "val_APCER", "val_BPCER", "val_AUC", "seconds"])
    v = validate(model, val_dl, device)
    best = (v["ACER"], -v["AUC"])
    torch.save(model.backbone.state_dict(), out / "models" / fname)
    print(f"[{patch}] pretrained: val ACER {v['ACER']:.2%} AUC {v['AUC']:.4f} | {len(tr)} train, {len(va)} val, {device}")

    for ep in range(1, a.epochs + 1):
        model.train()
        t0, n, tot, correct = time.time(), 0, 0.0, 0
        for x, ft, y in train_dl:
            x, ft, y = x.to(device), ft.to(device), y.to(device)
            with torch.autocast(device.type, torch.float16, enabled=device.type == "cuda"):
                logits, ft_pred = model(x)
            loss = live_vs_attack_loss(logits, y) if a.loss == "live_vs_attack" else F.cross_entropy(logits.float(), y)
            if ft_pred is not None:  # same 50/50 weighting as the original project
                loss = 0.5 * loss + 0.5 * F.mse_loss(ft_pred.float(), ft)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            n += len(y)
            tot += loss.item() * len(y)
            correct += ((F.softmax(logits.float(), 1)[:, REAL] >= 0.5).long() == y).sum().item()
        v = validate(model, val_dl, device)
        saved = (v["ACER"], -v["AUC"]) < best
        if saved:
            best = (v["ACER"], -v["AUC"])
            torch.save(model.backbone.state_dict(), out / "models" / fname)
        log.writerow([ep, tot / n, correct / n, v["ACER"], v["APCER"], v["BPCER"], v["AUC"], round(time.time() - t0, 1)])
        log_file.flush()
        print(f"[{patch}] epoch {ep}/{a.epochs} loss {tot / n:.4f} acc {correct / n:.2%} | "
              f"val ACER {v['ACER']:.2%} AUC {v['AUC']:.4f}{'  <- saved' if saved else ''}")
    log_file.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--aug", default="fraud", choices=["standard", "fraud"])
    ap.add_argument("--loss", default="live_vs_attack", choices=["ce", "live_vs_attack"])
    ap.add_argument("--no_fft", action="store_true", help="train without the FFT head")
    ap.add_argument("--disable", nargs="*", default=[], help="fraud transforms to switch off")
    ap.add_argument("--p_synthetic", type=float, default=0.2, help="chance a real face becomes a synthetic attack")
    ap.add_argument("--p_degrade", type=float, default=0.15, help="chance of each capture degradation")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--workers", type=int, default=os.cpu_count(), help="data loading processes (default: all CPUs)")
    ap.add_argument("--no_cache", action="store_true", help="read crops from disk every epoch instead of RAM")
    ap.add_argument("--limit", type=int, default=0, help="use only N training images (quick tests)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--patches", nargs="+", default=list(PRETRAINED), choices=list(PRETRAINED))
    ap.add_argument("--train_root", default="data/train")
    ap.add_argument("--val_root", default="data/val")
    ap.add_argument("--pretrained", default="pretrained/anti_spoof")
    ap.add_argument("--out", default="results/runs/ours")
    a = ap.parse_args()
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True  # fixed 80x80 input: let cuDNN pick the fastest kernels
    for p in a.patches:
        train(p, a, dev)

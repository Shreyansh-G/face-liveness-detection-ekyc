"""
CelebA-Spoof (Hugging Face mirror Ar4ikov/celebA_spoof) -> 80x80 face crops -> PyTorch dataset.

Crop layout: <root>/<split>/<scale>_80x80/<label>/<id>.png   label 1 = real, 0 = fake
Splits: train (training), val (best epoch + bank threshold), test (final numbers only).
Licence: CelebA-Spoof is non-commercial research only; never redistribute its images.
"""
import json
import os
import random
import urllib.request
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import partial
from pathlib import Path

import cv2
import numpy as np
import torch

from fas.augment import fft_target, standard_augment
from fas.engine import crop_face

HF_REPO = "Ar4ikov/celebA_spoof"
SPLITS = {"train": "train", "valid": "val", "test": "test"}  # mirror split -> folder
LABELS = {"live": 1, "spoof": 0}


def _download(f, out):
    dest = out / Path(f["path"]).name
    if dest.exists() and dest.stat().st_size == f["size"]:
        return
    with urllib.request.urlopen(f"https://huggingface.co/datasets/{HF_REPO}/resolve/main/{f['path']}") as r, \
            open(dest.with_suffix(".part"), "wb") as w:
        while chunk := r.read(1 << 22):
            w.write(chunk)
    dest.with_suffix(".part").replace(dest)
    print(f"downloaded {dest.name} ({f['size'] / 1e6:.0f} MB)", flush=True)


def download_celeba(n_train, n_valid, n_test, out, parallel=4):
    """Download the first N parquet files of each split (train ~4.5k images/file, valid ~5.2k, test ~0.9k)."""
    with urllib.request.urlopen(f"https://huggingface.co/api/datasets/{HF_REPO}/tree/main/data") as r:
        listing = json.load(r)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    files = [f for split, n in (("train", n_train), ("valid", n_valid), ("test", n_test))
             for f in sorted((f for f in listing if Path(f["path"]).name.startswith(split + "-")),
                             key=lambda f: f["path"])[:n]]
    with ThreadPoolExecutor(parallel) as pool:
        list(pool.map(partial(_download, out=out), files))


def _crop_file(f, out_dir, scales, size, delete_after, limit, split):
    """One parquet file -> crops. Runs in its own process."""
    import pyarrow.parquet as pq

    name = split or f.name.split("-")[0]
    prefix = "".join(f.stem.split("-")[:2])  # e.g. "train00003"
    counts, n = {0: 0, 1: 0}, 0
    for batch in pq.ParquetFile(f).iter_batches(batch_size=128, columns=["Filepath", "Bbox", "Class"]):
        for row in batch.to_pylist():
            if limit and n >= limit:
                break
            n += 1
            label, data = LABELS.get(row["Class"]), (row["Filepath"] or {}).get("bytes")
            img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR) if data else None
            x1, y1, x2, y2 = row["Bbox"]
            if label is None or img is None or x2 - x1 < 10 or y2 - y1 < 10:
                continue
            for s in scales:
                d = Path(out_dir) / SPLITS[name] / f"{s:g}_{size}x{size}" / str(label)
                d.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(d / f"{prefix}_{n:05d}.png"), crop_face(img, [x1, y1, x2 - x1 + 1, y2 - y1 + 1], s, size, size))
            counts[label] += 1
        if limit and n >= limit:
            break
    if delete_after:
        f.unlink()
    return f"{f.name} -> {SPLITS[name]}: {counts[1]} real, {counts[0]} fake"


def parquet_to_crops(raw_dir, out_dir, scales=(2.7, 4.0), size=80, delete_after=False, limit=0, split=None,
                     workers=os.cpu_count()):
    """Cut every face at each scale (one crop per pretrained model), one process per file.
    Face boxes come with the data."""
    files = sorted(Path(raw_dir).glob("*.parquet"))
    job = partial(_crop_file, out_dir=out_dir, scales=scales, size=size, delete_after=delete_after,
                  limit=limit, split=split)
    with ProcessPoolExecutor(min(workers, max(1, len(files)))) as pool:
        for msg in pool.map(job, files):
            print(msg, flush=True)


def list_images(root, limit=0, seed=0):
    """[(path, label)] from root/0 (fake) and root/1 (real)."""
    items = [(str(p), label) for label in (0, 1) for p in sorted((Path(root) / str(label)).glob("*.png"))]
    if limit:
        random.Random(seed).shuffle(items)
        items = items[:limit]
    return items


class CropDataset(torch.utils.data.Dataset):
    """train=True: fraud-aware augmentation (if given) -> FFT target -> standard augmentation.
    cache=True decodes every crop once into RAM (~0.9 GB per 45k crops) instead of every epoch."""

    def __init__(self, items, train=False, fraud_aug=None, use_fft=True, ft_hw=(10, 10), cache=False):
        self.items, self.train, self.fraud_aug, self.use_fft, self.ft_hw = items, train, fraud_aug, use_fft, ft_hw
        self.images = None
        if cache:
            with ThreadPoolExecutor(8) as pool:
                self.images = np.stack(list(pool.map(cv2.imread, (p for p, _ in items))))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, label = self.items[i]
        img = self.images[i].copy() if self.images is not None else cv2.imread(path)
        ft = np.zeros(self.ft_hw, np.float32)
        if self.train:
            if self.fraud_aug is not None:
                img, label = self.fraud_aug(img, label)
            if self.use_fft:  # spectrum after fraud transforms, so synthetic screen/print traces show up
                ft = fft_target(img, *self.ft_hw)
            img = standard_augment(img)
        x = torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1))).float()  # raw 0-255 BGR
        return x, torch.from_numpy(ft)[None], label

"""
Download a slice of CelebA-Spoof and cut 80x80 face crops (2.7x and 4x the face box).
Downloads 4 files at a time and crops one file per CPU core.

  python scripts/prepare_data.py --train 10 --valid 1 --test 6 --raw /tmp/celeba_raw --out data
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fas.data import download_celeba, parquet_to_crops  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=10, help="train files (of 93), ~4.5k images each")
    ap.add_argument("--valid", type=int, default=1, help="valid files (of 9), ~5.2k images each")
    ap.add_argument("--test", type=int, default=6, help="test files (of 65), ~0.9k images each")
    ap.add_argument("--raw", default="data/raw", help="where parquet files are downloaded")
    ap.add_argument("--out", default="data", help="where crops are written")
    ap.add_argument("--workers", type=int, default=os.cpu_count(), help="parallel cropping processes")
    ap.add_argument("--keep_raw", action="store_true", help="keep parquet files after cropping")
    ap.add_argument("--no_download", action="store_true", help="only crop files already in --raw")
    ap.add_argument("--limit", type=int, default=0, help="max rows per file (quick tests)")
    ap.add_argument("--split", choices=["train", "valid", "test"], help="treat every file as this split (quick tests)")
    a = ap.parse_args()

    if not a.no_download:
        download_celeba(a.train, a.valid, a.test, a.raw)
    parquet_to_crops(a.raw, a.out, delete_after=not a.keep_raw, limit=a.limit, split=a.split, workers=a.workers)

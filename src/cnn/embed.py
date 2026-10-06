"""Extract frozen-backbone embeddings for every Flavia image, once per backbone.

This is the expensive step (VGG11 is ~7.6 GFLOPs per image on CPU), so embeddings are
cached to cache/cnn/<backbone>.npz and every later experiment -- linear probes,
t-SNE, the classifier comparison -- reads the cache instead of the images.

No augmentation here: these embeddings describe the images as they are. Augmentation
belongs to fine-tuning, where the network can actually learn from it.

Usage:
    python -m src.cnn.embed                      # all three backbones
    python -m src.cnn.embed --backbone resnet18
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from . import backbones
from .data import LeafDataset, eval_transform, load_index


def extract(name: str, out_dir: Path, batch_size: int, workers: int,
            force: bool) -> Path:
    out = out_dir / f"{name}.npz"
    if out.exists() and not force:
        print(f"  {name}: đã có {out}, bỏ qua (dùng --force để chạy lại)")
        return out

    paths, labels, classes = load_index()
    loader = DataLoader(LeafDataset(paths, labels, eval_transform()),
                        batch_size=batch_size, shuffle=False, num_workers=workers)

    bb = backbones.build(name)
    print(f"  {name}: {bb.params / 1e6:.1f}M tham số, embedding {bb.dim} chiều")
    start = time.time()
    features = backbones.embed(bb, loader).numpy()
    elapsed = time.time() - start

    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, X=features, y=labels, classes=np.array(classes),
                        paths=np.array([str(p) for p in paths]),
                        params=bb.params, seconds=elapsed)
    print(f"  {name}: {features.shape} trong {elapsed:.0f}s "
          f"({1000 * elapsed / len(labels):.0f} ms/ảnh) -> {out}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Trích đặc trưng bằng backbone đóng băng")
    ap.add_argument("--backbone", choices=backbones.NAMES, default=None,
                    help="mặc định: chạy cả ba")
    ap.add_argument("--out", type=Path, default=Path("cache/cnn"))
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    torch.set_num_threads(torch.get_num_threads())
    names = [args.backbone] if args.backbone else list(backbones.NAMES)
    for name in names:
        extract(name, args.out, args.batch_size, args.workers, args.force)


if __name__ == "__main__":
    main()

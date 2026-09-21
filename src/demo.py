"""Single-image demo for the submission video.

Predicts one image end-to-end and saves a figure showing the input, the segmentation
mask, the predicted species and -- for Flavia filenames -- whether it was right.

The model is trained on every cached image EXCEPT the one being demonstrated. A demo
that has seen its own test image proves nothing, and it is the first thing a viewer
would ask about.

`--rotate` turns the image before prediction, which shows the invariance finding live:
a leaf classified correctly upright can be misclassified after a 15° turn, while a 90°
turn changes nothing (texture features are invariant only to multiples of 90°).

Usage:
    python -m src.demo data/flavia/3120.jpg
    python -m src.demo data/flavia/3120.jpg --rotate 15
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import cv2
import numpy as np

from . import features
from .dataset import columns_for, load_cached
from .experiments import classifiers
from .invariance import _rotate
from .labels import from_id
from .preprocess import segment


def true_label(image: Path) -> str | None:
    """Species from the Flavia filename number, or None for a non-Flavia image."""
    m = re.search(r"(\d+)", image.stem)
    return from_id(int(m.group(1))) if m else None


def main() -> None:
    ap = argparse.ArgumentParser(description="Classify one leaf image")
    ap.add_argument("image", type=Path)
    ap.add_argument("--rotate", type=float, default=0.0,
                    help="rotate the image by this many degrees before predicting")
    ap.add_argument("--cache", type=Path, default=Path("cache/features.npz"))
    ap.add_argument("--data", type=Path, default=Path("data/flavia"))
    ap.add_argument("--model", default="SVM (RBF)")
    # Default = the configuration src.experiments found best. The vein group is left
    # out: it adds nothing measurable, see results/ablation.csv and vein_variants.csv.
    ap.add_argument("--groups", default="shape,color,texture")
    ap.add_argument("--out", type=Path, default=None,
                    help="figure path (default: results/demo_<name>[_rot<deg>].png)")
    args = ap.parse_args()

    groups = args.groups.split(",")
    data = load_cached(args.cache, args.data)
    cols = columns_for(data, groups)

    # Leave the demonstrated image out of training (matched by resolved path).
    target = args.image.resolve()
    keep = np.array([Path(str(p)).resolve() != target for p in data["paths"]])
    excluded = int((~keep).sum())
    model = classifiers()[args.model]
    model.fit(data["X"][keep][:, cols], data["y"][keep])

    bgr = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if bgr is None:
        raise SystemExit(f"cannot read {args.image}")
    if args.rotate:
        bgr = _rotate(bgr, args.rotate)
    sample = segment(bgr, args.image)
    vector = features.extract(sample, groups).reshape(1, -1)
    prediction = model.predict(vector)[0]

    truth = true_label(args.image)
    verdict = "" if truth is None else ("  ĐÚNG" if prediction == truth else f"  SAI (đúng là: {truth})")
    angle = f" xoay {args.rotate:g}°" if args.rotate else ""
    print(f"{args.image.name}{angle} -> {prediction}{verdict}")
    print(f"(mô hình huấn luyện trên {int(keep.sum())} ảnh, đã loại {excluded} ảnh đang demo)")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(sample.bgr[:, :, ::-1])
    axes[0].set_title(f"ảnh đầu vào{angle}")
    axes[1].imshow(sample.mask, cmap="gray")
    axes[1].set_title("phân đoạn")
    for ax in axes:
        ax.axis("off")
    colour = "black" if truth is None else ("green" if prediction == truth else "red")
    fig.suptitle(f"dự đoán: {prediction}{verdict}", color=colour)
    fig.tight_layout()

    out = args.out or Path("results") / (
        f"demo_{args.image.stem}" + (f"_rot{args.rotate:g}" if args.rotate else "") + ".png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"hình -> {out}")


if __name__ == "__main__":
    main()

"""Build (and cache) the feature matrix.

Extraction is the expensive half of this project and classification is nearly free,
so every group is extracted once into a single cached matrix; experiments then slice
columns out of it instead of re-reading images. Deleting the cache is the only way to
pick up a change to a feature extractor -- `--force` does that for you.

Usage:
    python -m src.dataset --data data/flavia --cache cache/features.npz
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np

from . import features
from .labels import from_id
from .preprocess import SegmentationError, load

# The default extraction set, and what the ablation grid ranges over. Deliberately
# NOT list(features.GROUPS): the alternative vein extractors live in the registry but
# are compared separately in src.vein_variants, not folded into the ablation.
ALL_GROUPS = list(features.ABLATION_GROUPS)
_IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
_ID_RE = re.compile(r"(\d+)")


def find_images(root: Path) -> list[tuple[Path, str]]:
    """Return (path, label) pairs.

    Directory-per-species layout wins when present; otherwise fall back to the Flavia
    filename-ID ranges.
    """
    subdirs = [d for d in sorted(root.iterdir()) if d.is_dir()] if root.is_dir() else []
    items: list[tuple[Path, str]] = []

    if subdirs:
        for d in subdirs:
            for p in sorted(d.rglob("*")):
                if p.suffix.lower() in _IMAGE_EXT:
                    items.append((p, d.name))
        return items

    unmapped = 0
    for p in sorted(root.glob("*")):
        if p.suffix.lower() not in _IMAGE_EXT:
            continue
        m = _ID_RE.search(p.stem)
        label = from_id(int(m.group(1))) if m else None
        if label is None:
            unmapped += 1
            continue
        items.append((p, label))
    if unmapped:
        print(f"warning: {unmapped} image(s) had no species range -- check src/labels.py", file=sys.stderr)
    return items


def build(root: Path, groups: list[str] | None = None) -> dict:
    groups = groups or ALL_GROUPS
    items = find_images(root)
    if not items:
        raise SystemExit(f"no images under {root} -- put the Flavia archive there first")

    rows, labels, used, failed = [], [], [], []
    for i, (path, label) in enumerate(items, 1):
        try:
            sample = load(path)
            rows.append(features.extract(sample, groups))
        except SegmentationError as exc:
            failed.append(str(exc))
            continue
        labels.append(label)
        used.append(str(path))
        if i % 50 == 0 or i == len(items):
            print(f"  {i}/{len(items)} images", end="\r", flush=True)

    print()
    for message in failed:
        print(f"skipped: {message}", file=sys.stderr)

    X = np.vstack(rows)
    # A constant or NaN column breaks StandardScaler silently; catch it at build time.
    if not np.isfinite(X).all():
        bad = np.unique(np.argwhere(~np.isfinite(X))[:, 1])
        names = features.names(groups)
        raise SystemExit(f"non-finite values in columns: {[names[b] for b in bad]}")

    return {
        "X": X,
        "y": np.array(labels),
        "feature_names": np.array(features.names(groups)),
        "groups": np.array(groups),
        "paths": np.array(used),
    }


def load_cached(cache: Path, root: Path, force: bool = False,
                groups: list[str] | None = None) -> dict:
    if cache.exists() and not force:
        data = dict(np.load(cache, allow_pickle=False))
        print(f"loaded cached features {data['X'].shape} from {cache}")
        return data
    print(f"extracting features from {root} ...")
    data = build(root, groups)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, **data)
    print(f"cached features {data['X'].shape} -> {cache}")
    return data


def columns_for(data: dict, groups: list[str]) -> np.ndarray:
    """Boolean column mask selecting the given feature groups from a cached matrix."""
    prefixes = tuple(f"{g}." for g in groups)
    return np.array([n.startswith(prefixes) for n in data["feature_names"]])


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract Flavia leaf features")
    ap.add_argument("--data", type=Path, default=Path("data/flavia"))
    ap.add_argument("--cache", type=Path, default=Path("cache/features.npz"))
    ap.add_argument("--force", action="store_true", help="re-extract even if cached")
    args = ap.parse_args()

    data = load_cached(args.cache, args.data, args.force)
    classes, counts = np.unique(data["y"], return_counts=True)
    print(f"{data['X'].shape[0]} samples, {data['X'].shape[1]} features, {len(classes)} classes")
    print(f"class sizes: min {counts.min()}, max {counts.max()}")


if __name__ == "__main__":
    main()

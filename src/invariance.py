"""Are the features actually invariant to what we claim they are?

The report asserts several invariances without measuring any of them: dimensionless
shape ratios and Hu moments should survive rotation and scaling, GLCM averaged over
four angles should survive rotation, HSV should be steadier than RGB under lighting
changes. This experiment tests those claims by transforming the *images* -- not the
feature vectors -- so segmentation is stress-tested too, which is the honest version
of the question: a classifier is only invariant if the whole pipeline in front of it is.

Protocol: train on clean features from the main cache, test on features re-extracted
from transformed copies of the held-out images. Each feature group is evaluated on its
own so a drop can be attributed rather than merely observed.

A segmentation failure counts as a misclassification -- the pipeline failed end to end
-- and the failure count is reported separately so the two causes stay distinguishable.

Usage:
    python -m src.invariance
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from . import features
from .dataset import ALL_GROUPS
from .experiments import RANDOM_STATE, classifiers
from .preprocess import SegmentationError, segment

def _prefix(*groups: str):
    return lambda name: name.startswith(tuple(f"{g}." for g in groups))


def _colour_space(letters: str):
    """Select one colour space out of the colour group.

    Feature names are `color.<channel>_<stat>`; RGB channels are r/g/b and HSV are
    h/s/v, so the first letter identifies the space unambiguously. Splitting them lets
    the report's claim that HSV is steadier under lighting changes be tested rather
    than repeated.
    """
    return lambda name: (name.startswith("color.")
                         and name.split(".", 1)[1][0] in letters)


# Feature sets scored separately, so a drop can be attributed to a specific group.
FEATURE_SETS = {
    "shape": _prefix("shape"),
    "color": _prefix("color"),
    "color RGB": _colour_space("rgb"),
    "color HSV": _colour_space("hsv"),
    "texture": _prefix("texture"),
    "vein": _prefix("vein"),
    "shape+color+texture": _prefix("shape", "color", "texture"),
}

_WHITE = (255, 255, 255)


def _rotate(bgr: np.ndarray, angle: float) -> np.ndarray:
    """Rotate about the centre into an expanded canvas, filling new area with white.

    Two details, both of which produced wrong results when first got wrong:

    The canvas must GROW to fit the rotated frame. Keeping the original 1600x1200 clips
    the leaf -- measured at 83% of test images for a 90° turn, losing a median 4.2% of
    leaf area -- and clipped leaves change every shape feature. That would be measuring
    a flaw in the transform and reporting it as non-invariance of the features.

    The fill colour must be white. Flavia's background is near-white, so warpAffine's
    default black would hand Otsu a second dark region and break segmentation for a
    reason that has nothing to do with rotation.
    """
    h, w = bgr.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    cos, sin = abs(m[0, 0]), abs(m[0, 1])
    new_w, new_h = int(h * sin + w * cos), int(h * cos + w * sin)
    m[0, 2] += (new_w - w) / 2
    m[1, 2] += (new_h - h) / 2
    return cv2.warpAffine(bgr, m, (new_w, new_h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=_WHITE)


def _scale(bgr: np.ndarray, factor: float) -> np.ndarray:
    """Shrink the leaf inside a canvas of unchanged size, padding with white.

    Resizing the canvas instead would leave the leaf occupying the same fraction of the
    frame, which is not a scale change at all for ratio-based features.
    """
    h, w = bgr.shape[:2]
    small = cv2.resize(bgr, (int(w * factor), int(h * factor)), interpolation=cv2.INTER_AREA)
    out = np.full_like(bgr, 255)
    y0, x0 = (h - small.shape[0]) // 2, (w - small.shape[1]) // 2
    out[y0:y0 + small.shape[0], x0:x0 + small.shape[1]] = small
    return out


def _brightness(bgr: np.ndarray, factor: float) -> np.ndarray:
    return cv2.convertScaleAbs(bgr, alpha=factor, beta=0)


def _noise(bgr: np.ndarray, sigma: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noisy = bgr.astype(np.int16) + rng.normal(0, sigma, bgr.shape).astype(np.int16)
    return np.clip(noisy, 0, 255).astype(np.uint8)


def _occlude(bgr: np.ndarray, fraction: float) -> np.ndarray:
    """Cover `fraction` of the leaf's bounding box with an opaque grey rectangle.

    The leaf must be located first, so this transform segments the clean image -- the
    occluder has to land on the leaf to test anything.
    """
    try:
        clean = segment(bgr)
    except SegmentationError:
        return bgr
    x, y, w, h = cv2.boundingRect(clean.contour)
    side = int(np.sqrt(fraction * w * h))
    cx, cy = x + w // 2, y + h // 2
    out = bgr.copy()
    cv2.rectangle(out, (cx - side // 2, cy - side // 2),
                  (cx + side // 2, cy + side // 2), (90, 90, 90), thickness=cv2.FILLED)
    return out


def transforms() -> dict:
    """name -> callable(bgr, index) -> bgr. The index only seeds the noise generator."""
    return {
        "identity": lambda img, i: img,
        "rotate 15°": lambda img, i: _rotate(img, 15),
        "rotate 45°": lambda img, i: _rotate(img, 45),
        "rotate 90°": lambda img, i: _rotate(img, 90),
        "scale 0.5": lambda img, i: _scale(img, 0.5),
        "bright ×1.3": lambda img, i: _brightness(img, 1.3),
        "dark ×0.7": lambda img, i: _brightness(img, 0.7),
        "noise σ=10": lambda img, i: _noise(img, 10.0, RANDOM_STATE + i),
        "occlude 10%": lambda img, i: _occlude(img, 0.10),
    }


def extract_transformed(paths: np.ndarray, name: str, fn, cache_dir: Path,
                        force: bool) -> tuple[np.ndarray, np.ndarray]:
    """Features for the test images under one transform. Returns (X, ok_mask)."""
    cache = cache_dir / f"{name.replace(' ', '_').replace('×', 'x').replace('σ', 's')}.npz"
    if cache.exists() and not force:
        d = np.load(cache, allow_pickle=False)
        return d["X"], d["ok"]

    width = len(features.names(ALL_GROUPS))
    X = np.zeros((len(paths), width))
    ok = np.zeros(len(paths), dtype=bool)
    for i, p in enumerate(paths):
        bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
        try:
            sample = segment(fn(bgr, i), Path(p))
            X[i] = features.extract(sample, ALL_GROUPS)
            ok[i] = True
        except SegmentationError:
            pass  # left as zeros; counted as a failure and scored as wrong
        if (i + 1) % 100 == 0:
            print(f"    {i + 1}/{len(paths)}", end="\r", flush=True)

    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, X=X, ok=ok)
    return X, ok


def drift_table(data: dict, names: list, test_idx: np.ndarray, args) -> pd.DataFrame:
    """How far each feature moves under each transform, in units of its own std.

    Accuracy says *that* a feature set broke; this says *which* features moved and by
    how much, which is what turns a number into an explanation. Expressing the shift in
    standard deviations makes features with wildly different units comparable, and it is
    the scale the classifier effectively sees after StandardScaler.
    """
    clean = data["X"][test_idx]
    sd = clean.std(axis=0)
    sd = np.where(sd > 0, sd, 1.0)

    # Sub-blocks worth separating: GLCM vs LBP answer different questions about
    # rotation, and the report cites them individually.
    blocks = {
        "shape": lambda n: n.startswith("shape."),
        "color": lambda n: n.startswith("color."),
        "texture": lambda n: n.startswith("texture."),
        "texture_glcm": lambda n: n.startswith("texture.glcm"),
        "texture_lbp": lambda n: n.startswith("texture.lbp"),
        "vein": lambda n: n.startswith("vein."),
    }

    rows = []
    for name, fn in transforms().items():
        X, _ = extract_transformed(data["paths"][test_idx], name, fn,
                                   args.transform_cache, False)
        shift = np.abs(X - clean).mean(axis=0) / sd
        for block, predicate in blocks.items():
            mask = np.array([predicate(str(n)) for n in names])
            rows.append({"transform": name, "block": block,
                         "drift_mean": float(shift[mask].mean())})
        worst = int(np.argmax(shift))
        rows.append({"transform": name, "block": "WORST:" + str(names[worst]),
                     "drift_mean": float(shift[worst])})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="Invariance stress test")
    ap.add_argument("--cache", type=Path, default=Path("cache/features.npz"))
    ap.add_argument("--transform-cache", type=Path, default=Path("cache/invariance"))
    ap.add_argument("--out", type=Path, default=Path("results"))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    data = dict(np.load(args.cache, allow_pickle=False))
    names = list(data["feature_names"])
    idx = np.arange(len(data["y"]))
    train_idx, test_idx = train_test_split(
        idx, test_size=0.25, stratify=data["y"], random_state=RANDOM_STATE
    )
    test_paths = data["paths"][test_idx]
    y_test = data["y"][test_idx]
    print(f"train {len(train_idx)}, test {len(test_idx)}")

    # One model per feature set, fitted once on clean training data and reused for
    # every transform -- the question is whether the features move, not whether the
    # classifier can be retrained to cope.
    models = {}
    for label, predicate in FEATURE_SETS.items():
        cols = np.array([predicate(str(n)) for n in names])
        model = classifiers()["SVM (RBF)"]
        model.fit(data["X"][np.ix_(train_idx, np.where(cols)[0])], data["y"][train_idx])
        models[label] = (model, cols)

    rows = []
    for name, fn in transforms().items():
        print(f"  {name}")
        X, ok = extract_transformed(test_paths, name, fn, args.transform_cache, args.force)
        for label, (model, cols) in models.items():
            pred = model.predict(X[:, cols])
            correct = (pred == y_test) & ok
            rows.append({
                "transform": name,
                "features": label,
                "accuracy": correct.mean(),
                "seg_failures": int((~ok).sum()),
            })
        acc = rows[-1]["accuracy"]
        print(f"    seg failures {int((~ok).sum())}, best-set accuracy {acc:.4f}")

    table = pd.DataFrame(rows)
    args.out.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out / "invariance.csv", index=False)
    drift_table(data, names, test_idx, args).to_csv(args.out / "invariance_drift.csv", index=False)
    pivot = table.pivot(index="transform", columns="features", values="accuracy")
    pivot.to_csv(args.out / "invariance_pivot.csv")
    print("\n" + pivot.to_string(float_format=lambda v: f"{v:.4f}"))

    control = pivot.loc["identity", "shape+color+texture"]
    print(f"\ncontrol (identity) accuracy: {control:.4f}")
    print("If this does not match the held-out accuracy from src.experiments, the "
          "re-extraction path is broken and every drop below is a bug, not a finding.")


if __name__ == "__main__":
    main()

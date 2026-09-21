"""Does a better vein extractor rescue the vein feature group?

The main ablation found the Wu-style vein group (Otsu on the opening residue) worthless
and slightly harmful. That is a claim about *this implementation*, not about vein
features in general, so this experiment swaps the binarisation step for two
alternatives -- a fixed relative threshold and Frangi vesselness -- and re-measures
against the same baseline under the identical protocol.

Every configuration is `shape+color+texture` (the best feature set) plus at most one
vein variant, evaluated with the same stratified 5-fold split, so the only thing that
differs between rows is how vein evidence is binarised.

Usage:
    python -m src.vein_variants
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel
from sklearn.model_selection import cross_val_score

from . import features
from .dataset import columns_for, load_cached
from .experiments import RANDOM_STATE, _cv, classifiers

BASE = ["shape", "color", "texture"]


def aligned_variants(main: dict, cache: Path, root: Path, force: bool) -> dict:
    """Extract the alternative vein groups and align them to the main matrix by path.

    Alignment is by filename rather than by position: a segmentation failure in either
    pass would otherwise silently shift every subsequent row against its label.
    """
    extra = load_cached(cache, root, force, groups=["vein_pct", "vein_frangi"])
    index = {p: i for i, p in enumerate(extra["paths"])}
    missing = [p for p in main["paths"] if p not in index]
    if missing:
        raise SystemExit(f"{len(missing)} image(s) missing from {cache}, e.g. {missing[0]}")
    order = [index[p] for p in main["paths"]]
    return {
        "X": extra["X"][order],
        "feature_names": extra["feature_names"],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare vein extraction variants")
    ap.add_argument("--data", type=Path, default=Path("data/flavia"))
    ap.add_argument("--cache", type=Path, default=Path("cache/features.npz"))
    ap.add_argument("--variant-cache", type=Path, default=Path("cache/vein_variants.npz"))
    ap.add_argument("--out", type=Path, default=Path("results"))
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    data = load_cached(args.cache, args.data)
    extra = aligned_variants(data, args.variant_cache, args.data, args.force)
    args.out.mkdir(parents=True, exist_ok=True)

    y = data["y"]
    cv = _cv(args.folds)
    base_cols = columns_for(data, BASE)

    # Each configuration: (label, feature matrix). The baseline carries no vein columns.
    configs: list[tuple[str, np.ndarray]] = [("(không dùng gân lá)", data["X"][:, base_cols])]
    configs.append((
        "vein (Otsu -- baseline)",
        np.hstack([data["X"][:, base_cols], data["X"][:, columns_for(data, ["vein"])]]),
    ))
    for group in ("vein_pct", "vein_frangi"):
        cols = np.array([n.startswith(f"{group}.") for n in extra["feature_names"]])
        configs.append((
            {"vein_pct": "vein_pct (ngưỡng cố định)", "vein_frangi": "vein_frangi (Frangi)"}[group],
            np.hstack([data["X"][:, base_cols], extra["X"][:, cols]]),
        ))

    # Per-fold scores are kept so each variant can be compared against the no-vein
    # baseline with a *paired* test: the folds are identical across configurations, so
    # pairing removes the fold-to-fold variance that otherwise swamps differences this
    # small. Without it, a gap of 0.001 against a fold std of 0.004 looks like a result.
    rows = []
    baseline_folds: dict[str, np.ndarray] = {}
    for label, X in configs:
        for name, model in classifiers().items():
            scores = cross_val_score(model, X, y, cv=cv, n_jobs=-1)
            if label == configs[0][0]:
                baseline_folds[name] = scores
                p_value = float("nan")
            else:
                p_value = float(ttest_rel(scores, baseline_folds[name]).pvalue)
            rows.append({
                "vein_variant": label,
                "n_features": X.shape[1],
                "classifier": name,
                "accuracy_mean": scores.mean(),
                "accuracy_std": scores.std(),
                "delta_vs_no_vein": scores.mean() - baseline_folds[name].mean(),
                "paired_p": p_value,
                "folds": " ".join(f"{s:.4f}" for s in scores),
            })
            print(f"  {label:28s} {name:15s} {scores.mean():.4f} +- {scores.std():.4f}"
                  f"  p={p_value:.3f}")

    table = pd.DataFrame(rows)
    table.to_csv(args.out / "vein_variants.csv", index=False)
    pivot = table.pivot(index="vein_variant", columns="classifier", values="accuracy_mean")
    pivot.to_csv(args.out / "vein_variants_pivot.csv")
    print("\n" + pivot.to_string(float_format=lambda v: f"{v:.4f}"))

    print("\nPaired vs no-vein baseline (delta, p) -- * marks p < 0.05:")
    for _, r in table[table["paired_p"].notna()].iterrows():
        star = "*" if r["paired_p"] < 0.05 else " "
        print(f"  {r['vein_variant']:28s} {r['classifier']:15s} "
              f"{r['delta_vs_no_vein']:+.4f}  p={r['paired_p']:.3f} {star}")


if __name__ == "__main__":
    main()

"""How much do the colour and texture groups actually overlap?

Section 6.1 of the report infers overlap indirectly: colour alone and texture alone
each lift the shape baseline by a similar amount, yet combining them adds little, so
"they must encode the same thing". That is a reasonable reading of accuracy numbers,
and it is also the weakest link in the report -- accuracy is a very blunt instrument
for measuring shared information. This module measures it directly on the cached
matrix, with no re-extraction.

Four measurements, from the bluntest to the sharpest:

  mutual information   how much each group knows about the label on its own
  canonical corr. (CCA) the strongest linear relationships *between* two groups
  ridge R^2            can colour features predict texture features outright?
  permutation importance which groups the fitted model actually leans on

CCA and ridge answer different questions and both are needed: CCA finds the best
correlated projections (symmetric, one direction at a time), while ridge asks whether
a whole block is reconstructible from another (asymmetric, per-feature).

Usage:
    python -m src.redundancy
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import CCA
from sklearn.feature_selection import mutual_info_classif
from sklearn.inspection import permutation_importance
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .dataset import ALL_GROUPS, columns_for, load_cached
from .experiments import RANDOM_STATE, classifiers

BEST = ["shape", "color", "texture"]


def group_mutual_information(data: dict) -> pd.DataFrame:
    """Mean MI between each feature and the label, averaged within each group.

    The mean rather than the sum: groups have different dimensionalities (16/18/22/7)
    and summing would rank them by size instead of by informativeness.
    """
    X = StandardScaler().fit_transform(data["X"])
    mi = mutual_info_classif(X, data["y"], random_state=RANDOM_STATE)
    rows = []
    for group in ALL_GROUPS:
        cols = columns_for(data, [group])
        rows.append({
            "group": group,
            "n_features": int(cols.sum()),
            "mi_mean": float(mi[cols].mean()),
            "mi_max": float(mi[cols].max()),
        })
    return pd.DataFrame(rows)


def canonical_correlations(data: dict, a: str, b: str, n: int = 5) -> list[float]:
    """Top-n canonical correlations between two feature blocks.

    1.0 would mean one block is a linear re-encoding of the other; near 0 means the
    blocks carry unrelated linear structure.
    """
    Xa = StandardScaler().fit_transform(data["X"][:, columns_for(data, [a])])
    Xb = StandardScaler().fit_transform(data["X"][:, columns_for(data, [b])])
    k = min(n, Xa.shape[1], Xb.shape[1])
    cca = CCA(n_components=k, max_iter=1000)
    Ua, Ub = cca.fit_transform(Xa, Xb)
    return [float(np.corrcoef(Ua[:, i], Ub[:, i])[0, 1]) for i in range(k)]


def cross_predictability(data: dict, src: str, dst: str) -> float:
    """Median held-out R^2 when predicting each `dst` feature from all of `src`.

    Median, not mean: a couple of unpredictable features would otherwise drag a
    genuinely high overlap down and understate it.
    """
    Xs = StandardScaler().fit_transform(data["X"][:, columns_for(data, [src])])
    Xd = StandardScaler().fit_transform(data["X"][:, columns_for(data, [dst])])
    tr, te = train_test_split(np.arange(len(Xs)), test_size=0.25, random_state=RANDOM_STATE)
    scores = []
    for j in range(Xd.shape[1]):
        model = RidgeCV(alphas=np.logspace(-2, 3, 12)).fit(Xs[tr], Xd[tr, j])
        scores.append(model.score(Xs[te], Xd[te, j]))
    return float(np.median(scores))


def group_importance(data: dict) -> pd.DataFrame:
    """Permutation importance on the best model, summed within each group.

    Summed here, unlike MI: importance is a drop in accuracy attributable to the group
    as a whole, so the contributions genuinely add up.
    """
    cols = columns_for(data, BEST)
    X, y = data["X"][:, cols], data["y"]
    names = np.array(data["feature_names"])[cols]
    tr, te = train_test_split(np.arange(len(y)), test_size=0.25, stratify=y,
                              random_state=RANDOM_STATE)
    model = classifiers()["SVM (RBF)"].fit(X[tr], y[tr])
    result = permutation_importance(model, X[te], y[te], n_repeats=10,
                                    random_state=RANDOM_STATE, n_jobs=-1)
    rows = []
    for group in BEST:
        mask = np.array([n.startswith(f"{group}.") for n in names])
        rows.append({
            "group": group,
            "importance_sum": float(result.importances_mean[mask].sum()),
            "top_feature": str(names[mask][np.argmax(result.importances_mean[mask])]),
        })
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="Quantify feature-group redundancy")
    ap.add_argument("--data", type=Path, default=Path("data/flavia"))
    ap.add_argument("--cache", type=Path, default=Path("cache/features.npz"))
    ap.add_argument("--out", type=Path, default=Path("results"))
    args = ap.parse_args()

    data = load_cached(args.cache, args.data)
    args.out.mkdir(parents=True, exist_ok=True)

    mi = group_mutual_information(data)
    print("\nMutual information with the label:")
    print(mi.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    pairs = [("color", "texture"), ("color", "shape"), ("texture", "shape")]
    rows = []
    for a, b in pairs:
        corrs = canonical_correlations(data, a, b)
        rows.append({
            "pair": f"{a}-{b}",
            "cca_1": corrs[0],
            "cca_2": corrs[1],
            "cca_3": corrs[2],
            "r2_forward": cross_predictability(data, a, b),
            "r2_backward": cross_predictability(data, b, a),
        })
    overlap = pd.DataFrame(rows)
    print("\nBetween-group overlap (canonical correlations, ridge R^2):")
    print(overlap.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    importance = group_importance(data)
    print("\nPermutation importance on shape+color+texture:")
    print(importance.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    mi.to_csv(args.out / "redundancy_mi.csv", index=False)
    overlap.to_csv(args.out / "redundancy_overlap.csv", index=False)
    importance.to_csv(args.out / "redundancy_importance.csv", index=False)
    print(f"\nwrote redundancy_{{mi,overlap,importance}}.csv -> {args.out}")


if __name__ == "__main__":
    main()

"""The two experiments the report is built on.

1. Feature ablation -- what does each group actually buy?
2. Classifier comparison -- SVM vs KNN vs Random Forest vs Naive Bayes on identical
   features, which is the only way the report's claim about SVM (max-margin + kernel
   trick on high-dimensional, low-sample data) can be supported rather than asserted.

Both run stratified k-fold, and the scaler lives inside the Pipeline so it is fit on
the training fold only -- fitting it on the whole matrix leaks test statistics and
inflates exactly the numbers the comparison rests on.

Usage:
    python -m src.experiments --cache cache/features.npz --out results
"""
from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .dataset import ALL_GROUPS, columns_for, load_cached

RANDOM_STATE = 42


def classifiers() -> dict[str, Pipeline]:
    """Scaling is mandatory for SVM/KNN (margin- and distance-based) and harmless for
    the tree and the Gaussian, so every model gets the same pipeline shape and the
    comparison stays apples-to-apples."""
    return {
        "SVM (RBF)": Pipeline([
            ("scale", StandardScaler()),
            ("clf", SVC(kernel="rbf", C=10.0, gamma="scale", random_state=RANDOM_STATE)),
        ]),
        "KNN (k=5)": Pipeline([
            ("scale", StandardScaler()),
            ("clf", KNeighborsClassifier(n_neighbors=5)),
        ]),
        "Random Forest": Pipeline([
            ("scale", StandardScaler()),
            ("clf", RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE)),
        ]),
        "Naive Bayes": Pipeline([
            ("scale", StandardScaler()),
            ("clf", GaussianNB()),
        ]),
    }


def group_combinations() -> list[list[str]]:
    """shape alone, then every superset of shape -- shape is the Wu 2007 baseline and
    the point of the ablation is what the later papers add to it."""
    extras = [g for g in ALL_GROUPS if g != "shape"]
    combos = []
    for r in range(len(extras) + 1):
        for subset in itertools.combinations(extras, r):
            combos.append(["shape", *subset])
    return combos


def _cv(folds: int) -> StratifiedKFold:
    # Stratified because Flavia class sizes range from ~30 to ~77 images; a plain
    # random split under-represents the small species.
    return StratifiedKFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)


def run_ablation(data: dict, folds: int) -> pd.DataFrame:
    rows = []
    cv = _cv(folds)
    for groups in group_combinations():
        cols = columns_for(data, groups)
        X, y = data["X"][:, cols], data["y"]
        for name, model in classifiers().items():
            scores = cross_val_score(model, X, y, cv=cv, n_jobs=-1)
            rows.append({
                "features": "+".join(groups),
                "n_features": int(cols.sum()),
                "classifier": name,
                "accuracy_mean": scores.mean(),
                "accuracy_std": scores.std(),
            })
            label = "+".join(groups)
            print(f"  {label:28s} {name:15s} {scores.mean():.4f} +- {scores.std():.4f}")
    return pd.DataFrame(rows)


def confusion_for_best(data: dict, table: pd.DataFrame, out: Path) -> None:
    best = table.loc[table["accuracy_mean"].idxmax()]
    groups = best["features"].split("+")
    cols = columns_for(data, groups)
    X_train, X_test, y_train, y_test = train_test_split(
        data["X"][:, cols], data["y"], test_size=0.25,
        stratify=data["y"], random_state=RANDOM_STATE,
    )
    model = classifiers()[best["classifier"]]
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    labels = sorted(set(data["y"]))
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    pd.DataFrame(cm, index=labels, columns=labels).to_csv(out / "confusion_matrix.csv")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(14, 14))
    ConfusionMatrixDisplay(cm, display_labels=labels).plot(
        ax=ax, xticks_rotation=90, colorbar=False, values_format="d"
    )
    ax.set_title(f"{best['classifier']} on {best['features']} ({best['accuracy_mean']:.3f} CV)")
    fig.tight_layout()
    fig.savefig(out / "confusion_matrix.png", dpi=150)

    print(f"\nbest: {best['classifier']} on {best['features']} -> {best['accuracy_mean']:.4f}")
    print(f"held-out accuracy {np.mean(y_pred == y_test):.4f}; confusion matrix in {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Ablation + classifier comparison")
    ap.add_argument("--data", type=Path, default=Path("data/flavia"))
    ap.add_argument("--cache", type=Path, default=Path("cache/features.npz"))
    ap.add_argument("--out", type=Path, default=Path("results"))
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    data = load_cached(args.cache, args.data)
    args.out.mkdir(parents=True, exist_ok=True)

    table = run_ablation(data, args.folds)
    table.to_csv(args.out / "ablation.csv", index=False)

    # Wide form: feature groups down the side, classifiers across -- this is the table
    # that goes straight into the report.
    pivot = table.pivot(index="features", columns="classifier", values="accuracy_mean")
    pivot.to_csv(args.out / "ablation_pivot.csv")
    print("\n" + pivot.to_string(float_format=lambda v: f"{v:.4f}"))

    confusion_for_best(data, table, args.out)


if __name__ == "__main__":
    main()

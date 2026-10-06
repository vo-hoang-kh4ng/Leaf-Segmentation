"""2-D views of the feature space: PCA and LDA, shape-only vs shape+color+texture.

PCA is unsupervised and keeps the directions of largest variance, which need not be the
directions that separate species; LDA (Fisher) uses the labels to find exactly those.
Both are shown because a PCA plot alone makes the 32 classes look hopelessly mixed,
when an SVM separates them at ~99% in the full space.

The picture cannot show *why* colour and texture help, so this module also measures it:
KNN accuracy on the first k LDA axes, with LDA refitted inside each fold (held-out
numbers, unlike the plot, which is fitted on all data and is illustration only). In
either feature set the first two axes are spent on the same few shape-distinctive
species; the colour/texture gain only appears from roughly the fifth axis on.

The report's figure and numbers come from here via report/make_assets.py.

Usage:
    python -m src.projection
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .experiments import RANDOM_STATE

# key -> (title shown on the figure, feature-name prefixes)
SETS = {
    "shape": ("Hình dạng", ("shape.",)),
    "best": ("Hình dạng + Màu + Kết cấu", ("shape.", "color.", "texture.")),
}
LDA_AXES = (2, 10)


def _matrix(data: dict, prefixes: tuple[str, ...]) -> np.ndarray:
    names = [str(n) for n in data["feature_names"]]
    return data["X"][:, [i for i, n in enumerate(names) if n.startswith(prefixes)]]


def analyse(data: dict) -> dict:
    """Numbers behind the figure. Fractions, not percentages; callers format them."""
    y = data["y"]
    cv = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)
    out: dict = {}
    for key, (_title, prefixes) in SETS.items():
        X = _matrix(data, prefixes)
        Xs = StandardScaler().fit_transform(X)
        out[f"pca_var_{key}"] = float(PCA(n_components=2).fit(Xs).explained_variance_ratio_.sum())
        lda = LinearDiscriminantAnalysis(n_components=2).fit(Xs, y)
        out[f"lda_var_{key}"] = float(lda.explained_variance_ratio_[:2].sum())
        for k in LDA_AXES:
            pipe = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis(n_components=k),
                                 KNeighborsClassifier(5))
            out[f"lda_knn_{key}_{k}"] = float(cross_val_score(pipe, X, y, cv=cv).mean())

    # The species that monopolise the first two discriminants of the full feature set.
    Z = LinearDiscriminantAnalysis(n_components=2).fit_transform(
        StandardScaler().fit_transform(_matrix(data, SETS["best"][1])), y)
    spread = {s: float(np.linalg.norm(Z[y == s].mean(axis=0))) for s in np.unique(y)}
    out["lda_outliers"] = sorted(spread, key=spread.get, reverse=True)[:3]
    return out


def plot(data: dict, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _species, y_idx = np.unique(data["y"], return_inverse=True)
    # 32 classes: tab20 + tab20b gives 40 distinct colours without cycling.
    palette = list(plt.cm.tab20.colors) + list(plt.cm.tab20b.colors)
    colours = [palette[i % len(palette)] for i in y_idx]

    fig, axes = plt.subplots(2, 2, figsize=(9, 7.4))
    for col, (title, prefixes) in enumerate(SETS.values()):
        X = StandardScaler().fit_transform(_matrix(data, prefixes))
        projections = (("PCA", PCA(n_components=2).fit_transform(X)),
                       ("LDA", LinearDiscriminantAnalysis(n_components=2).fit_transform(X, data["y"])))
        for row, (method, Z) in enumerate(projections):
            ax = axes[row, col]
            ax.scatter(Z[:, 0], Z[:, 1], c=colours, s=4, alpha=0.7, linewidths=0)
            ax.set_title(f"{method} — {title}", fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="PCA / LDA projections of the feature space")
    ap.add_argument("--cache", type=Path, default=Path("cache/features.npz"))
    ap.add_argument("--out", type=Path, default=Path("results/projection.png"))
    args = ap.parse_args()

    data = dict(np.load(args.cache, allow_pickle=False))
    r = analyse(data)

    def pct(v: float) -> str:
        return f"{100 * v:5.1f}%".replace(".", ",")

    print(f"{'':28s}{'Hình dạng':>12s}{'H.dạng+Màu+K.cấu':>20s}")
    print(f"{'PCA 2 chiều giữ phương sai':28s}{pct(r['pca_var_shape']):>12s}{pct(r['pca_var_best']):>20s}")
    for k in LDA_AXES:
        label = f"KNN trên {k} trục LDA"
        print(f"{label:28s}{pct(r[f'lda_knn_shape_{k}']):>12s}{pct(r[f'lda_knn_best_{k}']):>20s}")
    print(f"\nHai trục LDA đầu tiên dành cho: {', '.join(r['lda_outliers'])}")

    plot(data, args.out)
    print(f"hình -> {args.out}")


if __name__ == "__main__":
    main()

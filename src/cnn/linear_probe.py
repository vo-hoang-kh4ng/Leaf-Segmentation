"""Linear probe: how good is each frozen backbone's representation on its own?

A linear classifier on top of frozen features answers a specific question -- is the
information already there in the ImageNet representation, or does the network have to
be retrained to find it? Nothing but the last layer is learned, so a high score means
the pretrained features already separate the species.

Protocol is deliberately identical to assignment 1: the same stratified 5-fold
cross-validation over all 1907 images, plus the same held-out 25% split, so the numbers
sit in the same table as the 98.85% of the hand-crafted pipeline.

Usage:
    python -m src.cnn.linear_probe
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import backbones
from .data import split
from ..experiments import RANDOM_STATE


def probe() -> LogisticRegression:
    """Multinomial logistic regression -- the standard linear-probe classifier.

    Scaling first because embedding dimensions have very different magnitudes; `max_iter`
    is generous since 512-4096 dimensions over 1430 samples converges slowly.
    """
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=3000, C=1.0, random_state=RANDOM_STATE),
    )


def load(name: str, cache: Path) -> dict:
    path = cache / f"{name}.npz"
    if not path.exists():
        raise SystemExit(f"thiếu {path} -- chạy `python -m src.cnn.embed` trước")
    return dict(np.load(path, allow_pickle=False))


def main() -> None:
    ap = argparse.ArgumentParser(description="Linear probe trên backbone đóng băng")
    ap.add_argument("--cache", type=Path, default=Path("cache/cnn"))
    ap.add_argument("--out", type=Path, default=Path("results"))
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    cv = StratifiedKFold(args.folds, shuffle=True, random_state=RANDOM_STATE)
    rows, fold_scores = [], {}
    for name in backbones.NAMES:
        data = load(name, args.cache)
        X, y = data["X"], data["y"]
        train_idx, test_idx = split(y)

        start = time.time()
        scores = cross_val_score(probe(), X, y, cv=cv, n_jobs=-1)
        model = probe().fit(X[train_idx], y[train_idx])
        held_out = float((model.predict(X[test_idx]) == y[test_idx]).mean())
        fold_scores[name] = scores

        rows.append({
            "backbone": backbones.PRETTY[name],
            "params_m": float(data["params"]) / 1e6,
            "dim": X.shape[1],
            "embed_ms_per_image": 1000 * float(data["seconds"]) / len(y),
            "cv_mean": scores.mean(),
            "cv_std": scores.std(),
            "held_out": held_out,
            "probe_seconds": time.time() - start,
            "folds": " ".join(f"{s:.4f}" for s in scores),
        })
        print(f"  {backbones.PRETTY[name]:20s} CV {scores.mean():.4f} ± {scores.std():.4f}"
              f"   held-out {held_out:.4f}")

    table = pd.DataFrame(rows)
    args.out.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out / "cnn_linear_probe.csv", index=False)
    print("\n" + table[["backbone", "params_m", "dim", "cv_mean", "held_out",
                        "embed_ms_per_image"]].to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    # Differences this small need a paired test before any ranking claim -- the same
    # discipline assignment 1 arrived at over the vein features.
    from scipy.stats import ttest_rel
    names = list(fold_scores)
    print("\nKiểm định t ghép cặp giữa các backbone (trên cùng các fold):")
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            p = float(ttest_rel(fold_scores[a], fold_scores[b]).pvalue)
            delta = 100 * (fold_scores[a].mean() - fold_scores[b].mean())
            flag = "*" if p < 0.05 else " "
            print(f"  {backbones.PRETTY[a]:20s} vs {backbones.PRETTY[b]:20s} "
                  f"{delta:+.2f} điểm  p={p:.3f} {flag}")


if __name__ == "__main__":
    main()

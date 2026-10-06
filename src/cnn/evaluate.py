"""Confusion matrix, precision, recall and F1 -- as tables and as figures.

The assignment asks for these explicitly and asks for them illustrated, so every number
here is written twice: once to CSV (so the report can cite it without retyping) and once
to a figure.

Accuracy alone hides what matters with 32 unevenly sized classes: a species with 50
images can be wiped out entirely while overall accuracy barely moves. Per-class recall
shows that; the confusion matrix shows where the mass went.

Usage:
    python -m src.cnn.evaluate --source finetune
    python -m src.cnn.evaluate --source probe --backbone resnet18
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (ConfusionMatrixDisplay, classification_report,
                             confusion_matrix)

from . import backbones
from .data import split
from .linear_probe import load, probe


def predictions(source: str, backbone: str, results: Path, cache: Path):
    if source == "finetune":
        path = results / "cnn_finetune_predictions.npz"
        if not path.exists():
            raise SystemExit(f"thiếu {path} -- chạy `python -m src.cnn.finetune` trước")
        d = np.load(path, allow_pickle=False)
        return d["y_true"], d["y_pred"], [str(c) for c in d["classes"]], "finetune_resnet18"

    data = load(backbone, cache)
    X, y, classes = data["X"], data["y"], [str(c) for c in data["classes"]]
    train_idx, test_idx = split(y)
    model = probe().fit(X[train_idx], y[train_idx])
    return y[test_idx], model.predict(X[test_idx]), classes, f"probe_{backbone}"


def main() -> None:
    ap = argparse.ArgumentParser(description="Ma trận nhầm lẫn và precision/recall")
    ap.add_argument("--source", choices=("finetune", "probe"), default="finetune")
    ap.add_argument("--backbone", choices=backbones.NAMES, default="resnet18")
    ap.add_argument("--results", type=Path, default=Path("results"))
    ap.add_argument("--cache", type=Path, default=Path("cache/cnn"))
    args = ap.parse_args()

    y_true, y_pred, classes, tag = predictions(args.source, args.backbone,
                                               args.results, args.cache)
    labels = np.arange(len(classes))
    accuracy = float((y_true == y_pred).mean())

    report = classification_report(y_true, y_pred, labels=labels, target_names=classes,
                                   output_dict=True, zero_division=0)
    per_class = pd.DataFrame(
        [{"species": c, "precision": report[c]["precision"], "recall": report[c]["recall"],
          "f1": report[c]["f1-score"], "support": int(report[c]["support"])} for c in classes]
    ).sort_values("f1")
    per_class.to_csv(args.results / f"cnn_{tag}_per_class.csv", index=False)

    summary = pd.DataFrame([{
        "tag": tag, "accuracy": accuracy,
        "macro_precision": report["macro avg"]["precision"],
        "macro_recall": report["macro avg"]["recall"],
        "macro_f1": report["macro avg"]["f1-score"],
        "weighted_f1": report["weighted avg"]["f1-score"],
        "perfect_classes": int((per_class["f1"] == 1.0).sum()),
        "n_classes": len(classes),
    }])
    summary.to_csv(args.results / f"cnn_{tag}_summary.csv", index=False)

    print(f"{tag}: accuracy {accuracy:.4f}  "
          f"macro precision {report['macro avg']['precision']:.4f}  "
          f"macro recall {report['macro avg']['recall']:.4f}  "
          f"macro F1 {report['macro avg']['f1-score']:.4f}")
    print(f"{int((per_class['f1'] == 1.0).sum())}/{len(classes)} loài đạt F1 = 1,00")
    imperfect = per_class[per_class["f1"] < 1.0]
    if not imperfect.empty:
        print("\nCác loài chưa hoàn hảo:")
        print(imperfect.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    pd.DataFrame(cm, index=classes, columns=classes).to_csv(
        args.results / f"cnn_{tag}_confusion.csv")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(14, 14))
    ConfusionMatrixDisplay(cm, display_labels=classes).plot(
        ax=ax, xticks_rotation=90, colorbar=False, values_format="d")
    ax.set_title(f"{tag} — accuracy {accuracy:.4f}".replace(".", ","))
    fig.tight_layout()
    fig.savefig(args.results / f"cnn_{tag}_confusion.png", dpi=150)
    plt.close(fig)

    # Precision/recall per class, ordered by F1 so the weak classes are at the top and
    # the eye lands on them first.
    order = per_class.iloc[::-1] if len(per_class) <= 40 else per_class.head(40)
    fig, ax = plt.subplots(figsize=(9, 8))
    y_pos = np.arange(len(order))
    ax.barh(y_pos + 0.2, order["precision"], height=0.4, label="Precision")
    ax.barh(y_pos - 0.2, order["recall"], height=0.4, label="Recall")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(order["species"], fontsize=7)
    ax.set_xlim(0.0, 1.02)
    ax.axvline(1.0, color="grey", lw=0.8, ls=":")
    ax.set_xlabel("Giá trị")
    ax.set_title(f"Precision và Recall theo loài — {tag}")
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(args.results / f"cnn_{tag}_precision_recall.png", dpi=150)
    plt.close(fig)
    print(f"\n-> results/cnn_{tag}_{{confusion,precision_recall}}.png và các file .csv")


if __name__ == "__main__":
    main()

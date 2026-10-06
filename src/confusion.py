"""Print the held-out confusion matrix in a form that fits on a terminal.

A 32x32 matrix printed cell by cell is unreadable, and almost all of it is the diagonal
anyway. This prints what is worth reading -- totals, the species that were not
classified perfectly, and every off-diagonal cell -- and leaves the full matrix to the
figure src.experiments already saves.

Reads results/confusion_matrix.csv, written by `python -m src.experiments`.

Usage:
    python -m src.confusion
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarise the held-out confusion matrix")
    ap.add_argument("--csv", type=Path, default=Path("results/confusion_matrix.csv"))
    args = ap.parse_args()

    if not args.csv.exists():
        raise SystemExit(f"{args.csv} not found -- run `python -m src.experiments` first")
    cm = pd.read_csv(args.csv, index_col=0)
    matrix = cm.to_numpy()
    total, correct = int(matrix.sum()), int(matrix.trace())

    print(f"Ma trận nhầm lẫn {matrix.shape[0]}x{matrix.shape[1]} trên tập kiểm tra giữ lại")
    print(f"  đúng {correct}/{total} mẫu = {100 * correct / total:.2f}%".replace(".", ","))
    print(f"  {int((matrix.diagonal() == matrix.sum(axis=1)).sum())}/{len(cm)} loài "
          "được phân lớp đúng toàn bộ\n")

    errors = [(cm.index[i], cm.columns[j], int(v))
              for i, row in enumerate(matrix) for j, v in enumerate(row) if i != j and v]
    if not errors:
        print("Không có mẫu nào bị phân lớp sai.")
        return

    width = max(len(t) for t, _, _ in errors)
    print(f"  {'Nhãn thật':<{width}}   {'Dự đoán':<26} Số mẫu")
    for true, pred, n in sorted(errors, key=lambda e: -e[2]):
        print(f"  {true:<{width}} → {pred:<26} {n}")

    # Recall for the species that lost at least one sample.
    print("\n  Độ phủ (recall) của các loài có mẫu sai:")
    for i, species in enumerate(cm.index):
        row_total = int(matrix[i].sum())
        if matrix[i, i] < row_total:
            print(f"    {species:<{width}} {matrix[i, i]}/{row_total}")


if __name__ == "__main__":
    main()

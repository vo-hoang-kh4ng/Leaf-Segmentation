"""Measure what a YOLO epoch actually costs on this CPU before planning a schedule.

There is no GPU here, and the assignment needs three model sizes trained under an
identical protocol. Guessing the schedule and discovering halfway that `m` takes hours
per epoch would waste the week, so this times a real training epoch on a fraction of
the data and extrapolates.

`fraction` only shrinks the training split, so the measured time is the training pass
alone; validation is disabled for the same reason. Multiply by 1/fraction for a full
epoch, and remember a real run adds one validation pass per epoch on top.

Usage:
    python -m src.yolo.benchmark
    python -m src.yolo.benchmark --fraction 0.2 --models yolo11n yolo11s
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

DATA = "african-wildlife.yaml"


def time_epoch(model_name: str, imgsz: int, fraction: float, batch: int,
               workers: int) -> tuple[float, float]:
    """Return (training-pass seconds, wall-clock seconds) for one epoch.

    Ultralytics always runs a final validation pass after training, whatever `val` is
    set to, and it also warms up and builds the dataloader. Timing `model.train()` end
    to end therefore overstates an epoch badly on a short run -- the first measurement
    here came out at 6.5 min/epoch when the training pass was under 3. Callbacks around
    the epoch itself give the number that actually scales with epochs.
    """
    from ultralytics import YOLO

    marks: dict[str, float] = {}
    model = YOLO(f"{model_name}.pt")
    model.add_callback("on_train_epoch_start", lambda _t: marks.update(start=time.time()))
    model.add_callback("on_train_epoch_end", lambda _t: marks.update(end=time.time()))

    wall = time.time()
    model.train(
        data=DATA, epochs=1, imgsz=imgsz, batch=batch, workers=workers,
        fraction=fraction, device="cpu", val=False, plots=False, save=False,
        project="runs/benchmark", name=f"{model_name}_{imgsz}", exist_ok=True,
        verbose=False, seed=42,
    )
    return marks["end"] - marks["start"], time.time() - wall


def main() -> None:
    ap = argparse.ArgumentParser(description="Đo thời gian một epoch YOLO trên CPU")
    ap.add_argument("--models", nargs="+", default=["yolo11n", "yolo11s", "yolo11m"])
    ap.add_argument("--imgsz", nargs="+", type=int, default=[640, 416])
    ap.add_argument("--fraction", type=float, default=0.1)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", type=Path, default=Path("results/yolo_benchmark.csv"))
    args = ap.parse_args()

    rows = []
    for imgsz in args.imgsz:
        for model_name in args.models:
            train_s, wall_s = time_epoch(model_name, imgsz, args.fraction, args.batch,
                                         args.workers)
            full = train_s / args.fraction
            rows.append({
                "model": model_name, "imgsz": imgsz, "fraction": args.fraction,
                "train_pass_s": train_s, "wall_s": wall_s, "full_epoch_s": full,
                "epoch_min": full / 60, "h_for_50_epochs": full * 50 / 3600,
            })
            print(f"  {model_name:9s} imgsz={imgsz}  epoch trên {args.fraction:.0%} dữ liệu: "
                  f"{train_s:5.1f}s  ->  {full/60:5.1f} phút/epoch đầy đủ, "
                  f"{full*50/3600:4.1f} giờ cho 50 epoch", flush=True)

    table = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False)
    print("\n" + table.to_string(index=False, float_format=lambda v: f"{v:.2f}"))


if __name__ == "__main__":
    main()

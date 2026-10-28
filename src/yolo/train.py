"""Train yolo11 n/s/m on one task under an identical protocol, then score them.

This is the "code cấu hình huấn luyện" the assignment asks to submit: it configures and
drives Ultralytics, it does not reimplement any of it.

Three things it does beyond calling `model.train()`:

* Prints the time budget from results/yolo_benchmark.csv before starting, because a run
  here is hours, not minutes, and starting a 6-hour job by accident is easy.
* Evaluates the finished model on the **test** split, not the val split. Ultralytics
  selects `best.pt` by val mAP, so val numbers are optimistic for the same reason a
  held-out split existed in the earlier assignments.
* Writes one row per model into results/yolo_<task>_summary.csv together with the
  protocol, so the report can cite both the metrics and the conditions.

Usage:
    python -m src.yolo.train --task detect
    python -m src.yolo.train --task detect --models n          # one size only
    python -m src.yolo.train --task segment --dry-run          # show the plan, run nothing
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from .config import SIZES, VERSION, TrainConfig, weights

BENCHMARK = Path("results/yolo_benchmark.csv")


def budget(config: TrainConfig, sizes: list[str]) -> pd.DataFrame | None:
    """Estimated hours per model, from the measured epoch times."""
    if not BENCHMARK.exists():
        return None
    bench = pd.read_csv(BENCHMARK)
    bench = bench[bench.imgsz == config.imgsz]
    rows = []
    for size in sizes:
        match = bench[bench.model == f"{VERSION}{size}"]
        if match.empty:
            continue
        minutes = float(match.iloc[0]["epoch_min"]) * config.epochs
        rows.append({"model": f"{VERSION}{size}", "epoch_min": float(match.iloc[0]["epoch_min"]),
                     "hours": minutes / 60})
    return pd.DataFrame(rows) if rows else None


def train_one(size: str, config: TrainConfig) -> dict:
    from ultralytics import YOLO

    checkpoint = weights(size, config.task)
    print(f"\n=== {checkpoint}  ({config.task}, {config.imgsz}px, {config.epochs} epoch) ===",
          flush=True)
    model = YOLO(checkpoint)
    start = time.time()
    model.train(**config.train_args(size))
    elapsed = time.time() - start

    # Final score on the test split. `split="test"` only works if the dataset yaml
    # defines one; African Wildlife and carparts-seg both do.
    metrics = model.val(data=config.data, split="test", imgsz=config.imgsz,
                        device=config.device, plots=True,
                        name=f"{VERSION}{size}_test", exist_ok=True)

    record = {"model": f"{VERSION}{size}", "checkpoint": checkpoint,
              "train_hours": elapsed / 3600,
              "params_m": sum(p.numel() for p in model.model.parameters()) / 1e6}
    record.update(config.as_record())
    for key, value in metrics.results_dict.items():
        record[key.replace("metrics/", "")] = value
    return record


def main() -> None:
    ap = argparse.ArgumentParser(description="Huấn luyện yolo11 n/s/m theo cùng một giao thức")
    ap.add_argument("--task", choices=("detect", "segment"), default="detect")
    ap.add_argument("--models", nargs="+", default=list(SIZES),
                    help="các cỡ model, mặc định n s m")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--imgsz", type=int, default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--device", default=None,
                    help='"cpu", hoặc "0" cho GPU đầu tiên (Colab)')
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--dry-run", action="store_true", help="chỉ in kế hoạch, không chạy")
    args = ap.parse_args()

    overrides = {k: v for k, v in
                 (("epochs", args.epochs), ("imgsz", args.imgsz), ("batch", args.batch),
                  ("device", args.device), ("workers", args.workers))
                 if v is not None}
    config = TrainConfig(task=args.task, **overrides)
    out = args.out or Path(f"results/yolo_{args.task}_summary.csv")

    print(f"Task        : {config.task}")
    print(f"Dataset     : {config.data}")
    print(f"Giao thức   : {config.epochs} epoch, {config.imgsz}px, batch {config.batch}, "
          f"seed {config.seed}, device {config.device}")
    print(f"Model       : {', '.join(weights(s, config.task) for s in args.models)}")

    # The benchmark was measured on this machine's CPU; quoting it for a GPU run would
    # be nonsense, so it is only shown when the device matches what was measured.
    estimate = budget(config, args.models) if config.device == "cpu" else None
    if estimate is not None:
        print("\nƯớc tính thời gian (từ results/yolo_benchmark.csv):")
        print(estimate.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
        print(f"  tổng khoảng {estimate['hours'].sum():.1f} giờ")
    else:
        print("\n(chưa có benchmark; chạy `python -m src.yolo.benchmark` để ước tính thời gian)")

    if args.dry_run:
        print("\n--dry-run: không chạy gì.")
        return

    records = []
    for size in args.models:
        records.append(train_one(size, config))
        table = pd.DataFrame(records)
        out.parent.mkdir(parents=True, exist_ok=True)
        # Written after every model, not at the end: a run this long should not lose
        # three hours of results because the next model crashed.
        table.to_csv(out, index=False)
        print(f"  -> đã ghi {out}")

    columns = [c for c in ("model", "params_m", "train_hours", "mAP50(B)", "mAP50-95(B)",
                           "mAP50(M)", "mAP50-95(M)") if c in table.columns]
    print("\n" + table[columns].to_string(index=False, float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()

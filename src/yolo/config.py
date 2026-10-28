"""The one place the training protocol is defined.

The assignment compares three model sizes, so the comparison is only meaningful if the
three runs differ in exactly one thing: the checkpoint they start from. Epochs, image
size, batch, seed, optimiser and augmentation all live here, are passed verbatim to
every run, and are written into the results summary so the report can state them.

Settings chosen for a CPU-only machine, from the measurements in
results/yolo_benchmark.csv (one full epoch of African Wildlife, 1052 train images):

                 416px            640px
    yolo11n      1.3 min/epoch    2.9 min/epoch
    yolo11s      2.7 min/epoch    6.4 min/epoch
    yolo11m      7.7 min/epoch   16.4 min/epoch

At 640px the three runs together cost 21.3 hours; at 416px, 9.7 hours. With transfer
learning on ~1000 images, reaching convergence matters more than input resolution, and
the animals in this dataset are large in frame, so 416px with 50 epochs is the better
use of the same wall clock.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict

# Checkpoint families. Ultralytics appends "-seg" for segmentation weights.
SIZES = ("n", "s", "m")
VERSION = "yolo11"

DATASETS = {
    "detect": "african-wildlife.yaml",
    "segment": "carparts-seg.yaml",     # chốt lại sau khi kiểm tra kích thước
}


def weights(size: str, task: str) -> str:
    suffix = "-seg" if task == "segment" else ""
    return f"{VERSION}{size}{suffix}.pt"


@dataclass(frozen=True)
class TrainConfig:
    """Everything that must be identical across the three model sizes."""

    task: str = "detect"
    epochs: int = 50
    imgsz: int = 416
    batch: int = 8
    seed: int = 42
    workers: int = 4
    device: str = "cpu"
    # Stop early if val mAP has not improved for this many epochs. Generous because the
    # point is to compare sizes, not to find the shortest schedule; a model cut short
    # at a different epoch than its siblings would break the comparison.
    patience: int = 20
    optimizer: str = "auto"
    # Ultralytics' default augmentation is left alone: changing it per model would be
    # another difference between runs, and the defaults are what its documentation and
    # pretrained checkpoints assume.
    extra: dict = field(default_factory=dict)

    @property
    def data(self) -> str:
        return DATASETS[self.task]

    def train_args(self, size: str) -> dict:
        return {
            "data": self.data,
            "epochs": self.epochs,
            "imgsz": self.imgsz,
            "batch": self.batch,
            "seed": self.seed,
            "workers": self.workers,
            "device": self.device,
            "patience": self.patience,
            "optimizer": self.optimizer,
            # No `project`: Ultralytics already builds runs_dir/<task>/<name>, so adding
            # one nests the output twice (runs/detect/runs/detect/yolo11n).
            "name": f"{VERSION}{size}",
            "exist_ok": True,
            "plots": True,          # results.png, confusion matrix, PR/F1 curves
            "val": True,
            **self.extra,
        }

    def as_record(self) -> dict:
        record = asdict(self)
        record.pop("extra")
        record["data"] = self.data
        record["version"] = VERSION
        return record

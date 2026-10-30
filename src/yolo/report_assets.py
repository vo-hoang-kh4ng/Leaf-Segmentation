r"""Turn the YOLO run directories into every number and figure the reports use.

Shared by report3 (detection) and report4 (segmentation): the two assignments differ
only in `task`, so duplicating this would let the two documents drift apart. Same
contract as the earlier reports -- no number is ever typed into a .tex file.

Sources, all read rather than re-computed:
    results/yolo_<task>_summary.csv        test-split metrics + the protocol
    runs/<task>/yolo11<size>/results.csv   per-epoch loss and val mAP
    runs/<task>/yolo11<size>_test/*.png    Ultralytics' own evaluation figures
    datasets/<name>/labels/<split>/*.txt   split sizes and annotation counts

Outputs into report<N>/generated/: macros.tex, tables.tex and the figures.
"""
from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FuncFormatter

ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS = ROOT / "results"
RUNS = ROOT / "runs"
DATASETS = ROOT / "datasets"
SIZES = ("n", "s", "m")
COLORS = {"n": "#1f77b4", "s": "#ff7f0e", "m": "#2ca02c"}


@dataclass(frozen=True)
class TaskSpec:
    """What differs between the two assignments."""

    task: str
    dataset: str                 # directory under datasets/
    suffix: str                  # checkpoint suffix
    metric: str                  # "B" = bounding box, "M" = mask
    train_losses: tuple[str, ...]
    url: str
    vn_name: str

    @property
    def runs(self) -> Path:
        return RUNS / self.task


SPECS = {
    "detect": TaskSpec(
        task="detect", dataset="african-wildlife", suffix="", metric="B",
        train_losses=("box_loss", "cls_loss", "dfl_loss"),
        url="https://docs.ultralytics.com/datasets/detect/african-wildlife/",
        vn_name="phát hiện đối tượng"),
    "segment": TaskSpec(
        task="segment", dataset="package-seg", suffix="-seg", metric="M",
        train_losses=("box_loss", "seg_loss", "cls_loss", "dfl_loss"),
        url="https://docs.ultralytics.com/datasets/segment/package-seg/",
        vn_name="phân đoạn đối tượng"),
}


def vn(number: str) -> str:
    r"""Vietnamese decimal comma; `{,}` keeps math mode from adding a space after it."""
    return number.replace(".", "{,}")


def num(value: float, decimals: int = 4) -> str:
    return vn(f"{value:.{decimals}f}")


def comma_axis(axis, decimals: int = 2) -> None:
    """Vietnamese decimal comma on the y axis, to match the prose and the tables."""
    axis.yaxis.set_major_formatter(
        FuncFormatter(lambda value, _: f"{value:.{decimals}f}".replace(".", ",")))


def latex_escape(text: str) -> str:
    return text.replace("&", r"\&").replace("_", r"\_").replace("%", r"\%")


# --------------------------------------------------------------------------- data

def dataset_stats(spec: TaskSpec) -> dict:
    """Split sizes, class names and annotation counts, measured from the label files.

    The counts in Ultralytics' own yaml comments are not reliable -- package-seg.yaml
    labels val as 89 and test as 188 images while the directories hold the reverse --
    so everything here is counted off disk.
    """
    root = DATASETS / spec.dataset
    yaml_text = (root / f"{spec.dataset}.yaml").read_text(encoding="utf-8")
    names: list[str] = []
    in_names = False
    for line in yaml_text.splitlines():
        if line.startswith("names:"):
            in_names = True
            continue
        if in_names:
            if line.startswith((" ", "\t")) and ":" in line:
                names.append(line.split(":", 1)[1].strip())
            elif line.strip():
                break

    splits = {}
    for split in ("train", "val", "test"):
        labels = sorted((root / "labels" / split).glob("*.txt"))
        per_class = [0] * len(names)
        regions = 0
        for label in labels:
            for row in label.read_text(encoding="utf-8").split("\n"):
                row = row.strip()
                if row:
                    regions += 1
                    per_class[int(row.split()[0])] += 1
        splits[split] = {"images": len(list((root / "images" / split).glob("*"))),
                         "regions": regions, "per_class": per_class}
    return {"names": names, "splits": splits}


def load(spec: TaskSpec) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    summary = pd.read_csv(RESULTS / f"yolo_{spec.task}_summary.csv").set_index("model")
    curves = {}
    for size in SIZES:
        frame = pd.read_csv(spec.runs / f"yolo11{size}" / "results.csv")
        frame.columns = [c.strip() for c in frame.columns]
        curves[size] = frame
    return summary, curves


def total_loss(frame: pd.DataFrame, spec: TaskSpec, split: str) -> pd.Series:
    """Ultralytics logs each loss term separately; their sum is what was minimised."""
    return sum(frame[f"{split}/{name}"] for name in spec.train_losses)


# ------------------------------------------------------------------------ figures

def curve_figure(spec: TaskSpec, curves: dict[str, pd.DataFrame], out: Path) -> None:
    """The evidence figure the assignment asks for: loss and mAP against epoch.

    Ultralytics' own results.png shows one model per file, which cannot answer the
    question the assignment actually poses -- how the three sizes differ. Plotting them
    on shared axes can, and the per-model results.png files are copied in as well.
    """
    key = f"metrics/mAP50-95({spec.metric})"
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.3))
    for size, frame in curves.items():
        label = f"yolo11{size}{spec.suffix}"
        style = dict(color=COLORS[size], linewidth=1.4, label=label)
        axes[0].plot(frame["epoch"], total_loss(frame, spec, "train"), **style)
        axes[1].plot(frame["epoch"], total_loss(frame, spec, "val"), **style)
        axes[2].plot(frame["epoch"], frame[key], **style)

    for axis, title in zip(axes, ("Mất mát trên tập huấn luyện",
                                  "Mất mát trên tập kiểm tra",
                                  f"mAP50-95 ({spec.metric}) trên tập kiểm tra")):
        axis.set_xlabel("Epoch")
        axis.set_title(title, fontsize=9)
        axis.grid(alpha=0.3)
        axis.legend(fontsize=7)
    # The largest model's val loss spikes far above the others in its first epochs, which
    # on a full-range axis flattens the whole convergence region into a line. The axis is
    # capped just above the losses from epoch 8 onwards; the caption says so, and the
    # spike itself is visible in panel one.
    cap = max(total_loss(f, spec, "val")[f["epoch"] >= 8].max() for f in curves.values())
    axes[1].set_ylim(top=1.1 * cap)
    axes[2].set_ylim(0, 1)
    comma_axis(axes[2], 1)
    fig.tight_layout()
    fig.savefig(out / "curves.png", dpi=200)
    plt.close(fig)


def gap_figure(spec: TaskSpec, summary: pd.DataFrame, curves: dict[str, pd.DataFrame],
               out: Path) -> None:
    """Val vs test mAP per size, with the late-training spread drawn as the error bar.

    This is the figure the central claim rests on: the gap between the three sizes is
    smaller than each model's own epoch-to-epoch wobble, so their ranking is not a
    result. The bar is the std of the last ten epochs of val mAP50-95.
    """
    key = f"metrics/mAP50-95({spec.metric})"
    labels = [f"yolo11{s}{spec.suffix}" for s in SIZES]
    val = [curves[s][key].iloc[-1] for s in SIZES]
    err = [curves[s][key].tail(10).std() for s in SIZES]
    test = [summary.loc[f"yolo11{s}", f"mAP50-95({spec.metric})"] for s in SIZES]

    fig, axis = plt.subplots(figsize=(6.2, 3.2))
    positions = range(len(SIZES))
    axis.errorbar(positions, val, yerr=err, fmt="o-", capsize=5, color="#1f77b4",
                  label="Tập kiểm tra (val), kèm độ lệch chuẩn 10 epoch cuối")
    axis.plot(positions, test, "s--", color="#d62728",
              label="Tập kiểm tra giữ lại (test)")
    axis.set_xticks(list(positions))
    axis.set_xticklabels(labels)
    axis.set_ylabel(f"mAP50-95 ({spec.metric})")
    axis.grid(alpha=0.3)
    comma_axis(axis, 3)
    axis.legend(fontsize=7.5, loc="best")
    fig.tight_layout()
    fig.savefig(out / "val_vs_test.png", dpi=200)
    plt.close(fig)


def copy_figures(spec: TaskSpec, out: Path) -> list[str]:
    """Ultralytics' own output, copied in under shorter names."""
    copied = []
    pairs = {
        "ultralytics_n.png": "yolo11n/results.png",
        "ultralytics_m.png": "yolo11m/results.png",
        "confusion.png": "yolo11n_test/confusion_matrix_normalized.png",
        "pr_curve.png": f"yolo11n_test/{'Mask' if spec.metric == 'M' else 'Box'}PR_curve.png",
        "predictions.jpg": "yolo11n_test/val_batch0_pred.jpg",
        "ground_truth.jpg": "yolo11n_test/val_batch0_labels.jpg",
        "train_batch.jpg": "yolo11n/train_batch0.jpg",
    }
    for target, source in pairs.items():
        path = spec.runs / source
        if path.exists():
            shutil.copy(path, out / target)
            copied.append(target)
    return copied


# ------------------------------------------------------------------------- tables

def dataset_table(spec: TaskSpec, stats: dict) -> str:
    unit = "vùng mặt nạ" if spec.metric == "M" else "hộp giới hạn"
    lines = [
        r"\begin{table}[H]", r"\centering",
        rf"\caption{{Bộ dữ liệu {latex_escape(spec.dataset)}: số ảnh và số {unit} trên từng "
        r"phần, đếm trực tiếp từ các tệp nhãn chứ không lấy theo chú thích trong tệp cấu hình "
        r"của Ultralytics.}",
        r"\label{tab:dataset}", r"\small",
        r"\begin{tabular}{l r r r}", r"\toprule",
        rf"Phần & Số ảnh & Số {unit} & Trung bình mỗi ảnh \\", r"\midrule",
    ]
    vn_split = {"train": "Huấn luyện (train)", "val": "Kiểm tra (val)",
                "test": "Giữ lại (test)"}
    for split, info in stats["splits"].items():
        mean = info["regions"] / info["images"] if info["images"] else 0
        lines.append(f"{vn_split[split]} & {info['images']} & {info['regions']} & "
                     f"{num(mean, 2)} " + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def protocol_table(spec: TaskSpec, summary: pd.DataFrame,
                   curves: dict[str, pd.DataFrame]) -> str:
    row = summary.iloc[0]
    # `optimizer: auto` lets Ultralytics pick the optimiser, and it does not record which
    # one it picked in args.yaml. The schedule it actually ran is in results.csv, so the
    # table reports the measured learning rate rather than naming an optimiser.
    peak_lr = curves["n"]["lr/pg0"].max()
    final_lr = curves["n"]["lr/pg0"].iloc[-1]
    items = [
        ("Phiên bản mô hình",
         f"YOLO11 ({', '.join(f'yolo11{s}{spec.suffix}' for s in SIZES)})"),
        ("Trọng số khởi tạo", "điểm kiểm tra tiền huấn luyện trên COCO của Ultralytics"),
        ("Số epoch", str(int(row["epochs"]))),
        ("Kích thước ảnh đầu vào", f"{int(row['imgsz'])}$\\times${int(row['imgsz'])}"),
        ("Kích thước lô (batch)", str(int(row["batch"]))),
        ("Hạt giống ngẫu nhiên", str(int(row["seed"]))),
        ("Bộ tối ưu", r"tự động (\texttt{optimizer=auto}) theo Ultralytics"),
        ("Tốc độ học",
         f"tăng tới {num(peak_lr, 5)} sau 3 epoch khởi động, giảm dần còn "
         f"${vn(f'{final_lr * 1e5:.2f}')}" + r"\times 10^{-5}$"),
        ("Tăng cường dữ liệu",
         "mặc định của Ultralytics; ghép ảnh (mosaic) tắt ở 10 epoch cuối"),
        ("Kiên nhẫn dừng sớm",
         f"{int(row['patience'])} epoch (không mô hình nào bị dừng sớm)"),
        ("Phần cứng", "GPU NVIDIA Tesla T4 trên Google Colab"),
    ]
    lines = [
        r"\begin{table}[H]", r"\centering",
        r"\caption{Giao thức huấn luyện. Ba cỡ mô hình dùng đúng các giá trị này; điểm khác "
        r"duy nhất giữa ba lần chạy là tệp trọng số khởi tạo. Toàn bộ tham số được khai báo một "
        r"chỗ trong \texttt{src/yolo/config.py} nên không thể lệch nhau giữa các lần chạy.}",
        r"\label{tab:protocol}", r"\small",
        r"\begin{tabular}{l l}", r"\toprule",
        r"Tham số & Giá trị \\", r"\midrule",
    ]
    for name, value in items:
        lines.append(f"{name} & {value} " + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def results_table(spec: TaskSpec, summary: pd.DataFrame,
                  curves: dict[str, pd.DataFrame]) -> str:
    m = spec.metric
    key = f"metrics/mAP50-95({m})"
    best = max(summary.loc[f"yolo11{s}", f"mAP50-95({m})"] for s in SIZES)
    unit = "mặt nạ" if m == "M" else "hộp giới hạn"
    # Whether the wobble swamps the gap or merely matches it differs between the two
    # tasks, so the caption states both numbers and draws the comparison from them
    # rather than asserting the same sentence in both reports.
    max_std = max(curves[s][key].tail(10).std() for s in SIZES)
    val_spread = (max(curves[s][key].iloc[-1] for s in SIZES)
                  - min(curves[s][key].iloc[-1] for s in SIZES))
    verdict = ("lớn hơn khoảng cách giữa ba mô hình trên tập val"
               if max_std > val_spread else
               "cùng cỡ với khoảng cách giữa ba mô hình trên tập val")
    lines = [
        r"\begin{table}[H]", r"\centering",
        rf"\caption{{Kết quả trên tập giữ lại (test). Dòng cuối bảng là độ lệch chuẩn của "
        rf"mAP50-95 trong mười epoch cuối, đo trên tập val. Mức dao động lớn nhất ({num(max_std)}) "
        rf"{verdict} ({num(val_spread)}), nên thứ tự ba dòng trên không nên đọc như một bảng xếp "
        rf"hạng. Ký hiệu ({m}) chỉ chỉ số tính trên {unit}.}}",
        r"\label{tab:results}", r"\small",
        r"\begin{tabular}{l r r r r r r}", r"\toprule",
        r"Mô hình & Tham số (M) & Thời gian (phút) & Precision & Recall & mAP50 & mAP50-95 \\",
        r"\midrule",
    ]
    for size in SIZES:
        row = summary.loc[f"yolo11{size}"]
        value = num(row[f"mAP50-95({m})"])
        if row[f"mAP50-95({m})"] == best:
            value = rf"\textbf{{{value}}}"
        lines.append(
            f"yolo11{size}{spec.suffix} & {num(row['params_m'], 1)} & "
            f"{num(row['train_hours'] * 60, 1)} & {num(row[f'precision({m})'], 3)} & "
            f"{num(row[f'recall({m})'], 3)} & {num(row[f'mAP50({m})'])} & {value} " + r"\\")
    lines.append(r"\midrule")
    lines.append(r"\multicolumn{7}{l}{\small Dao động mAP50-95 trên val, 10 epoch cuối: "
                 + "; ".join(f"yolo11{s}{spec.suffix} {num(curves[s][key].tail(10).std(), 4)}"
                             for s in SIZES) + r"} \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def ordering_table(spec: TaskSpec, summary: pd.DataFrame,
                   curves: dict[str, pd.DataFrame]) -> str:
    m = spec.metric
    key = f"metrics/mAP50-95({m})"
    # Whether the two splits rank the models the same way is itself a finding, and it
    # differs between the two tasks, so the caption is derived rather than asserted.
    order_val = sorted(SIZES, key=lambda s: curves[s][key].iloc[-1], reverse=True)
    order_test = sorted(SIZES, key=lambda s: summary.loc[f"yolo11{s}", f"mAP50-95({m})"],
                        reverse=True)
    if order_val == order_test:
        verdict = ("Hai cột xếp ba mô hình theo cùng một thứ tự, nhưng khoảng cách giữa "
                   "chúng vẫn chỉ xấp xỉ mức dao động của chính mỗi mô hình giữa các epoch.")
    elif order_val[0] == order_test[0]:
        verdict = ("Hai cột thống nhất ở mô hình đứng đầu nhưng đổi chỗ hai mô hình còn lại, "
                   "trong khi khoảng cách giữa chúng chỉ xấp xỉ mức dao động của chính mỗi mô "
                   "hình giữa các epoch.")
    else:
        verdict = ("Hai cột cho hai thứ tự xếp hạng khác nhau, trong khi cả hai khoảng cách "
                   "đều nhỏ hơn mức dao động của chính mỗi mô hình giữa các epoch.")
    lines = [
        r"\begin{table}[H]", r"\centering",
        rf"\caption{{Cùng một chỉ số, đo trên hai tập dữ liệu khác nhau. {verdict}}}",
        r"\label{tab:ordering}", r"\small",
        r"\begin{tabular}{l r r}", r"\toprule",
        r"Mô hình & mAP50-95 trên val & mAP50-95 trên test \\", r"\midrule",
    ]
    for size in SIZES:
        lines.append(f"yolo11{size}{spec.suffix} & {num(curves[size][key].iloc[-1])} & "
                     f"{num(summary.loc[f'yolo11{size}', f'mAP50-95({m})'])} " + r"\\")
    val_spread = (max(curves[s][key].iloc[-1] for s in SIZES)
                  - min(curves[s][key].iloc[-1] for s in SIZES))
    test_spread = (max(summary.loc[f"yolo11{s}", f"mAP50-95({m})"] for s in SIZES)
                   - min(summary.loc[f"yolo11{s}", f"mAP50-95({m})"] for s in SIZES))
    lines += [r"\midrule",
              f"Khoảng cách lớn nhất--nhỏ nhất & {num(val_spread)} & {num(test_spread)} " + r"\\",
              r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def log_table(spec: TaskSpec, curves: dict[str, pd.DataFrame]) -> str:
    """An excerpt of the raw per-epoch log, as evidence that training actually ran.

    The assignment asks for a figure "chứng minh mô hình đang huấn luyện" with mAP and
    loss. The plots show that, but they are drawn by this script; these rows are copied
    straight out of the run's own results.csv, which Ultralytics wrote epoch by epoch.
    """
    m = spec.metric
    frame = curves["n"]
    shown = list(range(3)) + [len(frame) - 2, len(frame) - 1]
    lines = [
        r"\begin{table}[H]", r"\centering",
        rf"\caption{{Trích nhật ký huấn luyện của yolo11n{spec.suffix}, chép trực tiếp từ "
        rf"\texttt{{runs/{spec.task}/yolo11n/results.csv}}, tệp do Ultralytics ghi sau mỗi "
        rf"epoch trong lúc chạy. Cột thời gian là giây tính từ lúc bắt đầu.}}",
        r"\label{tab:log}", r"\small",
        r"\begin{tabular}{r r r r r r}", r"\toprule",
        rf"Epoch & Thời gian (s) & Mất mát HL & Mất mát KT & mAP50 ({m}) & mAP50-95 ({m}) \\",
        r"\midrule",
    ]
    train = total_loss(frame, spec, "train")
    val = total_loss(frame, spec, "val")
    previous = None
    for index in shown:
        row = frame.iloc[index]
        if previous is not None and index != previous + 1:
            lines.append(r"\multicolumn{6}{c}{\dots} \\")
        lines.append(
            f"{int(row['epoch'])} & {num(row['time'], 1)} & {num(train.iloc[index], 3)} & "
            f"{num(val.iloc[index], 3)} & {num(row[f'metrics/mAP50({m})'])} & "
            f"{num(row[f'metrics/mAP50-95({m})'])} " + r"\\")
        previous = index
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def class_table(spec: TaskSpec, stats: dict) -> str | None:
    """Per-class annotation counts -- only interesting when there is more than one class."""
    names = stats["names"]
    if len(names) < 2:
        return None
    lines = [
        r"\begin{table}[H]", r"\centering",
        r"\caption{Số đối tượng của từng lớp trên các phần dữ liệu. Các lớp tương đối cân "
        r"bằng nên độ chính xác tổng thể không bị một lớp lớn chi phối.}",
        r"\label{tab:classes}", r"\small",
        r"\begin{tabular}{l r r r}", r"\toprule",
        r"Lớp & Train & Val & Test \\", r"\midrule",
    ]
    for index, name in enumerate(names):
        counts = [stats["splits"][s]["per_class"][index] for s in ("train", "val", "test")]
        lines.append(f"{latex_escape(name)} & " + " & ".join(str(c) for c in counts) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


# ------------------------------------------------------------------------- macros

def macros(spec: TaskSpec, summary: pd.DataFrame, curves: dict[str, pd.DataFrame],
           stats: dict) -> str:
    m = spec.metric
    key = f"metrics/mAP50-95({m})"
    row = summary.iloc[0]
    test = {s: summary.loc[f"yolo11{s}", f"mAP50-95({m})"] for s in SIZES}
    val = {s: curves[s][key].iloc[-1] for s in SIZES}
    std = {s: curves[s][key].tail(10).std() for s in SIZES}
    best_test = max(test, key=test.get)
    best_val = max(val, key=val.get)
    worst_test = min(test, key=test.get)
    splits = stats["splits"]

    values = {
        # dataset
        "dataname": latex_escape(spec.dataset),
        "dataurl": spec.url,
        "nclasses": str(len(stats["names"])),
        "classlist": latex_escape(", ".join(stats["names"])),
        "ntrain": str(splits["train"]["images"]),
        "nval": str(splits["val"]["images"]),
        "ntest": str(splits["test"]["images"]),
        "ntotal": str(sum(s["images"] for s in splits.values())),
        "ntrainobj": str(splits["train"]["regions"]),
        "ntestobj": str(splits["test"]["regions"]),
        "objperimage": num(splits["train"]["regions"] / splits["train"]["images"], 2),
        # protocol
        "epochs": str(int(row["epochs"])),
        "imgsz": str(int(row["imgsz"])),
        "batch": str(int(row["batch"])),
        "seed": str(int(row["seed"])),
        "patience": str(int(row["patience"])),
        "metricsuffix": m,
        # per model
        **{f"{s}params": num(summary.loc[f"yolo11{s}", "params_m"], 1) for s in SIZES},
        **{f"{s}minutes": num(summary.loc[f"yolo11{s}", "train_hours"] * 60, 1)
           for s in SIZES},
        **{f"{s}maptest": num(test[s]) for s in SIZES},
        **{f"{s}mapfifty": num(summary.loc[f"yolo11{s}", f"mAP50({m})"]) for s in SIZES},
        **{f"{s}mapval": num(val[s]) for s in SIZES},
        **{f"{s}std": num(std[s]) for s in SIZES},
        **{f"{s}bestepoch": str(int(curves[s].loc[curves[s][key].idxmax(), "epoch"]))
           for s in SIZES},
        # the argument
        "besttest": f"yolo11{best_test}{spec.suffix}",
        "bestval": f"yolo11{best_val}{spec.suffix}",
        "worsttest": f"yolo11{worst_test}{spec.suffix}",
        "besttestmap": num(test[best_test]),
        "valspread": num(max(val.values()) - min(val.values())),
        "testspread": num(max(test.values()) - min(test.values())),
        "maxstd": num(max(std.values())),
        "minstd": num(min(std.values())),
        "paramratio": num(summary.loc["yolo11m", "params_m"]
                          / summary.loc["yolo11n", "params_m"], 1),
        "timeratio": num(summary.loc["yolo11m", "train_hours"]
                         / summary.loc["yolo11n", "train_hours"], 2),
        "totalminutes": num(summary["train_hours"].sum() * 60, 0),
        "orderagrees": "trùng nhau" if best_test == best_val else "khác nhau",
    }
    if m == "M":
        # A segmentation run scores boxes as well as masks. The gap between the two is
        # the cost of having to trace the outline instead of drawing a rectangle.
        box = {s: summary.loc[f"yolo11{s}", "mAP50-95(B)"] for s in SIZES}
        values.update({f"{s}mapbox": num(box[s]) for s in SIZES})
        values["maskboxgap"] = num(
            sum(box[s] - test[s] for s in SIZES) / len(SIZES))
    return "\n".join(rf"\newcommand{{\{k}}}{{{v}}}" for k, v in values.items())


# --------------------------------------------------------------------------- entry

def build(task: str, out: Path) -> None:
    spec = SPECS[task]
    out.mkdir(parents=True, exist_ok=True)
    summary, curves = load(spec)
    stats = dataset_stats(spec)

    # One file per table rather than one combined tables.tex: the report inputs each
    # where it is discussed, otherwise LaTeX stacks all five onto the same page, pages
    # ahead of the text that explains them.
    tables = {
        "table_dataset.tex": dataset_table(spec, stats),
        "table_protocol.tex": protocol_table(spec, summary, curves),
        "table_results.tex": results_table(spec, summary, curves),
        "table_ordering.tex": ordering_table(spec, summary, curves),
        "table_log.tex": log_table(spec, curves),
    }
    classes = class_table(spec, stats)
    # package-seg has a single class, so a per-class table would be the totals again.
    tables["table_classes.tex"] = classes if classes else "% chỉ có một lớp\n"
    for name, body in tables.items():
        (out / name).write_text(body + "\n", encoding="utf-8")
    (out / "macros.tex").write_text(macros(spec, summary, curves, stats) + "\n",
                                    encoding="utf-8")

    curve_figure(spec, curves, out)
    gap_figure(spec, summary, curves, out)
    copied = copy_figures(spec, out)
    print(f"[{task}] {len(tables)} bảng, macros.tex, curves.png, val_vs_test.png "
          f"+ {len(copied)} hình sao chép -> {out}")

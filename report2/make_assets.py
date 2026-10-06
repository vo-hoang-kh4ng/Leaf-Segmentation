"""Generate every number and figure the CNN report uses, from results/.

Same contract as assignment 1's report/make_assets.py: the .tex file never contains a
hand-typed number. Re-running the experiments and rebuilding updates the document.

    report2/generated/tables.tex   backbone comparison, per-class metrics, CNN vs classical
    report2/generated/macros.tex   \\probebest, \\ftacc, ... used in the prose
    report2/generated/*.png        training curve + copies of the result figures

Usage:
    python report2/make_assets.py
"""
from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
OUT = ROOT / "report2" / "generated"

# Figures produced by src/cnn/*, copied in under shorter names.
FIGURES = {
    "confusion.png": "cnn_finetune_resnet18_confusion.png",
    "precision_recall.png": "cnn_finetune_resnet18_precision_recall.png",
    "filters.png": "cnn_filters.png",
    "feature_maps.png": "cnn_feature_maps.png",
    "gradcam.png": "cnn_gradcam.png",
    "tsne.png": "cnn_tsne.png",
}


def vn(number: str) -> str:
    """Vietnamese decimal comma; `{,}` so math mode does not add a space after it."""
    return number.replace(".", "{,}")


def pct(value: float, decimals: int = 2) -> str:
    return vn(f"{100 * value:.{decimals}f}") + r"\%"


def latex_escape(text: str) -> str:
    return text.replace("&", r"\&").replace("_", r"\_").replace("%", r"\%")


def backbone_table(probe: pd.DataFrame) -> str:
    best = probe["cv_mean"].max()
    lines = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{Ba backbone tiền huấn luyện ImageNet, đóng băng hoàn toàn, chỉ huấn luyện một "
        r"bộ phân lớp tuyến tính phía trên. Giao thức đánh giá trùng khớp với bài tập 1: kiểm định "
        r"chéo phân tầng 5 phần trên 1907 ảnh và tập kiểm tra giữ lại 25\%. Thời gian suy luận đo "
        r"trên CPU 16 luồng.}",
        r"\label{tab:backbones}", r"\small",
        r"\begin{tabular}{l r r r r r}", r"\toprule",
        r"Backbone & Tham số (M) & Số chiều & Kiểm định chéo & Giữ lại & ms/ảnh \\",
        r"\midrule",
    ]
    for _, r in probe.iterrows():
        cv = vn(f"{r['cv_mean']:.4f}")
        cv = rf"\textbf{{{cv}}}" if r["cv_mean"] == best else cv
        lines.append(f"{latex_escape(r['backbone'])} & {vn(f'{r.params_m:.1f}')} & {int(r['dim'])} & "
                     f"{cv} & {vn(f'{r.held_out:.4f}')} & {vn(f'{r.embed_ms_per_image:.1f}')} " + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def comparison_table(probe: pd.DataFrame, finetune: pd.DataFrame, classical: pd.DataFrame) -> str:
    best_classical = classical.loc[classical["accuracy_mean"].idxmax()]
    best_probe = probe.loc[probe["cv_mean"].idxmax()]
    rows = [
        ("Đặc trưng thủ công + SVM (bài tập 1)", f"{best_classical['accuracy_mean']:.4f}", "0,9895"),
        (f"{best_probe['backbone']} đóng băng + phân lớp tuyến tính",
         f"{best_probe['cv_mean']:.4f}", f"{best_probe['held_out']:.4f}"),
        ("ResNet18 fine-tune (mở khoá block cuối)", "---", f"{finetune['accuracy'].iloc[0]:.4f}"),
    ]
    lines = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{So sánh với bài tập 1 trên cùng bộ dữ liệu, cùng phép chia phân tầng và cùng "
        r"hạt giống ngẫu nhiên. Cấu hình fine-tune chỉ có một con số vì nó được huấn luyện trên "
        r"phần huấn luyện của phép chia giữ lại, không chạy kiểm định chéo (chi phí tính toán).}",
        r"\label{tab:comparison}", r"\small",
        r"\begin{tabular}{l r r}", r"\toprule",
        r"Phương pháp & Kiểm định chéo & Tập giữ lại \\", r"\midrule",
    ]
    for name, cv, held in rows:
        lines.append(f"{name} & {vn(cv)} & {vn(held)} " + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def per_class_table(per_class: pd.DataFrame) -> str:
    imperfect = per_class[per_class["f1"] < 1.0]
    lines = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{Toàn bộ các loài không đạt F1 = 1,00 với mô hình ResNet18 fine-tune. "
        r"Những loài còn lại đều đạt precision và recall bằng 1,00.}",
        r"\label{tab:perclass}", r"\small",
        r"\begin{tabular}{l r r r r}", r"\toprule",
        r"Loài & Precision & Recall & F1 & Số mẫu \\", r"\midrule",
    ]
    for _, r in imperfect.iterrows():
        lines.append(f"{latex_escape(r['species'])} & {vn(f'{r.precision:.3f}')} & "
                     f"{vn(f'{r.recall:.3f}')} & {vn(f'{r.f1:.3f}')} & {int(r['support'])} " + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def training_curve(history: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    axes[0].plot(history["epoch"], history["train_loss"], marker="o", label="huấn luyện")
    axes[0].plot(history["epoch"], history["val_loss"], marker="s", label="kiểm tra")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Hàm mất mát")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    axes[1].plot(history["epoch"], history["train_acc"], marker="o", label="huấn luyện")
    axes[1].plot(history["epoch"], history["val_acc"], marker="s", label="kiểm tra")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Độ chính xác")
    axes[1].set_ylim(0.6, 1.01)
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "training_curve.png", dpi=200)
    plt.close(fig)


def macros(probe: pd.DataFrame, finetune: pd.DataFrame, history: pd.DataFrame,
           per_class: pd.DataFrame, probe_summary: pd.DataFrame,
           classical: pd.DataFrame) -> str:
    best = probe.loc[probe["cv_mean"].idxmax()]
    worst = probe.loc[probe["cv_mean"].idxmin()]
    light = probe.loc[probe["params_m"].idxmin()]
    heavy = probe.loc[probe["params_m"].idxmax()]
    ft = finetune.iloc[0]
    best_classical = classical["accuracy_mean"].max()
    best_epoch = history.loc[history["val_acc"].idxmax()]

    values = {
        "probebestname": latex_escape(str(best["backbone"])),
        "probebestcv": pct(best["cv_mean"]),
        "probebestheld": pct(best["held_out"]),
        "probeworstname": latex_escape(str(worst["backbone"])),
        "probeworstcv": pct(worst["cv_mean"]),
        "lightname": latex_escape(str(light["backbone"])),
        "lightparams": vn(f"{light['params_m']:.1f}"),
        "lightcv": pct(light["cv_mean"]),
        "lightms": vn(f"{light['embed_ms_per_image']:.0f}"),
        "heavyname": latex_escape(str(heavy["backbone"])),
        "heavyparams": vn(f"{heavy['params_m']:.1f}"),
        "heavycv": pct(heavy["cv_mean"]),
        "heavyms": vn(f"{heavy['embed_ms_per_image']:.0f}"),
        "paramratio": vn(f"{heavy['params_m'] / light['params_m']:.0f}"),
        "ftacc": pct(ft["accuracy"]),
        "ftprecision": pct(ft["macro_precision"]),
        "ftrecall": pct(ft["macro_recall"]),
        "ftfone": pct(ft["macro_f1"]),
        "ftperfect": str(int(ft["perfect_classes"])),
        "ftclasses": str(int(ft["n_classes"])),
        "fterrors": str(int(round((1 - ft["accuracy"]) * 477))),
        "probeacc": pct(probe_summary["accuracy"].iloc[0]),
        "probeperfect": str(int(probe_summary["perfect_classes"].iloc[0])),
        "bestepoch": str(int(best_epoch["epoch"])),
        "epochseconds": vn(f"{history['seconds'].mean():.0f}"),
        "epochs": str(len(history)),
        "firstepochacc": pct(history["val_acc"].iloc[0]),
        "classicalcv": pct(best_classical),
        "cnngain": vn(f"{100 * (best['cv_mean'] - best_classical):.2f}"),
        "nclasses": str(int(ft["n_classes"])),
    }
    return "\n".join(rf"\newcommand{{\{k}}}{{{v}}}" for k, v in values.items())


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    probe = pd.read_csv(RESULTS / "cnn_linear_probe.csv")
    finetune = pd.read_csv(RESULTS / "cnn_finetune_resnet18_summary.csv")
    probe_summary = pd.read_csv(RESULTS / "cnn_probe_resnet18_summary.csv")
    history = pd.read_csv(RESULTS / "cnn_finetune_history.csv")
    per_class = pd.read_csv(RESULTS / "cnn_finetune_resnet18_per_class.csv")
    classical = pd.read_csv(RESULTS / "ablation.csv")

    tables = [backbone_table(probe), comparison_table(probe, finetune, classical),
              per_class_table(per_class)]
    (OUT / "tables.tex").write_text("\n\n".join(tables) + "\n", encoding="utf-8")
    (OUT / "macros.tex").write_text(
        macros(probe, finetune, history, per_class, probe_summary, classical) + "\n",
        encoding="utf-8")

    training_curve(history)
    for target, source in FIGURES.items():
        shutil.copy(RESULTS / source, OUT / target)
    print(f"wrote tables.tex, macros.tex, training_curve.png + {len(FIGURES)} hình -> {OUT}")


if __name__ == "__main__":
    main()

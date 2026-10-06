"""Generate every number and figure the report uses, straight from results/.

The report must never hard-code an accuracy: re-running the experiments and
re-compiling should update the document. This script emits

    report/generated/tables.tex    the ablation table + error breakdown
    report/generated/macros.tex    \\bestacc, \\shapeonlyacc, ... used in the prose
    report/generated/ablation.png  grouped bar chart
    report/generated/confusion.png copy of the held-out confusion matrix

Usage:
    python report/make_assets.py
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
OUT = ROOT / "report" / "generated"

# Ablation rows are printed in this order: baseline first, then increasing groups, so
# the table reads as a narrative rather than alphabetically.
ORDER = [
    "shape",
    "shape+color",
    "shape+texture",
    "shape+vein",
    "shape+color+texture",
    "shape+color+vein",
    "shape+texture+vein",
    "shape+color+texture+vein",
]
PRETTY = {
    "shape": "Hình dạng (cấu hình cơ sở)",
    "shape+color": "Hình dạng + Màu",
    "shape+texture": "Hình dạng + Kết cấu",
    "shape+vein": "Hình dạng + Gân lá",
    "shape+color+texture": "Hình dạng + Màu + Kết cấu",
    "shape+color+vein": "Hình dạng + Màu + Gân lá",
    "shape+texture+vein": "Hình dạng + Kết cấu + Gân lá",
    "shape+color+texture+vein": "Tất cả bốn nhóm",
}


def vn(number: str) -> str:
    """Vietnamese decimal comma for anything printed in the report.

    `{,}` rather than a bare comma: inside math mode LaTeX would otherwise add a space
    after the comma, as it does after a list separator.
    """
    return number.replace(".", "{,}")


def latex_escape(text: str) -> str:
    return text.replace("&", r"\&").replace("_", r"\_").replace("%", r"\%")


def ablation_table(table: pd.DataFrame) -> str:
    pivot = table.pivot(index="features", columns="classifier", values="accuracy_mean")
    counts = table.groupby("features")["n_features"].first()
    classifiers = list(pivot.columns)
    best_value = pivot.to_numpy().max()

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Độ chính xác trung bình theo kiểm định chéo phân tầng 5 phần trên Flavia "
        r"(1907 ảnh, 32 loài). Giá trị in đậm là kết quả cao nhất của toàn bảng.}",
        r"\label{tab:ablation}",
        r"\small",
        r"\begin{tabular}{l r " + " ".join(["r"] * len(classifiers)) + "}",
        r"\toprule",
        r"Nhóm đặc trưng & \#chiều & " + " & ".join(latex_escape(c) for c in classifiers) + r" \\",
        r"\midrule",
    ]
    for key in ORDER:
        if key not in pivot.index:
            continue
        cells = []
        for c in classifiers:
            v = pivot.loc[key, c]
            cell = vn(f"{v:.4f}")
            cells.append(rf"\textbf{{{cell}}}" if v == best_value else cell)
        lines.append(f"{PRETTY[key]} & {counts[key]} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def error_table(cm: pd.DataFrame) -> str:
    errors = [
        (cm.index[i], cm.columns[j], int(v))
        for i, row in enumerate(cm.to_numpy())
        for j, v in enumerate(row)
        if i != j and v
    ]
    errors.sort(key=lambda e: -e[2])
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Toàn bộ mẫu bị phân lớp sai trên tập held-out 25\%.}",
        r"\label{tab:errors}",
        r"\small",
        r"\begin{tabular}{l l r}",
        r"\toprule",
        r"Nhãn thật & Dự đoán & Số mẫu \\",
        r"\midrule",
    ]
    for true, pred, n in errors:
        lines.append(f"{latex_escape(true)} & {latex_escape(pred)} & {n} " + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def macros(table: pd.DataFrame, cm: pd.DataFrame) -> str:
    pivot = table.pivot(index="features", columns="classifier", values="accuracy_mean")
    best = table.loc[table["accuracy_mean"].idxmax()]
    total = int(cm.to_numpy().sum())
    correct = int(cm.to_numpy().trace())

    def pct(v: float) -> str:
        return vn(f"{100 * v:.2f}") + r"\%"

    values = {
        "bestacc": pct(best["accuracy_mean"]),
        "bestmodel": latex_escape(str(best["classifier"])),
        "bestfeatures": PRETTY[str(best["features"])],
        "bestdims": str(int(best["n_features"])),
        # Fold-to-fold std of the best configuration, in percentage points: the yardstick
        # every "is this difference real?" argument in section 6 is measured against.
        "beststd": vn(f"{100 * best['accuracy_std']:.2f}"),
        "shapeonlyacc": pct(pivot.loc["shape", "SVM (RBF)"]),
        "shapeonlyrf": pct(pivot.loc["shape", "Random Forest"]),
        "shapecoloracc": pct(pivot.loc["shape+color", "SVM (RBF)"]),
        "allgroupsacc": pct(pivot.loc["shape+color+texture+vein", "SVM (RBF)"]),
        "colorgain": pct(pivot.loc["shape+color", "SVM (RBF)"] - pivot.loc["shape", "SVM (RBF)"]),
        "texturegain": pct(
            pivot.loc["shape+color+texture", "SVM (RBF)"] - pivot.loc["shape+color", "SVM (RBF)"]
        ),
        "veindelta": pct(
            pivot.loc["shape+color+texture+vein", "SVM (RBF)"]
            - pivot.loc["shape+color+texture", "SVM (RBF)"]
        ),
        "holdoutacc": pct(correct / total),
        "holdouttotal": str(total),
        "holdouterrors": str(total - correct),
    }

    # Per-group dimensionality, read from the cached feature names rather than typed
    # into the section headings -- a hand-written "16 chiều" for a 17-dimensional group
    # is exactly the error this closes.
    cache = ROOT / "cache" / "features.npz"
    if cache.exists():
        import numpy as np

        names = [str(n) for n in np.load(cache, allow_pickle=False)["feature_names"]]
        for group in ("shape", "color", "texture", "vein"):
            values[f"{group}dims"] = str(sum(n.startswith(f"{group}.") for n in names))
        values["totaldims"] = str(len(names))
    else:
        print(f"warning: {cache} missing, per-group dimension macros not generated")
    return "\n".join(rf"\newcommand{{\{k}}}{{{v}}}" for k, v in values.items())


def ablation_figure(table: pd.DataFrame) -> None:
    pivot = table.pivot(index="features", columns="classifier", values="accuracy_mean")
    pivot = pivot.reindex([k for k in ORDER if k in pivot.index])

    fig, ax = plt.subplots(figsize=(9, 4.2))
    width = 0.8 / len(pivot.columns)
    for i, clf in enumerate(pivot.columns):
        positions = [x + i * width for x in range(len(pivot))]
        ax.bar(positions, pivot[clf], width, label=clf)

    ax.set_xticks([x + 0.4 - width / 2 for x in range(len(pivot))])
    ax.set_xticklabels([k.replace("+", "\n+") for k in pivot.index], fontsize=8)
    # Start at 0.75: every bar is above 0.78, so a 0-based axis would compress all the
    # differences the ablation is about into the top 20% of the plot.
    ax.set_ylim(0.75, 1.0)
    ax.set_ylabel("Accuracy (5-fold CV)")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8, ncol=4, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "ablation.png", dpi=200)


def redundancy_macros(mi: pd.DataFrame, overlap: pd.DataFrame, imp: pd.DataFrame) -> str:
    """Numbers for section 6.1's direct overlap measurement."""
    def corr(pair: str, field: str) -> float:
        return float(overlap[overlap["pair"] == pair].iloc[0][field])

    values = {
        "ccacolortexture": f"{corr('color-texture', 'cca_1'):.3f}".replace(".", "{,}"),
        "ccatextureshape": f"{corr('texture-shape', 'cca_1'):.3f}".replace(".", "{,}"),
        "ccacolorshape": f"{corr('color-shape', 'cca_1'):.3f}".replace(".", "{,}"),
        "rtwocolortexture": f"{corr('color-texture', 'r2_forward'):.3f}".replace(".", "{,}"),
        "rtwotexturecolor": f"{corr('color-texture', 'r2_backward'):.3f}".replace(".", "{,}"),
    }
    for _, r in mi.iterrows():
        values[f"mi{r['group']}"] = f"{r['mi_mean']:.3f}".replace(".", "{,}")
    for _, r in imp.iterrows():
        values[f"imp{r['group']}"] = f"{r['importance_sum']:.4f}".replace(".", "{,}")
    return "\n".join(rf"\newcommand{{\{k}}}{{{v}}}" for k, v in values.items())


def vein_table(df: pd.DataFrame) -> str:
    pivot = df.pivot(index="vein_variant", columns="classifier", values="accuracy_mean")
    order = [i for i in [
        "(không dùng gân lá)", "vein (Otsu -- baseline)",
        "vein_pct (ngưỡng cố định)", "vein_frangi (Frangi)",
    ] if i in pivot.index]
    classifiers = list(pivot.columns)

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Ba phương pháp trích xuất gân lá, cùng ghép với nhóm \emph{Hình dạng + Màu + Kết cấu} "
        r"và đánh giá trên cùng các phần kiểm định chéo. Không ô nào khác biệt có ý nghĩa thống kê so với "
        r"dòng đầu (kiểm định $t$ ghép cặp; giá trị $p$ nhỏ nhất là "
        + vn(f"{df['paired_p'].min():.3f}") + r" $> 0{,}05$).}",
        r"\label{tab:vein}",
        r"\small",
        r"\begin{tabular}{l " + " ".join(["r"] * len(classifiers)) + "}",
        r"\toprule",
        r"Cách trích xuất gân lá & " + " & ".join(latex_escape(c) for c in classifiers) + r" \\",
        r"\midrule",
    ]
    # Row labels in the CSV are the experiment's internal keys (vein_macros looks them up
    # verbatim); the table shows reader-facing names instead.
    display = {
        "(không dùng gân lá)": "Không dùng gân lá",
        "vein (Otsu -- baseline)": "Ngưỡng Otsu (cách làm gốc)",
        "vein_pct (ngưỡng cố định)": "Ngưỡng cố định tương đối",
        "vein_frangi (Frangi)": "Bộ lọc Frangi",
    }
    for key in order:
        cells = [vn(f"{pivot.loc[key, c]:.4f}") for c in classifiers]
        lines.append(display.get(key, latex_escape(key)) + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def vein_macros(df: pd.DataFrame) -> str:
    """Deltas and p-values for the prose. Signed percentage points, not ratios."""
    def pick(variant: str, clf: str, field: str) -> float:
        row = df[(df["vein_variant"] == variant) & (df["classifier"] == clf)].iloc[0]
        return float(row[field])

    keys = {
        "veinotsu": "vein (Otsu -- baseline)",
        "veinpct": "vein_pct (ngưỡng cố định)",
        "veinfrangi": "vein_frangi (Frangi)",
    }
    out = []
    for short, variant in keys.items():
        for clf, tag in (("SVM (RBF)", "svm"), ("KNN (k=5)", "knn")):
            delta = vn(f"{100 * pick(variant, clf, 'delta_vs_no_vein'):+.2f}")
            p_val = vn(f"{pick(variant, clf, 'paired_p'):.2f}")
            out.append(rf"\newcommand{{\{short}{tag}}}{{{delta}}}")
            out.append(rf"\newcommand{{\{short}{tag}p}}{{{p_val}}}")
    return "\n".join(out)


_INV_ROWS = ["identity", "rotate 15°", "rotate 45°", "rotate 90°", "scale 0.5",
             "bright ×1.3", "dark ×0.7", "noise σ=10", "occlude 10%"]
_INV_COLS = ["shape", "color", "color RGB", "color HSV", "texture", "vein",
             "shape+color+texture"]


def invariance_figure(df: pd.DataFrame) -> None:
    pivot = df.pivot(index="transform", columns="features", values="accuracy")
    pivot = pivot.reindex(index=[r for r in _INV_ROWS if r in pivot.index],
                          columns=[c for c in _INV_COLS if c in pivot.columns])

    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    im = ax.imshow(pivot.to_numpy(), cmap="RdYlGn", vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=8)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.iat[i, j]
            # Black on the light middle of the colormap, white on the dark extremes.
            colour = "black" if 0.25 < v < 0.85 else "white"
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7, color=colour)
    fig.colorbar(im, ax=ax, shrink=0.8, label="Accuracy")
    fig.tight_layout()
    fig.savefig(OUT / "invariance.png", dpi=200)


def invariance_macros(acc: pd.DataFrame, drift: pd.DataFrame) -> str:
    def a(transform: str, features: str) -> str:
        row = acc[(acc["transform"] == transform) & (acc["features"] == features)]
        return f"{float(row.iloc[0]['accuracy']):.4f}".replace(".", "{,}")

    def d(transform: str, block: str) -> str:
        row = drift[(drift["transform"] == transform) & (drift["block"] == block)]
        return f"{float(row.iloc[0]['drift_mean']):.2f}".replace(".", "{,}")

    values = {
        "invcontrol": a("identity", "shape+color+texture"),
        "invtexid": a("identity", "texture"),
        "invtexrotninety": a("rotate 90°", "texture"),
        "invtexrotfifteen": a("rotate 15°", "texture"),
        "invshapeid": a("identity", "shape"),
        "invshaperotninety": a("rotate 90°", "shape"),
        "invshapescale": a("scale 0.5", "shape"),
        "invhsvid": a("identity", "color HSV"),
        "invhsvdark": a("dark ×0.7", "color HSV"),
        "invrgbdark": a("dark ×0.7", "color RGB"),
        "invhsvnoise": a("noise σ=10", "color HSV"),
        "invrgbnoise": a("noise σ=10", "color RGB"),
        "invshapenoise": a("noise σ=10", "shape"),
        "invcombonoise": a("noise σ=10", "shape+color+texture"),
        "invshapeoccl": a("occlude 10%", "shape"),
        "invcombooccl": a("occlude 10%", "shape+color+texture"),
        "driftglcmninety": d("rotate 90°", "texture_glcm"),
        "driftlbpninety": d("rotate 90°", "texture_lbp"),
        "driftglcmfifteen": d("rotate 15°", "texture_glcm"),
        "driftlbpfifteen": d("rotate 15°", "texture_lbp"),
        "driftsmoothscale": d("scale 0.5", "WORST:shape.smooth_factor"),
        "driftsmoothninety": d("rotate 90°", "WORST:shape.smooth_factor"),
    }
    return "\n".join(rf"\newcommand{{\{k}}}{{{v}}}" for k, v in values.items())


def projection_figure() -> str:
    """PCA/LDA figure and its numbers, from src.projection. Returns macros.

    The computation lives in src/projection.py so the demo video (which runs that module
    on screen) and the report show the same figure and the same numbers.
    """
    import sys

    import numpy as np

    sys.path.insert(0, str(ROOT))
    from src.projection import LDA_AXES, analyse, plot

    cache = ROOT / "cache" / "features.npz"
    if not cache.exists():
        print(f"warning: {cache} missing, projection figure not generated")
        return ""
    data = dict(np.load(cache, allow_pickle=False))
    plot(data, OUT / "projection.png")
    r = analyse(data)

    words = {2: "two", 10: "ten"}
    macros: dict[str, str] = {}
    for key in ("shape", "best"):
        macros[f"pcavar{key}"] = vn(f"{100 * r[f'pca_var_{key}']:.1f}")
        macros[f"ldavar{key}"] = vn(f"{100 * r[f'lda_var_{key}']:.1f}")
        for k in LDA_AXES:
            macros[f"ldaknn{key}{words[k]}"] = vn(f"{100 * r[f'lda_knn_{key}_{k}']:.1f}") + r"\%"
    macros["ldaoutliers"] = ", ".join(rf"\emph{{{s}}}" for s in r["lda_outliers"])
    return "\n".join(rf"\newcommand{{\{k}}}{{{v}}}" for k, v in macros.items())


def segmentation_figure(image_id: str = "3267") -> bool:
    """Four panels showing what `preprocess.stages` produces on one real leaf.

    Image 3267 is chosen deliberately: morphology changes ~21k pixels there (filling
    holes in the bright midrib), against a median of ~1.5k across the dataset, so
    panels (b) and (c) actually differ. On a typical leaf they look identical.

    Returns False (with a warning) when the dataset is absent, so the report still
    builds on a checkout without data/ -- the previously generated PNG is kept.
    """
    import sys

    sys.path.insert(0, str(ROOT))
    import cv2

    from src.preprocess import stages

    path = ROOT / "data" / "flavia" / f"{image_id}.jpg"
    if not path.exists():
        print(f"warning: {path} missing, keeping existing segmentation.png")
        return False

    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    s = stages(bgr, path)

    # Crop every panel to the leaf's bounding box (plus a margin): at full frame the
    # leaf occupies ~10-20% of the image and the interesting detail is invisible.
    x, y, w, h = cv2.boundingRect(s["contour"])
    pad = int(0.06 * max(w, h))
    y0, y1 = max(0, y - pad), min(bgr.shape[0], y + h + pad)
    x0, x1 = max(0, x - pad), min(bgr.shape[1], x + w + pad)

    def crop(img):
        return img[y0:y1, x0:x1]

    overlay = bgr.copy()
    cv2.drawContours(overlay, [s["contour"]], -1, (0, 0, 255), 6)

    panels = [
        (crop(bgr[:, :, ::-1]), "(a) Ảnh gốc", None),
        (crop(s["otsu"]), "(b) Ngưỡng Otsu", "gray"),
        (crop(s["cleaned"]), "(c) Sau closing + opening", "gray"),
        (crop(overlay[:, :, ::-1]), "(d) Contour lớn nhất, tô đặc", None),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.4))
    for ax, (img, title, cmap) in zip(axes, panels):
        ax.imshow(img, cmap=cmap)
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(OUT / "segmentation.png", dpi=170, bbox_inches="tight")
    return True


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    table = pd.read_csv(RESULTS / "ablation.csv")
    cm = pd.read_csv(RESULTS / "confusion_matrix.csv", index_col=0)

    tables = [ablation_table(table), error_table(cm)]
    macro_text = macros(table, cm)

    # The vein-variant experiment is optional: the report still builds (minus that
    # subsection's numbers) if src.vein_variants has not been run.
    # Redundancy analysis (section 6.1): numbers only, no table of its own.
    mi_csv, overlap_csv, imp_csv = (
        RESULTS / "redundancy_mi.csv",
        RESULTS / "redundancy_overlap.csv",
        RESULTS / "redundancy_importance.csv",
    )
    if mi_csv.exists() and overlap_csv.exists() and imp_csv.exists():
        macro_text += "\n" + redundancy_macros(
            pd.read_csv(mi_csv), pd.read_csv(overlap_csv), pd.read_csv(imp_csv)
        )
    else:
        print("warning: redundancy_*.csv missing, section 6.1 macros not generated")

    projection = projection_figure()
    if projection:
        macro_text += "\n" + projection

    inv_csv, drift_csv = RESULTS / "invariance.csv", RESULTS / "invariance_drift.csv"
    if inv_csv.exists() and drift_csv.exists():
        inv, drift = pd.read_csv(inv_csv), pd.read_csv(drift_csv)
        invariance_figure(inv)
        macro_text += "\n" + invariance_macros(inv, drift)
    else:
        print("warning: invariance CSVs missing, section 6.6 assets not generated")

    vein_csv = RESULTS / "vein_variants.csv"
    if vein_csv.exists():
        vein = pd.read_csv(vein_csv)
        tables.append(vein_table(vein))
        macro_text += "\n" + vein_macros(vein)
    else:
        print(f"warning: {vein_csv} missing, vein-variant table not generated")

    (OUT / "tables.tex").write_text("\n\n".join(tables) + "\n", encoding="utf-8")
    (OUT / "macros.tex").write_text(macro_text + "\n", encoding="utf-8")
    ablation_figure(table)
    shutil.copy(RESULTS / "confusion_matrix.png", OUT / "confusion.png")
    written = ["tables.tex", "macros.tex", "ablation.png", "confusion.png"]
    if segmentation_figure():
        written.append("segmentation.png")
    print(f"wrote {', '.join(written)} -> {OUT}")


if __name__ == "__main__":
    main()

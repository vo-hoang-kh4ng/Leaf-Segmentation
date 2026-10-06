"""Scripted walkthrough for the assignment-2 demo video (≤ 5 minutes, no narration).

Same idea as src/walkthrough.py: each step prints a heading and a caption saying what is
about to run and what to look at, shows the command, runs it for real, opens the figure,
then waits for Enter.

One trap avoided: the live fine-tuning step runs a single epoch into a throwaway output
directory. Running it against results/ would overwrite the 6-epoch history, the best
checkpoint and the predictions that the report's confusion matrix is built from.

Usage:
    python -m src.cnn.walkthrough
    python -m src.cnn.walkthrough --auto 6
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
BOLD, CYAN, GREEN, DIM, RESET = "\033[1m", "\033[96m", "\033[92m", "\033[2m", "\033[0m"


def report_numbers() -> dict[str, str]:
    """Headline numbers from the report's generated macros, so this can never disagree."""
    path = ROOT / "report2" / "generated" / "macros.tex"
    if not path.exists():
        return {}
    numbers = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("\\newcommand{\\"):
            name, _, value = line[len("\\newcommand{\\"):].partition("}{")
            numbers[name] = value[:-1].replace("{,}", ",").replace("\\%", "%")
    return numbers


class Walkthrough:
    def __init__(self, auto: float | None, open_figures: bool) -> None:
        self.auto, self.open_figures, self.step = auto, open_figures, 0

    def pause(self) -> None:
        if self.auto is None:
            input(f"\n{DIM}[Enter để tiếp tục]{RESET}")
        else:
            import time
            time.sleep(self.auto)

    def heading(self, title: str, *caption: str) -> None:
        self.step += 1
        bar = "═" * 78
        print(f"\n{CYAN}{bar}\n{BOLD} BƯỚC {self.step}. {title}{RESET}\n{CYAN}{bar}{RESET}")
        for line in caption:
            print(f" {line}")
        print()

    def run(self, *args: str) -> None:
        print(f"{GREEN}$ python -m {' '.join(args)}{RESET}\n", flush=True)
        subprocess.run([sys.executable, "-m", *args], cwd=ROOT,
                       env=dict(os.environ, PYTHONIOENCODING="utf-8"), check=True)

    def show(self, path: Path) -> None:
        if not path.exists():
            print(f"{DIM}(không tìm thấy {path}){RESET}")
            return
        if self.open_figures and sys.platform == "win32":
            os.startfile(path)  # noqa: S606
        else:
            print(f"{DIM}mở hình: {path}{RESET}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Walkthrough cho video demo bài tập 2")
    ap.add_argument("--auto", type=float, default=None)
    ap.add_argument("--no-open", action="store_true")
    args = ap.parse_args()

    if sys.platform == "win32":
        os.system("")
    sys.stdout.reconfigure(encoding="utf-8")
    w = Walkthrough(args.auto, open_figures=not args.no_open)
    n = report_numbers()
    results = ROOT / "results"

    print(f"\n{BOLD}PHÂN LỚP LÁ CÂY TRÊN FLAVIA BẰNG MẠNG NƠ-RON TÍCH CHẬP{RESET}")
    print(" 32 loài, 1907 ảnh. Học chuyển giao từ ImageNet với ResNet18, MobileNetV3 và VGG11.")
    print(" Dùng đúng phép chia dữ liệu của bài tập 1 để so sánh được trực tiếp.")
    w.pause()

    w.heading("Trích đặc trưng bằng backbone đóng băng",
              "Đưa 1907 ảnh qua MobileNetV3 tiền huấn luyện, lấy vectơ 576 chiều mỗi ảnh.",
              "Không một trọng số nào trong mạng được huấn luyện lại trên ảnh lá.")
    w.run("src.cnn.embed", "--backbone", "mobilenet_v3_small", "--force")
    w.pause()

    w.heading("So sánh ba kiến trúc (chỉ huấn luyện một tầng tuyến tính phía trên)",
              "Kiểm định chéo 5 phần trên 1907 ảnh, cùng giao thức với bài tập 1.",
              "Chú ý: cả ba đều vượt 98,85% của đặc trưng thủ công, và MobileNetV3 nhỏ hơn",
              "VGG11 52 lần nhưng cho kết quả cao hơn.")
    w.run("src.cnn.linear_probe")
    w.pause()

    w.heading("Tinh chỉnh ResNet18 — chạy thật 1 epoch",
              "Mở khoá block tích chập cuối và tầng phân lớp mới, hai tốc độ học khác nhau.",
              "Video chỉ chạy 1 epoch (~45 giây); kết quả đầy đủ 6 epoch đã chạy sẵn từ trước.")
    with tempfile.TemporaryDirectory() as tmp:
        # Throwaway output: the real 6-epoch results must not be overwritten.
        w.run("src.cnn.finetune", "--epochs", "1", "--out", tmp,
              "--checkpoint", str(Path(tmp) / "demo.pt"))
    print(f"\n{DIM}Đường học của bản chạy đầy đủ 6 epoch:{RESET}")
    w.show(ROOT / "report2" / "generated" / "training_curve.png")
    w.pause()

    w.heading("Ma trận nhầm lẫn, precision và recall",
              "Mô hình tinh chỉnh tốt nhất trên 477 ảnh kiểm tra giữ lại.",
              "Chú ý: 30/32 loài đạt F1 = 1,00; chỉ còn 1 ảnh bị phân lớp sai.")
    w.run("src.cnn.evaluate", "--source", "finetune")
    w.show(results / "cnn_finetune_resnet18_confusion.png")
    w.show(results / "cnn_finetune_resnet18_precision_recall.png")
    w.pause()

    w.heading("Mạng đã học được đặc trưng gì?",
              "Bộ lọc tầng 1, bản đồ đặc trưng, Grad-CAM và t-SNE của embedding.",
              "Grad-CAM cho thấy mạng nhìn vào phiến lá chứ không bám vào nền trắng.")
    w.run("src.cnn.visualise")
    for figure in ("cnn_filters.png", "cnn_feature_maps.png", "cnn_gradcam.png", "cnn_tsne.png"):
        w.show(results / figure)
    w.pause()

    w.heading("Kết quả chính",
              f"1. Backbone ĐÓNG BĂNG + phân lớp tuyến tính: {n.get('probebestcv', '?')} "
              f"({n.get('probebestname', '?')}),",
              f"   cao hơn {n.get('cnngain', '?')} điểm so với {n.get('classicalcv', '?')} của đặc trưng thủ công.",
              f"2. Tinh chỉnh ResNet18: {n.get('ftacc', '?')} trên tập giữ lại "
              f"({n.get('fterrors', '?')}/477 ảnh sai).",
              f"3. {n.get('lightname', '?')} ({n.get('lightparams', '?')}M tham số) vượt "
              f"{n.get('heavyname', '?')} ({n.get('heavyparams', '?')}M):",
              "   số tham số không dự báo chất lượng biểu diễn chuyển giao.",
              "4. Phần lớn thành tích đến từ biểu diễn ImageNet có sẵn, không từ huấn luyện lại.")
    w.show(ROOT / "report2" / "build" / "report.pdf")
    print(f"\n{DIM}Chi tiết trong report2/build/report.pdf{RESET}\n")


if __name__ == "__main__":
    main()

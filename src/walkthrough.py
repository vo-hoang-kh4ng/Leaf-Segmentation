"""Scripted walkthrough for the silent demo video (≤ 5 minutes).

The video has no narration, so everything the viewer needs has to be on screen: each
step prints a heading and a one- or two-line caption saying what is about to run and
what to look at, shows the exact command, runs it for real, and opens the resulting
figure. Between steps it waits for Enter so whoever is recording controls the pace
(`--auto N` advances by itself after N seconds instead).

Full feature extraction takes ~9 minutes for 1907 images -- too long for the video --
so step 2 extracts a genuine 64-image subset (2 per species) into a temporary folder
and says so on screen; later steps use the cached features of the full dataset.

Usage (from the repo root, in a maximised terminal):
    python -m src.walkthrough
    python -m src.walkthrough --auto 6
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from .labels import ID_RANGES

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "flavia"

BOLD, CYAN, GREEN, DIM, RESET = "\033[1m", "\033[96m", "\033[92m", "\033[2m", "\033[0m"

# Images checked beforehand with the real `src.demo` (model trained without them):
# distinct leaf shapes, all classified correctly upright.
SHOWCASE = [("1001", "tre -- lá dài, hẹp"), ("1300", "phong Nhật -- lá xẻ thuỳ"),
            ("2400", "tuyết tùng -- lá kim"), ("3600", "quýt")]
# Correct at 0° and 90°, wrong at 15° -- the invariance finding on a single leaf.
ROTATION_LEAF = "1083"


def report_numbers() -> dict[str, str]:
    """Headline numbers from the report's generated macros, so the closing slide can
    never disagree with the PDF. `98{,}85\\%` -> `98,85%`."""
    path = ROOT / "report" / "generated" / "macros.tex"
    if not path.exists():
        return {}
    numbers = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("\\newcommand{\\"):
            name, _, value = line[len("\\newcommand{\\"):].partition("}{")
            numbers[name] = value[:-1].replace("{,}", ",").replace("\\%", "%")
    return numbers


def as_percent(fraction: str | None) -> str:
    """'0,4382' -> '43,82%', matching how the other closing numbers are written."""
    if not fraction:
        return "?"
    return f"{100 * float(fraction.replace(',', '.')):.2f}".replace(".", ",") + "%"


class Walkthrough:
    def __init__(self, auto: float | None, open_figures: bool) -> None:
        self.auto = auto
        self.open_figures = open_figures
        self.step = 0

    def pause(self) -> None:
        if self.auto is None:
            input(f"\n{DIM}[Enter để tiếp tục]{RESET}")
        else:
            time.sleep(self.auto)

    def heading(self, title: str, *caption: str) -> None:
        self.step += 1
        bar = "═" * 78
        print(f"\n{CYAN}{bar}\n{BOLD} BƯỚC {self.step}. {title}{RESET}\n{CYAN}{bar}{RESET}")
        for line in caption:
            print(f" {line}")
        print()

    def run(self, *args: str) -> None:
        shown = " ".join(a if " " not in a else f'"{a}"' for a in args)
        # Flush first: the child writes straight to the terminal, and anything still in
        # this process's buffer would otherwise surface after the child's output.
        print(f"{GREEN}$ python -m {shown}{RESET}\n", flush=True)
        env = dict(os.environ, PYTHONIOENCODING="utf-8")
        subprocess.run([sys.executable, "-m", *args], cwd=ROOT, env=env, check=True)

    def show(self, path: Path) -> None:
        """Open a figure in the default viewer so it appears in the recording."""
        if not path.exists():
            print(f"{DIM}(không tìm thấy {path}){RESET}")
            return
        if self.open_figures and sys.platform == "win32":
            os.startfile(path)  # noqa: S606 -- local file we just wrote
        else:
            print(f"{DIM}mở hình: {path}{RESET}")


def sample_subset(folder: Path, per_species: int = 2) -> int:
    """Copy the first `per_species` images of every species into a flat folder."""
    copied = 0
    for low, _high, _species in ID_RANGES:
        for image_id in range(low, low + per_species):
            src = DATA / f"{image_id}.jpg"
            if src.exists():
                shutil.copy(src, folder / src.name)
                copied += 1
    return copied


def main() -> None:
    ap = argparse.ArgumentParser(description="Silent-video walkthrough")
    ap.add_argument("--auto", type=float, default=None,
                    help="advance automatically after this many seconds")
    ap.add_argument("--no-open", action="store_true",
                    help="print figure paths instead of opening them")
    args = ap.parse_args()

    if sys.platform == "win32":
        os.system("")  # enable ANSI colours in the Windows console
    sys.stdout.reconfigure(encoding="utf-8")
    w = Walkthrough(args.auto, open_figures=not args.no_open)
    n = report_numbers()

    print(f"\n{BOLD}PHÂN LỚP LÁ CÂY TRÊN BỘ DỮ LIỆU FLAVIA{RESET}")
    print(" 32 loài, 1907 ảnh. Đặc trưng thủ công (hình dạng, màu sắc, kết cấu, gân lá)")
    print(" kết hợp bộ phân lớp truyền thống (SVM, KNN, Random Forest, Naive Bayes).")
    w.pause()

    w.heading("Kiểm tra bảng nhãn",
              "Nhãn của Flavia được mã hoá trong số hiệu tên tệp. Kiểm tra bảng dải số hiệu",
              "phủ đúng 32 loài và 1907 ảnh trước khi làm bất cứ việc gì khác.")
    w.run("src.labels")
    w.pause()

    w.heading("Trích xuất đặc trưng (chạy thật trên 64 ảnh, 2 ảnh/loài)",
              "Mỗi ảnh: phân đoạn lá bằng ngưỡng Otsu, rồi trích 64 đặc trưng thuộc 4 nhóm.",
              "Toàn bộ 1907 ảnh mất khoảng 9 phút, nên các bước sau dùng đặc trưng đã lưu sẵn.")
    # A short, fixed, gitignored folder: its path appears on screen in the command line.
    subset = ROOT / "cache" / "demo_subset"
    shutil.rmtree(subset, ignore_errors=True)
    subset.mkdir(parents=True)
    copied = sample_subset(subset)
    print(f"{DIM}(đã chép {copied} ảnh vào cache/demo_subset){RESET}\n")
    w.run("src.dataset", "--data", "cache/demo_subset",
          "--cache", "cache/demo_subset/features.npz", "--force")
    shutil.rmtree(subset, ignore_errors=True)
    w.pause()

    w.heading("Ablation và so sánh 4 bộ phân lớp (kiểm định chéo 5 phần, 1907 ảnh)",
              "Mỗi hàng là một tổ hợp nhóm đặc trưng, mỗi cột là một bộ phân lớp.",
              "Chú ý: thêm MÀU tăng mạnh nhất; SVM tốt nhất khi đủ đặc trưng, nhưng thua",
              "Random Forest khi chỉ có hình dạng.")
    w.run("src.experiments")
    w.show(ROOT / "report" / "generated" / "ablation.png")
    w.pause()

    w.heading("Ma trận nhầm lẫn của cấu hình tốt nhất (tập kiểm tra giữ lại, 477 ảnh)",
              "Ma trận đầy đủ 32x32 mở trong cửa sổ ảnh; terminal in các ô ngoài đường chéo.",
              "Chú ý: mỗi cặp nhầm lẫn chỉ xuất hiện một lần, không có lỗi mang tính hệ thống.")
    w.run("src.confusion")
    w.show(ROOT / "results" / "confusion_matrix.png")
    w.pause()

    w.heading("Chiếu không gian đặc trưng xuống 2 chiều: PCA và LDA",
              "PCA không dùng nhãn; LDA dùng nhãn để tìm hướng tách lớp. Hai chiều không đủ",
              "để thấy lợi ích của màu và kết cấu: nó chỉ xuất hiện khi dùng nhiều trục LDA hơn.")
    w.run("src.projection")
    w.show(ROOT / "results" / "projection.png")
    w.pause()

    w.heading("Dự đoán từng ảnh",
              "Mô hình được huấn luyện trên 1906 ảnh còn lại, KHÔNG gồm ảnh đang dự đoán.",
              "Bốn loài có hình dạng rất khác nhau.")
    for image_id, description in SHOWCASE:
        print(f"{DIM}# {description}{RESET}")
        w.run("src.demo", f"data/flavia/{image_id}.jpg")
        w.show(ROOT / "results" / f"demo_{image_id}.png")
        print()
    w.pause()

    w.heading("Kiểm chứng tính bất biến trên một chiếc lá",
              "Cùng một lá, xoay 0°, 15° rồi 90°. Chú ý: ở 15° phân đoạn vẫn chính xác nhưng",
              "mô hình đoán sai; ở 90° lại đúng. Đặc trưng kết cấu (GLCM, LBP) tính trên lưới",
              "điểm ảnh nên chỉ bất biến với phép xoay là bội số của 90°.")
    for angle in (0, 15, 90):
        rotate = [] if angle == 0 else ["--rotate", str(angle)]
        w.run("src.demo", f"data/flavia/{ROTATION_LEAF}.jpg", *rotate)
        suffix = "" if angle == 0 else f"_rot{angle}"
        w.show(ROOT / "results" / f"demo_{ROTATION_LEAF}{suffix}.png")
        print()
    w.pause()

    w.heading("Kiểm chứng tính bất biến trên toàn tập kiểm tra (477 ảnh)",
              "Huấn luyện trên ảnh sạch, kiểm tra trên ảnh đã xoay, thu nhỏ, đổi độ sáng,",
              "thêm nhiễu, che khuất. Mỗi cột là một nhóm đặc trưng được đánh giá riêng.")
    w.run("src.invariance")
    w.show(ROOT / "report" / "generated" / "invariance.png")
    w.pause()

    w.heading("Kết quả chính",
              f"1. SVM trên hình dạng + màu + kết cấu đạt {n.get('bestacc', '?')} (kiểm định chéo 5 phần).",
              f"2. Thêm màu tăng {n.get('colorgain', '?')}, thêm kết cấu chỉ tăng {n.get('texturegain', '?')}:",
              f"   hai nhóm chồng lấn thông tin (tương quan chính tắc {n.get('ccacolortexture', '?')}).",
              f"3. Nhóm gân lá không có đóng góp đo được (kiểm định t ghép cặp, p = {n.get('veinotsusvmp', '?')}).",
              "4. Kết cấu chỉ bất biến với phép xoay bội số 90°; tổ hợp tốt nhất lại kém bền",
              f"   vững hơn hình dạng đơn lẻ khi có nhiễu ({as_percent(n.get('invcombonoise'))} so với "
              f"{as_percent(n.get('invshapenoise'))}).")
    w.show(ROOT / "report" / "build" / "report.pdf")
    print(f"\n{DIM}Chi tiết trong report/build/report.pdf{RESET}\n")


if __name__ == "__main__":
    main()

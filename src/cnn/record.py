"""Record the assignment-2 demo video: start screen capture, run the walkthrough, stop.

One command produces the submission video. ffmpeg captures the whole desktop with
gdigrab while `src.cnn.walkthrough` runs in this same console, so the terminal output
and the figure windows it opens are both in the recording.

Two details that matter:

* ffmpeg is stopped by sending it "q" on stdin, never by killing it. A killed ffmpeg
  leaves the MP4 without its moov atom and the file will not play.
* The walkthrough runs with `--auto`, so nothing waits for a keypress; the pacing is
  fixed and the video length is predictable. The assignment allows 5 minutes, and the
  script warns if the result goes over.

Usage:
    python -m src.cnn.record                 # ~3 minutes
    python -m src.cnn.record --auto 5        # shorter pauses
    python -m src.cnn.record --no-open       # do not pop figure windows
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
BOLD, CYAN, DIM, RESET = "\033[1m", "\033[96m", "\033[2m", "\033[0m"


def primary_monitor() -> tuple[int, int]:
    """Size of the primary monitor in real pixels.

    gdigrab's "desktop" input grabs every monitor side by side -- on this two-screen
    machine that is 6400x2236, which would make the terminal a small strip inside a
    very wide frame. The primary monitor is always at (0,0) in virtual coordinates, so
    capturing that offset at this size isolates it.
    """
    import ctypes

    user32 = ctypes.windll.user32
    user32.SetProcessDPIAware()  # otherwise Windows reports scaled, not real, pixels
    return user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)


def ffmpeg_command(out: Path, fps: int, crf: int, full_desktop: bool = False,
                   scale_width: int = 1920) -> list[str]:
    capture = ["-i", "desktop"]
    if not full_desktop and sys.platform == "win32":
        width, height = primary_monitor()
        capture = ["-offset_x", "0", "-offset_y", "0",
                   "-video_size", f"{width}x{height}", "-i", "desktop"]

    # Even dimensions and yuv420p, or the file will not play in PowerPoint, Google
    # Drive's preview or QuickTime.
    scale = (f"scale={scale_width}:-2" if scale_width
             else "scale=trunc(iw/2)*2:trunc(ih/2)*2")
    return [
        "ffmpeg", "-y",
        "-f", "gdigrab", "-framerate", str(fps), "-draw_mouse", "1", *capture,
        "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf),
        "-pix_fmt", "yuv420p", "-vf", scale,
        str(out),
    ]


def duration(path: Path) -> float:
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", str(path)],
        capture_output=True, text=True)
    try:
        return float(probe.stdout.strip())
    except ValueError:
        return 0.0


def main() -> int:
    ap = argparse.ArgumentParser(description="Quay video demo bài tập 2")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "demo_bai2.mp4")
    ap.add_argument("--auto", type=float, default=7.0,
                    help="số giây dừng ở mỗi bước (mặc định 7)")
    ap.add_argument("--fps", type=int, default=10, help="khung hình/giây (terminal không cần cao)")
    ap.add_argument("--crf", type=int, default=28, help="chất lượng: nhỏ hơn = nét hơn, file to hơn")
    ap.add_argument("--countdown", type=int, default=5,
                    help="số giây chờ trước khi bắt đầu quay")
    ap.add_argument("--no-open", action="store_true", help="không mở cửa sổ ảnh")
    ap.add_argument("--full-desktop", action="store_true",
                    help="quay tất cả màn hình thay vì chỉ màn hình chính")
    ap.add_argument("--scale-width", type=int, default=1920,
                    help="thu video về chiều rộng này (0 = giữ nguyên độ phân giải)")
    args = ap.parse_args()

    if sys.platform == "win32":
        os.system("")
    sys.stdout.reconfigure(encoding="utf-8")

    if not shutil.which("ffmpeg"):
        print("Không tìm thấy ffmpeg. Cài bằng: winget install Gyan.FFmpeg")
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    area = "tất cả màn hình" if args.full_desktop else f"màn hình chính {primary_monitor()[0]}x{primary_monitor()[1]}"
    print(f"\n{BOLD}QUAY VIDEO DEMO BÀI TẬP 2{RESET}")
    print(f" Ghi ra: {args.out}")
    print(f" Vùng quay: {area}" + (f", thu về rộng {args.scale_width}px" if args.scale_width else ""))
    print(f" {DIM}Phóng to cửa sổ terminal này ngay bây giờ. Toàn màn hình sẽ được ghi lại.{RESET}")
    for remaining in range(args.countdown, 0, -1):
        print(f" Bắt đầu sau {remaining}...", end="\r", flush=True)
        time.sleep(1)
    print(" " * 40, end="\r")

    command = ffmpeg_command(args.out, args.fps, args.crf, args.full_desktop,
                             args.scale_width)
    recorder = subprocess.Popen(command, stdin=subprocess.PIPE,
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(1.5)  # let ffmpeg open the capture before anything happens on screen
    print(f"{CYAN}● ĐANG QUAY{RESET}\n", flush=True)

    walkthrough = ["src.cnn.walkthrough", "--auto", str(args.auto)]
    if args.no_open:
        walkthrough.append("--no-open")
    started = time.time()
    try:
        subprocess.run([sys.executable, "-m", *walkthrough], cwd=ROOT,
                       env=dict(os.environ, PYTHONIOENCODING="utf-8"), check=False)
    finally:
        time.sleep(2)  # a beat on the last screen so it is readable in the video
        # "q", not kill: ffmpeg must finalise the MP4 container itself.
        try:
            recorder.communicate(input=b"q", timeout=30)
        except subprocess.TimeoutExpired:
            recorder.terminate()
            recorder.wait(timeout=10)

    seconds = duration(args.out)
    size_mb = args.out.stat().st_size / 1048576
    print(f"\n{BOLD}Xong.{RESET} {args.out}")
    print(f" Thời lượng: {int(seconds // 60)} phút {int(seconds % 60)} giây"
          f"   Dung lượng: {size_mb:.1f} MB   (chạy thực tế {time.time() - started:.0f}s)")
    if seconds > 300:
        over = seconds - 300
        print(f" CẢNH BÁO: vượt giới hạn 5 phút {int(over)} giây. "
              f"Chạy lại với --auto {max(1, args.auto - over / 6):.0f} để rút ngắn.")
    elif seconds == 0:
        print(" CẢNH BÁO: không đọc được thời lượng, kiểm tra lại file.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

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


def raise_window(title_fragment: str, maximise: bool = False) -> bool:
    """Bring a window to the front so the screen capture actually shows it.

    Region capture records whatever is on top of that rectangle, so a window left
    behind another one is simply not in the video.
    """
    import ctypes
    from ctypes import wintypes

    user32 = ctypes.windll.user32
    found: list[int] = []

    @ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)
    def visit(handle, _param):
        if user32.IsWindowVisible(handle):
            length = user32.GetWindowTextLengthW(handle)
            if length:
                buffer = ctypes.create_unicode_buffer(length + 1)
                user32.GetWindowTextW(handle, buffer, length + 1)
                if title_fragment.lower() in buffer.value.lower():
                    found.append(handle)
                    return False
        return True

    user32.EnumWindows(visit, 0)
    if not found:
        return False
    user32.ShowWindow(found[0], 3 if maximise else 9)  # SW_MAXIMIZE / SW_RESTORE
    user32.SetForegroundWindow(found[0])
    return True


def launch_demo_console(auto: float, no_open: bool, title: str) -> subprocess.Popen:
    """Run the walkthrough in its own visible console window.

    Running it as an ordinary child process would send the output to whatever console
    started the recorder -- which is not on screen when the recorder is driven by a
    tool or a script. A new console is a real window, so it can be filmed.
    """
    inner = f'{Path(sys.executable).name} -m src.cnn.walkthrough --auto {auto}'
    if no_open:
        inner += " --no-open"
    command = f'title {title} && mode con: cols=130 lines=42 && {inner} && timeout /t 5'
    return subprocess.Popen(["cmd", "/c", command], cwd=ROOT,
                            env=dict(os.environ, PYTHONIOENCODING="utf-8"),
                            creationflags=subprocess.CREATE_NEW_CONSOLE)


def window_region(title_fragment: str) -> tuple[int, int, int, int]:
    """Screen rectangle (x, y, w, h) of the first visible window matching the title.

    Capturing the *region* rather than the window itself is deliberate: gdigrab's
    `title=` input returns pure black for GPU-accelerated windows such as VS Code or
    Chrome, because it reads the window's GDI surface and those apps never draw into
    it. Grabbing the same rectangle off the composited desktop works.
    """
    import ctypes
    from ctypes import wintypes

    user32 = ctypes.windll.user32
    user32.SetProcessDPIAware()
    found: list[int] = []

    @ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)
    def visit(handle, _param):
        if not user32.IsWindowVisible(handle):
            return True
        length = user32.GetWindowTextLengthW(handle)
        if length:
            buffer = ctypes.create_unicode_buffer(length + 1)
            user32.GetWindowTextW(handle, buffer, length + 1)
            if title_fragment.lower() in buffer.value.lower():
                found.append(handle)
                return False
        return True

    user32.EnumWindows(visit, 0)
    if not found:
        raise SystemExit(f"không tìm thấy cửa sổ nào có tên chứa {title_fragment!r}")

    rect = wintypes.RECT()
    user32.GetWindowRect(found[0], ctypes.byref(rect))
    # Even offsets and sizes: libx264 needs even dimensions, and an odd offset shifts
    # the chroma plane.
    x, y = max(0, rect.left) // 2 * 2, max(0, rect.top) // 2 * 2
    return x, y, (rect.right - x) // 2 * 2, (rect.bottom - y) // 2 * 2


def ffmpeg_command(out: Path, fps: int, crf: int, full_desktop: bool = False,
                   scale_width: int = 1920, window: str | None = None) -> list[str]:
    capture = ["-i", "desktop"]
    if window:
        x, y, width, height = window_region(window)
        capture = ["-offset_x", str(x), "-offset_y", str(y),
                   "-video_size", f"{width}x{height}", "-i", "desktop"]
    elif not full_desktop and sys.platform == "win32":
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
    ap.add_argument("--new-console", action="store_true",
                    help="chạy demo trong cửa sổ console riêng thay vì terminal hiện tại")
    ap.add_argument("--scale-width", type=int, default=2560,
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
    print(f" {DIM}Phóng to terminal này và tăng cỡ chữ (Ctrl + lăn chuột) trước khi bắt đầu.{RESET}")
    for remaining in range(args.countdown, 0, -1):
        print(f" Bắt đầu sau {remaining}...", end="\r", flush=True)
        time.sleep(1)
    print(" " * 40, end="\r")

    command = ffmpeg_command(args.out, args.fps, args.crf, args.full_desktop,
                             args.scale_width)
    # By default the walkthrough runs in THIS console, so whatever terminal you launched
    # the recorder from is what gets filmed. `--new-console` opens a separate window
    # instead; that only works from an interactive session, not from a tool-driven shell
    # (there, CREATE_NEW_CONSOLE produces no visible window at all).
    demo = None
    if args.new_console:
        console_title = "DEMO-BAI-TAP-2"
        demo = launch_demo_console(args.auto, args.no_open, console_title)
        time.sleep(2.5)
        if not raise_window(console_title):
            print(f"{DIM}(không thấy cửa sổ demo; hãy bỏ --new-console và chạy trực tiếp){RESET}")

    recorder = subprocess.Popen(command, stdin=subprocess.PIPE,
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(1.5)  # let ffmpeg open the capture before anything happens on screen
    print(f"{CYAN}● ĐANG QUAY{RESET}\n", flush=True)

    started = time.time()
    try:
        if demo is not None:
            demo.wait()
        else:
            walkthrough = ["src.cnn.walkthrough", "--auto", str(args.auto)]
            if args.no_open:
                walkthrough.append("--no-open")
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

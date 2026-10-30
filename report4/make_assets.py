"""Generate report4's numbers and figures (segmentation) from results/ and runs/.

All of the work is in src/yolo/report_assets.py, shared with the other YOLO report so
the two documents cannot drift apart. build.py runs this file with the report
directory as the working directory, hence the explicit path to the repository root.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.yolo.report_assets import build  # noqa: E402

build("segment", Path(__file__).resolve().parent / "generated")

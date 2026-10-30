r"""Pack one submission archive per assignment (3 = detect, 4 = segment).

Each archive is self-contained: the report PDF, the configuration code that trained the
models, the Colab notebook that ran it, the dataset links, and the raw per-epoch logs
Ultralytics wrote. The report .tex and its generated assets go in too, so the PDF can be
rebuilt from the archive alone without cloning the repo.

Deliberately left out: runs/*/weights (*.pt), the training images, and the other
assignments' code.
"""
from pathlib import Path
import zipfile

ROOT = Path(r"D:\kh4ng\CV")
STUDENT = "VoHoangKhang_25C01034"

TASKS = {
    "3": {"task": "detect", "report": "report3",
          "title": "Bai3_ObjectDetection", "extra_results": ["yolo_benchmark.csv"]},
    "4": {"task": "segment", "report": "report4",
          "title": "Bai4_ObjectSegmentation", "extra_results": []},
}

README = """NOP BAI - THI GIAC MAY TINH
{title}
Hoc vien: Vo Hoang Khang - MSHV: 25C01034
Ma nguon day du: https://github.com/vo-hoang-kh4ng/Leaf-Segmentation

NOI DUNG
--------
{report_pdf}
    Bao cao ({pages} trang). Chua link + mo ta dataset, phan ma cau hinh,
    va cac hinh minh chung huan luyen (mAP, loss).

LINK_DATASET.txt
    Link dataset cua ca hai bai, kem mo ta va so lieu tung phan.

src/yolo/
    Phan ma cau hinh de huan luyen YOLO (khong kem source code YOLO):
      config.py        toan bo giao thuc huan luyen, dung chung cho ca ba co model
      train.py         goi Ultralytics va cham diem tren phan test
      benchmark.py     do thoi gian mot epoch truoc khi chay that
      report_assets.py sinh so lieu va hinh cho bao cao tu runs/ va results/

notebooks/yolo_colab.ipynb
    So tay da dung de huan luyen that tren GPU Tesla T4 cua Google Colab.

results/yolo_{task}_summary.csv
    Ket qua tren phan test cua ba model, kem toan bo tham so giao thuc.

runs/{task}/yolo11*/results.csv va args.yaml
    Nhat ky tung epoch va tham so day du do Ultralytics ghi trong luc chay.
    Day la so lieu goc ma moi hinh trong bao cao duoc dung tu do.

{report}/
    Ma nguon bao cao: report.tex, make_assets.py, build.py va thu muc generated/.
    Moi con so trong bao cao deu sinh tu generated/macros.tex chu khong go tay.
    De dung lai PDF: can tai dataset ve truoc (make_assets.py dem so anh va so
    doi tuong truc tiep tu cac tep nhan), roi chay "python {report}/build.py".

requirements.txt
    Thu vien can thiet.

GHI CHU
-------
Bai 3 va bai 4 dung chung thu muc src/yolo/; chuyen giua hai bai chi la doi
gia tri --task. Trong so model (*.pt) va anh dataset khong kem theo cho nhe;
dataset tu tai ve khi chay theo link trong LINK_DATASET.txt.
"""


def pages(report: str) -> int:
    """Page count as XeLaTeX itself reported it on the last build.

    Counting "/Type /Page" in the PDF returns zero here: the file compresses its object
    streams, so those strings are not in the raw bytes.
    """
    log = (ROOT / report / "build" / "pass2.log").read_text(encoding="utf-8",
                                                            errors="replace")
    return int(log.split("Output written on")[-1].split("(")[-1].split(" pages")[0])


def build(number: str, spec: dict) -> Path:
    task, report = spec["task"], spec["report"]
    pdf = ROOT / report / "build" / "report.pdf"
    out = ROOT / f"NopBai{number}_{STUDENT}.zip"
    pdf_name = f"BaoCao_Bai{number}_{STUDENT}.pdf"

    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("README.txt", README.format(
            title=spec["title"], report_pdf=pdf_name, pages=pages(report),
            task=task, report=report))
        archive.write(pdf, pdf_name)
        archive.write(ROOT / "LINK_DATASET.txt", "LINK_DATASET.txt")
        archive.write(ROOT / "requirements.txt", "requirements.txt")
        archive.write(ROOT / "notebooks" / "yolo_colab.ipynb",
                      "notebooks/yolo_colab.ipynb")

        for source in sorted((ROOT / "src" / "yolo").glob("*.py")):
            archive.write(source, f"src/yolo/{source.name}")

        for name in [f"yolo_{task}_summary.csv"] + spec["extra_results"]:
            archive.write(ROOT / "results" / name, f"results/{name}")

        for run in sorted((ROOT / "runs" / task).glob("yolo11?")):
            for name in ("results.csv", "args.yaml"):
                path = run / name
                if path.exists():
                    archive.write(path, f"runs/{task}/{run.name}/{name}")

        for source in sorted((ROOT / report).glob("*.py")):
            archive.write(source, f"{report}/{source.name}")
        archive.write(ROOT / report / "report.tex", f"{report}/report.tex")
        for source in sorted((ROOT / report / "generated").iterdir()):
            archive.write(source, f"{report}/generated/{source.name}")

    with zipfile.ZipFile(out) as archive:
        count = len(archive.namelist())
    print(f"{out}  ({out.stat().st_size / 1048576:.1f} MB, {count} tệp)")
    return out


for number, spec in TASKS.items():
    build(number, spec)

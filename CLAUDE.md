# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Three assignments in one repository

| | Code | Report | Task | Status |
|---|---|---|---|---|
| 1 | `src/` | `report/` | Flavia leaves, hand-crafted features + classical classifiers | submitted |
| 2 | `src/cnn/` | `report2/` | same problem with CNNs (transfer learning) | submitted |
| 3 | `src/yolo/` *(not built yet)* | `report3/` *(not built yet)* | **object detection with Ultralytics YOLO** | due ~18/11/2026 |

Each assignment is its own classification problem; only 1 and 2 share data. The "no deep learning"
rule stated under Assignment 1 applies to that assignment alone.

**Binding rule between 1 and 2:** `src/cnn/data.py` reuses assignment 1's image order, stratified
split and `RANDOM_STATE`, so CNN numbers sit in the same table as the 98.85%. Change the split and
every comparison in report2 becomes meaningless. Assignment 3 shares nothing with them but the
report-building machinery.

## Assignment 3 — Ultralytics YOLO (current work)

Given 28/10/2026 with a three-week window. Requirements as stated by the lecturer:

- Object detection with [Ultralytics](https://github.com/ultralytics/ultralytics). The student
  **prefers an existing dataset** over labelling their own; self-built is allowed at ~100 annotated
  samples (Roboflow or LabelMe).
- **Three model sizes of the same YOLO version, minimum `n`, `s`, `m`** — the comparison is the point,
  so the training protocol (epochs, imgsz, batch, seed, split) must be identical across the three or
  the table means nothing. Same discipline as assignment 2's backbone comparison.
- Transfer learning from Ultralytics' pretrained checkpoints, not training from scratch.
- Submission: dataset link + description in the report, the **training configuration code only** (not
  a copy of the YOLO source), and figures evidencing training — mAP and loss curves.

**The hard constraint is compute.** This machine has no GPU (`torch.cuda.is_available()` is False,
torch 2.13+cpu, 8 threads), and YOLO training is far heavier than assignment 2's fine-tune. Budget
before choosing a dataset: `m` costs roughly an order of magnitude more than `n` per epoch, and the
run has to happen three times. Keep the dataset small (a few hundred images), consider `imgsz=416`
rather than 640, and measure one epoch before committing to a full schedule. `ultralytics` is not
installed yet.

Ultralytics writes its own metrics: `runs/detect/<name>/results.csv` plus `results.png`,
`confusion_matrix.png` and the PR/F1 curves. Those files are the evidence the assignment asks for —
read them rather than re-plotting from scratch, and keep `runs/` out of git.

**Dataset: African Wildlife** (`african-wildlife.yaml`, ships with Ultralytics, auto-downloads,
100 MB). 4 classes — buffalo, elephant, rhino, zebra — and a ready train/val/test split:

| split | images | labels | boxes |
|---|---|---|---|
| train | 1052 | 1052 | 1931 |
| val | 225 | 225 | 379 |
| test | 227 | 227 | 375 |

Boxes per class are reasonably balanced (399–575 in train), and the public page
(`docs.ultralytics.com/datasets/detect/african-wildlife/`) is the link the report must cite. Chosen
because it is ready-made, small enough for CPU training, and has a separate test split so the final
numbers are not read off the split that drove early stopping.

Ultralytics stores its paths in a global settings file, not in the repo. Both were pointed at this
project (`datasets_dir` -> `D:\kh4ng\CV\datasets`, `runs_dir` -> `D:\kh4ng\CV\runs`); `runs_dir`
had been left pointing at an unrelated project, so check it before a long run or the output lands
somewhere else entirely.

Timing anything with `model.train()` end-to-end is misleading: Ultralytics always runs a final
validation pass (whatever `val=` says) plus warm-up, which on a short run dominates. The first
measurement here read 6.5 min/epoch when the training pass was under 3. `src/yolo/benchmark.py`
brackets the epoch with `on_train_epoch_start`/`on_train_epoch_end` callbacks instead.

## Assignment 1 brief

Individual coursework (Computer Vision): **leaf classification on the Flavia dataset** using classical
hand-crafted features + a classical classifier — no deep learning.

Deliverables required by the assignment:
1. Source code
2. Demo video (≤ 5 min) showing the run pipeline and sample results
3. Report (≤ 10 pages, Vietnamese) explaining the method and the student's understanding of it

Academic basis: Wu et al. 2007 (12 geometric features + Probabilistic Neural Network, ~90%); later work
(Zhang 2012, Kadir 2011) adds color moments, GLCM texture and vein features and reaches 93–99% with SVM.
Reference repo for pipeline shape: `AayushG159/Plant-Leaf-Identification`.

**Anti-plagiarism constraint — this drives the design.** Re-running the reference notebook unchanged is not
acceptable; the work must show original analysis. Two mandatory contributions:
- **Feature ablation**: add GLCM / LBP texture features on top of shape + color, and report accuracy
  before/after so the contribution of each feature group is measurable.
- **Classifier comparison**: SVM vs KNN vs Random Forest vs Naive Bayes on the *same* feature matrix, to
  support the report's argument about why SVM wins on this feature space (max-margin + kernel trick on
  high-dimensional, low-sample-count data).

Every experiment must therefore emit numbers that can be pasted into the report — accuracy tables,
confusion matrices, per-feature-group ablation rows. Code that produces a single accuracy print is not
enough.

## Commands

```
python -m src.dataset [--data data/flavia] [--cache cache/features.npz] [--force]
python -m src.experiments [--cache cache/features.npz] [--out results] [--folds 5]
python -m src.demo <image> [--rotate DEG] [--groups shape,color,texture]  # leaves <image> out of training
python -m src.vein_variants   # vein extractor comparison + paired t-tests -> results/vein_variants.csv
python -m src.redundancy      # CCA / ridge / MI / permutation importance -> results/redundancy_*.csv
python -m src.invariance      # transform test-set images, measure per-group drop -> results/invariance*.csv
python -m src.projection      # PCA/LDA 2-D views + KNN on k LDA axes -> results/projection.png
python -m src.confusion       # held-out confusion matrix, off-diagonal cells only (reads results/)
python report/build.py     # regenerate numbers/figures from results/, then build report/build/report.pdf
python -m src.walkthrough [--auto SEC] [--no-open]  # scripted silent demo video; ~70 s of compute
```

Assignment 2 (CNN), CPU-only — torch 2.13+cpu, no GPU:

```
python -m src.cnn.embed [--backbone NAME] [--force]   # frozen-backbone features -> cache/cnn/*.npz
python -m src.cnn.linear_probe        # 3 backbones + paired t-tests -> results/cnn_linear_probe.csv
python -m src.cnn.finetune [--epochs 6]               # ~45 s/epoch -> results/cnn_finetune_*
python -m src.cnn.evaluate --source finetune|probe    # confusion matrix + precision/recall figures
python -m src.cnn.visualise           # filters, feature maps, Grad-CAM, t-SNE -> results/cnn_*.png
python report2/build.py               # -> report2/build/report.pdf
python -m src.cnn.walkthrough [--auto SEC] [--no-open]  # demo video script; ~2 min of compute
python -m src.cnn.record [--auto SEC] [--scale-width PX]  # ffmpeg screen capture + walkthrough -> mp4
```

`src.cnn.finetune` writes to `results/` by default; anything that re-runs it for show (the
walkthrough) must redirect `--out` and `--checkpoint` to a temp dir or it destroys the 6-epoch run
the report's confusion matrix is built from.

`src.cnn.linear_probe` rewrites `results/cnn_linear_probe.csv` on every run, and two of its columns
are wall-clock timings (`embed_ms_per_image`, `probe_seconds`). Accuracies are deterministic, those
are not — so a demo run shows up as a git diff, and rebuilding report2 afterwards shifts the ms/image
figures quoted in it (MobileNetV3 moved 10.6 -> 11.3 after one such run). Harmless, but do not treat a
diff in that file as a change in results.

`src.experiments` builds the cache itself when it is missing. **After editing any feature extractor you must
pass `--force`** to `src.dataset`, otherwise every experiment silently keeps scoring the old matrix.

There is no test suite yet. To smoke-test a change without the full dataset, point `--data` at a small folder
of images laid out one subdirectory per class.

## Structure

```
src/preprocess.py       # grayscale -> blur -> Otsu(INV) -> largest contour -> filled mask (LeafSample)
src/features/shape.py   # dimensionless Wu-2007 geometry + log Hu moments (17)
src/features/color.py   # color moments over masked pixels, RGB + HSV (18)
src/features/texture.py # GLCM over 32 levels, 2 distances x 4 angles + uniform LBP hist (22)
src/features/vein.py    # morphological opening residue ratios at radii 1..4 (7)
src/features/__init__.py# GROUPS registry -- the seam the ablation slices on
src/labels.py           # Flavia filename-ID -> species ranges (flat-archive layout)
src/dataset.py          # images -> X, y, feature_names; cached to .npz
src/experiments.py      # ablation grid + 4 classifiers -> results/ tables
src/demo.py             # one image -> prediction + segmentation figure (for the video)
report/report.tex       # the write-up (XeLaTeX + babel vietnamese, Times New Roman)
report/make_assets.py   # results/*.csv -> report/generated/{macros,tables}.tex + figures
data/flavia/            # dataset, gitignored, never commit
```

The report never hard-codes a number: every accuracy in the prose is a macro (`\bestacc`, `\colorgain`, ...)
defined in `report/generated/macros.tex`, which `make_assets.py` derives from `results/ablation.csv` and
`results/confusion_matrix.csv`. Re-running the experiments and `report/build.py` updates the document. If you need a
number in the text that has no macro, add it to `macros()` rather than typing the digits into `report.tex`.

`\todo{...}` marks the spots the student must fill in; `\showtodofalse` hides them all for submission.

**Page count:** the assignment says "không quá 10 trang"; the student got approval for **11**. Currently the
body + references end on page 9 and a Phụ lục (appendix, after the bibliography) holds the 32x32 confusion
matrix on page 10. Put bulky supporting material in the appendix rather than the body, and report the page
count `build.py` prints whenever it approaches 11.

Report conventions (Vietnamese academic register, as a master's student writes):
- Impersonal voice ("Báo cáo trình bày...", "Kết quả cho thấy...") rather than "chúng tôi"; Vietnamese
  terms with the English in parentheses on first use (độ chính xác, kiểm định chéo, tập kiểm tra giữ lại).
- Decimal comma everywhere. `make_assets.vn()` formats every generated number; signed deltas go in math
  mode (`$\veinpctknn$`) so the minus prints as a minus, not a hyphen.
- Every method named in the text carries a `\cite`. The 13 entries were each verified against the
  publisher/arXiv (IEEE style, numbered in order of first citation) -- keep that order when adding one.
  Only claim literature numbers you can cite: Kadir et al. report 93.75% on Flavia; there is no verified
  source for the "up to 99%" figure that once sat in the conclusion.

Edit `report.tex` with the Edit tool, never through non-raw Python strings: `\ref` becomes CR+`ef`,
`\textbf` becomes TAB+`extbf`. These build without any LaTeX error and just print as body text.
`build.py` refuses to build when it detects them.

Architectural rule that makes the ablation possible: **extractors are independent**, each exposing
`NAMES` and `extract(sample) -> np.ndarray` of matching length, and all contour/mask work happens once in
`preprocess.segment`. No extractor may read another's output. `NAMES` order is part of the cache format --
changing it invalidates cached matrices, and feature names are stored alongside `X` because the report needs
to name which features mattered.

Extraction is the expensive step and classification is nearly free, so all four groups are always extracted
into one cached matrix; `dataset.columns_for` then slices column subsets per experiment.

## Environment

System Python is 3.14 and the whole stack installs on it (numpy 2.4, opencv 5.0, scikit-image 0.26,
scikit-learn 1.8, torch 2.13+cpu): `python -m pip install -r requirements.txt`. No virtualenv is in
use. `ultralytics` is not installed yet — assignment 3 needs it.

**No GPU on this machine.** `torch.cuda.is_available()` is False and torch reports 8 threads. Every
deep-learning number in this repository was produced on CPU, and anything new has to be budgeted the
same way: measure one epoch before planning a schedule.

Shell here is PowerShell on Windows; a Bash tool is also available. Paths in code use `pathlib`, never
hard-coded backslashes.

## Findings so far (full Flavia run, 5-fold stratified)

Best: SVM (RBF) on shape+color+texture, 0.9885 CV / 0.9895 held-out. Shape-only baseline 0.86 (SVM) --
0.90 (RF), matching Wu 2007's ~90%. Three results the report is built on: color contributes far more than
texture (+11.3 vs +1.3 points) and the two overlap heavily; the vein group contributes nothing measurable;
and SVM loses to Random Forest on shape-only, only winning once the feature space is rich enough for the
margin to pay off.

On the vein group, be careful with the wording: it is NOT harmful. The -0.05 point difference is an order of
magnitude below the fold-to-fold std (0.43), and a paired t-test gives p=0.70. `src.vein_variants` re-ran it
with two better extractors (fixed relative threshold, Frangi vesselness); both beat Otsu for all four
classifiers and help the weak ones most (KNN +0.89/+0.73 vs +0.31), but nothing reaches p<0.05 at 5 folds.
Read that as: the vein signal is real but redundant with color+texture. Any claim about differences under
~1 point in this project needs a paired test before it goes in the report.

## Invariance findings (src.invariance)

Train on clean, test on transformed images. Control: the identity transform must reproduce the
held-out accuracy (0.9895) or the re-extraction path is broken and every drop is a bug.

- Texture is invariant only to multiples of 90 degrees (0.8323 at 90 deg, 0.3354 at 15 deg). GLCM and
  LBP drift equally at 15 and 45 deg, so the cause is pixel-lattice interpolation, not the angle set.
- `shape.smooth_factor` is the model's most important feature AND its least invariant: 0.39 std drift
  under 90 deg rotation, 8.13 std under 0.5x scaling, because it smooths with a fixed 5x5 window. If you
  ever make it scale-invariant, scale the window with the leaf perimeter -- and re-run everything.
- The best feature set is the most brittle: under noise, shape alone scores 0.8470 but
  shape+color+texture only 0.4382; under 10% occlusion, 0.7002 vs 0.0419.

A transform can itself be buggy and the identity control will not catch it. Rotation originally kept
the canvas size and clipped 83% of leaves (median 4.2% of leaf area), which looked exactly like
shape features failing to be rotation-invariant. Any new transform needs its own sanity check.

## Recording the demo video

`src.cnn.record` starts ffmpeg (gdigrab), runs the walkthrough with `--auto`, then stops ffmpeg by
sending "q" on stdin. Never kill ffmpeg instead: the MP4 loses its moov atom and will not play.

The walkthrough runs in the console that launched the recorder, so the recorder must be started from
a terminal the camera can see. A tool-driven shell cannot help here: `CREATE_NEW_CONSOLE` from such a
session produces no visible window at all (verified -- `EnumWindows` never sees it), so the student has
to run `python -m src.cnn.record` themselves. `--new-console` exists for interactive sessions only.

Window capture (`-i title=...`) is useless for VS Code, Chrome or any GPU-accelerated window: gdigrab
reads the window's GDI surface, which those apps never draw into, and every frame comes out pure black
(measured: 100% of pixels below 8/255). Capture the screen *region* instead -- but then whatever sits
on top of that rectangle is what gets filmed, so raise the window first.

gdigrab's "desktop" input grabs every monitor side by side -- 6400x2236 on this two-screen machine,
which buries the terminal in a wide strip. The script captures the primary monitor only (offset 0,0,
`GetSystemMetrics(0/1)` after `SetProcessDPIAware`) and scales to 1920 wide; about 21 MB for a
3-minute run. `--full-desktop` restores the old behaviour.

## CNN findings (assignment 2)

Frozen ImageNet backbones + a linear probe already beat the hand-crafted pipeline, with no training
inside the network at all: ResNet18 0.9974 CV, MobileNetV3 0.9963, VGG11 0.9932, against 0.9885.
Fine-tuning ResNet18's last block reaches 0.9979 held-out (1 error in 477). Paired t-tests: only
ResNet18 vs VGG11 is significant (p=0.035); the rest is noise, so do not rank by decimals.

MobileNetV3 has 52x fewer parameters than VGG11 and scores higher at 5x the speed — parameter count
does not predict transfer quality. The report's headline is that most of the performance comes from
the pretrained representation, not from training on Flavia; the t-SNE figure shows species already
clustered in the frozen embedding, whereas assignment 1's PCA/LDA projections could not separate
them in 2-D.

Accuracy is at the dataset's ceiling, so new comparisons here are mostly measuring noise.

## Report machinery worth reusing

Both reports are built the same way, and assignment 3 should copy the pattern rather than invent one:

- `reportN/make_assets.py` reads `results/*.csv` and emits `generated/macros.tex` (one `\newcommand`
  per number), `generated/tables.tex` and the figures. **No number is ever typed into the .tex.**
- `reportN/build.py` runs that, then XeLaTeX twice, and refuses to build when it detects escapes eaten
  by a non-raw Python string (`\ref` becoming CR+`ef`). It prints the page count.
- `report2/make_docx.js` builds a Word copy from the same macros via the `docx` npm package, so the
  .docx cannot drift from the .pdf. Verify a .docx by converting it with Word (COM, see the session
  notes) and looking at the rendered pages — layout bugs are invisible in the XML.
- `src/*/walkthrough.py` is a scripted, captioned run of the whole pipeline for the silent demo video;
  `src/cnn/record.py` films it with ffmpeg.

## Evaluation discipline

- Stratified train/test split (or stratified k-fold) — the 32 Flavia species are unevenly sized, so a plain
  random split skews small classes.
- Standardize features (`StandardScaler`) before SVM and KNN; scale-sensitive classifiers will otherwise be
  dominated by the raw-pixel-magnitude features.
- Fit the scaler inside the CV fold / on train only. Leaking it across the split inflates accuracy and
  invalidates the comparison the report is built on.
- Report accuracy *and* a confusion matrix; the interesting failure mode is confusion between visually
  similar species, and the report should discuss it.

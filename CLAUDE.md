# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

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
python report/build.py     # regenerate numbers/figures from results/, then build report/build/report.pdf
python -m src.walkthrough [--auto SEC] [--no-open]  # scripted silent demo video; ~70 s of compute
```

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
scikit-learn 1.8): `python -m pip install -r requirements.txt`. No virtualenv is in use.

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

## Evaluation discipline

- Stratified train/test split (or stratified k-fold) — the 32 Flavia species are unevenly sized, so a plain
  random split skews small classes.
- Standardize features (`StandardScaler`) before SVM and KNN; scale-sensitive classifiers will otherwise be
  dominated by the raw-pixel-magnitude features.
- Fit the scaler inside the CV fold / on train only. Leaking it across the split inflates accuracy and
  invalidates the comparison the report is built on.
- Report accuracy *and* a confusion matrix; the interesting failure mode is confusion between visually
  similar species, and the report should discuss it.

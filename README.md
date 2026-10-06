# Leaf classification on Flavia (shape + color + texture + vein -> SVM)

Classical computer-vision pipeline: segment the leaf, extract hand-crafted features in
four independent groups, then compare classifiers. No deep learning.

## Setup

```
python -m pip install -r requirements.txt
```

Download the Flavia dataset and unpack it into `data/flavia/`, either as the flat
folder of numbered JPEGs (labels come from the ID ranges in `src/labels.py`) or as one
subdirectory per species (subdirectory names win).

## Run

```
python -m src.dataset                      # extract features -> cache/features.npz
python -m src.experiments                  # ablation + classifier comparison -> results/
python -m src.demo data/flavia/1083.jpg              # single-image demo figure for the video
python -m src.demo data/flavia/1083.jpg --rotate 15  # same leaf turned 15° -> misclassified
python -m src.walkthrough                  # the whole demo video, step by step with on-screen captions
```

`src.experiments` calls `src.dataset` itself if the cache is missing, so the first
command is only needed when you want to inspect the matrix on its own. After editing
any feature extractor, re-run with `python -m src.dataset --force`.

## Outputs

- `results/ablation.csv` -- one row per (feature groups x classifier), mean/std CV accuracy
- `results/ablation_pivot.csv` -- the same as a table ready for the report
- `results/confusion_matrix.{csv,png}` -- held-out confusion matrix for the best combination

## Report

```
python report/build.py     # -> report/build/report.pdf
```

Needs XeLaTeX (MiKTeX/TeX Live) and the Times New Roman font. `build.py` regenerates
`report/generated/` from `results/` first, so the PDF always matches the last
experiment run -- no accuracy is typed by hand into `report.tex`.

## Vein extractor comparison

```
python -m src.vein_variants     # -> results/vein_variants.csv
```

Compares the Wu-style Otsu binarisation against a fixed relative threshold and Frangi
vesselness, all on top of `shape+color+texture`, with paired t-tests against the
no-vein baseline. Uses its own cache (`cache/vein_variants.npz`); Frangi extraction
takes ~15 min for the full dataset.

## Deeper analyses

```
python -m src.redundancy     # how much colour and texture overlap -> results/redundancy_*.csv
python -m src.invariance     # invariance stress test -> results/invariance.csv, invariance_drift.csv
python -m src.projection     # PCA/LDA projections of the feature space -> results/projection.png
python -m src.confusion      # print the held-out confusion matrix (errors only)
```

`src.invariance` trains on clean features and tests on transformed copies of the
held-out images (rotation, scaling, brightness, noise, occlusion), scoring each feature
group separately. Transformed features are cached under `cache/invariance/`; delete a
file there to re-extract that transform.

## Assignment 2 — CNN transfer learning

```
python -m src.cnn.embed          # frozen-backbone embeddings for 3 architectures
python -m src.cnn.linear_probe   # ResNet18 / MobileNetV3 / VGG11 + paired t-tests
python -m src.cnn.finetune       # fine-tune ResNet18's last block (~45 s/epoch on CPU)
python -m src.cnn.evaluate --source finetune   # confusion matrix, precision, recall
python -m src.cnn.visualise      # first-layer filters, feature maps, Grad-CAM, t-SNE
python report2/build.py          # -> report2/build/report.pdf
python -m src.cnn.walkthrough    # scripted demo for the video
```

CPU-only (torch 2.13+cpu). Frozen ImageNet features with a linear classifier already reach
99.74% (5-fold CV) against 98.85% for the hand-crafted pipeline; fine-tuning ResNet18 reaches
99.79% on the held-out split. The split and protocol are identical to assignment 1, so the
numbers are directly comparable.

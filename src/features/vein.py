"""Vein features, in three variants that share one skeleton.

Baseline (`vein`, Wu et al. 2007): opening the grayscale leaf with a disk of radius r
removes structures thinner than r, so the residue original-minus-opened is the venation.
Binarising that residue with Otsu is the step this module exists to interrogate: Otsu
always splits a histogram into two classes, so on a leaf with faint veins it invents a
threshold anyway and the resulting "vein area" is mostly noise. Section 6.2 of the
report measures the cost of that.

Two alternatives are provided so the choice can be tested rather than assumed:

  vein_pct     same residue, but thresholded at a fixed fraction of the leaf's own
               residue range (alpha * P99). Still contrast-adaptive, but it cannot
               manufacture veins where the residue is flat -- if nothing exceeds the
               level, the area is genuinely zero.
  vein_frangi  Frangi vesselness, a filter built for curvilinear structures: it scores
               each pixel by the eigenvalues of the Hessian, responding where one
               principal curvature is large and the other near zero -- i.e. ridges.

All three export 7 dimensions with the same meaning (4 area ratios at increasing
scale + 3 ratios relative to the finest scale), so they are directly comparable and
interchangeable in the ablation.
"""
from __future__ import annotations

import cv2
import numpy as np

from ..preprocess import LeafSample

_RADII = (1, 2, 3, 4)
_ALPHA = 0.30  # vein_pct: threshold at 30% of the leaf's 99th-percentile residue
_FRANGI_MAX_DIM = 512  # vein_frangi: Frangi is O(pixels) with a large constant
_FRANGI_LEVEL = 0.15  # fixed level on the max-normalised vesselness map

NAMES = [f"vein_area_ratio_r{r}" for r in _RADII] + [
    f"vein_ratio_r{r}_over_r1" for r in _RADII[1:]
]


def _ratios(areas: list[float], leaf_area: float) -> np.ndarray:
    """Turn absolute vein areas into the 7 scale-invariant ratios."""
    values = [a / leaf_area if leaf_area else 0.0 for a in areas]
    base = areas[0]
    values += [a / base if base else 0.0 for a in areas[1:]]
    return np.array(values, dtype=np.float64)


def _residues(sample: LeafSample) -> list[np.ndarray]:
    """Opening residue at each radius -- the part shared by `vein` and `vein_pct`."""
    gray = cv2.bitwise_and(sample.gray, sample.gray, mask=sample.mask)
    out = []
    for r in _RADII:
        se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
        out.append(cv2.subtract(gray, cv2.morphologyEx(gray, cv2.MORPH_OPEN, se)))
    return out


def extract(sample: LeafSample) -> np.ndarray:
    """Baseline: Otsu on the opening residue."""
    leaf = sample.mask > 0
    areas = []
    for residue in _residues(sample):
        inside = residue[leaf]
        if inside.size == 0 or inside.max() == 0:
            areas.append(0.0)
            continue
        level, _ = cv2.threshold(inside, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        areas.append(float(np.count_nonzero((residue >= level) & leaf)))
    return _ratios(areas, float(np.count_nonzero(leaf)))


def extract_pct(sample: LeafSample) -> np.ndarray:
    """Fixed relative threshold instead of Otsu.

    The level is alpha * P99 of the residue inside the leaf. P99 rather than the max
    because a single specular pixel would otherwise set the scale for the whole leaf.
    Unlike Otsu this can return zero area, which is the correct answer for a leaf
    whose veins the opening did not resolve.
    """
    leaf = sample.mask > 0
    areas = []
    for residue in _residues(sample):
        inside = residue[leaf]
        if inside.size == 0:
            areas.append(0.0)
            continue
        level = _ALPHA * float(np.percentile(inside, 99))
        areas.append(float(np.count_nonzero((residue > level) & leaf)) if level > 0 else 0.0)
    return _ratios(areas, float(np.count_nonzero(leaf)))


def extract_frangi(sample: LeafSample) -> np.ndarray:
    """Frangi vesselness at four scales, in place of the opening residue.

    Run on the leaf's bounding box downscaled to at most 512px: Frangi builds a Hessian
    per scale, which at 1600x1200 costs seconds per image for detail far finer than the
    veins. `black_ridges=False` because Flavia veins are *lighter* than the lamina.
    """
    from skimage.filters import frangi

    ys, xs = np.nonzero(sample.mask)
    if ys.size == 0:
        return np.zeros(len(NAMES))
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    gray = sample.gray[y0:y1, x0:x1]
    mask = sample.mask[y0:y1, x0:x1]

    scale = _FRANGI_MAX_DIM / max(gray.shape)
    if scale < 1.0:
        size = (int(gray.shape[1] * scale), int(gray.shape[0] * scale))
        gray = cv2.resize(gray, size, interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)

    leaf = mask > 0
    normalised = gray.astype(np.float64) / 255.0
    areas = []
    for r in _RADII:
        response = frangi(normalised, sigmas=[r], black_ridges=False)
        response[~leaf] = 0.0
        peak = response.max()
        if peak <= 0:
            areas.append(0.0)
            continue
        areas.append(float(np.count_nonzero(response > _FRANGI_LEVEL * peak)))
    return _ratios(areas, float(np.count_nonzero(leaf)))

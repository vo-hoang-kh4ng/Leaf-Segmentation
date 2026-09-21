"""GLCM (Haralick) + LBP texture, the Kadir 2011 addition to the Wu feature set.

This module exists to be switched off: the ablation experiment compares shape+color
against shape+color+texture, so its contribution has to be measurable in isolation.
"""
from __future__ import annotations

import numpy as np
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern

from ..preprocess import LeafSample

_LEVELS = 32
_DISTANCES = (1, 3)
_PROPS = ("contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM")
_LBP_P = 8
_LBP_R = 1
_LBP_BINS = _LBP_P + 2  # uniform LBP: P+1 uniform codes + 1 catch-all for the rest

NAMES = [f"glcm_{p}_d{d}" for d in _DISTANCES for p in _PROPS] + [
    f"lbp_{i}" for i in range(_LBP_BINS)
]


def extract(sample: LeafSample) -> np.ndarray:
    gray, mask = _crop_to_leaf(sample)

    # Quantise to 32 levels: a 256-level GLCM over a few 10k pixels is mostly zeros,
    # which makes correlation/energy numerically unstable.
    quantised = (gray.astype(np.uint16) * _LEVELS // 256).astype(np.uint8)
    # Background is forced to level 0 and level 0 is dropped from the matrix below,
    # so co-occurrences that straddle the leaf edge do not pollute the texture stats.
    quantised[mask == 0] = 0

    glcm = graycomatrix(
        quantised,
        distances=list(_DISTANCES),
        angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
        levels=_LEVELS,
        symmetric=True,
        normed=False,
    )
    glcm = glcm[1:, 1:, :, :].astype(np.float64)
    totals = glcm.sum(axis=(0, 1), keepdims=True)
    glcm = np.divide(glcm, totals, out=np.zeros_like(glcm), where=totals > 0)

    values: list[float] = []
    for di in range(len(_DISTANCES)):
        for prop in _PROPS:
            # Average over the four angles -> rotation-invariant, and a leaf's
            # orientation on the scanner is arbitrary.
            values.append(float(graycoprops(glcm[:, :, di : di + 1, :], prop).mean()))

    lbp = local_binary_pattern(gray, _LBP_P, _LBP_R, method="uniform")
    codes = lbp[mask > 0]
    hist, _ = np.histogram(codes, bins=_LBP_BINS, range=(0, _LBP_BINS))
    hist = hist.astype(np.float64)
    if hist.sum() > 0:
        hist /= hist.sum()

    return np.array(values + hist.tolist(), dtype=np.float64)


def _crop_to_leaf(sample: LeafSample) -> tuple[np.ndarray, np.ndarray]:
    ys, xs = np.nonzero(sample.mask)
    if ys.size == 0:
        return sample.gray, sample.mask
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    return sample.gray[y0:y1, x0:x1], sample.mask[y0:y1, x0:x1]

"""Geometric features (Wu et al. 2007) + log-scaled Hu moments.

Only dimensionless ratios are exported: raw area/perimeter would encode how close the
leaf sat to the scanner, not what species it is. Hu moments are already invariant to
scale/rotation/translation; log|.| compresses their huge dynamic range so that
StandardScaler has something well-behaved to work with.
"""
from __future__ import annotations

import cv2
import numpy as np

from ..preprocess import LeafSample

NAMES = [
    "smooth_factor",
    "aspect_ratio",
    "form_factor",
    "rectangularity",
    "narrow_factor",
    "perim_diameter_ratio",
    "perim_lw_ratio",
    "solidity",
    "eccentricity",
    "extent",
] + [f"hu{i}" for i in range(7)]


def extract(sample: LeafSample) -> np.ndarray:
    contour = sample.contour
    area = float(cv2.contourArea(contour))
    perimeter = float(cv2.arcLength(contour, True))

    # Physiological length/width = the sides of the minimum-area rectangle, i.e. the
    # leaf measured along its own axis rather than along the image axes.
    (_, _), (w, h), _ = cv2.minAreaRect(contour)
    length, width = max(w, h), min(w, h)

    # Wu's "diameter": the longest chord of the leaf, approximated by the minimum
    # enclosing circle so we avoid an O(n^2) sweep over contour points.
    _, radius = cv2.minEnclosingCircle(contour)
    diameter = 2.0 * radius

    hull_area = float(cv2.contourArea(cv2.convexHull(contour)))

    # Smooth factor: how much perimeter survives a 5x5 average -- a serrated margin
    # loses far more than an entire one. This is the feature that separates e.g.
    # Chinese Toon from Ford Woodlotus, which have near-identical global shape.
    smoothed = cv2.blur(sample.mask, (5, 5))
    _, smoothed = cv2.threshold(smoothed, 127, 255, cv2.THRESH_BINARY)
    sc, _ = cv2.findContours(smoothed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    smooth_perim = float(cv2.arcLength(max(sc, key=cv2.contourArea), True)) if sc else perimeter

    if len(contour) >= 5:
        (_, _), (ma, MA), _ = cv2.fitEllipse(contour)
        major, minor = max(ma, MA), min(ma, MA)
        eccentricity = float(np.sqrt(max(0.0, 1.0 - (minor / major) ** 2))) if major > 0 else 0.0
    else:
        eccentricity = 0.0

    x, y, bw, bh = cv2.boundingRect(contour)

    values = [
        _safe(smooth_perim, perimeter),
        _safe(length, width),
        4.0 * np.pi * _safe(area, perimeter**2),
        _safe(length * width, area),
        _safe(diameter, length),
        _safe(perimeter, diameter),
        _safe(perimeter, length + width),
        _safe(area, hull_area),
        eccentricity,
        _safe(area, float(bw * bh)),
    ]

    hu = cv2.HuMoments(cv2.moments(sample.mask, binaryImage=True)).flatten()
    hu = [float(-np.sign(v) * np.log10(abs(v) + 1e-30)) for v in hu]

    return np.array(values + hu, dtype=np.float64)


def _safe(num: float, den: float) -> float:
    return float(num / den) if den else 0.0

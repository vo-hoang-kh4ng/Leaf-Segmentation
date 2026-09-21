"""Color moments (mean, std, skewness) per channel, in RGB and HSV.

Statistics are taken over masked pixels only -- including the white background would
make every leaf look mostly-white and wash the feature out. HSV is carried alongside
RGB because hue separates the species that differ in tint (e.g. the blue-green
conifers) while staying comparatively stable under scanner illumination.
"""
from __future__ import annotations

import cv2
import numpy as np

from ..preprocess import LeafSample

_SPACES = (("r", "g", "b"), ("h", "s", "v"))
NAMES = [
    f"{ch}_{stat}"
    for channels in _SPACES
    for ch in channels
    for stat in ("mean", "std", "skew")
]


def extract(sample: LeafSample) -> np.ndarray:
    pixels = sample.mask > 0
    rgb = cv2.cvtColor(sample.bgr, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(sample.bgr, cv2.COLOR_BGR2HSV)

    values: list[float] = []
    for image in (rgb, hsv):
        for c in range(3):
            values.extend(_moments(image[:, :, c][pixels].astype(np.float64)))
    return np.array(values, dtype=np.float64)


def _moments(channel: np.ndarray) -> tuple[float, float, float]:
    if channel.size == 0:
        return 0.0, 0.0, 0.0
    mean = float(channel.mean())
    std = float(channel.std())
    # Third root of the third central moment, keeping the sign -- same units as the
    # channel itself, unlike scipy's normalised skew, and what the color-moments
    # literature (Stricker & Orengo) actually specifies.
    third = float(((channel - mean) ** 3).mean())
    skew = float(np.sign(third) * abs(third) ** (1.0 / 3.0))
    return mean, std, skew

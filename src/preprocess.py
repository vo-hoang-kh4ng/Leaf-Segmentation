"""Segmentation: raw leaf photo -> binary mask + contour.

Every feature extractor consumes a LeafSample, so contour/mask work happens exactly
once per image. Extractors never call back into this module.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class LeafSample:
    """One segmented leaf. `mask` is uint8 {0,255}, leaf = 255."""

    path: Path
    bgr: np.ndarray
    gray: np.ndarray
    mask: np.ndarray
    contour: np.ndarray


class SegmentationError(RuntimeError):
    """Raised when no plausible leaf blob is found; caller skips the image."""


def stages(bgr: np.ndarray, path: Path | None = None) -> dict[str, np.ndarray]:
    """Run the segmentation and return every intermediate image.

    `segment` keeps only the final result; the report figure needs the steps in
    between. Both go through here so the figure can never drift from what the feature
    extractors actually see.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    cleaned = cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, kernel, iterations=2)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise SegmentationError(f"no contour found in {path}")

    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < 0.01 * cleaned.size:
        # A blob under 1% of the frame is scanner noise, not a leaf.
        raise SegmentationError(f"largest blob too small in {path}")

    mask = np.zeros(gray.shape, dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)

    return {"gray": gray, "otsu": otsu, "cleaned": cleaned, "mask": mask, "contour": contour}


def segment(bgr: np.ndarray, path: Path | None = None) -> LeafSample:
    """Flavia images are a dark leaf on a near-white background.

    Otsu on the blurred grayscale separates them; THRESH_BINARY_INV makes the leaf
    the foreground. Closing fills the small holes that veins and specular highlights
    punch into the blob, then the largest contour is kept and filled solid -- filling
    matters because the shape features assume a solid silhouette.
    """
    s = stages(bgr, path)
    return LeafSample(
        path=path or Path("<memory>"),
        bgr=bgr,
        gray=s["gray"],
        mask=s["mask"],
        contour=s["contour"],
    )


def load(path: Path) -> LeafSample:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise SegmentationError(f"unreadable image {path}")
    return segment(bgr, path)

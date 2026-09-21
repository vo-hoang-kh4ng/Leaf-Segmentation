"""Feature-group registry.

Each group exposes `NAMES: list[str]` and `extract(sample) -> np.ndarray` of matching
length. Independence is the point: experiments select subsets of these groups by name,
so no extractor may read another's output, and the order of NAMES must stay stable
across runs (the cached feature matrix is keyed by it).

`ABLATION_GROUPS` is the four-group set the main ablation and the report are built on.
`GROUPS` additionally holds the two alternative vein extractors, which are compared
against the baseline in `src.vein_variants` rather than mixed into the ablation grid --
they measure the same thing three different ways, so combining them is meaningless.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from . import color, shape, texture, vein
from ..preprocess import LeafSample

ABLATION_GROUPS = ["shape", "color", "texture", "vein"]

GROUPS = {
    "shape": shape,
    "color": color,
    "texture": texture,
    "vein": vein,
    # Same 7 feature names, different binarisation of the vein evidence.
    "vein_pct": SimpleNamespace(NAMES=vein.NAMES, extract=vein.extract_pct),
    "vein_frangi": SimpleNamespace(NAMES=vein.NAMES, extract=vein.extract_frangi),
}

VEIN_VARIANTS = ["vein", "vein_pct", "vein_frangi"]


def names(groups: list[str]) -> list[str]:
    return [f"{g}.{n}" for g in groups for n in GROUPS[g].NAMES]


def extract(sample: LeafSample, groups: list[str]) -> np.ndarray:
    return np.concatenate([GROUPS[g].extract(sample) for g in groups])

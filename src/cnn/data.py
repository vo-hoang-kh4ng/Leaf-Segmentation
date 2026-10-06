"""Flavia as a torch Dataset, on exactly the split used by assignment 1.

The split must match or the comparison is meaningless, so the image order and labels are
read from the hand-crafted feature cache when it exists (that file stores the paths in
the order the classical pipeline used), and the same `train_test_split(test_size=0.25,
stratify=y, random_state=42)` is applied to indices.

Images are resized to 224x224 rather than short-side-resized and centre-cropped: a crop
would cut the tip off the longer leaves, and leaf tips are exactly what distinguishes
several species.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision import transforms

from ..dataset import find_images
from ..experiments import RANDOM_STATE

IMAGE_SIZE = 224
# ImageNet statistics: the backbones were pretrained with these, so the inputs have to
# arrive in the same distribution for the pretrained filters to mean anything.
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def eval_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def train_transform() -> transforms.Compose:
    """Augmentation kept mild and label-preserving.

    Flips and rotations are safe: a leaf has no canonical orientation on a scanner.
    Colour jitter is kept small because colour genuinely separates species here (the
    hand-crafted study measured +11.3 points from colour alone) -- strong jitter would
    destroy signal, not add invariance.
    """
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20, fill=(255, 255, 255)),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


class LeafDataset(Dataset):
    def __init__(self, paths: list[Path], labels: np.ndarray, transform) -> None:
        self.paths = [Path(p) for p in paths]
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int]:
        image = Image.open(self.paths[i]).convert("RGB")
        return self.transform(image), int(self.labels[i])


def load_index(root: Path = Path("data/flavia"),
               cache: Path = Path("cache/features.npz")) -> tuple[list[Path], np.ndarray, list[str]]:
    """Image paths, integer labels and class names, in the assignment-1 order."""
    if cache.exists():
        data = np.load(cache, allow_pickle=False)
        paths = [Path(str(p)) for p in data["paths"]]
        species = [str(s) for s in data["y"]]
    else:
        items = find_images(root)
        paths = [p for p, _ in items]
        species = [s for _, s in items]
    classes = sorted(set(species))
    index = {name: i for i, name in enumerate(classes)}
    return paths, np.array([index[s] for s in species]), classes


def split(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The same stratified 75/25 split assignment 1 used for its held-out numbers."""
    return train_test_split(np.arange(len(labels)), test_size=0.25,
                            stratify=labels, random_state=RANDOM_STATE)

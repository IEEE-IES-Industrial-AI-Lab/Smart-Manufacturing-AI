"""
NEU Surface Defect Dataset Loader
-----------------------------------
Reference: "A New Steel Surface Defect Database for Machine Vision Research"
           Song & Yan, IEEE Transactions on Instrumentation and Measurement, 2013.

Dataset: http://faculty.neu.edu.cn/yunhyan/NEU_surface_defect_database.html
6 types of surface defects on hot-rolled steel strips:
  Rolled-in Scale (RS), Patches (Pa), Crazing (Cr),
  Pitted Surface (PS), Inclusion (In), Scratches (Sc)
1800 images (300 per class), 200×200 pixels, grayscale.
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Callable, Dict, List, Literal, Optional, Tuple

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T


NEU_CLASSES = {
    "Cr": 0,  # Crazing
    "In": 1,  # Inclusion
    "Pa": 2,  # Patches
    "PS": 3,  # Pitted Surface
    "RS": 4,  # Rolled-in Scale
    "Sc": 5,  # Scratches
}

NEU_CLASS_NAMES = {v: k for k, v in NEU_CLASSES.items()}

NEU_CLASS_DESCRIPTIONS = {
    "Cr": "Crazing — network of fine cracks on the surface",
    "In": "Inclusion — embedded foreign materials",
    "Pa": "Patches — large discolored regions",
    "PS": "Pitted Surface — small holes or cavities",
    "RS": "Rolled-in Scale — scale pressed into the surface",
    "Sc": "Scratches — elongated marks from abrasion",
}


class NEUSurfaceDataset(Dataset):
    """
    PyTorch Dataset for the NEU Steel Surface Defect database.

    The dataset is expected to be organized as:
        <root>/
          Cr/  ← 300 images
          In/
          Pa/
          PS/
          RS/
          Sc/

    Parameters
    ----------
    root : str | Path
        Root directory of the NEU Surface Defect dataset.
    split : {"train", "val", "test", "all"}
        Dataset split. Splits are generated deterministically from the
        full dataset using ``train_ratio`` and ``val_ratio``.
    train_ratio : float
        Fraction of data used for training. Default 0.7.
    val_ratio : float
        Fraction of data used for validation. Default 0.15.
    transform : callable, optional
        Transform applied to PIL images.
    random_seed : int
        Seed for reproducible train/val/test splits.
    """

    def __init__(
        self,
        root: str | Path,
        split: Literal["train", "val", "test", "all"] = "train",
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        transform: Optional[Callable] = None,
        random_seed: int = 42,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.transform = transform or self._default_transform(split)
        self.random_seed = random_seed

        if not self.root.exists():
            raise FileNotFoundError(
                f"NEU Surface dataset not found at {self.root}. "
                "Run datasets/download_datasets.sh first."
            )

        all_samples = self._discover_samples()
        self.samples = self._split_samples(
            all_samples, split, train_ratio, val_ratio, random_seed
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _default_transform(self, split: str) -> Callable:
        normalize = T.Normalize(mean=[0.5], std=[0.5])  # grayscale
        if split == "train":
            return T.Compose([
                T.Resize((224, 224)),
                T.RandomHorizontalFlip(),
                T.RandomVerticalFlip(),
                T.RandomRotation(10),
                T.Grayscale(num_output_channels=3),  # replicate to 3ch for CNN
                T.ToTensor(),
                T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])
        else:
            return T.Compose([
                T.Resize((224, 224)),
                T.Grayscale(num_output_channels=3),
                T.ToTensor(),
                T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])

    def _discover_samples(self) -> List[Tuple[Path, int]]:
        samples = []
        for class_name, label in NEU_CLASSES.items():
            class_dir = self.root / class_name
            if not class_dir.exists():
                continue
            for img_path in sorted(class_dir.glob("*.jpg")) + sorted(class_dir.glob("*.bmp")):
                samples.append((img_path, label))
        if not samples:
            raise FileNotFoundError(
                f"No images found in {self.root}. "
                "Expected subdirectories: Cr, In, Pa, PS, RS, Sc."
            )
        return samples

    @staticmethod
    def _split_samples(
        all_samples: List[Tuple[Path, int]],
        split: str,
        train_ratio: float,
        val_ratio: float,
        seed: int,
    ) -> List[Tuple[Path, int]]:
        if split == "all":
            return all_samples

        # Stratified split per class
        from collections import defaultdict
        per_class: Dict[int, list] = defaultdict(list)
        for item in all_samples:
            per_class[item[1]].append(item)

        train, val, test = [], [], []
        rng = random.Random(seed)

        for label, items in per_class.items():
            items = items.copy()
            rng.shuffle(items)
            n = len(items)
            n_train = int(n * train_ratio)
            n_val = int(n * val_ratio)
            train.extend(items[:n_train])
            val.extend(items[n_train:n_train + n_val])
            test.extend(items[n_train + n_val:])

        return {"train": train, "val": val, "test": test}[split]

    # ------------------------------------------------------------------
    # Dataset API
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("L")
        image = self.transform(image)
        return image, label

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def class_distribution(self) -> Dict[str, int]:
        counts: Dict[str, int] = {k: 0 for k in NEU_CLASSES}
        for _, label in self.samples:
            counts[NEU_CLASS_NAMES[label]] += 1
        return counts

    def __repr__(self) -> str:
        dist = self.class_distribution()
        return (
            f"NEUSurfaceDataset(split={self.split}, "
            f"samples={len(self)}, distribution={dist})"
        )


def get_neu_transforms(
    image_size: int = 224,
    augment: bool = True,
) -> Tuple[Callable, Callable]:
    """
    Return (train_transform, val_transform) for NEU Surface.

    Parameters
    ----------
    image_size : int
        Target spatial resolution.
    augment : bool
        Whether to include training augmentations.
    """
    to_3ch = T.Grayscale(num_output_channels=3)
    normalize = T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    if augment:
        train_transform = T.Compose([
            T.Resize((image_size + 24, image_size + 24)),
            T.RandomCrop(image_size),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomRotation(15),
            to_3ch,
            T.ToTensor(),
            normalize,
        ])
    else:
        train_transform = T.Compose([
            T.Resize((image_size, image_size)),
            to_3ch,
            T.ToTensor(),
            normalize,
        ])

    val_transform = T.Compose([
        T.Resize((image_size, image_size)),
        to_3ch,
        T.ToTensor(),
        normalize,
    ])

    return train_transform, val_transform

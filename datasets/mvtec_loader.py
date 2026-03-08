"""
MVTec Anomaly Detection Dataset Loader
---------------------------------------
Paper: "MVTec AD — A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection"
       Bergmann et al., CVPR 2019. https://doi.org/10.1109/CVPR.2019.00982

Dataset: https://www.mvtec.com/company/research/datasets/mvtec-ad
15 categories (5 textures + 10 objects), ~5000 high-resolution images.

License: Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Literal, Optional, Tuple

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T


MVTEC_CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid",
    "hazelnut", "leather", "metal_nut", "pill", "screw",
    "tile", "toothbrush", "transistor", "wood", "zipper",
]

# Expected folder layout (after download):
#   <root>/<category>/train/good/*.png
#   <root>/<category>/test/good/*.png
#   <root>/<category>/test/<defect_type>/*.png
#   <root>/<category>/ground_truth/<defect_type>/*.png


class MVTecDataset(Dataset):
    """
    PyTorch Dataset for the MVTec Anomaly Detection benchmark.

    Supports:
    - Binary classification (good=0, anomaly=1)
    - Optional pixel-level ground-truth masks for segmentation evaluation

    Parameters
    ----------
    root : str | Path
        Root directory containing the MVTec AD download.
    category : str
        One of the 15 MVTec categories.
    split : {"train", "test"}
        Dataset split. Train contains only good samples.
    transform : callable, optional
        Image transform applied to each sample.
    mask_transform : callable, optional
        Transform applied to ground-truth masks (test split only).
    return_mask : bool
        If True, __getitem__ returns (image, label, mask). Default False.
    """

    def __init__(
        self,
        root: str | Path,
        category: str,
        split: Literal["train", "test"] = "train",
        transform: Optional[Callable] = None,
        mask_transform: Optional[Callable] = None,
        return_mask: bool = False,
    ) -> None:
        if category not in MVTEC_CATEGORIES:
            raise ValueError(
                f"Unknown category '{category}'. "
                f"Choose from: {MVTEC_CATEGORIES}"
            )

        self.root = Path(root) / category
        self.split = split
        self.transform = transform or self._default_transform()
        self.mask_transform = mask_transform or self._default_mask_transform()
        self.return_mask = return_mask

        self.samples: list[Tuple[Path, int, Optional[Path]]] = []
        self._load_samples()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _default_transform(self) -> Callable:
        return T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

    def _default_mask_transform(self) -> Callable:
        return T.Compose([
            T.Resize((224, 224), interpolation=T.InterpolationMode.NEAREST),
            T.ToTensor(),
        ])

    def _load_samples(self) -> None:
        split_dir = self.root / self.split

        if self.split == "train":
            # Only good samples during training
            good_dir = split_dir / "good"
            if not good_dir.exists():
                raise FileNotFoundError(
                    f"Training data not found at {good_dir}. "
                    "Run datasets/download_datasets.sh first."
                )
            for img_path in sorted(good_dir.glob("*.png")):
                self.samples.append((img_path, 0, None))

        elif self.split == "test":
            for defect_dir in sorted(split_dir.iterdir()):
                if not defect_dir.is_dir():
                    continue
                label = 0 if defect_dir.name == "good" else 1
                for img_path in sorted(defect_dir.glob("*.png")):
                    mask_path: Optional[Path] = None
                    if label == 1:
                        mask_path = (
                            self.root
                            / "ground_truth"
                            / defect_dir.name
                            / img_path.name.replace(".png", "_mask.png")
                        )
                        if not mask_path.exists():
                            mask_path = None
                    self.samples.append((img_path, label, mask_path))

    # ------------------------------------------------------------------
    # Dataset API
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, label, mask_path = self.samples[idx]

        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        if not self.return_mask:
            return image, label

        if mask_path is not None and mask_path.exists():
            mask = Image.open(mask_path).convert("L")
            mask = self.mask_transform(mask)
        else:
            # Good sample — zero mask
            mask = np.zeros((1, 224, 224), dtype=np.float32)
            import torch
            mask = torch.from_numpy(mask)

        return image, label, mask

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def class_distribution(self) -> dict:
        """Return counts of good vs. anomalous samples."""
        labels = [s[1] for s in self.samples]
        return {"good": labels.count(0), "anomaly": labels.count(1)}

    def __repr__(self) -> str:
        dist = self.class_distribution()
        return (
            f"MVTecDataset(category={self.root.name}, split={self.split}, "
            f"samples={len(self)}, good={dist['good']}, anomaly={dist['anomaly']})"
        )


def get_mvtec_transforms(
    image_size: int = 224,
    augment: bool = True,
) -> Tuple[Callable, Callable]:
    """
    Return (train_transform, val_transform) for MVTec AD.

    Parameters
    ----------
    image_size : int
        Target spatial resolution.
    augment : bool
        Whether to apply training augmentations.
    """
    normalize = T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    if augment:
        train_transform = T.Compose([
            T.Resize((image_size + 32, image_size + 32)),
            T.RandomCrop(image_size),
            T.RandomHorizontalFlip(),
            T.RandomRotation(15),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            T.ToTensor(),
            normalize,
        ])
    else:
        train_transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            normalize,
        ])

    val_transform = T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        normalize,
    ])

    return train_transform, val_transform

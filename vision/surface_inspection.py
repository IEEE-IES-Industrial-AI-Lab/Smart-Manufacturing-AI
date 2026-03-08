"""
Surface Inspection Pipeline
-----------------------------
Sliding-window inspection of high-resolution industrial surface images.

Pipeline:
  1. Divide the full image into overlapping patches
  2. Run each patch through a DefectDetector
  3. Aggregate per-patch predictions into a spatial heatmap
  4. Threshold the heatmap to produce a binary defect mask

This mimics real-world inline inspection systems that scan surfaces
faster than full-image classifiers can process them.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T

from vision.defect_detection import DefectDetector


class SurfaceInspector:
    """
    Sliding-window surface inspection using a DefectDetector backbone.

    Parameters
    ----------
    model : DefectDetector
        A trained DefectDetector instance.
    patch_size : int
        Size of each square patch (pixels).
    stride : int
        Stride of the sliding window. Smaller stride = more overlap = finer heatmap.
    defect_class : int
        The class index that corresponds to "defect". Default 1.
    threshold : float
        Probability threshold above which a patch is classified as defective.
    device : str
        Inference device.
    """

    def __init__(
        self,
        model: DefectDetector,
        patch_size: int = 224,
        stride: int = 112,
        defect_class: int = 1,
        threshold: float = 0.5,
        device: str = "auto",
    ) -> None:
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model = model.to(device).eval()
        self.patch_size = patch_size
        self.stride = stride
        self.defect_class = defect_class
        self.threshold = threshold

        self._patch_transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def inspect(
        self,
        image: Image.Image | np.ndarray,
        return_heatmap: bool = True,
    ) -> dict:
        """
        Run sliding-window inspection on a full surface image.

        Parameters
        ----------
        image : PIL.Image or np.ndarray
            High-resolution surface image.
        return_heatmap : bool
            If True, include the probability heatmap in the output.

        Returns
        -------
        dict with keys:
          - "is_defective" : bool — overall verdict
          - "defect_ratio" : float — fraction of patches classified as defective
          - "heatmap" : np.ndarray (H, W) — defect probability map, or None
          - "defect_mask" : np.ndarray (H, W) bool — thresholded mask
          - "patch_results" : list of dicts with per-patch info
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        image = image.convert("RGB")

        W, H = image.size
        patches, positions = self._extract_patches(image)

        if not patches:
            return {
                "is_defective": False,
                "defect_ratio": 0.0,
                "heatmap": None,
                "defect_mask": np.zeros((H, W), dtype=bool),
                "patch_results": [],
            }

        probabilities = self._score_patches(patches)

        heatmap = self._build_heatmap(probabilities, positions, H, W)
        defect_mask = heatmap >= self.threshold

        defect_probs = probabilities[:, self.defect_class]
        defect_ratio = float((defect_probs >= self.threshold).mean())

        patch_results = [
            {
                "position": pos,
                "defect_prob": float(prob[self.defect_class]),
                "is_defective": float(prob[self.defect_class]) >= self.threshold,
            }
            for pos, prob in zip(positions, probabilities)
        ]

        return {
            "is_defective": defect_ratio > 0.0,
            "defect_ratio": defect_ratio,
            "heatmap": heatmap if return_heatmap else None,
            "defect_mask": defect_mask,
            "patch_results": patch_results,
        }

    def visualize(
        self,
        image: Image.Image,
        result: dict,
        alpha: float = 0.5,
    ) -> np.ndarray:
        """
        Overlay the heatmap on the original image.

        Parameters
        ----------
        image : PIL.Image
        result : dict
            Output from :meth:`inspect`.
        alpha : float
            Opacity of the heatmap overlay.

        Returns
        -------
        np.ndarray (H, W, 3) uint8 overlay image.
        """
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm

        img_np = np.array(image.convert("RGB"))
        heatmap = result.get("heatmap")

        if heatmap is None:
            return img_np

        from PIL import Image as PILImage
        heatmap_resized = np.array(
            PILImage.fromarray((heatmap * 255).astype(np.uint8)).resize(
                (img_np.shape[1], img_np.shape[0]),
                PILImage.BILINEAR,
            )
        ) / 255.0

        colormap = cm.get_cmap("jet")
        heatmap_colored = (colormap(heatmap_resized)[:, :, :3] * 255).astype(np.uint8)

        overlay = (img_np * (1 - alpha) + heatmap_colored * alpha).astype(np.uint8)
        return overlay

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_patches(
        self,
        image: Image.Image,
    ) -> Tuple[List[Image.Image], List[Tuple[int, int, int, int]]]:
        W, H = image.size
        patches, positions = [], []

        for y in range(0, H - self.patch_size + 1, self.stride):
            for x in range(0, W - self.patch_size + 1, self.stride):
                box = (x, y, x + self.patch_size, y + self.patch_size)
                patch = image.crop(box)
                patches.append(patch)
                positions.append(box)

        # Catch right/bottom edge patches
        for y in range(0, H - self.patch_size + 1, self.stride):
            x = W - self.patch_size
            if x > 0:
                box = (x, y, W, y + self.patch_size)
                if box not in positions:
                    patches.append(image.crop(box))
                    positions.append(box)

        return patches, positions

    def _score_patches(self, patches: List[Image.Image]) -> np.ndarray:
        """Run batched inference on a list of patch PIL images."""
        tensors = torch.stack([self._patch_transform(p) for p in patches])
        tensors = tensors.to(self.device)

        all_probs = []
        batch_size = 32
        with torch.no_grad():
            for i in range(0, len(tensors), batch_size):
                batch = tensors[i:i + batch_size]
                logits = self.model(batch)
                probs = F.softmax(logits, dim=1).cpu().numpy()
                all_probs.append(probs)

        return np.vstack(all_probs)  # (N_patches, num_classes)

    def _build_heatmap(
        self,
        probabilities: np.ndarray,
        positions: List[Tuple[int, int, int, int]],
        H: int,
        W: int,
    ) -> np.ndarray:
        """
        Accumulate per-patch defect probabilities into a dense heatmap
        using averaging over overlapping regions.
        """
        heatmap_sum = np.zeros((H, W), dtype=np.float32)
        heatmap_count = np.zeros((H, W), dtype=np.float32)

        ps = self.patch_size
        for (x1, y1, x2, y2), prob in zip(positions, probabilities):
            defect_prob = prob[self.defect_class]
            heatmap_sum[y1:y2, x1:x2] += defect_prob
            heatmap_count[y1:y2, x1:x2] += 1.0

        # Avoid division by zero in unvisited regions
        mask = heatmap_count > 0
        heatmap = np.zeros_like(heatmap_sum)
        heatmap[mask] = heatmap_sum[mask] / heatmap_count[mask]
        return heatmap

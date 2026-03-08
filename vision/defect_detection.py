"""
CNN-based Defect Detector
--------------------------
Wraps ResNet-50 and EfficientNet-B4 backbones for industrial defect classification.

Supports:
  - Binary classification (good / defective)
  - Multi-class fault type classification
  - Transfer learning with selective layer freezing
  - Grad-CAM saliency maps for explainability

References:
  He et al., "Deep Residual Learning for Image Recognition", CVPR 2016.
  Tan & Le, "EfficientNet: Rethinking Model Scaling for CNNs", ICML 2019.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models
from tqdm import tqdm


class DefectDetector(nn.Module):
    """
    CNN backbone fine-tuned for industrial defect classification.

    Parameters
    ----------
    backbone : {"resnet50", "efficientnet_b4", "resnet18", "efficientnet_b0"}
        Pre-trained backbone architecture.
    num_classes : int
        Number of output classes (2 for binary defect detection).
    pretrained : bool
        Whether to initialize with ImageNet weights.
    dropout : float
        Dropout rate before the classification head.
    freeze_backbone : bool
        If True, freeze all backbone parameters (only train the head).
    """

    SUPPORTED_BACKBONES = {
        "resnet18": (models.resnet18, models.ResNet18_Weights.IMAGENET1K_V1, 512),
        "resnet50": (models.resnet50, models.ResNet50_Weights.IMAGENET1K_V2, 2048),
        "efficientnet_b0": (models.efficientnet_b0, models.EfficientNet_B0_Weights.IMAGENET1K_V1, 1280),
        "efficientnet_b4": (models.efficientnet_b4, models.EfficientNet_B4_Weights.IMAGENET1K_V1, 1792),
    }

    def __init__(
        self,
        backbone: str = "resnet50",
        num_classes: int = 2,
        pretrained: bool = True,
        dropout: float = 0.3,
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()

        if backbone not in self.SUPPORTED_BACKBONES:
            raise ValueError(
                f"Unsupported backbone '{backbone}'. "
                f"Choose from: {list(self.SUPPORTED_BACKBONES.keys())}"
            )

        self.backbone_name = backbone
        self.num_classes = num_classes

        model_fn, weights, feature_dim = self.SUPPORTED_BACKBONES[backbone]
        weights_arg = weights if pretrained else None
        base_model = model_fn(weights=weights_arg)

        # Strip the classification head, keep feature extractor
        if backbone.startswith("resnet"):
            self.features = nn.Sequential(*list(base_model.children())[:-2])
            self.pool = nn.AdaptiveAvgPool2d(1)
        elif backbone.startswith("efficientnet"):
            self.features = base_model.features
            self.pool = nn.AdaptiveAvgPool2d(1)

        if freeze_backbone:
            for param in self.features.parameters():
                param.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout / 2),
            nn.Linear(256, num_classes),
        )

        self._feature_dim = feature_dim
        self._grad_cam_hooks: list = []

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)          # (B, C, H, W)
        feat = self.pool(feat)           # (B, C, 1, 1)
        feat = feat.flatten(1)           # (B, C)
        return self.classifier(feat)     # (B, num_classes)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Return class indices. Convenience wrapper around forward()."""
        with torch.no_grad():
            logits = self(x)
            return logits.argmax(dim=1)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return class probabilities."""
        with torch.no_grad():
            return F.softmax(self(x), dim=1)

    # ------------------------------------------------------------------
    # Training utilities
    # ------------------------------------------------------------------

    def unfreeze_backbone(self) -> None:
        """Unfreeze all backbone parameters for full fine-tuning."""
        for param in self.features.parameters():
            param.requires_grad = True

    def freeze_backbone(self) -> None:
        """Freeze backbone parameters."""
        for param in self.features.parameters():
            param.requires_grad = False

    def count_parameters(self) -> Dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable, "frozen": total - trainable}

    def __repr__(self) -> str:
        counts = self.count_parameters()
        return (
            f"DefectDetector(backbone={self.backbone_name}, "
            f"num_classes={self.num_classes}, "
            f"trainable_params={counts['trainable']:,})"
        )


# =============================================================================
# Training loop
# =============================================================================

class DefectDetectorTrainer:
    """
    High-level training wrapper for DefectDetector.

    Parameters
    ----------
    model : DefectDetector
    device : str
        "cuda", "mps", or "cpu".
    learning_rate : float
    weight_decay : float
    freeze_backbone_epochs : int
        Train only the head for this many epochs, then unfreeze the backbone.
    """

    def __init__(
        self,
        model: DefectDetector,
        device: str = "auto",
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        freeze_backbone_epochs: int = 3,
    ) -> None:
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model = model.to(device)
        self.lr = learning_rate
        self.weight_decay = weight_decay
        self.freeze_backbone_epochs = freeze_backbone_epochs
        self.history: Dict[str, list] = {"train_loss": [], "val_loss": [], "val_acc": []}

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 50,
        early_stopping_patience: int = 10,
        checkpoint_path: Optional[str] = None,
    ) -> Dict[str, list]:
        """
        Train the model.

        Parameters
        ----------
        train_loader : DataLoader
        val_loader : DataLoader
        epochs : int
        early_stopping_patience : int
        checkpoint_path : str, optional
            Path to save the best model checkpoint.

        Returns
        -------
        dict
            Training history with keys train_loss, val_loss, val_acc.
        """
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        criterion = nn.CrossEntropyLoss()

        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(1, epochs + 1):
            # Warm-up: unfreeze backbone after freeze_backbone_epochs
            if epoch == self.freeze_backbone_epochs + 1:
                self.model.unfreeze_backbone()
                optimizer = torch.optim.AdamW(
                    self.model.parameters(),
                    lr=self.lr * 0.1,  # lower LR for backbone
                    weight_decay=self.weight_decay,
                )
                print(f"Epoch {epoch}: Backbone unfrozen. Continuing with LR={self.lr * 0.1:.2e}")

            train_loss = self._train_epoch(train_loader, optimizer, criterion)
            val_loss, val_acc = self._eval_epoch(val_loader, criterion)
            scheduler.step()

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)

            print(
                f"Epoch {epoch:3d}/{epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val Acc: {val_acc:.4f}"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                if checkpoint_path:
                    self.save(checkpoint_path)
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"Early stopping triggered at epoch {epoch}.")
                    break

        return self.history

    def _train_epoch(self, loader: DataLoader, optimizer, criterion) -> float:
        self.model.train()
        total_loss = 0.0
        for images, labels in tqdm(loader, desc="Training", leave=False):
            images = images.to(self.device)
            labels = labels.to(self.device)
            optimizer.zero_grad()
            logits = self.model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(images)
        return total_loss / len(loader.dataset)

    def _eval_epoch(self, loader: DataLoader, criterion) -> Tuple[float, float]:
        self.model.eval()
        total_loss, correct = 0.0, 0
        with torch.no_grad():
            for images, labels in tqdm(loader, desc="Validation", leave=False):
                images = images.to(self.device)
                labels = labels.to(self.device)
                logits = self.model(images)
                loss = criterion(logits, labels)
                total_loss += loss.item() * len(images)
                preds = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
        n = len(loader.dataset)
        return total_loss / n, correct / n

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "backbone": self.model.backbone_name,
            "num_classes": self.model.num_classes,
            "history": self.history,
        }, path)

    def load(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.history = checkpoint.get("history", {})


# =============================================================================
# Grad-CAM
# =============================================================================

class GradCAM:
    """
    Gradient-weighted Class Activation Mapping for DefectDetector.

    Produces spatial attention heatmaps showing which image regions
    contributed most to the predicted class.

    Reference: Selvaraju et al., "Grad-CAM: Visual Explanations from Deep
               Networks via Gradient-based Localization", ICCV 2017.

    Parameters
    ----------
    model : DefectDetector
    target_layer : nn.Module
        The convolutional layer to extract activations from.
        Default: last conv layer of the backbone.
    """

    def __init__(self, model: DefectDetector, target_layer: Optional[nn.Module] = None) -> None:
        self.model = model
        self.model.eval()

        if target_layer is None:
            # Select last conv block of the backbone
            if model.backbone_name.startswith("resnet"):
                target_layer = list(model.features.children())[-1]
            else:
                target_layer = list(model.features.children())[-1]

        self._activations: Optional[torch.Tensor] = None
        self._gradients: Optional[torch.Tensor] = None

        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self._activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self._gradients = grad_output[0].detach()

    def __call__(self, image: torch.Tensor, class_idx: Optional[int] = None) -> np.ndarray:
        """
        Generate a Grad-CAM heatmap.

        Parameters
        ----------
        image : torch.Tensor
            Input image tensor of shape (1, C, H, W).
        class_idx : int, optional
            Target class. If None, uses the predicted class.

        Returns
        -------
        np.ndarray
            Heatmap of shape (H, W), values in [0, 1].
        """
        image.requires_grad_(True)
        logits = self.model(image)

        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()

        self.model.zero_grad()
        logits[0, class_idx].backward()

        # Global average pooling of gradients
        weights = self._gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        cam = (weights * self._activations).sum(dim=1, keepdim=True)  # (1, 1, H, W)
        cam = F.relu(cam)
        cam = cam.squeeze().cpu().numpy()

        # Normalize to [0, 1]
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()

        return cam

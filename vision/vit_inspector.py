"""
Vision Transformer Inspector
------------------------------
ViT-B/16-based fine-grained defect classifier with attention rollout
for spatial explainability.

Attention rollout recursively multiplies attention matrices across
all Transformer layers to produce a single attention map showing which
image patches the model attended to.

References:
  Dosovitskiy et al., "An Image is Worth 16x16 Words: Transformers for Image
  Recognition at Scale", ICLR 2021.
  Abnar & Zuidema, "Quantifying Attention Flow in Transformers", ACL 2020.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False


class ViTInspector(nn.Module):
    """
    Vision Transformer inspector for defect classification.

    Uses timm's ViT-B/16 (or ViT-S/16 for lighter deployment) as the
    backbone, with a linear classification head.

    Supports:
    - Standard forward pass for classification
    - Attention rollout for spatial explainability

    Parameters
    ----------
    model_name : str
        timm model name. Default "vit_base_patch16_224".
    num_classes : int
        Number of output classes.
    pretrained : bool
        Use ImageNet-21k pre-trained weights.
    dropout : float
        Dropout before the classification head.
    """

    def __init__(
        self,
        model_name: str = "vit_base_patch16_224",
        num_classes: int = 2,
        pretrained: bool = True,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        if not TIMM_AVAILABLE:
            raise ImportError(
                "timm is required for ViTInspector. Install with: pip install timm"
            )

        self.model_name = model_name
        self.num_classes = num_classes

        # Build backbone without classification head
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,  # remove timm's classifier
        )
        feature_dim = self.backbone.num_features

        self.head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(feature_dim, num_classes),
        )

        # Storage for attention rollout
        self._attention_maps: List[torch.Tensor] = []
        self._register_attention_hooks()

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._attention_maps.clear()
        features = self.backbone(x)   # (B, feature_dim) — CLS token
        return self.head(features)    # (B, num_classes)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self(x).argmax(dim=1)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return F.softmax(self(x), dim=1)

    # ------------------------------------------------------------------
    # Attention rollout
    # ------------------------------------------------------------------

    def _register_attention_hooks(self) -> None:
        """Hook into each transformer block to capture attention weights."""
        for block in self.backbone.blocks:
            block.attn.register_forward_hook(self._attention_hook)

    def _attention_hook(self, module, input, output):
        # timm attention modules store the attention weights in the module
        # after forward; we need to capture them explicitly
        # For timm >= 0.9, we use the fused_attn path or attn_drop
        with torch.no_grad():
            B, N, C = input[0].shape
            qkv = module.qkv(input[0])
            scale = module.scale
            num_heads = module.num_heads
            head_dim = C // num_heads

            qkv = qkv.reshape(B, N, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)
            attn = (q @ k.transpose(-2, -1)) * scale
            attn = attn.softmax(dim=-1)
            self._attention_maps.append(attn.cpu())

    def attention_rollout(
        self,
        image: torch.Tensor,
        discard_ratio: float = 0.9,
        head_fusion: str = "mean",
    ) -> np.ndarray:
        """
        Compute attention rollout for the input image.

        Parameters
        ----------
        image : torch.Tensor
            Input of shape (1, C, H, W).
        discard_ratio : float
            Fraction of lowest attention values to discard before rollout.
        head_fusion : {"mean", "max", "min"}
            How to aggregate across attention heads.

        Returns
        -------
        np.ndarray (patch_grid_size, patch_grid_size)
            Attention rollout map. Resize to image resolution for overlay.
        """
        self._attention_maps.clear()
        with torch.no_grad():
            _ = self(image)

        if not self._attention_maps:
            raise RuntimeError("No attention maps captured. Ensure the model processed an input.")

        # Fuse heads for each layer
        attn_layers = []
        for attn in self._attention_maps:
            if head_fusion == "mean":
                fused = attn.mean(dim=1)   # (B, N, N)
            elif head_fusion == "max":
                fused = attn.max(dim=1).values
            elif head_fusion == "min":
                fused = attn.min(dim=1).values
            else:
                raise ValueError(f"Unknown head_fusion: {head_fusion}")

            # Discard low-attention tokens
            flat = fused.view(fused.size(0), -1)
            threshold = torch.quantile(flat, discard_ratio, dim=1, keepdim=True)
            threshold = threshold.unsqueeze(-1)
            fused = torch.where(fused >= threshold, fused, torch.zeros_like(fused))

            # Add residual connection (attention flow)
            identity = torch.eye(fused.size(-1)).unsqueeze(0)
            fused = fused + identity
            fused = fused / fused.sum(dim=-1, keepdim=True)
            attn_layers.append(fused)

        # Rollout: multiply attention matrices
        rollout = attn_layers[0]
        for attn in attn_layers[1:]:
            rollout = attn @ rollout

        # Extract CLS → patch attention (first row, skip CLS token itself)
        mask = rollout[0, 0, 1:]  # (num_patches,)

        # Reshape to 2D grid
        num_patches = mask.shape[0]
        grid_size = int(num_patches ** 0.5)
        mask = mask.reshape(grid_size, grid_size).numpy()

        # Normalize to [0, 1]
        mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)
        return mask

    # ------------------------------------------------------------------
    # Parameter info
    # ------------------------------------------------------------------

    def count_parameters(self) -> Dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}

    def __repr__(self) -> str:
        counts = self.count_parameters()
        return (
            f"ViTInspector(model={self.model_name}, "
            f"num_classes={self.num_classes}, "
            f"params={counts['total']:,})"
        )


# =============================================================================
# Training loop
# =============================================================================

class ViTInspectorTrainer:
    """
    Training wrapper for ViTInspector with layer-wise learning rate decay.

    Lower layers of the ViT backbone are given smaller learning rates
    (10× decay per group) to preserve pre-trained representations.

    Parameters
    ----------
    model : ViTInspector
    device : str
    learning_rate : float
        Base learning rate for the classification head.
    lr_decay : float
        Per-group decay applied deeper into the backbone.
    weight_decay : float
    """

    def __init__(
        self,
        model: ViTInspector,
        device: str = "auto",
        learning_rate: float = 1e-4,
        lr_decay: float = 0.75,
        weight_decay: float = 1e-5,
    ) -> None:
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model = model.to(device)
        self.lr = learning_rate
        self.lr_decay = lr_decay
        self.weight_decay = weight_decay
        self.history: Dict[str, list] = {"train_loss": [], "val_loss": [], "val_acc": []}

    def _build_optimizer(self) -> torch.optim.Optimizer:
        """Layer-wise LR decay optimizer."""
        param_groups = [{"params": self.model.head.parameters(), "lr": self.lr}]

        # Group backbone blocks by depth
        blocks = list(self.model.backbone.blocks)
        num_blocks = len(blocks)
        for i, block in enumerate(reversed(blocks)):
            lr = self.lr * (self.lr_decay ** (i + 1))
            param_groups.append({"params": block.parameters(), "lr": lr})

        # Patch embedding and position embeddings
        param_groups.append({
            "params": self.model.backbone.patch_embed.parameters(),
            "lr": self.lr * (self.lr_decay ** (num_blocks + 1)),
        })

        return torch.optim.AdamW(
            param_groups,
            weight_decay=self.weight_decay,
        )

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 30,
        early_stopping_patience: int = 8,
        checkpoint_path: Optional[str] = None,
    ) -> Dict[str, list]:
        optimizer = self._build_optimizer()
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        criterion = nn.CrossEntropyLoss()

        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(1, epochs + 1):
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
                    self._save(checkpoint_path)
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"Early stopping at epoch {epoch}.")
                    break

        return self.history

    def _train_epoch(self, loader, optimizer, criterion) -> float:
        self.model.train()
        total_loss = 0.0
        for images, labels in tqdm(loader, desc="Training", leave=False):
            images, labels = images.to(self.device), labels.to(self.device)
            optimizer.zero_grad()
            loss = criterion(self.model(images), labels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item() * len(images)
        return total_loss / len(loader.dataset)

    def _eval_epoch(self, loader, criterion) -> Tuple[float, float]:
        self.model.eval()
        total_loss, correct = 0.0, 0
        with torch.no_grad():
            for images, labels in tqdm(loader, desc="Validation", leave=False):
                images, labels = images.to(self.device), labels.to(self.device)
                logits = self.model(images)
                total_loss += criterion(logits, labels).item() * len(images)
                correct += (logits.argmax(1) == labels).sum().item()
        n = len(loader.dataset)
        return total_loss / n, correct / n

    def _save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "model_name": self.model.model_name,
            "num_classes": self.model.num_classes,
            "history": self.history,
        }, path)

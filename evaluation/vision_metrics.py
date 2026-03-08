"""
Vision Evaluation Metrics
--------------------------
Standard metrics for industrial defect detection and surface inspection.

Implemented metrics:
  - AUROC (Area Under ROC Curve)
  - Average Precision (area under precision-recall curve)
  - F1 score at optimal threshold
  - Pixel-level IoU (Intersection over Union) for segmentation masks
  - Per-class confusion matrix

All functions accept NumPy arrays and return scalars or dicts.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    auc,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


# =============================================================================
# Core metrics
# =============================================================================

def compute_auroc(
    y_true: np.ndarray,
    y_score: np.ndarray,
    multi_class: str = "ovr",
) -> float:
    """
    Compute Area Under the ROC Curve.

    Parameters
    ----------
    y_true : (N,) int array — ground-truth class indices
    y_score : (N,) or (N, C) float array — predicted probabilities
    multi_class : str
        "ovr" (one-vs-rest) or "ovo" (one-vs-one) for multi-class.

    Returns
    -------
    float : AUROC in [0, 1]
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    n_classes = len(np.unique(y_true))
    if n_classes == 2:
        # Binary: y_score is probability of positive class
        score = y_score[:, 1] if y_score.ndim == 2 else y_score
        return float(roc_auc_score(y_true, score))
    else:
        return float(roc_auc_score(y_true, y_score, multi_class=multi_class))


def compute_average_precision(
    y_true: np.ndarray,
    y_score: np.ndarray,
) -> float:
    """
    Compute Average Precision (area under precision-recall curve).

    Parameters
    ----------
    y_true : (N,) binary int array
    y_score : (N,) float array — predicted probability of positive class

    Returns
    -------
    float : AP in [0, 1]
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    if y_score.ndim == 2:
        y_score = y_score[:, 1]
    return float(average_precision_score(y_true, y_score))


def compute_f1_at_optimal_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
) -> Tuple[float, float]:
    """
    Find the threshold that maximizes F1 score.

    Parameters
    ----------
    y_true : (N,) binary int array
    y_score : (N,) float array — positive class probability

    Returns
    -------
    f1 : float
    best_threshold : float
    """
    y_score = np.asarray(y_score)
    if y_score.ndim == 2:
        y_score = y_score[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_score)
    f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-8)
    best_idx = f1_scores.argmax()
    return float(f1_scores[best_idx]), float(thresholds[min(best_idx, len(thresholds) - 1)])


def compute_pixel_iou(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Compute pixel-level Intersection over Union for binary segmentation masks.

    Parameters
    ----------
    pred_mask : (H, W) float array — predicted defect probability map
    gt_mask : (H, W) float or bool array — ground-truth binary mask
    threshold : float
        Binarization threshold for pred_mask.

    Returns
    -------
    dict with keys: iou, dice, pixel_accuracy
    """
    pred_bin = (pred_mask >= threshold).astype(bool)
    gt_bin = (gt_mask > 0).astype(bool)

    intersection = (pred_bin & gt_bin).sum()
    union = (pred_bin | gt_bin).sum()
    total = gt_bin.size

    iou = float(intersection) / (float(union) + 1e-8)
    dice = 2.0 * float(intersection) / (float(pred_bin.sum() + gt_bin.sum()) + 1e-8)
    pixel_acc = float((pred_bin == gt_bin).sum()) / float(total)

    return {"iou": iou, "dice": dice, "pixel_accuracy": pixel_acc}


def compute_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
) -> Dict:
    """
    Compute confusion matrix and derived per-class metrics.

    Parameters
    ----------
    y_true : (N,) int array
    y_pred : (N,) int array
    class_names : list[str], optional

    Returns
    -------
    dict with keys: matrix, precision, recall, f1, class_names
    """
    cm = confusion_matrix(y_true, y_pred)
    n = cm.shape[0]

    precision = np.zeros(n)
    recall = np.zeros(n)
    f1 = np.zeros(n)

    for i in range(n):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        precision[i] = tp / (tp + fp + 1e-8)
        recall[i] = tp / (tp + fn + 1e-8)
        f1[i] = 2 * precision[i] * recall[i] / (precision[i] + recall[i] + 1e-8)

    names = class_names or [str(i) for i in range(n)]
    return {
        "matrix": cm,
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "f1": f1.tolist(),
        "class_names": names,
        "macro_f1": float(f1.mean()),
    }


# =============================================================================
# Full evaluation suite
# =============================================================================

def evaluate_detector(
    y_true: np.ndarray,
    y_score: np.ndarray,
    y_pred: Optional[np.ndarray] = None,
    class_names: Optional[List[str]] = None,
) -> Dict:
    """
    Run the full evaluation suite for a defect detector.

    Parameters
    ----------
    y_true : (N,) int array — ground-truth labels
    y_score : (N, C) float array — predicted probabilities
    y_pred : (N,) int array, optional — predicted class indices.
             If None, argmax(y_score) is used.
    class_names : list[str], optional

    Returns
    -------
    dict with all metrics
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    if y_pred is None:
        y_pred = y_score.argmax(axis=1)

    n_classes = y_score.shape[1] if y_score.ndim == 2 else 2
    is_binary = n_classes == 2

    results: Dict = {}

    results["auroc"] = compute_auroc(y_true, y_score)
    if is_binary:
        results["average_precision"] = compute_average_precision(y_true, y_score[:, 1])
        results["f1"], results["optimal_threshold"] = compute_f1_at_optimal_threshold(
            y_true, y_score[:, 1]
        )

    cm_results = compute_confusion_matrix(y_true, y_pred, class_names)
    results.update({
        "confusion_matrix": cm_results["matrix"].tolist(),
        "per_class_precision": cm_results["precision"],
        "per_class_recall": cm_results["recall"],
        "per_class_f1": cm_results["f1"],
        "macro_f1": cm_results["macro_f1"],
        "class_names": cm_results["class_names"],
    })

    return results


# =============================================================================
# Visualization helpers
# =============================================================================

def plot_roc_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
    title: str = "ROC Curve",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot the ROC curve for a binary classifier.

    Parameters
    ----------
    y_true : (N,) binary int array
    y_score : (N,) or (N, 2) float array
    title : str
    save_path : str, optional

    Returns
    -------
    matplotlib Figure
    """
    y_score = np.asarray(y_score)
    if y_score.ndim == 2:
        y_score = y_score[:, 1]

    fpr, tpr, _ = roc_curve(y_true, y_score)
    auroc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, lw=2, label=f"AUROC = {auroc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random (0.5000)")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_precision_recall_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
    title: str = "Precision-Recall Curve",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot the Precision-Recall curve.

    Parameters
    ----------
    y_true : (N,) binary int
    y_score : (N,) or (N, 2) float
    title : str
    save_path : str, optional

    Returns
    -------
    matplotlib Figure
    """
    y_score = np.asarray(y_score)
    if y_score.ndim == 2:
        y_score = y_score[:, 1]

    precision, recall, _ = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, lw=2, label=f"AP = {ap:.4f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_confusion_matrix_heatmap(
    cm: np.ndarray,
    class_names: List[str],
    title: str = "Confusion Matrix",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot a confusion matrix as a seaborn heatmap.

    Parameters
    ----------
    cm : (C, C) int array
    class_names : list[str]
    title : str
    save_path : str, optional

    Returns
    -------
    matplotlib Figure
    """
    try:
        import seaborn as sns
    except ImportError:
        sns = None

    fig, ax = plt.subplots(figsize=(max(5, len(class_names)), max(4, len(class_names))))

    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)

    if sns:
        sns.heatmap(
            cm_norm,
            annot=cm,
            fmt="d",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
            ax=ax,
        )
    else:
        im = ax.imshow(cm_norm, interpolation="nearest", cmap=plt.cm.Blues)
        plt.colorbar(im, ax=ax)
        ax.set_xticks(range(len(class_names)))
        ax.set_xticklabels(class_names, rotation=45, ha="right")
        ax.set_yticks(range(len(class_names)))
        ax.set_yticklabels(class_names)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color="white" if cm_norm[i, j] > 0.5 else "black")

    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")
    ax.set_title(title)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig

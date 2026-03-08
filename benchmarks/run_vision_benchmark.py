"""
Vision Benchmark: Defect Detection
------------------------------------
Trains and evaluates DefectDetector (ResNet-50, EfficientNet-B4) and
ViTInspector (ViT-B/16) on the MVTec AD and NEU Surface Defect datasets.

Results are saved to benchmarks/results/vision_benchmark.csv.

Usage:
  python benchmarks/run_vision_benchmark.py \
    --dataset mvtec \
    --category bottle \
    --backbones resnet50 efficientnet_b4 \
    --epochs 30 \
    --output benchmarks/results/vision_benchmark.csv
"""

import argparse
import csv
import os
import sys
import time
from pathlib import Path

# Ensure project root on path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from torch.utils.data import DataLoader, random_split

from datasets.mvtec_loader import MVTecDataset, get_mvtec_transforms
from datasets.neu_surface_loader import NEUSurfaceDataset, get_neu_transforms
from evaluation.vision_metrics import (
    evaluate_detector,
    plot_roc_curve,
    plot_confusion_matrix_heatmap,
)
from vision.defect_detection import DefectDetector, DefectDetectorTrainer

try:
    from vision.vit_inspector import ViTInspector, ViTInspectorTrainer
    VIT_AVAILABLE = True
except ImportError:
    VIT_AVAILABLE = False


# =============================================================================
# Helpers
# =============================================================================

def build_mvtec_loaders(
    root: str,
    category: str,
    batch_size: int,
    image_size: int = 224,
) -> tuple:
    train_tf, val_tf = get_mvtec_transforms(image_size, augment=True)
    train_ds = MVTecDataset(root, category, split="train", transform=train_tf)
    test_ds = MVTecDataset(root, category, split="test", transform=val_tf)

    # Split train → train / val
    n_val = max(1, int(0.2 * len(train_ds)))
    n_train = len(train_ds) - n_val
    train_subset, val_subset = random_split(train_ds, [n_train, n_val])

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4)
    return train_loader, val_loader, test_loader


def build_neu_loaders(root: str, batch_size: int, image_size: int = 224) -> tuple:
    train_tf, val_tf = get_neu_transforms(image_size, augment=True)
    train_ds = NEUSurfaceDataset(root, split="train", transform=train_tf)
    val_ds = NEUSurfaceDataset(root, split="val", transform=val_tf)
    test_ds = NEUSurfaceDataset(root, split="test", transform=val_tf)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4)
    return train_loader, val_loader, test_loader


def run_cnn_experiment(
    backbone: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    num_classes: int,
    epochs: int,
    device: str,
    checkpoint_dir: Path,
    results_dir: Path,
    label: str,
) -> dict:
    print(f"\n{'='*60}")
    print(f"Training {backbone} on {label}")
    print(f"{'='*60}")

    model = DefectDetector(
        backbone=backbone,
        num_classes=num_classes,
        pretrained=True,
        dropout=0.3,
        freeze_backbone=True,
    )
    print(model)

    trainer = DefectDetectorTrainer(
        model=model,
        device=device,
        learning_rate=1e-4,
        weight_decay=1e-5,
        freeze_backbone_epochs=3,
    )

    checkpoint_path = str(checkpoint_dir / f"{backbone}_{label.replace('/', '_')}.pth")
    t0 = time.time()
    trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        early_stopping_patience=8,
        checkpoint_path=checkpoint_path,
    )
    train_time = time.time() - t0

    # Inference on test set
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            probs = model.predict_proba(images).cpu().numpy()
            all_probs.extend(probs.tolist())
            all_labels.extend(labels.numpy().tolist())

    import numpy as np
    results = evaluate_detector(
        y_true=np.array(all_labels),
        y_score=np.array(all_probs),
        class_names=["good", "defect"] if num_classes == 2 else None,
    )
    results["backbone"] = backbone
    results["dataset"] = label
    results["epochs_trained"] = epochs
    results["train_time_s"] = round(train_time, 1)
    results["num_params"] = model.count_parameters()["trainable"]

    print(f"AUROC: {results['auroc']:.4f} | Macro F1: {results['macro_f1']:.4f}")

    # Save ROC curve
    if num_classes == 2:
        plot_roc_curve(
            y_true=np.array(all_labels),
            y_score=np.array(all_probs),
            title=f"ROC — {backbone} on {label}",
            save_path=str(results_dir / f"roc_{backbone}_{label.replace('/', '_')}.png"),
        )

    return results


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Vision Defect Detection Benchmark")
    parser.add_argument(
        "--dataset", choices=["mvtec", "neu"], default="mvtec",
        help="Dataset to benchmark on."
    )
    parser.add_argument(
        "--mvtec_root", default="datasets/raw/mvtec_ad",
        help="Root directory of MVTec AD dataset."
    )
    parser.add_argument(
        "--neu_root", default="datasets/raw/neu_surface",
        help="Root directory of NEU Surface dataset."
    )
    parser.add_argument(
        "--category", default="bottle",
        help="MVTec category (ignored for NEU)."
    )
    parser.add_argument(
        "--backbones", nargs="+",
        default=["resnet50", "efficientnet_b4"],
        choices=["resnet18", "resnet50", "efficientnet_b0", "efficientnet_b4"],
        help="Backbone architectures to benchmark."
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument(
        "--output", default="benchmarks/results/vision_benchmark.csv",
        help="Output CSV path."
    )
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    results_dir = Path("benchmarks/results")
    results_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = Path("checkpoints/vision_benchmark")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    for backbone in args.backbones:
        if args.dataset == "mvtec":
            label = f"mvtec_{args.category}"
            try:
                train_loader, val_loader, test_loader = build_mvtec_loaders(
                    root=args.mvtec_root,
                    category=args.category,
                    batch_size=args.batch_size,
                )
                num_classes = 2
            except FileNotFoundError as e:
                print(f"WARNING: {e}. Skipping {label}.")
                continue

        elif args.dataset == "neu":
            label = "neu_surface"
            try:
                train_loader, val_loader, test_loader = build_neu_loaders(
                    root=args.neu_root,
                    batch_size=args.batch_size,
                )
                num_classes = 6
            except FileNotFoundError as e:
                print(f"WARNING: {e}. Skipping {label}.")
                continue

        result = run_cnn_experiment(
            backbone=backbone,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            num_classes=num_classes,
            epochs=args.epochs,
            device=device,
            checkpoint_dir=checkpoint_dir,
            results_dir=results_dir,
            label=label,
        )
        all_results.append(result)

    # Save to CSV
    if all_results:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        flat_results = []
        for r in all_results:
            flat = {
                k: v for k, v in r.items()
                if not isinstance(v, (list, dict))
            }
            flat_results.append(flat)

        fieldnames = list(flat_results[0].keys())
        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(flat_results)

        print(f"\nBenchmark results saved to {output_path}")
        print("\n=== Summary ===")
        for r in flat_results:
            print(
                f"  {r['backbone']:20s} | {r['dataset']:20s} | "
                f"AUROC={r.get('auroc', 'N/A'):.4f} | "
                f"Macro F1={r.get('macro_f1', 'N/A'):.4f} | "
                f"Time={r.get('train_time_s', 0):.1f}s"
            )


if __name__ == "__main__":
    main()

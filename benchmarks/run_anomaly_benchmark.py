"""
Anomaly Detection Benchmark: Robot Joint Sensor Stream
---------------------------------------------------------
Trains and evaluates RobotAnomalyDetector (LSTM Autoencoder) on
robot sensor data with varying model sizes and threshold strategies.

Metrics reported:
  - AUROC (treating anomaly detection as binary classification)
  - F1 at optimal threshold
  - Average Precision
  - False Positive Rate at 95% True Positive Rate (FPR@95TPR)

Results saved to benchmarks/results/anomaly_benchmark.csv.

Usage:
  python benchmarks/run_anomaly_benchmark.py \
    --data_root datasets/raw/robot_sensors \
    --seq_len 50 \
    --epochs 80 \
    --output benchmarks/results/anomaly_benchmark.csv
"""

import argparse
import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from datasets.robot_dataset import RobotSensorDataset, generate_synthetic_robot_data
from evaluation.vision_metrics import (
    compute_auroc,
    compute_average_precision,
    compute_f1_at_optimal_threshold,
)
from robotics.robot_anomaly_detection import RobotAnomalyDetector


# =============================================================================
# FPR@95TPR
# =============================================================================

def fpr_at_95_tpr(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    False Positive Rate when True Positive Rate = 0.95.
    Standard metric for anomaly detection evaluation.
    """
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    # Find threshold closest to TPR=0.95
    idx = np.argmin(np.abs(tpr - 0.95))
    return float(fpr[idx])


# =============================================================================
# Experiment runner
# =============================================================================

def run_experiment(
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    config: dict,
    device: str,
    checkpoint_dir: Path,
) -> dict:
    label = config["label"]
    print(f"\n{'='*60}")
    print(f"Config: {label}")
    print(f"{'='*60}")

    detector = RobotAnomalyDetector(
        input_dim=config["input_dim"],
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"],
        latent_dim=config["latent_dim"],
        seq_len=config["seq_len"],
        dropout=config.get("dropout", 0.2),
        bidirectional=config.get("bidirectional", False),
        device=device,
    )
    print(detector.model)

    checkpoint_path = str(checkpoint_dir / f"{label.replace(' ', '_')}.pth")

    t0 = time.time()
    detector.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=config.get("epochs", 80),
        learning_rate=config.get("lr", 1e-3),
        early_stopping_patience=15,
        checkpoint_path=checkpoint_path,
    )
    train_time = time.time() - t0

    # Calibrate threshold on nominal validation data
    # Filter val_loader to nominal-only samples
    nominal_indices = [
        i for i, (_, label_val) in enumerate(val_loader.dataset)
        if label_val == 0
    ]
    if nominal_indices:
        nominal_subset = Subset(val_loader.dataset, nominal_indices)
        nominal_loader = DataLoader(
            nominal_subset, batch_size=val_loader.batch_size, shuffle=False
        )
        detector.calibrate_threshold(
            nominal_loader=nominal_loader,
            method=config.get("threshold_method", "percentile"),
            percentile=config.get("threshold_percentile", 95.0),
        )
    else:
        detector.calibrate_threshold(
            nominal_loader=val_loader,
            method="percentile",
            percentile=95.0,
        )

    # Evaluate on test set
    all_errors, all_labels = [], []
    detector.model.eval()
    with torch.no_grad():
        for windows, labels in test_loader:
            windows = windows.to(device)
            errors = detector.model.reconstruction_error(windows).cpu().numpy()
            all_errors.extend(errors.tolist())
            all_labels.extend(labels.numpy().tolist())

    y_true = np.array(all_labels)
    y_score = np.array(all_errors)

    auroc = compute_auroc(y_true, y_score)
    ap = compute_average_precision(y_true, y_score)
    f1, opt_thresh = compute_f1_at_optimal_threshold(y_true, y_score)
    fpr95 = fpr_at_95_tpr(y_true, y_score)

    result = {
        "config": label,
        "hidden_dim": config["hidden_dim"],
        "num_layers": config["num_layers"],
        "latent_dim": config["latent_dim"],
        "bidirectional": config.get("bidirectional", False),
        "auroc": round(auroc, 4),
        "average_precision": round(ap, 4),
        "f1_optimal": round(f1, 4),
        "optimal_threshold": round(opt_thresh, 6),
        "fpr_at_95tpr": round(fpr95, 4),
        "train_time_s": round(train_time, 1),
        "num_params": detector.model.count_parameters(),
        "epochs": config.get("epochs", 80),
    }

    print(
        f"AUROC={auroc:.4f} | AP={ap:.4f} | F1={f1:.4f} | FPR@95={fpr95:.4f}"
    )
    return result


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Robot Anomaly Detection Benchmark")
    parser.add_argument(
        "--data_root", default="datasets/raw/robot_sensors",
        help="Directory with robot sensor CSV files."
    )
    parser.add_argument("--seq_len", type=int, default=50)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument(
        "--output", default="benchmarks/results/anomaly_benchmark.csv"
    )
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    results_dir = Path("benchmarks/results")
    results_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = Path("checkpoints/anomaly_benchmark")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Generate synthetic data if real data not present
    data_root = Path(args.data_root)
    if not data_root.exists() or not list(data_root.glob("*.csv")):
        print(f"No data found at {data_root}. Generating synthetic robot sensor data...")
        generate_synthetic_robot_data(
            output_path=data_root / "robot_data.csv",
            n_nominal_steps=15000,
            n_fault_steps=3000,
            random_seed=42,
        )

    train_ds = RobotSensorDataset(
        root=data_root, sequence_length=args.seq_len, stride=args.stride,
        split="train", normalize=True,
    )
    val_ds = RobotSensorDataset(
        root=data_root, sequence_length=args.seq_len, stride=args.stride,
        split="val", normalize=True,
    )
    test_ds = RobotSensorDataset(
        root=data_root, sequence_length=args.seq_len, stride=args.stride,
        split="test", normalize=True,
    )
    print(train_ds)
    print(val_ds)
    print(test_ds)

    input_dim = train_ds.num_channels

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # Define benchmark configurations
    configs = [
        {
            "label": "LSTM-AE-small",
            "input_dim": input_dim, "hidden_dim": 32,
            "num_layers": 1, "latent_dim": 8,
            "seq_len": args.seq_len, "epochs": args.epochs,
        },
        {
            "label": "LSTM-AE-base",
            "input_dim": input_dim, "hidden_dim": 64,
            "num_layers": 2, "latent_dim": 16,
            "seq_len": args.seq_len, "epochs": args.epochs,
        },
        {
            "label": "LSTM-AE-large",
            "input_dim": input_dim, "hidden_dim": 128,
            "num_layers": 2, "latent_dim": 32,
            "seq_len": args.seq_len, "epochs": args.epochs,
        },
        {
            "label": "BiLSTM-AE-base",
            "input_dim": input_dim, "hidden_dim": 64,
            "num_layers": 2, "latent_dim": 16,
            "seq_len": args.seq_len, "epochs": args.epochs,
            "bidirectional": True,
        },
    ]

    all_results = []
    for config in configs:
        result = run_experiment(
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            config=config,
            device=device,
            checkpoint_dir=checkpoint_dir,
        )
        all_results.append(result)

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(all_results[0].keys())
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)

    print(f"\nBenchmark results saved to {output_path}")
    print("\n=== Summary ===")
    print(f"{'Config':<22} {'AUROC':>6} {'AP':>6} {'F1':>6} {'FPR@95':>7} {'Params':>10}")
    print("-" * 65)
    for r in all_results:
        print(
            f"{r['config']:<22} "
            f"{r['auroc']:>6.4f} "
            f"{r['average_precision']:>6.4f} "
            f"{r['f1_optimal']:>6.4f} "
            f"{r['fpr_at_95tpr']:>7.4f} "
            f"{r['num_params']:>10,}"
        )


if __name__ == "__main__":
    main()

"""
Robot Sensor Time-Series Dataset
----------------------------------
Loader for robot joint sensor streams used in anomaly detection.

Expected data format: CSV files with columns for joint torques, velocities,
and a binary 'fault' label. Compatible with:
  - ROS bag exports (rosbag2 CSV export)
  - ABB RobotStudio log format
  - KUKA WorkVisual CSV export
  - Custom multi-joint sensor logs

Default sensor channels (12 channels for a 6-DOF robot arm):
  joint_{1..6}_torque, joint_{1..6}_velocity
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import torch


class RobotSensorDataset(Dataset):
    """
    Sliding-window time-series dataset for robot joint sensor streams.

    Data is segmented into overlapping windows of length ``sequence_length``
    with stride ``stride``. Windows are labelled as nominal (0) or faulty (1).

    The LSTM Autoencoder in ``robotics/robot_anomaly_detection.py`` is
    trained exclusively on nominal windows.

    Parameters
    ----------
    root : str | Path
        Directory containing CSV log files.
    sequence_length : int
        Number of time steps per window.
    stride : int
        Stride for the sliding window.
    split : {"train", "val", "test", "all"}
        Which split to return.
    train_ratio : float
        Fraction of nominal data for training.
    sensor_columns : list[str], optional
        Column names to use as features. If None, all numeric columns
        except 'fault' and 'timestamp' are used.
    label_column : str
        Name of the binary fault label column. Use None if unlabelled.
    normalize : bool
        Z-score normalize per-channel using training statistics.
    random_seed : int
        Seed for reproducible splits.
    """

    def __init__(
        self,
        root: str | Path,
        sequence_length: int = 50,
        stride: int = 10,
        split: Literal["train", "val", "test", "all"] = "train",
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        sensor_columns: Optional[List[str]] = None,
        label_column: Optional[str] = "fault",
        normalize: bool = True,
        random_seed: int = 42,
    ) -> None:
        self.root = Path(root)
        self.sequence_length = sequence_length
        self.stride = stride
        self.split = split
        self.label_column = label_column
        self.normalize = normalize
        self.random_seed = random_seed

        # Load and concatenate all CSV files
        df = self._load_csvs(sensor_columns)

        # Determine feature columns
        exclude = {"fault", "timestamp", "time", "label"}
        if sensor_columns is not None:
            self.feature_cols = sensor_columns
        else:
            self.feature_cols = [
                c for c in df.columns
                if c not in exclude and pd.api.types.is_numeric_dtype(df[c])
            ]

        self.labels_available = (label_column is not None and label_column in df.columns)

        # Build windows
        features = df[self.feature_cols].values.astype(np.float32)
        fault_labels = (
            df[label_column].values.astype(np.int64)
            if self.labels_available
            else np.zeros(len(df), dtype=np.int64)
        )

        windows, window_labels = self._create_windows(features, fault_labels)

        # Normalize using training split statistics
        self._mean: Optional[np.ndarray] = None
        self._std: Optional[np.ndarray] = None
        if normalize:
            train_windows = self._select_split(windows, window_labels, "train", train_ratio, val_ratio)
            if len(train_windows) > 0:
                self._mean = train_windows.mean(axis=(0, 1), keepdims=True)
                self._std = train_windows.std(axis=(0, 1), keepdims=True) + 1e-8

        if normalize and self._mean is not None:
            windows = (windows - self._mean) / self._std

        # Select correct split
        if split == "all":
            self.windows = windows
            self.window_labels = window_labels
        else:
            split_windows = self._select_split(windows, window_labels, split, train_ratio, val_ratio)
            split_labels = self._select_split(window_labels, window_labels, split, train_ratio, val_ratio)
            self.windows = split_windows
            self.window_labels = split_labels

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_csvs(self, sensor_columns: Optional[List[str]]) -> pd.DataFrame:
        if not self.root.exists():
            raise FileNotFoundError(
                f"Robot sensor data not found at {self.root}. "
                "Run datasets/download_datasets.sh or provide custom CSV files."
            )
        csv_files = sorted(self.root.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {self.root}.")

        dfs = []
        for f in csv_files:
            df = pd.read_csv(f)
            dfs.append(df)
        return pd.concat(dfs, ignore_index=True)

    def _create_windows(
        self,
        features: np.ndarray,
        labels: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        n = len(features)
        windows, window_labels = [], []
        for start in range(0, n - self.sequence_length + 1, self.stride):
            end = start + self.sequence_length
            window = features[start:end]
            # Window is faulty if ANY time step in the window is faulty
            label = int(labels[start:end].any())
            windows.append(window)
            window_labels.append(label)
        return np.stack(windows), np.array(window_labels, dtype=np.int64)

    def _select_split(
        self,
        data: np.ndarray,
        labels: np.ndarray,
        split: str,
        train_ratio: float,
        val_ratio: float,
    ) -> np.ndarray:
        n = len(data)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        if split == "train":
            return data[:n_train]
        elif split == "val":
            return data[n_train:n_train + n_val]
        elif split == "test":
            return data[n_train + n_val:]
        return data

    # ------------------------------------------------------------------
    # Dataset API
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int):
        window = torch.from_numpy(self.windows[idx])   # (T, C)
        label = int(self.window_labels[idx])
        return window, label

    @property
    def num_channels(self) -> int:
        return len(self.feature_cols)

    @property
    def fault_ratio(self) -> float:
        if len(self.window_labels) == 0:
            return 0.0
        return float(self.window_labels.mean())

    def __repr__(self) -> str:
        return (
            f"RobotSensorDataset(split={self.split}, "
            f"windows={len(self)}, seq_len={self.sequence_length}, "
            f"channels={self.num_channels}, fault_ratio={self.fault_ratio:.3f})"
        )


def generate_synthetic_robot_data(
    output_path: str | Path,
    n_nominal_steps: int = 10_000,
    n_fault_steps: int = 2_000,
    num_joints: int = 6,
    random_seed: int = 42,
) -> Path:
    """
    Generate synthetic robot joint sensor data for testing purposes.

    Creates a CSV file with joint torques and velocities, with injected
    fault segments (elevated torque, velocity spikes).

    Parameters
    ----------
    output_path : str | Path
        Path to save the generated CSV file.
    n_nominal_steps : int
        Number of nominal (healthy) time steps.
    n_fault_steps : int
        Number of fault time steps to append.
    num_joints : int
        Number of robot joints.
    random_seed : int
        RNG seed.

    Returns
    -------
    Path
        Path to the saved CSV file.
    """
    rng = np.random.default_rng(random_seed)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torque_nominal = rng.normal(loc=0.0, scale=1.0, size=(n_nominal_steps, num_joints))
    vel_nominal = rng.normal(loc=0.0, scale=0.5, size=(n_nominal_steps, num_joints))

    torque_fault = rng.normal(loc=3.0, scale=2.0, size=(n_fault_steps, num_joints))
    vel_fault = rng.normal(loc=0.0, scale=2.0, size=(n_fault_steps, num_joints))

    torque = np.vstack([torque_nominal, torque_fault])
    velocity = np.vstack([vel_nominal, vel_fault])
    fault_labels = np.array(
        [0] * n_nominal_steps + [1] * n_fault_steps, dtype=np.int64
    )

    columns = (
        [f"joint_{j+1}_torque" for j in range(num_joints)]
        + [f"joint_{j+1}_velocity" for j in range(num_joints)]
        + ["fault"]
    )
    data = np.hstack([torque, velocity, fault_labels.reshape(-1, 1)])
    df = pd.DataFrame(data, columns=columns)
    df["fault"] = df["fault"].astype(int)
    df.to_csv(output_path, index=False)

    print(f"Synthetic robot data saved to {output_path} "
          f"({n_nominal_steps} nominal + {n_fault_steps} fault steps)")
    return output_path

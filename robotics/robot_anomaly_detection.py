"""
Robot Joint Anomaly Detection
-------------------------------
LSTM Autoencoder for detecting anomalous behavior in industrial robot arms.

Principle:
  The autoencoder is trained exclusively on nominal (healthy) sensor data.
  At inference time, windows with reconstruction error above a learned
  threshold are flagged as anomalous.

Architecture:
  Encoder: stacked LSTM → bottleneck latent representation
  Decoder: stacked LSTM → reconstruct input sequence

Sensor inputs: joint torques and velocities for a 6-DOF robot arm
               (12 channels by default).

References:
  Malhotra et al., "LSTM-based Encoder-Decoder for Multi-sensor Anomaly
  Detection", ICML 2016 Anomaly Detection Workshop.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Literal, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


# =============================================================================
# Model
# =============================================================================

class LSTMEncoder(nn.Module):
    """LSTM encoder that compresses a time-series window to a latent vector."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        latent_dim: int,
        dropout: float,
        bidirectional: bool,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        lstm_out_dim = hidden_dim * (2 if bidirectional else 1)
        self.fc = nn.Linear(lstm_out_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x : (B, T, input_dim)

        Returns
        -------
        latent : (B, latent_dim)
        lstm_out : (B, T, hidden_dim) — for skip connections
        """
        lstm_out, (h_n, _) = self.lstm(x)
        # Use last hidden state of the final layer
        if self.lstm.bidirectional:
            h = torch.cat([h_n[-2], h_n[-1]], dim=1)
        else:
            h = h_n[-1]
        latent = self.fc(h)
        return latent, lstm_out


class LSTMDecoder(nn.Module):
    """LSTM decoder that reconstructs the input sequence from a latent vector."""

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        num_layers: int,
        output_dim: int,
        seq_len: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.fc = nn.Linear(latent_dim, hidden_dim)
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.out = nn.Linear(hidden_dim, output_dim)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        latent : (B, latent_dim)

        Returns
        -------
        reconstruction : (B, T, output_dim)
        """
        h = self.fc(latent)                         # (B, hidden_dim)
        h = h.unsqueeze(1).repeat(1, self.seq_len, 1)  # (B, T, hidden_dim)
        lstm_out, _ = self.lstm(h)
        return self.out(lstm_out)                   # (B, T, output_dim)


class LSTMAutoencoder(nn.Module):
    """
    LSTM Autoencoder for time-series anomaly detection.

    Parameters
    ----------
    input_dim : int
        Number of sensor channels.
    hidden_dim : int
        LSTM hidden state size.
    num_layers : int
        Number of stacked LSTM layers.
    latent_dim : int
        Size of the bottleneck latent representation.
    seq_len : int
        Input sequence length (must match training windows).
    dropout : float
    bidirectional : bool
        Use bidirectional LSTM in the encoder.
    """

    def __init__(
        self,
        input_dim: int = 12,
        hidden_dim: int = 64,
        num_layers: int = 2,
        latent_dim: int = 16,
        seq_len: int = 50,
        dropout: float = 0.2,
        bidirectional: bool = False,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.seq_len = seq_len

        self.encoder = LSTMEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            latent_dim=latent_dim,
            dropout=dropout,
            bidirectional=bidirectional,
        )
        self.decoder = LSTMDecoder(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            output_dim=input_dim,
            seq_len=seq_len,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, T, input_dim)

        Returns
        -------
        reconstruction : (B, T, input_dim)
        """
        latent, _ = self.encoder(x)
        return self.decoder(latent)

    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute per-sample mean squared reconstruction error.

        Parameters
        ----------
        x : (B, T, input_dim)

        Returns
        -------
        errors : (B,) float tensor
        """
        with torch.no_grad():
            recon = self(x)
            return ((x - recon) ** 2).mean(dim=(1, 2))

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def __repr__(self) -> str:
        return (
            f"LSTMAutoencoder(input_dim={self.input_dim}, "
            f"seq_len={self.seq_len}, "
            f"params={self.count_parameters():,})"
        )


# =============================================================================
# High-level detector wrapper
# =============================================================================

class RobotAnomalyDetector:
    """
    High-level wrapper for training, threshold calibration, and inference.

    Workflow:
      1. Train autoencoder on nominal data only
      2. Compute reconstruction errors on a held-out nominal set
      3. Set threshold at (e.g.) the 95th percentile of nominal errors
      4. Flag test windows above threshold as anomalous

    Parameters
    ----------
    input_dim : int
        Number of sensor channels.
    hidden_dim : int
    num_layers : int
    latent_dim : int
    seq_len : int
    dropout : float
    bidirectional : bool
    device : str
    """

    def __init__(
        self,
        input_dim: int = 12,
        hidden_dim: int = 64,
        num_layers: int = 2,
        latent_dim: int = 16,
        seq_len: int = 50,
        dropout: float = 0.2,
        bidirectional: bool = False,
        device: str = "auto",
    ) -> None:
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        self.model = LSTMAutoencoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            latent_dim=latent_dim,
            seq_len=seq_len,
            dropout=dropout,
            bidirectional=bidirectional,
        ).to(device)

        self.threshold: Optional[float] = None
        self.history: Dict[str, list] = {"train_loss": [], "val_loss": []}

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 100,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        early_stopping_patience: int = 15,
        checkpoint_path: Optional[str] = None,
    ) -> Dict[str, list]:
        """
        Train the LSTM autoencoder on nominal data.

        Parameters
        ----------
        train_loader : DataLoader
            Should yield (window, label) tuples; only nominal windows
            should be in the training set.
        val_loader : DataLoader, optional
        epochs : int
        learning_rate : float
        weight_decay : float
        early_stopping_patience : int
        checkpoint_path : str, optional

        Returns
        -------
        dict : training history
        """
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=30, gamma=0.5
        )
        criterion = nn.MSELoss()

        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(1, epochs + 1):
            train_loss = self._train_epoch(train_loader, optimizer, criterion)
            self.history["train_loss"].append(train_loss)

            val_loss_str = ""
            if val_loader is not None:
                val_loss = self._eval_epoch(val_loader, criterion)
                self.history["val_loss"].append(val_loss)
                val_loss_str = f" | Val Loss: {val_loss:.6f}"

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    if checkpoint_path:
                        self.save(checkpoint_path)
                else:
                    patience_counter += 1
                    if patience_counter >= early_stopping_patience:
                        print(f"Early stopping at epoch {epoch}.")
                        break

            if epoch % 10 == 0 or epoch == 1:
                print(
                    f"Epoch {epoch:4d}/{epochs} | "
                    f"Train Loss: {train_loss:.6f}{val_loss_str}"
                )
            scheduler.step()

        return self.history

    def _train_epoch(self, loader: DataLoader, optimizer, criterion) -> float:
        self.model.train()
        total_loss = 0.0
        for batch in tqdm(loader, desc="Training", leave=False):
            x = batch[0].to(self.device)  # (B, T, C)
            optimizer.zero_grad()
            recon = self.model(x)
            loss = criterion(recon, x)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item() * len(x)
        return total_loss / len(loader.dataset)

    def _eval_epoch(self, loader: DataLoader, criterion) -> float:
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for batch in loader:
                x = batch[0].to(self.device)
                recon = self.model(x)
                total_loss += criterion(recon, x).item() * len(x)
        return total_loss / len(loader.dataset)

    # ------------------------------------------------------------------
    # Threshold calibration
    # ------------------------------------------------------------------

    def calibrate_threshold(
        self,
        nominal_loader: DataLoader,
        method: Literal["percentile", "sigma", "fixed"] = "percentile",
        percentile: float = 95.0,
        sigma: float = 3.0,
        fixed_value: Optional[float] = None,
    ) -> float:
        """
        Compute the anomaly detection threshold from nominal validation data.

        Parameters
        ----------
        nominal_loader : DataLoader
            DataLoader yielding nominal (fault=0) windows.
        method : str
            "percentile" — threshold at `percentile`-th percentile of nominal errors.
            "sigma"      — mean + sigma × std of nominal errors.
            "fixed"      — use `fixed_value` directly.
        percentile : float
        sigma : float
        fixed_value : float, optional

        Returns
        -------
        threshold : float
        """
        self.model.eval()
        errors = []
        with torch.no_grad():
            for batch in tqdm(nominal_loader, desc="Calibrating threshold"):
                x = batch[0].to(self.device)
                err = self.model.reconstruction_error(x)
                errors.extend(err.cpu().numpy().tolist())

        errors = np.array(errors)

        if method == "percentile":
            self.threshold = float(np.percentile(errors, percentile))
        elif method == "sigma":
            self.threshold = float(errors.mean() + sigma * errors.std())
        elif method == "fixed":
            if fixed_value is None:
                raise ValueError("fixed_value must be provided when method='fixed'.")
            self.threshold = fixed_value
        else:
            raise ValueError(f"Unknown threshold method: {method}")

        print(
            f"Threshold calibrated: {self.threshold:.6f} "
            f"(method={method}, errors: mean={errors.mean():.6f}, "
            f"std={errors.std():.6f}, p95={np.percentile(errors, 95):.6f})"
        )
        return self.threshold

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, x: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict anomaly labels and reconstruction errors.

        Parameters
        ----------
        x : (B, T, input_dim) tensor

        Returns
        -------
        predictions : np.ndarray (B,) int — 1 = anomaly, 0 = nominal
        errors : np.ndarray (B,) float — per-sample reconstruction error
        """
        if self.threshold is None:
            raise RuntimeError(
                "Threshold not set. Call calibrate_threshold() first."
            )
        self.model.eval()
        x = x.to(self.device)
        errors = self.model.reconstruction_error(x).cpu().numpy()
        predictions = (errors >= self.threshold).astype(int)
        return predictions, errors

    def score_stream(
        self,
        sensor_stream: np.ndarray,
        stride: int = 10,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply sliding-window anomaly detection to a continuous sensor stream.

        Parameters
        ----------
        sensor_stream : np.ndarray (T_total, input_dim)
            Full sensor recording.
        stride : int
            Sliding window stride.

        Returns
        -------
        timestamps : np.ndarray (N_windows,) — center time index of each window
        anomaly_scores : np.ndarray (N_windows,) — reconstruction errors
        """
        seq_len = self.model.seq_len
        n = len(sensor_stream)
        scores, timestamps = [], []

        for start in range(0, n - seq_len + 1, stride):
            end = start + seq_len
            window = torch.from_numpy(
                sensor_stream[start:end].astype(np.float32)
            ).unsqueeze(0)
            err = self.model.reconstruction_error(window.to(self.device))
            scores.append(err.item())
            timestamps.append(start + seq_len // 2)

        return np.array(timestamps), np.array(scores)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "model_config": {
                "input_dim": self.model.input_dim,
                "seq_len": self.model.seq_len,
            },
            "threshold": self.threshold,
            "history": self.history,
        }, path)
        print(f"Checkpoint saved: {path}")

    def load(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.threshold = checkpoint.get("threshold")
        self.history = checkpoint.get("history", {})
        print(f"Checkpoint loaded: {path}  |  threshold={self.threshold}")

"""
Digital Twin Simulator
-----------------------
Discrete-event simulation of a multi-stage manufacturing production line.

Each stage has:
  - A nominal cycle time with stochastic variation
  - A random failure rate (Poisson process)
  - A mean time to repair (MTTR)
  - An inter-stage buffer (finite capacity WIP queue)

The simulator generates synthetic sensor streams (temperature, vibration,
pressure, cycle time) that mimic real industrial IoT data. These streams
are used to:
  1. Train and evaluate anomaly detection models
  2. Test digital twin synchronization (twin_sync.py)
  3. Benchmark the production RL agent against a simulation target

References:
  Grieves, "Digital Twin: Manufacturing Excellence through Virtual Factory
  Replication", White Paper, 2014.
"""

from __future__ import annotations

import csv
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Generator, List, Optional

import numpy as np


# =============================================================================
# Stage model
# =============================================================================

@dataclass
class StageConfig:
    name: str
    nominal_cycle_time: float       # seconds
    cycle_time_std: float
    failure_rate: float             # failures per time step
    mttr: float                     # mean time to repair (seconds)


@dataclass
class StageState:
    name: str
    status: str = "idle"            # idle | running | failed | repairing
    cycle_time_remaining: float = 0.0
    repair_time_remaining: float = 0.0
    total_produced: int = 0
    total_downtime: float = 0.0
    total_failures: int = 0

    # Sensor readings (last observed values)
    temperature: float = 20.0
    vibration: float = 0.0
    pressure: float = 1.0
    last_cycle_time: float = 0.0


# =============================================================================
# Simulator
# =============================================================================

class TwinSimulator:
    """
    Discrete-event simulation of a multi-stage manufacturing line.

    The simulator is parameterized by a YAML config (see
    configs/digital_twin.yaml) or constructed directly.

    Parameters
    ----------
    stage_configs : list[StageConfig]
        Configuration for each production stage in order.
    inter_stage_capacity : int
        WIP buffer capacity between stages.
    time_step : float
        Simulation time step in seconds.
    sensor_noise_std : float
        Standard deviation of Gaussian noise on sensor readings.
    random_seed : int
    """

    def __init__(
        self,
        stage_configs: Optional[List[StageConfig]] = None,
        inter_stage_capacity: int = 10,
        time_step: float = 1.0,
        sensor_noise_std: float = 0.01,
        random_seed: int = 42,
    ) -> None:
        self.stage_configs = stage_configs or self._default_stages()
        self.inter_stage_capacity = inter_stage_capacity
        self.time_step = time_step
        self.noise_std = sensor_noise_std
        self.rng = np.random.default_rng(random_seed)

        self.num_stages = len(self.stage_configs)
        self._stages: List[StageState] = []
        self._buffers: List[int] = []  # WIP units between stages
        self._sim_time: float = 0.0
        self._history: List[Dict] = []

        self._init_states()

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def _default_stages(self) -> List[StageConfig]:
        return [
            StageConfig("machining",  nominal_cycle_time=30, cycle_time_std=2.0, failure_rate=0.005, mttr=60),
            StageConfig("assembly",   nominal_cycle_time=45, cycle_time_std=3.0, failure_rate=0.003, mttr=90),
            StageConfig("inspection", nominal_cycle_time=15, cycle_time_std=1.0, failure_rate=0.001, mttr=30),
            StageConfig("packaging",  nominal_cycle_time=20, cycle_time_std=1.5, failure_rate=0.002, mttr=45),
        ]

    def _init_states(self) -> None:
        self._stages = [
            StageState(name=cfg.name, status="idle")
            for cfg in self.stage_configs
        ]
        # n-1 buffers between n stages
        self._buffers = [0] * (self.num_stages - 1)
        self._sim_time = 0.0
        self._history = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self, random_seed: Optional[int] = None) -> None:
        """Reset simulator to initial state."""
        if random_seed is not None:
            self.rng = np.random.default_rng(random_seed)
        self._init_states()

    def step(self) -> Dict:
        """
        Advance the simulation by one time step.

        Returns
        -------
        dict
            Snapshot of the system state and sensor readings for this step.
        """
        self._sim_time += self.time_step
        self._tick_stages()
        snapshot = self._build_snapshot()
        self._history.append(snapshot)
        return snapshot

    def run(self, num_steps: int, verbose: bool = False) -> List[Dict]:
        """
        Run the simulation for a fixed number of steps.

        Parameters
        ----------
        num_steps : int
        verbose : bool
            Print a progress summary every 100 steps.

        Returns
        -------
        list[dict]
            Full simulation history.
        """
        self.reset()
        for i in range(num_steps):
            snap = self.step()
            if verbose and i % 100 == 0:
                self._print_summary(snap)
        return self._history

    def stream(self, num_steps: int) -> Generator[Dict, None, None]:
        """
        Generator that yields one snapshot per time step.
        Useful for real-time interfaces and the TwinSync module.
        """
        self.reset()
        for _ in range(num_steps):
            yield self.step()

    # ------------------------------------------------------------------
    # Core simulation logic
    # ------------------------------------------------------------------

    def _tick_stages(self) -> None:
        dt = self.time_step

        for i, (cfg, state) in enumerate(zip(self.stage_configs, self._stages)):

            if state.status == "repairing":
                state.repair_time_remaining -= dt
                state.total_downtime += dt
                if state.repair_time_remaining <= 0:
                    state.status = "idle"
                    state.repair_time_remaining = 0.0

            elif state.status == "running":
                state.cycle_time_remaining -= dt
                if state.cycle_time_remaining <= 0:
                    # Job completed
                    state.total_produced += 1
                    state.last_cycle_time = abs(state.cycle_time_remaining) + dt
                    state.status = "idle"
                    state.cycle_time_remaining = 0.0
                    # Push to downstream buffer
                    if i < self.num_stages - 1:
                        if self._buffers[i] < self.inter_stage_capacity:
                            self._buffers[i] += 1

            elif state.status == "idle":
                # Check for failure
                if self.rng.random() < cfg.failure_rate * dt:
                    state.status = "repairing"
                    state.total_failures += 1
                    state.repair_time_remaining = self.rng.exponential(cfg.mttr)
                    continue

                # Pick up job from upstream buffer (or raw material feed at stage 0)
                can_start = (i == 0) or (self._buffers[i - 1] > 0)
                if can_start:
                    if i > 0:
                        self._buffers[i - 1] -= 1
                    cycle_time = max(
                        self.rng.normal(cfg.nominal_cycle_time, cfg.cycle_time_std),
                        1.0,
                    )
                    state.status = "running"
                    state.cycle_time_remaining = cycle_time

            # Update sensor readings
            self._update_sensors(state, cfg)

    def _update_sensors(self, state: StageState, cfg: StageConfig) -> None:
        noise = lambda: self.rng.normal(0, self.noise_std)

        if state.status == "running":
            # Elevated temperature and vibration during operation
            state.temperature = 60.0 + 20.0 * (
                1.0 - state.cycle_time_remaining / max(cfg.nominal_cycle_time, 1)
            ) + noise() * 5
            state.vibration = 2.5 + noise() * 0.5
            state.pressure = 3.0 + noise() * 0.1
        elif state.status == "repairing":
            # High temperature during failure/repair
            state.temperature = 90.0 + noise() * 10
            state.vibration = 5.0 + noise()
            state.pressure = 1.5 + noise() * 0.2
        else:
            # Idle baseline
            state.temperature = 25.0 + noise() * 2
            state.vibration = 0.2 + noise() * 0.05
            state.pressure = 1.0 + noise() * 0.02

    # ------------------------------------------------------------------
    # Snapshot & reporting
    # ------------------------------------------------------------------

    def _build_snapshot(self) -> Dict:
        snap: Dict = {"sim_time": self._sim_time}
        for i, state in enumerate(self._stages):
            prefix = f"stage_{i}_{state.name}"
            snap[f"{prefix}_status"] = state.status
            snap[f"{prefix}_temperature"] = round(state.temperature, 3)
            snap[f"{prefix}_vibration"] = round(state.vibration, 4)
            snap[f"{prefix}_pressure"] = round(state.pressure, 4)
            snap[f"{prefix}_cycle_time_remaining"] = round(state.cycle_time_remaining, 2)
            snap[f"{prefix}_total_produced"] = state.total_produced
            snap[f"{prefix}_total_downtime"] = round(state.total_downtime, 2)
            snap[f"{prefix}_total_failures"] = state.total_failures
        for i, buf in enumerate(self._buffers):
            snap[f"buffer_{i}"] = buf
        return snap

    def _print_summary(self, snap: Dict) -> None:
        t = snap["sim_time"]
        produced = [
            snap.get(f"stage_{i}_{s.name}_total_produced", 0)
            for i, s in enumerate(self._stages)
        ]
        print(f"t={t:6.1f}s | produced={produced}")

    def summary(self) -> Dict:
        """Return aggregate statistics from the last simulation run."""
        if not self._history:
            return {}
        stats: Dict = {}
        for i, state in enumerate(self._stages):
            stats[f"stage_{i}_{state.name}"] = {
                "total_produced": state.total_produced,
                "total_failures": state.total_failures,
                "total_downtime_s": round(state.total_downtime, 2),
                "availability": round(
                    1.0 - state.total_downtime / max(self._sim_time, 1), 4
                ),
            }
        stats["total_sim_time_s"] = self._sim_time
        return stats

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_csv(self, output_path: str | Path) -> Path:
        """
        Export simulation history to CSV.

        Parameters
        ----------
        output_path : str | Path

        Returns
        -------
        Path
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if not self._history:
            raise RuntimeError("No simulation data. Run .run() or .step() first.")

        fieldnames = list(self._history[0].keys())
        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self._history)

        print(f"Simulation data exported to {output_path} ({len(self._history)} steps)")
        return output_path

    @classmethod
    def from_config(cls, config: Dict) -> "TwinSimulator":
        """
        Construct a TwinSimulator from a parsed YAML config dict.

        Parameters
        ----------
        config : dict
            Parsed contents of configs/digital_twin.yaml.

        Returns
        -------
        TwinSimulator
        """
        stages = [
            StageConfig(
                name=s["name"],
                nominal_cycle_time=s["nominal_cycle_time"],
                cycle_time_std=s.get("cycle_time_std", 1.0),
                failure_rate=s.get("failure_rate", 0.002),
                mttr=s.get("mttr", 60),
            )
            for s in config.get("stages", [])
        ]
        sim_cfg = config.get("simulation", {})
        return cls(
            stage_configs=stages or None,
            inter_stage_capacity=config.get("buffers", {}).get("inter_stage_capacity", 10),
            time_step=sim_cfg.get("time_step_seconds", 1.0),
            sensor_noise_std=config.get("sensors", {}).get("noise_std", 0.01),
            random_seed=sim_cfg.get("random_seed", 42),
        )

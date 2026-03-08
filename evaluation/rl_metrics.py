"""
Reinforcement Learning Evaluation Metrics
-------------------------------------------
Metrics and visualization for the production scheduling RL agent.

Metrics:
  - Cumulative episode reward curve
  - Overall Equipment Effectiveness (OEE) proxy
  - Throughput (jobs completed per episode)
  - Machine utilization
  - Tardiness rate

OEE formula used:
  OEE = Availability × Performance × Quality
  Simplified proxy: completed_jobs / theoretical_max_jobs
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =============================================================================
# OEE
# =============================================================================

def compute_oee(
    completed_jobs: int,
    max_episode_steps: int,
    num_machines: int,
    avg_processing_time: float,
    availability: Optional[float] = None,
) -> Dict[str, float]:
    """
    Compute Overall Equipment Effectiveness (OEE) proxy.

    OEE = Availability × Performance × Quality

    Parameters
    ----------
    completed_jobs : int
        Total jobs completed in the episode.
    max_episode_steps : int
        Total number of time steps in the episode.
    num_machines : int
        Number of machines in the production line.
    avg_processing_time : float
        Average processing time per job (steps).
    availability : float, optional
        Fraction of time machines were not broken. If None, inferred as 1.0.

    Returns
    -------
    dict with keys: oee, availability, performance, quality, throughput_per_step
    """
    theoretical_max = max_episode_steps * num_machines / max(avg_processing_time, 1)
    performance = min(completed_jobs / max(theoretical_max, 1), 1.0)
    avail = availability if availability is not None else 1.0
    quality = 1.0  # assume all completed jobs pass quality — adjust as needed
    oee = avail * performance * quality
    throughput_per_step = completed_jobs / max(max_episode_steps, 1)

    return {
        "oee": round(oee, 4),
        "availability": round(avail, 4),
        "performance": round(performance, 4),
        "quality": round(quality, 4),
        "throughput_per_step": round(throughput_per_step, 6),
        "completed_jobs": completed_jobs,
        "theoretical_max": round(theoretical_max, 2),
    }


# =============================================================================
# Reward analysis
# =============================================================================

def compute_throughput(episode_infos: List[Dict]) -> Dict[str, float]:
    """
    Compute throughput statistics across multiple episodes.

    Parameters
    ----------
    episode_infos : list[dict]
        List of info dicts returned by ManufacturingEnv at episode end.
        Expected keys: total_completed, total_late.

    Returns
    -------
    dict with mean/std of completed jobs and tardiness ratio.
    """
    completed = np.array([info.get("total_completed", 0) for info in episode_infos])
    late = np.array([info.get("total_late", 0) for info in episode_infos])

    return {
        "mean_completed": float(completed.mean()),
        "std_completed": float(completed.std()),
        "min_completed": float(completed.min()),
        "max_completed": float(completed.max()),
        "mean_tardiness": float(late.mean()),
        "tardiness_ratio": float((late / np.maximum(completed, 1)).mean()),
    }


def smooth_rewards(rewards: List[float], window: int = 10) -> np.ndarray:
    """Apply a simple moving average to reward history for plotting."""
    if len(rewards) < window:
        return np.array(rewards)
    kernel = np.ones(window) / window
    return np.convolve(rewards, kernel, mode="valid")


# =============================================================================
# Visualization
# =============================================================================

def plot_reward_curve(
    rewards: List[float],
    title: str = "Training Reward Curve",
    smoothing_window: int = 20,
    save_path: Optional[str] = None,
    xlabel: str = "Episode",
) -> plt.Figure:
    """
    Plot cumulative episode rewards over training.

    Parameters
    ----------
    rewards : list[float]
        Episode rewards in order.
    title : str
    smoothing_window : int
        Moving average window for the smoothed overlay.
    save_path : str, optional
    xlabel : str

    Returns
    -------
    matplotlib Figure
    """
    rewards_np = np.array(rewards)
    episodes = np.arange(1, len(rewards_np) + 1)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(episodes, rewards_np, alpha=0.3, color="steelblue", linewidth=0.8, label="Raw")

    if len(rewards) >= smoothing_window:
        smoothed = smooth_rewards(rewards, smoothing_window)
        smooth_x = np.arange(smoothing_window, len(rewards) + 1)
        ax.plot(smooth_x, smoothed, color="steelblue", linewidth=2,
                label=f"Smoothed (w={smoothing_window})")

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Episode Reward")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_oee_comparison(
    oee_results: Dict[str, Dict],
    title: str = "OEE Comparison: Agents",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Bar chart comparing OEE components across multiple agents or conditions.

    Parameters
    ----------
    oee_results : dict[str, dict]
        Keys = agent names, values = output of compute_oee().
    title : str
    save_path : str, optional

    Returns
    -------
    matplotlib Figure
    """
    agents = list(oee_results.keys())
    metrics = ["availability", "performance", "quality", "oee"]
    x = np.arange(len(agents))
    width = 0.18

    fig, ax = plt.subplots(figsize=(max(7, len(agents) * 2), 5))

    for i, metric in enumerate(metrics):
        values = [oee_results[a].get(metric, 0) for a in agents]
        ax.bar(x + i * width, values, width, label=metric.capitalize())

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(agents, rotation=20, ha="right")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.15)
    ax.set_title(title)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_machine_utilization(
    machine_utilization: List[float],
    machine_names: Optional[List[str]] = None,
    title: str = "Machine Utilization",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Horizontal bar chart of per-machine utilization rates.

    Parameters
    ----------
    machine_utilization : list[float]
        Utilization fraction [0, 1] for each machine.
    machine_names : list[str], optional
    title : str
    save_path : str, optional

    Returns
    -------
    matplotlib Figure
    """
    n = len(machine_utilization)
    names = machine_names or [f"Machine {i+1}" for i in range(n)]
    colors = ["#2ecc71" if u >= 0.7 else "#e74c3c" if u < 0.4 else "#f39c12"
              for u in machine_utilization]

    fig, ax = plt.subplots(figsize=(7, max(3, n * 0.5)))
    y_pos = np.arange(n)
    ax.barh(y_pos, machine_utilization, color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.set_xlim(0, 1.0)
    ax.axvline(0.85, color="green", linestyle="--", alpha=0.6, label="Target (85%)")
    ax.set_xlabel("Utilization")
    ax.set_title(title)
    ax.legend()
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def summarize_training(
    rewards: List[float],
    episode_infos: Optional[List[Dict]] = None,
) -> Dict:
    """
    Produce a summary dictionary of training results.

    Parameters
    ----------
    rewards : list[float]
    episode_infos : list[dict], optional

    Returns
    -------
    dict
    """
    rewards_np = np.array(rewards)
    summary: Dict = {
        "total_episodes": len(rewards),
        "mean_reward": float(rewards_np.mean()),
        "std_reward": float(rewards_np.std()),
        "best_reward": float(rewards_np.max()),
        "worst_reward": float(rewards_np.min()),
        "last_10_mean": float(rewards_np[-10:].mean()) if len(rewards) >= 10 else float(rewards_np.mean()),
    }
    if episode_infos:
        throughput = compute_throughput(episode_infos)
        summary.update(throughput)
    return summary

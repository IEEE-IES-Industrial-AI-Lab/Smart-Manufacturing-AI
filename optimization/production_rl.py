"""
RL-based Production Scheduling Optimization
---------------------------------------------
Formulates job-shop production scheduling as a Markov Decision Process
and trains a PPO agent to maximize throughput while minimizing tardiness
and machine downtime.

Environment:
  - N machines, M job types
  - Stochastic processing times, random machine breakdowns
  - Discrete action: assign next queued job to a specific machine
  - Reward: throughput gain − tardiness penalty − downtime cost

Algorithm: Proximal Policy Optimization (PPO) via Stable-Baselines3.

References:
  Schulman et al., "Proximal Policy Optimization Algorithms", arXiv 2017.
  Zhang et al., "Learning to Dispatch for Job Shop Scheduling via Deep RL",
  NeurIPS 2020.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import (
        BaseCallback,
        CheckpointCallback,
        EvalCallback,
    )
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False


# =============================================================================
# Manufacturing Environment
# =============================================================================

class ManufacturingEnv(gym.Env):
    """
    Gymnasium environment for job-shop production scheduling.

    State space (flat vector):
      - Machine states: 0=idle, 1=busy, 2=broken (num_machines,)
      - Machine remaining time: steps until current job completes (num_machines,)
      - Machine repair time: steps until repair done (num_machines,)
      - Queue lengths: number of waiting jobs per type (num_job_types,)
      - Normalized step count (1,)
      Total: 3*num_machines + num_job_types + 1

    Action space:
      Discrete(num_machines + 1)
      - Action i (0..num_machines-1): dispatch next queued job to machine i
      - Action num_machines: "wait" — do nothing this step

    Reward:
      +reward_weights["throughput"] for each completed job
      +reward_weights["tardiness_penalty"] for each job late (negative weight)
      +reward_weights["downtime_penalty"] for idle capable machines (negative weight)

    Parameters
    ----------
    num_machines : int
    num_job_types : int
    max_queue_length : int
    processing_time_range : tuple[int, int]
    breakdown_probability : float
    repair_time : int
    max_episode_steps : int
    reward_weights : dict
    random_seed : int
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(
        self,
        num_machines: int = 5,
        num_job_types: int = 4,
        max_queue_length: int = 20,
        processing_time_range: Tuple[int, int] = (1, 10),
        breakdown_probability: float = 0.02,
        repair_time: int = 5,
        max_episode_steps: int = 500,
        reward_weights: Optional[Dict[str, float]] = None,
        random_seed: Optional[int] = None,
    ) -> None:
        super().__init__()

        self.num_machines = num_machines
        self.num_job_types = num_job_types
        self.max_queue_length = max_queue_length
        self.proc_time_low, self.proc_time_high = processing_time_range
        self.breakdown_prob = breakdown_probability
        self.repair_time = repair_time
        self.max_episode_steps = max_episode_steps
        self.reward_weights = reward_weights or {
            "throughput": 1.0,
            "tardiness_penalty": -0.5,
            "downtime_penalty": -0.3,
        }

        self.rng = np.random.default_rng(random_seed)

        # Observation: machine_state + remaining_time + repair_time + queue + step
        obs_dim = 3 * num_machines + num_job_types + 1
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(obs_dim,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(num_machines + 1)

        # Internal state (initialized in reset)
        self._machine_state: np.ndarray = np.zeros(num_machines, dtype=np.int32)
        self._machine_remaining: np.ndarray = np.zeros(num_machines, dtype=np.int32)
        self._machine_repair: np.ndarray = np.zeros(num_machines, dtype=np.int32)
        self._job_queue: np.ndarray = np.zeros(num_job_types, dtype=np.int32)
        self._step_count: int = 0
        self._total_completed: int = 0
        self._total_late: int = 0

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self._machine_state = np.zeros(self.num_machines, dtype=np.int32)
        self._machine_remaining = np.zeros(self.num_machines, dtype=np.int32)
        self._machine_repair = np.zeros(self.num_machines, dtype=np.int32)
        self._job_queue = self.rng.integers(
            0, self.max_queue_length // 2,
            size=self.num_job_types,
            dtype=np.int32,
        )
        self._step_count = 0
        self._total_completed = 0
        self._total_late = 0

        return self._get_obs(), self._get_info()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        reward = 0.0

        # --- Execute action ---
        if action < self.num_machines:
            machine_id = action
            if (
                self._machine_state[machine_id] == 0       # idle
                and self._job_queue.sum() > 0              # jobs available
            ):
                # Select a random job type from the queue
                available = np.where(self._job_queue > 0)[0]
                job_type = self.rng.choice(available)
                self._job_queue[job_type] -= 1

                proc_time = self.rng.integers(
                    self.proc_time_low,
                    self.proc_time_high + 1,
                )
                self._machine_state[machine_id] = 1          # busy
                self._machine_remaining[machine_id] = proc_time

        # --- Tick machines ---
        completed_this_step = 0
        for m in range(self.num_machines):
            if self._machine_state[m] == 1:  # busy
                self._machine_remaining[m] -= 1
                if self._machine_remaining[m] <= 0:
                    self._machine_state[m] = 0  # idle
                    self._machine_remaining[m] = 0
                    completed_this_step += 1

            elif self._machine_state[m] == 2:  # broken
                self._machine_repair[m] -= 1
                if self._machine_repair[m] <= 0:
                    self._machine_state[m] = 0  # repaired
                    self._machine_repair[m] = 0

            elif self._machine_state[m] == 0:  # idle
                # Random breakdown
                if self.rng.random() < self.breakdown_prob:
                    self._machine_state[m] = 2  # broken
                    self._machine_repair[m] = self.repair_time

        # --- Arrivals ---
        for j in range(self.num_job_types):
            if self.rng.random() < 0.3:
                new_jobs = self.rng.integers(1, 4)
                self._job_queue[j] = min(
                    self._job_queue[j] + new_jobs,
                    self.max_queue_length,
                )

        # --- Reward ---
        self._total_completed += completed_this_step
        reward += self.reward_weights["throughput"] * completed_this_step

        # Tardiness: jobs waiting > 10 steps
        overdue = int((self._job_queue > 10).sum())
        reward += self.reward_weights["tardiness_penalty"] * overdue
        self._total_late += overdue

        # Downtime: idle machines while queue is non-empty
        idle_machines = int(
            ((self._machine_state == 0) & (self._job_queue.sum() > 0)).sum()
        )
        reward += self.reward_weights["downtime_penalty"] * idle_machines

        self._step_count += 1
        terminated = False
        truncated = self._step_count >= self.max_episode_steps

        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def render(self) -> None:
        print(
            f"Step {self._step_count:4d} | "
            f"Machines: {self._machine_state} | "
            f"Queue: {self._job_queue} | "
            f"Completed: {self._total_completed}"
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        machine_state_norm = self._machine_state.astype(np.float32) / 2.0
        remaining_norm = self._machine_remaining.astype(np.float32) / max(self.proc_time_high, 1)
        repair_norm = self._machine_repair.astype(np.float32) / max(self.repair_time, 1)
        queue_norm = self._job_queue.astype(np.float32) / max(self.max_queue_length, 1)
        step_norm = np.array([self._step_count / self.max_episode_steps], dtype=np.float32)

        return np.concatenate([
            machine_state_norm,
            remaining_norm,
            repair_norm,
            queue_norm,
            step_norm,
        ])

    def _get_info(self) -> dict:
        return {
            "total_completed": self._total_completed,
            "total_late": self._total_late,
            "step": self._step_count,
            "queue_total": int(self._job_queue.sum()),
        }

    def compute_oee(self) -> float:
        """
        Compute a simplified Overall Equipment Effectiveness proxy.
        OEE = (completed jobs) / (theoretical max throughput)
        """
        theoretical_max = (
            self.max_episode_steps
            * self.num_machines
            / ((self.proc_time_low + self.proc_time_high) / 2)
        )
        return min(self._total_completed / max(theoretical_max, 1), 1.0)


# =============================================================================
# Agent wrapper
# =============================================================================

class ProductionAgent:
    """
    PPO agent for production scheduling.

    Wraps Stable-Baselines3's PPO implementation with convenience methods
    for training, evaluation, and loading checkpoints.

    Parameters
    ----------
    env_kwargs : dict
        Keyword arguments passed to ManufacturingEnv.
    ppo_kwargs : dict
        Keyword arguments passed to PPO (overrides defaults).
    device : str
    """

    def __init__(
        self,
        env_kwargs: Optional[Dict[str, Any]] = None,
        ppo_kwargs: Optional[Dict[str, Any]] = None,
        device: str = "auto",
    ) -> None:
        if not SB3_AVAILABLE:
            raise ImportError(
                "stable-baselines3 is required. "
                "Install with: pip install stable-baselines3"
            )

        env_kwargs = env_kwargs or {}
        ppo_kwargs = ppo_kwargs or {}

        def make_env():
            env = ManufacturingEnv(**env_kwargs)
            env = Monitor(env)
            return env

        self.env = DummyVecEnv([make_env])
        self.eval_env = DummyVecEnv([make_env])

        default_ppo = {
            "policy": "MlpPolicy",
            "learning_rate": 3e-4,
            "n_steps": 2048,
            "batch_size": 64,
            "n_epochs": 10,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "ent_coef": 0.01,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
            "verbose": 1,
            "device": device,
        }
        default_ppo.update(ppo_kwargs)
        self.model = PPO(env=self.env, **default_ppo)

    def train(
        self,
        total_timesteps: int = 1_000_000,
        eval_freq: int = 10_000,
        n_eval_episodes: int = 10,
        checkpoint_dir: str = "checkpoints/production_rl",
        log_dir: str = "logs/production_rl",
    ) -> None:
        """
        Train the PPO agent.

        Parameters
        ----------
        total_timesteps : int
        eval_freq : int
        n_eval_episodes : int
        checkpoint_dir : str
        log_dir : str
        """
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(log_dir).mkdir(parents=True, exist_ok=True)

        callbacks = [
            CheckpointCallback(
                save_freq=max(eval_freq // 2, 1),
                save_path=checkpoint_dir,
                name_prefix="ppo_manufacturing",
            ),
            EvalCallback(
                self.eval_env,
                best_model_save_path=checkpoint_dir,
                log_path=log_dir,
                eval_freq=eval_freq,
                n_eval_episodes=n_eval_episodes,
                deterministic=True,
                verbose=1,
            ),
        ]

        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            tb_log_name="PPO_Manufacturing",
            reset_num_timesteps=True,
        )

    def evaluate(
        self,
        n_episodes: int = 20,
        deterministic: bool = True,
    ) -> Dict[str, float]:
        """
        Evaluate the trained agent.

        Returns
        -------
        dict with mean_reward, std_reward, mean_oee, mean_completed
        """
        episode_rewards, episode_oee, episode_completed = [], [], []
        obs = self.eval_env.reset()

        for ep in range(n_episodes):
            done = False
            ep_reward = 0.0
            while not done:
                action, _ = self.model.predict(obs, deterministic=deterministic)
                obs, reward, done, info = self.eval_env.step(action)
                ep_reward += reward[0]

            episode_rewards.append(ep_reward)
            if info and info[0]:
                ep_info = info[0]
                # Recompute OEE from completed jobs
                completed = ep_info.get("total_completed", 0)
                episode_completed.append(completed)

            obs = self.eval_env.reset()

        results = {
            "mean_reward": float(np.mean(episode_rewards)),
            "std_reward": float(np.std(episode_rewards)),
            "mean_completed": float(np.mean(episode_completed)) if episode_completed else 0.0,
        }
        print(
            f"Evaluation over {n_episodes} episodes: "
            f"mean_reward={results['mean_reward']:.2f} ± {results['std_reward']:.2f}, "
            f"mean_completed={results['mean_completed']:.1f}"
        )
        return results

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.model.save(path)
        print(f"Agent saved: {path}")

    def load(self, path: str) -> None:
        self.model = PPO.load(path, env=self.env)
        print(f"Agent loaded: {path}")

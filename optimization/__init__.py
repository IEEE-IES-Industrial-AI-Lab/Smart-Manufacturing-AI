"""
optimization
------------
Reinforcement learning for production scheduling optimization.

Classes:
  - ManufacturingEnv  : Custom Gymnasium environment for job-shop scheduling
  - ProductionAgent   : PPO-based agent wrapper for training and inference
"""

from optimization.production_rl import ManufacturingEnv, ProductionAgent

__all__ = ["ManufacturingEnv", "ProductionAgent"]

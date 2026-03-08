"""
datasets
--------
Dataset loaders for Smart-Manufacturing-AI.

Available loaders:
  - MVTecDataset       : MVTec Anomaly Detection benchmark
  - NEUSurfaceDataset  : NEU Steel Surface Defect dataset
  - RobotSensorDataset : Robot joint sensor time-series dataset
"""

from datasets.mvtec_loader import MVTecDataset
from datasets.neu_surface_loader import NEUSurfaceDataset
from datasets.robot_dataset import RobotSensorDataset

__all__ = ["MVTecDataset", "NEUSurfaceDataset", "RobotSensorDataset"]

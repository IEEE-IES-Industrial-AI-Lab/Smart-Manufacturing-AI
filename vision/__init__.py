"""
vision
------
Vision-based defect detection and surface inspection for smart manufacturing.

Classes:
  - DefectDetector    : CNN-based classifier (ResNet-50 / EfficientNet-B4)
  - SurfaceInspector  : Sliding-window inspection pipeline with heatmap output
  - ViTInspector      : Vision Transformer inspector with attention rollout
"""

from vision.defect_detection import DefectDetector
from vision.surface_inspection import SurfaceInspector
from vision.vit_inspector import ViTInspector

__all__ = ["DefectDetector", "SurfaceInspector", "ViTInspector"]

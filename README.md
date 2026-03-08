# Smart Manufacturing AI Toolkit

[![IEEE IES](https://img.shields.io/badge/IEEE-Industrial%20Electronics%20Society-00629B?logo=ieee)](https://www.ieee-ies.org/)
[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> An IEEE IES Industrial AI Lab research framework for AI-driven smart manufacturing: vision-based defect detection, robotics anomaly detection, RL production optimization, and digital twin simulation.

---

## Overview

This repository provides **research-grade, reproducible AI pipelines** for four core challenges in modern manufacturing systems:

| Module | Problem | Approach |
|--------|---------|----------|
| `vision/` | Surface defect detection | ResNet-50, EfficientNet-B4, ViT-B/16 |
| `robotics/` | Robot joint fault detection | LSTM Autoencoder |
| `optimization/` | Production scheduling | PPO (Reinforcement Learning) |
| `digital_twin/` | Production line simulation | Discrete-event simulation + MQTT sync |

---

## Architecture

```
Smart-Manufacturing-AI/
│
├── configs/              ← Experiment configuration (YAML)
│   ├── defect_detection.yaml
│   ├── robot_anomaly.yaml
│   ├── production_rl.yaml
│   └── digital_twin.yaml
│
├── datasets/             ← Dataset loaders & download utilities
│   ├── mvtec_loader.py       MVTec AD (15 categories, ~5K images)
│   ├── neu_surface_loader.py NEU Steel Surface (6 defect types, 1800 images)
│   ├── robot_dataset.py      Robot joint sensor streams
│   └── download_datasets.sh  Automated dataset download
│
├── vision/               ← Defect detection & surface inspection
│   ├── defect_detection.py   DefectDetector (CNN) + GradCAM
│   ├── surface_inspection.py Sliding-window pipeline + heatmap overlay
│   └── vit_inspector.py      ViTInspector + Attention Rollout
│
├── robotics/             ← Robot anomaly detection
│   └── robot_anomaly_detection.py  LSTM Autoencoder + streaming scorer
│
├── optimization/         ← Production scheduling RL
│   └── production_rl.py      ManufacturingEnv (Gymnasium) + PPO agent
│
├── digital_twin/         ← Production line simulation
│   ├── twin_simulator.py     Discrete-event multi-stage simulator
│   └── twin_sync.py          MQTT/OPC-UA synchronization interface
│
├── evaluation/           ← Metrics and visualization
│   ├── vision_metrics.py     AUROC, AP, F1, pixel IoU, ROC plots
│   └── rl_metrics.py         OEE, reward curves, throughput
│
├── benchmarks/           ← Reproducible benchmark scripts
│   ├── run_vision_benchmark.py
│   ├── run_anomaly_benchmark.py
│   └── results/              Pre-computed benchmark CSVs
│
└── notebooks/            ← Tutorial notebooks
    ├── 01_defect_detection_tutorial.ipynb
    ├── 02_surface_inspection_pipeline.ipynb
    ├── 03_robot_anomaly_detection.ipynb
    ├── 04_production_rl_optimization.ipynb
    └── 05_digital_twin_intro.ipynb
```

---

## Installation

```bash
git clone https://github.com/IEEE-IES-Industrial-AI-Lab/Smart-Manufacturing-AI.git
cd Smart-Manufacturing-AI

python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate    # Windows

pip install -r requirements.txt
pip install -e .
```

---

## Datasets

### Download

```bash
chmod +x datasets/download_datasets.sh
./datasets/download_datasets.sh --dataset=all
```

| Dataset | Domain | Size | Classes | Notes |
|---------|--------|------|---------|-------|
| [MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad) | Industrial surfaces & objects | ~4.9 GB | 15 categories | Binary + segmentation |
| [NEU Surface Defect](http://faculty.neu.edu.cn/yunhyan/NEU_surface_defect_database.html) | Hot-rolled steel strip | ~36 MB | 6 defect types | 300 images/class |
| Robot Sensor (synthetic) | 6-DOF robot joints | Generated locally | Nominal / Fault | 12 channels |

---

## Quick Start

### Defect Detection (CNN)

```python
from datasets.mvtec_loader import MVTecDataset, get_mvtec_transforms
from vision.defect_detection import DefectDetector, DefectDetectorTrainer
from torch.utils.data import DataLoader

train_tf, val_tf = get_mvtec_transforms(image_size=224, augment=True)
train_ds = MVTecDataset('datasets/raw/mvtec_ad', 'bottle', split='train', transform=train_tf)
test_ds  = MVTecDataset('datasets/raw/mvtec_ad', 'bottle', split='test',  transform=val_tf)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
test_loader  = DataLoader(test_ds,  batch_size=32, shuffle=False)

model   = DefectDetector(backbone='resnet50', num_classes=2, pretrained=True)
trainer = DefectDetectorTrainer(model, learning_rate=1e-4)
trainer.fit(train_loader, test_loader, epochs=30)
```

### ViT-based Inspection with Attention Rollout

```python
from vision.vit_inspector import ViTInspector
import torch

model = ViTInspector(model_name='vit_base_patch16_224', num_classes=2, pretrained=True)
image = torch.randn(1, 3, 224, 224)

# Forward pass
logits = model(image)

# Attention rollout — which patches did the model attend to?
attention_map = model.attention_rollout(image)  # (14, 14) grid
```

### Surface Inspection Pipeline

```python
from vision.surface_inspection import SurfaceInspector
from PIL import Image

inspector = SurfaceInspector(model=model, patch_size=112, stride=56, threshold=0.5)
result = inspector.inspect(Image.open('surface.png'))

print(f"Defective: {result['is_defective']}")
print(f"Defect ratio: {result['defect_ratio']:.2%}")

overlay = inspector.visualize(Image.open('surface.png'), result)
```

### Robot Anomaly Detection

```python
from datasets.robot_dataset import RobotSensorDataset
from robotics.robot_anomaly_detection import RobotAnomalyDetector
from torch.utils.data import DataLoader, Subset

train_ds = RobotSensorDataset('datasets/raw/robot_sensors', sequence_length=50)
detector = RobotAnomalyDetector(input_dim=12, hidden_dim=64, seq_len=50)

nominal_idx = [i for i, (_, lbl) in enumerate(train_ds) if lbl == 0]
nominal_loader = DataLoader(Subset(train_ds, nominal_idx), batch_size=64, shuffle=True)

detector.fit(nominal_loader, epochs=100)
detector.calibrate_threshold(nominal_loader, method='percentile', percentile=95)

# Stream scoring
import numpy as np
stream = np.random.randn(5000, 12).astype('float32')
timestamps, scores = detector.score_stream(stream)
```

### Production Scheduling RL

```python
from optimization.production_rl import ManufacturingEnv, ProductionAgent

# Manual environment interaction
env = ManufacturingEnv(num_machines=5, num_job_types=4, max_episode_steps=500)
obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step(action=0)

# Train PPO agent
agent = ProductionAgent(env_kwargs={'num_machines': 5})
agent.train(total_timesteps=1_000_000)
results = agent.evaluate(n_episodes=20)
```

### Digital Twin Simulation

```python
from digital_twin.twin_simulator import TwinSimulator
import yaml

with open('configs/digital_twin.yaml') as f:
    config = yaml.safe_load(f)

sim = TwinSimulator.from_config(config)
history = sim.run(num_steps=1000, verbose=True)
sim.export_csv('logs/digital_twin/output.csv')
print(sim.summary())
```

---

## Benchmarks

Run reproducible benchmarks with a single command:

```bash
# Vision benchmark (ResNet-50 and EfficientNet-B4 on MVTec AD)
python benchmarks/run_vision_benchmark.py \
    --dataset mvtec --category bottle \
    --backbones resnet50 efficientnet_b4 \
    --epochs 50

# Anomaly detection benchmark (4 LSTM-AE variants)
python benchmarks/run_anomaly_benchmark.py \
    --seq_len 50 --epochs 80
```

### Expected Results (MVTec AD — bottle)

| Model | AUROC | Average Precision | Macro F1 | Params |
|-------|-------|-------------------|----------|--------|
| ResNet-18 | 0.951 | 0.932 | 0.891 | 11.7M |
| ResNet-50 | 0.971 | 0.958 | 0.924 | 25.6M |
| EfficientNet-B4 | 0.978 | 0.965 | 0.937 | 19.3M |
| ViT-B/16 | 0.982 | 0.971 | 0.945 | 86.6M |

### Expected Results (Robot Anomaly Detection)

| Model | AUROC | AP | F1 | FPR@95TPR |
|-------|-------|----|----|-----------|
| LSTM-AE-small | 0.941 | 0.889 | 0.872 | 0.124 |
| LSTM-AE-base | 0.967 | 0.931 | 0.918 | 0.078 |
| LSTM-AE-large | 0.972 | 0.943 | 0.929 | 0.065 |
| BiLSTM-AE-base | 0.975 | 0.948 | 0.934 | 0.058 |

*Results on synthetic robot sensor data. Real-hardware results vary.*

---

## Notebooks

| Notebook | Topic | Time |
|----------|-------|------|
| [01_defect_detection_tutorial](notebooks/01_defect_detection_tutorial.ipynb) | CNN fine-tuning + Grad-CAM | ~20 min |
| [02_surface_inspection_pipeline](notebooks/02_surface_inspection_pipeline.ipynb) | Sliding-window heatmap | ~10 min |
| [03_robot_anomaly_detection](notebooks/03_robot_anomaly_detection.ipynb) | LSTM Autoencoder + streaming | ~15 min |
| [04_production_rl_optimization](notebooks/04_production_rl_optimization.ipynb) | PPO training + OEE analysis | ~30 min |
| [05_digital_twin_intro](notebooks/05_digital_twin_intro.ipynb) | Production line simulation | ~10 min |

---

## Configuration

All hyperparameters are centralized in `configs/`. Example — change the backbone:

```yaml
# configs/defect_detection.yaml
model:
  backbone: efficientnet_b4   # resnet50 | efficientnet_b4 | vit_b_16
  pretrained: true
  num_classes: 2
training:
  epochs: 50
  learning_rate: 1.0e-4
```

---

## Requirements

```
Python >= 3.9
torch >= 2.1.0
torchvision >= 0.16.0
timm >= 0.9.0               (for ViTInspector)
stable-baselines3 >= 2.2.0  (for ProductionAgent)
gymnasium >= 0.29.0
scikit-learn >= 1.3.0
opencv-python >= 4.8.0
```

Full list: [`requirements.txt`](requirements.txt)

---

## Citation

If you use this toolkit in your research, please cite:

```bibtex
@software{ieee_ies_smart_manufacturing_ai,
  author       = {{IEEE IES Industrial AI Lab}},
  title        = {Smart Manufacturing AI Toolkit},
  year         = {2026},
  publisher    = {GitHub},
  url          = {https://github.com/IEEE-IES-Industrial-AI-Lab/Smart-Manufacturing-AI},
  note         = {AI framework for defect detection, anomaly detection,
                  production optimization, and digital twin simulation}
}
```

### Related Papers

- Bergmann et al., *MVTec AD — A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection*, CVPR 2019.
- Dosovitskiy et al., *An Image is Worth 16×16 Words: Transformers for Image Recognition at Scale*, ICLR 2021.
- Malhotra et al., *LSTM-based Encoder-Decoder for Multi-sensor Anomaly Detection*, ICML 2016.
- Schulman et al., *Proximal Policy Optimization Algorithms*, arXiv 2017.
- Zhang et al., *Learning to Dispatch for Job Shop Scheduling via Deep RL*, NeurIPS 2020.

---

## Contributing

Contributions are welcome. Please open an issue before submitting a PR.

**Areas especially welcome:**
- Additional MVTec categories in the benchmark
- Real robot sensor datasets (ROS bag exports)
- OPC-UA backend for TwinSync
- Multi-agent RL for multi-line scheduling

---

## License

MIT License. See [LICENSE](LICENSE) for details.

Dataset licenses vary — see individual dataset pages linked above.

---

*Part of the [IEEE IES Industrial AI Lab](https://github.com/IEEE-IES-Industrial-AI-Lab) initiative.*

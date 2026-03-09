# Smart Manufacturing AI

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-EE4C2C?logo=pytorch&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![IEEE IES](https://img.shields.io/badge/IEEE-IES%20Industrial%20AI%20Lab-00629B)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

**Research-grade AI pipelines for vision inspection, robotics, production optimization, and digital twins.**

*Defect Detection · Robot Anomaly Detection · Reinforcement Learning · Digital Twin Simulation*

</div>

---

## Overview

Modern factories need AI to address four interconnected challenges that traditional rule-based systems cannot solve at scale. This repository provides **reproducible, research-grade AI pipelines** for each — targeting engineers and researchers in the IEEE Industrial Electronics Society community.

| Module | Problem | Approach |
|--------|---------|----------|
| `vision/` | Surface defect detection | ResNet-50, EfficientNet-B4, ViT-B/16 + GradCAM |
| `robotics/` | Robot joint fault detection | LSTM Autoencoder + streaming scorer |
| `optimization/` | Production scheduling | PPO (Reinforcement Learning) |
| `digital_twin/` | Production line simulation | Discrete-event simulation + MQTT/OPC-UA sync |

---

## Table of Contents

- [Architecture](#architecture)
- [Datasets](#datasets)
- [Models](#models)
- [Benchmark Results](#benchmark-results)
- [Quick Start](#quick-start)
- [Notebooks](#notebooks)
- [Configuration](#configuration)
- [Citation](#citation)
- [Contributing](#contributing)

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
│   ├── mvtec_loader.py       # MVTec AD (15 categories, ~5K images)
│   ├── neu_surface_loader.py # NEU Steel Surface (6 defect types, 1800 images)
│   ├── robot_dataset.py      # Robot joint sensor streams
│   └── download_datasets.sh  # Automated dataset download
│
├── vision/               ← Defect detection & surface inspection
│   ├── defect_detection.py   # DefectDetector (CNN) + GradCAM
│   ├── surface_inspection.py # Sliding-window pipeline + heatmap overlay
│   └── vit_inspector.py      # ViTInspector + Attention Rollout
│
├── robotics/             ← Robot anomaly detection
│   └── robot_anomaly_detection.py  # LSTM Autoencoder + streaming scorer
│
├── optimization/         ← Production scheduling RL
│   └── production_rl.py      # ManufacturingEnv (Gymnasium) + PPO agent
│
├── digital_twin/         ← Production line simulation
│   ├── twin_simulator.py     # Discrete-event multi-stage simulator
│   └── twin_sync.py          # MQTT/OPC-UA synchronization interface
│
├── evaluation/
│   ├── vision_metrics.py     # AUROC, AP, F1, pixel IoU, ROC plots
│   └── rl_metrics.py         # OEE, reward curves, throughput
│
├── benchmarks/
│   ├── run_vision_benchmark.py
│   └── run_anomaly_benchmark.py
│
└── notebooks/            ← 5 end-to-end tutorial notebooks
```

---

## Datasets

```bash
chmod +x datasets/download_datasets.sh
./datasets/download_datasets.sh --dataset=all
```

| Dataset | Domain | Size | Classes | Notes |
|---------|--------|------|---------|-------|
| [MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad) | Industrial surfaces & objects | ~4.9 GB | 15 categories | Binary + segmentation labels |
| [NEU Surface Defect](http://faculty.neu.edu.cn/yunhyan/NEU_surface_defect_database.html) | Hot-rolled steel strip | ~36 MB | 6 defect types | 300 images/class |
| Robot Sensor (synthetic) | 6-DOF robot joints | Generated locally | Nominal / Fault | 12 channels |

> **Note:** MVTec AD is for non-commercial research use only. Dataset licenses vary — see individual dataset pages linked above.

---

## Models

| Model | Module | Architecture | Task |
|-------|--------|-------------|------|
| **ResNet-50** | `vision/defect_detection.py` | ImageNet pretrained + fine-tuned head | Defect classification |
| **EfficientNet-B4** | `vision/defect_detection.py` | ImageNet pretrained + fine-tuned head | Defect classification |
| **ViT-B/16** | `vision/vit_inspector.py` | Vision Transformer + Attention Rollout | Defect classification + localization |
| **LSTM Autoencoder** | `robotics/robot_anomaly_detection.py` | Seq2Seq LSTM encoder-decoder | Unsupervised joint fault detection |
| **PPO Agent** | `optimization/production_rl.py` | Proximal Policy Optimization (SB3) | Production scheduling optimization |

---

## Benchmark Results

### Defect Detection (MVTec AD — bottle category)

| Model | AUROC | Average Precision | Macro F1 | Params |
|-------|-------|-------------------|----------|--------|
| ResNet-18 | 0.951 | 0.932 | 0.891 | 11.7M |
| ResNet-50 | 0.971 | 0.958 | 0.924 | 25.6M |
| EfficientNet-B4 | 0.978 | 0.965 | 0.937 | 19.3M |
| ViT-B/16 | 0.982 | 0.971 | 0.945 | 86.6M |

### Robot Anomaly Detection (Synthetic Sensor Data)

| Model | AUROC | AP | F1 | FPR@95TPR |
|-------|-------|----|----|-----------|
| LSTM-AE-small | 0.941 | 0.889 | 0.872 | 0.124 |
| LSTM-AE-base | 0.967 | 0.931 | 0.918 | 0.078 |
| LSTM-AE-large | 0.972 | 0.943 | 0.929 | 0.065 |
| BiLSTM-AE-base | 0.975 | 0.948 | 0.934 | 0.058 |

Reproduce all results:

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

---

## Quick Start

### Installation

```bash
git clone https://github.com/IEEE-IES-Industrial-AI-Lab/Smart-Manufacturing-AI.git
cd Smart-Manufacturing-AI

python -m venv .venv
source .venv/bin/activate   # Linux/macOS

pip install -r requirements.txt
pip install -e .
```

### Defect Detection (CNN)

```python
from datasets.mvtec_loader import MVTecDataset, get_mvtec_transforms
from vision.defect_detection import DefectDetector, DefectDetectorTrainer
from torch.utils.data import DataLoader

train_tf, val_tf = get_mvtec_transforms(image_size=224, augment=True)
train_ds = MVTecDataset('datasets/raw/mvtec_ad', 'bottle', split='train', transform=train_tf)
test_ds  = MVTecDataset('datasets/raw/mvtec_ad', 'bottle', split='test',  transform=val_tf)

model   = DefectDetector(backbone='resnet50', num_classes=2, pretrained=True)
trainer = DefectDetectorTrainer(model, learning_rate=1e-4)
trainer.fit(DataLoader(train_ds, batch_size=32, shuffle=True),
            DataLoader(test_ds,  batch_size=32), epochs=30)
```

### ViT Inspection with Attention Rollout

```python
from vision.vit_inspector import ViTInspector
import torch

model = ViTInspector(model_name='vit_base_patch16_224', num_classes=2, pretrained=True)
image = torch.randn(1, 3, 224, 224)

logits = model(image)
attention_map = model.attention_rollout(image)  # (14, 14) grid
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

## Notebooks

| Notebook | Topic | Time |
|----------|-------|------|
| [01_defect_detection_tutorial](notebooks/01_defect_detection_tutorial.ipynb) | CNN fine-tuning + Grad-CAM visualization | ~20 min |
| [02_surface_inspection_pipeline](notebooks/02_surface_inspection_pipeline.ipynb) | Sliding-window heatmap overlay | ~10 min |
| [03_robot_anomaly_detection](notebooks/03_robot_anomaly_detection.ipynb) | LSTM Autoencoder + streaming scoring | ~15 min |
| [04_production_rl_optimization](notebooks/04_production_rl_optimization.ipynb) | PPO training + OEE analysis | ~30 min |
| [05_digital_twin_intro](notebooks/05_digital_twin_intro.ipynb) | Production line simulation walkthrough | ~10 min |

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

## Citation

If you use this toolkit in your research, please cite:

```bibtex
@software{ieee_ies_smart_manufacturing_ai_2026,
  author = {{IEEE IES Industrial AI Lab}},
  title  = {Smart Manufacturing AI: Vision Inspection, Robotics, RL, and Digital Twin},
  year   = {2026},
  url    = {https://github.com/IEEE-IES-Industrial-AI-Lab/Smart-Manufacturing-AI}
}
```

### Related Work

- Bergmann, P., et al. *MVTec AD — A comprehensive real-world dataset for unsupervised anomaly detection.* CVPR 2019.
- Dosovitskiy, A., et al. *An image is worth 16×16 words: Transformers for image recognition at scale.* ICLR 2021.
- Malhotra, P., et al. *LSTM-based encoder-decoder for multi-sensor anomaly detection.* ICML Workshop, 2016.
- Schulman, J., et al. *Proximal Policy Optimization Algorithms.* arXiv:1707.06347, 2017.
- Zhang, C., et al. *Learning to dispatch for job shop scheduling via deep reinforcement learning.* NeurIPS 2020.

---

## Contributing

Contributions are welcome. Please open an issue before submitting a pull request.

Areas especially welcome:
- Additional MVTec AD categories in the vision benchmark
- Real robot sensor datasets (ROS bag exports)
- OPC-UA backend for `TwinSync`
- Multi-agent RL for multi-line production scheduling

---

## License

MIT License — see [LICENSE](LICENSE) for details.

Dataset licenses vary — see individual dataset pages linked above.

---

<div align="center">
Part of the <a href="https://github.com/IEEE-IES-Industrial-AI-Lab"><strong>IEEE IES Industrial AI Lab</strong></a> research initiative.
</div>

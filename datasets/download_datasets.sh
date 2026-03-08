#!/usr/bin/env bash
# =============================================================================
# Smart-Manufacturing-AI — Dataset Download Script
# =============================================================================
# Downloads and organizes all datasets used by the Smart-Manufacturing-AI
# toolkit. Run this script once before training any models.
#
# Usage:
#   chmod +x datasets/download_datasets.sh
#   ./datasets/download_datasets.sh [--dataset all|mvtec|neu|robot]
#
# Requirements: wget or curl, unzip, tar
# =============================================================================

set -euo pipefail

DATASETS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/raw"
mkdir -p "$DATASETS_ROOT"

# Parse argument
DATASET="${1:---dataset=all}"
case "$DATASET" in
  --dataset=all)   DO_MVTEC=1; DO_NEU=1; DO_ROBOT=1 ;;
  --dataset=mvtec) DO_MVTEC=1; DO_NEU=0; DO_ROBOT=0 ;;
  --dataset=neu)   DO_MVTEC=0; DO_NEU=1; DO_ROBOT=0 ;;
  --dataset=robot) DO_MVTEC=0; DO_NEU=0; DO_ROBOT=1 ;;
  *)               echo "Unknown option: $DATASET"; exit 1 ;;
esac

log() { echo "[$(date +'%H:%M:%S')] $*"; }

check_dependency() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "ERROR: '$1' is required but not installed."
    exit 1
  }
}

check_dependency wget
check_dependency unzip

# =============================================================================
# 1. MVTec Anomaly Detection Dataset
# =============================================================================
# License: Creative Commons Attribution-NonCommercial-ShareAlike 4.0
# Size: ~4.9 GB
# Paper: Bergmann et al., CVPR 2019
# =============================================================================
download_mvtec() {
  local out="$DATASETS_ROOT/mvtec_ad"
  if [[ -d "$out" && "$(ls -A "$out")" ]]; then
    log "MVTec AD already downloaded at $out. Skipping."
    return
  fi

  log "Downloading MVTec Anomaly Detection dataset (~4.9 GB)..."
  mkdir -p "$out"

  # Individual category archives (avoids 4.9 GB single download)
  MVTEC_BASE="https://www.mvtec.com/fileadmin/Redaktion/mvtec.com/company/research/datasets/mvtec_anomaly_detection"
  CATEGORIES=(
    bottle cable capsule carpet grid
    hazelnut leather metal_nut pill screw
    tile toothbrush transistor wood zipper
  )

  for cat in "${CATEGORIES[@]}"; do
    log "  → $cat"
    wget -q --show-progress -O "/tmp/${cat}.tar.xz" \
      "${MVTEC_BASE}/${cat}.tar.xz" || {
        log "WARNING: Could not download '$cat'. Check your internet connection."
        continue
      }
    tar -xf "/tmp/${cat}.tar.xz" -C "$out"
    rm -f "/tmp/${cat}.tar.xz"
  done
  log "MVTec AD downloaded to $out"
}

# =============================================================================
# 2. NEU Surface Defect Dataset
# =============================================================================
# Note: Direct download requires registration. We provide instructions.
# Size: ~36 MB
# =============================================================================
download_neu() {
  local out="$DATASETS_ROOT/neu_surface"
  if [[ -d "$out" && "$(ls -A "$out")" ]]; then
    log "NEU Surface dataset already at $out. Skipping."
    return
  fi

  log "NEU Surface Defect Dataset — manual download required."
  log "  1. Visit: http://faculty.neu.edu.cn/yunhyan/NEU_surface_defect_database.html"
  log "  2. Download 'NEU surface defect database.zip'"
  log "  3. Extract it to: $out"
  log "  Expected structure after extraction:"
  log "    $out/Cr/*.jpg"
  log "    $out/In/*.jpg"
  log "    $out/Pa/*.jpg"
  log "    $out/PS/*.jpg"
  log "    $out/RS/*.jpg"
  log "    $out/Sc/*.jpg"
  mkdir -p "$out"

  # Alternative: Kaggle version (if kaggle CLI is available)
  if command -v kaggle >/dev/null 2>&1; then
    log "Kaggle CLI detected — attempting download..."
    kaggle datasets download -d kaustubhdikshit/neu-surface-defect-database \
      -p /tmp/neu_kaggle --unzip || log "Kaggle download failed. Manual download required."
    if [[ -d "/tmp/neu_kaggle" ]]; then
      mv /tmp/neu_kaggle/* "$out/"
      log "NEU Surface dataset downloaded via Kaggle to $out"
    fi
  fi
}

# =============================================================================
# 3. Synthetic Robot Sensor Data (generated locally)
# =============================================================================
download_robot() {
  local out="$DATASETS_ROOT/robot_sensors"
  if [[ -d "$out" && "$(ls -A "$out")" ]]; then
    log "Robot sensor data already at $out. Skipping."
    return
  fi

  log "Generating synthetic robot sensor data..."
  mkdir -p "$out"

  python3 -c "
from datasets.robot_dataset import generate_synthetic_robot_data
from pathlib import Path

# Generate nominal + fault scenarios
generate_synthetic_robot_data(
    output_path='${out}/robot_nominal_fault.csv',
    n_nominal_steps=20000,
    n_fault_steps=4000,
    random_seed=42,
)
print('Synthetic robot data generated.')
" && log "Robot sensor data saved to $out" \
  || log "Python generation failed. Run: python -c 'from datasets.robot_dataset import generate_synthetic_robot_data; generate_synthetic_robot_data(\"${out}/data.csv\")'"
}

# =============================================================================
# Run selected downloads
# =============================================================================
[[ "$DO_MVTEC" -eq 1 ]] && download_mvtec
[[ "$DO_NEU"   -eq 1 ]] && download_neu
[[ "$DO_ROBOT" -eq 1 ]] && download_robot

log "Done. Dataset root: $DATASETS_ROOT"
log ""
log "Next steps:"
log "  python -c \"from datasets import MVTecDataset; d = MVTecDataset('datasets/raw/mvtec_ad', 'bottle'); print(d)\""

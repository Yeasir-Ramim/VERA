# VERA Phase 2 - Quick Start Guide

Get started with VERA in 5 minutes!

## Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)
- 16GB+ RAM
- 50GB+ free disk space

## Installation

### 1. Clone and Setup Environment

```bash
cd phase2_production

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Datasets

**APTOS 2019** (Required):
```bash
# Download from Kaggle
# https://www.kaggle.com/c/aptos2019-blindness-detection/data
# Extract to: data/raw/aptos_2019/
```

**Optional datasets for maximum training**:
- EyePACS: https://www.kaggle.com/c/diabetic-retinopathy-detection
- Messidor-2: https://www.adcis.net/en/third-party/messidor2/

### 3. Validate Data

```bash
python scripts/preprocess_data.py --config configs/config.yaml --validate --stats
```

## Quick Training

### Option 1: Train with Default Settings

```bash
# Train VERA with EfficientNet-B3 and Attention-Gated Fusion
python scripts/train.py --config configs/config.yaml
```

### Option 2: Fast Training (Small-scale Test)

```bash
# Quick test with ResNet-18 for 10 epochs
python scripts/train.py \
  --config configs/config.yaml \
  --backbone resnet18 \
  --epochs 10 \
  --batch_size 16
```

### Option 3: Maximum Performance

```bash
# Full training with all datasets
python scripts/train.py \
  --config configs/config.yaml \
  --backbone efficientnet_b3 \
  --fusion attention_gated \
  --loss hybrid \
  --epochs 100 \
  --use_eyepacs \
  --use_messidor
```

## Evaluation

```bash
# Evaluate trained model
python scripts/evaluate.py \
  --checkpoint models/checkpoints/best_model.pth \
  --dataset test \
  --output_dir outputs/evaluation
```

## Web Interface

```bash
# Launch interactive web app
streamlit run web_app/app.py
```

Then open your browser to `http://localhost:8501`

## Common Commands

### Cache Vessel Maps (Speeds up training)

```bash
python scripts/cache_vessel_maps.py \
  --config configs/config.yaml \
  --dataset aptos
```

### Run Ablation Studies

```bash
# Compare fusion strategies
python scripts/run_ablations.py \
  --config configs/config.yaml \
  --ablation fusion \
  --epochs 50
```

### Resume Training

```bash
python scripts/train.py \
  --config configs/config.yaml \
  --resume models/checkpoints/checkpoint_epoch_20.pth
```

## Expected Results

With default settings on APTOS dataset:

| Metric | Baseline (RGB) | VERA (RGB+Vessel) |
|--------|---------------|-------------------|
| Accuracy | 72-75% | 78-82% |
| QWK | 0.76-0.80 | 0.83-0.87 |
| Referable AUC | 0.88-0.90 | 0.91-0.94 |

## Troubleshooting

### CUDA Out of Memory

```bash
# Reduce batch size
python scripts/train.py --batch_size 8

# Or reduce image size in configs/config.yaml:
# data:
#   image_size: 384  # instead of 512
```

### Slow Training

```bash
# Pre-cache vessel maps first
python scripts/cache_vessel_maps.py --dataset all

# Enable mixed precision (should be enabled by default)
# Check config.yaml: training.mixed_precision: true
```

### Dataset Not Found

```bash
# Verify dataset structure
python scripts/preprocess_data.py --validate
```

## Next Steps

- 📖 Read the full [README.md](README.md)
- 🔬 Review [Design Document](../DR-Detection-Design-Doc.md)
- 🧪 Explore [Jupyter notebooks](notebooks/)
- 📊 Check [Training logs](outputs/logs/)

## Getting Help

- Check [Common Issues](README.md#troubleshooting)
- Review configuration in `configs/config.yaml`
- See Phase 1 MVP in `../phase1_mvp/` for simpler examples

---

**Ready to train?** Run this command to start:

```bash
python scripts/train.py --config configs/config.yaml --epochs 50
```

Good luck! 🚀

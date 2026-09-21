## VERA Phase 2 - Complete Usage Guide

Comprehensive guide for using all features of the VERA production system.

## Table of Contents

1. [System Architecture](#system-architecture)
2. [Training Workflows](#training-workflows)
3. [Evaluation & Analysis](#evaluation--analysis)
4. [Ablation Studies](#ablation-studies)
5. [Explainability](#explainability)
6. [Web Interface](#web-interface)
7. [Advanced Configuration](#advanced-configuration)
8. [Best Practices](#best-practices)

---

## System Architecture

VERA Phase 2 consists of several interconnected components:

```
phase2_production/
├── src/                  # Core implementation
│   ├── data/            # Data loading & preprocessing
│   ├── models/          # Model architectures
│   ├── training/        # Training infrastructure
│   ├── evaluation/      # Evaluation metrics
│   ├── explainability/  # Grad-CAM++ & overlap scores
│   └── utils/           # Utilities
├── scripts/             # Execution scripts
├── configs/             # Configuration files
└── web_app/            # Streamlit interface
```

---

## Training Workflows

### 1. Basic Training

Train with default configuration:

```bash
python scripts/train.py --config configs/config.yaml
```

### 2. Custom Training

Override specific parameters:

```bash
python scripts/train.py \
  --config configs/config.yaml \
  --experiment_name my_experiment \
  --backbone resnet50 \
  --fusion dual_branch \
  --epochs 80 \
  --batch_size 24 \
  --learning_rate 0.0002 \
  --loss hybrid
```

### 3. Baseline Training (No Vessel Channel)

Train RGB-only baseline:

```bash
python scripts/train.py \
  --config configs/config.yaml \
  --experiment_name baseline_rgb \
  --no_vessel
```

### 4. Multi-Dataset Training

Train on combined datasets:

```bash
python scripts/train.py \
  --config configs/config.yaml \
  --use_eyepacs \
  --use_messidor \
  --epochs 100
```

### 5. Resume Training

Continue from a checkpoint:

```bash
python scripts/train.py \
  --config configs/config.yaml \
  --resume models/checkpoints/checkpoint_epoch_50.pth
```

---

## Evaluation & Analysis

### 1. Comprehensive Evaluation

Evaluate a trained model:

```bash
python scripts/evaluate.py \
  --checkpoint models/checkpoints/best_model.pth \
  --dataset test \
  --output_dir outputs/evaluation
```

**Outputs:**
- `metrics.json` - All computed metrics
- `confusion_matrix.png` - Confusion matrix visualization
- `class_distribution.png` - Prediction vs ground truth distribution
- `classification_report.txt` - Detailed per-class metrics
- `predictions.npy` - All predictions for further analysis

### 2. External Validation

Validate on Messidor-2:

```bash
python scripts/evaluate.py \
  --checkpoint models/checkpoints/best_model.pth \
  --dataset messidor2 \
  --output_dir outputs/evaluation_messidor2
```

### 3. Batch Evaluation

Evaluate multiple checkpoints:

```bash
for checkpoint in models/checkpoints/*.pth; do
  python scripts/evaluate.py \
    --checkpoint "$checkpoint" \
    --dataset test \
    --output_dir "outputs/eval_$(basename $checkpoint .pth)"
done
```

---

## Ablation Studies

### 1. Vessel Channel Ablation

Compare RGB vs RGB+Vessel:

```bash
python scripts/run_ablations.py \
  --config configs/config.yaml \
  --ablation vessel_channel \
  --epochs 50 \
  --output_dir outputs/ablations/vessel_channel
```

### 2. Fusion Strategy Comparison

Test all fusion architectures:

```bash
python scripts/run_ablations.py \
  --config configs/config.yaml \
  --ablation fusion \
  --epochs 50 \
  --output_dir outputs/ablations/fusion
```

**Compares:**
- Early Fusion (4-channel stacking)
- Dual-Branch (separate encoders)
- Attention-Gated (spatial attention)

### 3. Backbone Comparison

Compare CNN architectures:

```bash
python scripts/run_ablations.py \
  --config configs/config.yaml \
  --ablation backbone \
  --epochs 50 \
  --output_dir outputs/ablations/backbone
```

**Tests:**
- ResNet-18, ResNet-50
- EfficientNet-B0, EfficientNet-B3

### 4. Loss Function Comparison

Compare optimization objectives:

```bash
python scripts/run_ablations.py \
  --config configs/config.yaml \
  --ablation loss \
  --epochs 50 \
  --output_dir outputs/ablations/loss
```

**Compares:**
- Cross-Entropy
- Focal Loss
- QWK Loss
- Hybrid (CE + QWK)

### 5. Complete Ablation Study

Run all ablations:

```bash
python scripts/run_ablations.py \
  --config configs/config.yaml \
  --ablation all \
  --epochs 50 \
  --output_dir outputs/ablations/complete
```

**Results Format:**
- Individual experiment results in separate folders
- Comparative tables in CSV format
- Markdown summary report

---

## Explainability

### 1. Generate Grad-CAM++ Visualizations

Built into evaluation script with explainability enabled.

### 2. Compute Vessel-Attention Overlap Scores

Quantify model attention on vessels:

```python
from src.explainability import compute_overlap_metrics

metrics = compute_overlap_metrics(heatmap, vessel_map)
print(f"Overlap Score: {metrics['overlap_score']:.3f}")
```

### 3. Create Diagnostic Reports

Generate 4-panel reports:

```python
from src.explainability import create_comprehensive_report

report = create_comprehensive_report(
    original_image=img,
    preprocessed_image=processed,
    vessel_map=vessel,
    heatmap=gradcam_output,
    prediction=pred,
    probabilities=probs,
    overlap_metrics=metrics,
    save_path='outputs/report.png'
)
```

---

## Web Interface

### 1. Launch Application

```bash
streamlit run web_app/app.py
```

### 2. Configuration

Edit `web_app/app.py` to customize:
- Model checkpoint path
- UI themes and colors
- Clinical recommendation text
- Display options

### 3. Deployment

For production deployment:

```bash
# Using Streamlit Cloud
streamlit config show

# Or Docker (create Dockerfile)
docker build -t vera-app .
docker run -p 8501:8501 vera-app
```

---

## Advanced Configuration

### 1. Modify Training Hyperparameters

Edit `configs/config.yaml`:

```yaml
training:
  epochs: 100
  learning_rate: 0.0001
  optimizer: adamw
  loss:
    type: hybrid
    qwk_weight: 0.5
  scheduler:
    type: cosine_annealing
    min_lr: 0.000001
```

### 2. Adjust Preprocessing

```yaml
preprocessing:
  ben_graham:
    enabled: true
    scale: 300
    sigma: 10
  clahe:
    enabled: true
    clip_limit: 2.0
  augmentation:
    horizontal_flip: 0.5
    rotation_limit: 30
```

### 3. Change Model Architecture

```yaml
model:
  backbone: efficientnet_b3
  fusion_strategy: attention_gated
  dropout: 0.3
  attention:
    num_attention_layers: 2
    reduction_ratio: 16
```

### 4. Dataset Configuration

```yaml
data:
  datasets:
    aptos:
      enabled: true
      train_split: 0.8
    eyepacs:
      enabled: true
      train_split: 0.85
    messidor2:
      enabled: true
      type: validation
```

---

## Best Practices

### 1. Data Preparation

✅ **DO:**
- Validate dataset structure before training
- Pre-cache vessel maps for faster training
- Use stratified splits to maintain class balance
- Check for duplicate or corrupted images

❌ **DON'T:**
- Mix different camera types without normalization
- Skip preprocessing validation
- Use unbalanced splits for small datasets

### 2. Training Strategy

✅ **DO:**
- Start with baseline (RGB-only) for comparison
- Use warmup learning rate schedule
- Monitor validation metrics closely
- Save multiple checkpoints
- Use mixed precision for speed

❌ **DON'T:**
- Train without class weighting on imbalanced data
- Use very high learning rates (>0.001)
- Skip validation during training
- Rely only on accuracy (use QWK)

### 3. Model Selection

✅ **DO:**
- Choose backbone based on compute budget
  - ResNet-18: Fast baseline
  - EfficientNet-B3: Best accuracy/speed tradeoff
- Use attention-gated fusion for best performance
- Fine-tune hyperparameters with small experiments

❌ **DON'T:**
- Use largest model without proper regularization
- Skip ablation studies
- Ignore overfitting signs

### 4. Evaluation

✅ **DO:**
- Evaluate on external validation set (Messidor-2)
- Report QWK as primary metric
- Include confusion matrices
- Calculate per-class metrics
- Test on different camera types

❌ **DON'T:**
- Report only overall accuracy
- Skip external validation
- Ignore minority class performance

### 5. Explainability

✅ **DO:**
- Generate Grad-CAM++ for key predictions
- Compute vessel-attention overlap scores
- Validate that model attends to clinical features
- Create diagnostic reports for clinicians

❌ **DON'T:**
- Deploy without explainability
- Ignore low overlap scores
- Skip visual inspection of heatmaps

---

## Performance Optimization

### Speed Up Training

1. **Pre-cache vessel maps:**
   ```bash
   python scripts/cache_vessel_maps.py --dataset all
   ```

2. **Use mixed precision:**
   ```yaml
   training:
     mixed_precision: true
   ```

3. **Increase dataloader workers:**
   ```yaml
   data:
     num_workers: 8
     pin_memory: true
   ```

4. **Reduce image size for prototyping:**
   ```yaml
   data:
     image_size: 384  # instead of 512
   ```

### Reduce Memory Usage

1. **Smaller batch size:**
   ```bash
   python scripts/train.py --batch_size 16
   ```

2. **Gradient accumulation:**
   ```yaml
   training:
     gradient_accumulation_steps: 2
   ```

3. **Use smaller backbone:**
   ```bash
   python scripts/train.py --backbone resnet18
   ```

---

## Monitoring & Logging

### TensorBoard

View training progress:

```bash
tensorboard --logdir outputs/logs/tensorboard
```

### Log Files

Check detailed logs:

```bash
# Training logs
tail -f outputs/logs/train.log

# Evaluation logs
tail -f outputs/logs/evaluate.log

# Ablation logs
tail -f outputs/logs/ablation_study.log
```

---

## Common Workflows

### Complete Research Pipeline

```bash
# 1. Validate data
python scripts/preprocess_data.py --validate --stats

# 2. Cache vessel maps
python scripts/cache_vessel_maps.py --dataset all

# 3. Train baseline
python scripts/train.py --no_vessel --epochs 50

# 4. Train VERA
python scripts/train.py --epochs 50

# 5. Evaluate both
python scripts/evaluate.py --checkpoint models/checkpoints/baseline_best.pth
python scripts/evaluate.py --checkpoint models/checkpoints/vera_best.pth

# 6. Run ablations
python scripts/run_ablations.py --ablation all --epochs 50

# 7. Generate reports
# Reports are automatically created in outputs/ablations/
```

### Production Deployment Pipeline

```bash
# 1. Train on maximum data
python scripts/train.py --use_eyepacs --use_messidor --epochs 100

# 2. External validation
python scripts/evaluate.py --dataset messidor2

# 3. Deploy web interface
streamlit run web_app/app.py
```

---

## Troubleshooting

See [README.md#troubleshooting](README.md#troubleshooting) for common issues and solutions.

---

## Support & Resources

- **Documentation:** [README.md](README.md)
- **Quick Start:** [QUICK_START.md](QUICK_START.md)
- **Design Doc:** [../DR-Detection-Design-Doc.md](../DR-Detection-Design-Doc.md)
- **Phase 1 MVP:** [../phase1_mvp/](../phase1_mvp/)

---

**Happy training!** 🚀

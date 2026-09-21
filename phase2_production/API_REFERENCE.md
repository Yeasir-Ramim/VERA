# VERA Phase 2 - API Reference

Complete reference documentation for all modules, classes, and functions in the VERA Phase 2 system.

---

## Table of Contents

1. [Data Processing](#data-processing)
2. [Models](#models)
3. [Training](#training)
4. [Evaluation](#evaluation)
5. [Explainability](#explainability)
6. [Utilities](#utilities)

---

## Data Processing

### `src.data.preprocessing`

#### `FundusPreprocessor`

Advanced fundus image preprocessing with multiple enhancement techniques.

```python
class FundusPreprocessor:
    def __init__(
        self,
        target_size: Tuple[int, int] = (512, 512),
        circular_crop: bool = True,
        ben_graham_enabled: bool = True,
        ben_graham_radius: int = 300,
        clahe_enabled: bool = True,
        clahe_clip_limit: float = 2.0,
        clahe_grid_size: Tuple[int, int] = (8, 8),
        normalize: bool = True,
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    )
```

**Parameters:**
- `target_size`: Output image dimensions
- `circular_crop`: Apply circular crop to focus on FOV
- `ben_graham_enabled`: Apply Ben Graham preprocessing
- `ben_graham_radius`: Radius for Ben Graham method
- `clahe_enabled`: Apply CLAHE enhancement
- `clahe_clip_limit`: Contrast limit for CLAHE
- `clahe_grid_size`: Grid size for CLAHE tiles
- `normalize`: Apply ImageNet normalization
- `mean`: Normalization mean per channel
- `std`: Normalization std per channel

**Methods:**

##### `preprocess(image: np.ndarray) -> np.ndarray`

Preprocess a single fundus image.

**Args:**
- `image`: Input BGR image (H, W, 3)

**Returns:**
- Preprocessed RGB image (H, W, 3), float32, normalized

**Example:**
```python
preprocessor = FundusPreprocessor(
    target_size=(512, 512),
    ben_graham_enabled=True,
    clahe_enabled=True
)

image_bgr = cv2.imread('fundus.png')
processed = preprocessor.preprocess(image_bgr)
```

##### `ben_graham_preprocessing(image: np.ndarray, radius: int) -> np.ndarray`

Apply Ben Graham's method (subtract local average, clip, scale).

##### `circular_crop(image: np.ndarray) -> np.ndarray`

Crop image to largest inscribed circle.

##### `apply_clahe(image: np.ndarray) -> np.ndarray`

Apply Contrast Limited Adaptive Histogram Equalization.

---

### `src.data.augmentation`

#### `FundusAugmentation`

Domain-specific augmentations for fundus images.

```python
class FundusAugmentation:
    def __init__(
        self,
        p: float = 0.5,
        rotation_limit: int = 20,
        scale_limit: float = 0.1,
        brightness_limit: float = 0.2,
        contrast_limit: float = 0.2,
        hue_shift_limit: int = 10
    )
```

**Methods:**

##### `__call__(image: np.ndarray) -> np.ndarray`

Apply random augmentations to image.

**Example:**
```python
augmenter = FundusAugmentation(p=0.8)
augmented = augmenter(image)
```

---

### `src.data.datasets`

#### `APTOSDataset`

PyTorch dataset for APTOS 2019 Blindness Detection.

```python
class APTOSDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        csv_file: str,
        preprocessor: FundusPreprocessor,
        vessel_segmenter: Optional[nn.Module] = None,
        vessel_cache_dir: Optional[str] = None,
        augmentation: Optional[FundusAugmentation] = None,
        phase: str = 'train'
    )
```

**Parameters:**
- `root_dir`: Directory containing images/
- `csv_file`: CSV with columns ['id_code', 'diagnosis']
- `preprocessor`: FundusPreprocessor instance
- `vessel_segmenter`: Optional vessel segmentation model
- `vessel_cache_dir`: Directory to cache vessel maps
- `augmentation`: Optional augmentation pipeline
- `phase`: 'train', 'val', or 'test'

**Returns:**
```python
{
    'image': torch.Tensor,        # (3, H, W)
    'vessel': torch.Tensor,       # (1, H, W)
    'label': int,                 # 0-4
    'image_id': str
}
```

#### `EyePACSDataset`

PyTorch dataset for Kaggle DR Detection (EyePACS).

Similar API to APTOSDataset, expects CSV with ['image', 'level'] columns.

#### `Messidor2Dataset`

PyTorch dataset for Messidor-2.

Similar API to APTOSDataset, expects CSV with ['image_id', 'adjudicated_dr_grade'] columns.

---

## Models

### `src.models.vessel_segmentation`

#### `UNetVesselSegmenter`

U-Net based vessel segmentation using Segmentation Models PyTorch.

```python
class UNetVesselSegmenter(nn.Module):
    def __init__(
        self,
        encoder_name: str = 'resnet34',
        encoder_weights: str = 'imagenet',
        in_channels: int = 3,
        classes: int = 1,
        pretrained: bool = True
    )
```

**Methods:**

##### `forward(x: torch.Tensor) -> torch.Tensor`

**Args:**
- `x`: Input image tensor (B, 3, H, W)

**Returns:**
- Vessel probability map (B, 1, H, W), logits (apply sigmoid for probabilities)

**Example:**
```python
segmenter = UNetVesselSegmenter(encoder_name='resnet34', pretrained=True)
vessel_logits = segmenter(images)
vessel_probs = torch.sigmoid(vessel_logits)
```

##### `segment_image(image: torch.Tensor, threshold: float = 0.5) -> torch.Tensor`

Segment vessels from a single image with thresholding.

---

### `src.models.fusion`

#### `EarlyFusion`

Concatenates image and vessel channels at input level.

```python
class EarlyFusion(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        num_classes: int = 5,
        dropout_rate: float = 0.5
    )
```

**Forward:**
```python
output = model(image, vessel)  # image: (B,3,H,W), vessel: (B,1,H,W)
# Returns: (B, num_classes) logits
```

#### `DualBranchFusion`

Parallel processing of image and vessel with late fusion.

```python
class DualBranchFusion(nn.Module):
    def __init__(
        self,
        image_backbone: nn.Module,
        vessel_backbone: nn.Module,
        num_classes: int = 5,
        dropout_rate: float = 0.5,
        fusion_dim: int = 512
    )
```

#### `AttentionGatedFusion`

Attention mechanism to weight image and vessel features.

```python
class AttentionGatedFusion(nn.Module):
    def __init__(
        self,
        image_backbone: nn.Module,
        vessel_backbone: nn.Module,
        num_classes: int = 5,
        dropout_rate: float = 0.5,
        fusion_dim: int = 512
    )
```

**Example:**
```python
from src.models.fusion import AttentionGatedFusion
from src.models.backbones import get_backbone

backbone = get_backbone('resnet50', pretrained=True)
model = AttentionGatedFusion(
    image_backbone=backbone,
    vessel_backbone=get_backbone('resnet34', pretrained=True),
    num_classes=5
)

output = model(images, vessels)  # (B, 5) logits
```

---

### `src.models.classification`

#### `DRClassifier`

High-level wrapper for DR classification with flexible fusion.

```python
class DRClassifier(nn.Module):
    def __init__(
        self,
        backbone_name: str = 'resnet50',
        fusion_strategy: str = 'attention_gated',
        num_classes: int = 5,
        pretrained: bool = True,
        dropout_rate: float = 0.5
    )
```

**Parameters:**
- `backbone_name`: 'resnet50', 'efficientnet-b4', 'densenet121', etc.
- `fusion_strategy`: 'early_fusion', 'dual_branch', 'attention_gated'
- `num_classes`: Number of DR severity classes
- `pretrained`: Use ImageNet pretrained weights
- `dropout_rate`: Dropout probability

**Example:**
```python
model = DRClassifier(
    backbone_name='efficientnet-b4',
    fusion_strategy='attention_gated',
    num_classes=5,
    pretrained=True
)

logits = model(images, vessels)
predictions = logits.argmax(dim=1)
```

---

## Training

### `src.training.losses`

#### `QuadraticWeightedKappaLoss`

Differentiable surrogate for QWK metric.

```python
class QuadraticWeightedKappaLoss(nn.Module):
    def __init__(self, num_classes: int = 5, epsilon: float = 1e-7)
    
    def forward(
        self,
        y_pred: torch.Tensor,  # (B, C) logits
        y_true: torch.Tensor   # (B,) labels
    ) -> torch.Tensor
```

**Returns:** Scalar loss (1 - QWK), lower is better

#### `FocalLoss`

Handles class imbalance by down-weighting easy examples.

```python
class FocalLoss(nn.Module):
    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        reduction: str = 'mean'
    )
```

#### `HybridLoss`

Combination of QWK and Cross-Entropy.

```python
class HybridLoss(nn.Module):
    def __init__(
        self,
        qwk_weight: float = 0.5,
        ce_weight: float = 0.5,
        num_classes: int = 5
    )
```

**Example:**
```python
criterion = HybridLoss(qwk_weight=0.7, ce_weight=0.3)
loss = criterion(predictions, labels)
```

---

### `src.training.trainer`

#### `Trainer`

Complete training loop with validation, checkpointing, and logging.

```python
class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any] = None,
        device: str = 'cuda',
        num_epochs: int = 100,
        checkpoint_dir: str = './checkpoints',
        early_stopping_patience: int = 10,
        mixed_precision: bool = True,
        gradient_clip_val: Optional[float] = 1.0
    )
```

**Methods:**

##### `train() -> Dict[str, List[float]]`

Run full training loop.

**Returns:**
```python
{
    'train_loss': [...],
    'val_loss': [...],
    'val_qwk': [...],
    'val_accuracy': [...]
}
```

##### `train_epoch() -> float`

Train for one epoch.

##### `validate() -> Tuple[float, float, float]`

Validate current model.

**Returns:** `(val_loss, val_qwk, val_accuracy)`

**Example:**
```python
trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=HybridLoss(),
    optimizer=optimizer,
    scheduler=scheduler,
    device='cuda',
    num_epochs=50,
    early_stopping_patience=10
)

history = trainer.train()
```

---

## Evaluation

### `src.evaluation.metrics`

#### `compute_quadratic_weighted_kappa(y_true, y_pred, num_classes=5) -> float`

Compute QWK metric.

#### `compute_all_metrics(y_true, y_pred, y_prob, num_classes=5) -> Dict`

Compute comprehensive evaluation metrics.

**Returns:**
```python
{
    'accuracy': float,
    'qwk': float,
    'mae': float,
    'auc_ovr': float,
    'auc_ovo': float,
    'per_class_accuracy': [float, ...],
    'per_class_precision': [float, ...],
    'per_class_recall': [float, ...],
    'per_class_f1': [float, ...]
}
```

**Example:**
```python
metrics = compute_all_metrics(labels, predictions, probabilities)
print(f"QWK: {metrics['qwk']:.4f}")
print(f"AUC: {metrics['auc_ovr']:.4f}")
```

---

### `src.evaluation.evaluator`

#### `ModelEvaluator`

High-level evaluation with visualizations.

```python
class ModelEvaluator:
    def __init__(
        self,
        model: nn.Module,
        test_loader: DataLoader,
        device: str = 'cuda',
        num_classes: int = 5,
        class_names: List[str] = None
    )
```

**Methods:**

##### `evaluate() -> Dict`

Run full evaluation.

##### `plot_confusion_matrix(save_path: Optional[str] = None)`

Plot and save confusion matrix.

##### `plot_roc_curves(save_path: Optional[str] = None)`

Plot ROC curves for each class.

**Example:**
```python
evaluator = ModelEvaluator(model, test_loader, device='cuda')
results = evaluator.evaluate()
evaluator.plot_confusion_matrix('confusion_matrix.png')
evaluator.plot_roc_curves('roc_curves.png')
```

---

## Explainability

### `src.explainability.gradcam`

#### `GradCAMPlusPlus`

Enhanced Grad-CAM with better localization.

```python
class GradCAMPlusPlus:
    def __init__(
        self,
        model: nn.Module,
        target_layer: nn.Module
    )
```

**Methods:**

##### `generate_heatmap(image, vessel, target_class=None) -> np.ndarray`

Generate attention heatmap for a prediction.

**Args:**
- `image`: Input image tensor (B, 3, H, W)
- `vessel`: Vessel map tensor (B, 1, H, W)
- `target_class`: Class to explain (None = predicted class)

**Returns:** Heatmap (H, W), values in [0, 1]

**Example:**
```python
# Get last conv layer
target_layer = model.backbone.model.layer4[-1]

gradcam = GradCAMPlusPlus(model, target_layer)
heatmap = gradcam.generate_heatmap(image, vessel, target_class=3)

# Overlay on image
overlay = overlay_heatmap(original_image, heatmap)
```

---

### `src.explainability.overlap_score`

#### `compute_vessel_attention_overlap(heatmap, vessel_map, threshold=0.5) -> Dict`

Quantify alignment between attention and vessels.

**Args:**
- `heatmap`: Attention heatmap (H, W), [0, 1]
- `vessel_map`: Binary vessel map (H, W), {0, 1}
- `threshold`: Attention threshold

**Returns:**
```python
{
    'overlap_score': float,    # Overall overlap metric
    'coverage': float,          # % of vessels with high attention
    'precision': float,         # % of attention on vessels
    'vessel_area': float,       # % of image that is vessels
    'attention_area': float     # % of image with high attention
}
```

**Example:**
```python
metrics = compute_vessel_attention_overlap(heatmap, vessel_map)
print(f"Overlap Score: {metrics['overlap_score']:.3f}")
print(f"Coverage: {metrics['coverage']:.3f}")
```

---

## Utilities

### `src.utils.config`

#### `load_config(config_path: str) -> Dict`

Load YAML configuration file.

#### `save_config(config: Dict, save_path: str)`

Save configuration to YAML.

**Example:**
```python
config = load_config('configs/config.yaml')
print(config['model']['backbone'])
```

---

### `src.utils.logging_utils`

#### `setup_logger(name: str, log_file: str, level=logging.INFO) -> logging.Logger`

Create configured logger.

**Example:**
```python
logger = setup_logger('training', 'logs/train.log')
logger.info('Starting training...')
```

---

### `src.utils.random_utils`

#### `set_seed(seed: int = 42)`

Set random seed for reproducibility.

**Example:**
```python
set_seed(42)  # Ensures reproducible results
```

---

## Command-Line Scripts

### `scripts/train.py`

```bash
python scripts/train.py \
    --config configs/config.yaml \
    --gpu 0 \
    --resume checkpoints/last_model.pth
```

**Arguments:**
- `--config`: Path to config YAML
- `--gpu`: GPU device ID
- `--resume`: Resume from checkpoint
- `--seed`: Random seed

---

### `scripts/evaluate.py`

```bash
python scripts/evaluate.py \
    --checkpoint models/checkpoints/best_model.pth \
    --config configs/config.yaml \
    --output results/
```

**Arguments:**
- `--checkpoint`: Model checkpoint
- `--config`: Config file
- `--output`: Output directory for plots
- `--batch-size`: Evaluation batch size

---

### `scripts/run_ablations.py`

```bash
python scripts/run_ablations.py \
    --ablation fusion \
    --output ablation_results/
```

**Arguments:**
- `--ablation`: Which ablation ('fusion', 'backbone', 'loss', 'all')
- `--output`: Results directory
- `--epochs`: Epochs per experiment
- `--gpu`: GPU device

---

## Configuration File Format

Example `config.yaml`:

```yaml
data:
  image_size: 512
  batch_size: 16
  num_workers: 4
  datasets:
    - name: aptos
      root: /data/aptos
      csv: train.csv
      weight: 1.0
    - name: eyepacs
      root: /data/eyepacs
      csv: trainLabels.csv
      weight: 2.0

model:
  backbone: resnet50
  fusion_strategy: attention_gated
  num_classes: 5
  dropout_rate: 0.5

vessel_segmentation:
  encoder: resnet34
  checkpoint: models/vessel_segmenter.pth

training:
  num_epochs: 50
  learning_rate: 0.001
  weight_decay: 0.0001
  optimizer: adam
  scheduler: cosine
  early_stopping_patience: 10

loss:
  type: hybrid
  qwk_weight: 0.7
  ce_weight: 0.3

preprocessing:
  ben_graham: true
  clahe: true
  circular_crop: true

augmentation:
  enabled: true
  p: 0.8
  rotation_limit: 20
```

---

## Type Hints Reference

Common type aliases used throughout the codebase:

```python
from typing import Tuple, Optional, Dict, List, Union
import torch
import numpy as np

ImageTensor = torch.Tensor      # (B, C, H, W)
VesselTensor = torch.Tensor     # (B, 1, H, W)
LabelTensor = torch.Tensor      # (B,)
Heatmap = np.ndarray           # (H, W)
ImageArray = np.ndarray        # (H, W, C)
```

---

## Error Handling

All modules follow consistent error handling:

```python
try:
    result = function_call()
except FileNotFoundError:
    logger.error("File not found")
    raise
except ValueError as e:
    logger.error(f"Invalid value: {e}")
    raise
except Exception as e:
    logger.exception("Unexpected error")
    raise
```

---

## Logging Conventions

- `DEBUG`: Detailed information for debugging
- `INFO`: General progress updates
- `WARNING`: Non-critical issues
- `ERROR`: Errors that prevent operation
- `CRITICAL`: System-level failures

---

## Best Practices

1. **Always use config files** - Don't hardcode hyperparameters
2. **Set random seeds** - Use `set_seed()` for reproducibility
3. **Validate inputs** - Check tensor shapes and data types
4. **Cache vessel maps** - Use vessel_cache_dir to avoid re-computation
5. **Monitor metrics** - Use Trainer's history for debugging
6. **Mixed precision** - Enable for faster training on modern GPUs
7. **Checkpoint regularly** - Trainer saves best and last models
8. **Log everything** - Use logging_utils for consistent logs

---

## Common Workflows

### Training a New Model

```python
# 1. Load config
config = load_config('configs/config.yaml')

# 2. Setup data
preprocessor = FundusPreprocessor(**config['preprocessing'])
train_dataset = APTOSDataset(root, csv, preprocessor)
train_loader = DataLoader(train_dataset, **config['data'])

# 3. Create model
model = DRClassifier(**config['model'])

# 4. Setup training
criterion = HybridLoss(**config['loss'])
optimizer = torch.optim.Adam(model.parameters(), **config['training'])
trainer = Trainer(model, train_loader, val_loader, criterion, optimizer)

# 5. Train
history = trainer.train()
```

### Evaluating a Model

```python
# 1. Load checkpoint
checkpoint = torch.load('best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])

# 2. Evaluate
evaluator = ModelEvaluator(model, test_loader)
results = evaluator.evaluate()

# 3. Visualize
evaluator.plot_confusion_matrix()
evaluator.plot_roc_curves()
```

### Generating Explainability

```python
# 1. Get target layer
target_layer = model.backbone.model.layer4[-1]

# 2. Create Grad-CAM++
gradcam = GradCAMPlusPlus(model, target_layer)

# 3. Generate heatmap
heatmap = gradcam.generate_heatmap(image, vessel, target_class=prediction)

# 4. Compute overlap
metrics = compute_vessel_attention_overlap(heatmap, vessel_map)
```

---

*For more examples, see the notebooks/ directory and USAGE_GUIDE.md*

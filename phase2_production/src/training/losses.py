"""
VERA Phase 2 - Loss Functions

Implements multiple loss functions for DR classification including:
- Cross-Entropy Loss
- Focal Loss
- Quadratic Weighted Kappa (QWK) Loss
- Ordinal Loss
- Hybrid combinations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class QuadraticWeightedKappaLoss(nn.Module):
    """
    Differentiable Quadratic Weighted Kappa (QWK) Loss.
    
    QWK is the official metric for DR grading competitions. This implementation
    uses a differentiable surrogate to allow end-to-end training optimizing
    directly for the target metric.
    
    Reference:
    - https://www.kaggle.com/c/diabetic-retinopathy-detection/overview/evaluation
    - Cohen's Kappa with quadratic weights
    """
    
    def __init__(
        self,
        num_classes: int = 5,
        eps: float = 1e-10
    ):
        """
        Initialize QWK loss.
        
        Args:
            num_classes: Number of classes
            eps: Small constant for numerical stability
        """
        super(QuadraticWeightedKappaLoss, self).__init__()
        
        self.num_classes = num_classes
        self.eps = eps
        
        # Create weight matrix for quadratic weighting
        # Penalizes larger grade discrepancies more heavily
        self.register_buffer(
            'weight_matrix',
            self._create_weight_matrix(num_classes)
        )
        
        logger.info(f"Initialized QWK Loss with {num_classes} classes")
    
    def _create_weight_matrix(self, num_classes: int) -> torch.Tensor:
        """
        Create quadratic weight matrix.
        
        Weight(i,j) = (i - j)^2 / (num_classes - 1)^2
        """
        weights = torch.zeros(num_classes, num_classes)
        
        for i in range(num_classes):
            for j in range(num_classes):
                weights[i, j] = ((i - j) ** 2) / ((num_classes - 1) ** 2)
        
        return weights
    
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate QWK loss.
        
        Args:
            predictions: Model logits (B, num_classes)
            targets: Ground truth labels (B,)
            
        Returns:
            QWK loss (lower is better, 1 - kappa)
        """
        # Convert logits to probabilities
        probs = F.softmax(predictions, dim=1)  # (B, num_classes)
        
        # One-hot encode targets
        targets_one_hot = F.one_hot(targets, self.num_classes).float()  # (B, num_classes)
        
        # Expected confusion matrix (using soft predictions)
        batch_size = predictions.shape[0]
        
        # Compute histogram matrices
        pred_hist = probs.sum(dim=0) / batch_size  # (num_classes,)
        target_hist = targets_one_hot.sum(dim=0) / batch_size  # (num_classes,)
        
        # Outer product for expected matrix
        expected_matrix = torch.outer(target_hist, pred_hist)  # (num_classes, num_classes)
        
        # Observed matrix (soft version)
        observed_matrix = torch.matmul(
            targets_one_hot.t(), probs
        ) / batch_size  # (num_classes, num_classes)
        
        # Calculate kappa
        numerator = (self.weight_matrix * observed_matrix).sum()
        denominator = (self.weight_matrix * expected_matrix).sum() + self.eps
        
        kappa = 1 - numerator / denominator
        
        # Return loss (minimize 1 - kappa, i.e., maximize kappa)
        loss = 1 - kappa
        
        return loss


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    
    Focuses training on hard examples by down-weighting easy examples.
    
    Reference:
    Lin et al. "Focal Loss for Dense Object Detection" (2017)
    """
    
    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        reduction: str = 'mean'
    ):
        """
        Initialize Focal Loss.
        
        Args:
            alpha: Class weights (num_classes,)
            gamma: Focusing parameter (higher = more focus on hard examples)
            reduction: Reduction method ('mean', 'sum', 'none')
        """
        super(FocalLoss, self).__init__()
        
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
        logger.info(f"Initialized Focal Loss: gamma={gamma}")
    
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate Focal Loss.
        
        Args:
            predictions: Model logits (B, num_classes)
            targets: Ground truth labels (B,)
            
        Returns:
            Focal loss
        """
        # Cross entropy with logits
        ce_loss = F.cross_entropy(predictions, targets, reduction='none')
        
        # Get probabilities
        probs = F.softmax(predictions, dim=1)
        target_probs = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        
        # Focal term
        focal_weight = (1 - target_probs) ** self.gamma
        
        # Apply focal weight
        loss = focal_weight * ce_loss
        
        # Apply alpha if provided
        if self.alpha is not None:
            alpha_t = self.alpha.gather(0, targets)
            loss = alpha_t * loss
        
        # Reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class OrdinalLoss(nn.Module):
    """
    Ordinal Regression Loss for DR severity grading.
    
    Treats the problem as multiple binary classifications with
    ordinal relationships between classes.
    """
    
    def __init__(self, num_classes: int = 5):
        """
        Initialize Ordinal Loss.
        
        Args:
            num_classes: Number of ordinal classes
        """
        super(OrdinalLoss, self).__init__()
        
        self.num_classes = num_classes
        self.num_thresholds = num_classes - 1
        
        logger.info(f"Initialized Ordinal Loss with {num_classes} classes")
    
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate Ordinal Loss.
        
        Args:
            predictions: Cumulative logits (B, num_classes-1)
            targets: Ground truth labels (B,)
            
        Returns:
            Ordinal loss
        """
        # Create binary targets for each threshold
        # Target[i,j] = 1 if class > j, else 0
        batch_size = targets.shape[0]
        
        binary_targets = torch.zeros(
            batch_size, self.num_thresholds,
            device=targets.device, dtype=torch.float32
        )
        
        for i in range(self.num_thresholds):
            binary_targets[:, i] = (targets > i).float()
        
        # Binary cross-entropy for each threshold
        loss = F.binary_cross_entropy_with_logits(
            predictions,
            binary_targets,
            reduction='mean'
        )
        
        return loss


class HybridLoss(nn.Module):
    """
    Hybrid loss combining Cross-Entropy and QWK Loss.
    
    Balances local gradient signal (CE) with global metric optimization (QWK).
    """
    
    def __init__(
        self,
        num_classes: int = 5,
        ce_weight: float = 0.5,
        qwk_weight: float = 0.5,
        class_weights: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.0
    ):
        """
        Initialize Hybrid Loss.
        
        Args:
            num_classes: Number of classes
            ce_weight: Weight for cross-entropy term
            qwk_weight: Weight for QWK term
            class_weights: Optional class weights for CE
            label_smoothing: Label smoothing factor
        """
        super(HybridLoss, self).__init__()
        
        self.ce_weight = ce_weight
        self.qwk_weight = qwk_weight
        
        self.ce_loss = nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=label_smoothing
        )
        
        self.qwk_loss = QuadraticWeightedKappaLoss(num_classes=num_classes)
        
        logger.info(f"Initialized Hybrid Loss: CE weight={ce_weight}, QWK weight={qwk_weight}")
    
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """Calculate hybrid loss."""
        ce = self.ce_loss(predictions, targets)
        qwk = self.qwk_loss(predictions, targets)
        
        total_loss = self.ce_weight * ce + self.qwk_weight * qwk
        
        return total_loss


def create_loss_function(config: dict, class_weights: Optional[torch.Tensor] = None) -> nn.Module:
    """
    Factory function to create loss function from configuration.
    
    Args:
        config: Configuration dictionary
        class_weights: Optional class weights
        
    Returns:
        Loss function instance
    """
    loss_config = config.get('training', {}).get('loss', {})
    loss_type = loss_config.get('type', 'cross_entropy')
    
    num_classes = config.get('model', {}).get('num_classes', 5)
    
    # Handle class weights
    use_class_weights = loss_config.get('class_weights', 'auto')
    if use_class_weights == 'auto' and class_weights is not None:
        weights = class_weights
    elif use_class_weights == 'balanced':
        # Will be computed from data
        weights = class_weights
    else:
        weights = None
    
    # Create loss function based on type
    if loss_type == 'cross_entropy':
        label_smoothing = loss_config.get('label_smoothing', 0.0)
        
        loss_fn = nn.CrossEntropyLoss(
            weight=weights,
            label_smoothing=label_smoothing
        )
        logger.info(f"Using Cross-Entropy Loss with label smoothing={label_smoothing}")
    
    elif loss_type == 'focal':
        gamma = loss_config.get('focal_gamma', 2.0)
        alpha = loss_config.get('focal_alpha', None)
        
        if alpha is not None and weights is not None:
            alpha = weights
        
        loss_fn = FocalLoss(
            alpha=alpha,
            gamma=gamma,
            reduction='mean'
        )
        logger.info(f"Using Focal Loss with gamma={gamma}")
    
    elif loss_type == 'qwk':
        loss_fn = QuadraticWeightedKappaLoss(num_classes=num_classes)
        logger.info("Using QWK Loss")
    
    elif loss_type == 'ordinal':
        loss_fn = OrdinalLoss(num_classes=num_classes)
        logger.info("Using Ordinal Loss")
    
    elif loss_type == 'hybrid':
        ce_weight = loss_config.get('ce_weight', 0.5)
        qwk_weight = loss_config.get('qwk_weight', 0.5)
        label_smoothing = loss_config.get('label_smoothing', 0.0)
        
        loss_fn = HybridLoss(
            num_classes=num_classes,
            ce_weight=ce_weight,
            qwk_weight=qwk_weight,
            class_weights=weights,
            label_smoothing=label_smoothing
        )
        logger.info(f"Using Hybrid Loss: CE={ce_weight}, QWK={qwk_weight}")
    
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
    
    return loss_fn


def compute_qwk_metric(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int = 5
) -> float:
    """
    Compute Quadratic Weighted Kappa as a metric (not differentiable).
    
    Args:
        predictions: Predicted class labels (B,)
        targets: Ground truth labels (B,)
        num_classes: Number of classes
        
    Returns:
        QWK score (0 to 1, higher is better)
    """
    from sklearn.metrics import cohen_kappa_score
    
    # Convert to numpy
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()
    
    # Compute QWK
    qwk = cohen_kappa_score(
        targets,
        predictions,
        weights='quadratic',
        labels=list(range(num_classes))
    )
    
    return qwk

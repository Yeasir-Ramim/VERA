"""
VERA Phase 2 - Evaluation Metrics

Comprehensive metrics for DR classification evaluation.
"""

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score,
    recall_score, f1_score, confusion_matrix, roc_auc_score,
    cohen_kappa_score, classification_report
)
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)


def compute_all_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    probabilities: np.ndarray = None,
    num_classes: int = 5
) -> Dict[str, float]:
    """
    Compute comprehensive evaluation metrics.
    
    Args:
        predictions: Predicted class labels (N,)
        targets: Ground truth labels (N,)
        probabilities: Class probabilities (N, num_classes)
        num_classes: Number of classes
        
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    # Accuracy
    metrics['accuracy'] = accuracy_score(targets, predictions)
    metrics['balanced_accuracy'] = balanced_accuracy_score(targets, predictions)
    
    # Quadratic Weighted Kappa (QWK)
    metrics['qwk'] = cohen_kappa_score(
        targets, predictions,
        weights='quadratic',
        labels=list(range(num_classes))
    )
    
    # Linear Weighted Kappa
    metrics['lwk'] = cohen_kappa_score(
        targets, predictions,
        weights='linear',
        labels=list(range(num_classes))
    )
    
    # Unweighted Kappa
    metrics['kappa'] = cohen_kappa_score(targets, predictions)
    
    # Per-class metrics (macro average)
    metrics['precision_macro'] = precision_score(
        targets, predictions, average='macro', zero_division=0
    )
    metrics['recall_macro'] = recall_score(
        targets, predictions, average='macro', zero_division=0
    )
    metrics['f1_macro'] = f1_score(
        targets, predictions, average='macro', zero_division=0
    )
    
    # Weighted average (accounts for class imbalance)
    metrics['precision_weighted'] = precision_score(
        targets, predictions, average='weighted', zero_division=0
    )
    metrics['recall_weighted'] = recall_score(
        targets, predictions, average='weighted', zero_division=0
    )
    metrics['f1_weighted'] = f1_score(
        targets, predictions, average='weighted', zero_division=0
    )
    
    # Per-class metrics
    precision_per_class = precision_score(
        targets, predictions, average=None, zero_division=0
    )
    recall_per_class = recall_score(
        targets, predictions, average=None, zero_division=0
    )
    f1_per_class = f1_score(
        targets, predictions, average=None, zero_division=0
    )
    
    for i in range(num_classes):
        metrics[f'precision_class{i}'] = precision_per_class[i]
        metrics[f'recall_class{i}'] = recall_per_class[i]
        metrics[f'f1_class{i}'] = f1_per_class[i]
    
    # Binary referable DR metrics (0-1 vs 2-4)
    targets_binary = (targets >= 2).astype(int)
    predictions_binary = (predictions >= 2).astype(int)
    
    metrics['referable_accuracy'] = accuracy_score(targets_binary, predictions_binary)
    metrics['referable_precision'] = precision_score(
        targets_binary, predictions_binary, zero_division=0
    )
    metrics['referable_recall'] = recall_score(
        targets_binary, predictions_binary, zero_division=0
    )
    metrics['referable_f1'] = f1_score(
        targets_binary, predictions_binary, zero_division=0
    )
    
    # ROC-AUC for referable DR if probabilities provided
    if probabilities is not None:
        # Binary referable AUC
        probabilities_referable = probabilities[:, 2:].sum(axis=1)
        try:
            metrics['referable_auc'] = roc_auc_score(
                targets_binary, probabilities_referable
            )
        except:
            metrics['referable_auc'] = 0.0
        
        # Multi-class AUC (one-vs-rest)
        try:
            metrics['auc_ovr'] = roc_auc_score(
                targets, probabilities,
                multi_class='ovr',
                labels=list(range(num_classes))
            )
        except:
            metrics['auc_ovr'] = 0.0
    
    return metrics


def compute_confusion_matrix(
    predictions: np.ndarray,
    targets: np.ndarray,
    num_classes: int = 5,
    normalize: str = None
) -> np.ndarray:
    """
    Compute confusion matrix.
    
    Args:
        predictions: Predicted labels
        targets: True labels
        num_classes: Number of classes
        normalize: 'true', 'pred', 'all', or None
        
    Returns:
        Confusion matrix
    """
    cm = confusion_matrix(
        targets, predictions,
        labels=list(range(num_classes))
    )
    
    if normalize == 'true':
        cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    elif normalize == 'pred':
        cm = cm.astype('float') / cm.sum(axis=0, keepdims=True)
    elif normalize == 'all':
        cm = cm.astype('float') / cm.sum()
    
    return cm


def get_classification_report(
    predictions: np.ndarray,
    targets: np.ndarray,
    class_names: List[str] = None
) -> str:
    """
    Get detailed classification report.
    
    Args:
        predictions: Predicted labels
        targets: True labels
        class_names: List of class names
        
    Returns:
        Classification report string
    """
    if class_names is None:
        class_names = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative']
    
    report = classification_report(
        targets, predictions,
        target_names=class_names,
        digits=4
    )
    
    return report


def compute_grade_distribution(labels: np.ndarray, num_classes: int = 5) -> Dict:
    """
    Compute class distribution statistics.
    
    Args:
        labels: Array of labels
        num_classes: Number of classes
        
    Returns:
        Dictionary with distribution info
    """
    counts = np.bincount(labels, minlength=num_classes)
    total = len(labels)
    
    distribution = {
        'counts': counts.tolist(),
        'percentages': (counts / total * 100).tolist(),
        'total': total
    }
    
    return distribution

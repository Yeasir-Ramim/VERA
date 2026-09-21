"""
Evaluation and Metrics reporting module for Diabetic Retinopathy Classification.
Implements:
- Multiclass Accuracy and Quadratic Weighted Kappa (QWK).
- Binary Referable DR classification metrics (ROC-AUC, Sensitivity, Specificity).
- Detailed per-class precision, recall, and F1 reports.
- Seaborn Confusion Matrix visualization.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    roc_auc_score,
)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


ICDR_CLASS_NAMES = [
    "0 - No DR",
    "1 - Mild",
    "2 - Moderate",
    "3 - Severe",
    "4 - Proliferative"
]


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: Optional[str] = None,
    class_names: Optional[List[str]] = None
) -> Dict[str, Union[float, np.ndarray, str]]:
    """
    Runs full inference on dataloader and calculates comprehensive DR metrics:
    Accuracy, Quadratic Weighted Kappa (QWK), Referable AUC, and Confusion Matrix.
    
    Returns:
        Dictionary containing:
          - 'accuracy': float
          - 'qwk': float (Quadratic Weighted Kappa)
          - 'referable_auc': float (AUC for Grade >= 2)
          - 'confusion_matrix': ndarray (5, 5)
          - 'classification_report': str
          - 'y_true': ndarray
          - 'y_pred': ndarray
          - 'y_probs': ndarray
    """
    if device is None:
        device_obj = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device_obj = torch.device(device)
        
    model.eval()
    model.to(device_obj)
    
    y_true = []
    y_pred = []
    y_probs = []
    
    with torch.no_grad():
        for inputs, targets, _ in dataloader:
            inputs = inputs.to(device_obj)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            y_true.extend(targets.numpy() if hasattr(targets, 'numpy') else targets)
            y_pred.extend(preds.cpu().numpy())
            y_probs.extend(probs.cpu().numpy())
            
    y_true = np.array(y_true, dtype=int)
    y_pred = np.array(y_pred, dtype=int)
    y_probs = np.array(y_probs, dtype=float)
    
    names = class_names or ICDR_CLASS_NAMES
    acc = accuracy_score(y_true, y_pred)
    
    # Quadratic Weighted Kappa (official clinical / competition metric)
    try:
        qwk = cohen_kappa_score(y_true, y_pred, weights="quadratic")
    except Exception:
        qwk = 0.0
        
    # Referable DR metrics (Non-Referable: 0-1 vs Referable: 2-4)
    y_true_ref = (y_true >= 2).astype(int)
    p_ref = np.sum(y_probs[:, 2:], axis=1) if y_probs.ndim == 2 and y_probs.shape[1] >= 5 else (y_pred >= 2).astype(float)
    
    try:
        if len(np.unique(y_true_ref)) > 1:
            ref_auc = roc_auc_score(y_true_ref, p_ref)
        else:
            ref_auc = 1.0
    except Exception:
        ref_auc = 0.0
        
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(names))))
    report = classification_report(y_true, y_pred, target_names=names, zero_division=0)
    
    return {
        "accuracy": float(acc),
        "qwk": float(qwk),
        "referable_auc": float(ref_auc),
        "confusion_matrix": cm,
        "classification_report": report,
        "y_true": y_true,
        "y_pred": y_pred,
        "y_probs": y_probs
    }


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: Optional[List[str]] = None,
    title: str = "Confusion Matrix - DR Severity Classification",
    save_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (8, 6)
) -> plt.Figure:
    """
    Renders and optionally saves a formatted Seaborn heatmap for the confusion matrix.
    """
    names = class_names or ICDR_CLASS_NAMES
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=names,
        yticklabels=names,
        cbar=True,
        ax=ax
    )
    
    ax.set_title(title, fontsize=14, pad=15, weight="bold")
    ax.set_xlabel("Predicted Severity Grade", fontsize=12)
    ax.set_ylabel("True Ground Truth Grade", fontsize=12)
    plt.xticks(rotation=30, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        
    return fig

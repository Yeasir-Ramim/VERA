"""
Training and Optimization Pipeline for Diabetic Retinopathy Classification.
Implements:
- Class-weighted Cross-Entropy loss.
- Multi-Class Focal Loss for handling severe class imbalances.
- Differentiable Quadratic Weighted Kappa (QWK) Loss.
- Combined CE + QWK Hybrid Loss.
- AdamW optimizer with Cosine Annealing Learning Rate scheduling.
- Validation tracking with QWK and model checkpointing.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from sklearn.metrics import cohen_kappa_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class QuadraticWeightedKappaLoss(nn.Module):
    """
    Differentiable surrogate loss for Quadratic Weighted Kappa (QWK).
    Directly optimizes the competition metric by penalizing large grade discrepancies quadratically.
    """
    def __init__(self, num_classes: int = 5, epsilon: float = 1e-7):
        super().__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        
        # Quadratic weight matrix: W_ij = (i - j)^2 / (K - 1)^2
        w = np.zeros((num_classes, num_classes), dtype=np.float32)
        for i in range(num_classes):
            for j in range(num_classes):
                w[i, j] = ((i - j) ** 2) / ((num_classes - 1) ** 2)
        self.register_buffer("weight_matrix", torch.tensor(w))
        
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits, dim=1)  # (B, K)
        batch_size = probs.size(0)
        
        if targets.ndim == 1:
            targets_one_hot = torch.zeros_like(probs).scatter_(1, targets.unsqueeze(1), 1.0)
        else:
            targets_one_hot = targets.float()
            
        # Observed agreement matrix: O = P^T * Y / B
        O = torch.matmul(probs.t(), targets_one_hot) / max(1, batch_size)
        
        # Expected agreement matrix: E = (P_marginal)^T * (Y_marginal)
        hist_p = probs.mean(dim=0, keepdim=True)
        hist_y = targets_one_hot.mean(dim=0, keepdim=True)
        E = torch.matmul(hist_p.t(), hist_y)
        
        # Weighted agreement sums
        num = torch.sum(self.weight_matrix * O)
        den = torch.sum(self.weight_matrix * E) + self.epsilon
        
        qwk = 1.0 - (num / den)
        # Loss: minimize (1 - QWK)
        return 1.0 - qwk


class FocalLoss(nn.Module):
    """
    Multi-Class Focal Loss:
    FL(p_t) = - alpha_t * (1 - p_t)^gamma * log(p_t)
    Down-weights well-classified easy samples to focus learning on hard minority cases.
    """
    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        reduction: str = "mean"
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_p = nn.functional.log_softmax(logits, dim=1)
        p = torch.exp(log_p)
        
        # Gather probabilities for target classes
        target_log_p = log_p.gather(1, targets.unsqueeze(1)).squeeze(1)
        target_p = p.gather(1, targets.unsqueeze(1)).squeeze(1)
        
        focal_weight = (1.0 - target_p) ** self.gamma
        loss = -focal_weight * target_log_p
        
        if self.alpha is not None:
            alpha_weight = self.alpha.to(logits.device)[targets]
            loss = alpha_weight * loss
            
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class HybridCEQWKLoss(nn.Module):
    """
    Hybrid Loss combining CrossEntropy (for stable class logit margins)
    and differentiable Quadratic Weighted Kappa (for ordinal distance penalization):
        L = L_CE + lambda_qwk * L_QWK
    """
    def __init__(self, class_weights: Optional[torch.Tensor] = None, lambda_qwk: float = 1.0, num_classes: int = 5):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=class_weights)
        self.qwk = QuadraticWeightedKappaLoss(num_classes=num_classes)
        self.lambda_qwk = lambda_qwk
        
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        loss_ce = self.ce(logits, targets)
        loss_qwk = self.qwk(logits, targets)
        return loss_ce + self.lambda_qwk * loss_qwk


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
) -> Tuple[float, float]:
    """
    Trains the model for one epoch.
    
    Returns:
        (avg_loss, accuracy)
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, targets, _ in dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += torch.sum(preds == targets).item()
        total += targets.size(0)
        
    if scheduler is not None:
        scheduler.step()
        
    epoch_loss = running_loss / max(1, total)
    epoch_acc = correct / max(1, total)
    return epoch_loss, epoch_acc


def validate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float, float, np.ndarray, np.ndarray]:
    """
    Evaluates the model on validation set.
    
    Returns:
        (avg_loss, accuracy, qwk, y_true, y_pred)
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    all_targets = []
    all_preds = []
    
    with torch.no_grad():
        for inputs, targets, _ in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == targets).item()
            total += targets.size(0)
            
            all_targets.extend(targets.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            
    val_loss = running_loss / max(1, total)
    val_acc = correct / max(1, total)
    
    y_true_arr = np.array(all_targets)
    y_pred_arr = np.array(all_preds)
    try:
        val_qwk = float(cohen_kappa_score(y_true_arr, y_pred_arr, weights="quadratic"))
    except Exception:
        val_qwk = 0.0
        
    return val_loss, val_acc, val_qwk, y_true_arr, y_pred_arr


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 10,
    learning_rate: float = 3e-4,
    weight_decay: float = 1e-4,
    class_weights: Optional[torch.Tensor] = None,
    loss_type: str = "ce",
    save_path: Optional[Union[str, Path]] = None,
    device: Optional[str] = None
) -> Dict[str, List[float]]:
    """
    Full training orchestration with loss function selection, validation tracking,
    and best-checkpoint saving (ranked by QWK and accuracy).
    
    Args:
        loss_type: 'ce' (CrossEntropy), 'focal' (FocalLoss), 'qwk' (QWKLoss), or 'hybrid' (CE + QWK).
    """
    if device is None:
        device_obj = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device_obj = torch.device(device)
        
    model.to(device_obj)
    
    # Configure loss criterion
    if loss_type == "focal":
        weights = class_weights.to(device_obj) if class_weights is not None else None
        criterion = FocalLoss(alpha=weights, gamma=2.0)
    elif loss_type == "qwk":
        criterion = QuadraticWeightedKappaLoss(num_classes=5)
    elif loss_type in ["hybrid", "combo"]:
        weights = class_weights.to(device_obj) if class_weights is not None else None
        criterion = HybridCEQWKLoss(class_weights=weights, lambda_qwk=1.0)
    else:
        # Standard Cross-Entropy with optional class weights
        weights = class_weights.to(device_obj) if class_weights is not None else None
        criterion = nn.CrossEntropyLoss(weight=weights)
        
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "val_qwk": []
    }
    
    best_score = -1.0  # Tracks combination of QWK + Acc
    
    print(f"Starting training on device: {device_obj} for {num_epochs} epochs with [{loss_type.upper()}] loss...")
    for epoch in range(num_epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device_obj, scheduler
        )
        val_loss, val_acc, val_qwk, _, _ = validate_epoch(
            model, val_loader, criterion, device_obj
        )
        
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_qwk"].append(val_qwk)
        
        print(f"Epoch [{epoch+1:02d}/{num_epochs:02d}] "
              f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val QWK: {val_qwk:.4f}")
        
        # Save model based on highest QWK (with acc tie-breaker)
        current_score = val_qwk * 0.7 + val_acc * 0.3
        if current_score > best_score:
            best_score = current_score
            if save_path:
                Path(save_path).parent.mkdir(parents=True, exist_ok=True)
                torch.save({
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "val_acc": val_acc,
                    "val_qwk": val_qwk,
                    "val_loss": val_loss
                }, str(save_path))
                
    print(f"Training complete. Best Validation Combined Score: {best_score:.4f}")
    return history

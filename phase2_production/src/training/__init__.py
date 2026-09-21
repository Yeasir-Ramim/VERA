"""
VERA Phase 2 - Training Module

Exports training infrastructure including losses, optimizers, schedulers, and trainer.
"""

from .losses import (
    QuadraticWeightedKappaLoss,
    FocalLoss,
    OrdinalLoss,
    HybridLoss,
    create_loss_function,
    compute_qwk_metric
)

from .optimizers import create_optimizer
from .schedulers import create_scheduler, get_warmup_scheduler
from .trainer import Trainer, create_trainer

__all__ = [
    # Losses
    'QuadraticWeightedKappaLoss',
    'FocalLoss',
    'OrdinalLoss',
    'HybridLoss',
    'create_loss_function',
    'compute_qwk_metric',
    
    # Optimizers
    'create_optimizer',
    
    # Schedulers
    'create_scheduler',
    'get_warmup_scheduler',
    
    # Trainer
    'Trainer',
    'create_trainer',
]

"""
VERA Phase 2 - Learning Rate Schedulers

Factory functions for creating LR schedulers with configuration.
"""

import torch.optim as optim
from torch.optim.lr_scheduler import (
    StepLR, ExponentialLR, CosineAnnealingLR,
    ReduceLROnPlateau, OneCycleLR, CosineAnnealingWarmRestarts
)
from typing import Dict
import logging

logger = logging.getLogger(__name__)


def create_scheduler(
    optimizer: optim.Optimizer,
    config: Dict,
    steps_per_epoch: int = None
):
    """
    Create learning rate scheduler from configuration.
    
    Args:
        optimizer: Optimizer instance
        config: Configuration dictionary
        steps_per_epoch: Number of training steps per epoch (for OneCycle)
        
    Returns:
        Scheduler instance
    """
    training_config = config.get('training', {})
    scheduler_config = training_config.get('scheduler', {})
    
    scheduler_type = scheduler_config.get('type', 'cosine_annealing')
    
    if scheduler_type == 'step':
        step_size = scheduler_config.get('step_size', 30)
        gamma = scheduler_config.get('factor', 0.1)
        
        scheduler = StepLR(
            optimizer,
            step_size=step_size,
            gamma=gamma
        )
        logger.info(f"Using StepLR: step_size={step_size}, gamma={gamma}")
    
    elif scheduler_type == 'exponential':
        gamma = scheduler_config.get('factor', 0.95)
        
        scheduler = ExponentialLR(
            optimizer,
            gamma=gamma
        )
        logger.info(f"Using ExponentialLR: gamma={gamma}")
    
    elif scheduler_type == 'cosine_annealing':
        epochs = training_config.get('epochs', 100)
        min_lr = scheduler_config.get('min_lr', 1e-6)
        
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=epochs,
            eta_min=min_lr
        )
        logger.info(f"Using CosineAnnealingLR: T_max={epochs}, min_lr={min_lr}")
    
    elif scheduler_type == 'cosine_annealing_warm_restarts':
        T_0 = scheduler_config.get('T_0', 10)
        T_mult = scheduler_config.get('T_mult', 2)
        min_lr = scheduler_config.get('min_lr', 1e-6)
        
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=T_0,
            T_mult=T_mult,
            eta_min=min_lr
        )
        logger.info(f"Using CosineAnnealingWarmRestarts: T_0={T_0}, T_mult={T_mult}")
    
    elif scheduler_type == 'reduce_on_plateau':
        mode = scheduler_config.get('mode', 'min')
        factor = scheduler_config.get('factor', 0.5)
        patience = scheduler_config.get('patience', 10)
        min_lr = scheduler_config.get('min_lr', 1e-6)
        
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode=mode,
            factor=factor,
            patience=patience,
            min_lr=min_lr,
            verbose=True
        )
        logger.info(f"Using ReduceLROnPlateau: mode={mode}, factor={factor}, "
                   f"patience={patience}")
    
    elif scheduler_type == 'onecycle':
        if steps_per_epoch is None:
            raise ValueError("steps_per_epoch required for OneCycleLR")
        
        epochs = training_config.get('epochs', 100)
        max_lr = training_config.get('learning_rate', 0.001)
        
        scheduler = OneCycleLR(
            optimizer,
            max_lr=max_lr,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            pct_start=0.3,
            anneal_strategy='cos'
        )
        logger.info(f"Using OneCycleLR: max_lr={max_lr}, epochs={epochs}")
    
    elif scheduler_type == 'none' or scheduler_type is None:
        scheduler = None
        logger.info("No learning rate scheduler")
    
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")
    
    return scheduler


def get_warmup_scheduler(
    optimizer: optim.Optimizer,
    warmup_epochs: int,
    base_scheduler=None
):
    """
    Create a warmup scheduler wrapper.
    
    Args:
        optimizer: Optimizer instance
        warmup_epochs: Number of warmup epochs
        base_scheduler: Base scheduler to use after warmup
        
    Returns:
        Scheduler with warmup
    """
    # Note: This is a simplified version
    # For production, consider using transformers library's get_linear_schedule_with_warmup
    
    class WarmupScheduler:
        def __init__(self, optimizer, warmup_epochs, base_scheduler=None):
            self.optimizer = optimizer
            self.warmup_epochs = warmup_epochs
            self.base_scheduler = base_scheduler
            self.current_epoch = 0
            self.base_lr = optimizer.param_groups[0]['lr']
        
        def step(self, epoch=None):
            self.current_epoch += 1
            
            if self.current_epoch <= self.warmup_epochs:
                # Linear warmup
                lr = self.base_lr * (self.current_epoch / self.warmup_epochs)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr
            elif self.base_scheduler is not None:
                self.base_scheduler.step(epoch)
        
        def state_dict(self):
            state = {
                'current_epoch': self.current_epoch,
                'base_lr': self.base_lr
            }
            if self.base_scheduler is not None:
                state['base_scheduler'] = self.base_scheduler.state_dict()
            return state
        
        def load_state_dict(self, state_dict):
            self.current_epoch = state_dict['current_epoch']
            self.base_lr = state_dict['base_lr']
            if self.base_scheduler is not None and 'base_scheduler' in state_dict:
                self.base_scheduler.load_state_dict(state_dict['base_scheduler'])
    
    return WarmupScheduler(optimizer, warmup_epochs, base_scheduler)

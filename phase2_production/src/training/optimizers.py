"""
VERA Phase 2 - Optimizers

Factory functions for creating optimizers with configuration.
"""

import torch.optim as optim
from torch.optim import Optimizer
from typing import Dict
import logging

logger = logging.getLogger(__name__)


def create_optimizer(
    parameters,
    config: Dict
) -> Optimizer:
    """
    Create optimizer from configuration.
    
    Args:
        parameters: Model parameters
        config: Configuration dictionary
        
    Returns:
        Optimizer instance
    """
    training_config = config.get('training', {})
    
    optimizer_name = training_config.get('optimizer', 'adamw').lower()
    learning_rate = training_config.get('learning_rate', 0.0001)
    weight_decay = training_config.get('weight_decay', 0.0001)
    
    if optimizer_name == 'adam':
        optimizer = optim.Adam(
            parameters,
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        logger.info(f"Using Adam optimizer: lr={learning_rate}, wd={weight_decay}")
    
    elif optimizer_name == 'adamw':
        optimizer = optim.AdamW(
            parameters,
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        logger.info(f"Using AdamW optimizer: lr={learning_rate}, wd={weight_decay}")
    
    elif optimizer_name == 'sgd':
        momentum = training_config.get('momentum', 0.9)
        nesterov = training_config.get('nesterov', True)
        
        optimizer = optim.SGD(
            parameters,
            lr=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=nesterov
        )
        logger.info(f"Using SGD optimizer: lr={learning_rate}, momentum={momentum}, "
                   f"wd={weight_decay}, nesterov={nesterov}")
    
    elif optimizer_name == 'rmsprop':
        optimizer = optim.RMSprop(
            parameters,
            lr=learning_rate,
            weight_decay=weight_decay,
            momentum=0.9
        )
        logger.info(f"Using RMSprop optimizer: lr={learning_rate}, wd={weight_decay}")
    
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    return optimizer

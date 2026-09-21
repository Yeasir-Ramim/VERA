"""
VERA Phase 2 - Configuration Utilities

Functions for loading and validating configuration files.
"""

import yaml
from pathlib import Path
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Configuration dictionary
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    logger.info(f"Loaded configuration from {config_path}")
    
    return config


def save_config(config: Dict[str, Any], output_path: str):
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        output_path: Output file path
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    logger.info(f"Saved configuration to {output_path}")


def validate_config(config: Dict[str, Any]) -> bool:
    """
    Validate configuration structure and required fields.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        True if valid, raises ValueError otherwise
    """
    required_sections = ['experiment', 'data', 'model', 'training']
    
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required configuration section: {section}")
    
    # Validate data section
    if 'datasets' not in config['data']:
        raise ValueError("Missing 'datasets' in data configuration")
    
    # Validate model section
    if 'backbone' not in config['model']:
        raise ValueError("Missing 'backbone' in model configuration")
    
    # Validate training section
    if 'optimizer' not in config['training']:
        raise ValueError("Missing 'optimizer' in training configuration")
    
    logger.info("Configuration validation passed")
    return True


def update_config(config: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    """
    Update configuration with new values (nested update).
    
    Args:
        config: Original configuration dictionary
        updates: Dictionary with updates
        
    Returns:
        Updated configuration
    """
    def nested_update(d, u):
        for k, v in u.items():
            if isinstance(v, dict) and k in d:
                d[k] = nested_update(d.get(k, {}), v)
            else:
                d[k] = v
        return d
    
    return nested_update(config.copy(), updates)


def get_config_value(config: Dict[str, Any], key_path: str, default: Any = None) -> Any:
    """
    Get configuration value using dot notation path.
    
    Args:
        config: Configuration dictionary
        key_path: Dot-separated path (e.g., 'model.backbone')
        default: Default value if path not found
        
    Returns:
        Configuration value or default
    """
    keys = key_path.split('.')
    value = config
    
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return default
    
    return value

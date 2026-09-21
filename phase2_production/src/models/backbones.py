"""
VERA Phase 2 - CNN Backbones

Flexible backbone selection supporting ResNet, EfficientNet, and others.
Handles modification for 4-channel input (RGB + Vessel).
"""

import torch
import torch.nn as nn
import timm
from typing import Optional
import logging

logger = logging.getLogger(__name__)


def modify_first_conv_for_4channel(
    model: nn.Module,
    backbone_name: str
) -> nn.Module:
    """
    Modify first convolutional layer to accept 4 input channels.
    
    Adds an extra channel for vessel map while preserving pretrained
    RGB weights by initializing the vessel channel as the mean of RGB channels.
    
    Args:
        model: Pretrained model with 3-channel input
        backbone_name: Name of backbone architecture
        
    Returns:
        Modified model with 4-channel input
    """
    # Locate first conv layer based on architecture
    if 'resnet' in backbone_name.lower():
        first_conv = model.conv1
        first_conv_name = 'conv1'
    elif 'efficientnet' in backbone_name.lower():
        first_conv = model.conv_stem
        first_conv_name = 'conv_stem'
    elif 'densenet' in backbone_name.lower():
        first_conv = model.features.conv0
        first_conv_name = 'features.conv0'
    else:
        # Try to find first conv layer generically
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d) and module.in_channels == 3:
                first_conv = module
                first_conv_name = name
                break
        else:
            raise ValueError(f"Could not find first conv layer in {backbone_name}")
    
    # Get current weights
    old_weight = first_conv.weight.data  # (out_channels, 3, H, W)
    
    # Create new conv layer with 4 input channels
    new_conv = nn.Conv2d(
        in_channels=4,
        out_channels=first_conv.out_channels,
        kernel_size=first_conv.kernel_size,
        stride=first_conv.stride,
        padding=first_conv.padding,
        bias=(first_conv.bias is not None)
    )
    
    # Initialize weights
    with torch.no_grad():
        # Copy RGB weights
        new_conv.weight[:, :3, :, :] = old_weight
        
        # Initialize vessel channel as mean of RGB channels
        new_conv.weight[:, 3:4, :, :] = old_weight.mean(dim=1, keepdim=True)
        
        # Copy bias if exists
        if first_conv.bias is not None:
            new_conv.bias = first_conv.bias
    
    # Replace first conv layer
    if 'resnet' in backbone_name.lower():
        model.conv1 = new_conv
    elif 'efficientnet' in backbone_name.lower():
        model.conv_stem = new_conv
    elif 'densenet' in backbone_name.lower():
        model.features.conv0 = new_conv
    else:
        # Generic replacement
        parent_name = '.'.join(first_conv_name.split('.')[:-1])
        child_name = first_conv_name.split('.')[-1]
        if parent_name:
            parent = model
            for part in parent_name.split('.'):
                parent = getattr(parent, part)
            setattr(parent, child_name, new_conv)
        else:
            setattr(model, child_name, new_conv)
    
    logger.info(f"Modified {backbone_name} first conv from 3 to 4 input channels")
    
    return model


def create_backbone(
    backbone_name: str,
    pretrained: bool = True,
    in_channels: int = 3,
    num_classes: int = 0,
    drop_rate: float = 0.0,
    drop_path_rate: float = 0.0
) -> nn.Module:
    """
    Create CNN backbone using timm library.
    
    Args:
        backbone_name: Name of backbone (resnet18, resnet50, efficientnet_b0, etc.)
        pretrained: Load pretrained ImageNet weights
        in_channels: Number of input channels (3 for RGB, 4 for RGB+V)
        num_classes: Number of output classes (0 for feature extraction only)
        drop_rate: Dropout rate
        drop_path_rate: DropPath/StochasticDepth rate
        
    Returns:
        CNN backbone model
    """
    # Create model using timm
    model = timm.create_model(
        backbone_name,
        pretrained=pretrained,
        num_classes=num_classes,
        drop_rate=drop_rate,
        drop_path_rate=drop_path_rate
    )
    
    # Modify for 4-channel input if needed
    if in_channels == 4:
        model = modify_first_conv_for_4channel(model, backbone_name)
    elif in_channels == 1:
        model = modify_first_conv_for_1channel(model, backbone_name)
    elif in_channels != 3:
        raise ValueError(f"Unsupported number of input channels: {in_channels}")
    
    logger.info(f"Created {backbone_name} backbone: pretrained={pretrained}, "
               f"in_channels={in_channels}, num_classes={num_classes}")
    
    return model


def modify_first_conv_for_1channel(
    model: nn.Module,
    backbone_name: str
) -> nn.Module:
    """
    Modify first convolutional layer for single-channel input (vessel maps).
    
    Args:
        model: Pretrained model with 3-channel input
        backbone_name: Name of backbone architecture
        
    Returns:
        Modified model with 1-channel input
    """
    # Locate first conv layer
    if 'resnet' in backbone_name.lower():
        first_conv = model.conv1
    elif 'efficientnet' in backbone_name.lower():
        first_conv = model.conv_stem
    elif 'densenet' in backbone_name.lower():
        first_conv = model.features.conv0
    else:
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d) and module.in_channels == 3:
                first_conv = module
                break
        else:
            raise ValueError(f"Could not find first conv layer in {backbone_name}")
    
    # Get current weights
    old_weight = first_conv.weight.data  # (out_channels, 3, H, W)
    
    # Create new conv layer with 1 input channel
    new_conv = nn.Conv2d(
        in_channels=1,
        out_channels=first_conv.out_channels,
        kernel_size=first_conv.kernel_size,
        stride=first_conv.stride,
        padding=first_conv.padding,
        bias=(first_conv.bias is not None)
    )
    
    # Initialize as mean of RGB channels
    with torch.no_grad():
        new_conv.weight[:, 0:1, :, :] = old_weight.mean(dim=1, keepdim=True)
        
        if first_conv.bias is not None:
            new_conv.bias = first_conv.bias
    
    # Replace
    if 'resnet' in backbone_name.lower():
        model.conv1 = new_conv
    elif 'efficientnet' in backbone_name.lower():
        model.conv_stem = new_conv
    elif 'densenet' in backbone_name.lower():
        model.features.conv0 = new_conv
    
    logger.info(f"Modified {backbone_name} first conv from 3 to 1 input channel")
    
    return model


def get_backbone_output_features(
    backbone: nn.Module,
    input_size: int = 224
) -> int:
    """
    Get the number of output features from a backbone.
    
    Args:
        backbone: Backbone model
        input_size: Input image size
        
    Returns:
        Number of output features
    """
    # Try to get from model attributes
    if hasattr(backbone, 'num_features'):
        return backbone.num_features
    
    if hasattr(backbone, 'fc') and hasattr(backbone.fc, 'in_features'):
        return backbone.fc.in_features
    
    if hasattr(backbone, 'classifier') and hasattr(backbone.classifier, 'in_features'):
        return backbone.classifier.in_features
    
    # Fallback: forward pass with dummy input
    with torch.no_grad():
        dummy_input = torch.randn(1, 3, input_size, input_size)
        try:
            features = backbone(dummy_input)
            if isinstance(features, tuple):
                features = features[-1]
            
            # Apply global pooling if spatial
            if len(features.shape) == 4:
                features = torch.nn.functional.adaptive_avg_pool2d(features, 1)
                features = features.flatten(1)
            
            return features.shape[1]
        except Exception as e:
            logger.error(f"Could not determine backbone output features: {e}")
            raise


class BackboneWrapper(nn.Module):
    """
    Wrapper for backbones to provide consistent interface.
    
    Handles feature extraction with optional pooling and ensures
    consistent output format.
    """
    
    def __init__(
        self,
        backbone: nn.Module,
        return_features_only: bool = True,
        global_pool: bool = False
    ):
        """
        Initialize backbone wrapper.
        
        Args:
            backbone: CNN backbone
            return_features_only: Return only features (remove classifier)
            global_pool: Apply global pooling to features
        """
        super(BackboneWrapper, self).__init__()
        
        self.backbone = backbone
        self.return_features_only = return_features_only
        self.global_pool = global_pool
        
        # Remove classifier if needed
        if return_features_only:
            if hasattr(backbone, 'fc'):
                backbone.fc = nn.Identity()
            elif hasattr(backbone, 'classifier'):
                backbone.classifier = nn.Identity()
            elif hasattr(backbone, 'head'):
                backbone.head = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        features = self.backbone(x)
        
        # Handle different output formats
        if isinstance(features, tuple):
            features = features[-1]
        
        # Apply global pooling if requested and features are spatial
        if self.global_pool and len(features.shape) == 4:
            features = torch.nn.functional.adaptive_avg_pool2d(features, 1)
            features = features.flatten(1)
        
        return features


def list_available_backbones() -> list:
    """
    List all available backbone architectures from timm.
    
    Returns:
        List of backbone names
    """
    return timm.list_models(pretrained=True)


def get_recommended_backbones() -> dict:
    """
    Get recommended backbones for different use cases.
    
    Returns:
        Dictionary of use cases and recommended backbones
    """
    return {
        'lightweight': ['resnet18', 'efficientnet_b0', 'mobilenetv3_small_100'],
        'balanced': ['resnet34', 'resnet50', 'efficientnet_b1', 'efficientnet_b2'],
        'high_capacity': ['resnet101', 'efficientnet_b3', 'efficientnet_b4', 'densenet121'],
        'best_accuracy': ['efficientnet_b3', 'efficientnet_b4', 'resnet50', 'resnet101']
    }

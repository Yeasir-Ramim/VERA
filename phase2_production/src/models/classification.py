"""
VERA Phase 2 - DR Classification Models

Complete DR classification models combining backbones and fusion strategies.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional
import logging

from .backbones import create_backbone, BackboneWrapper
from .fusion import create_fusion_model, EarlyFusion, DualBranchFusion, AttentionGatedFusion

logger = logging.getLogger(__name__)


class DRClassifier(nn.Module):
    """
    Complete Diabetic Retinopathy classification model.
    
    Integrates backbone architecture, fusion strategy, and classification head.
    """
    
    def __init__(
        self,
        backbone_name: str = 'efficientnet_b3',
        fusion_strategy: str = 'attention_gated',
        num_classes: int = 5,
        pretrained: bool = True,
        use_vessel_channel: bool = True,
        dropout: float = 0.3,
        **fusion_kwargs
    ):
        """
        Initialize DR classifier.
        
        Args:
            backbone_name: Name of CNN backbone
            fusion_strategy: Fusion strategy ('early_fusion', 'dual_branch', 'attention_gated')
            num_classes: Number of DR severity classes
            pretrained: Use pretrained ImageNet weights
            use_vessel_channel: Include vessel channel
            dropout: Dropout probability
            **fusion_kwargs: Additional arguments for fusion model
        """
        super(DRClassifier, self).__init__()
        
        self.backbone_name = backbone_name
        self.fusion_strategy = fusion_strategy
        self.num_classes = num_classes
        self.use_vessel_channel = use_vessel_channel
        
        # Determine input channels
        if fusion_strategy == 'early_fusion' and use_vessel_channel:
            in_channels = 4  # RGB + Vessel
        else:
            in_channels = 3  # RGB only for other strategies
        
        # Create backbone(s)
        if fusion_strategy == 'dual_branch' and use_vessel_channel:
            # Separate backbones for RGB and vessel
            self.backbone_rgb = create_backbone(
                backbone_name=backbone_name,
                pretrained=pretrained,
                in_channels=3,
                num_classes=0  # Feature extraction only
            )
            
            self.backbone_vessel = create_backbone(
                backbone_name=backbone_name,
                pretrained=pretrained,
                in_channels=1,
                num_classes=0
            )
            
            # Create fusion model
            config = {
                'model': {
                    'fusion_strategy': fusion_strategy,
                    'num_classes': num_classes,
                    'dropout': dropout,
                    **fusion_kwargs
                }
            }
            
            self.model = create_fusion_model(
                config=config,
                backbone_rgb=self.backbone_rgb,
                backbone_vessel=self.backbone_vessel
            )
        
        else:
            # Single backbone
            self.backbone = create_backbone(
                backbone_name=backbone_name,
                pretrained=pretrained,
                in_channels=in_channels,
                num_classes=0
            )
            
            # Wrap backbone
            self.backbone = BackboneWrapper(
                backbone=self.backbone,
                return_features_only=True,
                global_pool=False
            )
            
            # Create fusion model
            config = {
                'model': {
                    'fusion_strategy': fusion_strategy,
                    'num_classes': num_classes,
                    'dropout': dropout,
                    **fusion_kwargs
                }
            }
            
            self.model = create_fusion_model(
                config=config,
                backbone_rgb=self.backbone,
                backbone_vessel=None
            )
        
        logger.info(f"Initialized DRClassifier: backbone={backbone_name}, "
                   f"fusion={fusion_strategy}, classes={num_classes}")
    
    def forward(
        self,
        rgb: torch.Tensor,
        vessel: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            rgb: RGB image tensor (B, 3, H, W)
            vessel: Optional vessel map tensor (B, 1, H, W)
            
        Returns:
            Classification logits (B, num_classes)
        """
        if not self.use_vessel_channel or vessel is None:
            # Create dummy vessel map if not using vessel channel
            vessel = torch.zeros(
                rgb.shape[0], 1, rgb.shape[2], rgb.shape[3],
                device=rgb.device, dtype=rgb.dtype
            )
        
        return self.model(rgb, vessel)
    
    def get_features(
        self,
        rgb: torch.Tensor,
        vessel: Optional[torch.Tensor] = None
    ):
        """Extract features for explainability."""
        if not self.use_vessel_channel or vessel is None:
            vessel = torch.zeros(
                rgb.shape[0], 1, rgb.shape[2], rgb.shape[3],
                device=rgb.device, dtype=rgb.dtype
            )
        
        if hasattr(self.model, 'get_features'):
            return self.model.get_features(rgb, vessel)
        else:
            # Fallback
            return None


def create_dr_classifier(config: Dict) -> DRClassifier:
    """
    Factory function to create DR classifier from configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        DRClassifier instance
    """
    model_config = config.get('model', {})
    
    # Extract configuration
    backbone_name = model_config.get('backbone', 'efficientnet_b3')
    fusion_strategy = model_config.get('fusion_strategy', 'attention_gated')
    num_classes = model_config.get('num_classes', 5)
    pretrained = model_config.get('pretrained', True)
    use_vessel_channel = model_config.get('use_vessel_channel', True)
    dropout = model_config.get('dropout', 0.3)
    
    # Fusion-specific kwargs
    fusion_kwargs = {}
    
    if fusion_strategy == 'dual_branch':
        fusion_kwargs['fusion_method'] = model_config.get('dual_branch_fusion', 'concat')
    
    elif fusion_strategy == 'attention_gated':
        attention_config = model_config.get('attention', {})
        fusion_kwargs.update({
            'num_attention_layers': attention_config.get('num_attention_layers', 2),
            'reduction_ratio': attention_config.get('reduction_ratio', 16),
            'use_spatial_attention': attention_config.get('use_spatial_attention', True),
            'use_channel_attention': attention_config.get('use_channel_attention', True),
        })
    
    # Create model
    model = DRClassifier(
        backbone_name=backbone_name,
        fusion_strategy=fusion_strategy,
        num_classes=num_classes,
        pretrained=pretrained,
        use_vessel_channel=use_vessel_channel,
        dropout=dropout,
        **fusion_kwargs
    )
    
    return model


def load_dr_classifier(
    checkpoint_path: str,
    config: Optional[Dict] = None,
    device: str = 'cuda'
) -> DRClassifier:
    """
    Load DR classifier from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        config: Optional configuration (will try to load from checkpoint)
        device: Device to load model on
        
    Returns:
        Loaded model
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract configuration
    if config is None:
        if 'config' in checkpoint:
            config = checkpoint['config']
        else:
            raise ValueError("No configuration provided and none found in checkpoint")
    
    # Create model
    model = create_dr_classifier(config)
    
    # Load weights
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    logger.info(f"Loaded DR classifier from {checkpoint_path}")
    
    return model


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """
    Count model parameters.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with parameter counts
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total': total_params,
        'trainable': trainable_params,
        'non_trainable': total_params - trainable_params
    }

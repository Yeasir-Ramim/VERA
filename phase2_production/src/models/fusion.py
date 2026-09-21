"""
VERA Phase 2 - Fusion Architectures

Multiple strategies for fusing RGB fundus images with vessel segmentation maps:
1. Early Fusion: Simple channel concatenation [R, G, B, V]
2. Dual-Branch: Separate encoders with late fusion
3. Attention-Gated: Spatial attention mechanism for vessel-guided features
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class EarlyFusion(nn.Module):
    """
    Early Fusion Strategy: Simple channel concatenation.
    
    Stacks RGB and vessel channels [R, G, B, V] and processes
    with a single CNN backbone. Simplest approach but allows
    the network to learn joint representations from the start.
    """
    
    def __init__(
        self,
        backbone: nn.Module,
        num_classes: int = 5,
        dropout: float = 0.3
    ):
        """
        Initialize early fusion model.
        
        Args:
            backbone: CNN backbone (modified for 4 input channels)
            num_classes: Number of output classes
            dropout: Dropout probability
        """
        super(EarlyFusion, self).__init__()
        
        self.backbone = backbone
        self.num_classes = num_classes
        
        # Get backbone output features
        self.feature_dim = self._get_backbone_features()
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.BatchNorm1d(self.feature_dim),
            nn.Dropout(dropout),
            nn.Linear(self.feature_dim, num_classes)
        )
        
        logger.info(f"Initialized EarlyFusion with {self.feature_dim} features")
    
    def _get_backbone_features(self) -> int:
        """Determine backbone output feature dimension."""
        with torch.no_grad():
            dummy_input = torch.randn(1, 4, 224, 224)
            features = self.backbone(dummy_input)
            if isinstance(features, tuple):
                features = features[-1]
            return features.shape[1]
    
    def forward(
        self,
        rgb: torch.Tensor,
        vessel: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            rgb: RGB image tensor (B, 3, H, W)
            vessel: Vessel map tensor (B, 1, H, W)
            
        Returns:
            Classification logits (B, num_classes)
        """
        # Concatenate RGB and vessel channels
        x = torch.cat([rgb, vessel], dim=1)  # (B, 4, H, W)
        
        # Extract features
        features = self.backbone(x)
        
        # Handle different backbone outputs
        if isinstance(features, tuple):
            features = features[-1]
        
        # Classify
        logits = self.classifier(features)
        
        return logits
    
    def get_features(
        self,
        rgb: torch.Tensor,
        vessel: torch.Tensor
    ) -> torch.Tensor:
        """Extract features without classification (for explainability)."""
        x = torch.cat([rgb, vessel], dim=1)
        features = self.backbone(x)
        
        if isinstance(features, tuple):
            features = features[-1]
        
        return features


class DualBranchFusion(nn.Module):
    """
    Dual-Branch Fusion Strategy: Separate encoders with late fusion.
    
    Processes RGB and vessel inputs through separate CNN backbones,
    then fuses their features before classification. Allows specialized
    feature learning for each modality.
    """
    
    def __init__(
        self,
        rgb_backbone: nn.Module,
        vessel_backbone: nn.Module,
        num_classes: int = 5,
        fusion_method: str = 'concat',
        dropout: float = 0.3
    ):
        """
        Initialize dual-branch fusion model.
        
        Args:
            rgb_backbone: CNN backbone for RGB images
            vessel_backbone: CNN backbone for vessel maps
            num_classes: Number of output classes
            fusion_method: Fusion method ('concat', 'add', 'multiply', 'attention')
            dropout: Dropout probability
        """
        super(DualBranchFusion, self).__init__()
        
        self.rgb_backbone = rgb_backbone
        self.vessel_backbone = vessel_backbone
        self.fusion_method = fusion_method
        self.num_classes = num_classes
        
        # Get feature dimensions
        self.rgb_feature_dim = self._get_backbone_features(rgb_backbone, 3)
        self.vessel_feature_dim = self._get_backbone_features(vessel_backbone, 1)
        
        # Fusion layer
        if fusion_method == 'concat':
            self.fusion_dim = self.rgb_feature_dim + self.vessel_feature_dim
            self.fusion_layer = None
        elif fusion_method == 'add':
            # Ensure same dimensions
            assert self.rgb_feature_dim == self.vessel_feature_dim, \
                "RGB and vessel features must have same dimension for addition"
            self.fusion_dim = self.rgb_feature_dim
            self.fusion_layer = None
        elif fusion_method == 'multiply':
            assert self.rgb_feature_dim == self.vessel_feature_dim, \
                "RGB and vessel features must have same dimension for multiplication"
            self.fusion_dim = self.rgb_feature_dim
            self.fusion_layer = None
        elif fusion_method == 'attention':
            # Cross-attention fusion
            self.fusion_dim = self.rgb_feature_dim
            self.fusion_layer = CrossModalAttention(
                self.rgb_feature_dim,
                self.vessel_feature_dim
            )
        else:
            raise ValueError(f"Unknown fusion method: {fusion_method}")
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(self.fusion_dim),
            nn.Dropout(dropout),
            nn.Linear(self.fusion_dim, num_classes)
        )
        
        logger.info(f"Initialized DualBranchFusion with fusion method: {fusion_method}, "
                   f"fusion_dim: {self.fusion_dim}")
    
    def _get_backbone_features(self, backbone: nn.Module, in_channels: int) -> int:
        """Determine backbone output feature dimension."""
        with torch.no_grad():
            dummy_input = torch.randn(1, in_channels, 224, 224)
            features = backbone(dummy_input)
            if isinstance(features, tuple):
                features = features[-1]
            
            # Apply global pooling if spatial
            if len(features.shape) == 4:
                features = F.adaptive_avg_pool2d(features, 1).flatten(1)
            
            return features.shape[1]
    
    def forward(
        self,
        rgb: torch.Tensor,
        vessel: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            rgb: RGB image tensor (B, 3, H, W)
            vessel: Vessel map tensor (B, 1, H, W)
            
        Returns:
            Classification logits (B, num_classes)
        """
        # Extract RGB features
        rgb_features = self.rgb_backbone(rgb)
        if isinstance(rgb_features, tuple):
            rgb_features = rgb_features[-1]
        
        # Extract vessel features
        vessel_features = self.vessel_backbone(vessel)
        if isinstance(vessel_features, tuple):
            vessel_features = vessel_features[-1]
        
        # Global pooling if needed
        if len(rgb_features.shape) == 4:
            rgb_features = F.adaptive_avg_pool2d(rgb_features, 1).flatten(1)
        if len(vessel_features.shape) == 4:
            vessel_features = F.adaptive_avg_pool2d(vessel_features, 1).flatten(1)
        
        # Fuse features
        if self.fusion_method == 'concat':
            fused = torch.cat([rgb_features, vessel_features], dim=1)
        elif self.fusion_method == 'add':
            fused = rgb_features + vessel_features
        elif self.fusion_method == 'multiply':
            fused = rgb_features * vessel_features
        elif self.fusion_method == 'attention':
            fused = self.fusion_layer(rgb_features, vessel_features)
        
        # Classify
        logits = self.classifier(fused)
        
        return logits
    
    def get_features(
        self,
        rgb: torch.Tensor,
        vessel: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract features from both branches and fused."""
        rgb_features = self.rgb_backbone(rgb)
        if isinstance(rgb_features, tuple):
            rgb_features = rgb_features[-1]
        
        vessel_features = self.vessel_backbone(vessel)
        if isinstance(vessel_features, tuple):
            vessel_features = vessel_features[-1]
        
        if len(rgb_features.shape) == 4:
            rgb_features_pooled = F.adaptive_avg_pool2d(rgb_features, 1).flatten(1)
        else:
            rgb_features_pooled = rgb_features
        
        if len(vessel_features.shape) == 4:
            vessel_features_pooled = F.adaptive_avg_pool2d(vessel_features, 1).flatten(1)
        else:
            vessel_features_pooled = vessel_features
        
        if self.fusion_method == 'concat':
            fused = torch.cat([rgb_features_pooled, vessel_features_pooled], dim=1)
        elif self.fusion_method == 'add':
            fused = rgb_features_pooled + vessel_features_pooled
        elif self.fusion_method == 'multiply':
            fused = rgb_features_pooled * vessel_features_pooled
        elif self.fusion_method == 'attention':
            fused = self.fusion_layer(rgb_features_pooled, vessel_features_pooled)
        
        return rgb_features, vessel_features, fused


class AttentionGatedFusion(nn.Module):
    """
    Attention-Gated Fusion Strategy: Spatial attention for vessel-guided features.
    
    Uses vessel maps to generate spatial attention masks that modulate
    RGB features. Allows the network to focus on vessel-relevant regions
    while maintaining a single backbone.
    """
    
    def __init__(
        self,
        backbone: nn.Module,
        num_classes: int = 5,
        num_attention_layers: int = 2,
        reduction_ratio: int = 16,
        use_spatial_attention: bool = True,
        use_channel_attention: bool = True,
        dropout: float = 0.3
    ):
        """
        Initialize attention-gated fusion model.
        
        Args:
            backbone: CNN backbone for feature extraction
            num_classes: Number of output classes
            num_attention_layers: Number of attention modules
            reduction_ratio: Channel reduction ratio for attention
            use_spatial_attention: Use spatial attention
            use_channel_attention: Use channel attention
            dropout: Dropout probability
        """
        super(AttentionGatedFusion, self).__init__()
        
        self.backbone = backbone
        self.num_classes = num_classes
        
        # Vessel processing branch
        self.vessel_encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        
        # Attention modules
        self.attention_modules = nn.ModuleList([
            SpatialChannelAttention(
                channels=256,
                reduction_ratio=reduction_ratio,
                use_spatial=use_spatial_attention,
                use_channel=use_channel_attention
            )
            for _ in range(num_attention_layers)
        ])
        
        # Get backbone features
        self.feature_dim = self._get_backbone_features()
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.BatchNorm1d(self.feature_dim),
            nn.Dropout(dropout),
            nn.Linear(self.feature_dim, num_classes)
        )
        
        logger.info(f"Initialized AttentionGatedFusion with {num_attention_layers} attention layers")
    
    def _get_backbone_features(self) -> int:
        """Determine backbone output feature dimension."""
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            features = self.backbone(dummy_input)
            if isinstance(features, tuple):
                features = features[-1]
            return features.shape[1]
    
    def forward(
        self,
        rgb: torch.Tensor,
        vessel: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass with attention gating.
        
        Args:
            rgb: RGB image tensor (B, 3, H, W)
            vessel: Vessel map tensor (B, 1, H, W)
            
        Returns:
            Classification logits (B, num_classes)
        """
        # Process vessel map to get attention features
        vessel_features = self.vessel_encoder(vessel)  # (B, 256, H', W')
        
        # Apply attention modules
        attended_vessel = vessel_features
        for attention_module in self.attention_modules:
            attended_vessel = attention_module(attended_vessel)
        
        # Extract RGB features
        rgb_features = self.backbone(rgb)
        if isinstance(rgb_features, tuple):
            rgb_features = rgb_features[-1]
        
        # Resize vessel attention to match RGB features if needed
        if attended_vessel.shape[2:] != rgb_features.shape[2:]:
            attended_vessel = F.interpolate(
                attended_vessel,
                size=rgb_features.shape[2:],
                mode='bilinear',
                align_corners=True
            )
        
        # Apply vessel-guided attention to RGB features
        if attended_vessel.shape[1] != rgb_features.shape[1]:
            # Project to same channel dimension
            projection = nn.Conv2d(
                attended_vessel.shape[1],
                rgb_features.shape[1],
                kernel_size=1
            ).to(rgb_features.device)
            attended_vessel = projection(attended_vessel)
        
        # Modulate RGB features with vessel attention
        attended_features = rgb_features * torch.sigmoid(attended_vessel)
        
        # Combine original and attended features
        fused_features = rgb_features + attended_features
        
        # Classify
        logits = self.classifier(fused_features)
        
        return logits
    
    def get_attention_maps(
        self,
        rgb: torch.Tensor,
        vessel: torch.Tensor
    ) -> torch.Tensor:
        """Get attention maps for visualization."""
        vessel_features = self.vessel_encoder(vessel)
        
        attended_vessel = vessel_features
        for attention_module in self.attention_modules:
            attended_vessel = attention_module(attended_vessel)
        
        return torch.sigmoid(attended_vessel)


class SpatialChannelAttention(nn.Module):
    """
    Combined spatial and channel attention module.
    
    Applies both spatial and channel-wise attention to enhance
    important features.
    """
    
    def __init__(
        self,
        channels: int,
        reduction_ratio: int = 16,
        use_spatial: bool = True,
        use_channel: bool = True
    ):
        """
        Initialize attention module.
        
        Args:
            channels: Number of input channels
            reduction_ratio: Reduction ratio for channel attention
            use_spatial: Use spatial attention
            use_channel: Use channel attention
        """
        super(SpatialChannelAttention, self).__init__()
        
        self.use_spatial = use_spatial
        self.use_channel = use_channel
        
        # Channel attention
        if use_channel:
            self.channel_attention = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(channels, channels // reduction_ratio, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels // reduction_ratio, channels, 1),
                nn.Sigmoid()
            )
        
        # Spatial attention
        if use_spatial:
            self.spatial_attention = nn.Sequential(
                nn.Conv2d(channels, channels // 2, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels // 2, 1, kernel_size=1),
                nn.Sigmoid()
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply attention."""
        identity = x
        
        # Channel attention
        if self.use_channel:
            channel_att = self.channel_attention(x)
            x = x * channel_att
        
        # Spatial attention
        if self.use_spatial:
            spatial_att = self.spatial_attention(x)
            x = x * spatial_att
        
        return x + identity


class CrossModalAttention(nn.Module):
    """
    Cross-modal attention for fusing RGB and vessel features.
    
    Allows RGB features to attend to vessel features and vice versa.
    """
    
    def __init__(
        self,
        rgb_dim: int,
        vessel_dim: int,
        hidden_dim: Optional[int] = None
    ):
        """
        Initialize cross-modal attention.
        
        Args:
            rgb_dim: RGB feature dimension
            vessel_dim: Vessel feature dimension
            hidden_dim: Hidden dimension for attention
        """
        super(CrossModalAttention, self).__init__()
        
        if hidden_dim is None:
            hidden_dim = min(rgb_dim, vessel_dim)
        
        self.query = nn.Linear(rgb_dim, hidden_dim)
        self.key = nn.Linear(vessel_dim, hidden_dim)
        self.value = nn.Linear(vessel_dim, rgb_dim)
        
        self.scale = hidden_dim ** -0.5
    
    def forward(
        self,
        rgb_features: torch.Tensor,
        vessel_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply cross-modal attention.
        
        Args:
            rgb_features: RGB features (B, D_rgb)
            vessel_features: Vessel features (B, D_vessel)
            
        Returns:
            Attended RGB features (B, D_rgb)
        """
        # Compute attention
        q = self.query(rgb_features)  # (B, hidden_dim)
        k = self.key(vessel_features)  # (B, hidden_dim)
        v = self.value(vessel_features)  # (B, D_rgb)
        
        # Attention scores
        attention_scores = torch.sum(q * k, dim=1, keepdim=True) * self.scale  # (B, 1)
        attention_weights = torch.sigmoid(attention_scores)
        
        # Apply attention
        attended = rgb_features + attention_weights * v
        
        return attended


def create_fusion_model(
    config: dict,
    backbone_rgb: nn.Module,
    backbone_vessel: Optional[nn.Module] = None
) -> nn.Module:
    """
    Factory function to create fusion model from configuration.
    
    Args:
        config: Configuration dictionary
        backbone_rgb: RGB backbone network
        backbone_vessel: Optional separate vessel backbone (for dual-branch)
        
    Returns:
        Fusion model instance
    """
    model_config = config.get('model', {})
    fusion_strategy = model_config.get('fusion_strategy', 'early_fusion')
    num_classes = model_config.get('num_classes', 5)
    dropout = model_config.get('dropout', 0.3)
    
    if fusion_strategy == 'early_fusion':
        model = EarlyFusion(
            backbone=backbone_rgb,
            num_classes=num_classes,
            dropout=dropout
        )
    
    elif fusion_strategy == 'dual_branch':
        if backbone_vessel is None:
            # Use same architecture as RGB but with 1 input channel
            backbone_vessel = backbone_rgb  # Should be modified for 1 channel
        
        fusion_method = model_config.get('dual_branch_fusion', 'concat')
        
        model = DualBranchFusion(
            rgb_backbone=backbone_rgb,
            vessel_backbone=backbone_vessel,
            num_classes=num_classes,
            fusion_method=fusion_method,
            dropout=dropout
        )
    
    elif fusion_strategy == 'attention_gated':
        attention_config = model_config.get('attention', {})
        
        model = AttentionGatedFusion(
            backbone=backbone_rgb,
            num_classes=num_classes,
            num_attention_layers=attention_config.get('num_attention_layers', 2),
            reduction_ratio=attention_config.get('reduction_ratio', 16),
            use_spatial_attention=attention_config.get('use_spatial_attention', True),
            use_channel_attention=attention_config.get('use_channel_attention', True),
            dropout=dropout
        )
    
    else:
        raise ValueError(f"Unknown fusion strategy: {fusion_strategy}")
    
    logger.info(f"Created fusion model with strategy: {fusion_strategy}")
    
    return model

"""
Model architectures for Diabetic Retinopathy Classification.
Implements:
1. Baseline CNN (3-Channel RGB input).
2. Early Fusion / Vessel-Aware CNN (4-Channel RGB + Vessel Probability Map input).
3. Dual-Branch Late Concatenation Fusion CNN (RGB Backbone + Vascular Backbone).
4. Spatial Attention-Gated CNN (Vessel map dynamically gates intermediate features).

Supports ResNet-18, ResNet-50, EfficientNet-B0, and EfficientNet-B3 backbones with weight preservation.
"""

from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
import torchvision.models as models


def adapt_first_conv_to_4channels(orig_conv: nn.Conv2d) -> nn.Conv2d:
    """
    Creates a new Conv2d layer accepting 4 input channels instead of 3.
    Initializes channels 0-2 with original pretrained weights and channel 3 with the mean of channels 0-2.
    
    Args:
        orig_conv: The original 3-channel Conv2d layer from pretrained backbone.
        
    Returns:
        New Conv2d layer with in_channels=4 and properly initialized weights.
    """
    out_channels = orig_conv.out_channels
    kernel_size = orig_conv.kernel_size
    stride = orig_conv.stride
    padding = orig_conv.padding
    dilation = orig_conv.dilation
    groups = orig_conv.groups
    has_bias = orig_conv.bias is not None
    
    new_conv = nn.Conv2d(
        in_channels=4,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        bias=has_bias
    )
    
    with torch.no_grad():
        orig_weight = orig_conv.weight.data  # Shape: (out_channels, 3, k_h, k_w)
        # Copy RGB weights
        new_conv.weight[:, 0:3, :, :] = orig_weight
        # 4th channel is the average of RGB channels
        vessel_weight = torch.mean(orig_weight, dim=1, keepdim=True)  # Shape: (out_channels, 1, k_h, k_w)
        new_conv.weight[:, 3:4, :, :] = vessel_weight
        
        if has_bias:
            new_conv.bias.data = orig_conv.bias.data
            
    return new_conv


def adapt_first_conv_to_1channel(orig_conv: nn.Conv2d) -> nn.Conv2d:
    """
    Creates a new Conv2d layer accepting 1 input channel (for vessel maps).
    Initializes weights as the average of the 3 pretrained RGB channels.
    """
    out_channels = orig_conv.out_channels
    kernel_size = orig_conv.kernel_size
    stride = orig_conv.stride
    padding = orig_conv.padding
    dilation = orig_conv.dilation
    groups = orig_conv.groups
    has_bias = orig_conv.bias is not None
    
    new_conv = nn.Conv2d(
        in_channels=1,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        bias=has_bias
    )
    
    with torch.no_grad():
        orig_weight = orig_conv.weight.data
        new_conv.weight[:, 0:1, :, :] = torch.mean(orig_weight, dim=1, keepdim=True)
        if has_bias:
            new_conv.bias.data = orig_conv.bias.data
            
    return new_conv


class DRClassifier(nn.Module):
    """
    Unified Diabetic Retinopathy Classifier supporting both 3-channel Baseline
    and 4-channel Vessel-Aware architectures (Early Fusion / Channel Stacking).
    """
    def __init__(
        self,
        backbone_name: str = "resnet18",
        num_classes: int = 5,
        is_4channel: bool = False,
        pretrained: bool = True,
        dropout_rate: float = 0.3
    ):
        super().__init__()
        self.backbone_name = backbone_name.lower()
        self.num_classes = num_classes
        self.is_4channel = is_4channel
        
        if self.backbone_name == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            self.backbone = models.resnet18(weights=weights)
            if is_4channel:
                self.backbone.conv1 = adapt_first_conv_to_4channels(self.backbone.conv1)
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Sequential(
                nn.Dropout(p=dropout_rate),
                nn.Linear(in_features, num_classes)
            )
            
        elif self.backbone_name == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            self.backbone = models.resnet50(weights=weights)
            if is_4channel:
                self.backbone.conv1 = adapt_first_conv_to_4channels(self.backbone.conv1)
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Sequential(
                nn.Dropout(p=dropout_rate),
                nn.Linear(in_features, num_classes)
            )
            
        elif self.backbone_name == "efficientnet_b0":
            weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            self.backbone = models.efficientnet_b0(weights=weights)
            if is_4channel:
                orig_first = self.backbone.features[0][0]
                self.backbone.features[0][0] = adapt_first_conv_to_4channels(orig_first)
            in_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Sequential(
                nn.Dropout(p=dropout_rate),
                nn.Linear(in_features, num_classes)
            )
            
        elif self.backbone_name == "efficientnet_b3":
            weights = models.EfficientNet_B3_Weights.DEFAULT if pretrained else None
            self.backbone = models.efficientnet_b3(weights=weights)
            if is_4channel:
                orig_first = self.backbone.features[0][0]
                self.backbone.features[0][0] = adapt_first_conv_to_4channels(orig_first)
            in_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Sequential(
                nn.Dropout(p=dropout_rate),
                nn.Linear(in_features, num_classes)
            )
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}. Choose resnet18, resnet50, efficientnet_b0, or efficientnet_b3.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def get_target_layer_for_gradcam(self) -> nn.Module:
        """
        Returns the final convolutional feature layer for Grad-CAM visualization.
        """
        if "resnet" in self.backbone_name:
            layer4 = self.backbone.layer4
            last_block = layer4[-1]
            return last_block.conv2 if hasattr(last_block, "conv2") else (last_block.conv3 if hasattr(last_block, "conv3") else last_block)
        elif "efficientnet" in self.backbone_name:
            return self.backbone.features[-1]
        raise ValueError(f"Grad-CAM target layer not configured for {self.backbone_name}")


class DualBranchDRClassifier(nn.Module):
    """
    Dual-Branch Retinopathy Classifier (Ablation 2: Late Concatenation Fusion).
    Branch 1: RGB Backbone processes tissue features (3 channels).
    Branch 2: Vascular Backbone processes vessel morphology (1 channel).
    Both feature representations are concatenated and classified.
    """
    def __init__(
        self,
        backbone_name: str = "resnet18",
        num_classes: int = 5,
        pretrained: bool = True,
        dropout_rate: float = 0.3
    ):
        super().__init__()
        self.backbone_name = backbone_name.lower()
        self.num_classes = num_classes
        
        if "resnet18" in self.backbone_name:
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            base_rgb = models.resnet18(weights=weights)
            rgb_dim = base_rgb.fc.in_features
            self.rgb_features = nn.Sequential(*list(base_rgb.children())[:-1])
            
            base_vessel = models.resnet18(weights=weights)
            base_vessel.conv1 = adapt_first_conv_to_1channel(base_vessel.conv1)
            vessel_dim = base_vessel.fc.in_features
            self.vessel_features = nn.Sequential(*list(base_vessel.children())[:-1])
            
        elif "resnet50" in self.backbone_name:
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            base_rgb = models.resnet50(weights=weights)
            rgb_dim = base_rgb.fc.in_features
            self.rgb_features = nn.Sequential(*list(base_rgb.children())[:-1])
            
            base_vessel = models.resnet50(weights=weights)
            base_vessel.conv1 = adapt_first_conv_to_1channel(base_vessel.conv1)
            vessel_dim = base_vessel.fc.in_features
            self.vessel_features = nn.Sequential(*list(base_vessel.children())[:-1])
            
        elif "efficientnet" in self.backbone_name:
            if "b3" in self.backbone_name:
                weights = models.EfficientNet_B3_Weights.DEFAULT if pretrained else None
                base_rgb = models.efficientnet_b3(weights=weights)
                base_vessel = models.efficientnet_b3(weights=weights)
            else:
                weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
                base_rgb = models.efficientnet_b0(weights=weights)
                base_vessel = models.efficientnet_b0(weights=weights)
                
            rgb_dim = base_rgb.classifier[1].in_features
            self.rgb_features = nn.Sequential(base_rgb.features, base_rgb.avgpool)
            
            orig_first = base_vessel.features[0][0]
            base_vessel.features[0][0] = adapt_first_conv_to_1channel(orig_first)
            vessel_dim = base_vessel.classifier[1].in_features
            self.vessel_features = nn.Sequential(base_vessel.features, base_vessel.avgpool)
        else:
            raise ValueError(f"Unsupported backbone for dual branch: {backbone_name}")
            
        total_dim = rgb_dim + vessel_dim
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(total_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(1) == 4:
            x_rgb = x[:, :3, :, :]
            x_vessel = x[:, 3:4, :, :]
        elif x.size(1) == 3:
            x_rgb = x
            x_vessel = torch.mean(x, dim=1, keepdim=True)
        else:
            raise ValueError(f"Expected 3 or 4 channels, got {x.size(1)}")
            
        feat_rgb = self.rgb_features(x_rgb)
        feat_vessel = self.vessel_features(x_vessel)
        fused = torch.cat([torch.flatten(feat_rgb, 1), torch.flatten(feat_vessel, 1)], dim=1)
        return self.classifier(fused)

    def get_target_layer_for_gradcam(self) -> nn.Module:
        if "resnet" in self.backbone_name:
            layer4 = self.rgb_features[7]
            last_block = layer4[-1]
            return last_block.conv2 if hasattr(last_block, "conv2") else (last_block.conv3 if hasattr(last_block, "conv3") else last_block)
        elif "efficientnet" in self.backbone_name:
            return self.rgb_features[0][-1]
        return self.rgb_features[-1]


class SpatialAttentionGatedDRClassifier(nn.Module):
    """
    Spatial Attention-Gated Classifier (Ablation 2: Spatial Attention Fusion).
    The retinal vascular map acts as a spatial soft-attention prior gating intermediate
    convolutional feature representations of the RGB backbone.
    """
    def __init__(
        self,
        backbone_name: str = "resnet18",
        num_classes: int = 5,
        pretrained: bool = True,
        dropout_rate: float = 0.3
    ):
        super().__init__()
        self.backbone_name = backbone_name.lower()
        self.num_classes = num_classes
        
        if self.backbone_name == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            base = models.resnet18(weights=weights)
            self.conv1 = base.conv1
            self.bn1 = base.bn1
            self.relu = base.relu
            self.maxpool = base.maxpool
            self.layer1 = base.layer1
            self.layer2 = base.layer2
            self.layer3 = base.layer3
            self.layer4 = base.layer4
            self.avgpool = base.avgpool
            in_features = base.fc.in_features
        elif self.backbone_name == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            base = models.resnet50(weights=weights)
            self.conv1 = base.conv1
            self.bn1 = base.bn1
            self.relu = base.relu
            self.maxpool = base.maxpool
            self.layer1 = base.layer1
            self.layer2 = base.layer2
            self.layer3 = base.layer3
            self.layer4 = base.layer4
            self.avgpool = base.avgpool
            in_features = base.fc.in_features
        elif "efficientnet" in self.backbone_name:
            if "b3" in self.backbone_name:
                weights = models.EfficientNet_B3_Weights.DEFAULT if pretrained else None
                base = models.efficientnet_b3(weights=weights)
            else:
                weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
                base = models.efficientnet_b0(weights=weights)
            self.features = base.features
            self.avgpool = base.avgpool
            in_features = base.classifier[1].in_features
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")
            
        self.attention_gate = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, num_classes)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(1) == 4:
            x_rgb = x[:, :3, :, :]
            x_vessel = x[:, 3:4, :, :]
        else:
            x_rgb = x
            x_vessel = torch.mean(x, dim=1, keepdim=True)
            
        if "resnet" in self.backbone_name:
            h = self.relu(self.bn1(self.conv1(x_rgb)))
            h = self.maxpool(h)
            h = self.layer1(h)
            h = self.layer2(h)
            h = self.layer3(h)
            feat = self.layer4(h)
        else:
            feat = self.features(x_rgb)
            
        # Spatial Attention Mask
        H_feat, W_feat = feat.size(2), feat.size(3)
        vessel_down = nn.functional.interpolate(x_vessel, size=(H_feat, W_feat), mode="bilinear", align_corners=False)
        attn_mask = self.attention_gate(vessel_down)
        
        # Modulate feature map: Soft residual gating
        attended_feat = feat * (1.0 + attn_mask)
        pooled = self.avgpool(attended_feat)
        return self.fc(pooled)

    def get_target_layer_for_gradcam(self) -> nn.Module:
        if "resnet" in self.backbone_name:
            last_block = self.layer4[-1]
            return last_block.conv2 if hasattr(last_block, "conv2") else (last_block.conv3 if hasattr(last_block, "conv3") else last_block)
        return self.features[-1]


def build_model(
    model_type: str = "vessel_aware",
    backbone: str = "resnet18",
    num_classes: int = 5,
    pretrained: bool = True,
    dropout_rate: float = 0.3
) -> nn.Module:
    """
    Factory function to construct classifiers across all Phase 2 ablation topologies:
    - 'baseline': 3-channel standard RGB model
    - 'vessel_aware' / 'early_fusion': 4-channel [R, G, B, V] stacked model
    - 'dual_branch': Separate RGB and Vessel CNN backbones with late feature concatenation
    - 'attention_gated': Spatial attention gating via vascular probability maps
    
    Backbones: 'resnet18', 'resnet50', 'efficientnet_b0', 'efficientnet_b3'
    """
    mtype = model_type.lower()
    if mtype in ["baseline", "3channel"]:
        return DRClassifier(
            backbone_name=backbone,
            num_classes=num_classes,
            is_4channel=False,
            pretrained=pretrained,
            dropout_rate=dropout_rate
        )
    elif mtype in ["vessel_aware", "4channel", "early_fusion", "channel_stacking"]:
        return DRClassifier(
            backbone_name=backbone,
            num_classes=num_classes,
            is_4channel=True,
            pretrained=pretrained,
            dropout_rate=dropout_rate
        )
    elif mtype in ["dual_branch", "late_fusion"]:
        return DualBranchDRClassifier(
            backbone_name=backbone,
            num_classes=num_classes,
            pretrained=pretrained,
            dropout_rate=dropout_rate
        )
    elif mtype in ["attention_gated", "spatial_attention"]:
        return SpatialAttentionGatedDRClassifier(
            backbone_name=backbone,
            num_classes=num_classes,
            pretrained=pretrained,
            dropout_rate=dropout_rate
        )
    else:
        raise ValueError(
            f"Unknown model_type: '{model_type}'. Choose from: 'baseline', 'vessel_aware', 'dual_branch', 'attention_gated'."
        )

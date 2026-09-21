"""
VERA Phase 2 - Models Module

Exports all model architectures including vessel segmentation and DR classification.
"""

from .vessel_segmentation import (
    UNetVesselSegmenter,
    SimpleUNet,
    VesselSegmentationLoss,
    create_vessel_segmenter,
    load_vessel_segmenter,
    train_vessel_segmenter,
    inference_vessel_segmenter
)

from .backbones import (
    create_backbone,
    modify_first_conv_for_4channel,
    modify_first_conv_for_1channel,
    get_backbone_output_features,
    BackboneWrapper,
    list_available_backbones,
    get_recommended_backbones
)

from .fusion import (
    EarlyFusion,
    DualBranchFusion,
    AttentionGatedFusion,
    SpatialChannelAttention,
    CrossModalAttention,
    create_fusion_model
)

from .classification import (
    DRClassifier,
    create_dr_classifier,
    load_dr_classifier,
    count_parameters
)

__all__ = [
    # Vessel Segmentation
    'UNetVesselSegmenter',
    'SimpleUNet',
    'VesselSegmentationLoss',
    'create_vessel_segmenter',
    'load_vessel_segmenter',
    'train_vessel_segmenter',
    'inference_vessel_segmenter',
    
    # Backbones
    'create_backbone',
    'modify_first_conv_for_4channel',
    'modify_first_conv_for_1channel',
    'get_backbone_output_features',
    'BackboneWrapper',
    'list_available_backbones',
    'get_recommended_backbones',
    
    # Fusion
    'EarlyFusion',
    'DualBranchFusion',
    'AttentionGatedFusion',
    'SpatialChannelAttention',
    'CrossModalAttention',
    'create_fusion_model',
    
    # Classification
    'DRClassifier',
    'create_dr_classifier',
    'load_dr_classifier',
    'count_parameters',
]

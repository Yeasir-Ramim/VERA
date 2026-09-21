"""
VERA Phase 2 - Data Module

Exports all data loading, preprocessing, and augmentation functionality.
"""

from .preprocessing import (
    FundusPreprocessor,
    VesselPreprocessor,
    create_preprocessor,
    create_vessel_preprocessor,
    load_and_preprocess_image,
    visualize_preprocessing_steps
)

from .augmentation import (
    FundusAugmentation,
    VesselAugmentation,
    TestTimeAugmentation,
    create_train_augmentation,
    create_vessel_augmentation,
    create_tta
)

from .datasets import (
    DRClassificationDataset,
    APTOSDataset,
    EyePACSDataset,
    Messidor2Dataset,
    VesselSegmentationDataset,
    create_dataloaders,
    get_class_weights
)

from .vessel_cache import (
    VesselCache,
    batch_cache_vessel_maps,
    create_vessel_cache
)

__all__ = [
    # Preprocessing
    'FundusPreprocessor',
    'VesselPreprocessor',
    'create_preprocessor',
    'create_vessel_preprocessor',
    'load_and_preprocess_image',
    'visualize_preprocessing_steps',
    
    # Augmentation
    'FundusAugmentation',
    'VesselAugmentation',
    'TestTimeAugmentation',
    'create_train_augmentation',
    'create_vessel_augmentation',
    'create_tta',
    
    # Datasets
    'DRClassificationDataset',
    'APTOSDataset',
    'EyePACSDataset',
    'Messidor2Dataset',
    'VesselSegmentationDataset',
    'create_dataloaders',
    'get_class_weights',
    
    # Vessel Cache
    'VesselCache',
    'batch_cache_vessel_maps',
    'create_vessel_cache',
]

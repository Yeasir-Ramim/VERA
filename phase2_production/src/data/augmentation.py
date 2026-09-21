"""
VERA Phase 2 - Data Augmentation

Advanced augmentation strategies for fundus images using Albumentations.
Includes geometric, color, and artifact augmentations specific to retinal imaging.
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class FundusAugmentation:
    """
    Fundus-specific augmentation pipeline using Albumentations.
    
    Implements realistic augmentations that preserve clinical features
    while increasing dataset diversity.
    """
    
    def __init__(
        self,
        image_size: int = 512,
        horizontal_flip: float = 0.5,
        vertical_flip: float = 0.5,
        rotation_limit: int = 30,
        shift_limit: float = 0.1,
        scale_limit: float = 0.15,
        brightness_limit: float = 0.2,
        contrast_limit: float = 0.2,
        blur_limit: int = 3,
        use_cutout: bool = True,
        cutout_num_holes: int = 8,
        cutout_max_h_size: int = 32,
        cutout_max_w_size: int = 32,
        use_advanced: bool = True,
        p: float = 0.9
    ):
        """
        Initialize augmentation pipeline.
        
        Args:
            image_size: Target image size
            horizontal_flip: Probability of horizontal flip
            vertical_flip: Probability of vertical flip
            rotation_limit: Rotation angle limit in degrees
            shift_limit: Translation limit as fraction of image size
            scale_limit: Scaling factor limit
            brightness_limit: Brightness adjustment limit
            contrast_limit: Contrast adjustment limit
            blur_limit: Maximum blur kernel size
            use_cutout: Use cutout/coarse dropout
            cutout_num_holes: Number of cutout rectangles
            cutout_max_h_size: Maximum height of cutout
            cutout_max_w_size: Maximum width of cutout
            use_advanced: Use advanced augmentations
            p: Overall probability of applying augmentation
        """
        self.image_size = image_size
        self.p = p
        
        # Build augmentation pipeline
        transforms_list = []
        
        # Geometric augmentations
        if horizontal_flip > 0:
            transforms_list.append(
                A.HorizontalFlip(p=horizontal_flip)
            )
        
        if vertical_flip > 0:
            transforms_list.append(
                A.VerticalFlip(p=vertical_flip)
            )
        
        if rotation_limit > 0:
            transforms_list.append(
                A.ShiftScaleRotate(
                    shift_limit=shift_limit,
                    scale_limit=scale_limit,
                    rotate_limit=rotation_limit,
                    border_mode=0,  # Constant border
                    value=0,
                    p=0.8
                )
            )
        
        # Color augmentations (preserve clinical appearance)
        color_transforms = []
        
        if brightness_limit > 0 or contrast_limit > 0:
            color_transforms.append(
                A.RandomBrightnessContrast(
                    brightness_limit=brightness_limit,
                    contrast_limit=contrast_limit,
                    p=0.8
                )
            )
        
        # Gamma correction (simulates different exposure)
        color_transforms.append(
            A.RandomGamma(gamma_limit=(80, 120), p=0.5)
        )
        
        # Hue/Saturation shift (small changes to simulate camera variations)
        color_transforms.append(
            A.HueSaturationValue(
                hue_shift_limit=10,
                sat_shift_limit=20,
                val_shift_limit=10,
                p=0.5
            )
        )
        
        if color_transforms:
            transforms_list.append(
                A.OneOf(color_transforms, p=0.7)
            )
        
        # Blur augmentations (simulates focus issues)
        if blur_limit > 0:
            blur_transforms = [
                A.Blur(blur_limit=blur_limit, p=1.0),
                A.GaussianBlur(blur_limit=blur_limit, p=1.0),
                A.MedianBlur(blur_limit=blur_limit, p=1.0)
            ]
            transforms_list.append(
                A.OneOf(blur_transforms, p=0.3)
            )
        
        # Advanced augmentations
        if use_advanced:
            advanced_transforms = []
            
            # Elastic transform (subtle deformations)
            advanced_transforms.append(
                A.ElasticTransform(
                    alpha=1,
                    sigma=50,
                    alpha_affine=50,
                    border_mode=0,
                    value=0,
                    p=1.0
                )
            )
            
            # Grid distortion (lens distortion simulation)
            advanced_transforms.append(
                A.GridDistortion(
                    num_steps=5,
                    distort_limit=0.3,
                    border_mode=0,
                    value=0,
                    p=1.0
                )
            )
            
            # Optical distortion
            advanced_transforms.append(
                A.OpticalDistortion(
                    distort_limit=0.5,
                    shift_limit=0.5,
                    border_mode=0,
                    value=0,
                    p=1.0
                )
            )
            
            if advanced_transforms:
                transforms_list.append(
                    A.OneOf(advanced_transforms, p=0.3)
                )
        
        # Cutout / Coarse Dropout (simulates occlusions)
        if use_cutout:
            transforms_list.append(
                A.CoarseDropout(
                    max_holes=cutout_num_holes,
                    max_height=cutout_max_h_size,
                    max_width=cutout_max_w_size,
                    min_holes=cutout_num_holes // 2,
                    min_height=cutout_max_h_size // 2,
                    min_width=cutout_max_w_size // 2,
                    fill_value=0,
                    p=0.3
                )
            )
        
        # Compose all transforms
        self.transform = A.Compose(transforms_list, p=self.p)
        
        logger.info(f"Initialized FundusAugmentation with {len(transforms_list)} transform groups")
    
    def __call__(self, image: np.ndarray, mask: Optional[np.ndarray] = None) -> Dict:
        """
        Apply augmentation to image and optionally mask.
        
        Args:
            image: Input image (H, W, C)
            mask: Optional segmentation mask
            
        Returns:
            Dictionary with augmented image and mask
        """
        if mask is not None:
            augmented = self.transform(image=image, mask=mask)
            return {
                'image': augmented['image'],
                'mask': augmented['mask']
            }
        else:
            augmented = self.transform(image=image)
            return augmented


class VesselAugmentation:
    """
    Augmentation pipeline for vessel segmentation training.
    
    Includes augmentations suitable for both fundus images and
    vessel segmentation masks.
    """
    
    def __init__(
        self,
        image_size: int = 512,
        rotation_limit: int = 45,
        scale_limit: float = 0.2,
        p: float = 0.95
    ):
        """
        Initialize vessel segmentation augmentation.
        
        Args:
            image_size: Target image size
            rotation_limit: Maximum rotation angle
            scale_limit: Maximum scale factor
            p: Probability of applying augmentation
        """
        self.transform = A.Compose([
            # Geometric transforms (with mask)
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=scale_limit,
                rotate_limit=rotation_limit,
                border_mode=0,
                value=0,
                mask_value=0,
                p=0.8
            ),
            
            # Elastic deformation
            A.ElasticTransform(
                alpha=1,
                sigma=50,
                alpha_affine=50,
                border_mode=0,
                value=0,
                mask_value=0,
                p=0.3
            ),
            
            # Grid distortion
            A.GridDistortion(
                num_steps=5,
                distort_limit=0.3,
                border_mode=0,
                value=0,
                mask_value=0,
                p=0.3
            ),
            
            # Color transforms (image only)
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.7
            ),
            A.RandomGamma(gamma_limit=(80, 120), p=0.5),
            
            # Blur (image only)
            A.OneOf([
                A.Blur(blur_limit=3, p=1.0),
                A.GaussianBlur(blur_limit=3, p=1.0)
            ], p=0.3),
            
        ], p=p)
        
        logger.info("Initialized VesselAugmentation for segmentation")
    
    def __call__(self, image: np.ndarray, mask: np.ndarray) -> Dict:
        """
        Apply augmentation to image and mask.
        
        Args:
            image: Input fundus image
            mask: Vessel segmentation mask
            
        Returns:
            Dictionary with augmented image and mask
        """
        augmented = self.transform(image=image, mask=mask)
        return {
            'image': augmented['image'],
            'mask': augmented['mask']
        }


class TestTimeAugmentation:
    """
    Test-time augmentation (TTA) for improved inference.
    
    Applies multiple augmentations during inference and averages predictions.
    """
    
    def __init__(
        self,
        num_augmentations: int = 5,
        horizontal_flip: bool = True,
        vertical_flip: bool = True,
        rotation_angles: list = None
    ):
        """
        Initialize TTA pipeline.
        
        Args:
            num_augmentations: Number of augmentation variants
            horizontal_flip: Include horizontal flip
            vertical_flip: Include vertical flip
            rotation_angles: List of rotation angles to try
        """
        self.num_augmentations = num_augmentations
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        self.rotation_angles = rotation_angles or [0, 90, 180, 270]
        
        # Build TTA transforms
        self.transforms = self._build_tta_transforms()
        
        logger.info(f"Initialized TTA with {len(self.transforms)} augmentation variants")
    
    def _build_tta_transforms(self):
        """Build list of TTA transforms."""
        transforms = []
        
        # Original (no augmentation)
        transforms.append(A.Compose([]))
        
        # Horizontal flip
        if self.horizontal_flip:
            transforms.append(A.Compose([A.HorizontalFlip(p=1.0)]))
        
        # Vertical flip
        if self.vertical_flip:
            transforms.append(A.Compose([A.VerticalFlip(p=1.0)]))
        
        # Rotations
        for angle in self.rotation_angles[:self.num_augmentations - len(transforms)]:
            if angle != 0:
                transforms.append(
                    A.Compose([
                        A.Rotate(limit=(angle, angle), border_mode=0, value=0, p=1.0)
                    ])
                )
        
        return transforms[:self.num_augmentations]
    
    def __call__(self, image: np.ndarray):
        """
        Apply all TTA transforms to image.
        
        Args:
            image: Input image
            
        Returns:
            List of augmented images
        """
        augmented_images = []
        
        for transform in self.transforms:
            augmented = transform(image=image)
            augmented_images.append(augmented['image'])
        
        return augmented_images


def create_train_augmentation(config: Dict) -> FundusAugmentation:
    """
    Create training augmentation pipeline from config.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        FundusAugmentation instance
    """
    aug_config = config.get('preprocessing', {}).get('augmentation', {})
    data_config = config.get('data', {})
    
    if not aug_config.get('enabled', True):
        logger.info("Augmentation disabled in config")
        return None
    
    # Extract cutout config
    cutout_config = aug_config.get('cutout', {})
    
    augmentation = FundusAugmentation(
        image_size=data_config.get('image_size', 512),
        horizontal_flip=aug_config.get('horizontal_flip', 0.5),
        vertical_flip=aug_config.get('vertical_flip', 0.5),
        rotation_limit=aug_config.get('rotation_limit', 30),
        shift_limit=aug_config.get('shift_limit', 0.1),
        scale_limit=aug_config.get('scale_limit', 0.15),
        brightness_limit=aug_config.get('brightness_limit', 0.2),
        contrast_limit=aug_config.get('contrast_limit', 0.2),
        blur_limit=aug_config.get('blur_limit', 3),
        use_cutout=cutout_config.get('enabled', True),
        cutout_num_holes=cutout_config.get('num_holes', 8),
        cutout_max_h_size=cutout_config.get('max_h_size', 32),
        cutout_max_w_size=cutout_config.get('max_w_size', 32),
        use_advanced=True,
        p=0.9
    )
    
    return augmentation


def create_vessel_augmentation(config: Dict) -> VesselAugmentation:
    """
    Create vessel segmentation augmentation from config.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        VesselAugmentation instance
    """
    data_config = config.get('data', {})
    
    augmentation = VesselAugmentation(
        image_size=data_config.get('image_size', 512),
        rotation_limit=45,
        scale_limit=0.2,
        p=0.95
    )
    
    return augmentation


def create_tta(config: Dict) -> Optional[TestTimeAugmentation]:
    """
    Create test-time augmentation from config.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        TestTimeAugmentation instance or None if disabled
    """
    tta_config = config.get('evaluation', {}).get('tta', {})
    
    if not tta_config.get('enabled', False):
        return None
    
    tta = TestTimeAugmentation(
        num_augmentations=tta_config.get('num_augmentations', 5),
        horizontal_flip=True,
        vertical_flip=True,
        rotation_angles=[0, 90, 180, 270]
    )
    
    return tta

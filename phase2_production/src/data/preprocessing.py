"""
VERA Phase 2 - Advanced Preprocessing Pipeline

Implements state-of-the-art fundus image preprocessing including:
- Circular cropping and masking
- Ben Graham's local average color subtraction
- Green-channel CLAHE enhancement
- Color normalization and standardization
"""

import cv2
import numpy as np
from typing import Tuple, Optional, Dict
import logging

logger = logging.getLogger(__name__)


class FundusPreprocessor:
    """
    Comprehensive fundus image preprocessing pipeline.
    
    Methods include circular cropping, Ben Graham preprocessing,
    CLAHE enhancement, and color normalization.
    """
    
    def __init__(
        self,
        target_size: int = 512,
        circular_crop: bool = True,
        crop_threshold: int = 10,
        ben_graham_enabled: bool = True,
        ben_graham_scale: int = 300,
        ben_graham_sigma: float = 10.0,
        clahe_enabled: bool = True,
        clahe_clip_limit: float = 2.0,
        clahe_tile_grid_size: Tuple[int, int] = (8, 8),
        clahe_channels: list = None,
        normalize: bool = True,
        normalize_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        normalize_std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    ):
        """
        Initialize preprocessor with configuration.
        
        Args:
            target_size: Output image size (square)
            circular_crop: Whether to crop to retinal boundary
            crop_threshold: Threshold for circular masking
            ben_graham_enabled: Enable Ben Graham preprocessing
            ben_graham_scale: Gaussian blur scale for local average
            ben_graham_sigma: Sigma for Gaussian kernel
            clahe_enabled: Enable CLAHE enhancement
            clahe_clip_limit: CLAHE clip limit
            clahe_tile_grid_size: CLAHE tile grid size
            clahe_channels: Channels to apply CLAHE (default: green only)
            normalize: Apply ImageNet normalization
            normalize_mean: Mean for normalization
            normalize_std: Std for normalization
        """
        self.target_size = target_size
        self.circular_crop = circular_crop
        self.crop_threshold = crop_threshold
        
        self.ben_graham_enabled = ben_graham_enabled
        self.ben_graham_scale = ben_graham_scale
        self.ben_graham_sigma = ben_graham_sigma
        
        self.clahe_enabled = clahe_enabled
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_tile_grid_size = clahe_tile_grid_size
        self.clahe_channels = clahe_channels or ["green"]
        
        self.normalize = normalize
        self.normalize_mean = np.array(normalize_mean).reshape(1, 1, 3)
        self.normalize_std = np.array(normalize_std).reshape(1, 1, 3)
        
        # Initialize CLAHE object
        if self.clahe_enabled:
            self.clahe = cv2.createCLAHE(
                clipLimit=self.clahe_clip_limit,
                tileGridSize=self.clahe_tile_grid_size
            )
        
        logger.info(f"Initialized FundusPreprocessor: target_size={target_size}, "
                   f"ben_graham={ben_graham_enabled}, clahe={clahe_enabled}")
    
    def preprocess(
        self,
        image: np.ndarray,
        return_mask: bool = False,
        return_steps: bool = False
    ) -> np.ndarray:
        """
        Apply full preprocessing pipeline to fundus image.
        
        Args:
            image: Input BGR image (H, W, 3)
            return_mask: Return circular mask along with image
            return_steps: Return intermediate preprocessing steps
            
        Returns:
            Preprocessed image (H, W, 3) or dict with steps
        """
        steps = {} if return_steps else None
        
        # Store original
        if return_steps:
            steps['original'] = image.copy()
        
        # 1. Circular cropping and masking
        if self.circular_crop:
            image, mask = self._apply_circular_crop(image)
            if return_steps:
                steps['circular_crop'] = image.copy()
                steps['mask'] = mask
        else:
            mask = np.ones(image.shape[:2], dtype=np.uint8) * 255
        
        # 2. Ben Graham preprocessing
        if self.ben_graham_enabled:
            image = self._apply_ben_graham(image, mask)
            if return_steps:
                steps['ben_graham'] = image.copy()
        
        # 3. CLAHE enhancement
        if self.clahe_enabled:
            image = self._apply_clahe(image)
            if return_steps:
                steps['clahe'] = image.copy()
        
        # 4. Resize to target size
        if image.shape[0] != self.target_size or image.shape[1] != self.target_size:
            image = cv2.resize(image, (self.target_size, self.target_size), 
                             interpolation=cv2.INTER_AREA)
            mask = cv2.resize(mask, (self.target_size, self.target_size),
                            interpolation=cv2.INTER_NEAREST)
        
        if return_steps:
            steps['resized'] = image.copy()
        
        # 5. Normalize to [0, 1] and apply ImageNet normalization
        if self.normalize:
            image = image.astype(np.float32) / 255.0
            image = (image - self.normalize_mean) / self.normalize_std
            if return_steps:
                steps['normalized'] = image.copy()
        else:
            image = image.astype(np.float32) / 255.0
        
        if return_steps:
            steps['final'] = image
            return steps
        
        if return_mask:
            return image, mask
        
        return image
    
    def _apply_circular_crop(
        self,
        image: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect and crop to circular retinal boundary.
        
        Args:
            image: Input BGR image
            
        Returns:
            Cropped image and binary mask
        """
        # Convert to grayscale for contour detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Threshold to find retinal region
        _, thresh = cv2.threshold(gray, self.crop_threshold, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) == 0:
            # No contour found, return original with full mask
            mask = np.ones(image.shape[:2], dtype=np.uint8) * 255
            return image, mask
        
        # Get largest contour (main retinal region)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Expand slightly to avoid cutting retinal edges
        margin = 5
        x = max(0, x - margin)
        y = max(0, y - margin)
        w = min(image.shape[1] - x, w + 2 * margin)
        h = min(image.shape[0] - y, h + 2 * margin)
        
        # Crop to square (use larger dimension)
        size = max(w, h)
        
        # Center the crop
        center_x = x + w // 2
        center_y = y + h // 2
        x_start = max(0, center_x - size // 2)
        y_start = max(0, center_y - size // 2)
        x_end = min(image.shape[1], x_start + size)
        y_end = min(image.shape[0], y_start + size)
        
        # Adjust if crop goes out of bounds
        if x_end - x_start < size:
            x_start = max(0, x_end - size)
        if y_end - y_start < size:
            y_start = max(0, y_end - size)
        
        # Crop image
        cropped_image = image[y_start:y_end, x_start:x_end]
        
        # Create circular mask
        mask = np.zeros(cropped_image.shape[:2], dtype=np.uint8)
        center = (cropped_image.shape[1] // 2, cropped_image.shape[0] // 2)
        radius = min(center[0], center[1])
        cv2.circle(mask, center, radius, 255, -1)
        
        # Apply mask to image
        cropped_image = cv2.bitwise_and(cropped_image, cropped_image, mask=mask)
        
        return cropped_image, mask
    
    def _apply_ben_graham(
        self,
        image: np.ndarray,
        mask: np.ndarray
    ) -> np.ndarray:
        """
        Apply Ben Graham's local average color subtraction preprocessing.
        
        This method removes local illumination variations by subtracting
        a heavily blurred version of the image, normalizing brightness
        across the fundus.
        
        Reference:
        Graham, B. (2015). Kaggle Diabetic Retinopathy Detection Competition Report.
        
        Args:
            image: Input BGR image
            mask: Binary mask of retinal region
            
        Returns:
            Preprocessed image with normalized illumination
        """
        # Convert to float
        image_float = image.astype(np.float32)
        
        # Calculate kernel size based on scale
        # The scale parameter controls how local the averaging is
        kernel_size = int(self.ben_graham_scale)
        if kernel_size % 2 == 0:
            kernel_size += 1  # Ensure odd kernel size
        
        # Apply Gaussian blur to get local average
        blurred = cv2.GaussianBlur(
            image_float,
            (kernel_size, kernel_size),
            self.ben_graham_sigma
        )
        
        # Subtract local average and add back a constant (128 for mid-gray)
        # This normalizes local brightness while preserving contrast
        processed = image_float - blurred + 128.0
        
        # Clip to valid range
        processed = np.clip(processed, 0, 255)
        
        # Apply mask to keep only retinal region
        processed = processed.astype(np.uint8)
        processed = cv2.bitwise_and(processed, processed, mask=mask)
        
        return processed
    
    def _apply_clahe(self, image: np.ndarray) -> np.ndarray:
        """
        Apply CLAHE (Contrast Limited Adaptive Histogram Equalization).
        
        Enhances local contrast in fundus images, particularly useful
        for making small vessels and microaneurysms more visible.
        
        Args:
            image: Input BGR image
            
        Returns:
            CLAHE-enhanced image
        """
        # Split channels
        b, g, r = cv2.split(image)
        
        # Apply CLAHE to specified channels
        if "blue" in self.clahe_channels or "all" in self.clahe_channels:
            b = self.clahe.apply(b)
        
        if "green" in self.clahe_channels or "all" in self.clahe_channels:
            g = self.clahe.apply(g)
        
        if "red" in self.clahe_channels or "all" in self.clahe_channels:
            r = self.clahe.apply(r)
        
        # Merge channels back
        enhanced = cv2.merge([b, g, r])
        
        return enhanced
    
    def __call__(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """Allow preprocessor to be called as a function."""
        return self.preprocess(image, **kwargs)


class VesselPreprocessor:
    """
    Specialized preprocessor for vessel segmentation models.
    
    Optimized for U-Net vessel segmentation with focus on
    enhancing vascular structures.
    """
    
    def __init__(
        self,
        target_size: int = 512,
        enhance_vessels: bool = True,
        use_green_channel: bool = True
    ):
        """
        Initialize vessel-focused preprocessor.
        
        Args:
            target_size: Output image size
            enhance_vessels: Apply vessel enhancement
            use_green_channel: Use only green channel (best vessel contrast)
        """
        self.target_size = target_size
        self.enhance_vessels = enhance_vessels
        self.use_green_channel = use_green_channel
        
        # CLAHE for vessel enhancement
        self.clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        
        logger.info(f"Initialized VesselPreprocessor: target_size={target_size}")
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess fundus image for vessel segmentation.
        
        Args:
            image: Input BGR image
            
        Returns:
            Preprocessed image optimized for vessel segmentation
        """
        # Extract green channel (best vessel contrast)
        if self.use_green_channel:
            if len(image.shape) == 3:
                image = image[:, :, 1]  # Green channel
        
        # Enhance vessels with CLAHE
        if self.enhance_vessels:
            image = self.clahe.apply(image)
        
        # Resize
        if image.shape[0] != self.target_size or image.shape[1] != self.target_size:
            image = cv2.resize(image, (self.target_size, self.target_size),
                             interpolation=cv2.INTER_AREA)
        
        # Normalize to [0, 1]
        image = image.astype(np.float32) / 255.0
        
        # Add channel dimension if using single channel
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=-1)
        
        return image
    
    def __call__(self, image: np.ndarray) -> np.ndarray:
        """Allow preprocessor to be called as a function."""
        return self.preprocess(image)


def create_preprocessor(config: Dict) -> FundusPreprocessor:
    """
    Factory function to create preprocessor from configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configured FundusPreprocessor instance
    """
    preprocessing_config = config.get('preprocessing', {})
    data_config = config.get('data', {})
    
    # Extract CLAHE settings
    clahe_config = preprocessing_config.get('clahe', {})
    clahe_channels = clahe_config.get('apply_to_channels', ['green'])
    
    # Extract Ben Graham settings
    ben_graham_config = preprocessing_config.get('ben_graham', {})
    
    preprocessor = FundusPreprocessor(
        target_size=data_config.get('image_size', 512),
        circular_crop=preprocessing_config.get('circular_crop', True),
        crop_threshold=preprocessing_config.get('crop_threshold', 10),
        ben_graham_enabled=ben_graham_config.get('enabled', True),
        ben_graham_scale=ben_graham_config.get('scale', 300),
        ben_graham_sigma=ben_graham_config.get('sigma', 10.0),
        clahe_enabled=clahe_config.get('enabled', True),
        clahe_clip_limit=clahe_config.get('clip_limit', 2.0),
        clahe_tile_grid_size=tuple(clahe_config.get('tile_grid_size', [8, 8])),
        clahe_channels=clahe_channels,
        normalize=True,
        normalize_mean=tuple(data_config.get('normalize_mean', [0.485, 0.456, 0.406])),
        normalize_std=tuple(data_config.get('normalize_std', [0.229, 0.224, 0.225]))
    )
    
    return preprocessor


def create_vessel_preprocessor(config: Dict) -> VesselPreprocessor:
    """
    Factory function to create vessel preprocessor from configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configured VesselPreprocessor instance
    """
    data_config = config.get('data', {})
    
    preprocessor = VesselPreprocessor(
        target_size=data_config.get('image_size', 512),
        enhance_vessels=True,
        use_green_channel=True
    )
    
    return preprocessor


# Utility functions for preprocessing
def load_and_preprocess_image(
    image_path: str,
    preprocessor: FundusPreprocessor
) -> np.ndarray:
    """
    Load and preprocess a single fundus image.
    
    Args:
        image_path: Path to image file
        preprocessor: Preprocessor instance
        
    Returns:
        Preprocessed image array
    """
    # Load image
    image = cv2.imread(image_path)
    
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    # Preprocess
    processed = preprocessor.preprocess(image)
    
    return processed


def visualize_preprocessing_steps(
    image: np.ndarray,
    preprocessor: FundusPreprocessor,
    save_path: Optional[str] = None
) -> Dict[str, np.ndarray]:
    """
    Visualize all preprocessing steps for debugging/analysis.
    
    Args:
        image: Input image
        preprocessor: Preprocessor instance
        save_path: Optional path to save visualization
        
    Returns:
        Dictionary with all intermediate preprocessing steps
    """
    steps = preprocessor.preprocess(image, return_steps=True)
    
    if save_path:
        import matplotlib.pyplot as plt
        
        # Create visualization
        num_steps = len(steps)
        fig, axes = plt.subplots(2, (num_steps + 1) // 2, figsize=(15, 6))
        axes = axes.flatten()
        
        for idx, (name, img) in enumerate(steps.items()):
            if name == 'mask':
                axes[idx].imshow(img, cmap='gray')
            else:
                # Handle normalized images
                if img.dtype == np.float32 and img.min() < 0:
                    # Denormalize for visualization
                    img_vis = img * preprocessor.normalize_std + preprocessor.normalize_mean
                    img_vis = np.clip(img_vis * 255, 0, 255).astype(np.uint8)
                else:
                    img_vis = img
                
                # Convert BGR to RGB for matplotlib
                if len(img_vis.shape) == 3:
                    img_vis = cv2.cvtColor(img_vis.astype(np.uint8), cv2.COLOR_BGR2RGB)
                
                axes[idx].imshow(img_vis)
            
            axes[idx].set_title(name.replace('_', ' ').title())
            axes[idx].axis('off')
        
        # Hide unused subplots
        for idx in range(len(steps), len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved preprocessing visualization to {save_path}")
    
    return steps

"""
Preprocessing module for Fundus Images.
Implements:
1. Circular Crop (Black border removal via contour/thresholding).
2. CLAHE (Contrast Limited Adaptive Histogram Equalization).
3. Resizing and standard normalization.
"""

from typing import Tuple, Union
import cv2
import numpy as np
from PIL import Image


def crop_fundus_circle(img: np.ndarray, tol: int = 10) -> np.ndarray:
    """
    Crops black borders from a retinal fundus image based on pixel intensity thresholding.
    
    Args:
        img: Input image as RGB numpy array (H, W, C).
        tol: Intensity tolerance threshold for detecting background black pixels.
        
    Returns:
        Cropped RGB numpy array focusing on the circular fundus region.
    """
    if img.ndim == 2:
        mask = img > tol
        if not mask.any():
            return img
        return img[np.ix_(mask.any(1), mask.any(0))]
    
    # 3-channel image
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    mask = gray > tol
    
    # Check if mask is valid
    if not mask.any():
        return img
    
    # Bounding box for non-black area
    row_idx = mask.any(axis=1)
    col_idx = mask.any(axis=0)
    
    ymin, ymax = np.where(row_idx)[0][[0, -1]]
    xmin, xmax = np.where(col_idx)[0][[0, -1]]
    
    # Add a slight safety margin if possible
    h, w = gray.shape
    ymin = max(0, ymin - 2)
    ymax = min(h, ymax + 2)
    xmin = max(0, xmin - 2)
    xmax = min(w, xmax + 2)
    
    cropped = img[ymin:ymax, xmin:xmax]
    return cropped


def apply_clahe(
    img: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: Tuple[int, int] = (8, 8)
) -> np.ndarray:
    """
    Applies CLAHE (Contrast Limited Adaptive Histogram Equalization) on the L-channel in LAB space.
    Enhances vessel and lesion contrast while preserving retinal color balance.
    
    Args:
        img: RGB image numpy array (H, W, 3) in range [0, 255].
        clip_limit: Threshold for contrast limiting.
        tile_grid_size: Size of grid for histogram equalization.
        
    Returns:
        Enhanced RGB image numpy array (H, W, 3).
    """
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    cl = clahe.apply(l)
    
    merged_lab = cv2.merge((cl, a, b))
    enhanced_rgb = cv2.cvtColor(merged_lab, cv2.COLOR_LAB2RGB)
    return enhanced_rgb


def apply_ben_graham(
    img: np.ndarray,
    sigma: float = 10.0,
    mask_tolerance: int = 10
) -> np.ndarray:
    """
    Applies Ben Graham's local color subtraction method (Kaggle DR 1st place):
        I_bg = 4 * I - 4 * GaussianBlur(I, sigma) + 128
    Coupled with a foreground mask to prevent boundary halo artifacts.
    
    Args:
        img: RGB image numpy array (H, W, 3) in range [0, 255].
        sigma: Standard deviation for Gaussian kernel.
        mask_tolerance: Pixel intensity threshold to separate retinal disk from black background.
        
    Returns:
        Enhanced RGB image numpy array (H, W, 3) uint8.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim == 3 else img
    mask = gray > mask_tolerance
    
    # Gaussian blur captures low-frequency illumination variation across camera types
    blur = cv2.GaussianBlur(img, (0, 0), sigmaX=sigma, sigmaY=sigma)
    
    # 4 * I - 4 * blur + 128
    bg = cv2.addWeighted(img, 4.0, blur, -4.0, 128)
    
    # Restore background to clean zero to eliminate halo artifacts
    if img.ndim == 3:
        for c in range(3):
            bg[:, :, c] = np.where(mask, bg[:, :, c], 0)
    else:
        bg = np.where(mask, bg, 0)
        
    return np.clip(bg, 0, 255).astype(np.uint8)


def preprocess_fundus(
    image: Union[np.ndarray, Image.Image, str],
    target_size: Tuple[int, int] = (224, 224),
    apply_crop: bool = True,
    apply_enhancement: bool = True,
    enhancement_method: str = "clahe"
) -> np.ndarray:
    """
    Full preprocessing pipeline for fundus images:
    Read/Convert -> Circular Crop -> Enhancement (CLAHE / Ben Graham / Combined) -> Resize.
    
    Args:
        image: Filepath, PIL Image, or numpy RGB array.
        target_size: Desired output dimensions (height, width).
        apply_crop: Whether to crop black background borders.
        apply_enhancement: Whether to apply enhancement.
        enhancement_method: One of ['clahe', 'ben_graham', 'combined', 'none'].
        
    Returns:
        Preprocessed RGB numpy array of shape (target_size[0], target_size[1], 3) uint8.
    """
    if isinstance(image, str):
        img_bgr = cv2.imread(image)
        if img_bgr is None:
            raise FileNotFoundError(f"Image not found at path: {image}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    elif isinstance(image, Image.Image):
        img_rgb = np.array(image.convert("RGB"))
    elif isinstance(image, np.ndarray):
        if image.ndim == 2:
            img_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            img_rgb = image.copy()
    else:
        raise TypeError(f"Unsupported image type: {type(image)}")
        
    if apply_crop:
        img_rgb = crop_fundus_circle(img_rgb)
        
    if apply_enhancement and enhancement_method != "none":
        if enhancement_method == "clahe":
            img_rgb = apply_clahe(img_rgb)
        elif enhancement_method == "ben_graham":
            img_rgb = apply_ben_graham(img_rgb)
        elif enhancement_method == "combined":
            img_rgb = apply_clahe(img_rgb)
            img_rgb = apply_ben_graham(img_rgb)
        else:
            raise ValueError(f"Unknown enhancement_method: {enhancement_method}")
        
    if target_size is not None:
        img_rgb = cv2.resize(img_rgb, (target_size[1], target_size[0]), interpolation=cv2.INTER_AREA)
        
    return img_rgb

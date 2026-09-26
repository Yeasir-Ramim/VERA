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


def validate_fundus_image(img_rgb: np.ndarray) -> Tuple[bool, str, float]:
    """
    Validates whether an input image is a legitimate Retinal Fundus photograph related to DR.
    Checks:
    1. Channel structure (3-channel RGB image).
    2. Ocular color profile & Red channel dominance (Fundus tissue is dominantly orange/reddish).
    3. Foreground/background ratio (Circular aperture geometry).
    4. Green channel vascular structural contrast.
    
    Args:
        img_rgb: Input RGB image array of shape (H, W, 3).
        
    Returns:
        Tuple of (is_valid: bool, reason_message: str, score: float)
    """
    if not isinstance(img_rgb, np.ndarray) or img_rgb.ndim != 3 or img_rgb.shape[2] != 3:
        return False, "Invalid image format: Input must be a 3-channel RGB image.", 0.0
        
    h, w, _ = img_rgb.shape
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    
    # Check foreground (non-black border pixels)
    fg_mask = gray > 15
    fg_pixels = fg_mask.sum()
    total_pixels = h * w
    fg_ratio = fg_pixels / total_pixels
    
    if fg_pixels < 100 or fg_ratio < 0.10:
        return False, "Uploaded image is too dark or empty. Please upload a clear Retinal Fundus photograph.", 0.0
        
    # Color profile check on foreground
    r_fg = img_rgb[:, :, 0][fg_mask].astype(float)
    g_fg = img_rgb[:, :, 1][fg_mask].astype(float)
    b_fg = img_rgb[:, :, 2][fg_mask].astype(float)
    
    mean_r = float(np.mean(r_fg))
    mean_g = float(np.mean(g_fg))
    mean_b = float(np.mean(b_fg))
    
    # Retinal fundus tissue has strong Red dominance over Blue (mean_r > mean_b)
    # and warm hue spectrum (R > B by significant margin)
    red_blue_ratio = (mean_r + 1.0) / (mean_b + 1.0)
    red_green_ratio = (mean_r + 1.0) / (mean_g + 1.0)
    
    # Check HSV hue spectrum (Ocular fundus hues are in red/orange: H < 30 or H > 150)
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    hues = hsv[:, :, 0][fg_mask]
    warm_hue_pixels = np.logical_or(hues < 35, hues > 145).sum()
    warm_hue_ratio = float(warm_hue_pixels) / len(hues) if len(hues) > 0 else 0.0
    
    # Structural variance in Green channel (where blood vessels/lesions create contrast)
    green_ch = img_rgb[:, :, 1]
    laplacian_var = float(cv2.Laplacian(green_ch, cv2.CV_64F).var())
    
    # Scoring system (0.0 to 1.0)
    score = 0.0
    if red_blue_ratio > 1.2: score += 0.35
    if red_green_ratio > 0.95: score += 0.20
    if warm_hue_ratio > 0.45: score += 0.30
    if laplacian_var > 5.0: score += 0.15
    
    is_valid = score >= 0.50
    
    if is_valid:
        msg = f"[OK] Valid Retinal Fundus Photograph verified (Confidence: {score*100:.1f}%)."
    else:
        msg = (
            "[WARNING] Non-Retinal / Invalid Image Detected: The uploaded photograph does not match the characteristic color or structural features of a Retinal Fundus image. "
            "Please upload a valid DR-related Retinal Fundus photograph (e.g., APTOS 2019, EyePACS, or Messidor fundus image) for Diabetic Retinopathy prediction."
        )

        
    return is_valid, msg, round(score, 3)


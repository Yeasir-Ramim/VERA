"""
Retinal Vessel Segmentation Module.
Produces single-channel vessel probability maps [0, 1] for fundus images.
Includes:
- Multi-scale matched filter / morphological vessel feature extraction.
- PyTorch U-Net vessel segmentation architecture (DRIVE/CHASE inference).
- Disk caching mechanism for fast batch training.
"""

import os
from pathlib import Path
from typing import Optional, Tuple, Union
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UNetVesselSegmenter(nn.Module):
    """
    Standard U-Net architecture for retinal vessel segmentation.
    Expects single-channel green channel or 3-channel RGB fundus image.
    Outputs 1-channel vessel probability map in range [0, 1].
    """
    def __init__(self, in_channels: int = 1, out_channels: int = 1, init_features: int = 32):
        super().__init__()
        features = init_features
        self.encoder1 = ConvBlock(in_channels, features)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.encoder2 = ConvBlock(features, features * 2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.encoder3 = ConvBlock(features * 2, features * 4)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.bottleneck = ConvBlock(features * 4, features * 8)
        
        self.upconv3 = nn.ConvTranspose2d(features * 8, features * 4, kernel_size=2, stride=2)
        self.decoder3 = ConvBlock(features * 8, features * 4)
        
        self.upconv2 = nn.ConvTranspose2d(features * 4, features * 2, kernel_size=2, stride=2)
        self.decoder2 = ConvBlock(features * 4, features * 2)
        
        self.upconv1 = nn.ConvTranspose2d(features * 2, features, kernel_size=2, stride=2)
        self.decoder1 = ConvBlock(features * 2, features)
        
        self.head = nn.Conv2d(features, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        
        bottleneck = self.bottleneck(self.pool3(enc3))
        
        dec3 = self.upconv3(bottleneck)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        
        logits = self.head(dec1)
        return self.sigmoid(logits)


def extract_vessel_map_multiscale(
    img_rgb: np.ndarray,
    target_size: Optional[Tuple[int, int]] = None
) -> np.ndarray:
    """
    Extracts retinal vessel probability map using multi-scale morphological top-hat
    filtering and matched filtering on the green channel.
    This provides a deterministic, high-quality vessel map without requiring external model weights.
    
    Args:
        img_rgb: Preprocessed RGB fundus image (H, W, 3) in [0, 255].
        target_size: Optional resize dimensions (height, width).
        
    Returns:
        Vessel probability map (H, W) float32 in [0.0, 1.0].
    """
    if target_size is not None and (img_rgb.shape[0] != target_size[0] or img_rgb.shape[1] != target_size[1]):
        img_rgb = cv2.resize(img_rgb, (target_size[1], target_size[0]), interpolation=cv2.INTER_AREA)
        
    # Green channel has strongest vessel absorption/contrast
    green = img_rgb[:, :, 1]
    
    # 1. Background illumination normalization using median blur
    background = cv2.medianBlur(green, 15)
    normalized = cv2.addWeighted(green, 1.5, background, -0.5, 0)
    
    # 2. Invert so vessels are bright
    inverted = cv2.bitwise_not(normalized)
    
    # 3. Multi-scale Top-Hat morphological enhancement for thin and thick vessel branches
    vessel_acc = np.zeros_like(inverted, dtype=np.float32)
    scales = [3, 5, 7, 9]
    for scale in scales:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (scale, scale))
        tophat = cv2.morphologyEx(inverted, cv2.MORPH_TOPHAT, kernel)
        vessel_acc += tophat.astype(np.float32) / len(scales)
        
    # 4. Multi-angle Gabor / directional matched filter for tubular vessel response
    angles = np.arange(0, 180, 15)
    gabor_max = np.zeros_like(vessel_acc)
    for theta in angles:
        kernel = cv2.getGaborKernel(
            ksize=(9, 9),
            sigma=1.5,
            theta=np.deg2rad(theta),
            lambd=4.0,
            gamma=0.5,
            psi=0,
            ktype=cv2.CV_32F
        )
        filtered = cv2.filter2D(vessel_acc, cv2.CV_32F, kernel)
        gabor_max = np.maximum(gabor_max, filtered)
        
    # Combine morphology and directional filter
    vessel_combined = 0.5 * vessel_acc + 0.5 * gabor_max
    
    # 5. Mask out fundus boundary to prevent border edge artifacts
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    mask = (gray > 15).astype(np.uint8)
    kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask_eroded = cv2.erode(mask, kernel_erode)
    vessel_combined = vessel_combined * mask_eroded
    
    # 6. Normalize to [0.0, 1.0] probability range
    v_min, v_max = vessel_combined.min(), vessel_combined.max()
    if v_max > v_min:
        vessel_map = (vessel_combined - v_min) / (v_max - v_min)
    else:
        vessel_map = np.zeros_like(vessel_combined, dtype=np.float32)
        
    # Contrast stretch and gamma correction for sharp vessel profiles
    vessel_map = np.clip(np.power(vessel_map, 0.85), 0.0, 1.0).astype(np.float32)
    return vessel_map


class VesselSegmenter:
    """
    Vessel Segmentation Engine with caching support.
    Supports multi-scale filter inference and neural U-Net inference.
    """
    def __init__(
        self,
        cache_dir: Optional[Union[str, Path]] = None,
        model_weights_path: Optional[str] = None,
        device: str = "cpu"
    ):
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            
        self.device = torch.device(device if torch.cuda.is_available() and device != "cpu" else "cpu")
        self.model: Optional[UNetVesselSegmenter] = None
        
        if model_weights_path and os.path.exists(model_weights_path):
            self.model = UNetVesselSegmenter(in_channels=1, out_channels=1)
            state_dict = torch.load(model_weights_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()

    def get_vessel_map(
        self,
        img_rgb: np.ndarray,
        image_id: Optional[str] = None,
        target_size: Tuple[int, int] = (224, 224)
    ) -> np.ndarray:
        """
        Retrieves vessel map from cache if present, otherwise computes and caches it.
        
        Args:
            img_rgb: Preprocessed RGB image (H, W, 3).
            image_id: Unique identifier for caching.
            target_size: Output shape (H, W).
            
        Returns:
            Vessel probability map (H, W) float32 in range [0.0, 1.0].
        """
        cache_file = None
        if self.cache_dir and image_id:
            cache_file = self.cache_dir / f"{image_id}_vessel.png"
            if cache_file.exists():
                cached_img = cv2.imread(str(cache_file), cv2.IMREAD_GRAYSCALE)
                if cached_img is not None:
                    if cached_img.shape != target_size:
                        cached_img = cv2.resize(cached_img, (target_size[1], target_size[0]))
                    return (cached_img / 255.0).astype(np.float32)

        # Compute map
        if self.model is not None:
            # Neural U-Net inference
            green = img_rgb[:, :, 1].astype(np.float32) / 255.0
            if green.shape != target_size:
                green = cv2.resize(green, (target_size[1], target_size[0]))
            tensor_in = torch.from_numpy(green).unsqueeze(0).unsqueeze(0).to(self.device)
            with torch.no_grad():
                pred = self.model(tensor_in).squeeze().cpu().numpy()
            vessel_map = np.clip(pred, 0.0, 1.0).astype(np.float32)
        else:
            vessel_map = extract_vessel_map_multiscale(img_rgb, target_size=target_size)

        # Save to cache
        if cache_file is not None:
            save_img = (vessel_map * 255.0).astype(np.uint8)
            cv2.imwrite(str(cache_file), save_img)

        return vessel_map

    def predict(self, img_rgb: np.ndarray, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
        """Convenience method for direct inference without mandatory disk caching."""
        return self.get_vessel_map(img_rgb, image_id=None, target_size=target_size)

    def precompute_dataset_vessels(
        self,
        df: pd.DataFrame,
        image_dir: Union[str, Path],
        target_size: Tuple[int, int] = (224, 224)
    ):
        """Precomputes and stores vessel probability maps into cache directory for fast dataloading."""
        image_dir = Path(image_dir)
        id_col = "image_id" if "image_id" in df.columns else ("id_code" if "id_code" in df.columns else df.columns[0])
        from .preprocessing import preprocess_fundus
        for _, row in df.iterrows():
            img_id = str(row[id_col])
            for ext in ["", ".png", ".jpg", ".jpeg", ".PNG", ".JPG"]:
                candidate = image_dir / f"{img_id}{ext}"
                if candidate.exists() and candidate.is_file():
                    try:
                        img_rgb = preprocess_fundus(str(candidate), target_size=target_size)
                        self.get_vessel_map(img_rgb, image_id=img_id, target_size=target_size)
                    except Exception:
                        pass
                    break


"""
Dataset module for Diabetic Retinopathy Detection.
Supports:
- APTOS 2019 dataset format (CSV with 'id_code' or 'image_path' and 'diagnosis' label 0-4).
- 3-Channel RGB loading.
- 4-Channel (RGB + Vessel Probability Map) stacking.
- Albumentations data augmentation for fundus images (rotation, flip, jitter).
- Balanced class weight computation.
"""

from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import cv2

from .preprocessing import preprocess_fundus
from .vessel_segmentation import VesselSegmenter


# Standard ImageNet normalization statistics
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Vessel channel normalization statistics (mean ~0.15, std ~0.25)
VESSEL_MEAN = [0.150]
VESSEL_STD = [0.250]


def standardize_dataset_df(df: pd.DataFrame, dataset_name: str = "aptos") -> pd.DataFrame:
    """
    Standardizes DataFrame schema across APTOS 2019, EyePACS, and Messidor-2 datasets.
    Ensures consistent presence of:
      - 'image_id': String image filename or ID
      - 'diagnosis': ICDR DR grade (0: None, 1: Mild, 2: Moderate, 3: Severe, 4: Proliferative)
      - 'referable': Binary indicator (0: Non-Referable [0-1], 1: Referable [2-4])
    """
    df = df.copy()
    
    # Standardize image identifier column
    for candidate in ["id_code", "image_id", "image", "Image name", "Image_ID"]:
        if candidate in df.columns:
            df["image_id"] = df[candidate].astype(str)
            break
    if "image_id" not in df.columns:
        df["image_id"] = df.iloc[:, 0].astype(str)
        
    # Standardize DR grade target column
    for candidate in ["diagnosis", "level", "Retinopathy grade", "label", "dr_grade", "adjudicated_dr_grade"]:
        if candidate in df.columns:
            df["diagnosis"] = pd.to_numeric(df[candidate], errors="coerce").fillna(0).astype(int)
            break
    if "diagnosis" not in df.columns:
        df["diagnosis"] = 0
        
    df["diagnosis"] = df["diagnosis"].clip(0, 4)
    df["referable"] = (df["diagnosis"] >= 2).astype(int)
    return df


class APTOSDataset(Dataset):
    """
    PyTorch Dataset for Diabetic Retinopathy grading (supports APTOS, EyePACS, Messidor-2).
    """
    def __init__(
        self,
        df: pd.DataFrame,
        image_dir: Union[str, Path],
        target_size: Tuple[int, int] = (224, 224),
        is_4channel: bool = False,
        vessel_segmenter: Optional[VesselSegmenter] = None,
        transform: Optional[Callable] = None,
        is_training: bool = False,
        enhancement_method: str = "clahe"
    ):
        """
        Args:
            df: DataFrame containing image IDs and diagnosis labels (0-4).
            image_dir: Directory containing the fundus image files.
            target_size: (H, W) for model input.
            is_4channel: If True, returns 4-channel tensor (RGB + Vessel map).
            vessel_segmenter: VesselSegmenter instance for extracting/caching vessel maps.
            transform: Optional Albumentations transform pipeline.
            is_training: Whether dataset is in training mode (enables augmentations).
            enhancement_method: 'clahe', 'ben_graham', 'combined', or 'none'.
        """
        self.df = standardize_dataset_df(df).reset_index(drop=True)
        self.image_dir = Path(image_dir)
        self.target_size = target_size
        self.is_4channel = is_4channel
        self.vessel_segmenter = vessel_segmenter or VesselSegmenter()
        self.transform = transform
        self.is_training = is_training
        self.enhancement_method = enhancement_method
        
        self.id_col = "image_id"
        self.target_col = "diagnosis"

    def __len__(self) -> int:
        return len(self.df)

    def _find_image_path(self, img_id: str) -> Path:
        """Finds image with matching id_code across common extensions."""
        img_id_clean = str(img_id).strip()
        # Direct check
        direct = self.image_dir / img_id_clean
        if direct.exists():
            return direct
        for ext in [".png", ".jpg", ".jpeg", ".PNG", ".JPG"]:
            candidate = self.image_dir / f"{img_id_clean}{ext}"
            if candidate.exists():
                return candidate
        raise FileNotFoundError(f"Image {img_id_clean} not found in {self.image_dir}")

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        row = self.df.iloc[idx]
        img_id = str(row[self.id_col])
        label = int(row[self.target_col]) if self.target_col in row else 0
        
        img_path = self._find_image_path(img_id)
        
        # 1. Preprocess RGB fundus image (Crop + Enhancement [CLAHE/Ben Graham] + Resize)
        rgb_img = preprocess_fundus(
            str(img_path),
            target_size=self.target_size,
            enhancement_method=self.enhancement_method
        )
        
        # 2. Extract / Retrieve vessel probability map if 4-channel
        vessel_map = None
        if self.is_4channel:
            vessel_map = self.vessel_segmenter.get_vessel_map(
                rgb_img,
                image_id=img_id,
                target_size=self.target_size
            )
            
        # 3. Apply spatial augmentations (consistent across RGB and Vessel map)
        if self.transform is not None:
            if self.is_4channel and vessel_map is not None:
                # Augment both together
                augmented = self.transform(image=rgb_img, mask=vessel_map)
                rgb_img = augmented["image"]
                vessel_map = augmented["mask"]
            else:
                augmented = self.transform(image=rgb_img)
                rgb_img = augmented["image"]
        elif self.is_training:
            # Default lightweight numpy augmentations
            if np.random.rand() > 0.5:
                rgb_img = np.fliplr(rgb_img).copy()
                if vessel_map is not None:
                    vessel_map = np.fliplr(vessel_map).copy()
            if np.random.rand() > 0.5:
                rgb_img = np.flipud(rgb_img).copy()
                if vessel_map is not None:
                    vessel_map = np.flipud(vessel_map).copy()

        # 4. Normalize & Convert to PyTorch Tensor
        rgb_norm = rgb_img.astype(np.float32) / 255.0
        # Standardize RGB with ImageNet stats
        for c in range(3):
            rgb_norm[:, :, c] = (rgb_norm[:, :, c] - IMAGENET_MEAN[c]) / IMAGENET_STD[c]
            
        # Shape: (3, H, W)
        rgb_tensor = torch.from_numpy(rgb_norm.transpose(2, 0, 1)).float()
        
        if self.is_4channel and vessel_map is not None:
            # Standardize Vessel channel: (H, W) -> (1, H, W)
            vessel_norm = (vessel_map - VESSEL_MEAN[0]) / VESSEL_STD[0]
            vessel_tensor = torch.from_numpy(vessel_norm).unsqueeze(0).float()
            
            # Stack into 4-channel tensor: (4, H, W)
            tensor_out = torch.cat([rgb_tensor, vessel_tensor], dim=0)
        else:
            tensor_out = rgb_tensor
            
        return tensor_out, label, img_id


def compute_class_weights(df: pd.DataFrame, target_col: str = "diagnosis", num_classes: int = 5) -> torch.Tensor:
    """
    Computes inverse-frequency class weights for cross-entropy loss to handle class imbalance.
    """
    counts = df[target_col].value_counts().to_dict()
    total_samples = len(df)
    
    weights = []
    for c in range(num_classes):
        cnt = counts.get(c, 0)
        if cnt > 0:
            # Standard balanced weighting: N / (K * N_c)
            w = total_samples / (num_classes * cnt)
        else:
            w = 1.0
        weights.append(w)
        
    weights_tensor = torch.tensor(weights, dtype=torch.float32)
    # Normalize weights so mean is 1.0
    weights_tensor = weights_tensor / weights_tensor.mean()
    return weights_tensor


def create_dataloaders(
    df: pd.DataFrame,
    image_dir: Union[str, Path],
    val_split: float = 0.2,
    batch_size: int = 16,
    is_4channel: bool = False,
    target_size: Tuple[int, int] = (224, 224),
    vessel_cache_dir: Optional[Union[str, Path]] = None,
    random_state: int = 42,
    num_workers: int = 2,
    enhancement_method: str = "clahe"
) -> Tuple[DataLoader, DataLoader, torch.Tensor]:
    """
    Splits DataFrame into train/validation sets with stratification and creates DataLoaders.
    """
    target_col = "diagnosis" if "diagnosis" in df.columns else "label"
    
    # Stratified Train/Val split
    from sklearn.model_selection import train_test_split
    train_df, val_df = train_test_split(
        df,
        test_size=val_split,
        random_state=random_state,
        stratify=df[target_col] if df[target_col].nunique() > 1 else None
    )
    
    class_weights = compute_class_weights(train_df, target_col=target_col)
    vessel_segmenter = VesselSegmenter(cache_dir=vessel_cache_dir)
    
    train_dataset = APTOSDataset(
        df=train_df,
        image_dir=image_dir,
        target_size=target_size,
        is_4channel=is_4channel,
        vessel_segmenter=vessel_segmenter,
        is_training=True,
        enhancement_method=enhancement_method
    )
    
    val_dataset = APTOSDataset(
        df=val_df,
        image_dir=image_dir,
        target_size=target_size,
        is_4channel=is_4channel,
        vessel_segmenter=vessel_segmenter,
        is_training=False,
        enhancement_method=enhancement_method
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    return train_loader, val_loader, class_weights


# Alias for cross-dataset generalization (APTOS, EyePACS, Messidor-2)
RetinalDRDataset = APTOSDataset
FundusDataset = APTOSDataset  # Main export name

# Transform functions for compatibility
def get_train_transform():
    """Returns training augmentation transform."""
    return None  # Will use default numpy augmentations in dataset

def get_val_transform():
    """Returns validation transform (no augmentation)."""
    return None


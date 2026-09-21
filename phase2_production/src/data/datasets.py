"""
VERA Phase 2 - Multi-Dataset Loaders

Unified dataset loading supporting multiple DR grading datasets:
- APTOS 2019 Blindness Detection
- EyePACS Diabetic Retinopathy Detection
- Messidor-2

Plus vessel segmentation datasets:
- DRIVE
- CHASE_DB1
"""

import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Callable
import logging
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


class DRClassificationDataset(Dataset):
    """
    Generic Diabetic Retinopathy classification dataset.
    
    Supports multiple dataset formats with unified interface.
    """
    
    def __init__(
        self,
        image_paths: List[str],
        labels: List[int],
        preprocessor: Optional[Callable] = None,
        augmentation: Optional[Callable] = None,
        vessel_cache: Optional[object] = None,
        use_vessel_channel: bool = True,
        return_vessel_map: bool = True,
        transform: Optional[Callable] = None
    ):
        """
        Initialize DR classification dataset.
        
        Args:
            image_paths: List of image file paths
            labels: List of DR severity labels (0-4)
            preprocessor: Preprocessing function
            augmentation: Augmentation function
            vessel_cache: VesselCache instance for loading vessel maps
            use_vessel_channel: Include vessel channel in output
            return_vessel_map: Return vessel map separately
            transform: Additional transforms (e.g., ToTensor)
        """
        self.image_paths = image_paths
        self.labels = labels
        self.preprocessor = preprocessor
        self.augmentation = augmentation
        self.vessel_cache = vessel_cache
        self.use_vessel_channel = use_vessel_channel
        self.return_vessel_map = return_vessel_map
        self.transform = transform
        
        assert len(self.image_paths) == len(self.labels), \
            "Number of images and labels must match"
        
        logger.info(f"Initialized DRClassificationDataset with {len(self.image_paths)} samples")
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Get a single sample.
        
        Returns:
            Dictionary with:
                - image: Preprocessed RGB image tensor (3, H, W)
                - vessel: Vessel probability map tensor (1, H, W)
                - label: DR severity label (0-4)
                - image_id: Image identifier
                - (optionally) vessel_map: Vessel map as separate output
        """
        # Load image
        image_path = self.image_paths[idx]
        image = cv2.imread(image_path)
        
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        # Get label
        label = self.labels[idx]
        
        # Get image ID
        image_id = Path(image_path).stem
        
        # Preprocess
        if self.preprocessor is not None:
            image = self.preprocessor.preprocess(image)
        else:
            # Basic preprocessing: resize and normalize
            image = cv2.resize(image, (512, 512))
            image = image.astype(np.float32) / 255.0
        
        # Load or generate vessel map
        vessel_map = None
        if self.use_vessel_channel and self.vessel_cache is not None:
            vessel_map = self.vessel_cache.load(image_id)
            
            if vessel_map is None:
                # Vessel map not cached, create dummy or skip
                logger.warning(f"Vessel map not cached for {image_id}, using zeros")
                vessel_map = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
        
        # Apply augmentation (to both image and vessel map if available)
        if self.augmentation is not None:
            if vessel_map is not None:
                augmented = self.augmentation(image=image, mask=vessel_map)
                image = augmented['image']
                vessel_map = augmented.get('mask', vessel_map)
            else:
                augmented = self.augmentation(image=image)
                image = augmented['image']
        
        # Convert to tensor format
        # Image: (H, W, 3) -> (3, H, W)
        if len(image.shape) == 3:
            image_tensor = torch.from_numpy(image).permute(2, 0, 1).float()
        else:
            image_tensor = torch.from_numpy(image).unsqueeze(0).float()
        
        # Vessel map: (H, W) -> (1, H, W)
        if vessel_map is not None:
            if len(vessel_map.shape) == 2:
                vessel_tensor = torch.from_numpy(vessel_map).unsqueeze(0).float()
            else:
                vessel_tensor = torch.from_numpy(vessel_map).permute(2, 0, 1).float()
        else:
            # Create empty vessel map
            vessel_tensor = torch.zeros(1, image_tensor.shape[1], image_tensor.shape[2])
        
        # Additional transforms
        if self.transform is not None:
            image_tensor = self.transform(image_tensor)
            vessel_tensor = self.transform(vessel_tensor)
        
        # Prepare output
        output = {
            'image': image_tensor,
            'vessel': vessel_tensor,
            'label': torch.tensor(label, dtype=torch.long),
            'image_id': image_id
        }
        
        if self.return_vessel_map:
            output['vessel_map'] = vessel_tensor
        
        return output


class APTOSDataset:
    """
    APTOS 2019 Blindness Detection dataset loader.
    
    Dataset structure:
    aptos_2019/
    ├── train_images/
    │   ├── 000c1434d8d7.png
    │   └── ...
    ├── test_images/
    │   ├── 0005cfc8afb2.png
    │   └── ...
    └── train.csv (columns: id_code, diagnosis)
    """
    
    @staticmethod
    def load(
        data_path: str,
        split: str = 'train',
        train_split: float = 0.8,
        val_split: float = 0.1,
        test_split: float = 0.1,
        random_seed: int = 42
    ) -> Tuple[List[str], List[int]]:
        """
        Load APTOS dataset.
        
        Args:
            data_path: Path to APTOS dataset directory
            split: 'train', 'val', or 'test'
            train_split: Fraction for training
            val_split: Fraction for validation
            test_split: Fraction for testing
            random_seed: Random seed for reproducibility
            
        Returns:
            Tuple of (image_paths, labels)
        """
        data_path = Path(data_path)
        
        # Load CSV
        csv_path = data_path / 'train.csv'
        if not csv_path.exists():
            raise FileNotFoundError(f"APTOS train.csv not found at {csv_path}")
        
        df = pd.read_csv(csv_path)
        
        # Add full image paths
        image_dir = data_path / 'train_images'
        df['image_path'] = df['id_code'].apply(
            lambda x: str(image_dir / f"{x}.png")
        )
        
        # Verify images exist
        df = df[df['image_path'].apply(os.path.exists)]
        
        logger.info(f"Loaded {len(df)} samples from APTOS dataset")
        
        # Split dataset
        if split in ['train', 'val', 'test']:
            # Split into train, val, test
            train_df, temp_df = train_test_split(
                df,
                test_size=(val_split + test_split),
                random_state=random_seed,
                stratify=df['diagnosis']
            )
            
            if val_split > 0 and test_split > 0:
                val_df, test_df = train_test_split(
                    temp_df,
                    test_size=test_split / (val_split + test_split),
                    random_state=random_seed,
                    stratify=temp_df['diagnosis']
                )
            else:
                val_df = temp_df if val_split > 0 else pd.DataFrame()
                test_df = temp_df if test_split > 0 else pd.DataFrame()
            
            # Select appropriate split
            if split == 'train':
                selected_df = train_df
            elif split == 'val':
                selected_df = val_df
            else:  # test
                selected_df = test_df
            
            image_paths = selected_df['image_path'].tolist()
            labels = selected_df['diagnosis'].tolist()
            
            logger.info(f"APTOS {split} split: {len(image_paths)} samples")
            
            return image_paths, labels
        
        else:
            raise ValueError(f"Invalid split: {split}")


class EyePACSDataset:
    """
    EyePACS Diabetic Retinopathy Detection dataset loader.
    
    Dataset structure:
    eyepacs/
    ├── train/
    │   ├── 10_left.jpeg
    │   └── ...
    ├── test/
    │   ├── 10000_left.jpeg
    │   └── ...
    ├── trainLabels.csv (columns: image, level)
    └── retinopathy_solution.csv (test labels)
    """
    
    @staticmethod
    def load(
        data_path: str,
        split: str = 'train',
        train_split: float = 0.85,
        val_split: float = 0.15,
        random_seed: int = 42,
        sample_fraction: Optional[float] = None
    ) -> Tuple[List[str], List[int]]:
        """
        Load EyePACS dataset.
        
        Args:
            data_path: Path to EyePACS dataset directory
            split: 'train', 'val', or 'test'
            train_split: Fraction for training
            val_split: Fraction for validation
            random_seed: Random seed
            sample_fraction: Optionally subsample (for faster experiments)
            
        Returns:
            Tuple of (image_paths, labels)
        """
        data_path = Path(data_path)
        
        # Load appropriate CSV
        if split in ['train', 'val']:
            csv_path = data_path / 'trainLabels.csv'
            image_dir = data_path / 'train'
        else:  # test
            csv_path = data_path / 'retinopathy_solution.csv'
            image_dir = data_path / 'test'
        
        if not csv_path.exists():
            raise FileNotFoundError(f"EyePACS CSV not found at {csv_path}")
        
        df = pd.read_csv(csv_path)
        
        # Rename columns for consistency
        if 'level' in df.columns:
            df = df.rename(columns={'level': 'diagnosis'})
        elif 'Level' in df.columns:
            df = df.rename(columns={'Level': 'diagnosis'})
        
        # Add full image paths (try different extensions)
        def find_image_path(image_name, image_dir):
            for ext in ['.jpeg', '.jpg', '.png']:
                path = image_dir / f"{image_name}{ext}"
                if path.exists():
                    return str(path)
            return None
        
        df['image_path'] = df['image'].apply(
            lambda x: find_image_path(x, image_dir)
        )
        
        # Filter out missing images
        df = df[df['image_path'].notna()]
        
        # Optional subsampling
        if sample_fraction is not None and sample_fraction < 1.0:
            df = df.sample(frac=sample_fraction, random_state=random_seed)
            logger.info(f"Subsampled EyePACS to {len(df)} samples ({sample_fraction*100}%)")
        
        logger.info(f"Loaded {len(df)} samples from EyePACS dataset")
        
        # Split dataset
        if split in ['train', 'val']:
            train_df, val_df = train_test_split(
                df,
                test_size=val_split,
                random_state=random_seed,
                stratify=df['diagnosis']
            )
            
            selected_df = train_df if split == 'train' else val_df
        else:  # test
            selected_df = df
        
        image_paths = selected_df['image_path'].tolist()
        labels = selected_df['diagnosis'].tolist()
        
        logger.info(f"EyePACS {split} split: {len(image_paths)} samples")
        
        return image_paths, labels


class Messidor2Dataset:
    """
    Messidor-2 dataset loader.
    
    Used primarily for external validation.
    
    Dataset structure:
    messidor2/
    ├── images/
    │   ├── 20051019_38557_0100_PP.png
    │   └── ...
    └── messidor2.csv (columns: image_id, adjudicated_dr_grade)
    """
    
    @staticmethod
    def load(
        data_path: str,
        split: str = 'test'
    ) -> Tuple[List[str], List[int]]:
        """
        Load Messidor-2 dataset.
        
        Args:
            data_path: Path to Messidor-2 dataset directory
            split: Always 'test' for external validation
            
        Returns:
            Tuple of (image_paths, labels)
        """
        data_path = Path(data_path)
        
        # Load CSV
        csv_path = data_path / 'messidor2.csv'
        if not csv_path.exists():
            # Try alternative naming
            csv_path = data_path / 'messidor_data.csv'
        
        if not csv_path.exists():
            raise FileNotFoundError(f"Messidor-2 CSV not found at {data_path}")
        
        df = pd.read_csv(csv_path)
        
        # Find image directory
        image_dir = data_path / 'images'
        if not image_dir.exists():
            image_dir = data_path
        
        # Add full image paths
        def find_messidor_image(image_id, image_dir):
            for ext in ['.png', '.jpg', '.jpeg', '.tif']:
                path = image_dir / f"{image_id}{ext}"
                if path.exists():
                    return str(path)
            return None
        
        # Try different column names
        if 'image_id' in df.columns:
            id_col = 'image_id'
        elif 'Image name' in df.columns:
            id_col = 'Image name'
        else:
            id_col = df.columns[0]
        
        if 'adjudicated_dr_grade' in df.columns:
            label_col = 'adjudicated_dr_grade'
        elif 'Retinopathy grade' in df.columns:
            label_col = 'Retinopathy grade'
        else:
            label_col = df.columns[1]
        
        df['image_path'] = df[id_col].apply(
            lambda x: find_messidor_image(x, image_dir)
        )
        
        # Filter out missing images
        df = df[df['image_path'].notna()]
        
        logger.info(f"Loaded {len(df)} samples from Messidor-2 dataset")
        
        image_paths = df['image_path'].tolist()
        labels = df[label_col].tolist()
        
        return image_paths, labels


class VesselSegmentationDataset(Dataset):
    """
    Dataset for vessel segmentation (DRIVE, CHASE_DB1).
    """
    
    def __init__(
        self,
        image_paths: List[str],
        mask_paths: List[str],
        preprocessor: Optional[Callable] = None,
        augmentation: Optional[Callable] = None,
        transform: Optional[Callable] = None
    ):
        """
        Initialize vessel segmentation dataset.
        
        Args:
            image_paths: List of fundus image paths
            mask_paths: List of vessel mask paths
            preprocessor: Preprocessing function
            augmentation: Augmentation function
            transform: Additional transforms
        """
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.preprocessor = preprocessor
        self.augmentation = augmentation
        self.transform = transform
        
        assert len(self.image_paths) == len(self.mask_paths), \
            "Number of images and masks must match"
        
        logger.info(f"Initialized VesselSegmentationDataset with {len(self.image_paths)} samples")
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Dict:
        """Get a single sample."""
        # Load image and mask
        image = cv2.imread(self.image_paths[idx])
        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)
        
        if image is None or mask is None:
            raise ValueError(f"Failed to load image or mask at index {idx}")
        
        # Preprocess
        if self.preprocessor is not None:
            image = self.preprocessor.preprocess(image)
        else:
            image = cv2.resize(image, (512, 512))
            image = image.astype(np.float32) / 255.0
        
        # Resize mask
        mask = cv2.resize(mask, (512, 512), interpolation=cv2.INTER_NEAREST)
        mask = (mask > 127).astype(np.float32)  # Binary mask
        
        # Apply augmentation
        if self.augmentation is not None:
            augmented = self.augmentation(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        
        # Convert to tensors
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float()
        mask_tensor = torch.from_numpy(mask).unsqueeze(0).float()
        
        return {
            'image': image_tensor,
            'mask': mask_tensor,
            'image_id': Path(self.image_paths[idx]).stem
        }


def create_dataloaders(
    config: Dict,
    preprocessor,
    augmentation,
    vessel_cache=None
) -> Dict[str, DataLoader]:
    """
    Create train, val, and test dataloaders from configuration.
    
    Args:
        config: Configuration dictionary
        preprocessor: Preprocessing function
        augmentation: Augmentation function
        vessel_cache: VesselCache instance
        
    Returns:
        Dictionary with 'train', 'val', 'test' dataloaders
    """
    data_config = config.get('data', {})
    datasets_config = data_config.get('datasets', {})
    
    # Collect all dataset splits
    all_train_paths, all_train_labels = [], []
    all_val_paths, all_val_labels = [], []
    all_test_paths, all_test_labels = [], []
    
    # Load APTOS
    if datasets_config.get('aptos', {}).get('enabled', False):
        aptos_config = datasets_config['aptos']
        aptos_path = aptos_config['path']
        
        train_paths, train_labels = APTOSDataset.load(
            aptos_path,
            split='train',
            train_split=aptos_config.get('train_split', 0.8),
            val_split=aptos_config.get('val_split', 0.1),
            test_split=aptos_config.get('test_split', 0.1)
        )
        val_paths, val_labels = APTOSDataset.load(
            aptos_path,
            split='val',
            train_split=aptos_config.get('train_split', 0.8),
            val_split=aptos_config.get('val_split', 0.1),
            test_split=aptos_config.get('test_split', 0.1)
        )
        test_paths, test_labels = APTOSDataset.load(
            aptos_path,
            split='test',
            train_split=aptos_config.get('train_split', 0.8),
            val_split=aptos_config.get('val_split', 0.1),
            test_split=aptos_config.get('test_split', 0.1)
        )
        
        all_train_paths.extend(train_paths)
        all_train_labels.extend(train_labels)
        all_val_paths.extend(val_paths)
        all_val_labels.extend(val_labels)
        all_test_paths.extend(test_paths)
        all_test_labels.extend(test_labels)
    
    # Load EyePACS
    if datasets_config.get('eyepacs', {}).get('enabled', False):
        eyepacs_config = datasets_config['eyepacs']
        eyepacs_path = eyepacs_config['path']
        
        if os.path.exists(eyepacs_path):
            train_paths, train_labels = EyePACSDataset.load(
                eyepacs_path,
                split='train',
                train_split=eyepacs_config.get('train_split', 0.85),
                val_split=eyepacs_config.get('val_split', 0.15)
            )
            val_paths, val_labels = EyePACSDataset.load(
                eyepacs_path,
                split='val',
                train_split=eyepacs_config.get('train_split', 0.85),
                val_split=eyepacs_config.get('val_split', 0.15)
            )
            
            all_train_paths.extend(train_paths)
            all_train_labels.extend(train_labels)
            all_val_paths.extend(val_paths)
            all_val_labels.extend(val_labels)
        else:
            logger.warning(f"EyePACS path not found: {eyepacs_path}")
    
    # Load Messidor-2 (validation only)
    if datasets_config.get('messidor2', {}).get('enabled', False):
        messidor2_config = datasets_config['messidor2']
        messidor2_path = messidor2_config['path']
        
        if os.path.exists(messidor2_path):
            test_paths, test_labels = Messidor2Dataset.load(messidor2_path)
            # Messidor-2 is external validation, can add to test or val
            all_test_paths.extend(test_paths)
            all_test_labels.extend(test_labels)
        else:
            logger.warning(f"Messidor-2 path not found: {messidor2_path}")
    
    # Create datasets
    model_config = config.get('model', {})
    use_vessel = model_config.get('use_vessel_channel', True)
    
    train_dataset = DRClassificationDataset(
        all_train_paths,
        all_train_labels,
        preprocessor=preprocessor,
        augmentation=augmentation,  # Augmentation only for training
        vessel_cache=vessel_cache,
        use_vessel_channel=use_vessel
    )
    
    val_dataset = DRClassificationDataset(
        all_val_paths,
        all_val_labels,
        preprocessor=preprocessor,
        augmentation=None,  # No augmentation for validation
        vessel_cache=vessel_cache,
        use_vessel_channel=use_vessel
    )
    
    test_dataset = DRClassificationDataset(
        all_test_paths,
        all_test_labels,
        preprocessor=preprocessor,
        augmentation=None,
        vessel_cache=vessel_cache,
        use_vessel_channel=use_vessel
    )
    
    # Create dataloaders
    batch_size = data_config.get('batch_size', 32)
    num_workers = data_config.get('num_workers', 4)
    pin_memory = data_config.get('pin_memory', True)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    logger.info(f"Created dataloaders: train={len(train_loader)} batches, "
               f"val={len(val_loader)} batches, test={len(test_loader)} batches")
    
    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }


def get_class_weights(labels: List[int], num_classes: int = 5) -> torch.Tensor:
    """
    Calculate class weights for imbalanced datasets.
    
    Args:
        labels: List of labels
        num_classes: Number of classes
        
    Returns:
        Tensor of class weights
    """
    class_counts = np.bincount(labels, minlength=num_classes)
    total_samples = len(labels)
    
    # Inverse frequency weighting
    class_weights = total_samples / (num_classes * class_counts + 1e-6)
    
    return torch.FloatTensor(class_weights)

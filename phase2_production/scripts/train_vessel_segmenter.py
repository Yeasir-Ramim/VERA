"""
VERA Phase 2 - Vessel Segmentation Training Script

Train or fine-tune U-Net vessel segmentation model on DRIVE/CHASE_DB1 datasets.
"""

import os
import sys
import argparse
import yaml
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.vessel_segmentation import (
    create_vessel_segmenter,
    train_vessel_segmenter,
    VesselSegmentationLoss
)
from src.data.datasets import VesselSegmentationDataset
from src.data.preprocessing import create_vessel_preprocessor
from src.data.augmentation import create_vessel_augmentation
from src.utils.logging import setup_logging
from src.utils.config import load_config
from src.utils.random import set_seed

logger = logging.getLogger(__name__)


def load_vessel_dataset(
    dataset_path: str,
    preprocessor,
    augmentation=None
):
    """
    Load DRIVE or CHASE_DB1 vessel segmentation dataset.
    
    Args:
        dataset_path: Path to dataset directory
        preprocessor: Preprocessing function
        augmentation: Augmentation function
        
    Returns:
        VesselSegmentationDataset instance
    """
    dataset_path = Path(dataset_path)
    
    # Find images and masks
    image_paths = []
    mask_paths = []
    
    # Try DRIVE structure
    train_images_dir = dataset_path / 'training' / 'images'
    train_masks_dir = dataset_path / 'training' / '1st_manual'
    
    if not train_images_dir.exists():
        # Try CHASE_DB1 structure
        train_images_dir = dataset_path / 'images'
        train_masks_dir = dataset_path / 'masks'
    
    if not train_images_dir.exists():
        raise FileNotFoundError(f"Could not find images in {dataset_path}")
    
    # Collect image and mask pairs
    for img_path in sorted(train_images_dir.glob('*')):
        if img_path.suffix.lower() in ['.png', '.jpg', '.jpeg', '.tif', '.tiff']:
            # Find corresponding mask
            img_stem = img_path.stem
            
            # Try different mask naming patterns
            mask_patterns = [
                f"{img_stem}_manual1*",
                f"{img_stem}_1stHO*",
                f"{img_stem}*",
            ]
            
            mask_found = False
            for pattern in mask_patterns:
                mask_candidates = list(train_masks_dir.glob(pattern))
                if mask_candidates:
                    image_paths.append(str(img_path))
                    mask_paths.append(str(mask_candidates[0]))
                    mask_found = True
                    break
            
            if not mask_found:
                logger.warning(f"No mask found for {img_path.name}")
    
    if len(image_paths) == 0:
        raise ValueError(f"No valid image-mask pairs found in {dataset_path}")
    
    logger.info(f"Loaded {len(image_paths)} image-mask pairs from {dataset_path}")
    
    dataset = VesselSegmentationDataset(
        image_paths=image_paths,
        mask_paths=mask_paths,
        preprocessor=preprocessor,
        augmentation=augmentation
    )
    
    return dataset


def main():
    parser = argparse.ArgumentParser(
        description='VERA Phase 2 - Train Vessel Segmentation Model'
    )
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--drive_path', type=str, default='data/raw/DRIVE',
                       help='Path to DRIVE dataset')
    parser.add_argument('--chase_path', type=str, default='data/raw/CHASE_DB1',
                       help='Path to CHASE_DB1 dataset')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.0001,
                       help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device for training')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--output_dir', type=str, default='models/checkpoints',
                       help='Output directory for checkpoints')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(log_file='outputs/logs/train_vessel_segmenter.log')
    
    logger.info("="*60)
    logger.info("VERA PHASE 2 - VESSEL SEGMENTATION TRAINING")
    logger.info("="*60)
    
    # Set random seed
    set_seed(args.seed)
    
    # Load configuration
    config = load_config(args.config)
    
    # Update config with command line args
    if 'vessel_segmentation' not in config:
        config['vessel_segmentation'] = {}
    
    config['vessel_segmentation']['epochs'] = args.epochs
    config['vessel_segmentation']['learning_rate'] = args.learning_rate
    
    # Set device
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA not available, using CPU")
        device = 'cpu'
    
    logger.info(f"Using device: {device}")
    
    # Create preprocessor and augmentation
    preprocessor = create_vessel_preprocessor(config)
    augmentation = create_vessel_augmentation(config)
    
    # Load datasets
    train_datasets = []
    
    # DRIVE dataset
    if Path(args.drive_path).exists():
        logger.info(f"\nLoading DRIVE dataset from {args.drive_path}")
        try:
            drive_dataset = load_vessel_dataset(
                args.drive_path,
                preprocessor=preprocessor,
                augmentation=augmentation
            )
            train_datasets.append(drive_dataset)
        except Exception as e:
            logger.error(f"Error loading DRIVE: {e}")
    else:
        logger.warning(f"DRIVE dataset not found at {args.drive_path}")
    
    # CHASE_DB1 dataset
    if Path(args.chase_path).exists():
        logger.info(f"\nLoading CHASE_DB1 dataset from {args.chase_path}")
        try:
            chase_dataset = load_vessel_dataset(
                args.chase_path,
                preprocessor=preprocessor,
                augmentation=augmentation
            )
            train_datasets.append(chase_dataset)
        except Exception as e:
            logger.error(f"Error loading CHASE_DB1: {e}")
    else:
        logger.warning(f"CHASE_DB1 dataset not found at {args.chase_path}")
    
    if len(train_datasets) == 0:
        logger.error("No vessel segmentation datasets loaded. Exiting.")
        return
    
    # Combine datasets
    from torch.utils.data import ConcatDataset
    combined_dataset = ConcatDataset(train_datasets)
    
    logger.info(f"\nTotal training samples: {len(combined_dataset)}")
    
    # Split into train and validation
    from torch.utils.data import random_split
    val_size = int(0.2 * len(combined_dataset))
    train_size = len(combined_dataset) - val_size
    
    train_dataset, val_dataset = random_split(
        combined_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed)
    )
    
    logger.info(f"Training samples: {len(train_dataset)}")
    logger.info(f"Validation samples: {len(val_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Create model
    logger.info("\nCreating vessel segmentation model...")
    model = create_vessel_segmenter(config)
    logger.info(f"Model parameters: {model.get_trainable_params():,}")
    
    # Train model
    logger.info("\nStarting training...")
    history = train_vessel_segmenter(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.epochs,
        learning_rate=args.learning_rate,
        device=device,
        checkpoint_dir=args.output_dir,
        save_best_only=True
    )
    
    # Plot training history
    try:
        import matplotlib.pyplot as plt
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss plot
        ax1.plot(history['train_loss'], label='Train Loss', marker='o')
        ax1.plot(history['val_loss'], label='Val Loss', marker='s')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Dice plot
        ax2.plot(history['val_dice'], label='Val Dice', marker='o', color='green')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Dice Coefficient')
        ax2.set_title('Validation Dice Coefficient')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        output_path = Path(args.output_dir) / 'vessel_training_history.png'
        plt.savefig(output_path, dpi=150)
        logger.info(f"\nSaved training history plot to {output_path}")
        
    except Exception as e:
        logger.warning(f"Could not plot training history: {e}")
    
    logger.info("\n" + "="*60)
    logger.info("TRAINING COMPLETE")
    logger.info("="*60)
    logger.info(f"Best validation loss: {min(history['val_loss']):.4f}")
    logger.info(f"Best validation Dice: {max(history['val_dice']):.4f}")
    logger.info(f"Model saved to: {args.output_dir}/vessel_segmentation_best.pth")


if __name__ == '__main__':
    main()

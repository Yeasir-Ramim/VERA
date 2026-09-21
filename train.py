"""
VERA Training Script - Production DR Classification System
Trains vessel-aware 4-channel CNN for diabetic retinopathy detection.

Usage:
    python train.py --dataset aptos --epochs 50 --batch_size 32
    python train.py --dataset eyepacs --backbone efficientnet_b3 --fusion attention_gated
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    cohen_kappa_score,
    roc_auc_score,
    accuracy_score
)

# Import VERA modules
from src.dataset import FundusDataset, get_train_transform, get_val_transform
from src.models import VesselAwareClassifier
from src.vessel_segmentation import VesselSegmenter
from src.preprocessing import preprocess_fundus
from src.utils import set_seed, save_checkpoint, load_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description='Train VERA DR Classification Model')
    
    # Dataset arguments
    parser.add_argument('--dataset', type=str, default='aptos',
                       choices=['aptos', 'eyepacs', 'messidor2'],
                       help='Dataset to use')
    parser.add_argument('--data_dir', type=str, default='data/raw',
                       help='Path to dataset directory')
    parser.add_argument('--vessel_cache_dir', type=str, default='data/processed/vessel_cache',
                       help='Directory for cached vessel maps')
    
    # Model arguments
    parser.add_argument('--backbone', type=str, default='resnet50',
                       choices=['resnet18', 'resnet50', 'efficientnet_b0', 'efficientnet_b3'],
                       help='Backbone architecture')
    parser.add_argument('--fusion', type=str, default='early',
                       choices=['early', 'dual_branch', 'attention_gated'],
                       help='Fusion strategy for vessel channel')
    parser.add_argument('--pretrained', action='store_true', default=True,
                       help='Use ImageNet pretrained weights')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='Weight decay')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    
    # Loss function
    parser.add_argument('--loss', type=str, default='ce',
                       choices=['ce', 'focal', 'qwk'],
                       help='Loss function (ce=cross-entropy, focal=focal loss, qwk=QWK loss)')
    parser.add_argument('--class_weights', action='store_true', default=True,
                       help='Use class weights for imbalanced data')
    
    # Optimization
    parser.add_argument('--optimizer', type=str, default='adamw',
                       choices=['adam', 'adamw', 'sgd'],
                       help='Optimizer')
    parser.add_argument('--scheduler', type=str, default='cosine',
                       choices=['none', 'step', 'cosine', 'plateau'],
                       help='Learning rate scheduler')
    parser.add_argument('--mixed_precision', action='store_true', default=True,
                       help='Use mixed precision training')
    
    # Regularization
    parser.add_argument('--dropout', type=float, default=0.3,
                       help='Dropout rate')
    parser.add_argument('--label_smoothing', type=float, default=0.1,
                       help='Label smoothing factor')
    
    # Data split
    parser.add_argument('--train_split', type=float, default=0.8,
                       help='Training set proportion')
    parser.add_argument('--val_split', type=float, default=0.1,
                       help='Validation set proportion')
    
    # Checkpoint and logging
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                       help='Directory to save checkpoints')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--save_every', type=int, default=5,
                       help='Save checkpoint every N epochs')
    
    # Other
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cuda/cpu)')
    
    return parser.parse_args()


def load_dataset(args) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load and split dataset into train/val/test."""
    data_path = Path(args.data_dir)
    
    if args.dataset == 'aptos':
        csv_file = data_path / 'aptos2019' / 'train.csv'
        image_dir = data_path / 'aptos2019' / 'train_images'
        label_col = 'diagnosis'
        id_col = 'id_code'
        
    elif args.dataset == 'eyepacs':
        csv_file = data_path / 'eyepacs' / 'trainLabels.csv'
        image_dir = data_path / 'eyepacs' / 'train'
        label_col = 'level'
        id_col = 'image'
        
    elif args.dataset == 'messidor2':
        csv_file = data_path / 'messidor2' / 'messidor2_annotations.csv'
        image_dir = data_path / 'messidor2' / 'images'
        label_col = 'grade'
        id_col = 'image_id'
    
    if not csv_file.exists():
        raise FileNotFoundError(
            f"Dataset CSV not found: {csv_file}\n"
            f"Please run: python setup_datasets.py --dataset {args.dataset}"
        )
    
    # Load CSV
    df = pd.read_csv(csv_file)
    df['image_path'] = df[id_col].apply(lambda x: str(image_dir / f"{x}.png"))
    df['label'] = df[label_col]
    
    # Check if images exist
    missing_count = 0
    for path in df['image_path'][:100]:  # Check first 100
        if not Path(path).exists():
            missing_count += 1
    
    if missing_count > 0:
        raise FileNotFoundError(
            f"Images not found in {image_dir}\n"
            f"Please download and extract the dataset properly."
        )
    
    # Stratified split
    from sklearn.model_selection import train_test_split
    
    train_df, temp_df = train_test_split(
        df, 
        test_size=(1 - args.train_split),
        stratify=df['label'],
        random_state=args.seed
    )
    
    val_size = args.val_split / (args.val_split + (1 - args.train_split - args.val_split))
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1 - val_size),
        stratify=temp_df['label'],
        random_state=args.seed
    )
    
    print(f"\nDataset: {args.dataset}")
    print(f"Total images: {len(df)}")
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    print(f"\nClass distribution (train):")
    print(train_df['label'].value_counts().sort_index())
    
    return train_df, val_df, test_df


def get_class_weights(train_df: pd.DataFrame, num_classes: int = 5) -> torch.Tensor:
    """Compute class weights for imbalanced dataset."""
    class_counts = train_df['label'].value_counts().sort_index().values
    total = len(train_df)
    weights = total / (num_classes * class_counts)
    return torch.FloatTensor(weights)


def get_dataloaders(args, train_df, val_df, test_df, vessel_segmenter):
    """Create data loaders."""
    
    # Create datasets
    train_dataset = FundusDataset(
        train_df,
        vessel_segmenter=vessel_segmenter,
        vessel_cache_dir=args.vessel_cache_dir,
        transform=get_train_transform(),
        target_size=(224, 224)
    )
    
    val_dataset = FundusDataset(
        val_df,
        vessel_segmenter=vessel_segmenter,
        vessel_cache_dir=args.vessel_cache_dir,
        transform=get_val_transform(),
        target_size=(224, 224)
    )
    
    test_dataset = FundusDataset(
        test_df,
        vessel_segmenter=vessel_segmenter,
        vessel_cache_dir=args.vessel_cache_dir,
        transform=get_val_transform(),
        target_size=(224, 224)
    )
    
    # Create weighted sampler for train set (handle class imbalance)
    if args.class_weights:
        class_counts = train_df['label'].value_counts().sort_index().values
        sample_weights = [1.0 / class_counts[label] for label in train_df['label']]
        sampler = WeightedRandomSampler(
            sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
        shuffle = False
    else:
        sampler = None
        shuffle = True
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


def get_loss_function(args, class_weights=None):
    """Get loss function."""
    if args.loss == 'ce':
        if class_weights is not None and args.class_weights:
            criterion = nn.CrossEntropyLoss(
                weight=class_weights,
                label_smoothing=args.label_smoothing
            )
        else:
            criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    
    elif args.loss == 'focal':
        from src.losses import FocalLoss
        criterion = FocalLoss(
            alpha=class_weights if args.class_weights else None,
            gamma=2.0
        )
    
    elif args.loss == 'qwk':
        from src.losses import QWKLoss
        criterion = QWKLoss(num_classes=5)
    
    return criterion


def train_one_epoch(model, train_loader, criterion, optimizer, device, scaler, epoch):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch} [Train]')
    
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        # Mixed precision training
        with autocast(enabled=(scaler is not None)):
            outputs = model(images)
            loss = criterion(outputs, labels)
        
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': running_loss / (pbar.n + 1),
            'acc': 100. * correct / total
        })
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc


@torch.no_grad()
def validate(model, val_loader, criterion, device, epoch):
    """Validate model."""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(val_loader, desc=f'Epoch {epoch} [Val]')
    
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        pbar.set_postfix({'loss': running_loss / (pbar.n + 1)})
    
    # Compute metrics
    val_loss = running_loss / len(val_loader)
    val_acc = accuracy_score(all_labels, all_preds) * 100
    val_kappa = cohen_kappa_score(all_labels, all_preds, weights='quadratic')
    
    return val_loss, val_acc, val_kappa, all_preds, all_labels


def train(args):
    """Main training function."""
    
    # Set random seeds
    set_seed(args.seed)
    
    # Device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Create checkpoint directory
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    print("\nLoading dataset...")
    train_df, val_df, test_df = load_dataset(args)
    
    # Initialize vessel segmenter
    print("\nInitializing vessel segmenter...")
    vessel_segmenter = VesselSegmenter(cache_dir=args.vessel_cache_dir)
    
    # Create data loaders
    print("\nCreating data loaders...")
    train_loader, val_loader, test_loader = get_dataloaders(
        args, train_df, val_df, test_df, vessel_segmenter
    )
    
    # Get class weights
    class_weights = None
    if args.class_weights:
        class_weights = get_class_weights(train_df).to(device)
        print(f"\nClass weights: {class_weights}")
    
    # Create model
    print(f"\nCreating model: {args.backbone} with {args.fusion} fusion...")
    model = VesselAwareClassifier(
        backbone=args.backbone,
        fusion_type=args.fusion,
        num_classes=5,
        pretrained=args.pretrained,
        dropout=args.dropout
    )
    model = model.to(device)
    
    # Loss function
    criterion = get_loss_function(args, class_weights)
    
    # Optimizer
    if args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    
    # Scheduler
    if args.scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    elif args.scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    elif args.scheduler == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5)
    else:
        scheduler = None
    
    # Mixed precision scaler
    scaler = GradScaler() if args.mixed_precision else None
    
    # Resume from checkpoint
    start_epoch = 0
    best_val_kappa = 0.0
    
    if args.resume:
        checkpoint = load_checkpoint(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_kappa = checkpoint.get('best_val_kappa', 0.0)
        print(f"\nResumed from epoch {start_epoch}")
    
    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...")
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'val_kappa': []
    }
    
    for epoch in range(start_epoch, args.epochs):
        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler, epoch + 1
        )
        
        # Validate
        val_loss, val_acc, val_kappa, val_preds, val_labels = validate(
            model, val_loader, criterion, device, epoch + 1
        )
        
        # Update scheduler
        if scheduler is not None:
            if args.scheduler == 'plateau':
                scheduler.step(val_loss)
            else:
                scheduler.step()
        
        # Log results
        print(f"\nEpoch {epoch + 1}/{args.epochs}:")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, Val Kappa: {val_kappa:.4f}")
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_kappa'].append(val_kappa)
        
        # Save checkpoint
        is_best = val_kappa > best_val_kappa
        if is_best:
            best_val_kappa = val_kappa
        
        if (epoch + 1) % args.save_every == 0 or is_best:
            checkpoint_path = checkpoint_dir / f'checkpoint_epoch_{epoch + 1}.pth'
            save_checkpoint(
                model, optimizer, epoch, val_loss, val_kappa,
                checkpoint_path, is_best=is_best
            )
            
            if is_best:
                best_path = checkpoint_dir / 'best_model.pth'
                save_checkpoint(
                    model, optimizer, epoch, val_loss, val_kappa,
                    best_path, is_best=True
                )
                print(f"  ✓ New best model saved! (Kappa: {val_kappa:.4f})")
    
    # Save training history
    history_path = checkpoint_dir / 'training_history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\n✓ Training complete!")
    print(f"Best validation kappa: {best_val_kappa:.4f}")
    print(f"Best model saved to: {checkpoint_dir / 'best_model.pth'}")
    
    return model, history


if __name__ == '__main__':
    args = parse_args()
    model, history = train(args)

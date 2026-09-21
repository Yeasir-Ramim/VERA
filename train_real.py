"""
VERA Real Dataset Training Script
Simplified version that works with existing src modules
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix
from sklearn.model_selection import train_test_split

# Import from existing src modules
from src.dataset import APTOSDataset, compute_class_weights
from src.models import build_model
from src.vessel_segmentation import VesselSegmenter
from src.utils import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description='Train VERA on Real Dataset')
    parser.add_argument('--dataset', type=str, default='aptos')
    parser.add_argument('--data_dir', type=str, default='data/raw')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--backbone', type=str, default='resnet50')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    print(f"Training on {args.dataset.upper()} dataset")
    print(f"Epochs: {args.epochs}, Batch size: {args.batch_size}")
    print(f"Backbone: {args.backbone}\n")
    
    # Load dataset
    data_path = Path(args.data_dir) / 'aptos2019'
    csv_file = data_path / 'train.csv'
    image_dir = data_path / 'train_images'
    
    if not csv_file.exists():
        print(f"ERROR: Dataset not found at {csv_file}")
        return
    
    df = pd.read_csv(csv_file)
    print(f"Loaded {len(df)} images")
    print(f"Class distribution:\n{df['diagnosis'].value_counts().sort_index()}\n")
    
    # Split dataset
    train_df, val_df = train_test_split(
        df, test_size=0.2, random_state=args.seed, stratify=df['diagnosis']
    )
    
    print(f"Train: {len(train_df)}, Val: {len(val_df)}\n")
    
    # Create datasets
    vessel_segmenter = VesselSegmenter(cache_dir='data/processed/vessel_cache')
    
    train_dataset = APTOSDataset(
        df=train_df,
        image_dir=image_dir,
        target_size=(224, 224),
        is_4channel=True,
        vessel_segmenter=vessel_segmenter,
        is_training=True
    )
    
    val_dataset = APTOSDataset(
        df=val_df,
        image_dir=image_dir,
        target_size=(224, 224),
        is_4channel=True,
        vessel_segmenter=vessel_segmenter,
        is_training=False
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    # Create model
    print("Creating model...")
    model = build_model(
        model_type='early_fusion',
        backbone=args.backbone,
        pretrained=True,
        num_classes=5
    )
    model = model.to(device)
    
    # Loss and optimizer
    class_weights = compute_class_weights(train_df, target_col='diagnosis')
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Training loop
    best_kappa = 0.0
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'val_kappa': []}
    
    print("\nStarting training...\n")
    
    for epoch in range(args.epochs):
        # Train
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.epochs} [Train]')
        for images, labels, _ in pbar:
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({
                'loss': train_loss / (pbar.n + 1),
                'acc': 100. * train_correct / train_total
            })
        
        train_loss /= len(train_loader)
        train_acc = 100. * train_correct / train_total
        
        # Validate
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{args.epochs} [Val]')
            for images, labels, _ in pbar:
                images = images.to(device)
                labels = labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                pbar.set_postfix({'loss': val_loss / (pbar.n + 1)})
        
        val_loss /= len(val_loader)
        val_acc = accuracy_score(all_labels, all_preds) * 100
        val_kappa = cohen_kappa_score(all_labels, all_preds, weights='quadratic')
        
        scheduler.step()
        
        # Log
        print(f"\nEpoch {epoch+1}/{args.epochs}:")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, Val Kappa: {val_kappa:.4f}")
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_kappa'].append(val_kappa)
        
        # Save best model
        if val_kappa > best_kappa:
            best_kappa = val_kappa
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_kappa': val_kappa,
                'val_acc': val_acc,
                'backbone': args.backbone
            }
            torch.save(checkpoint, checkpoint_dir / 'best_model.pth')
            print(f"  ✓ New best model saved! (Kappa: {val_kappa:.4f})")
        
        print()
    
    # Save history
    with open(checkpoint_dir / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\n✓ Training complete!")
    print(f"Best validation kappa: {best_kappa:.4f}")
    print(f"Model saved to: {checkpoint_dir / 'best_model.pth'}")


if __name__ == '__main__':
    main()

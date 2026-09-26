"""
Resume VERA Training from Checkpoint
Continues training from where it left off
"""

import argparse
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
from sklearn.metrics import accuracy_score, cohen_kappa_score
from sklearn.model_selection import train_test_split

# Import from existing src modules
from src.dataset import APTOSDataset, compute_class_weights
from src.models import build_model
from src.vessel_segmentation import VesselSegmenter
from src.utils import set_seed, save_checkpoint, load_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description='Resume VERA Training')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/production_model/best_model.pth')
    parser.add_argument('--dataset', type=str, default='aptos')
    parser.add_argument('--data_dir', type=str, default='data/raw')
    parser.add_argument('--total_epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/production_model')
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(dataloader, desc=f"[Train]")
    for batch_idx, (inputs, labels, _) in enumerate(pbar):
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        preds = outputs.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
        
        acc = accuracy_score(all_labels, all_preds)
        pbar.set_postfix({'loss': running_loss / (batch_idx + 1), 'acc': acc * 100})
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = accuracy_score(all_labels, all_preds)
    return epoch_loss, epoch_acc


def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f"[Val]")
        for batch_idx, (inputs, labels, _) in enumerate(pbar):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            
            pbar.set_postfix({'loss': running_loss / (batch_idx + 1)})
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = accuracy_score(all_labels, all_preds)
    epoch_kappa = cohen_kappa_score(all_labels, all_preds, weights='quadratic')
    
    return epoch_loss, epoch_acc, epoch_kappa


def main():
    args = parse_args()
    set_seed(args.seed)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print(f"🔄 RESUMING VERA TRAINING")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Target Total Epochs: {args.total_epochs}")
    print(f"Batch size: {args.batch_size}")
    
    # Load checkpoint to get starting epoch
    print(f"\n📂 Loading checkpoint...")
    ckpt = torch.load(args.checkpoint, map_location=device)
    start_epoch = ckpt['epoch'] + 1
    best_kappa = ckpt.get('val_kappa', 0.0)
    backbone = ckpt.get('backbone', 'resnet50')
    
    print(f"✓ Loaded checkpoint from Epoch {ckpt['epoch']}")
    print(f"✓ Best Kappa so far: {best_kappa:.4f}")
    print(f"✓ Backbone: {backbone}")
    print(f"✓ Resuming from Epoch {start_epoch}")
    
    if start_epoch >= args.total_epochs:
        print(f"\n✅ Training already complete! ({start_epoch}/{args.total_epochs} epochs)")
        return
    
    # Dataset
    print(f"\n📊 Loading {args.dataset.upper()} dataset...")
    dataset_map = {
        'aptos': 'aptos2019',
        'eyepacs': 'eyepacs',
        'messidor2': 'messidor2'
    }
    
    data_path = Path(args.data_dir) / dataset_map[args.dataset]
    
    if args.dataset == 'aptos':
        csv_path = data_path / 'train.csv'
        img_dir = data_path / 'train_images'
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} images")
        print(f"Class distribution:\n{df['diagnosis'].value_counts().sort_index()}")
        
        # Split
        train_df, val_df = train_test_split(df, test_size=0.2, random_state=args.seed, stratify=df['diagnosis'])
        print(f"Train: {len(train_df)}, Val: {len(val_df)}")
        
        # Datasets with vessel segmenter
        vessel_cache_dir = Path("data/processed/vessel_cache")
        vessel_cache_dir.mkdir(parents=True, exist_ok=True)
        vessel_segmenter = VesselSegmenter(cache_dir=str(vessel_cache_dir))
        
        train_dataset = APTOSDataset(train_df, img_dir, is_4channel=True, 
                                     vessel_segmenter=vessel_segmenter, is_training=True)
        val_dataset = APTOSDataset(val_df, img_dir, is_4channel=True, 
                                   vessel_segmenter=vessel_segmenter, is_training=False)
        
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                                 num_workers=0, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, 
                               num_workers=0, pin_memory=True)
    
    # Model
    print(f"\n🧠 Creating model...")
    model = build_model(model_type='early_fusion', backbone=backbone, pretrained=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(device)
    print(f"✓ Model loaded with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    if 'optimizer_state_dict' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        print(f"✓ Optimizer state restored")
    
    # Loss with class weights
    class_weights = compute_class_weights(train_df, target_col='diagnosis')
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    
    # Checkpoint directory
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Training loop
    print(f"\n{'='*60}")
    print(f"🚀 STARTING TRAINING FROM EPOCH {start_epoch}")
    print(f"{'='*60}\n")
    
    for epoch in range(start_epoch, args.total_epochs):
        print(f"Epoch {epoch + 1}/{args.total_epochs}")
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_acc, val_kappa = validate_epoch(model, val_loader, criterion, device)
        
        # Print results
        print(f"Epoch {epoch + 1}/{args.total_epochs}:")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc * 100:.2f}%")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc * 100:.2f}%, Val Kappa: {val_kappa:.4f}")
        
        # Save checkpoint if improved
        if val_kappa > best_kappa:
            best_kappa = val_kappa
            checkpoint_path = checkpoint_dir / 'best_model.pth'
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                val_loss=val_loss,
                val_kappa=val_kappa,
                path=str(checkpoint_path),
                is_best=True
            )
            # Add backbone and accuracy to checkpoint
            ckpt_data = torch.load(checkpoint_path, map_location='cpu')
            ckpt_data['backbone'] = backbone
            ckpt_data['val_acc'] = val_acc * 100
            torch.save(ckpt_data, checkpoint_path)
            print(f"  ✓ New best model saved! (Kappa: {val_kappa:.4f})")
        
        print()
    
    print(f"\n{'='*60}")
    print(f"✅ TRAINING COMPLETE!")
    print(f"{'='*60}")
    print(f"Best Validation Kappa: {best_kappa:.4f}")
    print(f"Final Model: {checkpoint_dir / 'best_model.pth'}")
    print(f"\n🎉 Model ready for deployment!")


if __name__ == '__main__':
    main()

"""
VERA Phase 2 Multi-Dataset Training Script for Google Colab & Kaggle (40,000+ Real Images)

Usage on Google Colab or Kaggle GPU:
    python VERA_Phase2_Colab_Kaggle_Training.py --epochs 30 --batch_size 32 --backbone resnet50
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, cohen_kappa_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

try:
    import kagglehub
except ImportError:
    kagglehub = None


def parse_args():
    parser = argparse.ArgumentParser(description='VERA Multi-Dataset Training on Colab / Kaggle GPU')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--backbone', type=str, default='resnet50')
    parser.add_argument('--output_dir', type=str, default='checkpoints/production_model')
    return parser.parse_args()


def crop_fundus_circle(img: np.ndarray, tol: int = 10) -> np.ndarray:
    if img.ndim == 2:
        mask = img > tol
        if not mask.any(): return img
        return img[np.ix_(mask.any(1), mask.any(0))]
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    mask = gray > tol
    if not mask.any(): return img
    row_idx = mask.any(axis=1)
    col_idx = mask.any(axis=0)
    ymin, ymax = np.where(row_idx)[0][[0, -1]]
    xmin, xmax = np.where(col_idx)[0][[0, -1]]
    return img[max(0, ymin-2):min(gray.shape[0], ymax+2), max(0, xmin-2):min(gray.shape[1], xmax+2)]


def apply_clahe(img: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2RGB)


def extract_vessel_map(img_rgb: np.ndarray) -> np.ndarray:
    green = img_rgb[:, :, 1]
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(green)
    inverted = 255 - enhanced
    blurred = cv2.GaussianBlur(inverted, (5, 5), 0)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    tophat = cv2.morphologyEx(blurred, cv2.MORPH_TOPHAT, kernel)
    return cv2.normalize(tophat, None, 0.0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)


class VERA4ChannelResNet50(nn.Module):
    def __init__(self, num_classes: int = 5, pretrained: bool = True):
        super().__init__()
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
        
        # Modify conv1 to accept 4-channel tensor [R, G, B, V]
        old_conv = self.backbone.conv1
        new_conv = nn.Conv2d(4, old_conv.out_channels, kernel_size=old_conv.kernel_size,
                             stride=old_conv.stride, padding=old_conv.padding, bias=old_conv.bias is not None)
        with torch.no_grad():
            new_conv.weight[:, :3, :, :] = old_conv.weight
            new_conv.weight[:, 3:4, :, :] = old_conv.weight.mean(dim=1, keepdim=True)
        self.backbone.conv1 = new_conv
        
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class VERAMultiDataset(Dataset):
    def __init__(self, df: pd.DataFrame, target_size=(224, 224)):
        self.df = df.reset_index(drop=True)
        self.target_size = target_size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = cv2.imread(row['image_path'])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cropped = crop_fundus_circle(img)
        clahe = apply_clahe(cropped)
        resized = cv2.resize(clahe, (self.target_size[1], self.target_size[0]))
        v_map = extract_vessel_map(resized)
        
        rgb_norm = (resized.astype(np.float32) / 255.0 - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        v_norm = (v_map - 0.15) / 0.25
        
        t_rgb = torch.from_numpy(rgb_norm.transpose(2, 0, 1)).float()
        t_v = torch.from_numpy(v_norm).unsqueeze(0).float()
        tensor_4ch = torch.cat([t_rgb, t_v], dim=0)
        
        label = torch.tensor(row['diagnosis'], dtype=torch.long)
        return tensor_4ch, label


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🚀 VERA Multi-Dataset Training Starting...")
    print(f"Device: {device}")
    print(f"Epochs: {args.epochs} | Batch Size: {args.batch_size} | Backbone: {args.backbone}")
    
    # Dataset Auto-Discovery
    manifest_records = []
    
    # Check APTOS 2019
    aptos_paths = [Path("data/raw/aptos2019"), Path("aptos2019-blindness-detection")]
    if kagglehub:
        try:
            downloaded = kagglehub.dataset_download("mariafe/aptos-2019-blindness-detection")
            aptos_paths.append(Path(downloaded))
        except Exception:
            pass
            
    for p in aptos_paths:
        csv_p = p / "train.csv"
        if csv_p.exists():
            df_a = pd.read_csv(csv_p)
            for _, r in df_a.iterrows():
                ip = p / "train_images" / f"{r['id_code']}.png"
                if ip.exists():
                    manifest_records.append({'image_path': str(ip), 'diagnosis': int(r['diagnosis']), 'dataset': 'aptos'})
            print(f"✓ Loaded {len(manifest_records)} APTOS images from {p}")
            break
            
    if not manifest_records:
        print("⚠️ No real dataset images found. Please ensure APTOS 2019 / EyePACS datasets are in data/raw/")
        return

    df_manifest = pd.DataFrame(manifest_records)
    print(f"\nDataset Total: {len(df_manifest)} images")
    print(df_manifest['diagnosis'].value_counts().sort_index())

    train_df, val_df = train_test_split(df_manifest, test_size=0.2, random_state=42, stratify=df_manifest['diagnosis'])
    train_loader = DataLoader(VERAMultiDataset(train_df), batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(VERAMultiDataset(val_df), batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model = VERA4ChannelResNet50(num_classes=5, pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler()

    best_kappa = 0.0
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()
            pbar.set_postfix({'loss': train_loss / (pbar.n + 1)})

        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                outputs = model(images)
                preds = outputs.argmax(dim=1).cpu().numpy()
                val_preds.extend(preds)
                val_labels.extend(labels.numpy())

        val_acc = accuracy_score(val_labels, val_preds) * 100
        val_kappa = cohen_kappa_score(val_labels, val_preds, weights='quadratic')
        scheduler.step()

        print(f"Epoch {epoch+1}/{args.epochs}: Val Acc = {val_acc:.2f}%, Val QWK Kappa = {val_kappa:.4f}")

        if val_kappa > best_kappa:
            best_kappa = val_kappa
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_kappa': val_kappa,
                'val_acc': val_acc,
                'backbone': args.backbone
            }, output_dir / 'best_model.pth')
            print(f"  ✓ Saved new best model to {output_dir / 'best_model.pth'} (Kappa: {val_kappa:.4f})")

    print("\n✅ Training Complete!")
    print(f"Best Validation Kappa: {best_kappa:.4f}")
    print(f"Model saved at: {output_dir / 'best_model.pth'}")


if __name__ == '__main__':
    main()

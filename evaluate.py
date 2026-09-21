"""
VERA Model Evaluation Script
Comprehensive evaluation with metrics, confusion matrix, and visualizations.

Usage:
    python evaluate.py --checkpoint checkpoints/best_model.pth --dataset aptos
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    cohen_kappa_score,
    roc_auc_score,
    accuracy_score,
    precision_recall_fscore_support
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import FundusDataset, get_val_transform
from src.models import VesselAwareClassifier
from src.vessel_segmentation import VesselSegmenter
from src.explainability import GradCAM, compute_vessel_attention_overlap
from src.utils import load_checkpoint


CLASS_NAMES = ['No DR', 'Mild NPDR', 'Moderate NPDR', 'Severe NPDR', 'Proliferative DR']


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate VERA Model')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--dataset', type=str, default='aptos',
                       choices=['aptos', 'eyepacs', 'messidor2'])
    parser.add_argument('--data_dir', type=str, default='data/raw')
    parser.add_argument('--vessel_cache_dir', type=str, default='data/processed/vessel_cache')
    parser.add_argument('--output_dir', type=str, default='evaluation_results')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--device', type=str, default=None)
    return parser.parse_args()


@torch.no_grad()
def evaluate_model(model, dataloader, device):
    """Run model inference and collect predictions."""
    model.eval()
    
    all_preds = []
    all_probs = []
    all_labels = []
    
    for images, labels in tqdm(dataloader, desc='Evaluating'):
        images = images.to(device)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)
        _, preds = outputs.max(1)
        
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
        all_labels.extend(labels.numpy())
    
    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


def compute_metrics(y_true, y_pred, y_probs):
    """Compute comprehensive evaluation metrics."""
    metrics = {}
    
    # Accuracy
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    
    # Quadratic Weighted Kappa
    metrics['quadratic_kappa'] = cohen_kappa_score(y_true, y_pred, weights='quadratic')
    
    # Linear Weighted Kappa
    metrics['linear_kappa'] = cohen_kappa_score(y_true, y_pred, weights='linear')
    
    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=[0, 1, 2, 3, 4]
    )
    
    for i, class_name in enumerate(CLASS_NAMES):
        metrics[f'precision_class_{i}'] = precision[i]
        metrics[f'recall_class_{i}'] = recall[i]
        metrics[f'f1_class_{i}'] = f1[i]
        metrics[f'support_class_{i}'] = int(support[i])
    
    # Macro and weighted averages
    metrics['precision_macro'] = precision.mean()
    metrics['recall_macro'] = recall.mean()
    metrics['f1_macro'] = f1.mean()
    
    precision_w, recall_w, f1_w, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted'
    )
    metrics['precision_weighted'] = precision_w
    metrics['recall_weighted'] = recall_w
    metrics['f1_weighted'] = f1_w
    
    # Referable DR metrics (binary: 0-1 vs 2-4)
    y_true_binary = (y_true >= 2).astype(int)
    y_pred_binary = (y_pred >= 2).astype(int)
    y_probs_binary = y_probs[:, 2:].sum(axis=1)
    
    metrics['referable_accuracy'] = accuracy_score(y_true_binary, y_pred_binary)
    metrics['referable_precision'], metrics['referable_recall'], metrics['referable_f1'], _ = \
        precision_recall_fscore_support(y_true_binary, y_pred_binary, average='binary')
    
    try:
        metrics['referable_auc'] = roc_auc_score(y_true_binary, y_probs_binary)
    except:
        metrics['referable_auc'] = 0.0
    
    return metrics


def plot_confusion_matrix(y_true, y_pred, output_path):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASS_NAMES,
                yticklabels=CLASS_NAMES)
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Confusion matrix saved to: {output_path}")


def plot_normalized_confusion_matrix(y_true, y_pred, output_path):
    """Plot normalized confusion matrix (percentages)."""
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues',
                xticklabels=CLASS_NAMES,
                yticklabels=CLASS_NAMES)
    plt.title('Normalized Confusion Matrix', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Normalized confusion matrix saved to: {output_path}")


def plot_class_distribution(y_true, y_pred, output_path):
    """Plot true vs predicted class distribution."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # True distribution
    true_counts = pd.Series(y_true).value_counts().sort_index()
    ax1.bar(range(5), true_counts.values, color='skyblue', edgecolor='black')
    ax1.set_xticks(range(5))
    ax1.set_xticklabels(CLASS_NAMES, rotation=45, ha='right')
    ax1.set_title('True Label Distribution', fontweight='bold')
    ax1.set_ylabel('Count')
    ax1.grid(axis='y', alpha=0.3)
    
    # Predicted distribution
    pred_counts = pd.Series(y_pred).value_counts().sort_index()
    ax2.bar(range(5), pred_counts.values, color='lightcoral', edgecolor='black')
    ax2.set_xticks(range(5))
    ax2.set_xticklabels(CLASS_NAMES, rotation=45, ha='right')
    ax2.set_title('Predicted Label Distribution', fontweight='bold')
    ax2.set_ylabel('Count')
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Class distribution plot saved to: {output_path}")


def plot_per_class_metrics(metrics, output_path):
    """Plot per-class precision, recall, F1."""
    classes = range(5)
    precision = [metrics[f'precision_class_{i}'] for i in classes]
    recall = [metrics[f'recall_class_{i}'] for i in classes]
    f1 = [metrics[f'f1_class_{i}'] for i in classes]
    
    x = np.arange(5)
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width, precision, width, label='Precision', color='skyblue')
    ax.bar(x, recall, width, label='Recall', color='lightcoral')
    ax.bar(x + width, f1, width, label='F1-Score', color='lightgreen')
    
    ax.set_xlabel('DR Severity Grade')
    ax.set_ylabel('Score')
    ax.set_title('Per-Class Performance Metrics', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 1.1)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Per-class metrics plot saved to: {output_path}")


def save_metrics_report(metrics, output_path):
    """Save metrics to text file."""
    with open(output_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("VERA Model Evaluation Report\n")
        f.write("="*80 + "\n\n")
        
        f.write("Overall Metrics:\n")
        f.write("-"*80 + "\n")
        f.write(f"Accuracy:                {metrics['accuracy']:.4f}\n")
        f.write(f"Quadratic Weighted Kappa: {metrics['quadratic_kappa']:.4f}\n")
        f.write(f"Linear Weighted Kappa:    {metrics['linear_kappa']:.4f}\n\n")
        
        f.write("Macro-Averaged Metrics:\n")
        f.write("-"*80 + "\n")
        f.write(f"Precision (Macro):        {metrics['precision_macro']:.4f}\n")
        f.write(f"Recall (Macro):           {metrics['recall_macro']:.4f}\n")
        f.write(f"F1-Score (Macro):         {metrics['f1_macro']:.4f}\n\n")
        
        f.write("Weighted-Averaged Metrics:\n")
        f.write("-"*80 + "\n")
        f.write(f"Precision (Weighted):     {metrics['precision_weighted']:.4f}\n")
        f.write(f"Recall (Weighted):        {metrics['recall_weighted']:.4f}\n")
        f.write(f"F1-Score (Weighted):      {metrics['f1_weighted']:.4f}\n\n")
        
        f.write("Referable DR Metrics (Binary Classification: Grade 0-1 vs 2-4):\n")
        f.write("-"*80 + "\n")
        f.write(f"Accuracy:                 {metrics['referable_accuracy']:.4f}\n")
        f.write(f"Precision:                {metrics['referable_precision']:.4f}\n")
        f.write(f"Recall (Sensitivity):     {metrics['referable_recall']:.4f}\n")
        f.write(f"F1-Score:                 {metrics['referable_f1']:.4f}\n")
        f.write(f"AUC-ROC:                  {metrics['referable_auc']:.4f}\n\n")
        
        f.write("Per-Class Metrics:\n")
        f.write("-"*80 + "\n")
        f.write(f"{'Class':<25} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}\n")
        f.write("-"*80 + "\n")
        
        for i, class_name in enumerate(CLASS_NAMES):
            precision = metrics[f'precision_class_{i}']
            recall = metrics[f'recall_class_{i}']
            f1 = metrics[f'f1_class_{i}']
            support = metrics[f'support_class_{i}']
            f.write(f"{class_name:<25} {precision:<12.4f} {recall:<12.4f} {f1:<12.4f} {support:<10}\n")
        
        f.write("\n" + "="*80 + "\n")
    
    print(f"✓ Metrics report saved to: {output_path}")


def main():
    args = parse_args()
    
    # Device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load checkpoint
    print(f"\nLoading checkpoint: {args.checkpoint}")
    checkpoint = load_checkpoint(args.checkpoint)
    
    # Create model
    model = VesselAwareClassifier(
        backbone=checkpoint.get('backbone', 'resnet50'),
        fusion_type=checkpoint.get('fusion_type', 'early'),
        num_classes=5,
        pretrained=False
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print("✓ Model loaded successfully")
    
    # Load test dataset
    print(f"\nLoading {args.dataset} test set...")
    data_path = Path(args.data_dir)
    
    if args.dataset == 'aptos':
        csv_file = data_path / 'aptos2019' / 'train.csv'
        image_dir = data_path / 'aptos2019' / 'train_images'
        label_col = 'diagnosis'
        id_col = 'id_code'
    # Add other datasets as needed
    
    df = pd.read_csv(csv_file)
    df['image_path'] = df[id_col].apply(lambda x: str(image_dir / f"{x}.png"))
    df['label'] = df[label_col]
    
    # Use last 10% as test set (or load from separate test CSV)
    test_df = df.iloc[int(0.9 * len(df)):]
    
    print(f"Test set size: {len(test_df)}")
    
    # Initialize vessel segmenter
    vessel_segmenter = VesselSegmenter(cache_dir=args.vessel_cache_dir)
    
    # Create test dataset and loader
    test_dataset = FundusDataset(
        test_df,
        vessel_segmenter=vessel_segmenter,
        vessel_cache_dir=args.vessel_cache_dir,
        transform=get_val_transform(),
        target_size=(224, 224)
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Run evaluation
    print("\nRunning evaluation...")
    y_true, y_pred, y_probs = evaluate_model(model, test_loader, device)
    
    # Compute metrics
    print("\nComputing metrics...")
    metrics = compute_metrics(y_true, y_pred, y_probs)
    
    # Print key metrics
    print("\n" + "="*80)
    print("Evaluation Results")
    print("="*80)
    print(f"Accuracy:              {metrics['accuracy']:.4f}")
    print(f"Quadratic Kappa:       {metrics['quadratic_kappa']:.4f}")
    print(f"Referable DR AUC:      {metrics['referable_auc']:.4f}")
    print("="*80)
    
    # Save results
    print("\nGenerating visualizations...")
    
    # Save metrics
    metrics_json = output_dir / 'metrics.json'
    with open(metrics_json, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"✓ Metrics JSON saved to: {metrics_json}")
    
    # Save text report
    save_metrics_report(metrics, output_dir / 'evaluation_report.txt')
    
    # Generate plots
    plot_confusion_matrix(y_true, y_pred, output_dir / 'confusion_matrix.png')
    plot_normalized_confusion_matrix(y_true, y_pred, output_dir / 'confusion_matrix_normalized.png')
    plot_class_distribution(y_true, y_pred, output_dir / 'class_distribution.png')
    plot_per_class_metrics(metrics, output_dir / 'per_class_metrics.png')
    
    print(f"\n✓ Evaluation complete! Results saved to: {output_dir}")


if __name__ == '__main__':
    main()

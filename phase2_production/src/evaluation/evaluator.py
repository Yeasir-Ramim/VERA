"""
VERA Phase 2 - Model Evaluator

Comprehensive model evaluation with metrics, confusion matrices, and visualizations.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
from typing import Dict, Optional, List
import logging

from .metrics import (
    compute_all_metrics,
    compute_confusion_matrix,
    get_classification_report,
    compute_grade_distribution
)

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Comprehensive model evaluator for DR classification.
    """
    
    def __init__(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        device: str = 'cuda',
        num_classes: int = 5,
        class_names: List[str] = None
    ):
        """
        Initialize evaluator.
        
        Args:
            model: PyTorch model
            dataloader: Evaluation dataloader
            device: Device for evaluation
            num_classes: Number of classes
            class_names: List of class names
        """
        self.model = model.to(device)
        self.dataloader = dataloader
        self.device = device
        self.num_classes = num_classes
        
        if class_names is None:
            self.class_names = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative']
        else:
            self.class_names = class_names
        
        self.model.eval()
    
    @torch.no_grad()
    def evaluate(self) -> Dict:
        """
        Run comprehensive evaluation.
        
        Returns:
            Dictionary with all evaluation results
        """
        logger.info("Running model evaluation...")
        
        all_predictions = []
        all_targets = []
        all_probabilities = []
        
        for batch in tqdm(self.dataloader, desc='Evaluating'):
            images = batch['image'].to(self.device)
            vessels = batch['vessel'].to(self.device)
            labels = batch['label'].to(self.device)
            
            # Forward pass
            outputs = self.model(images, vessels)
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            
            # Collect
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
            all_probabilities.append(probabilities.cpu().numpy())
        
        # Convert to arrays
        predictions = np.array(all_predictions)
        targets = np.array(all_targets)
        probabilities = np.vstack(all_probabilities)
        
        # Compute metrics
        metrics = compute_all_metrics(predictions, targets, probabilities, self.num_classes)
        
        # Confusion matrix
        cm = compute_confusion_matrix(predictions, targets, self.num_classes)
        cm_normalized = compute_confusion_matrix(predictions, targets, self.num_classes, normalize='true')
        
        # Classification report
        report = get_classification_report(predictions, targets, self.class_names)
        
        # Grade distributions
        pred_distribution = compute_grade_distribution(predictions, self.num_classes)
        target_distribution = compute_grade_distribution(targets, self.num_classes)
        
        results = {
            'metrics': metrics,
            'confusion_matrix': cm,
            'confusion_matrix_normalized': cm_normalized,
            'classification_report': report,
            'predictions': predictions,
            'targets': targets,
            'probabilities': probabilities,
            'pred_distribution': pred_distribution,
            'target_distribution': target_distribution
        }
        
        return results
    
    def print_results(self, results: Dict):
        """Print evaluation results."""
        logger.info("\n" + "="*80)
        logger.info("EVALUATION RESULTS")
        logger.info("="*80)
        
        metrics = results['metrics']
        
        logger.info(f"\nOverall Metrics:")
        logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"  Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
        logger.info(f"  Quadratic Weighted Kappa: {metrics['qwk']:.4f}")
        logger.info(f"  F1 Score (macro): {metrics['f1_macro']:.4f}")
        
        logger.info(f"\nReferable DR Metrics:")
        logger.info(f"  Accuracy: {metrics['referable_accuracy']:.4f}")
        logger.info(f"  Precision: {metrics['referable_precision']:.4f}")
        logger.info(f"  Recall: {metrics['referable_recall']:.4f}")
        logger.info(f"  F1 Score: {metrics['referable_f1']:.4f}")
        if 'referable_auc' in metrics:
            logger.info(f"  AUC: {metrics['referable_auc']:.4f}")
        
        logger.info(f"\nPer-Class Metrics:")
        for i, name in enumerate(self.class_names):
            logger.info(f"  {name} (Grade {i}):")
            logger.info(f"    Precision: {metrics[f'precision_class{i}']:.4f}")
            logger.info(f"    Recall: {metrics[f'recall_class{i}']:.4f}")
            logger.info(f"    F1: {metrics[f'f1_class{i}']:.4f}")
        
        logger.info(f"\nClassification Report:")
        logger.info(results['classification_report'])
    
    def plot_confusion_matrix(
        self,
        results: Dict,
        save_path: Optional[str] = None,
        normalize: bool = True
    ):
        """
        Plot confusion matrix.
        
        Args:
            results: Evaluation results
            save_path: Path to save figure
            normalize: Use normalized confusion matrix
        """
        cm = results['confusion_matrix_normalized'] if normalize else results['confusion_matrix']
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        sns.heatmap(
            cm,
            annot=True,
            fmt='.2f' if normalize else 'd',
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            cbar_kws={'label': 'Proportion' if normalize else 'Count'},
            ax=ax
        )
        
        ax.set_xlabel('Predicted Grade', fontsize=12, fontweight='bold')
        ax.set_ylabel('True Grade', fontsize=12, fontweight='bold')
        title = 'Normalized Confusion Matrix' if normalize else 'Confusion Matrix'
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved confusion matrix to {save_path}")
        
        plt.close()
    
    def plot_class_distribution(
        self,
        results: Dict,
        save_path: Optional[str] = None
    ):
        """Plot class distribution comparison."""
        pred_dist = results['pred_distribution']
        target_dist = results['target_distribution']
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x = np.arange(self.num_classes)
        width = 0.35
        
        ax.bar(x - width/2, target_dist['counts'], width,
              label='Ground Truth', color='#3498db', alpha=0.8)
        ax.bar(x + width/2, pred_dist['counts'], width,
              label='Predictions', color='#e74c3c', alpha=0.8)
        
        ax.set_xlabel('DR Severity Grade', fontsize=12, fontweight='bold')
        ax.set_ylabel('Number of Samples', fontsize=12, fontweight='bold')
        ax.set_title('Class Distribution: Ground Truth vs Predictions',
                    fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(self.class_names)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved distribution plot to {save_path}")
        
        plt.close()
    
    def save_results(self, results: Dict, output_dir: str):
        """Save all evaluation results."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save metrics as JSON
        import json
        metrics_path = output_dir / 'metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(results['metrics'], f, indent=2)
        
        # Save classification report
        report_path = output_dir / 'classification_report.txt'
        with open(report_path, 'w') as f:
            f.write(results['classification_report'])
        
        # Save confusion matrices
        np.save(output_dir / 'confusion_matrix.npy', results['confusion_matrix'])
        np.save(output_dir / 'confusion_matrix_normalized.npy',
               results['confusion_matrix_normalized'])
        
        # Save predictions
        np.save(output_dir / 'predictions.npy', results['predictions'])
        np.save(output_dir / 'targets.npy', results['targets'])
        np.save(output_dir / 'probabilities.npy', results['probabilities'])
        
        # Plot confusion matrix
        self.plot_confusion_matrix(results, str(output_dir / 'confusion_matrix.png'))
        self.plot_confusion_matrix(results, str(output_dir / 'confusion_matrix_normalized.png'),
                                  normalize=True)
        
        # Plot distributions
        self.plot_class_distribution(results, str(output_dir / 'class_distribution.png'))
        
        logger.info(f"Saved all evaluation results to {output_dir}")

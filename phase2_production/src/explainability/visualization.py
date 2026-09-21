"""
VERA Phase 2 - Explainability Visualizations

Functions for generating comprehensive diagnostic reports with explainability.
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Optional, List
import logging

from .gradcam import overlay_heatmap_on_image
from .overlap_score import compute_overlap_metrics, visualize_overlap

logger = logging.getLogger(__name__)


def create_4panel_report(
    original_image: np.ndarray,
    preprocessed_image: np.ndarray,
    vessel_map: np.ndarray,
    heatmap: np.ndarray,
    prediction: int,
    confidence: float,
    overlap_metrics: Dict[str, float],
    save_path: Optional[str] = None,
    class_names: List[str] = None
) -> np.ndarray:
    """
    Create 4-panel diagnostic report.
    
    Layout:
    ┌─────────────┬─────────────┐
    │   Original  │ Preprocessed│
    ├─────────────┼─────────────┤
    │ Vessel Map  │  Grad-CAM++ │
    └─────────────┴─────────────┘
    
    Args:
        original_image: Original fundus image (H, W, 3)
        preprocessed_image: Preprocessed image (H, W, 3)
        vessel_map: Vessel segmentation (H, W)
        heatmap: Grad-CAM heatmap (H, W)
        prediction: Predicted class
        confidence: Prediction confidence
        overlap_metrics: Overlap score metrics
        save_path: Optional path to save figure
        class_names: List of class names
        
    Returns:
        Figure as numpy array
    """
    if class_names is None:
        class_names = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative']
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    # Original image
    axes[0, 0].imshow(original_image)
    axes[0, 0].set_title('Original Fundus Image', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')
    
    # Preprocessed image
    axes[0, 1].imshow(preprocessed_image)
    axes[0, 1].set_title('Preprocessed (CLAHE + Ben Graham)', fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')
    
    # Vessel map
    axes[1, 0].imshow(vessel_map, cmap='gray')
    axes[1, 0].set_title('Vessel Segmentation', fontsize=14, fontweight='bold')
    axes[1, 0].axis('off')
    
    # Grad-CAM overlay
    heatmap_overlay = overlay_heatmap_on_image(original_image, heatmap, alpha=0.4)
    axes[1, 1].imshow(heatmap_overlay)
    axes[1, 1].set_title('Grad-CAM++ Attention', fontsize=14, fontweight='bold')
    axes[1, 1].axis('off')
    
    # Add prediction and metrics text
    pred_text = (
        f"Prediction: {class_names[prediction]} (Grade {prediction})\n"
        f"Confidence: {confidence:.1%}\n"
        f"Vessel Overlap: {overlap_metrics.get('overlap_score', 0):.3f}"
    )
    
    fig.text(0.5, 0.02, pred_text, ha='center', fontsize=12,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved 4-panel report to {save_path}")
    
    # Convert to array
    fig.canvas.draw()
    img_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
    plt.close(fig)
    
    return img_array


def create_comprehensive_report(
    original_image: np.ndarray,
    preprocessed_image: np.ndarray,
    vessel_map: np.ndarray,
    heatmap: np.ndarray,
    prediction: int,
    probabilities: np.ndarray,
    overlap_metrics: Dict[str, float],
    save_path: Optional[str] = None,
    class_names: List[str] = None
) -> np.ndarray:
    """
    Create comprehensive diagnostic report with all visualizations.
    
    Args:
        original_image: Original image
        preprocessed_image: Preprocessed image
        vessel_map: Vessel map
        heatmap: Grad-CAM heatmap
        prediction: Predicted class
        probabilities: Class probabilities (5,)
        overlap_metrics: Overlap metrics
        save_path: Path to save
        class_names: Class names
        
    Returns:
        Report image
    """
    if class_names is None:
        class_names = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative']
    
    # Create figure with more panels
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
    
    # Row 1: Images
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(original_image)
    ax1.set_title('Original', fontweight='bold')
    ax1.axis('off')
    
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(preprocessed_image)
    ax2.set_title('Preprocessed', fontweight='bold')
    ax2.axis('off')
    
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(vessel_map, cmap='gray')
    ax3.set_title('Vessel Segmentation', fontweight='bold')
    ax3.axis('off')
    
    ax4 = fig.add_subplot(gs[0, 3])
    heatmap_overlay = overlay_heatmap_on_image(original_image, heatmap, alpha=0.4)
    ax4.imshow(heatmap_overlay)
    ax4.set_title('Grad-CAM++ Heatmap', fontweight='bold')
    ax4.axis('off')
    
    # Row 2: Detailed visualizations
    ax5 = fig.add_subplot(gs[1, 0:2])
    # Heatmap only
    ax5.imshow(heatmap, cmap='jet')
    ax5.set_title('Attention Heatmap (Raw)', fontweight='bold')
    ax5.axis('off')
    ax5.set_aspect('equal')
    
    ax6 = fig.add_subplot(gs[1, 2:4])
    # Overlap visualization
    overlap_vis = visualize_overlap(original_image, heatmap, vessel_map)
    ax6.imshow(overlap_vis)
    ax6.set_title('Attention-Vessel Overlap', fontweight='bold')
    ax6.axis('off')
    
    # Row 3: Metrics and probabilities
    ax7 = fig.add_subplot(gs[2, 0:2])
    # Probability bar chart
    colors = ['#2ecc71', '#3498db', '#f39c12', '#e74c3c', '#8e44ad']
    bars = ax7.bar(class_names, probabilities, color=colors, alpha=0.7, edgecolor='black')
    bars[prediction].set_alpha(1.0)
    bars[prediction].set_edgecolor('red')
    bars[prediction].set_linewidth(3)
    ax7.set_ylabel('Probability', fontweight='bold')
    ax7.set_title('Class Probabilities', fontweight='bold')
    ax7.set_ylim([0, 1])
    ax7.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, prob in zip(bars, probabilities):
        height = bar.get_height()
        ax7.text(bar.get_x() + bar.get_width()/2., height,
                f'{prob:.1%}', ha='center', va='bottom', fontweight='bold')
    
    ax8 = fig.add_subplot(gs[2, 2:4])
    # Overlap metrics text
    ax8.axis('off')
    
    metrics_text = "Vessel-Attention Overlap Metrics:\n\n"
    metrics_text += f"Overall Overlap Score: {overlap_metrics.get('overlap_score', 0):.3f}\n"
    metrics_text += f"Spatial Correlation: {overlap_metrics.get('spatial_correlation', 0):.3f}\n\n"
    
    metrics_text += "Coverage (attention on vessels):\n"
    for key, value in overlap_metrics.items():
        if 'coverage' in key:
            threshold = key.split('_t')[-1]
            metrics_text += f"  Threshold {threshold}: {value:.3f}\n"
    
    metrics_text += "\nPrecision (vessels in high attention):\n"
    for key, value in overlap_metrics.items():
        if 'precision' in key:
            threshold = key.split('_t')[-1]
            metrics_text += f"  Threshold {threshold}: {value:.3f}\n"
    
    ax8.text(0.1, 0.5, metrics_text, fontsize=11, verticalalignment='center',
            fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    
    # Title
    fig.suptitle(
        f'VERA Diagnostic Report - Prediction: {class_names[prediction]} (Grade {prediction})',
        fontsize=16, fontweight='bold', y=0.98
    )
    
    # Save
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved comprehensive report to {save_path}")
    
    # Convert to array
    fig.canvas.draw()
    img_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
    plt.close(fig)
    
    return img_array


def generate_batch_reports(
    images: List[np.ndarray],
    vessel_maps: List[np.ndarray],
    heatmaps: List[np.ndarray],
    predictions: List[int],
    probabilities: List[np.ndarray],
    overlap_metrics_list: List[Dict],
    output_dir: str,
    class_names: List[str] = None
):
    """
    Generate reports for multiple samples.
    
    Args:
        images: List of original images
        vessel_maps: List of vessel maps
        heatmaps: List of heatmaps
        predictions: List of predictions
        probabilities: List of probability arrays
        overlap_metrics_list: List of overlap metrics dicts
        output_dir: Output directory
        class_names: Class names
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for idx, (img, vessel, heatmap, pred, probs, metrics) in enumerate(
        zip(images, vessel_maps, heatmaps, predictions, probabilities, overlap_metrics_list)
    ):
        save_path = output_dir / f'report_sample_{idx+1}_grade{pred}.png'
        
        create_comprehensive_report(
            original_image=img,
            preprocessed_image=img,  # Could pass actual preprocessed
            vessel_map=vessel,
            heatmap=heatmap,
            prediction=pred,
            probabilities=probs,
            overlap_metrics=metrics,
            save_path=str(save_path),
            class_names=class_names
        )
    
    logger.info(f"Generated {len(images)} reports in {output_dir}")

"""
VERA Phase 2 - Vessel-Attention Overlap Score

Quantitative metric for measuring model attention alignment with vessel structures.
"""

import torch
import torch.nn as nn
import numpy as np
import cv2
from typing import Tuple, Dict
import logging

logger = logging.getLogger(__name__)


def compute_vessel_attention_overlap(
    heatmap: np.ndarray,
    vessel_map: np.ndarray,
    threshold: float = 0.3,
    normalize: bool = True
) -> float:
    """
    Compute overlap between attention heatmap and vessel map.
    
    This metric quantifies how much of the model's attention is focused
    on vessel structures vs. non-vessel background. Higher scores indicate
    that the model is attending to clinically relevant vascular features.
    
    Args:
        heatmap: Grad-CAM heatmap (H, W) in range [0, 1]
        vessel_map: Vessel probability map (H, W) in range [0, 1]
        threshold: Threshold for binarizing attention (focus on high attention regions)
        normalize: Normalize heatmap before computing overlap
        
    Returns:
        Overlap score in range [0, 1]
    """
    # Ensure same size
    if heatmap.shape != vessel_map.shape:
        vessel_map = cv2.resize(vessel_map, (heatmap.shape[1], heatmap.shape[0]))
    
    # Normalize heatmap if requested
    if normalize:
        heatmap_norm = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    else:
        heatmap_norm = heatmap
    
    # Threshold attention map to focus on high-attention regions
    attention_mask = heatmap_norm > threshold
    
    # Compute overlap
    # Numerator: Sum of attention weights on vessel regions
    overlap_on_vessels = (heatmap_norm * vessel_map).sum()
    
    # Denominator: Total attention weight
    total_attention = heatmap_norm.sum()
    
    # Overlap score
    overlap_score = overlap_on_vessels / (total_attention + 1e-8)
    
    return float(overlap_score)


def compute_vessel_coverage(
    heatmap: np.ndarray,
    vessel_map: np.ndarray,
    threshold: float = 0.3
) -> float:
    """
    Compute what fraction of vessels are covered by high-attention regions.
    
    Args:
        heatmap: Grad-CAM heatmap (H, W)
        vessel_map: Vessel probability map (H, W)
        threshold: Threshold for high attention
        
    Returns:
        Coverage score [0, 1]
    """
    if heatmap.shape != vessel_map.shape:
        vessel_map = cv2.resize(vessel_map, (heatmap.shape[1], heatmap.shape[0]))
    
    # Normalize heatmap
    heatmap_norm = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    
    # High attention regions
    high_attention = heatmap_norm > threshold
    
    # Binary vessel map
    vessel_binary = vessel_map > 0.5
    
    # Coverage: fraction of vessel pixels with high attention
    vessel_pixels = vessel_binary.sum()
    covered_vessel_pixels = (high_attention & vessel_binary).sum()
    
    coverage = covered_vessel_pixels / (vessel_pixels + 1e-8)
    
    return float(coverage)


def compute_precision_on_vessels(
    heatmap: np.ndarray,
    vessel_map: np.ndarray,
    threshold: float = 0.3
) -> float:
    """
    Compute precision: fraction of high-attention regions that are on vessels.
    
    Args:
        heatmap: Grad-CAM heatmap (H, W)
        vessel_map: Vessel probability map (H, W)
        threshold: Threshold for high attention
        
    Returns:
        Precision score [0, 1]
    """
    if heatmap.shape != vessel_map.shape:
        vessel_map = cv2.resize(vessel_map, (heatmap.shape[1], heatmap.shape[0]))
    
    # Normalize heatmap
    heatmap_norm = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    
    # High attention regions
    high_attention = heatmap_norm > threshold
    
    # Binary vessel map
    vessel_binary = vessel_map > 0.5
    
    # Precision
    attention_pixels = high_attention.sum()
    correct_attention_pixels = (high_attention & vessel_binary).sum()
    
    precision = correct_attention_pixels / (attention_pixels + 1e-8)
    
    return float(precision)


def compute_overlap_metrics(
    heatmap: np.ndarray,
    vessel_map: np.ndarray,
    thresholds: list = None
) -> Dict[str, float]:
    """
    Compute comprehensive overlap metrics.
    
    Args:
        heatmap: Grad-CAM heatmap (H, W)
        vessel_map: Vessel probability map (H, W)
        thresholds: List of thresholds to try
        
    Returns:
        Dictionary of metrics
    """
    if thresholds is None:
        thresholds = [0.3, 0.5, 0.7]
    
    metrics = {}
    
    # Weighted overlap (main metric)
    metrics['overlap_score'] = compute_vessel_attention_overlap(heatmap, vessel_map)
    
    # Per-threshold metrics
    for threshold in thresholds:
        metrics[f'coverage_t{threshold}'] = compute_vessel_coverage(
            heatmap, vessel_map, threshold
        )
        metrics[f'precision_t{threshold}'] = compute_precision_on_vessels(
            heatmap, vessel_map, threshold
        )
    
    # Spatial correlation
    if heatmap.shape == vessel_map.shape:
        # Flatten and compute correlation
        heatmap_flat = heatmap.flatten()
        vessel_flat = vessel_map.flatten()
        
        correlation = np.corrcoef(heatmap_flat, vessel_flat)[0, 1]
        metrics['spatial_correlation'] = float(correlation)
    
    return metrics


def visualize_overlap(
    image: np.ndarray,
    heatmap: np.ndarray,
    vessel_map: np.ndarray,
    threshold: float = 0.3
) -> np.ndarray:
    """
    Create visualization showing overlap between attention and vessels.
    
    Args:
        image: Original image (H, W, 3)
        heatmap: Grad-CAM heatmap (H, W)
        vessel_map: Vessel probability map (H, W)
        threshold: Threshold for high attention
        
    Returns:
        Visualization image (H, W, 3)
    """
    # Resize to match image
    h, w = image.shape[:2]
    if heatmap.shape != (h, w):
        heatmap = cv2.resize(heatmap, (w, h))
    if vessel_map.shape != (h, w):
        vessel_map = cv2.resize(vessel_map, (w, h))
    
    # Normalize
    heatmap_norm = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    
    # Create colored overlay
    # Green for vessels, Red for high attention, Yellow for overlap
    visualization = image.copy()
    
    # High attention regions (red)
    high_attention = heatmap_norm > threshold
    visualization[high_attention] = visualization[high_attention] * 0.5 + np.array([255, 0, 0]) * 0.5
    
    # Vessel regions (green)
    vessels = vessel_map > 0.5
    visualization[vessels] = visualization[vessels] * 0.5 + np.array([0, 255, 0]) * 0.5
    
    # Overlap regions (yellow/orange)
    overlap = high_attention & vessels
    visualization[overlap] = visualization[overlap] * 0.3 + np.array([255, 200, 0]) * 0.7
    
    return visualization.astype(np.uint8)


class OverlapScoreEvaluator:
    """
    Evaluator for computing overlap scores across a dataset.
    """
    
    def __init__(
        self,
        model: nn.Module,
        gradcam,
        device: str = 'cuda'
    ):
        """
        Initialize evaluator.
        
        Args:
            model: PyTorch model
            gradcam: GradCAM instance
            device: Device for computation
        """
        self.model = model
        self.gradcam = gradcam
        self.device = device
        
        self.model.to(device)
        self.model.eval()
    
    @torch.no_grad()
    def evaluate_batch(
        self,
        images: torch.Tensor,
        vessels: torch.Tensor,
        labels: torch.Tensor
    ) -> Tuple[float, Dict]:
        """
        Evaluate overlap scores for a batch.
        
        Args:
            images: RGB images (B, 3, H, W)
            vessels: Vessel maps (B, 1, H, W)
            labels: Ground truth labels (B,)
            
        Returns:
            Average overlap score and detailed metrics
        """
        batch_size = images.shape[0]
        overlap_scores = []
        
        for i in range(batch_size):
            # Create 4-channel input
            input_tensor = torch.cat([images[i:i+1], vessels[i:i+1]], dim=1)
            
            # Generate heatmap
            heatmap = self.gradcam(input_tensor, target_class=labels[i].item())
            
            # Get vessel map
            vessel_map = vessels[i, 0].cpu().numpy()
            
            # Compute overlap
            overlap = compute_vessel_attention_overlap(heatmap, vessel_map)
            overlap_scores.append(overlap)
        
        avg_overlap = np.mean(overlap_scores)
        
        metrics = {
            'mean_overlap': avg_overlap,
            'std_overlap': np.std(overlap_scores),
            'min_overlap': np.min(overlap_scores),
            'max_overlap': np.max(overlap_scores)
        }
        
        return avg_overlap, metrics

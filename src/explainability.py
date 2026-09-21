"""
Explainability Module: Gradient-weighted Class Activation Mapping (Grad-CAM & Grad-CAM++).
Generates visual localization heatmaps and computes quantitative Vessel-Attention Overlap Scores.
"""

from typing import Dict, Optional, Tuple, Union
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseCAM:
    """Base class for Class Activation Mapping implementations."""
    def __init__(self, model: nn.Module, target_layer: Optional[nn.Module] = None):
        self.model = model
        self.target_layer = target_layer or self._get_default_target_layer()
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self._register_hooks()

    def _get_default_target_layer(self) -> nn.Module:
        if hasattr(self.model, "get_target_layer_for_gradcam"):
            return self.model.get_target_layer_for_gradcam()
        last_conv = None
        for module in self.model.modules():
            if isinstance(module, nn.Conv2d):
                last_conv = module
        if last_conv is None:
            raise ValueError("No Conv2d layer found in model for CAM.")
        return last_conv

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        if hasattr(self.target_layer, "register_full_backward_hook"):
            self.target_layer.register_full_backward_hook(backward_hook)
        else:
            self.target_layer.register_backward_hook(backward_hook)


class GradCAM(BaseCAM):
    """
    Standard Grad-CAM (Selvaraju et al., 2017).
    Weights feature maps by global-average-pooled first-order gradients.
    """
    def generate_cam(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None
    ) -> Tuple[np.ndarray, int, float, np.ndarray]:
        self.model.eval()
        self.model.zero_grad()
        
        if input_tensor.ndim == 3:
            input_tensor = input_tensor.unsqueeze(0)
            
        device = next(self.model.parameters()).device
        input_tensor = input_tensor.to(device)
        input_tensor.requires_grad_(True)
        
        logits = self.model(input_tensor)
        probs = F.softmax(logits, dim=1).squeeze(0).detach().cpu().numpy()
        pred_class = int(np.argmax(probs))
        confidence = float(probs[pred_class])
        
        if target_class is None:
            target_class = pred_class
            
        score = logits[0, target_class]
        score.backward(retain_graph=True)
        
        grads = self.gradients[0]   # (C, H, W)
        acts = self.activations[0]  # (C, H, W)
        
        weights = torch.mean(grads, dim=(1, 2))  # (C,)
        
        cam = torch.zeros(acts.shape[1:], dtype=torch.float32, device=acts.device)
        for i, w in enumerate(weights):
            cam += w * acts[i]
            
        cam = F.relu(cam)
        cam_np = cam.detach().cpu().numpy()
        
        c_min, c_max = cam_np.min(), cam_np.max()
        if c_max > c_min:
            cam_np = (cam_np - c_min) / (c_max - c_min)
        else:
            cam_np = np.zeros_like(cam_np)
            
        target_h, target_w = input_tensor.shape[2], input_tensor.shape[3]
        cam_resized = cv2.resize(cam_np, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        
        return cam_resized, pred_class, confidence, probs


class GradCAMPlusPlus(BaseCAM):
    """
    Grad-CAM++ (Chattopadhyay et al., 2018).
    Utilizes higher-order partial derivatives to provide superior localization
    for multiple distributed lesions (e.g. multiple microaneurysms and hemorrhages).
    """
    def generate_cam(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None
    ) -> Tuple[np.ndarray, int, float, np.ndarray]:
        self.model.eval()
        self.model.zero_grad()
        
        if input_tensor.ndim == 3:
            input_tensor = input_tensor.unsqueeze(0)
            
        device = next(self.model.parameters()).device
        input_tensor = input_tensor.to(device)
        input_tensor.requires_grad_(True)
        
        logits = self.model(input_tensor)
        probs = F.softmax(logits, dim=1).squeeze(0).detach().cpu().numpy()
        pred_class = int(np.argmax(probs))
        confidence = float(probs[pred_class])
        
        if target_class is None:
            target_class = pred_class
            
        # Target score exponentiation for stability in higher-order formulation
        score = logits[0, target_class]
        score.backward(retain_graph=True)
        
        grads = self.gradients[0]   # (C, H, W)
        acts = self.activations[0]  # (C, H, W)
        
        # Second and third-order gradient terms
        grads_power_2 = grads ** 2
        grads_power_3 = grads ** 3
        
        # Denominator: 2 * grads^2 + sum(acts * grads^3)
        sum_acts = torch.sum(acts, dim=(1, 2), keepdim=True)
        eps = 1e-7
        denom = 2.0 * grads_power_2 + sum_acts * grads_power_3 + eps
        
        # Pixel-wise alpha weights
        aij = grads_power_2 / denom
        
        # Positive gradients
        relu_grads = F.relu(grads)
        weights = torch.sum(aij * relu_grads, dim=(1, 2))  # (C,)
        
        cam = torch.zeros(acts.shape[1:], dtype=torch.float32, device=acts.device)
        for i, w in enumerate(weights):
            cam += w * acts[i]
            
        cam = F.relu(cam)
        cam_np = cam.detach().cpu().numpy()
        
        c_min, c_max = cam_np.min(), cam_np.max()
        if c_max > c_min:
            cam_np = (cam_np - c_min) / (c_max - c_min)
        else:
            cam_np = np.zeros_like(cam_np)
            
        target_h, target_w = input_tensor.shape[2], input_tensor.shape[3]
        cam_resized = cv2.resize(cam_np, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        
        return cam_resized, pred_class, confidence, probs


def compute_vessel_attention_overlap(
    cam_map: np.ndarray,
    vessel_map: np.ndarray,
    threshold: float = 0.2
) -> Dict[str, float]:
    """
    Quantifies the proportion of model attention focused on retinal vasculature.
    Used in Ablation Study 3 to formally verify reduction in shortcut learning.
    
    Args:
        cam_map: Normalized 2D Grad-CAM array in [0.0, 1.0].
        vessel_map: 2D Retinal vessel probability map in [0.0, 1.0].
        threshold: Binarization cutoff for high-attention regions.
        
    Returns:
        Dictionary with:
          - 'overlap_score': Continuous weighted intersection / total attention.
          - 'high_attention_vascular_ratio': Fraction of attention-activated pixels that coincide with vessels.
    """
    if cam_map.shape != vessel_map.shape:
        vessel_map = cv2.resize(vessel_map, (cam_map.shape[1], cam_map.shape[0]))
        
    vessel_map_clipped = np.clip(vessel_map, 0.0, 1.0)
    cam_map_clipped = np.clip(cam_map, 0.0, 1.0)
    
    # 1. Continuous Overlap Score: sum(CAM * Vessel) / sum(CAM)
    total_cam = float(np.sum(cam_map_clipped))
    if total_cam > 1e-6:
        overlap_score = float(np.sum(cam_map_clipped * vessel_map_clipped) / total_cam)
    else:
        overlap_score = 0.0
        
    # 2. Binarized overlap above threshold
    attn_mask = (cam_map_clipped >= threshold)
    total_active_pixels = int(np.sum(attn_mask))
    if total_active_pixels > 0:
        high_attn_ratio = float(np.sum(vessel_map_clipped[attn_mask] > 0.3) / total_active_pixels)
    else:
        high_attn_ratio = 0.0
        
    return {
        "overlap_score": float(np.clip(overlap_score, 0.0, 1.0)),
        "high_attention_vascular_ratio": float(np.clip(high_attn_ratio, 0.0, 1.0))
    }


def overlay_cam_on_image(
    img_rgb: np.ndarray,
    cam_map: np.ndarray,
    alpha: float = 0.5,
    colormap: int = cv2.COLORMAP_JET
) -> np.ndarray:
    """
    Overlays Grad-CAM heatmap onto RGB image.
    """
    h, w = img_rgb.shape[:2]
    if cam_map.shape[:2] != (h, w):
        cam_map = cv2.resize(cam_map, (w, h))
        
    heatmap_gray = np.uint8(255 * np.clip(cam_map, 0.0, 1.0))
    heatmap_bgr = cv2.applyColorMap(heatmap_gray, colormap)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
    
    blended = cv2.addWeighted(img_rgb, 1.0 - alpha, heatmap_rgb, alpha, 0)
    return blended

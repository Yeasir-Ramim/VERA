"""
VERA Phase 2 - Grad-CAM and Grad-CAM++ Implementation

Enhanced gradient-based visualization techniques for model explainability.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Optional, List, Tuple
import logging

logger = logging.getLogger(__name__)


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM).
    
    Reference:
    Selvaraju et al. "Grad-CAM: Visual Explanations from Deep Networks 
    via Gradient-based Localization" (2017)
    """
    
    def __init__(
        self,
        model: nn.Module,
        target_layer: Optional[nn.Module] = None,
        use_cuda: bool = True
    ):
        """
        Initialize Grad-CAM.
        
        Args:
            model: PyTorch model
            target_layer: Target layer for activation extraction
            use_cuda: Use CUDA if available
        """
        self.model = model
        self.target_layer = target_layer
        self.device = 'cuda' if use_cuda and torch.cuda.is_available() else 'cpu'
        
        self.gradients = None
        self.activations = None
        
        # Register hooks
        if target_layer is not None:
            self._register_hooks(target_layer)
        else:
            # Auto-detect last convolutional layer
            self.target_layer = self._find_target_layer()
            if self.target_layer is not None:
                self._register_hooks(self.target_layer)
        
        self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"Initialized Grad-CAM with target layer: {self.target_layer}")
    
    def _find_target_layer(self) -> Optional[nn.Module]:
        """Find the last convolutional layer."""
        target_layer = None
        
        for module in self.model.modules():
            if isinstance(module, nn.Conv2d):
                target_layer = module
        
        return target_layer
    
    def _register_hooks(self, layer: nn.Module):
        """Register forward and backward hooks."""
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        layer.register_forward_hook(forward_hook)
        layer.register_full_backward_hook(backward_hook)
    
    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate Grad-CAM heatmap.
        
        Args:
            input_tensor: Input image tensor (1, C, H, W)
            target_class: Target class index (None for predicted class)
            
        Returns:
            Grad-CAM heatmap (H, W) normalized to [0, 1]
        """
        # Forward pass
        input_tensor = input_tensor.to(self.device)
        self.model.zero_grad()
        
        output = self.model(input_tensor[:, :3], input_tensor[:, 3:4])
        
        # Get target class
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Backward pass
        self.model.zero_grad()
        output[0, target_class].backward()
        
        # Get gradients and activations
        gradients = self.gradients  # (1, C, H, W)
        activations = self.activations  # (1, C, H, W)
        
        # Global average pooling on gradients
        weights = gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        
        # Weighted combination
        cam = (weights * activations).sum(dim=1, keepdim=True)  # (1, 1, H, W)
        
        # ReLU
        cam = F.relu(cam)
        
        # Normalize
        cam = cam.squeeze().cpu().numpy()
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        return cam
    
    def __call__(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None
    ) -> np.ndarray:
        """Callable interface."""
        return self.generate(input_tensor, target_class)


class GradCAMPlusPlus(GradCAM):
    """
    Grad-CAM++ - improved version with better localization.
    
    Reference:
    Chattopadhay et al. "Grad-CAM++: Generalized Gradient-based Visual 
    Explanations for Deep Convolutional Networks" (2018)
    """
    
    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate Grad-CAM++ heatmap.
        
        Args:
            input_tensor: Input image tensor (1, C, H, W)
            target_class: Target class index
            
        Returns:
            Grad-CAM++ heatmap (H, W)
        """
        # Forward pass
        input_tensor = input_tensor.to(self.device)
        self.model.zero_grad()
        
        output = self.model(input_tensor[:, :3], input_tensor[:, 3:4])
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Backward pass
        self.model.zero_grad()
        output[0, target_class].backward(retain_graph=True)
        
        gradients = self.gradients  # (1, C, H, W)
        activations = self.activations  # (1, C, H, W)
        
        # Calculate alpha weights (Grad-CAM++ specific)
        gradients_power_2 = gradients ** 2
        gradients_power_3 = gradients ** 3
        
        sum_activations = activations.sum(dim=(2, 3), keepdim=True)
        
        alpha_num = gradients_power_2
        alpha_denom = 2 * gradients_power_2 + sum_activations * gradients_power_3 + 1e-8
        alpha = alpha_num / alpha_denom
        
        # ReLU on gradients
        relu_grad = F.relu(gradients)
        
        # Weighted combination
        weights = (alpha * relu_grad).sum(dim=(2, 3), keepdim=True)
        cam = (weights * activations).sum(dim=1, keepdim=True)
        
        # ReLU
        cam = F.relu(cam)
        
        # Normalize
        cam = cam.squeeze().cpu().numpy()
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        return cam


def overlay_heatmap_on_image(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.4,
    colormap: int = cv2.COLORMAP_JET
) -> np.ndarray:
    """
    Overlay heatmap on image.
    
    Args:
        image: Original image (H, W, 3) in RGB, range [0, 255]
        heatmap: Heatmap (H, W) in range [0, 1]
        alpha: Transparency of heatmap overlay
        colormap: OpenCV colormap
        
    Returns:
        Overlayed image (H, W, 3)
    """
    # Resize heatmap to image size
    if heatmap.shape != image.shape[:2]:
        heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    
    # Convert heatmap to uint8
    heatmap_uint8 = (heatmap * 255).astype(np.uint8)
    
    # Apply colormap
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, colormap)
    
    # Convert BGR to RGB (OpenCV uses BGR)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    # Ensure image is uint8
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)
    
    # Blend
    overlayed = cv2.addWeighted(image, 1 - alpha, heatmap_colored, alpha, 0)
    
    return overlayed


def create_gradcam(
    model: nn.Module,
    method: str = 'gradcam++',
    target_layer: Optional[nn.Module] = None,
    use_cuda: bool = True
):
    """
    Factory function to create Grad-CAM or Grad-CAM++.
    
    Args:
        model: PyTorch model
        method: 'gradcam' or 'gradcam++'
        target_layer: Target layer
        use_cuda: Use CUDA
        
    Returns:
        GradCAM or GradCAMPlusPlus instance
    """
    if method.lower() == 'gradcam':
        return GradCAM(model, target_layer, use_cuda)
    elif method.lower() == 'gradcam++':
        return GradCAMPlusPlus(model, target_layer, use_cuda)
    else:
        raise ValueError(f"Unknown method: {method}")

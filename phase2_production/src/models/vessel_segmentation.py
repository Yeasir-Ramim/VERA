"""
VERA Phase 2 - Vessel Segmentation Module

U-Net architecture for retinal vessel segmentation with fine-tuning capability.
Supports multiple encoder backbones and pretrained weights.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from typing import Optional, Dict, List
import logging

logger = logging.getLogger(__name__)


class UNetVesselSegmenter(nn.Module):
    """
    U-Net based vessel segmentation network.
    
    Uses segmentation_models_pytorch library for flexible backbone selection
    and pretrained weights.
    """
    
    def __init__(
        self,
        encoder_name: str = 'resnet34',
        encoder_weights: str = 'imagenet',
        in_channels: int = 3,
        classes: int = 1,
        activation: Optional[str] = None,
        encoder_depth: int = 5,
        decoder_channels: List[int] = None
    ):
        """
        Initialize U-Net vessel segmenter.
        
        Args:
            encoder_name: Encoder backbone (resnet18, resnet34, resnet50, efficientnet-b0, etc.)
            encoder_weights: Pretrained weights ('imagenet', None)
            in_channels: Number of input channels (1 for green channel, 3 for RGB)
            classes: Number of output classes (1 for binary segmentation)
            activation: Activation function (None, 'sigmoid', 'softmax')
            encoder_depth: Depth of encoder
            decoder_channels: List of decoder channel counts
        """
        super(UNetVesselSegmenter, self).__init__()
        
        if decoder_channels is None:
            decoder_channels = [256, 128, 64, 32, 16]
        
        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            activation=activation,
            encoder_depth=encoder_depth,
            decoder_channels=decoder_channels
        )
        
        self.encoder_name = encoder_name
        self.in_channels = in_channels
        self.classes = classes
        
        logger.info(f"Initialized UNetVesselSegmenter with encoder: {encoder_name}, "
                   f"in_channels: {in_channels}, pretrained: {encoder_weights}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (B, C, H, W)
            
        Returns:
            Segmentation mask (B, 1, H, W)
        """
        return self.model(x)
    
    def freeze_encoder(self):
        """Freeze encoder weights for fine-tuning decoder only."""
        for param in self.model.encoder.parameters():
            param.requires_grad = False
        logger.info("Froze encoder weights")
    
    def unfreeze_encoder(self):
        """Unfreeze encoder weights for full fine-tuning."""
        for param in self.model.encoder.parameters():
            param.requires_grad = True
        logger.info("Unfroze encoder weights")
    
    def get_trainable_params(self):
        """Get count of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class SimpleUNet(nn.Module):
    """
    Simple U-Net implementation from scratch (alternative to SMP).
    
    Useful for understanding the architecture or custom modifications.
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        features: List[int] = None
    ):
        """
        Initialize simple U-Net.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            features: List of feature counts per level
        """
        super(SimpleUNet, self).__init__()
        
        if features is None:
            features = [64, 128, 256, 512]
        
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Encoder
        for feature in features:
            self.encoder.append(self._block(in_channels, feature))
            in_channels = feature
        
        # Bottleneck
        self.bottleneck = self._block(features[-1], features[-1] * 2)
        
        # Decoder
        for feature in reversed(features):
            self.decoder.append(
                nn.ConvTranspose2d(
                    feature * 2, feature,
                    kernel_size=2, stride=2
                )
            )
            self.decoder.append(self._block(feature * 2, feature))
        
        # Final conv
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
        
        logger.info(f"Initialized SimpleUNet with features: {features}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with skip connections."""
        skip_connections = []
        
        # Encoder
        for down in self.encoder:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
        
        # Bottleneck
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]
        
        # Decoder
        for idx in range(0, len(self.decoder), 2):
            x = self.decoder[idx](x)
            skip = skip_connections[idx // 2]
            
            # Handle size mismatch
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
            
            concat_skip = torch.cat((skip, x), dim=1)
            x = self.decoder[idx + 1](concat_skip)
        
        return self.final_conv(x)
    
    @staticmethod
    def _block(in_channels: int, out_channels: int) -> nn.Sequential:
        """Create a double convolution block."""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )


class VesselSegmentationLoss(nn.Module):
    """
    Combined loss for vessel segmentation.
    
    Combines Binary Cross-Entropy with Dice loss for better
    handling of class imbalance in vessel segmentation.
    """
    
    def __init__(
        self,
        bce_weight: float = 0.5,
        dice_weight: float = 0.5,
        smooth: float = 1e-6
    ):
        """
        Initialize combined loss.
        
        Args:
            bce_weight: Weight for BCE loss
            dice_weight: Weight for Dice loss
            smooth: Smoothing factor for Dice
        """
        super(VesselSegmentationLoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.smooth = smooth
        
        self.bce_loss = nn.BCEWithLogitsLoss()
    
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate combined loss.
        
        Args:
            predictions: Model predictions (B, 1, H, W) - logits
            targets: Ground truth masks (B, 1, H, W) - binary
            
        Returns:
            Combined loss value
        """
        # BCE Loss
        bce = self.bce_loss(predictions, targets)
        
        # Dice Loss
        predictions_sigmoid = torch.sigmoid(predictions)
        
        # Flatten
        predictions_flat = predictions_sigmoid.view(-1)
        targets_flat = targets.view(-1)
        
        # Dice coefficient
        intersection = (predictions_flat * targets_flat).sum()
        dice_coeff = (2.0 * intersection + self.smooth) / (
            predictions_flat.sum() + targets_flat.sum() + self.smooth
        )
        
        dice_loss = 1 - dice_coeff
        
        # Combined loss
        total_loss = self.bce_weight * bce + self.dice_weight * dice_loss
        
        return total_loss


def create_vessel_segmenter(config: Dict) -> UNetVesselSegmenter:
    """
    Create vessel segmenter from configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        UNetVesselSegmenter instance
    """
    vessel_config = config.get('vessel_segmentation', {})
    
    # Determine input channels based on preprocessing
    preprocessing_config = config.get('preprocessing', {})
    use_green_channel = preprocessing_config.get('use_green_channel', True)
    in_channels = 1 if use_green_channel else 3
    
    model = UNetVesselSegmenter(
        encoder_name=vessel_config.get('encoder', 'resnet34'),
        encoder_weights='imagenet' if vessel_config.get('pretrained_encoder', True) else None,
        in_channels=in_channels,
        classes=1,
        activation=None  # Apply sigmoid separately
    )
    
    # Load pretrained weights if specified
    pretrained_path = vessel_config.get('pretrained_path')
    if pretrained_path and torch.cuda.is_available():
        try:
            checkpoint = torch.load(pretrained_path)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            logger.info(f"Loaded pretrained vessel segmenter from {pretrained_path}")
        except Exception as e:
            logger.warning(f"Could not load pretrained weights: {e}")
    
    # Freeze encoder if specified
    if vessel_config.get('freeze_encoder', False):
        model.freeze_encoder()
    
    return model


def load_vessel_segmenter(
    checkpoint_path: str,
    device: str = 'cuda'
) -> UNetVesselSegmenter:
    """
    Load vessel segmenter from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model on
        
    Returns:
        Loaded model
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract model configuration
    if 'config' in checkpoint:
        config = checkpoint['config']
        model = create_vessel_segmenter(config)
    else:
        # Default configuration
        model = UNetVesselSegmenter()
    
    # Load weights
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    logger.info(f"Loaded vessel segmenter from {checkpoint_path}")
    
    return model


def train_vessel_segmenter(
    model: UNetVesselSegmenter,
    train_loader,
    val_loader,
    num_epochs: int,
    learning_rate: float,
    device: str = 'cuda',
    checkpoint_dir: str = 'models/checkpoints',
    save_best_only: bool = True
):
    """
    Train vessel segmentation model.
    
    Args:
        model: Vessel segmentation model
        train_loader: Training dataloader
        val_loader: Validation dataloader
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        device: Device for training
        checkpoint_dir: Directory to save checkpoints
        save_best_only: Only save best model
        
    Returns:
        Training history
    """
    from pathlib import Path
    from tqdm import tqdm
    
    model = model.to(device)
    
    # Loss and optimizer
    criterion = VesselSegmentationLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_dice': []
    }
    
    best_val_loss = float('inf')
    
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        train_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        for batch in train_bar:
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_bar.set_postfix({'loss': loss.item()})
        
        train_loss /= len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_dice = 0.0
        
        with torch.no_grad():
            val_bar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
            for batch in val_bar:
                images = batch['image'].to(device)
                masks = batch['mask'].to(device)
                
                outputs = model(images)
                loss = criterion(outputs, masks)
                
                # Calculate Dice coefficient
                predictions = torch.sigmoid(outputs) > 0.5
                intersection = (predictions * masks).sum()
                dice = (2.0 * intersection) / (predictions.sum() + masks.sum() + 1e-6)
                
                val_loss += loss.item()
                val_dice += dice.item()
                val_bar.set_postfix({'loss': loss.item(), 'dice': dice.item()})
        
        val_loss /= len(val_loader)
        val_dice /= len(val_loader)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_dice'].append(val_dice)
        
        logger.info(f"Epoch {epoch+1}/{num_epochs} - "
                   f"Train Loss: {train_loss:.4f}, "
                   f"Val Loss: {val_loss:.4f}, "
                   f"Val Dice: {val_dice:.4f}")
        
        # Save checkpoint
        if val_loss < best_val_loss or not save_best_only:
            best_val_loss = val_loss
            
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_dice': val_dice,
                'history': history
            }
            
            checkpoint_path = Path(checkpoint_dir) / 'vessel_segmentation_best.pth'
            torch.save(checkpoint, checkpoint_path)
            logger.info(f"Saved checkpoint to {checkpoint_path}")
    
    return history


def inference_vessel_segmenter(
    model: UNetVesselSegmenter,
    image: torch.Tensor,
    device: str = 'cuda',
    threshold: float = 0.5
) -> torch.Tensor:
    """
    Run inference on a single image.
    
    Args:
        model: Vessel segmentation model
        image: Input image tensor (1, C, H, W) or (C, H, W)
        device: Device for inference
        threshold: Threshold for binary segmentation
        
    Returns:
        Binary vessel mask (H, W)
    """
    model = model.to(device)
    model.eval()
    
    # Add batch dimension if needed
    if image.dim() == 3:
        image = image.unsqueeze(0)
    
    image = image.to(device)
    
    with torch.no_grad():
        output = model(image)
        prediction = torch.sigmoid(output)
        binary_mask = (prediction > threshold).float()
    
    # Remove batch and channel dimensions
    binary_mask = binary_mask.squeeze().cpu()
    
    return binary_mask

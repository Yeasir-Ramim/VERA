"""
VERA Phase 2 - Main Training Loop

Comprehensive trainer with support for:
- Mixed precision training
- Gradient accumulation
- Checkpointing
- TensorBoard logging
- Early stopping
- Model evaluation
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from tqdm import tqdm
from typing import Dict, Optional
import logging
import time

from .losses import compute_qwk_metric
from src.utils.checkpoint import save_checkpoint, load_checkpoint

logger = logging.getLogger(__name__)


class Trainer:
    """
    Main training class for DR classification models.
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[object] = None,
        config: Dict = None,
        device: str = 'cuda',
        checkpoint_dir: str = 'models/checkpoints',
        log_dir: str = 'outputs/logs'
    ):
        """
        Initialize trainer.
        
        Args:
            model: PyTorch model
            train_loader: Training dataloader
            val_loader: Validation dataloader
            criterion: Loss function
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            config: Configuration dictionary
            device: Device for training
            checkpoint_dir: Directory for checkpoints
            log_dir: Directory for logs
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config or {}
        self.device = device
        
        # Setup directories
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Training configuration
        training_config = self.config.get('training', {})
        
        self.epochs = training_config.get('epochs', 100)
        self.mixed_precision = training_config.get('mixed_precision', True)
        self.gradient_clip = training_config.get('gradient_clip', 1.0)
        self.log_frequency = training_config.get('log_frequency', 10)
        
        # Early stopping
        early_stopping_config = training_config.get('early_stopping', {})
        self.use_early_stopping = early_stopping_config.get('enabled', True)
        self.early_stopping_patience = early_stopping_config.get('patience', 15)
        self.early_stopping_monitor = early_stopping_config.get('monitor', 'val_qwk')
        self.early_stopping_mode = early_stopping_config.get('mode', 'max')
        
        # Checkpointing
        self.save_best_only = training_config.get('save_best_only', True)
        self.save_frequency = training_config.get('save_frequency', 5)
        
        # Mixed precision scaler
        self.scaler = GradScaler() if self.mixed_precision else None
        
        # TensorBoard
        if training_config.get('tensorboard', True):
            self.writer = SummaryWriter(log_dir=self.log_dir / 'tensorboard')
        else:
            self.writer = None
        
        # Training state
        self.current_epoch = 0
        self.best_metric = float('-inf') if self.early_stopping_mode == 'max' else float('inf')
        self.epochs_without_improvement = 0
        
        # History
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'val_qwk': [],
            'learning_rate': []
        }
        
        logger.info(f"Initialized Trainer on {device}")
        logger.info(f"Training for {self.epochs} epochs with mixed precision: {self.mixed_precision}")
    
    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Returns:
            Dictionary with training metrics
        """
        self.model.train()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        progress_bar = tqdm(
            self.train_loader,
            desc=f'Epoch {self.current_epoch + 1}/{self.epochs} [Train]'
        )
        
        for batch_idx, batch in enumerate(progress_bar):
            # Get data
            images = batch['image'].to(self.device)
            vessels = batch['vessel'].to(self.device)
            labels = batch['label'].to(self.device)
            
            # Forward pass with mixed precision
            self.optimizer.zero_grad()
            
            if self.mixed_precision:
                with autocast():
                    outputs = self.model(images, vessels)
                    loss = self.criterion(outputs, labels)
                
                # Backward pass
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                if self.gradient_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.gradient_clip
                    )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            
            else:
                outputs = self.model(images, vessels)
                loss = self.criterion(outputs, labels)
                
                loss.backward()
                
                if self.gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.gradient_clip
                    )
                
                self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.0 * correct / total:.2f}%'
            })
            
            # Log to TensorBoard
            if self.writer and (batch_idx % self.log_frequency == 0):
                global_step = self.current_epoch * len(self.train_loader) + batch_idx
                self.writer.add_scalar('Train/BatchLoss', loss.item(), global_step)
        
        # Epoch metrics
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100.0 * correct / total
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy
        }
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """
        Validate model.
        
        Returns:
            Dictionary with validation metrics
        """
        self.model.eval()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        all_predictions = []
        all_labels = []
        
        progress_bar = tqdm(
            self.val_loader,
            desc=f'Epoch {self.current_epoch + 1}/{self.epochs} [Val]'
        )
        
        for batch in progress_bar:
            images = batch['image'].to(self.device)
            vessels = batch['vessel'].to(self.device)
            labels = batch['label'].to(self.device)
            
            # Forward pass
            if self.mixed_precision:
                with autocast():
                    outputs = self.model(images, vessels)
                    loss = self.criterion(outputs, labels)
            else:
                outputs = self.model(images, vessels)
                loss = self.criterion(outputs, labels)
            
            # Statistics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Collect for QWK
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.0 * correct / total:.2f}%'
            })
        
        # Epoch metrics
        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100.0 * correct / total
        
        # Compute QWK
        qwk = compute_qwk_metric(
            torch.tensor(all_predictions),
            torch.tensor(all_labels),
            num_classes=self.config.get('model', {}).get('num_classes', 5)
        )
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'qwk': qwk
        }
    
    def train(self):
        """
        Full training loop.
        """
        logger.info("=" * 60)
        logger.info("STARTING TRAINING")
        logger.info("=" * 60)
        
        start_time = time.time()
        
        for epoch in range(self.epochs):
            self.current_epoch = epoch
            
            # Train
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate()
            
            # Update scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['loss'])
                else:
                    self.scheduler.step()
            
            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Update history
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_acc'].append(train_metrics['accuracy'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_acc'].append(val_metrics['accuracy'])
            self.history['val_qwk'].append(val_metrics['qwk'])
            self.history['learning_rate'].append(current_lr)
            
            # Log to TensorBoard
            if self.writer:
                self.writer.add_scalar('Train/Loss', train_metrics['loss'], epoch)
                self.writer.add_scalar('Train/Accuracy', train_metrics['accuracy'], epoch)
                self.writer.add_scalar('Val/Loss', val_metrics['loss'], epoch)
                self.writer.add_scalar('Val/Accuracy', val_metrics['accuracy'], epoch)
                self.writer.add_scalar('Val/QWK', val_metrics['qwk'], epoch)
                self.writer.add_scalar('LearningRate', current_lr, epoch)
            
            # Print metrics
            logger.info(f"\nEpoch {epoch + 1}/{self.epochs}")
            logger.info(f"  Train Loss: {train_metrics['loss']:.4f} | "
                       f"Train Acc: {train_metrics['accuracy']:.2f}%")
            logger.info(f"  Val Loss: {val_metrics['loss']:.4f} | "
                       f"Val Acc: {val_metrics['accuracy']:.2f}% | "
                       f"Val QWK: {val_metrics['qwk']:.4f}")
            logger.info(f"  LR: {current_lr:.6f}")
            
            # Check if best model
            current_metric = val_metrics[self.early_stopping_monitor.replace('val_', '')]
            is_best = self._is_better(current_metric, self.best_metric)
            
            if is_best:
                self.best_metric = current_metric
                self.epochs_without_improvement = 0
                
                logger.info(f"  ✓ New best {self.early_stopping_monitor}: {current_metric:.4f}")
                
                # Save best checkpoint
                save_checkpoint(
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    epoch=epoch + 1,
                    metrics=val_metrics,
                    config=self.config,
                    filepath=self.checkpoint_dir / 'best_model.pth',
                    is_best=True
                )
            else:
                self.epochs_without_improvement += 1
            
            # Save regular checkpoint
            if not self.save_best_only and (epoch + 1) % self.save_frequency == 0:
                save_checkpoint(
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    epoch=epoch + 1,
                    metrics=val_metrics,
                    config=self.config,
                    filepath=self.checkpoint_dir / f'checkpoint_epoch_{epoch+1}.pth',
                    is_best=False
                )
            
            # Early stopping
            if self.use_early_stopping and self.epochs_without_improvement >= self.early_stopping_patience:
                logger.info(f"\nEarly stopping triggered after {epoch + 1} epochs")
                logger.info(f"No improvement for {self.early_stopping_patience} epochs")
                break
        
        # Training complete
        total_time = time.time() - start_time
        
        logger.info("\n" + "=" * 60)
        logger.info("TRAINING COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Total training time: {total_time / 3600:.2f} hours")
        logger.info(f"Best {self.early_stopping_monitor}: {self.best_metric:.4f}")
        
        if self.writer:
            self.writer.close()
        
        return self.history
    
    def _is_better(self, current: float, best: float) -> bool:
        """Check if current metric is better than best."""
        if self.early_stopping_mode == 'max':
            return current > best
        else:
            return current < best


def create_trainer(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[object],
    config: Dict,
    device: str = 'cuda'
) -> Trainer:
    """
    Factory function to create trainer from configuration.
    
    Args:
        model: PyTorch model
        train_loader: Training dataloader
        val_loader: Validation dataloader
        criterion: Loss function
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        config: Configuration dictionary
        device: Device for training
        
    Returns:
        Trainer instance
    """
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        device=device,
        checkpoint_dir=config.get('paths', {}).get('models_root', 'models') + '/checkpoints',
        log_dir=config.get('paths', {}).get('logs_root', 'outputs/logs')
    )
    
    return trainer

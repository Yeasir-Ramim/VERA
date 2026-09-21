"""
VERA Phase 2 - Main Training Script

Comprehensive training script with support for:
- Multi-dataset training
- Multiple fusion strategies
- Configurable loss functions and optimizers
- Mixed precision training
- Checkpointing and early stopping
- TensorBoard logging
"""

import os
import sys
import argparse
from pathlib import Path
import yaml
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import create_dr_classifier, count_parameters
from src.data import (
    create_preprocessor,
    create_train_augmentation,
    create_vessel_cache,
    create_dataloaders,
    get_class_weights
)
from src.training import (
    create_loss_function,
    create_optimizer,
    create_scheduler,
    create_trainer
)
from src.utils import (
    setup_logging,
    load_config,
    save_config,
    set_seed,
    get_logger
)

logger = get_logger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='VERA Phase 2 - Train DR Classification Model'
    )
    
    # Configuration
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--experiment_name', type=str, default=None,
                       help='Experiment name (overrides config)')
    
    # Model configuration
    parser.add_argument('--backbone', type=str, default=None,
                       choices=['resnet18', 'resnet50', 'efficientnet_b0', 'efficientnet_b3'],
                       help='CNN backbone')
    parser.add_argument('--fusion', type=str, default=None,
                       choices=['early_fusion', 'dual_branch', 'attention_gated'],
                       help='Fusion strategy')
    parser.add_argument('--no_vessel', action='store_true',
                       help='Disable vessel channel (baseline)')
    
    # Training configuration
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=None,
                       help='Learning rate')
    parser.add_argument('--loss', type=str, default=None,
                       choices=['cross_entropy', 'focal', 'qwk', 'hybrid'],
                       help='Loss function')
    
    # Data configuration
    parser.add_argument('--use_eyepacs', action='store_true',
                       help='Include EyePACS dataset')
    parser.add_argument('--use_messidor', action='store_true',
                       help='Include Messidor-2 for validation')
    
    # System configuration
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device for training')
    parser.add_argument('--num_workers', type=int, default=None,
                       help='Number of dataloader workers')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    # Output configuration
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory')
    parser.add_argument('--checkpoint_dir', type=str, default=None,
                       help='Checkpoint directory')
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume from checkpoint')
    
    # Flags
    parser.add_argument('--no_pretrained', action='store_true',
                       help='Do not use pretrained weights')
    parser.add_argument('--debug', action='store_true',
                       help='Debug mode (small subset)')
    
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()
    
    # Setup logging
    log_file = 'outputs/logs/train.log'
    if args.output_dir:
        log_file = Path(args.output_dir) / 'logs' / 'train.log'
    
    setup_logging(log_file=str(log_file))
    
    logger.info("="*80)
    logger.info("VERA PHASE 2 - MAIN TRAINING SCRIPT")
    logger.info("="*80)
    
    # Load configuration
    config = load_config(args.config)
    
    # Override with command line arguments
    if args.experiment_name:
        config['experiment']['name'] = args.experiment_name
    
    if args.backbone:
        config['model']['backbone'] = args.backbone
    
    if args.fusion:
        config['model']['fusion_strategy'] = args.fusion
    
    if args.no_vessel:
        config['model']['use_vessel_channel'] = False
    
    if args.epochs:
        config['training']['epochs'] = args.epochs
    
    if args.batch_size:
        config['data']['batch_size'] = args.batch_size
    
    if args.learning_rate:
        config['training']['learning_rate'] = args.learning_rate
    
    if args.loss:
        config['training']['loss']['type'] = args.loss
    
    if args.use_eyepacs:
        config['data']['datasets']['eyepacs']['enabled'] = True
    
    if args.use_messidor:
        config['data']['datasets']['messidor2']['enabled'] = True
    
    if args.num_workers:
        config['data']['num_workers'] = args.num_workers
    
    if args.no_pretrained:
        config['model']['pretrained'] = False
    
    # Set random seed
    set_seed(args.seed)
    logger.info(f"Set random seed to {args.seed}")
    
    # Set device
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA not available, using CPU")
        device = 'cpu'
    
    logger.info(f"Using device: {device}")
    
    # Log configuration
    logger.info("\n" + "="*80)
    logger.info("CONFIGURATION")
    logger.info("="*80)
    logger.info(f"Experiment: {config['experiment']['name']}")
    logger.info(f"Backbone: {config['model']['backbone']}")
    logger.info(f"Fusion: {config['model']['fusion_strategy']}")
    logger.info(f"Use Vessel Channel: {config['model']['use_vessel_channel']}")
    logger.info(f"Epochs: {config['training']['epochs']}")
    logger.info(f"Batch Size: {config['data']['batch_size']}")
    logger.info(f"Learning Rate: {config['training']['learning_rate']}")
    logger.info(f"Loss: {config['training']['loss']['type']}")
    
    # Create output directories
    output_dir = Path(args.output_dir) if args.output_dir else Path('outputs/experiments') / config['experiment']['name']
    output_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else output_dir / 'checkpoints'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    config_save_path = output_dir / 'config.yaml'
    save_config(config, str(config_save_path))
    logger.info(f"Saved configuration to {config_save_path}")
    
    # Create preprocessor
    logger.info("\n" + "="*80)
    logger.info("CREATING DATA PIPELINE")
    logger.info("="*80)
    
    preprocessor = create_preprocessor(config)
    augmentation = create_train_augmentation(config)
    
    # Create vessel cache
    vessel_cache = None
    if config['model'].get('use_vessel_channel', True):
        vessel_cache = create_vessel_cache(config)
        cache_stats = vessel_cache.get_cache_stats()
        logger.info(f"Vessel cache: {cache_stats['num_cached']} images, "
                   f"{cache_stats['cache_size_mb']:.2f} MB")
    
    # Create dataloaders
    dataloaders = create_dataloaders(
        config=config,
        preprocessor=preprocessor,
        augmentation=augmentation,
        vessel_cache=vessel_cache
    )
    
    train_loader = dataloaders['train']
    val_loader = dataloaders['val']
    test_loader = dataloaders['test']
    
    logger.info(f"Train batches: {len(train_loader)}")
    logger.info(f"Val batches: {len(val_loader)}")
    logger.info(f"Test batches: {len(test_loader)}")
    
    # Get class weights
    # Collect all training labels
    train_labels = []
    for batch in train_loader:
        train_labels.extend(batch['label'].tolist())
    
    class_weights = get_class_weights(train_labels, num_classes=5)
    logger.info(f"Class weights: {class_weights.numpy()}")
    
    # Create model
    logger.info("\n" + "="*80)
    logger.info("CREATING MODEL")
    logger.info("="*80)
    
    model = create_dr_classifier(config)
    model.to(device)
    
    params = count_parameters(model)
    logger.info(f"Model: {config['model']['backbone']} with {config['model']['fusion_strategy']} fusion")
    logger.info(f"Total parameters: {params['total']:,}")
    logger.info(f"Trainable parameters: {params['trainable']:,}")
    
    # Create loss function
    if config['training']['loss'].get('class_weights') in ['auto', 'balanced']:
        criterion = create_loss_function(config, class_weights.to(device))
    else:
        criterion = create_loss_function(config, None)
    
    logger.info(f"Loss function: {config['training']['loss']['type']}")
    
    # Create optimizer
    optimizer = create_optimizer(model.parameters(), config)
    logger.info(f"Optimizer: {config['training']['optimizer']}")
    
    # Create scheduler
    scheduler = create_scheduler(optimizer, config, steps_per_epoch=len(train_loader))
    if scheduler:
        logger.info(f"Scheduler: {config['training']['scheduler']['type']}")
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        from src.utils import load_checkpoint
        checkpoint = load_checkpoint(
            args.resume,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device
        )
        start_epoch = checkpoint.get('epoch', 0)
        logger.info(f"Resumed from checkpoint at epoch {start_epoch}")
    
    # Create trainer
    logger.info("\n" + "="*80)
    logger.info("STARTING TRAINING")
    logger.info("="*80)
    
    trainer = create_trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        device=device
    )
    
    # Train
    history = trainer.train()
    
    # Evaluate on test set
    logger.info("\n" + "="*80)
    logger.info("EVALUATING ON TEST SET")
    logger.info("="*80)
    
    test_metrics = trainer.validate()
    
    logger.info(f"Test Loss: {test_metrics['loss']:.4f}")
    logger.info(f"Test Accuracy: {test_metrics['accuracy']:.2f}%")
    logger.info(f"Test QWK: {test_metrics['qwk']:.4f}")
    
    # Save final results
    results = {
        'config': config,
        'history': history,
        'test_metrics': test_metrics,
        'model_parameters': params
    }
    
    import json
    results_path = output_dir / 'results.json'
    with open(results_path, 'w') as f:
        # Convert tensors to lists for JSON serialization
        serializable_results = {
            'experiment': config['experiment']['name'],
            'test_loss': test_metrics['loss'],
            'test_accuracy': test_metrics['accuracy'],
            'test_qwk': test_metrics['qwk'],
            'best_val_qwk': max(history['val_qwk']),
            'total_epochs': len(history['train_loss']),
            'parameters': params
        }
        json.dump(serializable_results, f, indent=2)
    
    logger.info(f"Saved results to {results_path}")
    
    logger.info("\n" + "="*80)
    logger.info("TRAINING COMPLETE")
    logger.info("="*80)
    logger.info(f"Best validation QWK: {max(history['val_qwk']):.4f}")
    logger.info(f"Test QWK: {test_metrics['qwk']:.4f}")
    logger.info(f"Checkpoints saved to: {checkpoint_dir}")
    logger.info(f"Results saved to: {output_dir}")


if __name__ == '__main__':
    main()

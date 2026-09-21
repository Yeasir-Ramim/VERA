"""
VERA Phase 2 - Model Evaluation Script

Evaluate trained models on test sets with comprehensive metrics and visualizations.
"""

import os
import sys
import argparse
from pathlib import Path
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import load_dr_classifier
from src.data import create_preprocessor, create_vessel_cache, create_dataloaders
from src.evaluation import ModelEvaluator
from src.utils import setup_logging, load_config, get_logger

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description='VERA Phase 2 - Evaluate Trained Model'
    )
    
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to configuration file (will try checkpoint if not provided)')
    parser.add_argument('--dataset', type=str, default='test',
                       choices=['test', 'val', 'messidor2'],
                       help='Dataset to evaluate on')
    parser.add_argument('--output_dir', type=str, default='outputs/evaluation',
                       help='Output directory for evaluation results')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device for evaluation')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for evaluation')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(log_file='outputs/logs/evaluate.log')
    
    logger.info("="*80)
    logger.info("VERA PHASE 2 - MODEL EVALUATION")
    logger.info("="*80)
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Dataset: {args.dataset}")
    
    # Set device
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA not available, using CPU")
        device = 'cpu'
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # Load configuration
    if args.config:
        config = load_config(args.config)
    elif 'config' in checkpoint:
        config = checkpoint['config']
    else:
        raise ValueError("No configuration provided and none found in checkpoint")
    
    logger.info(f"\nModel Configuration:")
    logger.info(f"  Backbone: {config['model']['backbone']}")
    logger.info(f"  Fusion: {config['model']['fusion_strategy']}")
    logger.info(f"  Vessel Channel: {config['model']['use_vessel_channel']}")
    
    # Override batch size if specified
    if args.batch_size:
        config['data']['batch_size'] = args.batch_size
    
    # Create data pipeline
    logger.info("\nCreating data pipeline...")
    preprocessor = create_preprocessor(config)
    vessel_cache = create_vessel_cache(config) if config['model'].get('use_vessel_channel', True) else None
    
    dataloaders = create_dataloaders(
        config=config,
        preprocessor=preprocessor,
        augmentation=None,  # No augmentation for evaluation
        vessel_cache=vessel_cache
    )
    
    # Select dataloader
    if args.dataset == 'test':
        dataloader = dataloaders['test']
    elif args.dataset == 'val':
        dataloader = dataloaders['val']
    else:
        # Messidor-2
        dataloader = dataloaders['test']  # Assuming Messidor-2 is in test set
    
    logger.info(f"Evaluation batches: {len(dataloader)}")
    
    # Load model
    logger.info("\nLoading model...")
    model = load_dr_classifier(args.checkpoint, config, device)
    
    # Create evaluator
    evaluator = ModelEvaluator(
        model=model,
        dataloader=dataloader,
        device=device,
        num_classes=config['model'].get('num_classes', 5)
    )
    
    # Run evaluation
    logger.info("\n" + "="*80)
    logger.info("RUNNING EVALUATION")
    logger.info("="*80)
    
    results = evaluator.evaluate()
    
    # Print results
    evaluator.print_results(results)
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    evaluator.save_results(results, str(output_dir))
    
    logger.info("\n" + "="*80)
    logger.info("EVALUATION COMPLETE")
    logger.info("="*80)
    logger.info(f"Results saved to: {output_dir}")
    logger.info(f"QWK: {results['metrics']['qwk']:.4f}")
    logger.info(f"Accuracy: {results['metrics']['accuracy']:.4f}")


if __name__ == '__main__':
    main()

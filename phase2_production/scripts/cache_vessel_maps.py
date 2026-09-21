"""
VERA Phase 2 - Vessel Map Caching Script

Pre-compute and cache vessel segmentation maps for all datasets
to accelerate training iterations.
"""

import os
import sys
import argparse
import yaml
from pathlib import Path
import torch
from tqdm import tqdm
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.datasets import APTOSDataset, EyePACSDataset, Messidor2Dataset
from src.data.vessel_cache import create_vessel_cache, batch_cache_vessel_maps
from src.models.vessel_segmentation import create_vessel_segmenter, load_vessel_segmenter
from src.utils.logging import setup_logging
from src.utils.config import load_config

logger = logging.getLogger(__name__)


def cache_dataset_vessel_maps(
    dataset_name: str,
    config: dict,
    vessel_model,
    vessel_cache,
    device: str = 'cuda',
    batch_size: int = 8
):
    """
    Cache vessel maps for a specific dataset.
    
    Args:
        dataset_name: Name of dataset ('aptos', 'eyepacs', 'messidor2')
        config: Configuration dictionary
        vessel_model: Vessel segmentation model
        vessel_cache: VesselCache instance
        device: Device for computation
        batch_size: Batch size for inference
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"CACHING VESSEL MAPS FOR {dataset_name.upper()}")
    logger.info(f"{'='*60}")
    
    datasets_config = config.get('data', {}).get('datasets', {})
    dataset_config = datasets_config.get(dataset_name, {})
    
    if not dataset_config.get('enabled', False):
        logger.warning(f"{dataset_name} is not enabled in configuration")
        return
    
    dataset_path = dataset_config['path']
    
    # Load dataset
    all_image_paths = []
    
    try:
        if dataset_name == 'aptos':
            train_paths, _ = APTOSDataset.load(dataset_path, split='train')
            val_paths, _ = APTOSDataset.load(dataset_path, split='val')
            test_paths, _ = APTOSDataset.load(dataset_path, split='test')
            all_image_paths = train_paths + val_paths + test_paths
        
        elif dataset_name == 'eyepacs':
            train_paths, _ = EyePACSDataset.load(dataset_path, split='train')
            val_paths, _ = EyePACSDataset.load(dataset_path, split='val')
            all_image_paths = train_paths + val_paths
        
        elif dataset_name == 'messidor2':
            test_paths, _ = Messidor2Dataset.load(dataset_path)
            all_image_paths = test_paths
        
        else:
            logger.error(f"Unknown dataset: {dataset_name}")
            return
        
        logger.info(f"Found {len(all_image_paths)} images to process")
        
        # Batch cache vessel maps
        batch_cache_vessel_maps(
            image_paths=all_image_paths,
            vessel_segmenter=vessel_model,
            cache=vessel_cache,
            batch_size=batch_size,
            device=device
        )
        
        logger.info(f"Successfully cached vessel maps for {dataset_name}")
        
    except Exception as e:
        logger.error(f"Error caching vessel maps for {dataset_name}: {e}")


def main():
    parser = argparse.ArgumentParser(
        description='VERA Phase 2 - Cache Vessel Segmentation Maps'
    )
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--dataset', type=str, default='all',
                       choices=['all', 'aptos', 'eyepacs', 'messidor2'],
                       help='Dataset to cache')
    parser.add_argument('--model_path', type=str, default=None,
                       help='Path to vessel segmentation model checkpoint')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size for inference')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device for computation')
    parser.add_argument('--clear_cache', action='store_true',
                       help='Clear existing cache before caching')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(log_file='outputs/logs/cache_vessel_maps.log')
    
    logger.info("="*60)
    logger.info("VERA PHASE 2 - VESSEL MAP CACHING")
    logger.info("="*60)
    
    # Load configuration
    config = load_config(args.config)
    
    # Set device
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA not available, using CPU")
        device = 'cpu'
    
    logger.info(f"Using device: {device}")
    
    # Create vessel cache
    vessel_cache = create_vessel_cache(config)
    
    if args.clear_cache:
        logger.info("Clearing existing cache...")
        vessel_cache.clear()
    
    # Print cache stats
    stats = vessel_cache.get_cache_stats()
    logger.info(f"\nCache Statistics:")
    logger.info(f"  Cache directory: {stats['cache_dir']}")
    logger.info(f"  Currently cached: {stats['num_cached']} images")
    logger.info(f"  Cache size: {stats['cache_size_mb']:.2f} MB")
    
    # Load or create vessel segmentation model
    if args.model_path:
        logger.info(f"\nLoading vessel segmenter from {args.model_path}")
        vessel_model = load_vessel_segmenter(args.model_path, device=device)
    else:
        logger.info("\nCreating vessel segmenter from config")
        vessel_model = create_vessel_segmenter(config)
        vessel_model.to(device)
        vessel_model.eval()
    
    # Cache vessel maps
    if args.dataset == 'all':
        datasets = ['aptos', 'eyepacs', 'messidor2']
    else:
        datasets = [args.dataset]
    
    for dataset_name in datasets:
        cache_dataset_vessel_maps(
            dataset_name=dataset_name,
            config=config,
            vessel_model=vessel_model,
            vessel_cache=vessel_cache,
            device=device,
            batch_size=args.batch_size
        )
    
    # Print final cache stats
    final_stats = vessel_cache.get_cache_stats()
    logger.info(f"\n{'='*60}")
    logger.info("CACHING COMPLETE")
    logger.info(f"{'='*60}")
    logger.info(f"  Total cached images: {final_stats['num_cached']}")
    logger.info(f"  Total cache size: {final_stats['cache_size_mb']:.2f} MB")
    logger.info(f"  Cache directory: {final_stats['cache_dir']}")


if __name__ == '__main__':
    main()

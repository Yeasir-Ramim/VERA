"""
VERA Phase 2 - Data Preprocessing Script

Validates dataset structure, generates statistics, and optionally
preprocesses all images for faster training.
"""

import os
import sys
import argparse
import yaml
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.datasets import APTOSDataset, EyePACSDataset, Messidor2Dataset
from src.data.preprocessing import create_preprocessor
from src.utils.logging import setup_logging

logger = logging.getLogger(__name__)


def validate_dataset_structure(config: dict):
    """
    Validate that all configured datasets exist and have correct structure.
    
    Args:
        config: Configuration dictionary
    """
    logger.info("=" * 60)
    logger.info("VALIDATING DATASET STRUCTURE")
    logger.info("=" * 60)
    
    datasets_config = config.get('data', {}).get('datasets', {})
    validation_results = {}
    
    # Check APTOS
    if datasets_config.get('aptos', {}).get('enabled', False):
        aptos_path = Path(datasets_config['aptos']['path'])
        logger.info(f"\nValidating APTOS dataset at: {aptos_path}")
        
        checks = {
            'directory_exists': aptos_path.exists(),
            'train_images_exists': (aptos_path / 'train_images').exists(),
            'train_csv_exists': (aptos_path / 'train.csv').exists(),
        }
        
        if all(checks.values()):
            logger.info("✓ APTOS dataset structure is valid")
            
            # Count images
            try:
                train_paths, train_labels = APTOSDataset.load(str(aptos_path), split='train')
                val_paths, val_labels = APTOSDataset.load(str(aptos_path), split='val')
                test_paths, test_labels = APTOSDataset.load(str(aptos_path), split='test')
                
                logger.info(f"  - Train samples: {len(train_paths)}")
                logger.info(f"  - Val samples: {len(val_paths)}")
                logger.info(f"  - Test samples: {len(test_paths)}")
                
                validation_results['aptos'] = {
                    'valid': True,
                    'train': len(train_paths),
                    'val': len(val_paths),
                    'test': len(test_paths)
                }
            except Exception as e:
                logger.error(f"✗ Error loading APTOS: {e}")
                validation_results['aptos'] = {'valid': False, 'error': str(e)}
        else:
            logger.error("✗ APTOS dataset structure is invalid")
            for check, result in checks.items():
                logger.error(f"  - {check}: {result}")
            validation_results['aptos'] = {'valid': False, 'checks': checks}
    
    # Check EyePACS
    if datasets_config.get('eyepacs', {}).get('enabled', False):
        eyepacs_path = Path(datasets_config['eyepacs']['path'])
        logger.info(f"\nValidating EyePACS dataset at: {eyepacs_path}")
        
        checks = {
            'directory_exists': eyepacs_path.exists(),
            'train_dir_exists': (eyepacs_path / 'train').exists(),
            'train_labels_exists': (eyepacs_path / 'trainLabels.csv').exists(),
        }
        
        if checks['directory_exists'] and checks['train_labels_exists']:
            logger.info("✓ EyePACS dataset structure is valid")
            
            try:
                train_paths, train_labels = EyePACSDataset.load(
                    str(eyepacs_path),
                    split='train',
                    sample_fraction=0.1  # Sample for validation
                )
                
                logger.info(f"  - Available train samples: {len(train_paths)} (10% sample)")
                
                validation_results['eyepacs'] = {
                    'valid': True,
                    'train': len(train_paths) * 10  # Estimate full size
                }
            except Exception as e:
                logger.warning(f"! EyePACS loading issue: {e}")
                validation_results['eyepacs'] = {'valid': False, 'error': str(e)}
        else:
            logger.warning("! EyePACS dataset not found or incomplete")
            validation_results['eyepacs'] = {'valid': False, 'checks': checks}
    
    # Check Messidor-2
    if datasets_config.get('messidor2', {}).get('enabled', False):
        messidor2_path = Path(datasets_config['messidor2']['path'])
        logger.info(f"\nValidating Messidor-2 dataset at: {messidor2_path}")
        
        if messidor2_path.exists():
            try:
                test_paths, test_labels = Messidor2Dataset.load(str(messidor2_path))
                logger.info("✓ Messidor-2 dataset is valid")
                logger.info(f"  - Test samples: {len(test_paths)}")
                
                validation_results['messidor2'] = {
                    'valid': True,
                    'test': len(test_paths)
                }
            except Exception as e:
                logger.warning(f"! Messidor-2 loading issue: {e}")
                validation_results['messidor2'] = {'valid': False, 'error': str(e)}
        else:
            logger.warning("! Messidor-2 dataset not found")
            validation_results['messidor2'] = {'valid': False}
    
    logger.info("\n" + "=" * 60)
    logger.info("VALIDATION SUMMARY")
    logger.info("=" * 60)
    
    for dataset_name, results in validation_results.items():
        status = "✓ VALID" if results.get('valid', False) else "✗ INVALID"
        logger.info(f"{dataset_name.upper()}: {status}")
    
    return validation_results


def generate_dataset_statistics(config: dict, output_dir: Path):
    """
    Generate statistics and visualizations for datasets.
    
    Args:
        config: Configuration dictionary
        output_dir: Output directory for figures
    """
    logger.info("\n" + "=" * 60)
    logger.info("GENERATING DATASET STATISTICS")
    logger.info("=" * 60)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    datasets_config = config.get('data', {}).get('datasets', {})
    
    all_stats = {}
    
    # APTOS statistics
    if datasets_config.get('aptos', {}).get('enabled', False):
        aptos_path = datasets_config['aptos']['path']
        
        try:
            train_paths, train_labels = APTOSDataset.load(str(aptos_path), split='train')
            val_paths, val_labels = APTOSDataset.load(str(aptos_path), split='val')
            test_paths, test_labels = APTOSDataset.load(str(aptos_path), split='test')
            
            all_labels = train_labels + val_labels + test_labels
            
            # Class distribution
            class_counts = np.bincount(all_labels, minlength=5)
            
            all_stats['aptos'] = {
                'total': len(all_labels),
                'train': len(train_labels),
                'val': len(val_labels),
                'test': len(test_labels),
                'class_distribution': class_counts.tolist(),
                'class_percentages': (class_counts / len(all_labels) * 100).tolist()
            }
            
            logger.info("\nAPTOS Dataset Statistics:")
            logger.info(f"  Total samples: {len(all_labels)}")
            logger.info(f"  Train: {len(train_labels)}")
            logger.info(f"  Val: {len(val_labels)}")
            logger.info(f"  Test: {len(test_labels)}")
            logger.info(f"  Class distribution:")
            for i, count in enumerate(class_counts):
                pct = count / len(all_labels) * 100
                logger.info(f"    Grade {i}: {count} ({pct:.1f}%)")
            
            # Plot class distribution
            plt.figure(figsize=(10, 6))
            grades = ['No DR\n(0)', 'Mild\n(1)', 'Moderate\n(2)', 'Severe\n(3)', 'Proliferative\n(4)']
            colors = ['#2ecc71', '#3498db', '#f39c12', '#e74c3c', '#8e44ad']
            
            plt.bar(grades, class_counts, color=colors, alpha=0.7, edgecolor='black')
            plt.xlabel('DR Severity Grade', fontsize=12, fontweight='bold')
            plt.ylabel('Number of Samples', fontsize=12, fontweight='bold')
            plt.title('APTOS 2019 - Class Distribution', fontsize=14, fontweight='bold')
            plt.grid(axis='y', alpha=0.3)
            
            # Add count labels on bars
            for i, count in enumerate(class_counts):
                plt.text(i, count + max(class_counts)*0.02, str(count), 
                        ha='center', va='bottom', fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(output_dir / 'aptos_class_distribution.png', dpi=150)
            plt.close()
            
            logger.info(f"  Saved class distribution plot to {output_dir / 'aptos_class_distribution.png'}")
            
        except Exception as e:
            logger.error(f"Error generating APTOS statistics: {e}")
    
    # Save statistics to JSON
    import json
    stats_file = output_dir / 'dataset_statistics.json'
    with open(stats_file, 'w') as f:
        json.dump(all_stats, f, indent=2)
    
    logger.info(f"\nSaved statistics to {stats_file}")
    
    return all_stats


def test_preprocessing_pipeline(config: dict, output_dir: Path):
    """
    Test preprocessing pipeline on sample images.
    
    Args:
        config: Configuration dictionary
        output_dir: Output directory for visualizations
    """
    logger.info("\n" + "=" * 60)
    logger.info("TESTING PREPROCESSING PIPELINE")
    logger.info("=" * 60)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create preprocessor
    preprocessor = create_preprocessor(config)
    
    # Load sample images
    datasets_config = config.get('data', {}).get('datasets', {})
    
    if datasets_config.get('aptos', {}).get('enabled', False):
        aptos_path = datasets_config['aptos']['path']
        
        try:
            train_paths, _ = APTOSDataset.load(str(aptos_path), split='train')
            
            # Test on first few images
            num_samples = min(3, len(train_paths))
            
            for i in range(num_samples):
                import cv2
                from src.data.preprocessing import visualize_preprocessing_steps
                
                image_path = train_paths[i]
                image = cv2.imread(image_path)
                
                if image is None:
                    continue
                
                # Visualize preprocessing steps
                output_path = output_dir / f'preprocessing_steps_sample_{i+1}.png'
                visualize_preprocessing_steps(image, preprocessor, str(output_path))
                
                logger.info(f"  Saved preprocessing visualization {i+1} to {output_path}")
            
        except Exception as e:
            logger.error(f"Error testing preprocessing: {e}")


def main():
    parser = argparse.ArgumentParser(description='VERA Phase 2 - Data Preprocessing')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--output_dir', type=str, default='outputs/data_analysis',
                       help='Output directory for analysis results')
    parser.add_argument('--validate', action='store_true',
                       help='Validate dataset structure')
    parser.add_argument('--stats', action='store_true',
                       help='Generate dataset statistics')
    parser.add_argument('--test_preprocessing', action='store_true',
                       help='Test preprocessing pipeline')
    parser.add_argument('--all', action='store_true',
                       help='Run all analyses')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(log_file='outputs/logs/preprocess_data.log')
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run analyses
    if args.all or args.validate:
        validate_dataset_structure(config)
    
    if args.all or args.stats:
        generate_dataset_statistics(config, output_dir)
    
    if args.all or args.test_preprocessing:
        test_preprocessing_pipeline(config, output_dir)
    
    logger.info("\n" + "=" * 60)
    logger.info("DATA PREPROCESSING COMPLETE")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()

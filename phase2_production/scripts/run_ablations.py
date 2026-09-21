"""
VERA Phase 2 - Ablation Study Script

Automated ablation experiments to evaluate:
1. Vessel channel contribution
2. Fusion topology comparison
3. Backbone architecture comparison
4. Loss function comparison
"""

import os
import sys
import argparse
from pathlib import Path
import yaml
import json
import subprocess
from typing import List, Dict
import pandas as pd
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import setup_logging, load_config, get_logger

logger = get_logger(__name__)


def run_training_experiment(
    config_path: str,
    experiment_name: str,
    overrides: Dict,
    output_base_dir: str
) -> Dict:
    """
    Run a single training experiment with configuration overrides.
    
    Args:
        config_path: Path to base configuration
        experiment_name: Name for this experiment
        overrides: Dictionary of configuration overrides
        output_base_dir: Base output directory
        
    Returns:
        Results dictionary
    """
    # Create experiment-specific output directory
    output_dir = Path(output_base_dir) / experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Build command
    cmd = [
        sys.executable,
        'scripts/train.py',
        '--config', config_path,
        '--experiment_name', experiment_name,
        '--output_dir', str(output_dir)
    ]
    
    # Add overrides
    for key, value in overrides.items():
        if key == 'backbone':
            cmd.extend(['--backbone', value])
        elif key == 'fusion':
            cmd.extend(['--fusion', value])
        elif key == 'loss':
            cmd.extend(['--loss', value])
        elif key == 'no_vessel':
            if value:
                cmd.append('--no_vessel')
        elif key == 'epochs':
            cmd.extend(['--epochs', str(value)])
        elif key == 'batch_size':
            cmd.extend(['--batch_size', str(value)])
    
    logger.info(f"\n{'='*80}")
    logger.info(f"Running experiment: {experiment_name}")
    logger.info(f"Command: {' '.join(cmd)}")
    logger.info(f"{'='*80}\n")
    
    # Run experiment
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.info(f"Experiment {experiment_name} completed successfully")
        
        # Load results
        results_path = output_dir / 'results.json'
        if results_path.exists():
            with open(results_path, 'r') as f:
                results = json.load(f)
            return results
        else:
            logger.warning(f"Results file not found for {experiment_name}")
            return {'experiment': experiment_name, 'status': 'completed_no_results'}
    
    except subprocess.CalledProcessError as e:
        logger.error(f"Experiment {experiment_name} failed:")
        logger.error(e.stderr)
        return {'experiment': experiment_name, 'status': 'failed', 'error': str(e)}


def ablation_vessel_channel(
    config_path: str,
    output_dir: str,
    num_epochs: int = 50
) -> List[Dict]:
    """
    Ablation Study 1: Vessel Channel Contribution.
    
    Compare:
    - Baseline 3-channel RGB
    - VERA 4-channel [R, G, B, V]
    """
    logger.info("\n" + "="*80)
    logger.info("ABLATION 1: VESSEL CHANNEL CONTRIBUTION")
    logger.info("="*80)
    
    experiments = [
        {
            'name': 'baseline_rgb_3channel',
            'overrides': {
                'no_vessel': True,
                'epochs': num_epochs
            }
        },
        {
            'name': 'vera_4channel_with_vessel',
            'overrides': {
                'no_vessel': False,
                'epochs': num_epochs
            }
        }
    ]
    
    results = []
    for exp in experiments:
        result = run_training_experiment(
            config_path=config_path,
            experiment_name=f"ablation1_{exp['name']}",
            overrides=exp['overrides'],
            output_base_dir=output_dir
        )
        results.append(result)
    
    return results


def ablation_fusion_topology(
    config_path: str,
    output_dir: str,
    num_epochs: int = 50
) -> List[Dict]:
    """
    Ablation Study 2: Fusion Topology Comparison.
    
    Compare:
    - Early Fusion (channel stacking)
    - Dual-Branch (separate encoders)
    - Attention-Gated (spatial attention)
    """
    logger.info("\n" + "="*80)
    logger.info("ABLATION 2: FUSION TOPOLOGY COMPARISON")
    logger.info("="*80)
    
    experiments = [
        {
            'name': 'early_fusion',
            'overrides': {
                'fusion': 'early_fusion',
                'epochs': num_epochs
            }
        },
        {
            'name': 'dual_branch',
            'overrides': {
                'fusion': 'dual_branch',
                'epochs': num_epochs
            }
        },
        {
            'name': 'attention_gated',
            'overrides': {
                'fusion': 'attention_gated',
                'epochs': num_epochs
            }
        }
    ]
    
    results = []
    for exp in experiments:
        result = run_training_experiment(
            config_path=config_path,
            experiment_name=f"ablation2_{exp['name']}",
            overrides=exp['overrides'],
            output_base_dir=output_dir
        )
        results.append(result)
    
    return results


def ablation_backbone_architecture(
    config_path: str,
    output_dir: str,
    num_epochs: int = 50
) -> List[Dict]:
    """
    Ablation Study 3: Backbone Architecture Comparison.
    
    Compare:
    - ResNet-18
    - ResNet-50
    - EfficientNet-B0
    - EfficientNet-B3
    """
    logger.info("\n" + "="*80)
    logger.info("ABLATION 3: BACKBONE ARCHITECTURE COMPARISON")
    logger.info("="*80)
    
    experiments = [
        {
            'name': 'resnet18',
            'overrides': {
                'backbone': 'resnet18',
                'epochs': num_epochs
            }
        },
        {
            'name': 'resnet50',
            'overrides': {
                'backbone': 'resnet50',
                'epochs': num_epochs
            }
        },
        {
            'name': 'efficientnet_b0',
            'overrides': {
                'backbone': 'efficientnet_b0',
                'epochs': num_epochs
            }
        },
        {
            'name': 'efficientnet_b3',
            'overrides': {
                'backbone': 'efficientnet_b3',
                'epochs': num_epochs
            }
        }
    ]
    
    results = []
    for exp in experiments:
        result = run_training_experiment(
            config_path=config_path,
            experiment_name=f"ablation3_{exp['name']}",
            overrides=exp['overrides'],
            output_base_dir=output_dir
        )
        results.append(result)
    
    return results


def ablation_loss_function(
    config_path: str,
    output_dir: str,
    num_epochs: int = 50
) -> List[Dict]:
    """
    Ablation Study 4: Loss Function Comparison.
    
    Compare:
    - Cross-Entropy
    - Focal Loss
    - QWK Loss
    - Hybrid (CE + QWK)
    """
    logger.info("\n" + "="*80)
    logger.info("ABLATION 4: LOSS FUNCTION COMPARISON")
    logger.info("="*80)
    
    experiments = [
        {
            'name': 'cross_entropy',
            'overrides': {
                'loss': 'cross_entropy',
                'epochs': num_epochs
            }
        },
        {
            'name': 'focal_loss',
            'overrides': {
                'loss': 'focal',
                'epochs': num_epochs
            }
        },
        {
            'name': 'qwk_loss',
            'overrides': {
                'loss': 'qwk',
                'epochs': num_epochs
            }
        },
        {
            'name': 'hybrid_ce_qwk',
            'overrides': {
                'loss': 'hybrid',
                'epochs': num_epochs
            }
        }
    ]
    
    results = []
    for exp in experiments:
        result = run_training_experiment(
            config_path=config_path,
            experiment_name=f"ablation4_{exp['name']}",
            overrides=exp['overrides'],
            output_base_dir=output_dir
        )
        results.append(result)
    
    return results


def create_comparison_table(results: List[Dict], ablation_name: str) -> pd.DataFrame:
    """
    Create comparison table from ablation results.
    
    Args:
        results: List of result dictionaries
        ablation_name: Name of ablation study
        
    Returns:
        Pandas DataFrame with comparison
    """
    data = []
    
    for result in results:
        if result.get('status') != 'failed':
            data.append({
                'Experiment': result.get('experiment', 'unknown'),
                'Test Accuracy (%)': result.get('test_accuracy', 0),
                'Test QWK': result.get('test_qwk', 0),
                'Best Val QWK': result.get('best_val_qwk', 0),
                'Epochs': result.get('total_epochs', 0),
                'Parameters (M)': result.get('parameters', {}).get('total', 0) / 1e6
            })
    
    df = pd.DataFrame(data)
    
    # Sort by Test QWK descending
    if len(df) > 0:
        df = df.sort_values('Test QWK', ascending=False)
    
    return df


def generate_ablation_report(
    all_results: Dict[str, List[Dict]],
    output_dir: str
):
    """
    Generate comprehensive ablation study report.
    
    Args:
        all_results: Dictionary mapping ablation names to results
        output_dir: Output directory
    """
    output_dir = Path(output_dir)
    report_dir = output_dir / 'ablation_reports'
    report_dir.mkdir(parents=True, exist_ok=True)
    
    # Create markdown report
    report_path = report_dir / 'ablation_summary.md'
    
    with open(report_path, 'w') as f:
        f.write("# VERA Phase 2 - Ablation Study Results\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("---\n\n")
        
        for ablation_name, results in all_results.items():
            f.write(f"## {ablation_name}\n\n")
            
            # Create comparison table
            df = create_comparison_table(results, ablation_name)
            
            if len(df) > 0:
                f.write(df.to_markdown(index=False))
                f.write("\n\n")
                
                # Save CSV
                csv_path = report_dir / f"{ablation_name.lower().replace(' ', '_')}.csv"
                df.to_csv(csv_path, index=False)
                f.write(f"*Detailed results saved to: {csv_path.name}*\n\n")
                
                # Highlight winner
                best = df.iloc[0]
                f.write(f"**Winner:** {best['Experiment']} "
                       f"(QWK: {best['Test QWK']:.4f}, Accuracy: {best['Test Accuracy (%)']:.2f}%)\n\n")
            else:
                f.write("*No results available*\n\n")
            
            f.write("---\n\n")
    
    logger.info(f"\n{'='*80}")
    logger.info("ABLATION REPORT GENERATED")
    logger.info(f"{'='*80}")
    logger.info(f"Report saved to: {report_path}")


def main():
    """Main ablation study function."""
    parser = argparse.ArgumentParser(
        description='VERA Phase 2 - Run Ablation Studies'
    )
    
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Path to base configuration file')
    parser.add_argument('--output_dir', type=str, default='outputs/ablations',
                       help='Output directory for ablation results')
    parser.add_argument('--ablation', type=str, default='all',
                       choices=['all', 'vessel_channel', 'fusion', 'backbone', 'loss'],
                       help='Which ablation study to run')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of epochs per experiment')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(log_file='outputs/logs/ablation_study.log')
    
    logger.info("="*80)
    logger.info("VERA PHASE 2 - ABLATION STUDY")
    logger.info("="*80)
    logger.info(f"Configuration: {args.config}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Epochs per experiment: {args.epochs}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run ablation studies
    all_results = {}
    
    if args.ablation in ['all', 'vessel_channel']:
        results = ablation_vessel_channel(args.config, str(output_dir), args.epochs)
        all_results['Ablation 1: Vessel Channel Contribution'] = results
    
    if args.ablation in ['all', 'fusion']:
        results = ablation_fusion_topology(args.config, str(output_dir), args.epochs)
        all_results['Ablation 2: Fusion Topology'] = results
    
    if args.ablation in ['all', 'backbone']:
        results = ablation_backbone_architecture(args.config, str(output_dir), args.epochs)
        all_results['Ablation 3: Backbone Architecture'] = results
    
    if args.ablation in ['all', 'loss']:
        results = ablation_loss_function(args.config, str(output_dir), args.epochs)
        all_results['Ablation 4: Loss Function'] = results
    
    # Generate report
    generate_ablation_report(all_results, str(output_dir))
    
    logger.info("\n" + "="*80)
    logger.info("ABLATION STUDY COMPLETE")
    logger.info("="*80)
    logger.info(f"All results saved to: {output_dir}")


if __name__ == '__main__':
    main()

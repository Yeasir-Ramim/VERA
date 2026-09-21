"""
Dataset Setup Script for VERA - Real Diabetic Retinopathy Detection
Downloads and prepares APTOS 2019, EyePACS, and Messidor-2 datasets.

Usage:
    python setup_datasets.py --dataset aptos --data_dir data/
    python setup_datasets.py --dataset all --data_dir data/
"""

import argparse
import os
import sys
from pathlib import Path
import pandas as pd
import requests
from tqdm import tqdm
import zipfile
import shutil


def download_file(url: str, destination: Path, chunk_size: int = 8192):
    """Download file with progress bar."""
    print(f"Downloading from {url}...")
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    destination.parent.mkdir(parents=True, exist_ok=True)
    
    with open(destination, 'wb') as f, tqdm(
        total=total_size,
        unit='B',
        unit_scale=True,
        desc=destination.name
    ) as pbar:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))


def setup_aptos_2019(data_dir: Path):
    """
    Setup APTOS 2019 Blindness Detection Dataset
    Kaggle Competition: https://www.kaggle.com/c/aptos2019-blindness-detection
    
    Manual Download Required:
    1. Visit: https://www.kaggle.com/c/aptos2019-blindness-detection/data
    2. Download train.csv and train_images.zip
    3. Place in data/raw/aptos2019/
    """
    aptos_dir = data_dir / 'raw' / 'aptos2019'
    aptos_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*80)
    print("APTOS 2019 Blindness Detection Dataset Setup")
    print("="*80)
    
    # Check if files exist
    train_csv = aptos_dir / 'train.csv'
    train_images_zip = aptos_dir / 'train_images.zip'
    train_images_dir = aptos_dir / 'train_images'
    
    if not train_csv.exists() or not (train_images_zip.exists() or train_images_dir.exists()):
        print("\n⚠️  APTOS 2019 dataset requires manual download from Kaggle")
        print("\nSteps:")
        print("1. Create a Kaggle account at https://www.kaggle.com")
        print("2. Visit: https://www.kaggle.com/c/aptos2019-blindness-detection/data")
        print("3. Click 'Download All' or download:")
        print("   - train.csv")
        print("   - train_images.zip (3.6 GB)")
        print(f"4. Place files in: {aptos_dir.absolute()}")
        print("\nOr use Kaggle API:")
        print("   kaggle competitions download -c aptos2019-blindness-detection")
        print(f"   mv *.csv *.zip {aptos_dir.absolute()}")
        return False
    
    # Extract images if needed
    if not train_images_dir.exists() and train_images_zip.exists():
        print(f"\nExtracting {train_images_zip.name}...")
        with zipfile.ZipFile(train_images_zip, 'r') as zip_ref:
            zip_ref.extractall(aptos_dir)
        print("✓ Extraction complete")
    
    # Validate dataset
    if train_csv.exists():
        df = pd.read_csv(train_csv)
        print(f"\n✓ APTOS 2019 dataset ready!")
        print(f"  - Total images: {len(df)}")
        print(f"  - Class distribution:")
        print(df['diagnosis'].value_counts().sort_index().to_string())
        
        # Create processed directory structure
        processed_dir = data_dir / 'processed' / 'aptos2019'
        processed_dir.mkdir(parents=True, exist_ok=True)
        
        return True
    
    return False


def setup_eyepacs(data_dir: Path):
    """
    Setup EyePACS Diabetic Retinopathy Dataset
    Kaggle Competition: https://www.kaggle.com/c/diabetic-retinopathy-detection
    
    Manual Download Required:
    1. Visit: https://www.kaggle.com/c/diabetic-retinopathy-detection/data
    2. Download trainLabels.csv and train.zip (35 GB)
    3. Place in data/raw/eyepacs/
    """
    eyepacs_dir = data_dir / 'raw' / 'eyepacs'
    eyepacs_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*80)
    print("EyePACS Diabetic Retinopathy Dataset Setup")
    print("="*80)
    
    train_csv = eyepacs_dir / 'trainLabels.csv'
    train_images_zip = eyepacs_dir / 'train.zip'
    train_images_dir = eyepacs_dir / 'train'
    
    if not train_csv.exists() or not (train_images_zip.exists() or train_images_dir.exists()):
        print("\n⚠️  EyePACS dataset requires manual download from Kaggle")
        print("\nSteps:")
        print("1. Visit: https://www.kaggle.com/c/diabetic-retinopathy-detection/data")
        print("2. Download:")
        print("   - trainLabels.csv")
        print("   - train.zip (~35 GB - contains ~35,000 images)")
        print(f"3. Place files in: {eyepacs_dir.absolute()}")
        print("\nOr use Kaggle API:")
        print("   kaggle competitions download -c diabetic-retinopathy-detection")
        print(f"   mv *.csv *.zip {eyepacs_dir.absolute()}")
        return False
    
    # Extract images if needed (this takes a while)
    if not train_images_dir.exists() and train_images_zip.exists():
        print(f"\n⚠️  Large extraction: {train_images_zip.name} (~35 GB)")
        response = input("Proceed with extraction? (y/n): ")
        if response.lower() == 'y':
            print("Extracting... This may take 10-30 minutes.")
            with zipfile.ZipFile(train_images_zip, 'r') as zip_ref:
                for file in tqdm(zip_ref.namelist(), desc="Extracting"):
                    zip_ref.extract(file, eyepacs_dir)
            print("✓ Extraction complete")
    
    # Validate dataset
    if train_csv.exists():
        df = pd.read_csv(train_csv)
        print(f"\n✓ EyePACS dataset ready!")
        print(f"  - Total images: {len(df)}")
        print(f"  - Class distribution:")
        print(df['level'].value_counts().sort_index().to_string())
        
        processed_dir = data_dir / 'processed' / 'eyepacs'
        processed_dir.mkdir(parents=True, exist_ok=True)
        
        return True
    
    return False


def setup_messidor2(data_dir: Path):
    """
    Setup Messidor-2 Dataset for External Validation
    Source: http://www.adcis.net/en/third-party/messidor2/
    
    Manual Download Required:
    1. Visit: http://www.adcis.net/en/third-party/messidor2/
    2. Fill out application form
    3. Download dataset after approval
    4. Place in data/raw/messidor2/
    """
    messidor_dir = data_dir / 'raw' / 'messidor2'
    messidor_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*80)
    print("Messidor-2 Dataset Setup")
    print("="*80)
    
    annotation_file = messidor_dir / 'messidor2_annotations.csv'
    images_dir = messidor_dir / 'images'
    
    if not annotation_file.exists() or not images_dir.exists():
        print("\n⚠️  Messidor-2 requires application and manual download")
        print("\nSteps:")
        print("1. Visit: http://www.adcis.net/en/third-party/messidor2/")
        print("2. Fill out application form for research access")
        print("3. Wait for approval email with download link")
        print("4. Download dataset (1,748 images)")
        print(f"5. Extract to: {messidor_dir.absolute()}")
        print("6. Ensure structure:")
        print("   messidor2/")
        print("   ├── images/")
        print("   └── messidor2_annotations.csv")
        return False
    
    if annotation_file.exists():
        df = pd.read_csv(annotation_file)
        print(f"\n✓ Messidor-2 dataset ready!")
        print(f"  - Total images: {len(df)}")
        
        processed_dir = data_dir / 'processed' / 'messidor2'
        processed_dir.mkdir(parents=True, exist_ok=True)
        
        return True
    
    return False


def setup_drive_chase(data_dir: Path):
    """
    Setup DRIVE and CHASE_DB1 for vessel segmentation training
    These are small datasets with ground truth vessel annotations.
    """
    vessel_dir = data_dir / 'raw' / 'vessel_segmentation'
    vessel_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*80)
    print("Vessel Segmentation Datasets (DRIVE & CHASE_DB1)")
    print("="*80)
    
    print("\n⚠️  Manual download required:")
    print("\n1. DRIVE Dataset:")
    print("   Visit: https://drive.grand-challenge.org/")
    print("   Download training and test sets")
    print(f"   Extract to: {vessel_dir / 'DRIVE'}")
    
    print("\n2. CHASE_DB1 Dataset:")
    print("   Visit: https://blogs.kingston.ac.uk/retinal/chasedb1/")
    print("   Download dataset")
    print(f"   Extract to: {vessel_dir / 'CHASE_DB1'}")
    
    return False


def create_directory_structure(data_dir: Path):
    """Create complete directory structure for the project."""
    print("\n" + "="*80)
    print("Creating Directory Structure")
    print("="*80)
    
    directories = [
        'raw/aptos2019',
        'raw/eyepacs',
        'raw/messidor2',
        'raw/vessel_segmentation/DRIVE',
        'raw/vessel_segmentation/CHASE_DB1',
        'processed/aptos2019',
        'processed/eyepacs',
        'processed/messidor2',
        'processed/vessel_cache',
        'models/checkpoints',
        'models/pretrained',
        'outputs/logs',
        'outputs/figures',
        'outputs/reports',
        'outputs/vessel_cache',
    ]
    
    for dir_path in directories:
        full_path = data_dir / dir_path
        full_path.mkdir(parents=True, exist_ok=True)
        print(f"✓ Created: {full_path}")
    
    # Create README in data directory
    readme_path = data_dir / 'README.md'
    with open(readme_path, 'w') as f:
        f.write("""# VERA Dataset Directory

This directory contains all datasets used for training and evaluating the VERA system.

## Directory Structure

```
data/
├── raw/                          # Original downloaded datasets
│   ├── aptos2019/               # APTOS 2019 (3.6k images)
│   │   ├── train_images/
│   │   └── train.csv
│   ├── eyepacs/                 # EyePACS (~35k images)
│   │   ├── train/
│   │   └── trainLabels.csv
│   ├── messidor2/               # Messidor-2 (1.7k images)
│   │   ├── images/
│   │   └── messidor2_annotations.csv
│   └── vessel_segmentation/     # Vessel ground truth
│       ├── DRIVE/
│       └── CHASE_DB1/
│
├── processed/                    # Preprocessed datasets
│   ├── aptos2019/               # Cropped, CLAHE-enhanced
│   ├── eyepacs/
│   ├── messidor2/
│   └── vessel_cache/            # Precomputed vessel maps
│
└── models/                       # Model weights
    ├── checkpoints/             # Training checkpoints
    └── pretrained/              # Pretrained weights
```

## Dataset Sources

### 1. APTOS 2019 Blindness Detection
- **Source**: Kaggle Competition
- **URL**: https://www.kaggle.com/c/aptos2019-blindness-detection
- **Size**: 3,662 images
- **Classes**: 5 (0-4, ICDR scale)

### 2. EyePACS Diabetic Retinopathy
- **Source**: Kaggle Competition
- **URL**: https://www.kaggle.com/c/diabetic-retinopathy-detection
- **Size**: ~35,000 images
- **Classes**: 5 (0-4)

### 3. Messidor-2
- **Source**: ADCIS (French Research)
- **URL**: http://www.adcis.net/en/third-party/messidor2/
- **Size**: 1,748 images
- **Classes**: DR severity grades

### 4. DRIVE (Vessel Segmentation)
- **Source**: Grand Challenge
- **URL**: https://drive.grand-challenge.org/
- **Size**: 40 images with vessel annotations

### 5. CHASE_DB1 (Vessel Segmentation)
- **Source**: Kingston University
- **URL**: https://blogs.kingston.ac.uk/retinal/chasedb1/
- **Size**: 28 images with vessel annotations

## Setup Instructions

Run the setup script to prepare datasets:

```bash
python setup_datasets.py --dataset all --data_dir data/
```

Or setup individual datasets:

```bash
python setup_datasets.py --dataset aptos --data_dir data/
python setup_datasets.py --dataset eyepacs --data_dir data/
python setup_datasets.py --dataset messidor2 --data_dir data/
```
""")
    print(f"\n✓ Created README: {readme_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Setup datasets for VERA Diabetic Retinopathy Detection'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        choices=['aptos', 'eyepacs', 'messidor2', 'vessel', 'all'],
        default='all',
        help='Which dataset to setup'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default='data',
        help='Base directory for datasets'
    )
    
    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    
    print("\n" + "="*80)
    print("VERA Dataset Setup Utility")
    print("="*80)
    print(f"Data directory: {data_dir.absolute()}")
    
    # Create directory structure
    create_directory_structure(data_dir)
    
    # Setup requested datasets
    results = {}
    
    if args.dataset in ['aptos', 'all']:
        results['APTOS 2019'] = setup_aptos_2019(data_dir)
    
    if args.dataset in ['eyepacs', 'all']:
        results['EyePACS'] = setup_eyepacs(data_dir)
    
    if args.dataset in ['messidor2', 'all']:
        results['Messidor-2'] = setup_messidor2(data_dir)
    
    if args.dataset in ['vessel', 'all']:
        results['Vessel Segmentation'] = setup_drive_chase(data_dir)
    
    # Print summary
    print("\n" + "="*80)
    print("Setup Summary")
    print("="*80)
    
    for dataset_name, success in results.items():
        status = "✓ Ready" if success else "⚠️  Manual download required"
        print(f"{dataset_name}: {status}")
    
    print("\n" + "="*80)
    print("Next Steps")
    print("="*80)
    print("\n1. Download any missing datasets (see instructions above)")
    print("2. Verify data integrity:")
    print("   python scripts/verify_datasets.py")
    print("3. Preprocess datasets:")
    print("   python scripts/preprocess_datasets.py")
    print("4. Train vessel segmentation model:")
    print("   python scripts/train_vessel_unet.py")
    print("5. Train DR classification model:")
    print("   python train.py --config configs/config.yaml")


if __name__ == '__main__':
    main()

"""
Download pre-trained VERA models for local inference.

This script downloads pre-trained models so you can run VERA locally
without training on your own machine.
"""

import os
import requests
from pathlib import Path
from tqdm import tqdm
import zipfile

def download_file(url: str, destination: str):
    """Download file with progress bar."""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    
    with open(destination, 'wb') as f, tqdm(
        desc=os.path.basename(destination),
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for data in response.iter_content(chunk_size=1024):
            size = f.write(data)
            pbar.update(size)

def download_pretrained_models():
    """Download pre-trained VERA models."""
    
    print("=" * 60)
    print("VERA - Pre-trained Model Downloader")
    print("=" * 60)
    print()
    
    # Model URLs (you'll need to host these after training)
    # For now, these are placeholder URLs
    models = {
        'vessel_segmenter': {
            'url': 'https://huggingface.co/YOUR_USERNAME/vera-vessel-segmenter/resolve/main/vessel_segmenter.pth',
            'path': 'models/checkpoints/vessel_segmenter.pth',
            'size': '~90 MB'
        },
        'dr_classifier': {
            'url': 'https://huggingface.co/YOUR_USERNAME/vera-dr-classifier/resolve/main/best_model.pth',
            'path': 'models/checkpoints/best_model.pth',
            'size': '~200 MB'
        }
    }
    
    print("📦 Available pre-trained models:")
    print()
    for name, info in models.items():
        print(f"  • {name}")
        print(f"    Size: {info['size']}")
        print(f"    Path: {info['path']}")
        print()
    
    print("⚠️  Note: You need to train models first or obtain them from:")
    print("   1. Train on Google Colab using Train_VERA_on_Colab.ipynb")
    print("   2. Download from Hugging Face (after you upload trained models)")
    print("   3. Use pre-trained ImageNet weights (limited performance)")
    print()
    
    choice = input("Download models? (yes/no): ").lower()
    
    if choice != 'yes':
        print("❌ Download cancelled.")
        return
    
    print()
    print("Downloading models...")
    print()
    
    for name, info in models.items():
        try:
            print(f"📥 Downloading {name}...")
            download_file(info['url'], info['path'])
            print(f"✅ {name} downloaded successfully!")
            print()
        except Exception as e:
            print(f"❌ Error downloading {name}: {e}")
            print(f"   Please train the model or download manually.")
            print()
    
    print("=" * 60)
    print("✨ Setup complete!")
    print()
    print("Next steps:")
    print("  1. Verify models exist in models/checkpoints/")
    print("  2. Run: streamlit run web_app/app.py")
    print("  3. Upload retinopathy images and get predictions!")
    print("=" * 60)

def use_pretrained_imagenet_weights():
    """
    Alternative: Use ImageNet pre-trained backbones (no DR-specific training).
    This will work but with reduced accuracy.
    """
    print()
    print("=" * 60)
    print("Alternative: Using ImageNet Pre-trained Weights")
    print("=" * 60)
    print()
    print("⚠️  This option uses ImageNet pre-trained backbones without")
    print("   DR-specific fine-tuning. Predictions will be less accurate.")
    print()
    print("To use this option:")
    print("  1. Set 'pretrained: true' in configs/config.yaml")
    print("  2. Run the web app - models will download automatically")
    print("  3. Expect lower performance (~60-70% accuracy)")
    print()
    print("For best results, train on Google Colab first!")
    print("=" * 60)

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--imagenet':
        use_pretrained_imagenet_weights()
    else:
        download_pretrained_models()

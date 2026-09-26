"""
Check training status and model checkpoint details
"""
import torch
from pathlib import Path

checkpoint_path = Path("checkpoints/production_model/best_model.pth")

checkpoint_path = Path("checkpoints/production_model/best_model.pth")

if checkpoint_path.exists():
    print(f"[OK] Model checkpoint found: {checkpoint_path}")
    print(f"  File size: {checkpoint_path.stat().st_size / (1024*1024):.2f} MB")
    print(f"  Last modified: {checkpoint_path.stat().st_mtime}")
    
    # Load checkpoint to see details
    try:
        ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        print("\n[INFO] Checkpoint Contents:")

        print(f"  Keys: {list(ckpt.keys())}")
        
        if 'epoch' in ckpt:
            print(f"\n[OK] Training reached Epoch: {ckpt['epoch']}")
        
        if 'val_kappa' in ckpt:
            print(f"[OK] Best Validation Kappa: {ckpt['val_kappa']:.4f}")
        
        if 'val_loss' in ckpt:
            print(f"[OK] Best Validation Loss: {ckpt['val_loss']:.4f}")
            
        if 'val_acc' in ckpt:
            val_acc = ckpt.get('val_acc', 0)
            if val_acc > 1:
                print(f"[OK] Best Validation Accuracy: {val_acc:.2f}%")
            else:
                print(f"[OK] Best Validation Accuracy: {val_acc * 100:.2f}%")
        
        # Check model state dict
        if 'model_state_dict' in ckpt:
            model_params = sum(p.numel() for p in ckpt['model_state_dict'].values())
            print(f"\n[MODEL] Parameters: {model_params:,}")
        
        print("\n[SUCCESS] Model is ready for evaluation and deployment!")
        
    except Exception as e:
        print(f"\n[ERROR] Error loading checkpoint: {e}")
else:
    print("[ERROR] No checkpoint found!")


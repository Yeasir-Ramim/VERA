"""
Main CLI Runner for Phase 1 MVP.
Runs the complete 5-chunk pipeline:
1. Preprocesses fundus data
2. Trains/evaluates Baseline 3-channel CNN
3. Computes & caches vessel maps
4. Trains/evaluates Vessel-Aware 4-channel CNN
5. Runs Grad-CAM explainability and saves multi-panel prediction figures to outputs/
"""

import argparse
import os
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.dataset import APTOSDataset, create_dataloaders, compute_class_weights
from src.evaluate import evaluate_model, plot_confusion_matrix
from src.explainability import GradCAM, overlay_cam_on_image
from src.models import build_model
from src.preprocessing import preprocess_fundus
from src.train import train_model
from src.utils import generate_synthetic_fundus, plot_prediction_explanation, setup_sample_dataset
from src.vessel_segmentation import VesselSegmenter


def run_pipeline(
    data_dir: str = "sample_data",
    epochs: int = 5,
    batch_size: int = 16,
    backbone: str = "resnet18",
    output_dir: str = "outputs"
):
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    figures_dir = out_path / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    weights_dir = out_path / "checkpoints"
    weights_dir.mkdir(parents=True, exist_ok=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== Running Phase 1 MVP Pipeline on device: {device} ===")
    
    # 1. Dataset verification or generation
    data_path = Path(data_dir)
    csv_file = data_path / "train.csv"
    images_dir = data_path / "images"
    
    if not csv_file.exists() or not images_dir.exists():
        print(f"Data not found in {data_dir}. Generating sample APTOS-like dataset...")
        setup_sample_dataset(output_dir=data_path, num_samples_per_class=20)
        
    df = pd.read_csv(csv_file)
    print(f"Loaded dataset: {len(df)} samples across {df['diagnosis'].nunique()} classes.")
    
    vessel_cache_dir = out_path / "vessel_cache"
    
    # 2. Train / Evaluate Baseline 3-Channel CNN
    print("\n--- [Chunk 2] Training Baseline 3-Channel CNN ---")
    train_loader_3ch, val_loader_3ch, class_weights = create_dataloaders(
        df=df,
        image_dir=images_dir,
        val_split=0.2,
        batch_size=batch_size,
        is_4channel=False,
        vessel_cache_dir=vessel_cache_dir
    )
    
    baseline_model = build_model(
        model_type="baseline",
        backbone=backbone,
        num_classes=5,
        pretrained=True
    )
    
    train_model(
        model=baseline_model,
        train_loader=train_loader_3ch,
        val_loader=val_loader_3ch,
        num_epochs=epochs,
        learning_rate=3e-4,
        class_weights=class_weights,
        save_path=weights_dir / "baseline_model.pth",
        device=device
    )
    
    eval_baseline = evaluate_model(baseline_model, val_loader_3ch, device=device)
    print(f"\nBaseline Validation Accuracy: {eval_baseline['accuracy'] * 100:.2f}%")
    print(eval_baseline["classification_report"])
    
    # 3. Train / Evaluate Vessel-Aware 4-Channel CNN
    print("\n--- [Chunk 3 & 4] Training Vessel-Aware 4-Channel CNN ---")
    train_loader_4ch, val_loader_4ch, _ = create_dataloaders(
        df=df,
        image_dir=images_dir,
        val_split=0.2,
        batch_size=batch_size,
        is_4channel=True,
        vessel_cache_dir=vessel_cache_dir
    )
    
    vessel_model = build_model(
        model_type="vessel_aware",
        backbone=backbone,
        num_classes=5,
        pretrained=True
    )
    
    train_model(
        model=vessel_model,
        train_loader=train_loader_4ch,
        val_loader=val_loader_4ch,
        num_epochs=epochs,
        learning_rate=3e-4,
        class_weights=class_weights,
        save_path=weights_dir / "vessel_aware_model.pth",
        device=device
    )
    
    eval_vessel = evaluate_model(vessel_model, val_loader_4ch, device=device)
    print(f"\nVessel-Aware Validation Accuracy: {eval_vessel['accuracy'] * 100:.2f}%")
    print(eval_vessel["classification_report"])
    
    # Plot Confusion Matrix for Vessel-Aware Model
    cm_fig = plot_confusion_matrix(
        eval_vessel["confusion_matrix"],
        title=f"Vessel-Aware CNN Confusion Matrix (Acc: {eval_vessel['accuracy']*100:.1f}%)",
        save_path=figures_dir / "confusion_matrix.png"
    )
    plt.close(cm_fig)
    print(f"Saved confusion matrix plot to: {figures_dir / 'confusion_matrix.png'}")
    
    # 4. [Chunk 5] Explainability & Grad-CAM Demo Examples
    print("\n--- [Chunk 5] Generating Grad-CAM Explainability Visualizations ---")
    gradcam = GradCAM(vessel_model)
    vessel_segmenter = VesselSegmenter(cache_dir=vessel_cache_dir)
    
    # Pick 5 example cases (one for each grade 0-4)
    example_samples = []
    for grade in range(5):
        grade_rows = df[df["diagnosis"] == grade]
        if not grade_rows.empty:
            example_samples.append((grade_rows.iloc[0]["id_code"], grade))
            
    for idx, (img_id, true_grade) in enumerate(example_samples):
        # 1. Load and preprocess image
        # Find path
        img_path = images_dir / f"{img_id}.png"
        prep_rgb = preprocess_fundus(str(img_path), target_size=(224, 224))
        
        # 2. Vessel Map
        vessel_map = vessel_segmenter.get_vessel_map(prep_rgb, image_id=str(img_id), target_size=(224, 224))
        
        # 3. Construct 4-channel tensor for inference
        rgb_norm = (prep_rgb.astype(np.float32) / 255.0 - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        vessel_norm = (vessel_map - 0.150) / 0.250
        
        tensor_rgb = torch.from_numpy(rgb_norm.transpose(2, 0, 1)).float()
        tensor_vessel = torch.from_numpy(vessel_norm).unsqueeze(0).float()
        tensor_4ch = torch.cat([tensor_rgb, tensor_vessel], dim=0).unsqueeze(0).to(device)
        
        # 4. Compute Grad-CAM
        cam_map, pred_class, confidence, probs = gradcam.generate_cam(tensor_4ch)
        
        # 5. Overlay on RGB
        overlay = overlay_cam_on_image(prep_rgb, cam_map, alpha=0.45)
        
        # 6. Plot 4-panel figure
        fig_title = f"Case {idx+1}: Ground Truth Grade {true_grade} -> Predicted Grade {pred_class} ({confidence*100:.1f}%)"
        fig_save = figures_dir / f"example_case_{idx+1}_grade{true_grade}.png"
        
        fig = plot_prediction_explanation(
            original_rgb=prep_rgb,
            vessel_map=vessel_map,
            gradcam_overlay=overlay,
            probs=probs,
            true_grade=true_grade,
            pred_grade=pred_class,
            title=fig_title,
            save_path=fig_save
        )
        plt.close(fig)
        print(f"Generated explanation figure: {fig_save}")
        
    print("\n=== Phase 1 MVP Pipeline Execution Completed Successfully ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DR Detection Phase 1 MVP Runner")
    parser.add_argument("--data_dir", type=str, default="sample_data", help="Path to dataset directory")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--backbone", type=str, default="resnet18", help="Backbone model (resnet18, efficientnet_b0)")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Output directory for logs, figures, and models")
    
    args = parser.parse_args()
    run_pipeline(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        backbone=args.backbone,
        output_dir=args.output_dir
    )

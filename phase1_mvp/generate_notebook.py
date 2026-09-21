"""
Helper script to generate notebooks/mvp_demo.ipynb
"""

import json
from pathlib import Path

def create_demo_notebook():
    nb = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# Explainable CNN-Based Diabetic Retinopathy Detection with Vessel-Aware Preprocessing\n",
                    "### Phase 1 MVP Deliverable\n",
                    "\n",
                    "**Project Summary:**\n",
                    "This pipeline implements an end-to-end Explainable Deep Learning system for 5-class Diabetic Retinopathy (DR) grading on the International Clinical Diabetic Retinopathy (ICDR) scale:\n",
                    "- **Grade 0:** No DR\n",
                    "- **Grade 1:** Mild NPDR\n",
                    "- **Grade 2:** Moderate NPDR\n",
                    "- **Grade 3:** Severe NPDR\n",
                    "- **Grade 4:** Proliferative DR (PDR)\n",
                    "\n",
                    "**Key Innovations:**\n",
                    "1. **Vessel-Aware Preprocessing:** Pretrained / multi-scale vascular feature extraction providing a dedicated vessel probability map.\n",
                    "2. **4-Channel CNN Fusion:** ImageNet-pretrained backbone adapted to accept 4 channels ($[R, G, B, V]$) via channel-stacking.\n",
                    "3. **Explainability Layer:** Gradient-weighted Class Activation Mapping (Grad-CAM) to visualize retinal regions influencing diagnostic decisions."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import os\n",
                    "import sys\n",
                    "from pathlib import Path\n",
                    "import cv2\n",
                    "import matplotlib.pyplot as plt\n",
                    "import numpy as np\n",
                    "import pandas as pd\n",
                    "import torch\n",
                    "import torch.nn as nn\n",
                    "\n",
                    "# Add project root to sys.path\n",
                    "sys.path.insert(0, str(Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()))\n",
                    "\n",
                    "from src.preprocessing import crop_fundus_circle, apply_clahe, preprocess_fundus\n",
                    "from src.vessel_segmentation import VesselSegmenter, extract_vessel_map_multiscale\n",
                    "from src.dataset import APTOSDataset, create_dataloaders\n",
                    "from src.models import build_model\n",
                    "from src.train import train_model\n",
                    "from src.evaluate import evaluate_model, plot_confusion_matrix\n",
                    "from src.explainability import GradCAM, overlay_cam_on_image\n",
                    "from src.utils import generate_synthetic_fundus, setup_sample_dataset, plot_prediction_explanation\n",
                    "\n",
                    "# Configure environment\n",
                    "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n",
                    "print(f'Using compute device: {device}')\n",
                    "if torch.cuda.is_available():\n",
                    "    print(f'GPU: {torch.cuda.get_device_name(0)}')\n",
                    "plt.rcParams['figure.dpi'] = 120\n"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 1. Data Pipeline: Dataset Setup & Preprocessing (Chunk 1)\n",
                    "Raw fundus photography typically exhibits dark unilluminated margins and uneven exposure across retinal zones.\n",
                    "Here we load the dataset and apply **circular crop** (black border removal) and **CLAHE** (Contrast Limited Adaptive Histogram Equalization) on LAB color space to enhance microaneurysms, hemorrhages, and vascular branches."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Setup sample dataset\n",
                    "data_dir, df = setup_sample_dataset(output_dir='sample_data', num_samples_per_class=20)\n",
                    "images_dir = data_dir / 'images'\n",
                    "\n",
                    "print(f'Dataset summary: {len(df)} images')\n",
                    "display(df.head())\n",
                    "\n",
                    "# Visual comparison: Raw vs Cropped vs CLAHE-enhanced\n",
                    "sample_raw = cv2.imread(str(images_dir / f\"{df.iloc[0]['id_code']}.png\"))\n",
                    "sample_raw = cv2.cvtColor(sample_raw, cv2.COLOR_BGR2RGB)\n",
                    "\n",
                    "sample_cropped = crop_fundus_circle(sample_raw)\n",
                    "sample_clahe = apply_clahe(sample_cropped)\n",
                    "sample_preprocessed = preprocess_fundus(sample_raw, target_size=(224, 224))\n",
                    "\n",
                    "fig, axs = plt.subplots(1, 4, figsize=(16, 4))\n",
                    "axs[0].imshow(sample_raw)\n",
                    "axs[0].set_title('1. Raw Fundus Photo')\n",
                    "axs[0].axis('off')\n",
                    "\n",
                    "axs[1].imshow(sample_cropped)\n",
                    "axs[1].set_title('2. Circular Crop')\n",
                    "axs[1].axis('off')\n",
                    "\n",
                    "axs[2].imshow(sample_clahe)\n",
                    "axs[2].set_title('3. CLAHE Enhanced')\n",
                    "axs[2].axis('off')\n",
                    "\n",
                    "axs[3].imshow(sample_preprocessed)\n",
                    "axs[3].set_title('4. Standard Resized (224x224)')\n",
                    "axs[3].axis('off')\n",
                    "\n",
                    "plt.tight_layout()\n",
                    "plt.show()\n"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### Data Pipeline Insights:\n",
                    "- **Circular Crop:** Eliminates non-informative black borders, optimizing the active pixel density for the convolutional receptive field.\n",
                    "- **Adaptive Equalization (CLAHE):** Enhances micro-vascular boundaries and localized hemorrhage edges without blowing out optic disc saturation."
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 2. Vessel Segmentation & Feature Caching (Chunk 3)\n",
                    "Extracting vascular probability maps ($V \\in [0, 1]$) to supply explicit morphological priors directly to the classifier."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "vessel_cache_dir = Path('outputs/vessel_cache')\n",
                    "segmenter = VesselSegmenter(cache_dir=vessel_cache_dir)\n",
                    "\n",
                    "# Extract vessel map for sample fundus\n",
                    "vessel_map = segmenter.get_vessel_map(sample_preprocessed, image_id='demo_sample', target_size=(224, 224))\n",
                    "\n",
                    "fig, axs = plt.subplots(1, 3, figsize=(14, 4.5))\n",
                    "axs[0].imshow(sample_preprocessed)\n",
                    "axs[0].set_title('Preprocessed RGB')\n",
                    "axs[0].axis('off')\n",
                    "\n",
                    "axs[1].imshow(sample_preprocessed[:, :, 1], cmap='gray')\n",
                    "axs[1].set_title('Green Channel (Max Hemoglobin Contrast)')\n",
                    "axs[1].axis('off')\n",
                    "\n",
                    "im = axs[2].imshow(vessel_map, cmap='magma', vmin=0, vmax=1)\n",
                    "axs[2].set_title('Extracted Vessel Probability Map [0, 1]')\n",
                    "axs[2].axis('off')\n",
                    "plt.colorbar(im, ax=axs[2], fraction=0.046, pad=0.04)\n",
                    "\n",
                    "plt.tight_layout()\n",
                    "plt.show()\n"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 3. 4-Channel Model Architecture (Chunk 4)\n",
                    "We adapt the first convolutional layer (`conv1`) of an ImageNet-pretrained ResNet-18 backbone:\n",
                    "- Original weights $W_{RGB} \\in \\mathbb{R}^{64 \\times 3 \\times 7 \\times 7}$ are preserved for channels 0, 1, and 2.\n",
                    "- Channel 3 (vessel map) is initialized with the channel-wise mean of $W_{RGB}$."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "model_4ch = build_model(model_type='vessel_aware', backbone='resnet18', num_classes=5, pretrained=True)\n",
                    "\n",
                    "print('Vessel-Aware ResNet-18 Conv1 Configuration:')\n",
                    "print(model_4ch.backbone.conv1)\n",
                    "print(f'Conv1 weight tensor shape: {model_4ch.backbone.conv1.weight.shape}')\n",
                    "print('Classification Head:')\n",
                    "print(model_4ch.backbone.fc)\n"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 4. Model Training & Evaluation (Chunks 2 & 4)\n",
                    "Training both the Baseline (3ch) and Vessel-Aware (4ch) CNNs with class-weighted cross-entropy loss."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Create DataLoaders\n",
                    "train_loader_3ch, val_loader_3ch, class_weights = create_dataloaders(\n",
                    "    df=df, image_dir=images_dir, val_split=0.2, batch_size=16, is_4channel=False, vessel_cache_dir=vessel_cache_dir\n",
                    ")\n",
                    "train_loader_4ch, val_loader_4ch, _ = create_dataloaders(\n",
                    "    df=df, image_dir=images_dir, val_split=0.2, batch_size=16, is_4channel=True, vessel_cache_dir=vessel_cache_dir\n",
                    ")\n",
                    "\n",
                    "# Train Baseline Model\n",
                    "print('=== Training Baseline 3-Channel CNN ===')\n",
                    "baseline_model = build_model(model_type='baseline', backbone='resnet18', num_classes=5, pretrained=True)\n",
                    "hist_baseline = train_model(\n",
                    "    model=baseline_model,\n",
                    "    train_loader=train_loader_3ch,\n",
                    "    val_loader=val_loader_3ch,\n",
                    "    num_epochs=5,\n",
                    "    learning_rate=3e-4,\n",
                    "    class_weights=class_weights,\n",
                    "    device=str(device)\n",
                    ")\n",
                    "\n",
                    "# Train Vessel-Aware Model\n",
                    "print('\\n=== Training Vessel-Aware 4-Channel CNN ===')\n",
                    "vessel_model = build_model(model_type='vessel_aware', backbone='resnet18', num_classes=5, pretrained=True)\n",
                    "hist_vessel = train_model(\n",
                    "    model=vessel_model,\n",
                    "    train_loader=train_loader_4ch,\n",
                    "    val_loader=val_loader_4ch,\n",
                    "    num_epochs=5,\n",
                    "    learning_rate=3e-4,\n",
                    "    class_weights=class_weights,\n",
                    "    device=str(device)\n",
                    ")\n"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Evaluate and Compare\n",
                    "eval_res = evaluate_model(vessel_model, val_loader_4ch, device=str(device))\n",
                    "print(f'Vessel-Aware Model Accuracy: {eval_res[\"accuracy\"]*100:.2f}%')\n",
                    "print('\\nClassification Report:')\n",
                    "print(eval_res['classification_report'])\n",
                    "\n",
                    "# Plot Confusion Matrix\n",
                    "plot_confusion_matrix(eval_res['confusion_matrix'], title='Vessel-Aware CNN Confusion Matrix')\n",
                    "plt.show()\n"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 5. Explainability & Grad-CAM Demo Visualizations (Chunk 5)\n",
                    "For each ICDR severity grade (0 to 4), we generate the 4-panel visual explanation deliverable:\n",
                    "$$\\text{Preprocessed Fundus} \\longrightarrow \\text{Vessel Probability Map} \\longrightarrow \\text{Grad-CAM Saliency Overlay} \\longrightarrow \\text{Class Probabilities Bar Chart}$$"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "gradcam = GradCAM(vessel_model)\n",
                    "\n",
                    "for grade in range(5):\n",
                    "    # Find a representative image of this grade\n",
                    "    row = df[df['diagnosis'] == grade].iloc[0]\n",
                    "    img_id = row['id_code']\n",
                    "    img_path = images_dir / f'{img_id}.png'\n",
                    "    \n",
                    "    # Preprocess RGB & Extract Vessel Map\n",
                    "    rgb_prep = preprocess_fundus(str(img_path), target_size=(224, 224))\n",
                    "    v_map = segmenter.get_vessel_map(rgb_prep, image_id=img_id, target_size=(224, 224))\n",
                    "    \n",
                    "    # Construct 4-channel tensor\n",
                    "    rgb_norm = (rgb_prep.astype(np.float32) / 255.0 - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])\n",
                    "    v_norm = (v_map - 0.150) / 0.250\n",
                    "    \n",
                    "    t_rgb = torch.from_numpy(rgb_norm.transpose(2, 0, 1)).float()\n",
                    "    t_v = torch.from_numpy(v_norm).unsqueeze(0).float()\n",
                    "    tensor_4ch = torch.cat([t_rgb, t_v], dim=0).unsqueeze(0).to(device)\n",
                    "    \n",
                    "    # Compute Grad-CAM\n",
                    "    cam_map, pred_class, confidence, probs = gradcam.generate_cam(tensor_4ch)\n",
                    "    overlay = overlay_cam_on_image(rgb_prep, cam_map, alpha=0.45)\n",
                    "    \n",
                    "    # Render 4-panel visual explanation\n",
                    "    fig = plot_prediction_explanation(\n",
                    "        original_rgb=rgb_prep,\n",
                    "        vessel_map=v_map,\n",
                    "        gradcam_overlay=overlay,\n",
                    "        probs=probs,\n",
                    "        true_grade=grade,\n",
                    "        pred_grade=pred_class,\n",
                    "        title=f'ICDR Grade {grade} Case Explanation'\n",
                    "    )\n",
                    "    plt.show()\n"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 6. Summary & Phase 1 MVP Conclusions\n",
                    "\n",
                    "- **Completed Deliverables:**\n",
                    "  1. Automated Data Pipeline with Circular Crop and LAB CLAHE.\n",
                    "  2. Retinal Vasculature Segmentation & Probability Map Caching.\n",
                    "  3. 4-Channel Vessel-Aware Conv1 Weight Adaptation with Pretrained Weight Transfer.\n",
                    "  4. Training & Validation Pipeline with Class Imbalance Weighting.\n",
                    "  5. Explainable Grad-CAM Visualizations with multi-panel diagnostic summaries.\n",
                    "\n",
                    "- **Next Steps (Phase 2):**\n",
                    "  - Scale pretraining on EyePACS and external validation on Messidor-2.\n",
                    "  - Incorporate Quadratic Weighted Kappa (QWK) and Ordinal/CORAL loss functions.\n",
                    "  - Dual-branch / Attention-gated fusion ablation studies.\n",
                    "  - Interactive Gradio / Streamlit clinician demo dashboard."
                ]
            }
        ],
        "metadata": {
            "language_info": {
                "name": "python",
                "version": "3.9"
            },
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }
    
    out_dir = Path("notebooks")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "mvp_demo.ipynb", "w") as f:
        json.dump(nb, f, indent=2)
    print("Created notebooks/mvp_demo.ipynb successfully.")

if __name__ == "__main__":
    create_demo_notebook()

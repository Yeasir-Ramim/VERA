"""
VERA: Vascular Explainable Retinopathy Assessment
Interactive Clinical Decision Support Web Application (Streamlit).

Features:
- Live fundus image upload or sample preset inspection across all 5 ICDR grades.
- Visual inspection of Circular Crop, CLAHE, and Ben Graham local color subtraction.
- Real-time retinal vascular segmentation probability map visualization.
- Multi-model inference supporting ResNet-18/50 and EfficientNet backbones with
  Early Fusion, Dual-Branch, and Spatial Attention Gating.
- Quantitative explainability with Grad-CAM and Grad-CAM++ heatmap overlays.
- Quantitative Vessel-Attention Overlap Score and Referable DR clinical risk triage.
"""

from pathlib import Path
from typing import Optional, Tuple
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import torch

from src.dataset import IMAGENET_MEAN, IMAGENET_STD, VESSEL_MEAN, VESSEL_STD
from src.evaluate import ICDR_CLASS_NAMES
from src.explainability import (
    GradCAM,
    GradCAMPlusPlus,
    compute_vessel_attention_overlap,
    overlay_cam_on_image,
)
from src.models import build_model
from src.preprocessing import apply_ben_graham, apply_clahe, crop_fundus_circle, preprocess_fundus
from src.vessel_segmentation import VesselSegmenter


st.set_page_config(
    page_title="VERA | Explainable Retinopathy Assessment",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F3F4F6;
        padding: 1.2rem;
        border-radius: 0.5rem;
        border-left: 5px solid #2563EB;
        margin-bottom: 1rem;
    }
    .referable-bad {
        background-color: #FEE2E2;
        border-left: 6px solid #DC2626;
        padding: 1rem;
        border-radius: 0.5rem;
        color: #991B1B;
        font-weight: 600;
    }
    .referable-good {
        background-color: #ECFDF5;
        border-left: 6px solid #059669;
        padding: 1rem;
        border-radius: 0.5rem;
        color: #065F46;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_vessel_segmenter():
    return VesselSegmenter(cache_dir="outputs/vessel_cache")


@st.cache_resource
def load_classification_model(model_type: str, backbone: str, checkpoint_path: Optional[str] = None):
    model = build_model(model_type=model_type, backbone=backbone, pretrained=True)
    if checkpoint_path and Path(checkpoint_path).exists():
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        state_dict = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


def prepare_input_tensor(img_rgb: np.ndarray, vessel_map: np.ndarray) -> torch.Tensor:
    """Standardizes RGB and Vessel map into a normalized 4-channel tensor."""
    h, w = img_rgb.shape[:2]
    if vessel_map.shape[:2] != (h, w):
        vessel_map = cv2.resize(vessel_map, (w, h))
        
    rgb_norm = img_rgb.astype(np.float32) / 255.0
    for c in range(3):
        rgb_norm[:, :, c] = (rgb_norm[:, :, c] - IMAGENET_MEAN[c]) / IMAGENET_STD[c]
    rgb_t = torch.from_numpy(rgb_norm.transpose(2, 0, 1)).float()
    
    vessel_norm = (vessel_map.astype(np.float32) - VESSEL_MEAN[0]) / VESSEL_STD[0]
    vessel_t = torch.from_numpy(vessel_norm).unsqueeze(0).float()
    
    tensor_4ch = torch.cat([rgb_t, vessel_t], dim=0).unsqueeze(0)
    return tensor_4ch


# Sidebar configuration
st.sidebar.image("https://img.icons8.com/fluency/96/ophthalmology.png", width=72)
st.sidebar.title("VERA Settings")

backbone_choice = st.sidebar.selectbox(
    "Backbone Architecture",
    ["resnet18", "resnet50", "efficientnet_b0", "efficientnet_b3"],
    index=0
)

fusion_choice = st.sidebar.selectbox(
    "Fusion Topology (Phase 2)",
    ["early_fusion", "dual_branch", "attention_gated", "baseline"],
    index=0,
    help="Select how retinal vascular structural maps are fused with the RGB fundus image."
)

enhancement_choice = st.sidebar.selectbox(
    "Preprocessing Enhancement",
    ["clahe", "ben_graham", "combined", "none"],
    index=0,
    help="Select clinical illumination correction algorithm."
)

cam_choice = st.sidebar.radio(
    "Explainability Algorithm",
    ["Grad-CAM++ (Recommended)", "Grad-CAM (Vanilla)"],
    index=0
)

cam_alpha = st.sidebar.slider("Heatmap Overlay Opacity", min_value=0.1, max_value=0.9, value=0.5, step=0.05)

# Main Title
st.markdown('<div class="main-title">VERA: Vascular Explainable Retinopathy Assessment</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Explainable multi-channel deep learning system for 5-class Diabetic Retinopathy screening with vessel-aware feature fusion.</div>', unsafe_allow_html=True)

# Sample Images or File Upload
col_mode1, col_mode2 = st.columns([1, 1])
with col_mode1:
    data_source = st.radio("Select Image Source:", ["Use Presets (Grades 0 to 4)", "Upload Fundus Image"], horizontal=True)

sample_img_path: Optional[str] = None
uploaded_pil: Optional[Image.Image] = None

if data_source == "Use Presets (Grades 0 to 4)":
    sample_dir = Path("sample_data/images")
    if sample_dir.exists():
        preset_files = sorted(list(sample_dir.glob("fundus_*.png")))
        if preset_files:
            preset_labels = [f"Grade {f.stem.split('_')[1]} — {f.name}" for f in preset_files[:5]]
            selected_preset = st.selectbox("Select Benchmark Case:", preset_labels, index=2)
            idx = preset_labels.index(selected_preset)
            sample_img_path = str(preset_files[idx])
        else:
            st.info("No sample preset images found in `sample_data/images`. Please upload an image.")
    else:
        st.info("Directory `sample_data/images` not found. Please upload a fundus photograph.")
else:
    uploaded_file = st.file_uploader("Upload Retinal Fundus Photograph (JPG/PNG)", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        uploaded_pil = Image.open(uploaded_file)

# Execute Pipeline
active_image = sample_img_path or uploaded_pil

if active_image is not None:
    # 1. Preprocessing
    if isinstance(active_image, str):
        raw_bgr = cv2.imread(active_image)
        raw_rgb = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)
    else:
        raw_rgb = np.array(active_image.convert("RGB"))
        
    preprocessed_rgb = preprocess_fundus(
        raw_rgb,
        target_size=(224, 224),
        apply_crop=True,
        apply_enhancement=True,
        enhancement_method=enhancement_choice
    )
    
    # 2. Vascular Segmentation
    segmenter = get_vessel_segmenter()
    vessel_prob_map = segmenter.predict(preprocessed_rgb)
    
    # 3. Model Inference
    # Check for available checkpoint
    ckpt_candidates = [
        Path("outputs/checkpoints/vessel_aware_model.pth"),
        Path(f"outputs/checkpoints/ablation1_vera_4ch.pth"),
        Path(f"outputs/checkpoints/ablation2_{fusion_choice}.pth")
    ]
    ckpt_to_use = None
    for c in ckpt_candidates:
        if c.exists():
            ckpt_to_use = str(c)
            break
            
    model = load_classification_model(fusion_choice, backbone_choice, ckpt_to_use)
    input_tensor = prepare_input_tensor(preprocessed_rgb, vessel_prob_map)
    
    # 4. Explainability & Overlap
    cam_engine = GradCAMPlusPlus(model) if "++" in cam_choice else GradCAM(model)
    cam_map, pred_class, confidence, all_probs = cam_engine.generate_cam(input_tensor)
    cam_overlay = overlay_cam_on_image(preprocessed_rgb, cam_map, alpha=cam_alpha)
    overlap_metrics = compute_vessel_attention_overlap(cam_map, vessel_prob_map)
    
    # Visual Inspection Panels
    st.write("### 1. Diagnostic Vision Pipeline")
    p1, p2, p3, p4 = st.columns(4)
    with p1:
        st.caption("**Original Fundus Image**")
        st.image(raw_rgb, use_container_width=True)
    with p2:
        st.caption(f"**Preprocessed ({enhancement_choice.upper()})**")
        st.image(preprocessed_rgb, use_container_width=True)
    with p3:
        st.caption("**Retinal Vascular Tree (U-Net)**")
        st.image((vessel_prob_map * 255).astype(np.uint8), clamp=True, use_container_width=True)
    with p4:
        st.caption(f"**{cam_choice} Overlay**")
        st.image(cam_overlay, use_container_width=True)
        
    st.write("---")
    
    # Clinical Prediction & Triage Panel
    st.write("### 2. Clinical Diagnostic Assessment")
    c1, c2 = st.columns([1, 1])
    
    with c1:
        st.subheader("Automated Grade Prediction")
        st.markdown(f"### **{ICDR_CLASS_NAMES[pred_class]}**")
        st.metric("Model Confidence", f"{confidence * 100:.1f}%")
        
        # Referable DR Triage Warning
        if pred_class >= 2:
            st.markdown("""
            <div class="referable-bad">
                ⚠️ <b>REFERABLE DIABETIC RETINOPATHY DETECTED</b><br>
                Grade &ge; 2 indicates presence of clinical hemorrhages, exudates, or neovascularization. 
                Immediate specialist ophthalmology referral recommended.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="referable-good">
                ✅ <b>NON-REFERABLE RETINOPATHY</b><br>
                Grade &le; 1 indicates no or mild microaneurysms only. 
                Routine annual diabetic eye screening schedule advised.
            </div>
            """, unsafe_allow_html=True)
            
        st.write("")
        # Overlap Metric
        st.subheader("Explainability Overlap Validation")
        overlap_score = overlap_metrics["overlap_score"]
        col_m1, col_m2 = st.columns(2)
        col_m1.metric("Vessel-Attention Overlap", f"{overlap_score:.3f}")
        col_m2.metric("Vascular Focus Ratio", f"{overlap_metrics['high_attention_vascular_ratio'] * 100:.1f}%")
        
        if overlap_score >= 0.35:
            st.success("High Clinical Alignment: Network attention is firmly grounded in vascular pathology.")
        else:
            st.warning("Moderate Alignment: Attention spans both vascular features and retinal background tissue.")
            
    with c2:
        st.subheader("Severity Probability Distribution")
        prob_df = pd.DataFrame({
            "ICDR Severity Grade": ICDR_CLASS_NAMES,
            "Probability": all_probs
        })
        st.bar_chart(prob_df.set_index("ICDR Severity Grade"), color="#2563EB")
        
        with st.expander("Technical Model Details"):
            st.write(f"- **Backbone:** `{backbone_choice}`")
            st.write(f"- **Fusion Topology:** `{fusion_choice}`")
            st.write(f"- **Input Dimensions:** `(4, 224, 224)`")
            st.write(f"- **Explainability Engine:** `{cam_choice}`")
            st.write(f"- **Active Checkpoint:** `{ckpt_to_use or 'Pretrained ImageNet Weights'}`")

else:
    st.info("Please select a sample case or upload an image to begin assessment.")

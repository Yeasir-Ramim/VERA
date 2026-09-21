"""
VERA Phase 2 - Simplified Demo Application

A lightweight demo that works with or without trained models.
If trained models are not available, uses ImageNet pre-trained weights.
"""

import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

from src.models.classification import DRClassifier
from src.models.vessel_segmentation import UNetVesselSegmenter
from src.data.preprocessing import FundusPreprocessor
from src.utils.config import load_config
import matplotlib.pyplot as plt


# Page configuration
st.set_page_config(
    page_title="VERA - DR Detection Demo",
    page_icon="🩺",
    layout="wide"
)

st.markdown("""
<style>
    .main-header {font-size: 2.5rem; color: #2c3e50; text-align: center; margin-bottom: 1rem;}
    .sub-header {font-size: 1.3rem; color: #34495e; margin-top: 1rem;}
    .grade-0 {color: #27ae60; font-weight: bold;}
    .grade-1 {color: #3498db; font-weight: bold;}
    .grade-2 {color: #f39c12; font-weight: bold;}
    .grade-3 {color: #e67e22; font-weight: bold;}
    .grade-4 {color: #c0392b; font-weight: bold;}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_models():
    """Load models (trained or ImageNet pre-trained)."""
    try:
        config_path = Path(__file__).parent / 'configs' / 'config.yaml'
        config = load_config(str(config_path))
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Check for trained models
        vessel_checkpoint_path = Path(__file__).parent / 'models' / 'checkpoints' / 'vessel_segmenter.pth'
        dr_checkpoint_path = Path(__file__).parent / 'models' / 'checkpoints' / 'best_model.pth'
        
        trained_models_available = vessel_checkpoint_path.exists() and dr_checkpoint_path.exists()
        
        if trained_models_available:
            st.sidebar.success("✅ Using trained models")
            
            # Load vessel segmenter
            vessel_segmenter = UNetVesselSegmenter(
                encoder_name='resnet34',
                pretrained=False
            ).to(device)
            vessel_checkpoint = torch.load(vessel_checkpoint_path, map_location=device)
            vessel_segmenter.load_state_dict(vessel_checkpoint['model_state_dict'])
            vessel_segmenter.eval()
            
            # Load DR classifier
            dr_model = DRClassifier(
                backbone_name=config['model']['backbone'],
                fusion_strategy=config['model']['fusion_strategy'],
                num_classes=5,
                pretrained=False
            ).to(device)
            dr_checkpoint = torch.load(dr_checkpoint_path, map_location=device)
            dr_model.load_state_dict(dr_checkpoint['model_state_dict'])
            dr_model.eval()
            
            return dr_model, vessel_segmenter, config, device, True
            
        else:
            st.sidebar.warning("⚠️ Using ImageNet pre-trained weights (limited accuracy)")
            st.sidebar.info("""
            For best results, train models first:
            1. Use Train_VERA_on_Colab.ipynb
            2. Download trained weights
            3. Place in models/checkpoints/
            """)
            
            # Load with ImageNet pre-trained weights
            vessel_segmenter = UNetVesselSegmenter(
                encoder_name='resnet34',
                pretrained=True
            ).to(device)
            vessel_segmenter.eval()
            
            dr_model = DRClassifier(
                backbone_name='resnet34',  # Lighter model for demo
                fusion_strategy='early_fusion',
                num_classes=5,
                pretrained=True
            ).to(device)
            dr_model.eval()
            
            return dr_model, vessel_segmenter, config, device, False
            
    except Exception as e:
        st.error(f"Error loading models: {e}")
        import traceback
        st.error(traceback.format_exc())
        return None, None, None, None, False


def preprocess_image(image, config):
    """Preprocess uploaded image."""
    image_np = np.array(image)
    
    if len(image_np.shape) == 3:
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    else:
        image_bgr = image_np
    
    preprocessor = FundusPreprocessor(
        target_size=(512, 512),
        circular_crop=True,
        ben_graham_enabled=True,
        clahe_enabled=True
    )
    
    processed = preprocessor.preprocess(image_bgr)
    return processed, image_bgr


def predict(dr_model, vessel_segmenter, image_tensor, device):
    """Run inference."""
    dr_model.eval()
    vessel_segmenter.eval()
    
    with torch.no_grad():
        image_batch = image_tensor.unsqueeze(0).to(device)
        
        # Segment vessels
        vessel_output = vessel_segmenter(image_batch)
        vessel_map = (torch.sigmoid(vessel_output) > 0.5).float()
        
        # Predict DR grade
        outputs = dr_model(image_batch, vessel_map)
        probabilities = torch.softmax(outputs, dim=1)
        prediction = outputs.argmax(dim=1).item()
        confidence = probabilities[0, prediction].item()
    
    return prediction, probabilities[0].cpu().numpy(), confidence, vessel_map.squeeze(0).cpu()


def main():
    # Header
    st.markdown('<h1 class="main-header">🩺 VERA - Diabetic Retinopathy Detection</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #7f8c8d; font-size: 1.1rem;">Vascular Explainable Retinopathy Assessment System - Demo</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("## 📊 About VERA")
        st.info("""
        VERA uses AI to detect diabetic retinopathy from fundus images.
        
        **Features:**
        - Vessel-aware neural networks
        - 5-level severity grading
        - Real-time analysis
        """)
        
        st.markdown("## 🎯 DR Severity Grades")
        st.markdown("""
        - <span class="grade-0">Grade 0</span>: No DR
        - <span class="grade-1">Grade 1</span>: Mild NPDR
        - <span class="grade-2">Grade 2</span>: Moderate NPDR  
        - <span class="grade-3">Grade 3</span>: Severe NPDR
        - <span class="grade-4">Grade 4</span>: Proliferative DR
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("### 📥 Sample Images")
        st.markdown("""
        Download test images:
        - [APTOS Dataset](https://www.kaggle.com/c/aptos2019-blindness-detection/data)
        - [EyePACS](https://www.kaggle.com/c/diabetic-retinopathy-detection/data)
        """)
    
    # Load models
    dr_model, vessel_segmenter, config, device, trained = load_models()
    
    if dr_model is None:
        st.error("Failed to load models. Please check installation.")
        return
    
    st.success(f"✅ Models loaded on {device.upper()}")
    
    if not trained:
        st.warning("""
        ⚠️ **Demo Mode**: Using ImageNet pre-trained weights without DR-specific training.
        Predictions may be less accurate. For production use, train models on Google Colab first.
        """)
    
    # Main content
    st.markdown('<h2 class="sub-header">📤 Upload Fundus Image</h2>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Choose a retinal fundus image...",
        type=['png', 'jpg', 'jpeg'],
        help="Upload a color fundus photograph"
    )
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Original Image")
            st.image(image, use_column_width=True)
        
        if st.button("🔍 Analyze Image", type="primary"):
            with st.spinner("Processing..."):
                # Preprocess
                processed, image_bgr = preprocess_image(image, config)
                image_tensor = torch.from_numpy(processed).permute(2, 0, 1).float()
                
                # Predict
                prediction, probabilities, confidence, vessel_map = predict(
                    dr_model, vessel_segmenter, image_tensor, device
                )
            
            # Display vessel segmentation
            with col2:
                st.markdown("### Vessel Segmentation")
                vessel_vis = vessel_map.squeeze().numpy() * 255
                st.image(vessel_vis.astype(np.uint8), use_column_width=True, caption="Detected Blood Vessels")
            
            # Results
            st.markdown('<h2 class="sub-header">📊 Diagnosis Results</h2>', unsafe_allow_html=True)
            
            grade_names = ['No DR', 'Mild NPDR', 'Moderate NPDR', 'Severe NPDR', 'Proliferative DR']
            grade_colors = ['#27ae60', '#3498db', '#f39c12', '#e67e22', '#c0392b']
            
            result_col1, result_col2, result_col3 = st.columns([2, 1, 1])
            
            with result_col1:
                st.markdown(f'<h1 style="color: {grade_colors[prediction]}; font-size: 3rem;">Grade {prediction}</h1>', 
                          unsafe_allow_html=True)
                st.markdown(f'<h3>{grade_names[prediction]}</h3>', unsafe_allow_html=True)
            
            with result_col2:
                st.metric("Confidence", f"{confidence*100:.1f}%")
            
            with result_col3:
                referable = "Yes" if prediction >= 2 else "No"
                st.metric("Referable", referable)
            
            # Probability chart
            st.markdown("### Class Probabilities")
            fig, ax = plt.subplots(figsize=(10, 4))
            bars = ax.bar(grade_names, probabilities, color=grade_colors, alpha=0.7, edgecolor='black')
            bars[prediction].set_alpha(1.0)
            bars[prediction].set_edgecolor('red')
            bars[prediction].set_linewidth(3)
            
            ax.set_ylabel('Probability', fontweight='bold')
            ax.set_ylim([0, 1])
            ax.grid(axis='y', alpha=0.3)
            
            for bar, prob in zip(bars, probabilities):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{prob:.1%}', ha='center', va='bottom', fontweight='bold')
            
            st.pyplot(fig)
            plt.close()
            
            # Clinical recommendations
            st.markdown("### 🏥 Clinical Recommendations")
            
            if prediction == 0:
                st.success("""
                **No Diabetic Retinopathy Detected**
                - Routine annual screening recommended
                - Continue diabetes management
                """)
            elif prediction == 1:
                st.info("""
                **Mild NPDR**
                - Follow-up in 6-12 months
                - Optimize blood glucose control
                """)
            elif prediction == 2:
                st.warning("""
                **Moderate NPDR**
                - Follow-up in 3-6 months
                - Refer to ophthalmologist
                """)
            elif prediction == 3:
                st.error("""
                **Severe NPDR**
                - Urgent ophthalmology referral
                - Follow-up within 1-2 months
                """)
            else:
                st.error("""
                **Proliferative DR**
                - **URGENT ophthalmology referral required**
                - Immediate treatment needed
                """)
            
            st.markdown("---")
            st.warning("""
            ⚠️ **Disclaimer**: This is an AI-assisted tool for demonstration purposes. 
            Always consult a qualified ophthalmologist for diagnosis and treatment.
            """)
    
    else:
        st.info("""
        👆 **How to use:**
        1. Upload a fundus photograph using the file uploader
        2. Click "Analyze Image"
        3. View AI-powered diagnosis and recommendations
        
        **Supported formats:** PNG, JPG, JPEG
        """)


if __name__ == '__main__':
    main()

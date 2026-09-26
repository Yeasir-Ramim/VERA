"""
VERA Phase 2 - Streamlit Web Application

Interactive web interface for DR classification with explainability.
"""

import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.classification import DRClassifier
from src.models.vessel_segmentation import UNetVesselSegmenter
from src.data.preprocessing import FundusPreprocessor
from src.explainability.gradcam import GradCAMPlusPlus
from src.explainability.overlap_score import compute_vessel_attention_overlap
from src.utils.config import load_config
import matplotlib.pyplot as plt


# Page configuration
st.set_page_config(
    page_title="VERA - DR Detection",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #34495e;
        margin-top: 2rem;
    }
    .metric-card {
        background-color: #ecf0f1;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .grade-0 {color: #27ae60; font-weight: bold;}
    .grade-1 {color: #3498db; font-weight: bold;}
    .grade-2 {color: #f39c12; font-weight: bold;}
    .grade-3 {color: #e67e22; font-weight: bold;}
    .grade-4 {color: #c0392b; font-weight: bold;}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model_and_config():
    """Load model and configuration (cached)."""
    try:
        config_path = Path(__file__).parent.parent / 'configs' / 'config.yaml'
        config = load_config(str(config_path))
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Load DR classifier
        checkpoint_path = Path(__file__).parent.parent / 'models' / 'checkpoints' / 'best_model.pth'
        
        if not checkpoint_path.exists():
            st.error(f"Model checkpoint not found at {checkpoint_path}")
            return None, None, None, None
        
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model = DRClassifier(
            backbone_name=config['model']['backbone'],
            fusion_strategy=config['model']['fusion_strategy'],
            num_classes=config['model']['num_classes'],
            pretrained=False
        ).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # Load vessel segmenter
        vessel_checkpoint_path = Path(__file__).parent.parent / 'models' / 'checkpoints' / 'vessel_segmenter.pth'
        vessel_segmenter = None
        
        if vessel_checkpoint_path.exists():
            vessel_segmenter = UNetVesselSegmenter(
                encoder_name=config['vessel_segmentation']['encoder'],
                pretrained=False
            ).to(device)
            vessel_checkpoint = torch.load(vessel_checkpoint_path, map_location=device)
            vessel_segmenter.load_state_dict(vessel_checkpoint['model_state_dict'])
            vessel_segmenter.eval()
        
        return model, vessel_segmenter, config, device
    except Exception as e:
        st.error(f"Error loading model: {e}")
        import traceback
        st.error(traceback.format_exc())
        return None, None, None, None


def preprocess_image(image, config):
    """Preprocess uploaded image."""
    # Convert PIL to numpy
    image_np = np.array(image)
    
    # Convert RGB to BGR for OpenCV
    if len(image_np.shape) == 3:
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    else:
        image_bgr = image_np
    
    # Create preprocessor
    preprocessor = FundusPreprocessor(
        target_size=config['data']['image_size'],
        circular_crop=True,
        ben_graham_enabled=True,
        clahe_enabled=True
    )
    
    # Preprocess
    processed = preprocessor.preprocess(image_bgr)
    
    return processed, image_bgr


def segment_vessels(vessel_segmenter, image_tensor, device):
    """Segment blood vessels from fundus image."""
    if vessel_segmenter is None:
        # Return empty vessel map if segmenter not available
        return torch.zeros(1, image_tensor.shape[1], image_tensor.shape[2])
    
    vessel_segmenter.eval()
    with torch.no_grad():
        image_batch = image_tensor.unsqueeze(0).to(device)
        vessel_output = vessel_segmenter(image_batch)
        vessel_map = torch.sigmoid(vessel_output) > 0.5
        return vessel_map.squeeze(0).cpu()


def predict(model, image_tensor, vessel_tensor, device):
    """Run inference."""
    model.eval()
    
    with torch.no_grad():
        # Add batch dimension
        image_batch = image_tensor.unsqueeze(0).to(device)
        vessel_batch = vessel_tensor.unsqueeze(0).to(device)
        
        # Forward pass
        outputs = model(image_batch, vessel_batch)
        probabilities = torch.softmax(outputs, dim=1)
        prediction = outputs.argmax(dim=1).item()
        confidence = probabilities[0, prediction].item()
    
    return prediction, probabilities[0].cpu().numpy(), confidence


def main():
    # Header
    st.markdown('<h1 class="main-header">🩺 VERA - Diabetic Retinopathy Detection</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #7f8c8d;">Vascular Explainable Retinopathy Assessment System</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("https://via.placeholder.com/300x100/3498db/ffffff?text=VERA", use_container_width=True)
        st.markdown("## About VERA")
        st.info("""
        VERA is an advanced AI system for detecting diabetic retinopathy (DR) 
        from fundus images. It uses:
        
        - **Vessel-aware neural networks** for better accuracy
        - **Grad-CAM++ explainability** for transparent predictions
        - **Multi-scale attention** for detailed analysis
        """)
        
        st.markdown("## DR Severity Grades")
        st.markdown("""
        - <span class="grade-0">Grade 0</span>: No DR
        - <span class="grade-1">Grade 1</span>: Mild NPDR
        - <span class="grade-2">Grade 2</span>: Moderate NPDR
        - <span class="grade-3">Grade 3</span>: Severe NPDR
        - <span class="grade-4">Grade 4</span>: Proliferative DR
        """, unsafe_allow_html=True)
    
    # Load model
    model, vessel_segmenter, config, device = load_model_and_config()
    
    if model is None:
        st.error("Failed to load model. Please ensure the model checkpoint exists.")
        return
    
    vessel_status = "✅ Loaded" if vessel_segmenter else "⚠️ Not available (will use empty vessel maps)"
    st.success(f"✅ DR Classifier loaded on {device.upper()}")
    st.info(f"Vessel Segmenter: {vessel_status}")
    
    # Main content
    st.markdown('<h2 class="sub-header">Upload Fundus Image</h2>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Choose a retinal fundus image...",
        type=['png', 'jpg', 'jpeg'],
        help="Upload a color fundus photograph of the retina"
    )
    
    if uploaded_file is not None:
        # Load image
        image = Image.open(uploaded_file)
        
        # Display original
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Original Image")
            st.image(image, use_container_width=True)
        
        # Process button
        if st.button("🔍 Analyze Image", type="primary"):
            with st.spinner("Processing image..."):
                # Preprocess
                processed, image_bgr = preprocess_image(image, config)
                
                # Convert to tensor
                image_tensor = torch.from_numpy(processed).permute(2, 0, 1).float()
                
                # Segment vessels
                vessel_tensor = segment_vessels(vessel_segmenter, image_tensor, device)
                
                # Show vessel segmentation
                with col2:
                    st.markdown("### Vessel Segmentation")
                    if vessel_segmenter:
                        vessel_vis = vessel_tensor.squeeze().numpy() * 255
                        st.image(vessel_vis.astype(np.uint8), use_container_width=True, caption="Detected Blood Vessels")
                    else:
                        st.warning("Vessel segmentation not available")
                
                # Predict
                prediction, probabilities, confidence = predict(
                    model, image_tensor, vessel_tensor, device
                )
            
            # Display results
            st.markdown('<h2 class="sub-header">📊 Diagnosis Results</h2>', unsafe_allow_html=True)
            
            # Grade names and colors
            grade_names = ['No DR', 'Mild NPDR', 'Moderate NPDR', 'Severe NPDR', 'Proliferative DR']
            grade_colors = ['#27ae60', '#3498db', '#f39c12', '#e67e22', '#c0392b']
            
            # Result card
            result_col1, result_col2, result_col3 = st.columns([2, 1, 1])
            
            with result_col1:
                st.markdown(f"### Predicted Grade")
                st.markdown(f'<h1 style="color: {grade_colors[prediction]}; font-size: 4rem;">{prediction}</h1>', 
                          unsafe_allow_html=True)
                st.markdown(f'<h3>{grade_names[prediction]}</h3>', unsafe_allow_html=True)
            
            with result_col2:
                st.metric("Confidence", f"{confidence*100:.1f}%")
            
            with result_col3:
                referable = "Yes" if prediction >= 2 else "No"
                st.metric("Referable", referable)
            
            # Probability distribution
            st.markdown("### Class Probabilities")
            prob_fig, ax = plt.subplots(figsize=(10, 4))
            bars = ax.bar(grade_names, probabilities, color=grade_colors, alpha=0.7, edgecolor='black')
            bars[prediction].set_alpha(1.0)
            bars[prediction].set_edgecolor('red')
            bars[prediction].set_linewidth(3)
            
            ax.set_ylabel('Probability', fontweight='bold')
            ax.set_ylim([0, 1])
            ax.grid(axis='y', alpha=0.3)
            
            # Add value labels
            for bar, prob in zip(bars, probabilities):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{prob:.1%}', ha='center', va='bottom', fontweight='bold')
            
            st.pyplot(prob_fig)
            plt.close()
            
            # Explainability with Grad-CAM++
            st.markdown("### 🔬 Model Explainability (Grad-CAM++)")
            
            try:
                with st.spinner("Generating attention heatmap..."):
                    # Get the backbone's last conv layer
                    target_layer = model.backbone.model.layer4[-1]
                    
                    # Create Grad-CAM++
                    gradcam = GradCAMPlusPlus(model.backbone, target_layer)
                    
                    # Generate heatmap for predicted class
                    image_batch = image_tensor.unsqueeze(0).to(device)
                    vessel_batch = vessel_tensor.unsqueeze(0).to(device)
                    
                    heatmap = gradcam.generate_heatmap(
                        image_batch,
                        vessel_batch,
                        target_class=prediction
                    )
                    
                    # Overlay on original
                    original_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
                    original_resized = cv2.resize(original_rgb, (image_tensor.shape[2], image_tensor.shape[1]))
                    
                    # Create colored heatmap
                    heatmap_colored = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
                    overlay = cv2.addWeighted(original_resized, 0.6, heatmap_colored, 0.4, 0)
                    
                    # Display
                    cam_col1, cam_col2 = st.columns(2)
                    
                    with cam_col1:
                        st.markdown("**Attention Heatmap**")
                        st.image(heatmap, use_container_width=True, clamp=True, caption="Model focus areas (red = high attention)")
                    
                    with cam_col2:
                        st.markdown("**Overlay on Original**")
                        st.image(overlay, use_container_width=True, caption="Attention overlaid on fundus image")
                    
                    # Compute vessel-attention overlap if vessel segmenter available
                    if vessel_segmenter:
                        overlap_metrics = compute_vessel_attention_overlap(
                            heatmap,
                            vessel_tensor.squeeze().numpy()
                        )
                        
                        st.markdown("**Vessel-Attention Overlap Analysis**")
                        metric_col1, metric_col2, metric_col3 = st.columns(3)
                        
                        with metric_col1:
                            st.metric("Overlap Score", f"{overlap_metrics['overlap_score']:.3f}")
                        with metric_col2:
                            st.metric("Vessel Coverage", f"{overlap_metrics['coverage']:.3f}")
                        with metric_col3:
                            st.metric("Attention Precision", f"{overlap_metrics['precision']:.3f}")
                        
                        st.info("""
                        **Overlap Score**: How well the model's attention aligns with blood vessels (higher = better vessel focus)
                        
                        **Coverage**: Proportion of vessel pixels that receive high attention
                        
                        **Precision**: Proportion of high-attention pixels that fall on vessels
                        """)
                
            except Exception as e:
                st.error(f"Error generating explainability: {e}")
                import traceback
                st.error(traceback.format_exc())
            
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
                - Monitor blood pressure
                """)
            elif prediction == 2:
                st.warning("""
                **Moderate NPDR**
                - Follow-up in 3-6 months
                - Refer to ophthalmologist
                - Strict glucose and BP control
                """)
            elif prediction == 3:
                st.error("""
                **Severe NPDR**
                - Urgent ophthalmology referral
                - Follow-up within 1-2 months
                - Consider panretinal photocoagulation
                """)
            else:  # prediction == 4
                st.error("""
                **Proliferative DR**
                - **URGENT ophthalmology referral required**
                - Immediate treatment needed
                - High risk of vision loss
                """)
            
            # Disclaimer
            st.markdown("---")
            st.warning("""
            ⚠️ **Important Disclaimer**: This is an AI-assisted diagnostic tool and should not replace 
            professional medical judgment. Always consult with a qualified ophthalmologist for final diagnosis 
            and treatment decisions.
            """)
    
    else:
        # Instructions
        st.info("""
        👆 **How to use:**
        1. Upload a color fundus photograph using the file uploader above
        2. Click the "Analyze Image" button
        3. View the AI-powered diagnosis and recommendations
        
        **Supported formats:** PNG, JPG, JPEG
        """)
        
        # Example images section
        st.markdown("### 📸 Example Images")
        st.markdown("""
        For testing, you can use fundus images from:
        - [APTOS 2019 Dataset](https://www.kaggle.com/c/aptos2019-blindness-detection)
        - [EyePACS Dataset](https://www.kaggle.com/c/diabetic-retinopathy-detection)
        """)


if __name__ == '__main__':
    main()

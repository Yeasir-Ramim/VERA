# VERA Phase 1 - MVP (Proof of Concept)

This folder contains the complete Phase 1 implementation of the VERA (Vascular Explainable Retinopathy Assessment) project.

## Phase 1 Overview

Phase 1 demonstrates the core proof-of-concept for vessel-aware diabetic retinopathy detection:

- **Single Dataset**: APTOS 2019 (3,662 images)
- **Preprocessing**: Circular masking + Green-channel CLAHE
- **Vessel Segmentation**: Pretrained U-Net (DRIVE weights)
- **Fusion Strategy**: Simple 4-channel stacking [R, G, B, V]
- **Backbones**: ResNet-18, EfficientNet-B0
- **Loss**: Class-weighted Cross-Entropy
- **Explainability**: Grad-CAM visual reports

## Directory Structure

```
phase1_mvp/
├── src/                    # Core implementation modules
│   ├── preprocessing.py    # Image preprocessing
│   ├── vessel_segmentation.py  # U-Net vessel extraction
│   ├── model.py           # 4-channel CNN models
│   ├── dataset.py         # PyTorch dataset loaders
│   ├── explainability.py  # Grad-CAM implementation
│   └── utils.py           # Helper functions
├── tests/                 # Unit tests
├── notebooks/             # Interactive demos
├── outputs/              # Generated results & checkpoints
├── run_mvp.py            # Main execution script
└── app.py                # Basic Streamlit interface
```

## Quick Start

```bash
# Install dependencies
pip install -r ../requirements.txt

# Run complete MVP pipeline
python run_mvp.py --epochs 10 --batch_size 16 --backbone resnet18

# Run tests
python -m unittest discover tests

# Launch demo notebook
jupyter notebook notebooks/mvp_demo.ipynb

# Launch basic web interface
streamlit run app.py
```

## Key Results (Phase 1)

- **Baseline 3-Channel Accuracy**: ~72%
- **VERA 4-Channel Accuracy**: ~78%
- **Improvement**: +6% with vessel-aware fusion
- **Explainability**: 4-panel Grad-CAM reports for all 5 DR grades

## Next Steps

See Phase 2 for production-ready implementation with:
- Multi-dataset support (EyePACS, Messidor-2)
- Advanced fusion architectures
- QWK loss optimization
- Enhanced explainability metrics
- Production web interface

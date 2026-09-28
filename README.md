# VERA: Vascular Explainable Retinopathy Assessment
### Clinical Decision Support System (CDSS) for Ophthalmic & Medical Retina Specialists

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.6](https://img.shields.io/badge/PyTorch-2.6-EE4C2C.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40%2B-FF4B4B.svg)](https://streamlit.io/)
[![PyMuPDF](https://img.shields.io/badge/PyMuPDF-1.28%2B-008080.svg)](https://pymupdf.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Diagnostic Standard](https://img.shields.io/badge/Standard-ICDR%205--Stage-brightgreen.svg)](https://www.aao.org/)

---

## 📌 Clinical Overview & Motivation

**Diabetic Retinopathy (DR)** is the leading cause of preventable blindness among working-age adults globally. While modern deep convolutional neural networks achieve high benchmark classification metrics, standard "black-box" vision models frequently suffer from **shortcut learning**—leveraging camera-specific vignetting, optic disc illumination artifacts, or pigmentation gradients rather than authentic microvascular lesions.

**VERA (Vascular Explainable Retinopathy Assessment)** is an advanced Clinical Decision Support System (CDSS) engineered from the ground up for **Medical Retina Specialists, Vitreoretinal Surgeons, and Clinical Ophthalmologists**. 

VERA couples deep learning with classical computer vision and domain-specific clinical pathology:
1. **Explicit 4-Channel Input:** Combines high-resolution fundus photography (RGB) with a specialized 4th channel representing the multi-scale retinal vascular tree.
2. **Quantitative Lesion Biomarkers:** Quantifies microaneurysms, intraretinal blot hemorrhages, hard lipid exudates, and cotton wool spots.
3. **ETDRS 4-2-1 Rule Engine:** Automated 4-quadrant anatomical distribution analysis for objective Severe NPDR vs. PDR differentiation.
4. **Macular Threat Triage (CSME / DME):** Measures hard exudate proximity to the Foveal Avascular Zone (FAZ) and Optic Disc diameter.
5. **Continuous Vessel-Attention Grounding:** Evaluates Grad-CAM++ activation maps against segmented microvessels to guarantee non-shortcut explanations.
6. **Multi-Model Consensus Verification:** Real-time cross-inference against distinct model topologies to safeguard against single-model edge-case failures.
7. **Certified Consultation PDF Export:** Generates high-resolution, multi-panel A4 clinical consultation reports ready for EHR and hospital archives.

---

## 🔬 System Architecture & Diagnostic Pipeline

```
                                  [ Retinal Fundus Image ]
                                             │
                      ┌──────────────────────┴──────────────────────┐
                      ▼                                             ▼
          [ Preprocessing Pipeline ]                   [ Multi-Scale Vessel Extractor ]
       (Ben Graham / CLAHE / Circle Crop)            (Frangi Vesselness + Morphological Top-Hat)
                      │                                             │
                      │ RGB (3-Channel)                             │ Vessel Map (1-Channel)
                      └──────────────────────┬──────────────────────┘
                                             ▼
                           [ 4-Channel Fused Input Tensor ]
                                 (B x 4 x 512 x 512)
                                             │
                      ┌──────────────────────┴──────────────────────┐
                      ▼                                             ▼
          [ Production ResNet-50 ]                     [ Specialist Biomarker Engine ]
         (4-Channel Early Fusion)                     • Microaneurysm & Hemorrhage Counts
                      │                               • Hard Exudate Foci & Total Area
                      ├────────────────────────┐      • Foveal Center (FAZ) Proximity
                      ▼                        ▼      • ETDRS 4-Quadrant Partitioning
             [ ICDR Staging ]          [ Grad-CAM++ Engine ]        │
            (Grades 0 to 4)                    │                    │
                      │                        ▼                    │
                      │            [ Vessel Overlap Score ]         │
                      │            (Quantitative Grounding)         │
                      └────────────────────────┬────────────────────┘
                                               ▼
                           [ Clinical Decision Support Studio ]
                         • 5-Panel Synchronized Multimodal Viewer
                         • Multi-Model Architectural Consensus
                         • Actionable Treatment Recommendation Plan
                         • One-Click Certified Consultation PDF Export
```

---

## 🩺 Diagnostic Staging Standard (ICDR)

VERA adheres strictly to the **International Clinical Diabetic Retinopathy (ICDR)** severity scale:

| Stage | Severity Grade | Clinical Ophthalmic Definition | Triage Urgency |
|:---:|:---|:---|:---|
| **0** | **No Apparent DR** | Intact retinal microvasculature. No microaneurysms or exudates. | Routine annual rescreen (12–24 mos) |
| **1** | **Mild NPDR** | Microaneurysms only. Scattered punctate red lesions in temporal arcade. | Review in 12 months; glycemic control |
| **2** | **Moderate NPDR** | Intraretinal blot hemorrhages and/or hard exudates (exceeds Grade 1, less than Grade 3). | Refer to specialist within 3–6 months |
| **3** | **Severe NPDR** | **ETDRS 4-2-1 Rule satisfied:** ≥20 intraretinal hemorrhages in all 4 quadrants, OR venous beading in ≥2 quadrants, OR prominent IRMA in ≥1 quadrant. | Urgent specialist review within 2–4 weeks |
| **4** | **Proliferative DR (PDR)** | Active neovascularization (NVD/NVE), preretinal/vitreous hemorrhage, or fibrovascular traction. | **Emergency consultation within 48–72 hours** (PRP / Anti-VEGF) |

---

## 📊 Model Topologies & Research Ablation Study

VERA implements and evaluates multiple architectural paradigms to validate the benefit of explicit vascular channel integration:

| Architecture | Input Channels | Topology Description | Test Accuracy | Quadratic Weighted Kappa (QWK) | Vessel Overlap Score |
|:---|:---:|:---|:---:|:---:|:---:|
| **ResNet-50 (Production)** | **4 Channels** | **Early Fusion (RGB + Vessel Mask)** | **88.2%** | **0.893** | **0.342** |
| ResNet-18 (Attention-Gated) | 4 Channels | Spatial Cross-Attention Vessel Gating | 85.5% | 0.861 | 0.328 |
| ResNet-18 (Dual-Branch) | 3ch + 1ch | Dual-Encoder Late Concatenation | 84.1% | 0.849 | 0.315 |
| ResNet-18 (Early Fusion) | 4 Channels | 4-Channel Stem Convolution | 83.8% | 0.843 | 0.310 |
| ResNet-18 (Baseline Standard) | 3 Channels | Standard RGB Baseline (No Vessels) | 81.2% | 0.804 | 0.221 |

> **Key Takeaway:** Integrating the explicit 4th retinal vessel channel yields a **+8.9% increase in Quadratic Weighted Kappa (QWK)** and a **+54.7% improvement in Vessel Overlap Score**, proving that the model bases its predictions on verified microvascular abnormalities rather than spurious background shortcuts.

---

## 📁 Repository Structure

```
VERA/
├── .streamlit/
│   └── config.toml             # Streamlit high-contrast dark theme configuration
├── app.py                      # Main Clinical Decision Support Web Application
├── checkpoints/
│   └── production_model/
│       └── best_model.pth      # Pretrained ResNet-50 (4-channel early fusion) weights
├── configs/
│   └── config.yaml             # Hyperparameters, paths, and preprocessing options
├── data/
│   └── sample_data/            # Representative clinical fundus cases across Grades 0 to 4
├── src/
│   ├── __init__.py             # Module exports
│   ├── clinical_biomarkers.py  # Lesion detection, FAZ fovea, ETDRS 4-2-1 rule engine
│   ├── dataset.py              # PyTorch Dataset loaders with 4th-channel integration
│   ├── evaluate.py             # Evaluation metrics (QWK, confusion matrix, ROC-AUC)
│   ├── explainability.py       # Grad-CAM, Grad-CAM++, and Vessel Overlap Score
│   ├── models.py               # Model builders & safe weights loading for PyTorch 2.6+
│   ├── pdf_report.py           # 300-DPI A4 clinical consultation PDF report generator
│   ├── preprocessing.py        # Ben Graham filter, CLAHE, and circle cropping
│   ├── utils.py                # Logging, seeding, and file helpers
│   └── vessel_segmentation.py  # Multi-scale Frangi filter and morphological vessel extractor
├── tests/
│   ├── test_pipeline.py        # End-to-end integration tests (inference, Grad-CAM, pipeline)
│   └── test_phase2.py          # Production components and checkpoint validation
├── requirements.txt            # Python dependencies
├── LICENSE                     # MIT License
└── README.md                   # Project documentation
```

---

## ⚡ Quickstart & Installation

### 1. Prerequisites
- Python 3.10, 3.11, 3.12, or 3.13
- CUDA-enabled GPU (optional, CPU inference fully supported)

### 2. Clone and Setup Environment

```bash
# Clone the repository
git clone https://github.com/Yeasir-Ramim/VERA.git
cd VERA

# Create and activate a virtual environment
python -m venv .venv

# Windows:
.venv\Scripts\activate
# Linux / macOS:
source .venv/bin/activate

# Install required packages
pip install -r requirements.txt
```

### 3. Launch the Clinical Decision Support Studio

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`.

---

## 🖥️ Specialist Workflow & Application Views

The VERA interface is structured into five specialist viewpoints:

### View 1: 🩺 Patient Diagnostic Evaluation
- **Automated Staging Banner:** High-contrast grade badges (0 to 4), confidence score, and entropy uncertainty metric.
- **5-Panel Multimodal Inspection Console:** Side-by-side synchronized view of:
  1. Raw Color Fundus
  2. Contrast-Enhanced Fundus (CLAHE / Ben Graham)
  3. Extracted Retinal Vascular Tree
  4. Grad-CAM++ Lesion Activation Heatmap
  5. Annotated Quadrant & Biomarker Map
- **Quantitative Lesion Counts:** Microaneurysms, hemorrhages, hard exudate foci, and cotton wool spots.
- **Diabetic Macular Edema (CSME) Triage:** Measures nearest exudate distance to FAZ (< 1 DD vs. < 2 DD).
- **ETDRS 4-Quadrant Distribution:** Quad-by-quad assessment (ST, IT, SN, IN) testing the clinical 4-2-1 criteria for severe NPDR.

### View 2: 🧭 Specialist Decision Support Matrix
- Comprehensive clinical reference guide explaining:
  - ICDR staging definitions and key lesions.
  - Actionable referral urgency timelines.
  - The diagnostic significance of every VERA feature in clinical practice.

### View 3: ⚖️ Multi-Model Consensus
- Runs synchronized inference across 4 distinct neural topologies.
- Alerts the specialist to potential edge-case divergence between models.

### View 4: 📋 Clinical Consultation Report & PDF Export
- Generates a certified clinical consultation document.
- **One-Click Download (PDF):** Standard A4 medical document formatted at 300 DPI with embedded fundus images, biomarker summary tables, and digital physician sign-off.
- **Markdown Export (.md):** Raw text summary ready for hospital EHR systems.

### View 5: 📊 Research Ablations & Benchmarks
- Interactive metrics tables and confusion matrix analyses comparing 4-channel fusion against 3-channel RGB baselines.

---

## 🧪 Running Automated Tests

Run the full automated test suite to verify model loading, vessel segmentation, Grad-CAM hooks, and biomarker calculation:

```bash
python -m unittest tests/test_pipeline.py tests/test_phase2.py
```

Expected output:
```text
...........
----------------------------------------------------------------------
Ran 11 tests in 43.924s

OK
```

---

## 📜 Ethical AI & Medical Disclaimer

> **IMPORTANT CLINICAL NOTICE:**  
> VERA is developed as an academic and translational research Clinical Decision Support System (CDSS) designed to augment and assist qualified ophthalmologists and retinal specialists. It is not an autonomous diagnostic device. All therapeutic interventions, laser photocoagulation schedules, and anti-VEGF administration must be independently verified by a licensed ophthalmologist following in-person dilated fundus biomicroscopy or fluorescein angiography.

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

"""
VERA: Vascular Explainable Retinopathy Assessment
Clinical Decision Support System for Diabetic Retinopathy.
"""

from .preprocessing import preprocess_fundus, crop_fundus_circle, apply_clahe, apply_ben_graham
from .vessel_segmentation import VesselSegmenter, UNetVesselSegmenter, extract_vessel_map_multiscale
from .models import build_model, DRClassifier, DualBranchDRClassifier, SpatialAttentionGatedDRClassifier, load_checkpoint_safe
from .explainability import GradCAM, GradCAMPlusPlus, compute_vessel_attention_overlap, overlay_cam_on_image
from .clinical_biomarkers import analyze_clinical_biomarkers, ClinicalBiomarkerReport, derive_etdrs_clinical_grade, reconcile_icdr_grade
from .evaluate import evaluate_model, plot_confusion_matrix, ICDR_CLASS_NAMES
from .pdf_report import generate_clinical_pdf_report

__version__ = "2.0.0"

__all__ = [
    "preprocess_fundus",
    "crop_fundus_circle",
    "apply_clahe",
    "apply_ben_graham",
    "VesselSegmenter",
    "UNetVesselSegmenter",
    "extract_vessel_map_multiscale",
    "build_model",
    "DRClassifier",
    "DualBranchDRClassifier",
    "SpatialAttentionGatedDRClassifier",
    "load_checkpoint_safe",
    "GradCAM",
    "GradCAMPlusPlus",
    "compute_vessel_attention_overlap",
    "overlay_cam_on_image",
    "analyze_clinical_biomarkers",
    "ClinicalBiomarkerReport",
    "derive_etdrs_clinical_grade",
    "reconcile_icdr_grade",
    "evaluate_model",
    "plot_confusion_matrix",
    "ICDR_CLASS_NAMES",
    "generate_clinical_pdf_report",
]

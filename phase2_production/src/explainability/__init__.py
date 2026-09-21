"""
VERA Phase 2 - Explainability Module

Exports explainability tools including Grad-CAM, overlap metrics, and visualization.
"""

from .gradcam import (
    GradCAM,
    GradCAMPlusPlus,
    overlay_heatmap_on_image,
    create_gradcam
)

from .overlap_score import (
    compute_vessel_attention_overlap,
    compute_vessel_coverage,
    compute_precision_on_vessels,
    compute_overlap_metrics,
    visualize_overlap,
    OverlapScoreEvaluator
)

from .visualization import (
    create_4panel_report,
    create_comprehensive_report,
    generate_batch_reports
)

__all__ = [
    # Grad-CAM
    'GradCAM',
    'GradCAMPlusPlus',
    'overlay_heatmap_on_image',
    'create_gradcam',
    
    # Overlap Metrics
    'compute_vessel_attention_overlap',
    'compute_vessel_coverage',
    'compute_precision_on_vessels',
    'compute_overlap_metrics',
    'visualize_overlap',
    'OverlapScoreEvaluator',
    
    # Visualization
    'create_4panel_report',
    'create_comprehensive_report',
    'generate_batch_reports',
]

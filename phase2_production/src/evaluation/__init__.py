"""
VERA Phase 2 - Evaluation Module

Exports evaluation metrics and evaluator.
"""

from .metrics import (
    compute_all_metrics,
    compute_confusion_matrix,
    get_classification_report,
    compute_grade_distribution
)

from .evaluator import ModelEvaluator

__all__ = [
    'compute_all_metrics',
    'compute_confusion_matrix',
    'get_classification_report',
    'compute_grade_distribution',
    'ModelEvaluator',
]

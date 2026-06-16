"""Machine-learning validation utilities for OBPI."""

from __future__ import annotations

from obpi.ml.ablation import run_ablation, write_ablation_report
from obpi.ml.correlation import (
    compare_benchmarks,
    cronbach_alpha,
    expert_correlation,
    orthogonal_variance_test,
    spearman_correlation,
)
from obpi.ml.explainability import (
    compute_permutation_importance,
    compute_shap,
    get_metric_weights,
    save_metric_weights,
)
from obpi.ml.validation import (
    METRIC_COLUMNS,
    TrainingPreparationResult,
    ValidationResult,
    create_labels,
    evaluate_estimator,
    evaluate_holdout_predictions,
    prepare_labeled_data,
    prepare_training_frame,
    save_training_preparation,
    train_logistic,
    train_svm,
    train_xgboost,
    validate,
)

__all__ = [
    "METRIC_COLUMNS",
    "TrainingPreparationResult",
    "ValidationResult",
    "compute_permutation_importance",
    "compute_shap",
    "create_labels",
    "evaluate_estimator",
    "evaluate_holdout_predictions",
    "get_metric_weights",
    "prepare_labeled_data",
    "prepare_training_frame",
    "compare_benchmarks",
    "cronbach_alpha",
    "expert_correlation",
    "orthogonal_variance_test",
    "run_ablation",
    "save_training_preparation",
    "save_metric_weights",
    "spearman_correlation",
    "train_logistic",
    "train_svm",
    "train_xgboost",
    "validate",
    "write_ablation_report",
]

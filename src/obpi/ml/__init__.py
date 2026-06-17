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
    ExplainabilityResult,
    compute_permutation_importance,
    compute_shap,
    get_metric_weights,
    run_explainability,
    save_explainability_results,
    save_metric_weights,
)
from obpi.ml.research_validation import (
    build_validation_audit,
    save_validation_audit,
)
from obpi.ml.validation import (
    METRIC_COLUMNS,
    TrainingPreparationResult,
    ValidationResult,
    create_labels,
    evaluate_estimator,
    evaluate_holdout_predictions,
    extract_prepared_xy,
    prepare_labeled_data,
    prepare_training_frame,
    save_training_preparation,
    save_validation_results,
    train_logistic,
    train_svm,
    train_xgboost,
    validate,
    validate_prepared_data,
)

__all__ = [
    "METRIC_COLUMNS",
    "ExplainabilityResult",
    "TrainingPreparationResult",
    "ValidationResult",
    "compute_permutation_importance",
    "compute_shap",
    "create_labels",
    "build_validation_audit",
    "extract_prepared_xy",
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
    "run_explainability",
    "save_validation_audit",
    "save_explainability_results",
    "save_training_preparation",
    "save_metric_weights",
    "save_validation_results",
    "spearman_correlation",
    "train_logistic",
    "train_svm",
    "train_xgboost",
    "validate",
    "validate_prepared_data",
    "write_ablation_report",
]

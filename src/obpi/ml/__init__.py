"""Machine-learning validation utilities for OBPI."""

from obpi.ml.validation import (
    METRIC_COLUMNS,
    ValidationResult,
    create_labels,
    evaluate_estimator,
    evaluate_holdout_predictions,
    prepare_labeled_data,
    train_logistic,
    train_svm,
    train_xgboost,
    validate,
)
from obpi.ml.explainability import (
    compute_permutation_importance,
    compute_shap,
    get_metric_weights,
    save_metric_weights,
)
from obpi.ml.ablation import run_ablation, write_ablation_report
from obpi.ml.correlation import (
    compare_benchmarks,
    cronbach_alpha,
    expert_correlation,
    orthogonal_variance_test,
    spearman_correlation,
)

__all__ = [
    "METRIC_COLUMNS",
    "ValidationResult",
    "compute_permutation_importance",
    "compute_shap",
    "create_labels",
    "evaluate_estimator",
    "evaluate_holdout_predictions",
    "get_metric_weights",
    "prepare_labeled_data",
    "compare_benchmarks",
    "cronbach_alpha",
    "expert_correlation",
    "orthogonal_variance_test",
    "run_ablation",
    "save_metric_weights",
    "spearman_correlation",
    "train_logistic",
    "train_svm",
    "train_xgboost",
    "validate",
    "write_ablation_report",
]

"""Explainability helpers for OBPI validation models."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from obpi.ml.validation import (
    METRIC_COLUMNS,
    extract_prepared_xy,
    train_logistic,
    train_svm,
    train_xgboost,
)


@dataclass(frozen=True)
class ExplainabilityResult:
    """Explainability artifacts and summary metadata."""

    model_name: str
    metric_weights: dict[str, float]
    permutation_importance: pd.DataFrame
    shap_values: pd.DataFrame | None
    report: dict[str, Any]


def compute_shap(model: Any, x: pd.DataFrame) -> pd.DataFrame:
    """Compute SHAP values for a fitted tree model.

    The Week 7 roadmap expects `shap.TreeExplainer` on the best XGBoost model.
    SHAP is imported lazily so the rest of the validation suite can run even
    in environments that have not installed the optional explainability stack.
    """
    try:
        import shap
    except ModuleNotFoundError as exc:
        raise ImportError("shap is required for compute_shap(); install shap") from exc

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(x)
    if isinstance(shap_values, list):
        shap_values = shap_values[-1]
    if getattr(shap_values, "ndim", 0) == 3:
        shap_values = shap_values[:, :, -1]
    return pd.DataFrame(np.asarray(shap_values), index=x.index, columns=x.columns)


def get_metric_weights(
    importance_values: pd.DataFrame | pd.Series | dict[str, float],
) -> dict[str, float]:
    """Normalize metric importances into weights that sum to 1.0."""
    if isinstance(importance_values, pd.DataFrame):
        importances = importance_values.abs().mean(axis=0)
    elif isinstance(importance_values, pd.Series):
        importances = importance_values.astype(float).abs()
    else:
        importances = pd.Series(importance_values, dtype=float).abs()

    total = float(importances.sum())
    if total <= 0.0:
        raise ValueError("importance values must contain at least one positive value")

    weights = (importances / total).sort_values(ascending=False)
    return {metric: float(weight) for metric, weight in weights.items()}


def compute_permutation_importance(
    model: Any,
    x: pd.DataFrame,
    y: pd.Series,
    scoring: str = "roc_auc",
    n_repeats: int = 10,
    random_state: int = 42,
) -> pd.DataFrame:
    """Compute permutation importance for a fitted estimator."""
    result = permutation_importance(
        model,
        x,
        y,
        scoring=scoring,
        n_repeats=n_repeats,
        random_state=random_state,
    )
    return pd.DataFrame(
        {
            "metric": x.columns,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    ).sort_values("importance_mean", ascending=False, ignore_index=True)


def save_metric_weights(weights: dict[str, float], output_path: str | Path) -> None:
    """Persist metric weights for dashboard/API consumption."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(weights, indent=2, sort_keys=True) + "\n")


def run_explainability(
    prepared_df: pd.DataFrame,
    metric_columns: list[str] | None = None,
    cv_splits: int = 5,
    prefer_model: str = "svm",
    include_xgboost: bool = False,
    permutation_repeats: int = 10,
) -> ExplainabilityResult:
    """Run Week 7 explainability on the prepared extreme-quartile dataset."""
    metric_columns = metric_columns or METRIC_COLUMNS
    x, y = extract_prepared_xy(prepared_df, metric_columns=metric_columns)
    estimator = None
    model_name = prefer_model
    xgboost_note = None

    if include_xgboost:
        try:
            xgb_grid = train_xgboost(x, y, cv_splits=cv_splits)
        except ImportError as exc:
            xgb_grid = None
            xgboost_note = str(exc)
        else:
            estimator = xgb_grid.best_estimator_
            model_name = "xgboost"
    else:
        xgb_grid = None

    if estimator is None:
        if prefer_model == "logistic":
            grid = train_logistic(x, y, cv_splits=cv_splits)
            estimator = grid.best_estimator_
            model_name = "logistic"
        else:
            grid = train_svm(x, y, cv_splits=cv_splits)
            estimator = grid.best_estimator_
            model_name = "svm"

    permutation_df = compute_permutation_importance(
        estimator,
        x,
        y,
        n_repeats=permutation_repeats,
    )
    notes: list[str] = []
    try:
        weights = get_metric_weights(
            permutation_df.set_index("metric")["importance_mean"]
        )
    except ValueError:
        uniform_weight = 1.0 / len(metric_columns)
        weights = dict.fromkeys(metric_columns, uniform_weight)
        notes.append(
            "Permutation importance returned all-zero importances; using uniform metric weights."
        )

    shap_df: pd.DataFrame | None = None
    shap_note = None
    if xgb_grid is not None:
        try:
            shap_df = compute_shap(estimator, x)
        except ImportError as exc:
            shap_note = str(exc)
    else:
        shap_note = "SHAP skipped because XGBoost was not selected for explainability."

    report = {
        "model_name": model_name,
        "metric_weights": weights,
        "permutation_importance_ranking": permutation_df["metric"].tolist(),
        "shap_available": shap_df is not None,
        "notes": notes + [note for note in [xgboost_note, shap_note] if note],
    }
    return ExplainabilityResult(
        model_name=model_name,
        metric_weights=weights,
        permutation_importance=permutation_df,
        shap_values=shap_df,
        report=report,
    )


def save_explainability_results(
    result: ExplainabilityResult,
    weights_path: str | Path,
    permutation_path: str | Path,
    report_path: str | Path,
    shap_path: str | Path | None = None,
) -> None:
    """Persist Week 7 explainability artifacts to disk."""
    save_metric_weights(result.metric_weights, weights_path)

    permutation_output = Path(permutation_path)
    permutation_output.parent.mkdir(parents=True, exist_ok=True)
    result.permutation_importance.to_csv(permutation_output, index=False)

    report_output = Path(report_path)
    report_output.parent.mkdir(parents=True, exist_ok=True)
    report_output.write_text(json.dumps(result.report, indent=2) + "\n", encoding="utf-8")

    if shap_path is not None and result.shap_values is not None:
        shap_output = Path(shap_path)
        shap_output.parent.mkdir(parents=True, exist_ok=True)
        result.shap_values.to_csv(shap_output, index=False)

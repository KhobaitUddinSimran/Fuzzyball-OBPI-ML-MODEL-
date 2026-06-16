"""Explainability helpers for OBPI validation models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance


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

"""High-level helpers for fitting and applying the fuzzy OBPI engine."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from obpi.fuzzy.engine import DEFAULT_METRICS, FuzzyEngine
from obpi.fuzzy.membership import build_metric_memberships


def fit_fuzzy_engine(
    metrics_df: pd.DataFrame,
    metric_names: Sequence[str] = DEFAULT_METRICS,
    metric_weights: Mapping[str, float] | None = None,
) -> FuzzyEngine:
    """Fit a data-calibrated fuzzy engine from normalized metric columns."""

    metric_list = list(metric_names)
    membership_functions = build_metric_memberships(metrics_df, metric_list)
    return FuzzyEngine(
        metric_names=metric_list,
        membership_functions=membership_functions,
        metric_weights=metric_weights,
    )


def score_metrics_dataframe(
    metrics_df: pd.DataFrame,
    engine: FuzzyEngine | None = None,
    metric_names: Sequence[str] = DEFAULT_METRICS,
    score_column: str = "obpi",
) -> pd.DataFrame:
    """Return a copy of ``metrics_df`` with one OBPI score per row."""

    metric_list = list(metric_names)
    missing = set(metric_list) - set(metrics_df.columns)
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(f"metrics_df is missing metric columns: {missing_text}")

    metric_values = metrics_df[metric_list]
    if not np.isfinite(metric_values.to_numpy(dtype=float)).all():
        raise ValueError("metric columns must contain only finite numeric values")

    scoring_engine = engine or fit_fuzzy_engine(metrics_df, metric_list)
    scored_df = metrics_df.copy()
    scored_df[score_column] = [
        scoring_engine.compute(row)
        for row in metric_values.to_dict(orient="records")
    ]
    return scored_df

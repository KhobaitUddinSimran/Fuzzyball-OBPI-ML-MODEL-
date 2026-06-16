"""Batch scoring helpers for the OBPI fuzzy engine."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import pandas as pd

from obpi.fuzzy.engine import DEFAULT_METRIC_NAMES, FuzzyEngine
from obpi.fuzzy.membership import build_metric_memberships_from_dataframe


def score_dataframe(
    metrics_df: pd.DataFrame,
    metric_columns: list[str] | tuple[str, ...] = DEFAULT_METRIC_NAMES,
    score_column: str = "obpi_score",
    calibrate_memberships: bool = True,
    metric_weights: Mapping[str, float] | None = None,
) -> pd.DataFrame:
    """Score a metric DataFrame and append an OBPI score column."""
    metric_columns = tuple(metric_columns)
    missing = sorted(set(metric_columns) - set(metrics_df.columns))
    if missing:
        raise ValueError(f"missing metric columns: {', '.join(missing)}")

    metric_values = metrics_df.loc[:, metric_columns].astype(float)
    if metric_values.isna().any().any():
        raise ValueError("metric columns cannot contain missing values")
    if ((metric_values < 0.0) | (metric_values > 1.0)).any().any():
        raise ValueError("metric columns must be normalized to [0, 1]")

    memberships = None
    if calibrate_memberships:
        memberships = build_metric_memberships_from_dataframe(metrics_df, metric_columns)

    engine = FuzzyEngine(
        metric_names=metric_columns,
        membership_funcs=memberships,
        metric_weights=metric_weights,
    )
    scored = metrics_df.copy()
    scored[score_column] = [
        engine.compute(row)
        for row in metric_values.to_dict(orient="records")
    ]
    return scored


def score_csv(
    input_path: str | Path,
    output_path: str | Path,
    metric_columns: list[str] | tuple[str, ...] = DEFAULT_METRIC_NAMES,
    calibrate_memberships: bool = True,
) -> pd.DataFrame:
    """Read a CSV of metrics, write scored CSV, and return the scored frame."""
    input_path = Path(input_path)
    output_path = Path(output_path)
    metrics_df = pd.read_csv(input_path)
    scored = score_dataframe(
        metrics_df,
        metric_columns=metric_columns,
        calibrate_memberships=calibrate_memberships,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(output_path, index=False)
    return scored


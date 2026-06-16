"""Normalize processed OBPI metrics and run fuzzy scoring on real data."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from obpi.fuzzy.scoring import fit_fuzzy_engine, score_metrics_dataframe

RAW_TO_NORMALIZED_METRICS = {
    "M1_SC": "M1",
    "M2_OIRC": "M2",
    "M3_BRPC": "M3",
    "M4_OBR90": "M4",
    "M5_RBTL": "M5",
    "M6_RUP": "M6",
    "M7_SCI": "M7",
    "M8_LPC": "M8",
    "M9_CBI": "M9",
}

NORMALIZED_METRICS = list(RAW_TO_NORMALIZED_METRICS.values())


def normalize_processed_metrics(
    metrics_df: pd.DataFrame,
    metric_map: Mapping[str, str] = RAW_TO_NORMALIZED_METRICS,
    lower_quantile: float = 0.05,
    upper_quantile: float = 0.95,
) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    """Scale raw processed metrics into the normalized fuzzy universe [0, 1].

    Each raw metric is winsorized between ``lower_quantile`` and
    ``upper_quantile`` and then min-max scaled into ``[0, 1]``.
    """
    missing = set(metric_map) - set(metrics_df.columns)
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(f"metrics_df is missing raw metric columns: {missing_text}")

    normalized = metrics_df.copy()
    summary: dict[str, dict[str, float]] = {}

    for raw_name, normalized_name in metric_map.items():
        series = pd.to_numeric(metrics_df[raw_name], errors="coerce")
        finite = series[series.notna()]
        if finite.empty:
            raise ValueError(f"metric column {raw_name} contains no finite values")

        low = float(finite.quantile(lower_quantile))
        high = float(finite.quantile(upper_quantile))
        if high <= low:
            low = float(finite.min())
            high = float(finite.max())
        if high <= low:
            normalized[normalized_name] = 0.5
            summary[normalized_name] = {
                "source_metric": raw_name,
                "low": low,
                "high": high,
            }
            continue

        clipped = series.clip(lower=low, upper=high)
        normalized[normalized_name] = ((clipped - low) / (high - low)).clip(0.0, 1.0)
        summary[normalized_name] = {
            "source_metric": raw_name,
            "low": low,
            "high": high,
        }

    return normalized, summary


def run_real_data_fuzzy_processing(
    metrics_df: pd.DataFrame,
    id_columns: Sequence[str] = (
        "player_id",
        "player_name",
        "team_id",
        "team_name",
        "match_id",
        "minutes",
        "starting_position_name",
    ),
    score_column: str = "obpi",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Normalize raw metric outputs, fit the fuzzy engine, and score all rows."""
    normalized_df, normalization_summary = normalize_processed_metrics(metrics_df)
    scoring_input = normalized_df[list(id_columns) + NORMALIZED_METRICS].copy()
    engine = fit_fuzzy_engine(scoring_input, metric_names=NORMALIZED_METRICS)
    scored = score_metrics_dataframe(
        scoring_input,
        engine=engine,
        metric_names=NORMALIZED_METRICS,
        score_column=score_column,
    )
    metadata: dict[str, Any] = {
        "normalization": normalization_summary,
        "memberships": {
            metric_name: membership.to_dict()
            for metric_name, membership in engine.membership_functions.items()
        },
        "score_column": score_column,
    }
    return scored, metadata


def save_fuzzy_outputs(
    scored_df: pd.DataFrame,
    metadata: Mapping[str, Any],
    output_path: str | Path,
    metadata_path: str | Path,
) -> None:
    """Persist scored outputs and metadata summaries to disk."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    scored_df.to_parquet(output, index=False)

    metadata_output = Path(metadata_path)
    metadata_output.parent.mkdir(parents=True, exist_ok=True)
    metadata_output.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

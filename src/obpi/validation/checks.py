"""Outcome-validation checks for OBPI metric DataFrames."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger("obpi.validation")

METRIC_COLUMNS = [
    "M1_SC",
    "M2_OIRC",
    "M3_BRPC",
    "M4_OBR90",
    "M5_RBTL",
    "M6_RUP",
    "M7_SCI",
    "M8_LPC",
    "M9_CBI",
]

REQUIRED_COLUMNS = ["player_id", "match_id"] + METRIC_COLUMNS


def _check_finite(df: pd.DataFrame) -> list[str]:
    errors: list[str] = []
    for col in METRIC_COLUMNS:
        if not np.isfinite(df[col]).all():
            bad = df.loc[~np.isfinite(df[col]), col]
            errors.append(
                f"{col}: {len(bad)} non-finite values (inf/nan)"
            )
    return errors


def _check_range(df: pd.DataFrame) -> list[str]:
    errors: list[str] = []
    # All metrics are designed to be non-negative
    for col in METRIC_COLUMNS:
        if (df[col] < 0).any():
            bad = df.loc[df[col] < 0, col]
            errors.append(f"{col}: {len(bad)} negative values")
    return errors


def _check_schema(df: pd.DataFrame) -> list[str]:
    errors: list[str] = []
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        errors.append(f"Missing columns: {missing}")
    extra = [c for c in df.columns if c not in REQUIRED_COLUMNS + ["_schema_version"]]
    if extra:
        errors.append(f"Unexpected columns: {extra}")
    return errors


def validate(df: pd.DataFrame) -> dict[str, Any]:
    """Run all validation checks on a metrics DataFrame.

    Args:
        df: Output from :func:`~obpi.pipeline.compute_all_metrics`.

    Returns:
        Dict with ``valid`` (bool), ``errors`` (list[str]), and
        ``summary`` (dict of per-metric stats).
    """
    errors = _check_schema(df)
    if errors:
        return {"valid": False, "errors": errors, "summary": {}}

    errors.extend(_check_finite(df))
    errors.extend(_check_range(df))

    summary: dict[str, dict[str, float]] = {}
    for col in METRIC_COLUMNS:
        summary[col] = {
            "mean": float(df[col].mean()),
            "std": float(df[col].std()),
            "min": float(df[col].min()),
            "max": float(df[col].max()),
        }

    valid = not errors
    if not valid:
        for err in errors:
            logger.warning("Validation error: %s", err)
    else:
        logger.info("Validation passed for %d rows", len(df))

    return {"valid": valid, "errors": errors, "summary": summary}

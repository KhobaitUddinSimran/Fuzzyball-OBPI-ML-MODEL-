"""Data-calibrated fuzzy membership functions."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Union

import numpy as np
import pandas as pd

MembershipValue = Union[float, np.ndarray]
MembershipCallable = Callable[[MembershipValue], MembershipValue]


@dataclass(frozen=True)
class MembershipFunctions:
    """Low/medium/high membership functions for one metric."""

    low: MembershipCallable
    medium: MembershipCallable
    high: MembershipCallable
    percentiles: dict[str, float]


def trapmf(x: float | np.ndarray, points: list[float]) -> float | np.ndarray:
    """Evaluate a trapezoidal membership function."""

    a, b, c, d = points
    values = np.asarray(x, dtype=float)
    result = np.zeros_like(values, dtype=float)

    if b > a:
        rising = (values > a) & (values < b)
        result[rising] = (values[rising] - a) / (b - a)
    result[(values >= b) & (values <= c)] = 1.0
    if d > c:
        falling = (values > c) & (values < d)
        result[falling] = (d - values[falling]) / (d - c)

    result = np.clip(result, 0.0, 1.0)
    if np.isscalar(x):
        return float(result)
    return result


def build_membership_functions(values: np.ndarray) -> MembershipFunctions:
    """Build low/medium/high trapezoids from empirical metric values.

    The roadmap defines Low as ``trapmf(0, 0, P20, P50)``, Medium as
    ``trapmf(P20, P40, P60, P80)``, and High as ``trapmf(P50, P80, 1, 1)``.
    Input values are clipped into the normalized metric universe [0, 1].
    """

    clean_values = np.asarray(values, dtype=float)
    clean_values = clean_values[np.isfinite(clean_values)]
    if clean_values.size == 0:
        raise ValueError("values must contain at least one finite number")

    clipped = np.clip(clean_values, 0.0, 1.0)
    p20, p40, p50, p60, p80 = np.percentile(clipped, [20, 40, 50, 60, 80])
    percentiles = {
        "p20": float(p20),
        "p40": float(p40),
        "p50": float(p50),
        "p60": float(p60),
        "p80": float(p80),
    }

    return MembershipFunctions(
        low=lambda x: trapmf(x, [0.0, 0.0, percentiles["p20"], percentiles["p50"]]),
        medium=lambda x: trapmf(
            x,
            [
                percentiles["p20"],
                percentiles["p40"],
                percentiles["p60"],
                percentiles["p80"],
            ],
        ),
        high=lambda x: trapmf(x, [percentiles["p50"], percentiles["p80"], 1.0, 1.0]),
        percentiles=percentiles,
    )


def build_metric_memberships(
    metrics_df: pd.DataFrame,
    metric_names: list[str],
) -> dict[str, MembershipFunctions]:
    """Build membership functions for every OBPI metric column."""

    missing = set(metric_names) - set(metrics_df.columns)
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(f"metrics_df is missing metric columns: {missing_text}")

    return {
        metric_name: build_membership_functions(metrics_df[metric_name].to_numpy())
        for metric_name in metric_names
    }

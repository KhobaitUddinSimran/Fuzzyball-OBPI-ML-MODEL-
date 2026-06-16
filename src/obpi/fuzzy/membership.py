"""Membership functions for the OBPI fuzzy inference layer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd


FUZZY_LABELS = ("Low", "Medium", "High")


@dataclass(frozen=True)
class MembershipFunction:
    """Trapezoidal membership function."""

    points: tuple[float, float, float, float]

    def __call__(self, values: float | np.ndarray) -> float | np.ndarray:
        """Return membership degree(s) in [0, 1]."""
        return trapezoidal_membership(values, self.points)


def trapezoidal_membership(
    values: float | np.ndarray,
    points: tuple[float, float, float, float],
) -> float | np.ndarray:
    """Evaluate a trapezoidal membership function.

    The four points represent the left foot, left shoulder, right shoulder,
    and right foot of the trapezoid.
    """
    a, b, c, d = points
    if not a <= b <= c <= d:
        raise ValueError("trapezoid points must be ordered as a <= b <= c <= d")

    x = np.asarray(values, dtype=float)
    membership = np.zeros_like(x, dtype=float)

    if a == b:
        membership = np.where(x <= b, 1.0, membership)
    else:
        rising = (a < x) & (x < b)
        membership = np.where(rising, (x - a) / (b - a), membership)

    plateau = (b <= x) & (x <= c)
    membership = np.where(plateau, 1.0, membership)

    if c == d:
        membership = np.where(x >= c, 1.0, membership)
    else:
        falling = (c < x) & (x < d)
        membership = np.where(falling, (d - x) / (d - c), membership)

    membership = np.clip(membership, 0.0, 1.0)
    if np.isscalar(values):
        return float(membership)
    return membership


def build_membership_functions(
    values: np.ndarray | list[float] | None = None,
) -> dict[str, MembershipFunction]:
    """Build Low/Medium/High membership functions from metric values.

    When no data is supplied, stable normalized defaults are returned so the
    fuzzy engine can be used before Person 1's processed metrics exist.
    """
    if values is None:
        p20, p40, p50, p60, p80 = 0.2, 0.4, 0.5, 0.6, 0.8
    else:
        arr = np.asarray(values, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            raise ValueError("values must contain at least one finite number")
        p20, p40, p50, p60, p80 = np.percentile(arr, [20, 40, 50, 60, 80])

    return {
        "Low": MembershipFunction(_ordered_points((0.0, 0.0, p20, p50))),
        "Medium": MembershipFunction(_ordered_points((p20, p40, p60, p80))),
        "High": MembershipFunction(_ordered_points((p50, p80, 1.0, 1.0))),
    }


def build_metric_memberships(
    metric_values: Mapping[str, np.ndarray | list[float]] | None = None,
    metric_names: list[str] | tuple[str, ...] | None = None,
) -> dict[str, dict[str, MembershipFunction]]:
    """Build membership functions for each metric."""
    if metric_values is None:
        names = metric_names or tuple(f"M{i}" for i in range(1, 10))
        return {name: build_membership_functions() for name in names}

    return {
        metric_name: build_membership_functions(values)
        for metric_name, values in metric_values.items()
    }


def build_metric_memberships_from_dataframe(
    metrics_df: pd.DataFrame,
    metric_columns: list[str] | tuple[str, ...] | None = None,
) -> dict[str, dict[str, MembershipFunction]]:
    """Build per-metric memberships from DataFrame columns."""
    columns = tuple(metric_columns or tuple(f"M{i}" for i in range(1, 10)))
    missing = sorted(set(columns) - set(metrics_df.columns))
    if missing:
        raise ValueError(f"missing metric columns: {', '.join(missing)}")

    return build_metric_memberships(
        {column: metrics_df[column].to_numpy(dtype=float) for column in columns}
    )


def _ordered_points(points: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    """Keep degenerate percentile inputs valid for trapezoids."""
    ordered = np.maximum.accumulate(np.asarray(points, dtype=float))
    return tuple(float(np.clip(value, 0.0, 1.0)) for value in ordered)

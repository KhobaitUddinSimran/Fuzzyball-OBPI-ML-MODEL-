"""Mamdani-style fuzzy aggregation for OBPI scores."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from obpi.fuzzy.membership import MembershipFunctions

DEFAULT_METRICS = tuple(f"M{i}" for i in range(1, 10))


class FuzzyEngine:
    """Aggregate 9 normalized OBPI metrics into a score in [0, 1]."""

    def __init__(
        self,
        metric_names: Sequence[str] = DEFAULT_METRICS,
        membership_functions: Mapping[str, MembershipFunctions] | None = None,
        metric_weights: Mapping[str, float] | None = None,
        universe: np.ndarray | None = None,
    ) -> None:
        """Initialize the fuzzy aggregation engine."""
        self.metric_names = tuple(metric_names)
        if not self.metric_names:
            raise ValueError("metric_names must not be empty")

        self.universe = universe if universe is not None else np.linspace(0.0, 1.0, 101)
        self.membership_functions = membership_functions or {}
        self.metric_weights = self._normalize_weights(metric_weights)

    def compute(self, metrics: Mapping[str, float]) -> float:
        """Compute one corrected OBPI score from crisp metric inputs."""
        missing = set(self.metric_names) - set(metrics)
        if missing:
            missing_text = ", ".join(sorted(missing))
            raise ValueError(f"missing metric values: {missing_text}")

        weighted_outputs = []
        weights = []
        for metric_name in self.metric_names:
            crisp_value = float(np.clip(metrics[metric_name], 0.0, 1.0))
            raw_score = self._score_metric(metric_name, crisp_value)
            weighted_outputs.append(raw_score * self.metric_weights[metric_name])
            weights.append(self.metric_weights[metric_name])

        raw_obpi = float(np.sum(weighted_outputs) / np.sum(weights))
        return self.correct_range(raw_obpi)

    def compute_many(self, rows: Sequence[Mapping[str, float]]) -> np.ndarray:
        """Compute corrected OBPI scores for multiple metric dictionaries."""
        return np.asarray([self.compute(row) for row in rows], dtype=float)

    @staticmethod
    def correct_range(value: float, low: float = 0.15, high: float = 0.85) -> float:
        """Linearly expand centroid-compressed scores into [0, 1]."""
        if high <= low:
            raise ValueError("high must be greater than low")
        corrected = (value - low) / (high - low)
        return float(np.clip(corrected, 0.0, 1.0))

    def _score_metric(self, metric_name: str, crisp_value: float) -> float:
        membership = self.membership_functions.get(metric_name)
        if membership is None:
            return crisp_value

        low_degree = float(membership.low(crisp_value))
        medium_degree = float(membership.medium(crisp_value))
        high_degree = float(membership.high(crisp_value))
        denominator = low_degree + medium_degree + high_degree
        if denominator == 0.0:
            return crisp_value

        return (
            (low_degree * 0.15)
            + (medium_degree * 0.50)
            + (high_degree * 0.85)
        ) / denominator

    def _normalize_weights(
        self,
        metric_weights: Mapping[str, float] | None,
    ) -> dict[str, float]:
        if metric_weights is None:
            return dict.fromkeys(self.metric_names, 1.0)

        missing = set(self.metric_names) - set(metric_weights)
        if missing:
            missing_text = ", ".join(sorted(missing))
            raise ValueError(f"missing metric weights: {missing_text}")

        weights = {
            metric_name: float(metric_weights[metric_name])
            for metric_name in self.metric_names
        }
        if any(weight < 0.0 for weight in weights.values()):
            raise ValueError("metric weights must be non-negative")
        if sum(weights.values()) == 0.0:
            raise ValueError("at least one metric weight must be positive")
        return weights

"""Mamdani fuzzy inference engine for OBPI scores."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from obpi.fuzzy.membership import MembershipFunction, build_metric_memberships


DEFAULT_METRIC_NAMES = tuple(f"M{i}" for i in range(1, 10))


@dataclass(frozen=True)
class FuzzyRule:
    """Single-input rule mapping a metric label to an OBPI label."""

    metric_name: str
    input_label: str
    output_label: str


class FuzzyEngine:
    """Aggregate nine crisp OBPI metrics into a score in [0, 1].

    This implementation mirrors a Mamdani control system without requiring
    `scikit-fuzzy`: metric memberships fire SISO rules, consequents are clipped
    by rule strength, outputs are aggregated with max, then defuzzified by
    centroid.
    """

    def __init__(
        self,
        metric_names: list[str] | tuple[str, ...] | None = None,
        membership_funcs: Mapping[str, Mapping[str, MembershipFunction]] | None = None,
        metric_weights: Mapping[str, float] | None = None,
        universe: np.ndarray | None = None,
        raw_range: tuple[float, float] = (0.15, 0.85),
    ) -> None:
        self.metric_names = tuple(metric_names or DEFAULT_METRIC_NAMES)
        self.universe = np.asarray(
            universe if universe is not None else np.linspace(0.0, 1.0, 101),
            dtype=float,
        )
        if self.universe.ndim != 1 or self.universe.size < 2:
            raise ValueError("universe must be a one-dimensional array with at least 2 values")

        self.membership_funcs = dict(membership_funcs or build_metric_memberships(metric_names=self.metric_names))
        self.metric_weights = self._normalize_weights(metric_weights)
        self.raw_range = raw_range
        self.consequents = {
            "Low": MembershipFunction((0.0, 0.0, 0.10, 0.25)),
            "Medium": MembershipFunction((0.25, 0.40, 0.60, 0.75)),
            "High": MembershipFunction((0.75, 0.90, 1.0, 1.0)),
        }
        self.rules = self._build_rules()
        self._validate_configuration()

    def compute(self, metrics: Mapping[str, float] | list[float] | tuple[float, ...] | np.ndarray) -> float:
        """Compute a corrected OBPI score in [0, 1]."""
        metric_map = self._coerce_metrics(metrics)
        weighted_scores: list[float] = []
        weights: list[float] = []

        for metric_name in self.metric_names:
            aggregated = self._aggregate_metric_output(metric_name, metric_map[metric_name])
            raw_score = self.defuzzify(aggregated)
            weighted_scores.append(self.correct_range(raw_score))
            weights.append(self.metric_weights[metric_name])

        return float(np.average(weighted_scores, weights=weights))

    def compute_many(self, rows: list[Mapping[str, float]]) -> np.ndarray:
        """Compute OBPI scores for multiple metric rows."""
        return np.asarray([self.compute(row) for row in rows], dtype=float)

    def defuzzify(self, aggregated_membership: np.ndarray) -> float:
        """Defuzzify an aggregated output shape with the centroid method."""
        denominator = float(np.sum(aggregated_membership))
        if denominator == 0.0:
            return 0.0
        numerator = float(np.sum(self.universe * aggregated_membership))
        return numerator / denominator

    def correct_range(self, raw_score: float) -> float:
        """Linearly scale the centroid-compressed score into [0, 1]."""
        lower, upper = self.raw_range
        if lower >= upper:
            raise ValueError("raw_range must be ordered as lower < upper")
        corrected = (raw_score - lower) / (upper - lower)
        return float(np.clip(corrected, 0.0, 1.0))

    def _aggregate_metric_output(self, metric_name: str, metric_value: float) -> np.ndarray:
        """Aggregate the Low/Medium/High consequents for one metric."""
        aggregated = np.zeros_like(self.universe, dtype=float)
        for rule in self.rules:
            if rule.metric_name != metric_name:
                continue
            input_degree = self.membership_funcs[metric_name][rule.input_label](metric_value)
            consequent_shape = self.consequents[rule.output_label](self.universe)
            clipped_consequent = np.minimum(float(input_degree), consequent_shape)
            aggregated = np.maximum(aggregated, clipped_consequent)
        return aggregated

    def _build_rules(self) -> list[FuzzyRule]:
        rules: list[FuzzyRule] = []
        for metric_name in self.metric_names:
            rules.extend(
                [
                    FuzzyRule(metric_name, "Low", "Low"),
                    FuzzyRule(metric_name, "Medium", "Medium"),
                    FuzzyRule(metric_name, "High", "High"),
                ]
            )
        return rules

    def _coerce_metrics(
        self,
        metrics: Mapping[str, float] | list[float] | tuple[float, ...] | np.ndarray,
    ) -> dict[str, float]:
        if isinstance(metrics, Mapping):
            missing = sorted(set(self.metric_names) - set(metrics))
            if missing:
                raise ValueError(f"missing metrics: {', '.join(missing)}")
            metric_map = {name: float(metrics[name]) for name in self.metric_names}
        else:
            values = np.asarray(metrics, dtype=float)
            if values.shape != (len(self.metric_names),):
                raise ValueError(f"expected {len(self.metric_names)} metric values")
            metric_map = dict(zip(self.metric_names, (float(value) for value in values)))

        invalid = [
            name
            for name, value in metric_map.items()
            if not np.isfinite(value) or value < 0.0 or value > 1.0
        ]
        if invalid:
            raise ValueError(f"metrics must be finite values in [0, 1]: {', '.join(invalid)}")
        return metric_map

    def _normalize_weights(self, metric_weights: Mapping[str, float] | None) -> dict[str, float]:
        if metric_weights is None:
            return {name: 1.0 for name in self.metric_names}

        missing = sorted(set(self.metric_names) - set(metric_weights))
        if missing:
            raise ValueError(f"missing metric weights: {', '.join(missing)}")

        weights = {name: float(metric_weights[name]) for name in self.metric_names}
        if any(not np.isfinite(value) or value < 0.0 for value in weights.values()):
            raise ValueError("metric weights must be finite non-negative values")

        total_weight = sum(weights.values())
        if total_weight == 0.0:
            raise ValueError("at least one metric weight must be positive")
        return weights

    def _validate_configuration(self) -> None:
        missing = sorted(set(self.metric_names) - set(self.membership_funcs))
        if missing:
            raise ValueError(f"missing membership functions: {', '.join(missing)}")

        for metric_name in self.metric_names:
            labels = set(self.membership_funcs[metric_name])
            missing_labels = {"Low", "Medium", "High"} - labels
            if missing_labels:
                joined = ", ".join(sorted(missing_labels))
                raise ValueError(f"{metric_name} is missing membership labels: {joined}")

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from obpi.ml.ablation import run_ablation
from obpi.ml.correlation import (
    compare_benchmarks,
    cronbach_alpha,
    expert_correlation,
    orthogonal_variance_test,
    spearman_correlation,
)


def test_run_ablation_returns_one_row_per_metric() -> None:
    metrics_df = _make_synthetic_metrics(80)

    ablation_df = run_ablation(metrics_df, cv_splits=3, model_name="logistic")

    assert len(ablation_df) == 9
    assert set(ablation_df["removed_metric"]) == {f"M{i}" for i in range(1, 10)}
    assert "delta_accuracy" in ablation_df.columns


def test_spearman_correlation_detects_monotonic_relationship() -> None:
    scores = pd.Series(np.linspace(0, 1, 20))
    result = spearman_correlation(scores, scores * 10)

    assert result["spearman_rho"] == pytest.approx(1.0)
    assert result["n"] == 20


def test_compare_benchmarks_returns_one_row_per_benchmark() -> None:
    scores = pd.Series(np.linspace(0, 1, 20))
    benchmarks = pd.DataFrame(
        {
            "xThreat": scores,
            "benchmark_rating": scores[::-1].to_numpy(),
        }
    )

    comparison = compare_benchmarks(scores, benchmarks)

    assert set(comparison["benchmark"]) == {"xThreat", "benchmark_rating"}
    assert comparison.iloc[0]["spearman_rho"] >= comparison.iloc[-1]["spearman_rho"]


def test_orthogonal_variance_test_returns_pca_loadings() -> None:
    scores_df = pd.DataFrame(
        {
            "obpi": np.linspace(0, 1, 20),
            "xThreat": np.linspace(0, 1, 20) ** 2,
            "benchmark": np.linspace(1, 0, 20),
        }
    )

    result = orthogonal_variance_test(scores_df)

    assert result["n"] == 20
    assert len(result["explained_variance_ratio"]) == 3
    assert result["loadings"].shape == (3, 3)


def test_expert_correlation_and_cronbach_alpha() -> None:
    obpi = pd.Series(np.linspace(0, 1, 10))
    expert_ratings = pd.DataFrame(
        {
            "expert_1": obpi * 10,
            "expert_2": obpi * 10 + 0.1,
            "expert_3": obpi * 10 - 0.1,
        }
    )

    result = expert_correlation(obpi, expert_ratings.median(axis=1))
    alpha = cronbach_alpha(expert_ratings)

    assert result["spearman_rho"] == pytest.approx(1.0)
    assert alpha > 0.99


def _make_synthetic_metrics(n_rows: int) -> pd.DataFrame:
    values = np.linspace(0.05, 0.95, n_rows)
    rows = []
    for idx, value in enumerate(values):
        row = {"player_id": f"P{idx:03d}", "match_id": "SYN-W89", "minutes": 90}
        for metric_idx in range(1, 10):
            offset = (metric_idx - 5) * 0.015
            row[f"M{metric_idx}"] = float(np.clip(value + offset, 0.0, 1.0))
        row["obpi"] = float(value)
        rows.append(row)
    return pd.DataFrame(rows)

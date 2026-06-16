from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from obpi.fuzzy.scoring import fit_fuzzy_engine, score_metrics_dataframe
from obpi.ml.explainability import (
    compute_permutation_importance,
    get_metric_weights,
    save_metric_weights,
)
from obpi.ml.validation import prepare_labeled_data, train_logistic


def test_get_metric_weights_normalizes_shap_dataframe() -> None:
    shap_df = pd.DataFrame(
        {
            "M1": [1.0, -1.0, 2.0],
            "M2": [0.5, 0.5, -0.5],
            "M3": [0.0, 0.0, 0.0],
        }
    )

    weights = get_metric_weights(shap_df)

    assert sum(weights.values()) == pytest.approx(1.0)
    assert list(weights)[0] == "M1"
    assert weights["M3"] == 0.0


def test_get_metric_weights_rejects_all_zero_importance() -> None:
    with pytest.raises(ValueError, match="at least one positive"):
        get_metric_weights({"M1": 0.0, "M2": 0.0})


def test_permutation_importance_returns_metric_rows() -> None:
    metrics_df = _make_synthetic_scored_frame(80)
    x, y = prepare_labeled_data(metrics_df)
    grid = train_logistic(x, y, cv_splits=3)

    importance_df = compute_permutation_importance(
        grid.best_estimator_,
        x,
        y,
        n_repeats=3,
    )

    assert list(importance_df.columns) == [
        "metric",
        "importance_mean",
        "importance_std",
    ]
    assert set(importance_df["metric"]) == {f"M{i}" for i in range(1, 10)}


def test_metric_weights_feed_back_into_fuzzy_engine() -> None:
    metrics_df = _make_synthetic_scored_frame(20).drop(columns=["obpi"])
    weights = {f"M{i}": 1.0 for i in range(1, 10)}
    weights["M1"] = 10.0
    weighted_engine = fit_fuzzy_engine(metrics_df, metric_weights=weights)
    uniform_engine = fit_fuzzy_engine(metrics_df)

    weighted_score = score_metrics_dataframe(
        metrics_df,
        engine=weighted_engine,
    )["obpi"].iloc[-1]
    uniform_score = score_metrics_dataframe(
        metrics_df,
        engine=uniform_engine,
    )["obpi"].iloc[-1]

    assert weighted_score >= uniform_score


def test_save_metric_weights_writes_json(tmp_path) -> None:
    output_path = tmp_path / "metric_weights.json"
    weights = {"M1": 0.75, "M2": 0.25}

    save_metric_weights(weights, output_path)

    assert json.loads(output_path.read_text()) == weights


def _make_synthetic_scored_frame(n_rows: int) -> pd.DataFrame:
    values = np.linspace(0.05, 0.95, n_rows)
    rows = []
    for idx, value in enumerate(values):
        row = {
            "player_id": f"P{idx:03d}",
            "match_id": "SYN-W7",
            "minutes": 90,
        }
        for metric_idx in range(1, 10):
            offset = (metric_idx - 5) * 0.015
            row[f"M{metric_idx}"] = float(np.clip(value + offset, 0.0, 1.0))
        row["obpi"] = float(value)
        rows.append(row)
    return pd.DataFrame(rows)

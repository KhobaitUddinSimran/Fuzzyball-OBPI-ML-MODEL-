from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from obpi.fuzzy.scoring import fit_fuzzy_engine, score_metrics_dataframe
from obpi.ml.explainability import (
    ExplainabilityResult,
    compute_permutation_importance,
    get_metric_weights,
    run_explainability,
    save_explainability_results,
    save_metric_weights,
)
from obpi.ml.validation import prepare_labeled_data, prepare_training_frame, train_logistic


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


def test_run_explainability_returns_ranked_artifacts() -> None:
    metrics_df = _make_synthetic_scored_frame(80)
    prepared = prepare_training_frame(metrics_df, cv_splits=4)

    result = run_explainability(
        prepared.prepared_df,
        cv_splits=4,
        prefer_model="logistic",
        include_xgboost=False,
        permutation_repeats=3,
    )

    assert isinstance(result, ExplainabilityResult)
    assert result.model_name == "logistic"
    assert result.shap_values is None
    assert result.report["shap_available"] is False
    assert set(result.metric_weights) == {f"M{i}" for i in range(1, 10)}
    assert sum(result.metric_weights.values()) == pytest.approx(1.0)
    assert list(result.permutation_importance.columns) == [
        "metric",
        "importance_mean",
        "importance_std",
    ]


def test_save_explainability_results_writes_artifacts(tmp_path) -> None:
    permutation_df = pd.DataFrame(
        {
            "metric": ["M1", "M2"],
            "importance_mean": [0.2, 0.1],
            "importance_std": [0.01, 0.02],
        }
    )
    shap_df = pd.DataFrame({"M1": [0.1, -0.1], "M2": [0.05, -0.05]})
    result = ExplainabilityResult(
        model_name="xgboost",
        metric_weights={"M1": 2 / 3, "M2": 1 / 3},
        permutation_importance=permutation_df,
        shap_values=shap_df,
        report={"model_name": "xgboost", "shap_available": True},
    )

    weights_path = tmp_path / "metric_weights.json"
    permutation_path = tmp_path / "permutation_importance.csv"
    report_path = tmp_path / "explainability_report.json"
    shap_path = tmp_path / "shap_values.csv"

    save_explainability_results(
        result,
        weights_path=weights_path,
        permutation_path=permutation_path,
        report_path=report_path,
        shap_path=shap_path,
    )

    assert json.loads(weights_path.read_text()) == {"M1": 2 / 3, "M2": 1 / 3}
    assert pd.read_csv(permutation_path).shape == (2, 3)
    assert json.loads(report_path.read_text())["model_name"] == "xgboost"
    assert pd.read_csv(shap_path).shape == (2, 2)


def test_run_explainability_supports_xgboost_path(monkeypatch) -> None:
    metrics_df = _make_synthetic_scored_frame(80)
    prepared = prepare_training_frame(metrics_df, cv_splits=4)

    class DummyGrid:
        best_estimator_ = object()

    def fake_train_xgboost(x, y, cv_splits=5):
        return DummyGrid()

    def fake_compute_permutation_importance(*args, **kwargs):
        return pd.DataFrame(
            {
                "metric": [f"M{i}" for i in range(1, 10)],
                "importance_mean": np.linspace(0.9, 0.1, 9),
                "importance_std": np.zeros(9),
            }
        )

    def fake_compute_shap(model, x):
        return pd.DataFrame(np.ones((len(x), x.shape[1])), columns=x.columns)

    monkeypatch.setattr("obpi.ml.explainability.train_xgboost", fake_train_xgboost)
    monkeypatch.setattr(
        "obpi.ml.explainability.compute_permutation_importance",
        fake_compute_permutation_importance,
    )
    monkeypatch.setattr("obpi.ml.explainability.compute_shap", fake_compute_shap)

    result = run_explainability(
        prepared.prepared_df,
        cv_splits=4,
        include_xgboost=True,
        permutation_repeats=3,
    )

    assert result.model_name == "xgboost"
    assert result.shap_values is not None
    assert result.report["shap_available"] is True


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

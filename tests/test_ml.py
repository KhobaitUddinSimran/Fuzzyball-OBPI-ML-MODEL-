from __future__ import annotations

import builtins

import numpy as np
import pandas as pd
import pytest

from obpi.ml.validation import (
    create_labels,
    evaluate_estimator,
    prepare_labeled_data,
    train_logistic,
    train_svm,
    train_xgboost,
    validate,
)


def test_create_labels_keeps_top_and_bottom_quartiles() -> None:
    scores = pd.Series(np.linspace(0.0, 1.0, 100))
    labels = create_labels(scores)

    assert len(labels) == 50
    assert labels.value_counts().to_dict() == {0: 25, 1: 25}
    assert labels.loc[0] == 0
    assert labels.loc[99] == 1


def test_prepare_labeled_data_returns_extreme_quartile_rows() -> None:
    metrics_df = _make_synthetic_scored_frame(40)

    x, y = prepare_labeled_data(metrics_df)

    assert x.shape == (20, 9)
    assert len(y) == 20
    assert set(y.unique()) == {0, 1}


def test_train_logistic_and_evaluate() -> None:
    x, y = prepare_labeled_data(_make_synthetic_scored_frame(80))

    grid = train_logistic(x, y, cv_splits=3)
    result = evaluate_estimator(grid.best_estimator_, x, y, "logistic", cv_splits=3)

    assert result.model_name == "logistic"
    assert result.accuracy_mean >= 0.5
    assert result.roc_auc_mean >= 0.5
    assert result.n_splits == 3


def test_train_svm_and_evaluate() -> None:
    x, y = prepare_labeled_data(_make_synthetic_scored_frame(80))

    grid = train_svm(x, y, cv_splits=3)
    result = evaluate_estimator(grid.best_estimator_, x, y, "svm", cv_splits=3)

    assert result.model_name == "svm"
    assert result.accuracy_mean >= 0.5
    assert result.roc_auc_mean >= 0.5
    assert result.n_splits == 3


def test_validate_returns_model_metrics() -> None:
    metrics_df = _make_synthetic_scored_frame(80)

    result = validate(metrics_df, cv_splits=3)

    assert result["n_samples"] == 40
    assert result["n_rows"] == 80
    assert set(result["models"]) == {"svm", "logistic"}
    assert result["models"]["svm"]["accuracy_mean"] >= 0.5


def test_train_xgboost_reports_missing_optional_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    x, y = prepare_labeled_data(_make_synthetic_scored_frame(40))
    real_import = builtins.__import__

    def fake_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "xgboost":
            raise ImportError("No module named xgboost")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ImportError, match="xgboost is required"):
        train_xgboost(x, y, cv_splits=2)


def _make_synthetic_scored_frame(n_rows: int) -> pd.DataFrame:
    values = np.linspace(0.05, 0.95, n_rows)
    rows = []
    for idx, value in enumerate(values):
        row = {
            "player_id": f"P{idx:03d}",
            "match_id": "SYN-ML",
            "minutes": 90,
        }
        for metric_idx in range(1, 10):
            offset = (metric_idx - 5) * 0.015
            row[f"M{metric_idx}"] = float(np.clip(value + offset, 0.0, 1.0))
        row["obpi"] = float(value)
        rows.append(row)
    return pd.DataFrame(rows)

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from obpi.fuzzy import score_dataframe
from obpi.ml.validation import (
    create_labels,
    evaluate_estimator,
    prepare_labeled_data,
    train_logistic,
    train_svm,
    train_xgboost,
    validate,
)


class LabelConstructionTests(unittest.TestCase):
    def test_create_labels_keeps_top_and_bottom_quartiles(self) -> None:
        scores = pd.Series(np.linspace(0.0, 1.0, 100))

        labels = create_labels(scores)

        self.assertEqual(len(labels), 50)
        self.assertEqual(int((labels == 0).sum()), 25)
        self.assertEqual(int((labels == 1).sum()), 25)

    def test_create_labels_rejects_empty_scores(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one non-null"):
            create_labels(pd.Series(dtype=float))


class ValidationPipelineTests(unittest.TestCase):
    def test_prepare_labeled_data_returns_extreme_quartile_rows(self) -> None:
        metrics_df = _make_synthetic_scored_frame(40)

        X, y = prepare_labeled_data(metrics_df)

        self.assertEqual(X.shape, (20, 9))
        self.assertEqual(len(y), 20)
        self.assertEqual(set(y.unique()), {0, 1})

    def test_train_logistic_and_evaluate(self) -> None:
        X, y = prepare_labeled_data(_make_synthetic_scored_frame(80))

        grid = train_logistic(X, y, n_splits=3)
        result = evaluate_estimator(grid.best_estimator_, X, y, "logistic", n_splits=3)

        self.assertEqual(result.model_name, "logistic")
        self.assertGreaterEqual(result.accuracy_mean, 0.5)
        self.assertGreaterEqual(result.roc_auc_mean, 0.5)
        self.assertEqual(result.n_splits, 3)

    def test_train_svm_and_evaluate(self) -> None:
        X, y = prepare_labeled_data(_make_synthetic_scored_frame(80))

        grid = train_svm(X, y, n_splits=3)
        result = evaluate_estimator(grid.best_estimator_, X, y, "svm", n_splits=3)

        self.assertEqual(result.model_name, "svm")
        self.assertGreaterEqual(result.accuracy_mean, 0.5)
        self.assertGreaterEqual(result.roc_auc_mean, 0.5)
        self.assertEqual(result.n_splits, 3)

    def test_validate_returns_report_for_available_models(self) -> None:
        metrics_df = _make_synthetic_scored_frame(80)

        report = validate(metrics_df, n_splits=3)

        self.assertEqual(report["n_rows"], 80)
        self.assertEqual(report["n_labeled_rows"], 40)
        self.assertIn("logistic", report["models"])
        self.assertIn("svm", report["models"])

    def test_train_xgboost_reports_missing_optional_dependency(self) -> None:
        X, y = prepare_labeled_data(_make_synthetic_scored_frame(40))

        with self.assertRaisesRegex(ImportError, "xgboost is required"):
            train_xgboost(X, y, n_splits=2)


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
        rows.append(row)
    return score_dataframe(pd.DataFrame(rows), calibrate_memberships=False)


if __name__ == "__main__":
    unittest.main()


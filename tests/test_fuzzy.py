from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from obpi.fuzzy import FuzzyEngine, build_membership_functions, score_dataframe
from obpi.fuzzy.scoring import score_csv


class MembershipFunctionTests(unittest.TestCase):
    def test_build_membership_functions_from_percentiles(self) -> None:
        functions = build_membership_functions(np.linspace(0.0, 1.0, 101))

        self.assertEqual(set(functions), {"Low", "Medium", "High"})
        self.assertAlmostEqual(functions["Low"](0.0), 1.0)
        self.assertAlmostEqual(functions["Medium"](0.5), 1.0)
        self.assertAlmostEqual(functions["High"](1.0), 1.0)


class FuzzyEngineTests(unittest.TestCase):
    def test_single_player_score_is_in_expected_range(self) -> None:
        engine = FuzzyEngine()

        score = engine.compute([0.3, 0.5, 0.7, 0.4, 0.6, 0.8, 0.2, 0.9, 0.1])

        self.assertGreaterEqual(score, 0.15)
        self.assertLessEqual(score, 0.85)

    def test_all_zero_metrics_score_near_zero(self) -> None:
        engine = FuzzyEngine()

        score = engine.compute([0.0] * 9)

        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 0.05)

    def test_all_one_metrics_score_near_one(self) -> None:
        engine = FuzzyEngine()

        score = engine.compute([1.0] * 9)

        self.assertGreaterEqual(score, 0.95)
        self.assertLessEqual(score, 1.0)

    def test_mixed_metrics_score_in_middle(self) -> None:
        engine = FuzzyEngine()

        score = engine.compute([0.5] * 9)

        self.assertGreaterEqual(score, 0.4)
        self.assertLessEqual(score, 0.6)

    def test_mapping_input_uses_metric_names(self) -> None:
        engine = FuzzyEngine()
        metrics = {f"M{i}": 0.5 for i in range(1, 10)}

        score = engine.compute(metrics)

        self.assertGreaterEqual(score, 0.4)
        self.assertLessEqual(score, 0.6)

    def test_rejects_missing_metrics(self) -> None:
        engine = FuzzyEngine()

        with self.assertRaisesRegex(ValueError, "missing metrics"):
            engine.compute({"M1": 0.5})

    def test_rejects_out_of_range_metrics(self) -> None:
        engine = FuzzyEngine()

        with self.assertRaisesRegex(ValueError, "metrics must be finite values"):
            engine.compute([0.5] * 8 + [1.2])

    def test_compute_many_returns_one_score_per_row(self) -> None:
        engine = FuzzyEngine()
        rows = [
            {f"M{i}": 0.0 for i in range(1, 10)},
            {f"M{i}": 1.0 for i in range(1, 10)},
        ]

        scores = engine.compute_many(rows)

        self.assertEqual(scores.shape, (2,))
        self.assertTrue(np.all((0.0 <= scores) & (scores <= 1.0)))


class FuzzyScoringTests(unittest.TestCase):
    def test_score_dataframe_appends_obpi_score(self) -> None:
        metrics_df = pd.DataFrame(
            [
                {"player_id": "low", **{f"M{i}": 0.1 for i in range(1, 10)}},
                {"player_id": "high", **{f"M{i}": 0.9 for i in range(1, 10)}},
            ]
        )

        scored = score_dataframe(metrics_df)

        self.assertIn("obpi_score", scored.columns)
        self.assertEqual(len(scored), 2)
        self.assertTrue(scored["obpi_score"].between(0.0, 1.0).all())
        self.assertLess(
            scored.loc[scored["player_id"] == "low", "obpi_score"].iloc[0],
            scored.loc[scored["player_id"] == "high", "obpi_score"].iloc[0],
        )

    def test_score_dataframe_rejects_missing_metric_column(self) -> None:
        metrics_df = pd.DataFrame([{f"M{i}": 0.5 for i in range(1, 9)}])

        with self.assertRaisesRegex(ValueError, "missing metric columns"):
            score_dataframe(metrics_df)

    def test_score_csv_writes_output_file(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        input_path = repo_root / "data" / "sample" / "sample_metrics.csv"

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "sample_obpi_scores_test.csv"
            scored = score_csv(input_path, output_path)

            self.assertTrue(output_path.exists())
            self.assertIn("obpi_score", scored.columns)
            self.assertTrue(scored["obpi_score"].between(0.0, 1.0).all())


if __name__ == "__main__":
    unittest.main()

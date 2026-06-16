"""Tests for validation checks."""

import numpy as np
import pandas as pd
import pytest

from obpi.validation.checks import METRIC_COLUMNS, REQUIRED_COLUMNS, validate


class TestValidate:
    def test_valid_df_passes(self) -> None:
        df = pd.DataFrame(
            {
                "player_id": [1, 2],
                "match_id": [100, 100],
                **{col: [0.5, 0.5] for col in METRIC_COLUMNS},
            }
        )
        result = validate(df)
        assert result["valid"] is True
        assert result["errors"] == []
        assert set(result["summary"].keys()) == set(METRIC_COLUMNS)

    def test_missing_columns_fails(self) -> None:
        df = pd.DataFrame({"player_id": [1], "match_id": [100]})
        result = validate(df)
        assert result["valid"] is False
        assert any("Missing columns" in e for e in result["errors"])

    def test_non_finite_fails(self) -> None:
        df = pd.DataFrame(
            {
                "player_id": [1],
                "match_id": [100],
                **{col: [np.nan] for col in METRIC_COLUMNS},
            }
        )
        result = validate(df)
        assert result["valid"] is False
        assert any("non-finite" in e for e in result["errors"])

    def test_negative_values_fails(self) -> None:
        df = pd.DataFrame(
            {
                "player_id": [1],
                "match_id": [100],
                **{col: [-0.1] for col in METRIC_COLUMNS},
            }
        )
        result = validate(df)
        assert result["valid"] is False
        assert any("negative" in e for e in result["errors"])

    def test_summary_stats(self) -> None:
        df = pd.DataFrame(
            {
                "player_id": [1],
                "match_id": [100],
                **{col: [0.5] for col in METRIC_COLUMNS},
            }
        )
        result = validate(df)
        for col in METRIC_COLUMNS:
            assert result["summary"][col]["mean"] == pytest.approx(0.5)

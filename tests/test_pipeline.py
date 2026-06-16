"""Integration tests for the OBPI pipeline orchestrator."""

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from obpi.pipeline import (
    _extract_unique_players,
    compute_all_metrics,
    run_pipeline,
)
from obpi.utils.xt_model import XTModel


def _make_synthetic_events(match_id: int = 999999) -> pd.DataFrame:
    """Build a small events DataFrame mimicking StatsBomb schema."""
    rows: list[dict[str, Any]] = [
        {
            "id": 1,
            "match_id": match_id,
            "index": 1,
            "period": 1,
            "timestamp": "00:01:00.000",
            "minute": 1,
            "second": 0,
            "type": {"id": 30, "name": "Pass"},
            "possession": 1,
            "possession_team": {"id": 1, "name": "Home"},
            "team": {"id": 1, "name": "Home"},
            "player": {"id": 1001, "name": "Alice"},
            "position": {"id": 15, "name": "LCM"},
            "location": [60.0, 40.0],
            "under_pressure": False,
            "play_pattern": {"id": 1, "name": "Regular Play"},
        },
        {
            "id": 2,
            "match_id": match_id,
            "index": 2,
            "period": 1,
            "timestamp": "00:01:02.000",
            "minute": 1,
            "second": 2,
            "type": {"id": 42, "name": "Ball Receipt*"},
            "possession": 1,
            "possession_team": {"id": 1, "name": "Home"},
            "team": {"id": 1, "name": "Home"},
            "player": {"id": 1001, "name": "Alice"},
            "position": {"id": 15, "name": "LCM"},
            "location": [70.0, 40.0],
            "under_pressure": False,
            "play_pattern": {"id": 1, "name": "Regular Play"},
        },
        {
            "id": 3,
            "match_id": match_id,
            "index": 3,
            "period": 1,
            "timestamp": "00:02:00.000",
            "minute": 2,
            "second": 0,
            "type": {"id": 30, "name": "Pass"},
            "possession": 1,
            "possession_team": {"id": 1, "name": "Home"},
            "team": {"id": 1, "name": "Home"},
            "player": {"id": 1002, "name": "Bob"},
            "position": {"id": 10, "name": "RCM"},
            "location": [80.0, 40.0],
            "under_pressure": True,
            "play_pattern": {"id": 1, "name": "Regular Play"},
        },
        {
            "id": 4,
            "match_id": match_id,
            "index": 4,
            "period": 1,
            "timestamp": "00:02:05.000",
            "minute": 2,
            "second": 5,
            "type": {"id": 42, "name": "Ball Receipt*"},
            "possession": 1,
            "possession_team": {"id": 1, "name": "Home"},
            "team": {"id": 1, "name": "Home"},
            "player": {"id": 1002, "name": "Bob"},
            "position": {"id": 10, "name": "RCM"},
            "location": [85.0, 40.0],
            "under_pressure": True,
            "play_pattern": {"id": 1, "name": "Regular Play"},
        },
    ]
    return pd.DataFrame(rows)


def _make_synthetic_frames() -> list[dict[str, Any]]:
    """Return two synthetic 360 freeze frames."""
    return [
        {
            "event_uuid": "evt-1",
            "match_id": 999999,
            "visible_area": [0, 0, 120, 80],
            "freeze_frame": [
                {"teammate": True, "actor": True, "keeper": False, "location": [60.0, 40.0]},
                {"teammate": True, "actor": False, "keeper": False, "location": [55.0, 35.0]},
                {"teammate": True, "actor": False, "keeper": False, "location": [62.0, 45.0]},
                {"teammate": False, "actor": False, "keeper": False, "location": [65.0, 42.0]},
                {"teammate": False, "actor": False, "keeper": False, "location": [68.0, 38.0]},
            ],
        },
        {
            "event_uuid": "evt-2",
            "match_id": 999999,
            "visible_area": [0, 0, 120, 80],
            "freeze_frame": [
                {"teammate": True, "actor": True, "keeper": False, "location": [60.0, 40.0]},
                {"teammate": True, "actor": False, "keeper": False, "location": [58.0, 36.0]},
                {"teammate": True, "actor": False, "keeper": False, "location": [64.0, 46.0]},
                {"teammate": False, "actor": False, "keeper": False, "location": [70.0, 42.0]},
                {"teammate": False, "actor": False, "keeper": False, "location": [72.0, 39.0]},
            ],
        },
    ]


class TestExtractUniquePlayers:
    """Unit tests for player extraction helper."""

    def test_extracts_sorted_ids(self) -> None:
        events = _make_synthetic_events()
        players = _extract_unique_players(events)
        assert players == [1001, 1002]

    def test_empty_events_returns_empty(self) -> None:
        events = pd.DataFrame()
        players = _extract_unique_players(events)
        assert players == []


class TestComputeAllMetrics:
    """End-to-end pipeline tests with mocked loader."""

    @patch("obpi.pipeline.StatsBombLoader")
    def test_returns_expected_columns(self, mock_loader_cls: MagicMock) -> None:
        mock_loader = MagicMock()
        mock_loader.get_events.return_value = _make_synthetic_events()
        mock_loader.get_freeze_frames.return_value = _make_synthetic_frames()
        mock_loader_cls.return_value = mock_loader

        xt_model = XTModel()
        df = compute_all_metrics(match_id=999999, xt_model=xt_model)

        expected_cols = [
            "player_id",
            "match_id",
            "M1_SC",
            "M2_OIRC",
            "M3_BRPC",
            "M4_OBR90",
            "M5_RBTL",
            "M6_RUP",
            "M7_SCI",
            "M8_LPC",
            "M9_CBI",
        ]
        assert list(df.columns) == expected_cols
        assert len(df) == 2  # Two unique players
        assert df["match_id"].unique().tolist() == [999999]

    @patch("obpi.pipeline.StatsBombLoader")
    def test_all_values_finite(self, mock_loader_cls: MagicMock) -> None:
        mock_loader = MagicMock()
        mock_loader.get_events.return_value = _make_synthetic_events()
        mock_loader.get_freeze_frames.return_value = _make_synthetic_frames()
        mock_loader_cls.return_value = mock_loader

        xt_model = XTModel()
        df = compute_all_metrics(match_id=999999, xt_model=xt_model)

        metric_cols = [c for c in df.columns if c.startswith("M")]
        for col in metric_cols:
            finite = df[col].apply(
                lambda x: isinstance(x, (int, float)) and x == x
            ).all()
            assert finite, f"Column {col} contains non-finite values"

    @patch("obpi.pipeline.StatsBombLoader")
    def test_no_frames_sets_spatial_temporal_to_zero(self, mock_loader_cls: MagicMock) -> None:
        mock_loader = MagicMock()
        mock_loader.get_events.return_value = _make_synthetic_events()
        mock_loader.get_freeze_frames.return_value = []
        mock_loader_cls.return_value = mock_loader

        xt_model = XTModel()
        df = compute_all_metrics(match_id=999999, xt_model=xt_model)

        assert (df["M7_SCI"] == 0.0).all()
        assert (df["M1_SC"] == 0.0).all()
        assert (df["M3_BRPC"] == 0.0).all()
        assert (df["M9_CBI"] == 0.0).all()


class TestRunPipeline:
    """Tests for the caching wrapper."""

    @patch("obpi.pipeline.StatsBombLoader")
    def test_caches_to_parquet(self, mock_loader_cls: MagicMock, tmp_path: Path) -> None:
        mock_loader = MagicMock()
        mock_loader.get_events.return_value = _make_synthetic_events()
        mock_loader.get_freeze_frames.return_value = _make_synthetic_frames()
        mock_loader_cls.return_value = mock_loader

        output_dir = tmp_path / "processed"
        df = run_pipeline(match_id=999999, output_dir=str(output_dir))

        assert len(df) == 2
        cached = output_dir / "999999_metrics.parquet"
        assert cached.exists()

    @patch("obpi.pipeline.StatsBombLoader")
    def test_reads_existing_cache(self, mock_loader_cls: MagicMock, tmp_path: Path) -> None:
        output_dir = tmp_path / "processed"
        output_dir.mkdir(parents=True)
        cached = output_dir / "999999_metrics.parquet"
        cached_df = pd.DataFrame(
            {
                "player_id": [1001],
                "match_id": [999999],
                "M1_SC": [0.5],
                "M2_OIRC": [0.5],
                "M3_BRPC": [0.5],
                "M4_OBR90": [0.5],
                "M5_RBTL": [0.5],
                "M6_RUP": [0.5],
                "M7_SCI": [0.5],
                "M8_LPC": [0.5],
                "M9_CBI": [0.5],
                "_schema_version": [2],
            }
        )
        cached_df.to_parquet(cached, index=False)

        df = run_pipeline(match_id=999999, output_dir=str(output_dir))
        assert len(df) == 1
        assert df.iloc[0]["M1_SC"] == pytest.approx(0.5)

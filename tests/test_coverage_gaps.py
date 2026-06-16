"""Targeted tests to cover previously missed lines."""

import logging
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from obpi.config.loader import Config, load_config
from obpi.data.loader import StatsBombLoader
from obpi.data.preprocessor import DeltaTGate
from obpi.metrics.movement import _exclude_set_pieces, compute_obr90, compute_oirc
from obpi.metrics.receiving import compute_brpc, compute_rup, get_receipt_events
from obpi.metrics.spatial import compute_sc, compute_sci
from obpi.metrics.temporal import _get_player_id, _is_pass, _is_receipt, compute_cbi, compute_lpc
from obpi.pipeline import _sc_from_frames, _sci_from_frames, run_pipeline
from obpi.utils.geometry import run_directness, to_goal_vector, voronoi_areas
from obpi.utils.kinematics import detect_runs
from obpi.utils.logger import setup_logging
from obpi.utils.units import meters_per_second_to_kmh as mps_to_kmh
from obpi.utils.xt_model import XTModel
from obpi.validation.checks import METRIC_COLUMNS, validate


class TestConfigLoader:
    def test_attribute_error_on_missing_key(self) -> None:
        cfg = Config({"a": 1})
        with pytest.raises(AttributeError):
            _ = cfg.nonexistent

    def test_file_not_found(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/path.yaml")

    def test_custom_path(self) -> None:
        path = Path(__file__).resolve().parents[1] / "config" / "default.yaml"
        cfg = load_config(str(path))
        assert "movement" in cfg

    def test_none_mapping(self) -> None:
        cfg = Config(None)
        assert dict(cfg) == {}


class TestDataLoader:
    def test_api_tier_competitions(self) -> None:
        loader = StatsBombLoader(tier="api")
        with pytest.raises(NotImplementedError):
            loader.get_competitions()

    def test_api_tier_matches(self) -> None:
        loader = StatsBombLoader(tier="api")
        with pytest.raises(NotImplementedError):
            loader.get_matches(55, 43)

    def test_api_tier_events(self) -> None:
        loader = StatsBombLoader(tier="api")
        with pytest.raises(NotImplementedError):
            loader.get_events(3794687)

    def test_api_tier_frames(self) -> None:
        loader = StatsBombLoader(tier="api")
        with pytest.raises(NotImplementedError):
            loader.get_freeze_frames(3794687)

    def test_invalid_tier(self) -> None:
        with pytest.raises(ValueError, match="tier must be"):
            StatsBombLoader(tier="invalid")

    def test_matches_with_360_no_column(self) -> None:
        loader = StatsBombLoader(tier="open")
        with patch.object(loader, "get_matches", return_value=pd.DataFrame({"match_id": [1, 2]})):
            df = loader.matches_with_360(55, 43)
            assert len(df) == 2


class TestPreprocessor:
    def test_deltat_gate_mismatched_lengths(self) -> None:
        gate = DeltaTGate()
        locs = np.array([[0.0, 0.0], [1.0, 1.0]])
        ts = np.array([0.0])
        with pytest.raises(ValueError, match="same length"):
            gate.filter_pairs(locs, ts)

    def test_deltat_gate_single_point(self) -> None:
        gate = DeltaTGate()
        locs = np.array([[0.0, 0.0]])
        ts = np.array([0.0])
        fl, ft, fd = gate.filter_pairs(locs, ts)
        assert len(fl) == 1
        assert len(fd) == 0


class TestMovementMetrics:
    def test_exclude_set_pieces_string_type(self) -> None:
        df = pd.DataFrame(
            {
                "play_pattern": [
                    "From Corner",
                    "Regular Play",
                    "From Free Kick",
                ]
            }
        )
        result = _exclude_set_pieces(df)
        assert len(result) == 1
        assert result.iloc[0]["play_pattern"] == "Regular Play"

    def test_compute_obr90_no_matching_player(self) -> None:
        events = pd.DataFrame(
            {
                "timestamp": ["00:01:00.000"],
                "location": [[60.0, 40.0]],
                "period": [1],
                "play_pattern": [{"name": "Regular Play"}],
                "player": [{"id": 2}],
            }
        )
        result = compute_obr90(events, player_id=1)
        assert result == 0.0

    def test_compute_oirc_no_runs(self) -> None:
        events = pd.DataFrame(
            {
                "timestamp": ["00:01:00.000"],
                "location": [[60.0, 40.0]],
                "period": [1],
                "play_pattern": [{"name": "Regular Play"}],
                "player": [{"id": 1}],
            }
        )
        result = compute_oirc(events, player_id=1)
        assert result == 0.0


class TestReceivingMetrics:
    def test_get_receipt_events_string_type(self) -> None:
        df = pd.DataFrame(
            {
                "type": ["Ball Receipt*", "Pass", "Ball Receipt"],
                "player": [{"id": 1}, {"id": 1}, {"id": 2}],
            }
        )
        result = get_receipt_events(df, player_id=1)
        assert len(result) == 1  # Only row 0 matches (player 1 + receipt)

    def test_get_receipt_events_missing_player(self) -> None:
        df = pd.DataFrame(
            {
                "type": [{"name": "Ball Receipt*"}],
                "player": [None],
            }
        )
        result = get_receipt_events(df, player_id=1)
        assert result.empty

    def test_compute_rup_no_receipts(self) -> None:
        df = pd.DataFrame({"type": [{"name": "Pass"}], "player": [{"id": 1}]})
        result = compute_rup(df, player_id=1)
        assert result == 0.0

    def test_compute_brpc_more_receipts_than_frames(self) -> None:
        events = pd.DataFrame(
            {
                "type": [{"name": "Ball Receipt*"}, {"name": "Ball Receipt*"}],
                "player": [{"id": 1}, {"id": 1}],
                "location": [[60.0, 40.0], [70.0, 40.0]],
            }
        )
        frames = [
            {"freeze_frame": [{"teammate": False, "location": [90.0, 40.0]}]}
        ]
        result = compute_brpc(events, frames, player_id=1, pressure_threshold=5.0)
        # 2 receipts, 1 frame → only 1 qualifies, BRPC = 1/2
        assert result == pytest.approx(0.5)


class TestSpatialMetrics:
    def test_sci_fewer_than_three_teammates(self) -> None:
        before = {"freeze_frame": [{"teammate": True, "location": [60.0, 40.0]}]}
        after = {"freeze_frame": [{"teammate": True, "location": [61.0, 41.0]}]}
        result = compute_sci([before], [after])
        assert result == 0.0

    def test_sc_fewer_than_three_opponents(self) -> None:
        before = {"freeze_frame": [{"teammate": True, "location": [60.0, 40.0]}]}
        after = {"freeze_frame": [{"teammate": True, "location": [60.0, 40.0]}]}
        result = compute_sc([before], [after], player_location=[60.0, 40.0])
        assert result == 0.0


class TestTemporalMetrics:
    def test_is_receipt_string(self) -> None:
        row = pd.Series({"type": "Ball Receipt*"})
        assert _is_receipt(row) is True

    def test_is_pass_string(self) -> None:
        row = pd.Series({"type": "Pass"})
        assert _is_pass(row) is True

    def test_get_player_id_none(self) -> None:
        row = pd.Series({"player": None})
        assert _get_player_id(row) is None

    def test_compute_lpc_player_events_lt_two(self) -> None:
        events = pd.DataFrame(
            {
                "type": [{"name": "Pass"}],
                "player": [{"id": 1}],
                "timestamp": ["00:01:00.000"],
                "period": [1],
            }
        )
        xt_model = XTModel()
        result = compute_lpc(events, player_id=1, xt_model=xt_model)
        assert result == 0.0

    def test_compute_lpc_loc_none(self) -> None:
        events = pd.DataFrame(
            {
                "type": [{"name": "Ball Receipt*"}, {"name": "Pass"}],
                "player": [{"id": 1}, {"id": 1}],
                "timestamp": ["00:01:00.000", "00:01:03.000"],
                "period": [1, 1],
                "location": [None, [70.0, 40.0]],
                "pass": [{}, {"end_location": [80.0, 40.0]}],
            }
        )
        xt_model = XTModel()
        result = compute_lpc(events, player_id=1, xt_model=xt_model, min_dt=1.2)
        assert result == 0.0

    def test_compute_cbi_empty_events(self) -> None:
        result = compute_cbi(pd.DataFrame(), [], player_id=1)
        assert result == 0.0

    def test_compute_cbi_no_receipts_for_player(self) -> None:
        events = pd.DataFrame(
            {
                "type": [{"name": "Pass"}],
                "player": [{"id": 2}],
                "timestamp": ["00:01:00.000"],
                "period": [1],
            }
        )
        result = compute_cbi(events, [{}], player_id=1)
        assert result == 0.0


class TestPipelineGuards:
    def test_sci_from_frames_lt_two(self) -> None:
        result = _sci_from_frames([{"freeze_frame": []}])
        assert result == 0.0

    def test_sc_from_frames_lt_two(self) -> None:
        result = _sc_from_frames([{"freeze_frame": []}], [60.0, 40.0])
        assert result == 0.0

    @patch("obpi.pipeline.StatsBombLoader")
    def test_run_pipeline_stale_cache(self, mock_loader_cls: MagicMock, tmp_path: Path) -> None:
        output_dir = tmp_path / "processed"
        output_dir.mkdir(parents=True)
        cached = output_dir / "999999_metrics.parquet"
        cached_df = pd.DataFrame(
            {
                "player_id": [1001],
                "match_id": [999999],
                **{col: [0.5] for col in METRIC_COLUMNS},
                "_schema_version": [1],  # stale version
            }
        )
        cached_df.to_parquet(cached, index=False)

        mock_loader = MagicMock()
        mock_loader.get_events.return_value = pd.DataFrame(
            {
                "type": [{"name": "Pass"}],
                "player": [{"id": 1}],
                "timestamp": ["00:01:00.000"],
                "period": [1],
                "location": [[60.0, 40.0]],
                "play_pattern": [{"name": "Regular Play"}],
            }
        )
        mock_loader.get_freeze_frames.return_value = []
        mock_loader_cls.return_value = mock_loader

        df = run_pipeline(match_id=999999, output_dir=str(output_dir))
        assert "_schema_version" not in df.columns


class TestGeometry:
    def test_run_directness_zero_norm(self) -> None:
        v_run = np.array([0.0, 0.0], dtype=np.float64)
        v_goal = np.array([10.0, 0.0], dtype=np.float64)
        assert run_directness(v_run, v_goal) == 0.0

    def test_voronoi_areas_empty(self) -> None:
        points = np.array([], dtype=np.float64).reshape(0, 2)
        result = voronoi_areas(points)
        assert result == []

    def test_to_goal_vector(self) -> None:
        vec = to_goal_vector([60.0, 40.0])
        assert vec[0] == pytest.approx(60.0)
        assert vec[1] == pytest.approx(0.0)


class TestKinematics:
    def test_detect_runs_extends_to_final_row(self) -> None:
        """A run that reaches the last row should still be captured."""
        df = pd.DataFrame(
            {
                "x": [0.0, 10.0, 20.0],
                "y": [0.0, 0.0, 0.0],
                "dt": [1.0, 1.0, 1.0],
                "velocity": [3.0, 3.0, 3.0],
            }
        )
        runs = detect_runs(df, v_threshold=2.5, duration_threshold=0.4)
        assert len(runs) == 1
        assert runs[0]["end_idx"] == 2


class TestLogger:
    def test_setup_logging_changes_level(self) -> None:
        setup_logging(level=logging.DEBUG)
        root = logging.getLogger("obpi")
        assert root.level == logging.DEBUG


class TestUnits:
    def test_mps_to_kmh(self) -> None:
        assert mps_to_kmh(1.0) == pytest.approx(3.6)


class TestXTModel:
    def test_default_grid(self) -> None:
        model = XTModel()
        grid = model._default_grid()
        assert grid.shape == (8, 12)
        assert grid.dtype == np.float64


class TestValidation:
    def test_validation_warning_logged(self, caplog: Any) -> None:
        df = pd.DataFrame(
            {
                "player_id": [1],
                "match_id": [100],
                "M1_SC": [-0.1],
                "M2_OIRC": [0.5],
                "M3_BRPC": [0.5],
                "M4_OBR90": [0.5],
                "M5_RBTL": [0.5],
                "M6_RUP": [0.5],
                "M7_SCI": [0.5],
                "M8_LPC": [0.5],
                "M9_CBI": [0.5],
            }
        )
        with caplog.at_level("WARNING"):
            result = validate(df)
        assert result["valid"] is False
        assert any("negative" in m for m in result["errors"])

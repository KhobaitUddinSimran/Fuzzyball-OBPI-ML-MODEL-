"""Tests for spatial (M7, M1) and temporal (M8, M9) metrics, plus xT model."""

import numpy as np
import pandas as pd
import pytest

from obpi.metrics.spatial import compute_sc, compute_sci
from obpi.metrics.temporal import compute_cbi, compute_lpc
from obpi.utils.geometry import voronoi_areas
from obpi.utils.xt_model import XTModel


class TestXTModel:
    """Unit tests for the expected-threat grid."""

    def test_default_grid_shape(self) -> None:
        model = XTModel()
        assert model.get_grid().shape == (8, 12)

    def test_pass_to_higher_xt_zone_returns_positive_delta(self) -> None:
        model = XTModel()
        origin = [10.0, 40.0]   # left side (low xT)
        destination = [70.0, 40.0]  # right side (high xT)
        dxt = model.delta_xt(origin, destination)
        assert dxt > 0.0

    def test_pass_backwards_returns_negative_delta(self) -> None:
        model = XTModel()
        origin = [70.0, 40.0]
        destination = [10.0, 40.0]
        dxt = model.delta_xt(origin, destination)
        assert dxt < 0.0


class TestVoronoiAreas:
    """Unit tests for Voronoi cell area computation."""

    def test_symmetric_four_players(self) -> None:
        points = np.array(
            [[40.0, 40.0], [80.0, 40.0], [40.0, 40.0], [80.0, 40.0]],
            dtype=np.float64,
        )
        areas = voronoi_areas(points)
        assert len(areas) == 4
        assert all(a > 0 for a in areas)

    def test_fewer_than_three_returns_zeros(self) -> None:
        points = np.array([[60.0, 40.0], [70.0, 40.0]], dtype=np.float64)
        areas = voronoi_areas(points)
        assert areas == [0.0, 0.0]


class TestSci:
    """End-to-end tests for M7 Space Control Improvement."""

    def _make_frame(
        self, teammate_locs: list[list[float]], opponent_locs: list[list[float]]
    ) -> dict:
        freeze = []
        for loc in teammate_locs:
            freeze.append({"teammate": True, "actor": False, "location": loc})
        for loc in opponent_locs:
            freeze.append({"teammate": False, "actor": False, "location": loc})
        return {"freeze_frame": freeze}

    def test_teammates_gain_area(self) -> None:
        """Synthetic: teammates spread out → area gain is positive."""
        before = self._make_frame(
            teammate_locs=[[50.0, 30.0], [52.0, 40.0], [54.0, 50.0]],
            opponent_locs=[[70.0, 40.0]],
        )
        after = self._make_frame(
            teammate_locs=[[50.0, 30.0], [60.0, 40.0], [70.0, 50.0]],
            opponent_locs=[[70.0, 40.0]],
        )
        sci = compute_sci([before], [after])
        assert sci > 0.0

    def test_empty_frames_returns_zero(self) -> None:
        sci = compute_sci([], [])
        assert sci == pytest.approx(0.0)


class TestSc:
    """End-to-end tests for M1 Screen Count."""

    def _make_frame(self, opponent_locs: list[list[float]]) -> dict:
        freeze = [
            {"teammate": True, "actor": False, "location": [60.0, 40.0]},
        ]
        for loc in opponent_locs:
            freeze.append({"teammate": False, "actor": False, "location": loc})
        return {"freeze_frame": freeze}

    def test_local_shift_exceeds_global(self) -> None:
        """Local defender shifts 4m, global shifts 2.25m → adjusted = 1.75m > 1.5m."""
        before = self._make_frame(
            opponent_locs=[[62.0, 40.0], [90.0, 40.0]],
        )
        after = self._make_frame(
            opponent_locs=[[66.0, 40.0], [90.5, 40.0]],
        )
        sc = compute_sc(
            [before],
            [after],
            player_location=[60.0, 40.0],
            box_size=(15.0, 15.0),
            threshold=1.5,
        )
        assert sc == pytest.approx(1.0)

    def test_no_opponents_returns_zero(self) -> None:
        before = self._make_frame(opponent_locs=[])
        after = self._make_frame(opponent_locs=[])
        sc = compute_sc([before], [after], player_location=[60.0, 40.0])
        assert sc == pytest.approx(0.0)


class TestLpc:
    """End-to-end tests for M8 Layoff Pause Coefficient."""

    def _make_events(self, rows: list[dict]) -> pd.DataFrame:
        return pd.DataFrame(rows)

    def test_pause_improves_xt(self) -> None:
        """Receipt → slow pass to higher-xT zone counts as successful pause."""
        rows = [
            {
                "type": {"name": "Ball Receipt*"},
                "player": {"id": 1},
                "location": [60.0, 40.0],
                "timestamp": "00:10:00.000",
                "period": 1,
                "pass": {},
            },
            {
                "type": {"name": "Pass"},
                "player": {"id": 1},
                "location": [60.0, 40.0],
                "timestamp": "00:10:02.000",
                "period": 1,
                "pass": {"end_location": [80.0, 40.0]},
            },
        ]
        events = self._make_events(rows)
        synthetic_grid = np.linspace(0.01, 0.30, 12) * np.ones((8, 1))
        xt_model = XTModel(grid=synthetic_grid)
        lpc = compute_lpc(events, player_id=1, xt_model=xt_model, min_dt=1.2, max_vel=0.5)
        assert lpc == pytest.approx(1.0)

    def test_fast_action_no_pause(self) -> None:
        """Receipt → immediate fast pass does not count as pause."""
        rows = [
            {
                "type": {"name": "Ball Receipt*"},
                "player": {"id": 1},
                "location": [60.0, 40.0],
                "timestamp": "00:10:00.000",
                "period": 1,
                "pass": {},
            },
            {
                "type": {"name": "Pass"},
                "player": {"id": 1},
                "location": [70.0, 40.0],
                "timestamp": "00:10:00.500",
                "period": 1,
                "pass": {"end_location": [80.0, 40.0]},
            },
        ]
        events = self._make_events(rows)
        synthetic_grid = np.linspace(0.01, 0.30, 12) * np.ones((8, 1))
        xt_model = XTModel(grid=synthetic_grid)
        lpc = compute_lpc(events, player_id=1, xt_model=xt_model, min_dt=1.2, max_vel=0.5)
        assert lpc == pytest.approx(0.0)

    def test_no_receipts_returns_zero(self) -> None:
        events = pd.DataFrame()
        xt_model = XTModel()
        lpc = compute_lpc(events, player_id=1, xt_model=xt_model)
        assert lpc == pytest.approx(0.0)


class TestCbi:
    """End-to-end tests for M9 Cutback Intelligence."""

    def _make_events(self, rows: list[dict]) -> pd.DataFrame:
        return pd.DataFrame(rows)

    def _make_frame(self, opponent_locs: list[list[float]]) -> dict:
        freeze = [
            {"teammate": True, "actor": True, "location": [60.0, 40.0]},
            {"teammate": False, "actor": False, "location": [90.0, 40.0]},
        ]
        for loc in opponent_locs:
            freeze.append({"teammate": False, "actor": False, "location": loc})
        return {"freeze_frame": freeze}

    def test_aligned_run_open_lane(self) -> None:
        """Run aligned with ball-to-player vector, open lane → count = 1."""
        rows = [
            {
                "type": {"name": "Pass"},
                "player": {"id": 1},
                "location": [55.0, 40.0],
                "timestamp": "00:09:59.000",
                "period": 1,
            },
            {
                "type": {"name": "Pass"},
                "player": {"id": 2},
                "location": [50.0, 40.0],
                "timestamp": "00:10:00.000",
                "period": 1,
            },
            {
                "type": {"name": "Ball Receipt*"},
                "player": {"id": 1},
                "location": [60.0, 40.0],
                "timestamp": "00:10:01.000",
                "period": 1,
            },
        ]
        events = self._make_events(rows)
        frames = [self._make_frame(opponent_locs=[])]
        cbi = compute_cbi(events, frames, player_id=1, angle_threshold=30.0, lane_buffer=1.5)
        # run_vec = [5, 0] (55→60), ball_to_player = [10, 0] (50→60) → aligned
        assert cbi == pytest.approx(1.0)

    def test_misaligned_run(self) -> None:
        """Run at 90° to ball vector → count = 0."""
        rows = [
            {
                "type": {"name": "Pass"},
                "player": {"id": 1},
                "location": [60.0, 35.0],
                "timestamp": "00:09:59.000",
                "period": 1,
            },
            {
                "type": {"name": "Pass"},
                "player": {"id": 2},
                "location": [50.0, 40.0],
                "timestamp": "00:10:00.000",
                "period": 1,
            },
            {
                "type": {"name": "Ball Receipt*"},
                "player": {"id": 1},
                "location": [60.0, 40.0],
                "timestamp": "00:10:01.000",
                "period": 1,
            },
        ]
        events = self._make_events(rows)
        frames = [self._make_frame(opponent_locs=[])]
        cbi = compute_cbi(events, frames, player_id=1, angle_threshold=30.0, lane_buffer=1.5)
        # run_vec = [0, 5] (35→40), ball_to_player = [10, 0] (50→60) → 90° angle
        assert cbi == pytest.approx(0.0)

    def test_blocked_lane(self) -> None:
        """Opponent on the passing line → count = 0."""
        rows = [
            {
                "type": {"name": "Pass"},
                "player": {"id": 1},
                "location": [55.0, 40.0],
                "timestamp": "00:09:59.000",
                "period": 1,
            },
            {
                "type": {"name": "Pass"},
                "player": {"id": 2},
                "location": [50.0, 40.0],
                "timestamp": "00:10:00.000",
                "period": 1,
            },
            {
                "type": {"name": "Ball Receipt*"},
                "player": {"id": 1},
                "location": [60.0, 40.0],
                "timestamp": "00:10:01.000",
                "period": 1,
            },
        ]
        events = self._make_events(rows)
        # Opponent sits directly on the passing line between 50,40 and 60,40
        frames = [self._make_frame(opponent_locs=[[55.0, 40.0]])]
        cbi = compute_cbi(events, frames, player_id=1, angle_threshold=30.0, lane_buffer=1.5)
        assert cbi == pytest.approx(0.0)

    def test_no_receipts_returns_zero(self) -> None:
        events = pd.DataFrame()
        cbi = compute_cbi(events, [], player_id=1)
        assert cbi == pytest.approx(0.0)

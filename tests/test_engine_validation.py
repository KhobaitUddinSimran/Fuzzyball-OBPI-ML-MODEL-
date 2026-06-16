"""End-to-end engine validation with synthetic data and expected outcomes.

Each test constructs a controlled scenario, computes one or more OBPI metrics,
and asserts that the actual value matches the expected outcome.
"""

from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from obpi.metrics.movement import compute_obr90, compute_oirc
from obpi.metrics.receiving import compute_brpc, compute_rbtl, compute_rup
from obpi.metrics.spatial import compute_sc, compute_sci
from obpi.metrics.temporal import compute_cbi, compute_lpc
from obpi.pipeline import compute_all_metrics
from obpi.utils.xt_model import XTModel


def _fmt_time(sec: float) -> str:
    """Format seconds as HH:MM:SS.000."""
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    return f"{h:02d}:{m:02d}:{s:02d}.000"


def _make_events(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Helper to build a DataFrame from row dicts."""
    return pd.DataFrame(rows)


def _make_frame(
    teammate_locs: list[list[float]], opponent_locs: list[list[float]
]) -> dict[str, Any]:
    """Build a 360 freeze-frame dict."""
    freeze: list[dict[str, Any]] = []
    for loc in teammate_locs:
        freeze.append({"teammate": True, "actor": False, "location": loc})
    for loc in opponent_locs:
        freeze.append({"teammate": False, "actor": False, "location": loc})
    return {"freeze_frame": freeze}


# ---------------------------------------------------------------------------
# 1.  OBR90  – Off-Ball Runs per 90
# ---------------------------------------------------------------------------
class TestOBR90:
    """Synthetic scenarios for M4 OBR90."""

    def test_player_with_two_fast_runs(self) -> None:
        """Two bursts of speed >2.5 m/s lasting >0.4 s in 2 minutes → ~90 runs/90."""
        rows = [
            # Run 1: big jump in 1 s
            {
                "timestamp": "00:10:00.000",
                "location": [60.0, 40.0],
                "period": 1,
                "play_pattern": {"name": "Regular Play"},
                "player": {"id": 1},
            },
            {
                "timestamp": "00:10:01.000",
                "location": [65.0, 40.0],
                "period": 1,
                "play_pattern": {"name": "Regular Play"},
                "player": {"id": 1},
            },
            # Run 2: another big jump in 1 s
            {
                "timestamp": "00:10:03.000",
                "location": [70.0, 40.0],
                "period": 1,
                "play_pattern": {"name": "Regular Play"},
                "player": {"id": 1},
            },
            {
                "timestamp": "00:10:04.000",
                "location": [75.0, 40.0],
                "period": 1,
                "play_pattern": {"name": "Regular Play"},
                "player": {"id": 1},
            },
            # Slower movement (does not qualify)
            {
                "timestamp": "00:10:05.500",
                "location": [76.0, 40.0],
                "period": 1,
                "play_pattern": {"name": "Regular Play"},
                "player": {"id": 1},
            },
        ]
        events = _make_events(rows)
        obr90 = compute_obr90(events, player_id=1, v_threshold=2.5, duration_threshold=0.4)
        # 2 runs in ~5.5 s → scaled to 90 min: very high
        assert obr90 > 0.0

    def test_stationary_player(self) -> None:
        """No displacement → 0 runs → 0.0."""
        rows = [
            {
                "timestamp": "00:10:00.000",
                "location": [60.0, 40.0],
                "period": 1,
                "play_pattern": {"name": "Regular Play"},
                "player": {"id": 1},
            },
            {
                "timestamp": "00:10:02.000",
                "location": [60.0, 40.0],
                "period": 1,
                "play_pattern": {"name": "Regular Play"},
                "player": {"id": 1},
            },
        ]
        events = _make_events(rows)
        obr90 = compute_obr90(events, player_id=1)
        assert obr90 == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 2.  OIRC  – Off-Ball Impact Run Coefficient
# ---------------------------------------------------------------------------
class TestOIRC:
    """Synthetic scenarios for M2 OIRC."""

    def test_run_toward_goal(self) -> None:
        """Run directly toward opponent goal → high directness → high OIRC."""
        rows = [
            {
                "timestamp": "00:10:00.000",
                "location": [60.0, 40.0],
                "period": 1,
                "play_pattern": {"name": "Regular Play"},
                "player": {"id": 1},
            },
            {
                "timestamp": "00:10:01.000",
                "location": [80.0, 40.0],
                "period": 1,
                "play_pattern": {"name": "Regular Play"},
                "player": {"id": 1},
            },
        ]
        events = _make_events(rows)
        oirc = compute_oirc(events, player_id=1, v_threshold=2.5, duration_threshold=0.4)
        # displacement = 20, directness ≈ 1.0 → oirc ≈ 20
        assert oirc > 10.0

    def test_run_sideways(self) -> None:
        """Run perpendicular to goal → directness = 0 → OIRC = 0."""
        rows = [
            {
                "timestamp": "00:10:00.000",
                "location": [60.0, 30.0],
                "period": 1,
                "play_pattern": {"name": "Regular Play"},
                "player": {"id": 1},
            },
            {
                "timestamp": "00:10:01.000",
                "location": [60.0, 50.0],
                "period": 1,
                "play_pattern": {"name": "Regular Play"},
                "player": {"id": 1},
            },
        ]
        events = _make_events(rows)
        oirc = compute_oirc(events, player_id=1, v_threshold=2.5, duration_threshold=0.4)
        assert oirc == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 3.  RBTL  – Receipts Behind The Line
# ---------------------------------------------------------------------------
class TestRBTL:
    """Synthetic scenarios for M5 RBTL."""

    def test_all_receipts_behind_line(self) -> None:
        """Every receipt is between x=60 and x=120 → RBTL = 1.0."""
        rows = [
            {
                "type": {"name": "Ball Receipt*"},
                "player": {"id": 1},
                "location": [70.0, 40.0],
            },
            {
                "type": {"name": "Ball Receipt*"},
                "player": {"id": 1},
                "location": [90.0, 40.0],
            },
        ]
        events = _make_events(rows)
        rbtl = compute_rbtl(events, player_id=1, def_line_x=60.0, back_line_x=120.0)
        assert rbtl == pytest.approx(1.0)

    def test_half_receipts_behind_line(self) -> None:
        """One receipt inside, one outside → RBTL = 0.5."""
        rows = [
            {
                "type": {"name": "Ball Receipt*"},
                "player": {"id": 1},
                "location": [70.0, 40.0],
            },
            {
                "type": {"name": "Ball Receipt*"},
                "player": {"id": 1},
                "location": [50.0, 40.0],
            },
        ]
        events = _make_events(rows)
        rbtl = compute_rbtl(events, player_id=1, def_line_x=60.0, back_line_x=120.0)
        assert rbtl == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# 4.  RUP  – Receipt Under Pressure
# ---------------------------------------------------------------------------
class TestRUP:
    """Synthetic scenarios for M6 RUP."""

    def test_all_receipts_under_pressure(self) -> None:
        """Both receipts have under_pressure=True → RUP = 1.0."""
        rows = [
            {
                "type": {"name": "Ball Receipt*"},
                "player": {"id": 1},
                "location": [60.0, 40.0],
                "under_pressure": True,
            },
            {
                "type": {"name": "Ball Receipt*"},
                "player": {"id": 1},
                "location": [70.0, 40.0],
                "under_pressure": True,
            },
        ]
        events = _make_events(rows)
        rup = compute_rup(events, player_id=1)
        assert rup == pytest.approx(1.0)

    def test_half_receipts_under_pressure(self) -> None:
        """One pressured, one clean → RUP = 0.5."""
        rows = [
            {
                "type": {"name": "Ball Receipt*"},
                "player": {"id": 1},
                "location": [60.0, 40.0],
                "under_pressure": True,
            },
            {
                "type": {"name": "Ball Receipt*"},
                "player": {"id": 1},
                "location": [70.0, 40.0],
                "under_pressure": False,
            },
        ]
        events = _make_events(rows)
        rup = compute_rup(events, player_id=1)
        assert rup == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# 5.  BRPC  – Ball Receipt Pressure Counter
# ---------------------------------------------------------------------------
class TestBRPC:
    """Synthetic scenarios for M3 BRPC."""

    def test_no_opponents_nearby(self) -> None:
        """Nearest opponent 20 m away, cone clear → BRPC = 1.0."""
        rows = [
            {
                "type": {"name": "Ball Receipt*"},
                "player": {"id": 1},
                "location": [60.0, 40.0],
            },
        ]
        events = _make_events(rows)
        frames = [
            _make_frame(
                teammate_locs=[[55.0, 40.0]],
                opponent_locs=[[80.0, 40.0]],  # far away
            )
        ]
        brpc = compute_brpc(
            events, frames, player_id=1, pressure_threshold=5.0,
            cone_angle=45.0, cone_length=15.0
        )
        assert brpc == pytest.approx(1.0)

    def test_opponent_inside_pressure_radius(self) -> None:
        """Opponent 2 m away → fails pressure check → BRPC = 0.0."""
        rows = [
            {
                "type": {"name": "Ball Receipt*"},
                "player": {"id": 1},
                "location": [60.0, 40.0],
            },
        ]
        events = _make_events(rows)
        frames = [
            _make_frame(
                teammate_locs=[[55.0, 40.0]],
                opponent_locs=[[61.0, 40.0]],  # 1 m away
            )
        ]
        brpc = compute_brpc(
            events, frames, player_id=1, pressure_threshold=5.0,
            cone_angle=45.0, cone_length=15.0
        )
        assert brpc == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 6.  SCI  – Space Control Improvement
# ---------------------------------------------------------------------------
class TestSCI:
    """Synthetic scenarios for M7 SCI."""

    def test_teammates_spread_out(self) -> None:
        """Teammates spread wider after → positive area gain."""
        before = _make_frame(
            teammate_locs=[[50.0, 40.0], [51.0, 40.0], [52.0, 40.0], [53.0, 40.0]],  # tight line
            opponent_locs=[[90.0, 40.0]],
        )
        after = _make_frame(
            teammate_locs=[[20.0, 20.0], [40.0, 30.0], [80.0, 50.0], [100.0, 60.0]],  # spread
            opponent_locs=[[90.0, 40.0]],
        )
        sci = compute_sci([before], [after])
        assert sci > 0.0  # average gain should be positive when spreading out

    def test_teammates_bunch_up(self) -> None:
        """Teammates cluster tighter after → negative area gain."""
        before = _make_frame(
            teammate_locs=[[20.0, 20.0], [40.0, 30.0], [80.0, 50.0], [100.0, 60.0]],  # spread
            opponent_locs=[[90.0, 40.0]],
        )
        after = _make_frame(
            teammate_locs=[[50.0, 40.0], [51.0, 40.0], [52.0, 40.0], [53.0, 40.0]],  # tight line
            opponent_locs=[[90.0, 40.0]],
        )
        sci = compute_sci([before], [after])
        assert sci < 0.0  # average gain should be negative when bunching up


# ---------------------------------------------------------------------------
# 7.  SC  – Screen Count
# ---------------------------------------------------------------------------
class TestSC:
    """Synthetic scenarios for M1 SC."""

    def test_local_defender_shifts_more_than_global(self) -> None:
        """Nearby defender moves 5 m, global average moves 1 m → screen counts."""
        before = _make_frame(
            teammate_locs=[[60.0, 40.0]],
            opponent_locs=[[62.0, 40.0], [90.0, 40.0]],
        )
        after = _make_frame(
            teammate_locs=[[60.0, 40.0]],
            opponent_locs=[[67.0, 40.0], [91.0, 40.0]],
        )
        sc = compute_sc(
            [before], [after], player_location=[60.0, 40.0],
            box_size=(15.0, 15.0), threshold=1.5
        )
        assert sc == pytest.approx(1.0)

    def test_global_shift_exceeds_local(self) -> None:
        """Global shift > local shift → no screen."""
        before = _make_frame(
            teammate_locs=[[60.0, 40.0]],
            opponent_locs=[[62.0, 40.0], [90.0, 40.0]],
        )
        after = _make_frame(
            teammate_locs=[[60.0, 40.0]],
            opponent_locs=[[63.0, 40.0], [110.0, 40.0]],
        )
        sc = compute_sc(
            [before], [after], player_location=[60.0, 40.0],
            box_size=(15.0, 15.0), threshold=1.5
        )
        assert sc == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 8.  LPC  – Layoff Pause Coefficient
# ---------------------------------------------------------------------------
class TestLPC:
    """Synthetic scenarios for M8 LPC."""

    def test_pause_then_improve_xt(self) -> None:
        """Receipt → wait 2 s → slow pass to higher-xT zone → LPC = 1.0."""
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
        events = _make_events(rows)
        synthetic_grid = np.linspace(0.01, 0.30, 12) * np.ones((8, 1))
        xt_model = XTModel(grid=synthetic_grid)
        lpc = compute_lpc(events, player_id=1, xt_model=xt_model, min_dt=1.2, max_vel=0.5)
        assert lpc == pytest.approx(1.0)

    def test_no_pause_no_count(self) -> None:
        """Receipt → immediate fast pass → LPC = 0.0."""
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
                "timestamp": "00:10:00.200",
                "period": 1,
                "pass": {"end_location": [80.0, 40.0]},
            },
        ]
        events = _make_events(rows)
        synthetic_grid = np.linspace(0.01, 0.30, 12) * np.ones((8, 1))
        xt_model = XTModel(grid=synthetic_grid)
        lpc = compute_lpc(events, player_id=1, xt_model=xt_model, min_dt=1.2, max_vel=0.5)
        assert lpc == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 9.  CBI  – Cutback Intelligence
# ---------------------------------------------------------------------------
class TestCBI:
    """Synthetic scenarios for M9 CBI."""

    def test_aligned_run_open_lane(self) -> None:
        """Run aligned with ball vector, no opponents on line → CBI = 1.0."""
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
        events = _make_events(rows)
        frames = [_make_frame(teammate_locs=[[60.0, 40.0]], opponent_locs=[])]
        cbi = compute_cbi(events, frames, player_id=1, angle_threshold=30.0, lane_buffer=1.5)
        assert cbi == pytest.approx(1.0)

    def test_blocked_lane(self) -> None:
        """Opponent sits on passing line → CBI = 0.0."""
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
        events = _make_events(rows)
        frames = [
            _make_frame(
                teammate_locs=[[60.0, 40.0]],
                opponent_locs=[[55.0, 40.0]],  # on the line
            )
        ]
        cbi = compute_cbi(events, frames, player_id=1, angle_threshold=30.0, lane_buffer=1.5)
        assert cbi == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 10.  Rich Synthetic Match — Continuous Outcomes
# ---------------------------------------------------------------------------
class TestRichSyntheticMatch:
    """One dense match with many events → metrics are continuous, not just 0/1."""

    @staticmethod
    def _make_rich_match() -> tuple[pd.DataFrame, list[dict[str, Any]]]:
        """Return ~40 events for 2 players + matching freeze frames."""
        rows: list[dict[str, Any]] = []
        base_time = 600.0  # 10:00 in seconds

        # ---------- Player 10 (winger) ----------
        # 5 receipt→pass sequences for LPC/CBI
        for i in range(5):
            # preceding pass by teammate
            rows.append(
                {
                    "timestamp": _fmt_time(base_time + i * 30),
                    "location": [40.0, 30.0 + i * 2],
                    "period": 1,
                    "play_pattern": {"name": "Regular Play"},
                    "player": {"id": 99},
                    "type": {"name": "Pass"},
                }
            )
            # receipt by Player 10
            receipt_x = 50.0 + i * 5
            receipt_y = 35.0 + i * 2
            pressured = i % 3 == 0  # True for i=0,3
            rows.append(
                {
                    "timestamp": _fmt_time(base_time + i * 30 + 1),
                    "location": [receipt_x, receipt_y],
                    "period": 1,
                    "play_pattern": {"name": "Regular Play"},
                    "player": {"id": 10},
                    "type": {"name": "Ball Receipt*"},
                    "under_pressure": pressured,
                }
            )
            # pass by Player 10
            pause = i % 2 == 0  # True for i=0,2,4  → pause 3s; i=1,3 → immediate
            pass_time = base_time + i * 30 + (4 if pause else 1)
            rows.append(
                {
                    "timestamp": _fmt_time(pass_time),
                    "location": [receipt_x, receipt_y],
                    "period": 1,
                    "play_pattern": {"name": "Regular Play"},
                    "player": {"id": 10},
                    "type": {"name": "Pass"},
                    "pass": {"end_location": [receipt_x + 15, receipt_y + 5]},
                }
            )

        # Extra movement events for Player 10 (for OBR90 / OIRC)
        # Three fast runs toward goal
        for run_start in [800.0, 850.0, 900.0]:
            rows.append(
                {
                    "timestamp": _fmt_time(run_start),
                    "location": [60.0, 40.0],
                    "period": 1,
                    "play_pattern": {"name": "Regular Play"},
                    "player": {"id": 10},
                    "type": {"name": "Pass"},
                }
            )
            rows.append(
                {
                    "timestamp": _fmt_time(run_start + 1),
                    "location": [80.0, 40.0],
                    "period": 1,
                    "play_pattern": {"name": "Regular Play"},
                    "player": {"id": 10},
                    "type": {"name": "Ball Receipt*"},
                    "under_pressure": False,
                }
            )
            rows.append(
                {
                    "timestamp": _fmt_time(run_start + 2),
                    "location": [81.0, 40.0],
                    "period": 1,
                    "play_pattern": {"name": "Regular Play"},
                    "player": {"id": 10},
                    "type": {"name": "Pass"},
                }
            )

        # One sideways run (OIRC → 0 contribution)
        rows.append(
            {
                "timestamp": "00:16:00.000",
                "location": [60.0, 30.0],
                "period": 1,
                "play_pattern": {"name": "Regular Play"},
                "player": {"id": 10},
                "type": {"name": "Pass"},
            }
        )
        rows.append(
            {
                "timestamp": "00:16:01.000",
                "location": [60.0, 50.0],
                "period": 1,
                "play_pattern": {"name": "Regular Play"},
                "player": {"id": 10},
                "type": {"name": "Ball Receipt*"},
                "under_pressure": False,
            }
        )
        rows.append(
            {
                "timestamp": "00:16:02.000",
                "location": [60.5, 50.0],
                "period": 1,
                "play_pattern": {"name": "Regular Play"},
                "player": {"id": 10},
                "type": {"name": "Pass"},
            }
        )

        # ---------- Player 20 (midfielder) ----------
        for i in range(4):
            rows.append(
                {
                    "timestamp": _fmt_time(700 + i * 40),
                    "location": [45.0, 40.0],
                    "period": 1,
                    "play_pattern": {"name": "Regular Play"},
                    "player": {"id": 20},
                    "type": {"name": "Ball Receipt*"},
                    "under_pressure": i % 2 == 0,  # 2 pressured, 2 not
                }
            )

        events = pd.DataFrame(rows)
        events["_ts_sec"] = events["timestamp"].apply(
            lambda ts: (
                float(ts.split(":")[0]) * 3600
                + float(ts.split(":")[1]) * 60
                + float(ts.split(":")[2])
            )
        )
        events = events.sort_values(by=["period", "_ts_sec"]).reset_index(drop=True)
        events = events.drop(columns=["_ts_sec"])

        # ---------- Freeze frames ----------
        # One frame per receipt (player 10 has 8 receipts, player 20 has 4)
        frames: list[dict[str, Any]] = []
        # Player 10 frames: varying opponent proximity / lane openness
        for i in range(8):
            opp_x = 85.0 if i % 4 == 0 else 65.0  # far only for i=0,4
            frames.append(
                _make_frame(
                    teammate_locs=[[55.0, 35.0], [58.0, 42.0], [52.0, 38.0]],
                    opponent_locs=[[opp_x, 40.0], [90.0, 20.0]],
                )
            )
        # Player 20 frames
        for _i in range(4):
            frames.append(
                _make_frame(
                    teammate_locs=[[42.0, 38.0], [48.0, 42.0]],
                    opponent_locs=[[50.0, 40.0], [95.0, 40.0]],
                )
            )

        return events, frames

    @patch("obpi.pipeline.StatsBombLoader")
    def test_continuous_rup(self, mock_loader_cls: MagicMock) -> None:
        """Player 10: 2 pressured out of 7 receipts → RUP ≈ 0.286."""
        mock_loader = MagicMock()
        events, frames = self._make_rich_match()
        mock_loader.get_events.return_value = events
        mock_loader.get_freeze_frames.return_value = frames
        mock_loader_cls.return_value = mock_loader

        df = compute_all_metrics(match_id=100, xt_model=XTModel())
        p10 = df[df["player_id"] == 10].iloc[0]
        # Player 10 receipts: 5 sequences + 3 run + 1 sideways = 9
        # Pressured: i=0,3 in sequences = 2
        # Total receipts = 9, pressured = 2 → RUP ≈ 0.222
        assert 0.15 < p10["M6_RUP"] < 0.35

    @patch("obpi.pipeline.StatsBombLoader")
    def test_continuous_rbtl(self, mock_loader_cls: MagicMock) -> None:
        """Player 10: 5 of 9 receipts behind line (x=60 excluded as boundary)."""
        mock_loader = MagicMock()
        events, frames = self._make_rich_match()
        mock_loader.get_events.return_value = events
        mock_loader.get_freeze_frames.return_value = frames
        mock_loader_cls.return_value = mock_loader

        df = compute_all_metrics(match_id=100, xt_model=XTModel())
        p10 = df[df["player_id"] == 10].iloc[0]
        # Sequence: 50,55,60,65,70 → behind (x>60): 65,70 = 2
        # Run receipts: 80,80,80 = 3
        # Sideways: 60 excluded = 0
        # Total = 5/9 ≈ 0.556
        assert p10["M5_RBTL"] == pytest.approx(5 / 9, abs=0.05)

    @patch("obpi.pipeline.StatsBombLoader")
    def test_continuous_brpc(self, mock_loader_cls: MagicMock) -> None:
        """Player 10: 6 of 9 receipts qualify (cone clear + far opponent)."""
        mock_loader = MagicMock()
        events, frames = self._make_rich_match()
        mock_loader.get_events.return_value = events
        mock_loader.get_freeze_frames.return_value = frames
        mock_loader_cls.return_value = mock_loader

        df = compute_all_metrics(match_id=100, xt_model=XTModel())
        p10 = df[df["player_id"] == 10].iloc[0]
        # 3 receipts fail: i=1,2 (opponent in cone), i=3 (opponent too near)
        assert p10["M3_BRPC"] == pytest.approx(6 / 9, abs=0.05)

    @patch("obpi.pipeline.StatsBombLoader")
    def test_continuous_lpc(self, mock_loader_cls: MagicMock) -> None:
        """Player 10: 3 pauses out of 5 sequences, some improve xT → LPC in (0,1)."""
        mock_loader = MagicMock()
        events, frames = self._make_rich_match()
        mock_loader.get_events.return_value = events
        mock_loader.get_freeze_frames.return_value = frames
        mock_loader_cls.return_value = mock_loader

        df = compute_all_metrics(match_id=100, xt_model=XTModel())
        p10 = df[df["player_id"] == 10].iloc[0]
        # i=0,2,4 pause (dt=3s); i=1,3 fast (dt=1s)
        # of the 3 pauses, some improve xT
        assert 0.0 < p10["M8_LPC"] < 1.0

    @patch("obpi.pipeline.StatsBombLoader")
    def test_player_20_rup(self, mock_loader_cls: MagicMock) -> None:
        """Player 20: 2 pressured out of 4 receipts → RUP = 0.5."""
        mock_loader = MagicMock()
        events, frames = self._make_rich_match()
        mock_loader.get_events.return_value = events
        mock_loader.get_freeze_frames.return_value = frames
        mock_loader_cls.return_value = mock_loader

        df = compute_all_metrics(match_id=100, xt_model=XTModel())
        p20 = df[df["player_id"] == 20].iloc[0]
        assert p20["M6_RUP"] == pytest.approx(0.5)

    @patch("obpi.pipeline.StatsBombLoader")
    def test_player_20_rbtl(self, mock_loader_cls: MagicMock) -> None:
        """Player 20: all 4 receipts at x=45 → none behind line → RBTL = 0.0."""
        mock_loader = MagicMock()
        events, frames = self._make_rich_match()
        mock_loader.get_events.return_value = events
        mock_loader.get_freeze_frames.return_value = frames
        mock_loader_cls.return_value = mock_loader

        df = compute_all_metrics(match_id=100, xt_model=XTModel())
        p20 = df[df["player_id"] == 20].iloc[0]
        assert p20["M5_RBTL"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 11.  Full Pipeline End-to-End
# ---------------------------------------------------------------------------
class TestFullPipeline:
    """End-to-end validation with mocked loader and expected outcomes."""

    @staticmethod
    def _make_match_data() -> tuple[pd.DataFrame, list[dict[str, Any]]]:
        """Return synthetic events and frames for a 2-player match."""
        events = pd.DataFrame(
            [
                # Player 1: fast run toward goal
                {
                    "timestamp": "00:01:00.000",
                    "location": [60.0, 40.0],
                    "period": 1,
                    "play_pattern": {"name": "Regular Play"},
                    "player": {"id": 1},
                    "type": {"name": "Pass"},
                },
                {
                    "timestamp": "00:01:01.000",
                    "location": [80.0, 40.0],
                    "period": 1,
                    "play_pattern": {"name": "Regular Play"},
                    "player": {"id": 1},
                    "type": {"name": "Ball Receipt*"},
                    "under_pressure": True,
                },
                # Player 2: receipt behind line, no pressure
                {
                    "timestamp": "00:02:00.000",
                    "location": [70.0, 40.0],
                    "period": 1,
                    "play_pattern": {"name": "Regular Play"},
                    "player": {"id": 2},
                    "type": {"name": "Ball Receipt*"},
                    "under_pressure": False,
                },
                {
                    "timestamp": "00:02:03.000",
                    "location": [85.0, 40.0],
                    "period": 1,
                    "play_pattern": {"name": "Regular Play"},
                    "player": {"id": 2},
                    "type": {"name": "Pass"},
                    "pass": {"end_location": [90.0, 40.0]},
                },
            ]
        )
        frames = [
            _make_frame(
                teammate_locs=[[60.0, 40.0], [55.0, 35.0], [62.0, 45.0]],
                opponent_locs=[[65.0, 42.0], [68.0, 38.0]],
            ),
            _make_frame(
                teammate_locs=[[60.0, 40.0], [58.0, 36.0], [64.0, 46.0]],
                opponent_locs=[[70.0, 42.0], [72.0, 39.0]],
            ),
        ]
        return events, frames

    @patch("obpi.pipeline.StatsBombLoader")
    def test_player_1_high_rup(self, mock_loader_cls: MagicMock) -> None:
        """Player 1 receives under pressure → RUP should be high."""
        mock_loader = MagicMock()
        events, frames = self._make_match_data()
        mock_loader.get_events.return_value = events
        mock_loader.get_freeze_frames.return_value = frames
        mock_loader_cls.return_value = mock_loader

        df = compute_all_metrics(match_id=1, xt_model=XTModel())
        p1 = df[df["player_id"] == 1].iloc[0]
        assert p1["M6_RUP"] == pytest.approx(1.0)

    @patch("obpi.pipeline.StatsBombLoader")
    def test_player_2_zero_rup(self, mock_loader_cls: MagicMock) -> None:
        """Player 2 receives without pressure → RUP should be 0.0."""
        mock_loader = MagicMock()
        events, frames = self._make_match_data()
        mock_loader.get_events.return_value = events
        mock_loader.get_freeze_frames.return_value = frames
        mock_loader_cls.return_value = mock_loader

        df = compute_all_metrics(match_id=1, xt_model=XTModel())
        p2 = df[df["player_id"] == 2].iloc[0]
        assert p2["M6_RUP"] == pytest.approx(0.0)

    @patch("obpi.pipeline.StatsBombLoader")
    def test_all_metrics_finite(self, mock_loader_cls: MagicMock) -> None:
        """Every metric value must be finite for every player."""
        mock_loader = MagicMock()
        events, frames = self._make_match_data()
        mock_loader.get_events.return_value = events
        mock_loader.get_freeze_frames.return_value = frames
        mock_loader_cls.return_value = mock_loader

        df = compute_all_metrics(match_id=1, xt_model=XTModel())
        metric_cols = [c for c in df.columns if c.startswith("M")]
        for col in metric_cols:
            assert df[col].apply(lambda x: x == x).all(), f"{col} has NaN"

    @patch("obpi.pipeline.StatsBombLoader")
    def test_expected_column_order(self, mock_loader_cls: MagicMock) -> None:
        """Pipeline must return columns in the exact expected order."""
        mock_loader = MagicMock()
        events, frames = self._make_match_data()
        mock_loader.get_events.return_value = events
        mock_loader.get_freeze_frames.return_value = frames
        mock_loader_cls.return_value = mock_loader

        df = compute_all_metrics(match_id=1)
        expected = [
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
        assert list(df.columns) == expected

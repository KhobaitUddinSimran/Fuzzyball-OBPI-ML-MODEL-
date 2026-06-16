"""Tests for receiving metrics (M5 RBTL, M6 RUP, M3 BRPC)."""

from __future__ import annotations

import pandas as pd
import pytest

from obpi.data.preprocessor import nearest_opponent_distance
from obpi.metrics.receiving import compute_brpc, compute_rbtl, compute_rup


def _make_receipt(
    player_id: int = 1,
    location: list[float] | None = None,
    under_pressure: bool = False,
    type_name: str = "Ball Receipt*",
) -> dict:
    return {
        "player": {"id": player_id, "name": "Test Player"},
        "location": location or [60.0, 40.0],
        "under_pressure": under_pressure,
        "type": {"id": 42, "name": type_name},
        "timestamp": "00:05:00.000",
        "period": 1,
    }


def _make_events(receipts: list[dict]) -> pd.DataFrame:
    """Wrap receipt dicts in a DataFrame."""
    return pd.DataFrame(receipts)


def _make_frame(opponent_locations: list[list[float]]) -> dict:
    """Build a 360 freeze-frame with teammates and opponents."""
    freeze = [
        {"teammate": True, "actor": True, "keeper": False, "location": [60.0, 40.0]},
    ]
    for loc in opponent_locations:
        freeze.append(
            {"teammate": False, "actor": False, "keeper": False, "location": loc}
        )
    return {"freeze_frame": freeze}


class TestNearestOpponentDistance:
    """Unit tests for the 360-frame proximity helper."""

    def test_finds_nearest_opponent(self) -> None:
        frame = _make_frame([[65.0, 40.0], [70.0, 40.0]])
        dist = nearest_opponent_distance(frame, [60.0, 40.0])
        assert dist == pytest.approx(5.0)

    def test_no_opponents_returns_inf(self) -> None:
        frame = _make_frame([])
        dist = nearest_opponent_distance(frame, [60.0, 40.0])
        assert dist == float("inf")


class TestRbtl:
    """End-to-end tests for M5 RBTL."""

    def test_two_in_half_space_out_of_four(self) -> None:
        """2 receipts inside half-space / 4 total → RBTL = 0.5."""
        receipts = [
            _make_receipt(location=[70.0, 40.0]),   # inside (60-120)
            _make_receipt(location=[80.0, 20.0]),   # inside
            _make_receipt(location=[50.0, 40.0]),   # outside
            _make_receipt(location=[55.0, 60.0]),   # outside
        ]
        events = _make_events(receipts)
        rbtl = compute_rbtl(events, player_id=1, def_line_x=60.0, back_line_x=120.0)
        assert rbtl == pytest.approx(0.5)

    def test_zero_receipts_returns_zero(self) -> None:
        events = _make_events([])
        rbtl = compute_rbtl(events, player_id=1)
        assert rbtl == pytest.approx(0.0)

    def test_other_player_receipts_ignored(self) -> None:
        receipts = [
            _make_receipt(player_id=2, location=[70.0, 40.0]),
        ]
        events = _make_events(receipts)
        rbtl = compute_rbtl(events, player_id=1)
        assert rbtl == pytest.approx(0.0)


class TestRup:
    """End-to-end tests for M6 RUP."""

    def test_half_under_pressure(self) -> None:
        """2 of 4 receipts under pressure → RUP = 0.5."""
        receipts = [
            _make_receipt(under_pressure=True),
            _make_receipt(under_pressure=True),
            _make_receipt(under_pressure=False),
            _make_receipt(under_pressure=False),
        ]
        events = _make_events(receipts)
        rup = compute_rup(events, player_id=1)
        assert rup == pytest.approx(0.5)

    def test_fallback_360_proximity(self) -> None:
        """When under_pressure flag is missing, use 360 frame fallback."""
        receipts = [
            {"player": {"id": 1}, "location": [60.0, 40.0], "type": {"name": "Ball Receipt*"}},
        ]
        events = _make_events(receipts)
        # Opponent at 2.0m → under pressure (threshold = 2.5)
        frames = [_make_frame([[62.0, 40.0]])]
        rup = compute_rup(events, player_id=1, frames=frames, proximity_threshold=2.5)
        assert rup == pytest.approx(1.0)

    def test_fallback_360_not_close(self) -> None:
        """Opponent farther than threshold → not under pressure."""
        receipts = [
            {"player": {"id": 1}, "location": [60.0, 40.0], "type": {"name": "Ball Receipt*"}},
        ]
        events = _make_events(receipts)
        # Opponent at 10.0m → not under pressure
        frames = [_make_frame([[70.0, 40.0]])]
        rup = compute_rup(events, player_id=1, frames=frames, proximity_threshold=2.5)
        assert rup == pytest.approx(0.0)

    def test_zero_receipts_returns_zero(self) -> None:
        events = _make_events([])
        rup = compute_rup(events, player_id=1)
        assert rup == pytest.approx(0.0)


class TestBrpc:
    """End-to-end tests for M3 BRPC."""

    def test_open_lane_no_pressure(self) -> None:
        """Defender at 6m + open cone → count = 1."""
        receipts = [
            _make_receipt(location=[60.0, 40.0]),
        ]
        events = _make_events(receipts)
        # Opponent far away (6m) and outside the forward cone
        frames = [_make_frame([[66.0, 60.0]])]
        brpc = compute_brpc(events, frames, player_id=1)
        assert brpc == pytest.approx(1.0)

    def test_blocked_lane(self) -> None:
        """Defender inside forward cone → count = 0."""
        receipts = [
            _make_receipt(location=[60.0, 40.0]),
        ]
        events = _make_events(receipts)
        # Opponent directly in front, inside the cone
        frames = [_make_frame([[65.0, 40.0]])]
        brpc = compute_brpc(events, frames, player_id=1)
        assert brpc == pytest.approx(0.0)

    def test_too_close(self) -> None:
        """Defender within 5m → count = 0 regardless of lane."""
        receipts = [
            _make_receipt(location=[60.0, 40.0]),
        ]
        events = _make_events(receipts)
        # Opponent at 3m — too close
        frames = [_make_frame([[63.0, 40.0]])]
        brpc = compute_brpc(events, frames, player_id=1)
        assert brpc == pytest.approx(0.0)

    def test_zero_receipts_returns_zero(self) -> None:
        events = _make_events([])
        brpc = compute_brpc(events, [], player_id=1)
        assert brpc == pytest.approx(0.0)

    def test_no_frames_returns_zero(self) -> None:
        receipts = [
            _make_receipt(location=[60.0, 40.0]),
        ]
        events = _make_events(receipts)
        brpc = compute_brpc(events, [], player_id=1)
        assert brpc == pytest.approx(0.0)

"""Tests for movement metrics (M4 OBR90 and M2 OIRC)."""

import pandas as pd
import pytest

from obpi.metrics.movement import compute_obr90, compute_oirc
from obpi.utils import geometry, kinematics


def _make_player_events(
    player_id: int = 1,
    locations: list[list[float]] | None = None,
    timestamps: list[str] | None = None,
    periods: list[int] | None = None,
    play_patterns: list[str] | None = None,
) -> pd.DataFrame:
    """Build a minimal StatsBomb-style events DataFrame for one player."""
    if locations is None:
        locations = [[60.0, 40.0]]
    n = len(locations)
    if timestamps is None:
        timestamps = ["00:00:00.000"] * n
    if periods is None:
        periods = [1] * n
    if play_patterns is None:
        play_patterns = ["Regular Play"] * n

    return pd.DataFrame(
        {
            "player": [{"id": player_id}] * n,
            "location": locations,
            "timestamp": timestamps,
            "period": periods,
            "play_pattern": [{"name": p} for p in play_patterns],
        }
    )


class TestInferVelocity:
    """Unit tests for the kinematics velocity-inference pipeline."""

    def test_basic_velocity_computation(self) -> None:
        events = _make_player_events(
            locations=[[60.0, 40.0], [70.0, 40.0], [71.0, 40.0]],
            timestamps=["00:00:00.000", "00:00:01.000", "00:00:02.000"],
        )
        vel = kinematics.infer_velocity(events, player_id=1, max_dt=1.5)
        assert vel["velocity"].iloc[0] != vel["velocity"].iloc[0]  # NaN
        assert vel["velocity"].iloc[1] == pytest.approx(10.0)
        assert vel["velocity"].iloc[2] == pytest.approx(1.0)

    def test_dt_gate(self) -> None:
        events = _make_player_events(
            locations=[[60.0, 40.0], [90.0, 40.0]],
            timestamps=["00:00:00.000", "00:00:02.000"],
        )
        vel = kinematics.infer_velocity(events, player_id=1, max_dt=1.5)
        # dt = 2.0 > 1.5 → velocity should be NaN
        assert vel["velocity"].iloc[1] != vel["velocity"].iloc[1]  # NaN


class TestDetectRuns:
    """Unit tests for run detection."""

    def test_detects_single_run(self) -> None:
        events = _make_player_events(
            locations=[[60.0, 40.0], [70.0, 40.0], [71.0, 40.0]],
            timestamps=["00:00:00.000", "00:00:01.000", "00:00:02.000"],
        )
        vel = kinematics.infer_velocity(events, player_id=1)
        runs = kinematics.detect_runs(vel, v_threshold=2.5, duration_threshold=0.4)
        assert len(runs) == 1
        assert runs[0]["duration"] == pytest.approx(1.0)
        assert runs[0]["displacement"] == pytest.approx(10.0)

    def test_no_runs_below_threshold(self) -> None:
        events = _make_player_events(
            locations=[[60.0, 40.0], [61.0, 40.0], [62.0, 40.0]],
            timestamps=["00:00:00.000", "00:00:01.000", "00:00:02.000"],
        )
        vel = kinematics.infer_velocity(events, player_id=1)
        runs = kinematics.detect_runs(vel, v_threshold=2.5, duration_threshold=0.4)
        assert len(runs) == 0


class TestGeometry:
    """Unit tests for geometric helpers."""

    def test_run_directness_perfect(self) -> None:
        import numpy as np
        v_run = np.array([10.0, 0.0], dtype=np.float64)
        v_goal = np.array([60.0, 0.0], dtype=np.float64)
        assert geometry.run_directness(v_run, v_goal) == pytest.approx(1.0)

    def test_run_directness_perpendicular(self) -> None:
        import numpy as np
        v_run = np.array([0.0, 10.0], dtype=np.float64)
        v_goal = np.array([60.0, 0.0], dtype=np.float64)
        assert geometry.run_directness(v_run, v_goal) == pytest.approx(0.0)

    def test_run_directness_clips_negative(self) -> None:
        import numpy as np
        v_run = np.array([-10.0, 0.0], dtype=np.float64)
        v_goal = np.array([60.0, 0.0], dtype=np.float64)
        assert geometry.run_directness(v_run, v_goal) == pytest.approx(0.0)


class TestObr90:
    """End-to-end tests for M4 Off-Ball Runs per 90."""

    def test_three_runs_in_ninety_minutes(self) -> None:
        """Synthetic player with 3 runs in 90 min → OBR90 = 3.0."""
        events = _make_player_events(
            locations=[
                [60.0, 40.0],   # 0  start
                [60.0, 40.0],   # 1  10 min later (dt>1.5)
                [70.0, 40.0],   # 2  fast → Run 1
                [70.5, 40.0],   # 3  slow
                [70.5, 40.0],   # 4  period 2 start (dt>1.5)
                [80.5, 40.0],   # 5  fast → Run 2
                [81.0, 40.0],   # 6  slow
                [81.0, 40.0],   # 7  10 min later (dt>1.5)
                [91.0, 40.0],   # 8  fast → Run 3
                [91.5, 40.0],   # 9  slow
                [91.5, 40.0],   # 10 end of match (dt>1.5)
            ],
            timestamps=[
                "00:00:00.000", "00:10:00.000", "00:10:01.000",
                "00:10:02.000",
                "00:00:00.000", "00:00:01.000", "00:00:02.000",
                "00:10:00.000", "00:10:01.000", "00:10:02.000",
                "00:45:00.000",
            ],
            periods=[
                1, 1, 1, 1,
                2, 2, 2, 2, 2, 2, 2,
            ],
        )
        obr = compute_obr90(events, player_id=1, minutes_played=90.0)
        assert obr == pytest.approx(3.0)

    def test_set_piece_exclusion(self) -> None:
        """Events during a corner should not count as a run."""
        events = _make_player_events(
            locations=[
                [60.0, 40.0],
                [70.0, 40.0],   # fast step
                [71.0, 40.0],   # slow
            ],
            timestamps=[
                "00:00:00.000", "00:00:01.000", "00:00:02.000",
            ],
            play_patterns=["From Corner", "From Corner", "From Corner"],
        )
        obr = compute_obr90(events, player_id=1, minutes_played=90.0)
        assert obr == pytest.approx(0.0)

    def test_zero_minutes_returns_zero(self) -> None:
        events = _make_player_events()
        obr = compute_obr90(events, player_id=1, minutes_played=0.0)
        assert obr == pytest.approx(0.0)


class TestOirc:
    """End-to-end tests for M2 Off-Ball Impact Run Coefficient."""

    def test_perfect_directness(self) -> None:
        """Three straight-ahead runs → OIRC = mean displacement = 10.0."""
        events = _make_player_events(
            locations=[
                [60.0, 40.0], [60.0, 40.0], [70.0, 40.0], [70.5, 40.0],
                [70.5, 40.0], [80.5, 40.0], [81.0, 40.0],
                [81.0, 40.0], [91.0, 40.0], [91.5, 40.0], [91.5, 40.0],
            ],
            timestamps=[
                "00:00:00.000", "00:10:00.000", "00:10:01.000", "00:10:02.000",
                "00:00:00.000", "00:00:01.000", "00:00:02.000",
                "00:10:00.000", "00:10:01.000", "00:10:02.000", "00:45:00.000",
            ],
            periods=[
                1, 1, 1, 1,
                2, 2, 2, 2, 2, 2, 2,
            ],
        )
        oirc = compute_oirc(events, player_id=1)
        assert oirc == pytest.approx(10.0)

    def test_sideways_run_zero_directness(self) -> None:
        """A sideways run has RD=0, so it contributes nothing to OIRC."""
        events = _make_player_events(
            locations=[
                [60.0, 40.0], [60.0, 50.0], [60.0, 51.0],
            ],
            timestamps=[
                "00:00:00.000", "00:00:01.000", "00:00:02.000",
            ],
        )
        # Only one fast step: 60,40 → 60,50 at t=1, v=10
        # Then 60,50 → 60,51 at t=2, v=1 (slow)
        # So one run of displacement 10, RD = 0 (sideways vs goal)
        oirc = compute_oirc(events, player_id=1)
        assert oirc == pytest.approx(0.0)

    def test_no_runs_returns_zero(self) -> None:
        events = _make_player_events()
        oirc = compute_oirc(events, player_id=1)
        assert oirc == pytest.approx(0.0)

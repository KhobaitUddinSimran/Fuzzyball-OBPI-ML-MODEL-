"""Movement-quality metrics: M4 OBR90 and M2 OIRC."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from obpi.utils import geometry, kinematics

_SET_PIECE_PATTERNS = {
    "From Corner",
    "From Free Kick",
    "From Throw In",
    "From Goal Kick",
    "From Penalty",
    "From Kick Off",
}


def _exclude_set_pieces(events: pd.DataFrame) -> pd.DataFrame:
    """Drop events that occur during set-piece phases."""
    if "play_pattern_name" in events.columns:
        return events[
            ~events["play_pattern_name"].isin(_SET_PIECE_PATTERNS)
        ].reset_index(drop=True)

    def _is_set_piece(pattern: Any) -> bool:
        if isinstance(pattern, dict):
            return pattern.get("name", "") in _SET_PIECE_PATTERNS
        if isinstance(pattern, str):
            return pattern in _SET_PIECE_PATTERNS
        return False

    mask = ~events["play_pattern"].apply(_is_set_piece)
    return events[mask].reset_index(drop=True)


def _estimate_minutes(events: pd.DataFrame) -> float:
    """Estimate minutes played from event timestamps.

    Returns the span from first to last event, capped at 90 minutes.
    """
    if events.empty:
        return 0.0
    first = kinematics._parse_timestamp(events.iloc[0]["timestamp"])
    last = kinematics._parse_timestamp(events.iloc[-1]["timestamp"])
    minutes = (last - first) / 60.0
    return min(minutes, 90.0)


def compute_obr90(
    events: pd.DataFrame,
    player_id: int,
    minutes_played: float | None = None,
    v_threshold: float = 2.5,
    duration_threshold: float = 0.4,
    max_dt: float = 1.5,
    exclude_set_pieces: bool = True,
) -> float:
    """Compute Off-Ball Runs per 90 minutes (M4).

    A "run" is a contiguous sequence of fast steps (``v > v_threshold``)
    lasting at least ``duration_threshold`` seconds. Set-piece contexts are
    excluded.

    Args:
        events: StatsBomb-style events DataFrame.
        player_id: Target player identifier.
        minutes_played: Minutes the player was on the pitch. If ``None``,
            estimated from event timestamps (capped at 90).
        v_threshold: Minimum speed (m/s) for a fast step.
        duration_threshold: Minimum run duration (s).
        max_dt: Maximum reliable inter-event gap for velocity inference.
        exclude_set_pieces: Whether to remove set-piece phases before analysis.

    Returns:
        OBR90 value. Returns ``0.0`` if no runs detected or minutes played
        is zero.
    """
    events_clean = _exclude_set_pieces(events) if exclude_set_pieces else events
    vel_df = kinematics.infer_velocity(events_clean, player_id, max_dt=max_dt)
    runs = kinematics.detect_runs(vel_df, v_threshold, duration_threshold)

    n_runs = len(runs)
    if n_runs == 0:
        return 0.0

    minutes = minutes_played if minutes_played is not None else _estimate_minutes(events_clean)
    if minutes <= 0.0:
        return 0.0

    return (n_runs / minutes) * 90.0


def compute_oirc(
    events: pd.DataFrame,
    player_id: int,
    goal_center: tuple[float, float] = (120.0, 40.0),
    v_threshold: float = 2.5,
    duration_threshold: float = 0.4,
    max_dt: float = 1.5,
    exclude_set_pieces: bool = True,
) -> float:
    """Compute Off-Ball Impact Run Coefficient (M2).

    For each detected run, the contribution is ``displacement * directness``,
    where directness is the cosine similarity between the run vector and the
    vector toward the opponent goal (clipped to ``[0, 1]``).

    Args:
        events: StatsBomb-style events DataFrame.
        player_id: Target player identifier.
        goal_center: Opponent goal center ``(x, y)``.
        v_threshold: Minimum speed (m/s) for a fast step.
        duration_threshold: Minimum run duration (s).
        max_dt: Maximum reliable inter-event gap for velocity inference.
        exclude_set_pieces: Whether to remove set-piece phases before analysis.

    Returns:
        OIRC value. Returns ``0.0`` if no runs detected.
    """
    events_clean = _exclude_set_pieces(events) if exclude_set_pieces else events
    vel_df = kinematics.infer_velocity(events_clean, player_id, max_dt=max_dt)
    runs = kinematics.detect_runs(vel_df, v_threshold, duration_threshold)

    if not runs:
        return 0.0

    total = 0.0
    for run in runs:
        run_vec = np.array([run["dx"], run["dy"]], dtype=np.float64)
        # Goal vector from the *start* of the run
        start_x = vel_df.at[run["start_idx"], "x"]
        start_y = vel_df.at[run["start_idx"], "y"]
        goal_vec = geometry.to_goal_vector([start_x, start_y], goal_center)
        rd = geometry.run_directness(run_vec, goal_vec)
        total += run["displacement"] * rd

    return total / len(runs)

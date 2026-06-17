"""Velocity inference from discrete event data with Δt gating."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

_SET_PIECE_PATTERNS = {
    "From Corner",
    "From Free Kick",
    "From Throw In",
    "From Goal Kick",
    "From Penalty",
    "From Kick Off",
}


def _parse_timestamp(ts: str) -> float:
    """Convert StatsBomb ``HH:MM:SS.sss`` timestamp to total seconds."""
    h, m, s = ts.split(":")
    return float(h) * 3600 + float(m) * 60 + float(s)


def infer_velocity(
    events: pd.DataFrame,
    player_id: int | None = None,
    max_dt: float = 1.5,
) -> pd.DataFrame:
    """Infer per-event velocity from discrete event locations and timestamps.

    Velocity is computed between consecutive events for the same player.
    If ``Δt > max_dt`` or ``Δt <= 0``, the velocity is marked as unreliable
    (NaN) to prevent spurious finite-difference approximations.

    Args:
        events: StatsBomb-style events DataFrame with ``timestamp``,
            ``location``, ``period``, and optionally ``play_pattern``.
        player_id: If provided, filter to this player's events only.
        max_dt: Maximum reliable inter-event gap in seconds.

    Returns:
        Copy of ``events`` with added columns:
        ``time_sec``, ``x``, ``y``, ``dt``, ``dx``, ``dy``,
        ``velocity``, ``vx``, ``vy``.
    """
    df = events.copy()

    if player_id is not None:
        if "player_id" in df.columns:
            player_ids = pd.to_numeric(df["player_id"], errors="coerce")
            df = df[player_ids == player_id]
        else:
            df = df[
                df["player"].apply(lambda p: p.get("id") if isinstance(p, dict) else None)
                == player_id
            ]

    if df.empty:
        df = df.copy()
        for col in ["time_sec", "x", "y", "dt", "dx", "dy", "velocity", "vx", "vy"]:
            df[col] = pd.Series(dtype=float)
        return df

    # Chronological sort
    df = df.sort_values(["period", "timestamp"]).reset_index(drop=True)

    # Convert timestamps to continuous seconds
    df["time_sec"] = df["timestamp"].apply(_parse_timestamp).astype(float)
    df["period_offset"] = df["period"].apply(lambda p: 0 if p == 1 else 45 * 60).astype(float)
    df["time_sec"] = df["time_sec"] + df["period_offset"]

    # Extract x, y from flat columns when available, otherwise from location lists.
    if {"location_x", "location_y"}.issubset(df.columns):
        df["x"] = pd.to_numeric(df["location_x"], errors="coerce")
        df["y"] = pd.to_numeric(df["location_y"], errors="coerce")
    else:
        df["x"] = df["location"].apply(
            lambda loc: loc[0] if isinstance(loc, (list, tuple)) else np.nan
        )
        df["y"] = df["location"].apply(
            lambda loc: loc[1] if isinstance(loc, (list, tuple)) else np.nan
        )

    # Finite differences
    df["dt"] = df["time_sec"].diff()
    df["dx"] = df["x"].diff()
    df["dy"] = df["y"].diff()

    # Speed and component velocities
    df["velocity"] = np.sqrt(df["dx"] ** 2 + df["dy"] ** 2) / df["dt"]
    df["vx"] = df["dx"] / df["dt"]
    df["vy"] = df["dy"] / df["dt"]

    # Gate: unreliable Δt
    unreliable = (df["dt"] > max_dt) | (df["dt"] <= 0) | df["dt"].isna()
    df.loc[unreliable, ["velocity", "vx", "vy"]] = np.nan

    # Gate: missing location
    missing_loc = df["x"].isna() | df["y"].isna()
    df.loc[missing_loc, ["velocity", "vx", "vy"]] = np.nan

    return df


def detect_runs(
    velocity_df: pd.DataFrame,
    v_threshold: float = 2.5,
    duration_threshold: float = 0.4,
) -> list[dict[str, Any]]:
    """Detect off-ball runs from a velocity-annotated event DataFrame.

    A run is a maximal contiguous sequence of fast steps (``v > v_threshold``)
    whose total duration is at least ``duration_threshold`` seconds.

    Args:
        velocity_df: DataFrame produced by :func:`infer_velocity`.
        v_threshold: Minimum speed (m/s) to qualify as a fast step.
        duration_threshold: Minimum total duration (s) of a run.

    Returns:
        List of run dicts with keys ``start_idx``, ``end_idx``, ``duration``,
        ``dx``, ``dy``, ``displacement``.
    """
    runs: list[dict[str, Any]] = []
    in_run = False
    run_start = 0
    run_duration = 0.0

    for i, row in velocity_df.iterrows():
        v = row["velocity"]
        dt = row["dt"]

        is_fast = not pd.isna(v) and v > v_threshold

        if is_fast:
            if not in_run:
                run_start = int(i)
                in_run = True
            if not pd.isna(dt):
                run_duration += float(dt)
        else:
            if in_run:
                if run_duration >= duration_threshold:
                    end_idx = int(i) - 1
                    _s = max(0, run_start - 1)
                    start_x = velocity_df.at[_s, "x"]
                    start_y = velocity_df.at[_s, "y"]
                    end_x = velocity_df.at[end_idx, "x"]
                    end_y = velocity_df.at[end_idx, "y"]
                    dx = end_x - start_x if not pd.isna(start_x) and not pd.isna(end_x) else 0.0
                    dy = end_y - start_y if not pd.isna(start_y) and not pd.isna(end_y) else 0.0
                    displacement = float(np.sqrt(dx**2 + dy**2))
                    runs.append(
                        {
                            "start_idx": run_start,
                            "end_idx": end_idx,
                            "duration": run_duration,
                            "dx": dx,
                            "dy": dy,
                            "displacement": displacement,
                        }
                    )
                in_run = False
                run_duration = 0.0

    # Handle run that extends to final row
    if in_run and run_duration >= duration_threshold:
        end_idx = len(velocity_df) - 1
        _s = max(0, run_start - 1)
        start_x = velocity_df.at[_s, "x"]
        start_y = velocity_df.at[_s, "y"]
        end_x = velocity_df.at[end_idx, "x"]
        end_y = velocity_df.at[end_idx, "y"]
        dx = end_x - start_x if not pd.isna(start_x) and not pd.isna(end_x) else 0.0
        dy = end_y - start_y if not pd.isna(start_y) and not pd.isna(end_y) else 0.0
        displacement = float(np.sqrt(dx**2 + dy**2))
        runs.append(
            {
                "start_idx": run_start,
                "end_idx": end_idx,
                "duration": run_duration,
                "dx": dx,
                "dy": dy,
                "displacement": displacement,
            }
        )

    return runs

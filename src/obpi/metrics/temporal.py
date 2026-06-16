"""Temporal metrics: M8 LPC and M9 CBI."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from obpi.utils.xt_model import XTModel

_BALL_RECEIPT_NAMES = {"Ball Receipt*", "Ball Receipt"}
_PASS_NAMES = {"Pass"}


def _is_receipt(row: pd.Series) -> bool:
    ev_type = row.get("type", {})
    if isinstance(ev_type, dict):
        return ev_type.get("name", "") in _BALL_RECEIPT_NAMES
    if isinstance(ev_type, str):
        return ev_type in _BALL_RECEIPT_NAMES
    return False


def _is_pass(row: pd.Series) -> bool:
    ev_type = row.get("type", {})
    if isinstance(ev_type, dict):
        return ev_type.get("name", "") in _PASS_NAMES
    if isinstance(ev_type, str):
        return ev_type in _PASS_NAMES
    return False


def _get_player_id(row: pd.Series) -> int | None:
    player = row.get("player", {})
    if isinstance(player, dict):
        return player.get("id")
    return None


def _parse_ts(ts: str) -> float:
    """Convert StatsBomb ``HH:MM:SS.sss`` to total seconds."""
    h, m, s = ts.split(":")
    return float(h) * 3600 + float(m) * 60 + float(s)


def compute_lpc(
    events: pd.DataFrame,
    player_id: int,
    xt_model: XTModel,
    min_dt: float = 1.2,
    max_vel: float = 0.5,
) -> float:
    """Compute Layoff Pause Coefficient (M8).

    Identifies receipt events where the same player pauses (velocity <
    ``max_vel`` and Δt ≥ ``min_dt``) before making a pass that improves xT.

    Args:
        events: StatsBomb-style events DataFrame sorted chronologically.
        player_id: Target player identifier.
        xt_model: Instance of :class:`~obpi.utils.xt_model.XTModel`.
        min_dt: Minimum pause duration in seconds.
        max_vel: Maximum velocity (m/s) during the pause.

    Returns:
        Ratio of successful layoff pauses to total receipt→pass sequences.
        Returns ``0.0`` if no qualifying sequences.
    """
    if events.empty:
        return 0.0

    df = events.copy()
    df["is_receipt"] = df.apply(_is_receipt, axis=1)
    df["is_pass"] = df.apply(_is_pass, axis=1)
    df["player_id"] = df.apply(_get_player_id, axis=1)
    df["time_sec"] = df["timestamp"].apply(_parse_ts)
    df["period_offset"] = df["period"].apply(lambda p: 0 if p == 1 else 45 * 60)
    df["time_sec"] = df["time_sec"] + df["period_offset"]

    player_events = df[df["player_id"] == player_id].reset_index(drop=True)
    if len(player_events) < 2:
        return 0.0

    count = 0
    opportunities = 0

    for i in range(len(player_events) - 1):
        row = player_events.iloc[i]
        next_row = player_events.iloc[i + 1]

        if not row["is_receipt"] or not next_row["is_pass"]:
            continue

        opportunities += 1
        dt = next_row["time_sec"] - row["time_sec"]
        if dt < min_dt:
            continue

        # Velocity between receipt and next pass (simple finite difference)
        loc_a = row.get("location")
        loc_b = next_row.get("location")
        if loc_a is None or loc_b is None:
            continue
        dx = float(loc_b[0]) - float(loc_a[0])
        dy = float(loc_b[1]) - float(loc_a[1])
        vel = np.sqrt(dx**2 + dy**2) / dt if dt > 0 else float("inf")
        if vel >= max_vel:
            continue

        # xT improvement on the pass
        pass_end = next_row.get("pass", {})
        end_loc = (
            pass_end.get("end_location") if isinstance(pass_end, dict) else None
        )
        if end_loc is None:
            end_loc = loc_b
        dxt = xt_model.delta_xt(loc_a, end_loc)
        if dxt > 0:
            count += 1

    return count / opportunities if opportunities > 0 else 0.0


def compute_cbi(
    events: pd.DataFrame,
    frames: list[dict[str, Any]],
    player_id: int,
    angle_threshold: float = 30.0,
    lane_buffer: float = 1.5,
) -> float:
    """Compute Cutback Intelligence (M9).

    For each receipt opportunity, checks whether the player's run vector
    is aligned with the ball-to-player vector (within ``angle_threshold``)
    and whether the passing lane is open (no opponent within ``lane_buffer``
    metres of the line).

    Args:
        events: StatsBomb-style events DataFrame.
        frames: 360 freeze-frame dicts aligned with receipt events.
        player_id: Target player identifier.
        angle_threshold: Maximum allowable angle deviation in degrees.
        lane_buffer: Minimum clearance from passing line in metres.

    Returns:
        CBI value in ``[0.0, 1.0]``. Returns ``0.0`` if no opportunities.
    """
    if events.empty or not frames:
        return 0.0

    df = events.copy()
    df["is_receipt"] = df.apply(_is_receipt, axis=1)
    df["player_id"] = df.apply(_get_player_id, axis=1)

    receipts = df[(df["is_receipt"]) & (df["player_id"] == player_id)]
    if receipts.empty:
        return 0.0

    count = 0
    opportunities = 0

    for i, (_, row) in enumerate(receipts.iterrows()):
        opportunities += 1
        loc = row.get("location")
        if loc is None or i >= len(frames):
            continue

        # Ball-to-player vector: from previous event location (pass origin) to receipt
        prev_events = df[df.index < row.name]
        if prev_events.empty:
            continue
        prev = prev_events.iloc[-1]
        ball_loc = prev.get("location")
        if ball_loc is None:
            continue
        ball_to_player = np.array(
            [float(loc[0]) - float(ball_loc[0]), float(loc[1]) - float(ball_loc[1])],
            dtype=np.float64,
        )

        # Run vector: receiver's own previous location → receipt location
        receiver_prev = prev_events[prev_events.apply(
            lambda r, pid=player_id: (
                _get_player_id(r) == pid
                and r.get("location") is not None
            ),
            axis=1,
        )]
        if receiver_prev.empty:
            continue
        run_origin = receiver_prev.iloc[-1].get("location")
        if run_origin is None:
            continue
        run_vec = np.array(
            [float(loc[0]) - float(run_origin[0]), float(loc[1]) - float(run_origin[1])],
            dtype=np.float64,
        )

        # Alignment check
        run_norm = float(np.linalg.norm(run_vec))
        ball_norm = float(np.linalg.norm(ball_to_player))
        if run_norm == 0.0 or ball_norm == 0.0:
            continue
        cos_angle = float(np.dot(run_vec, ball_to_player) / (run_norm * ball_norm))
        angle = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
        if angle > angle_threshold:
            continue

        # Lane check: no opponent within buffer of the line from ball to player
        frame = frames[i]
        ff = frame.get("freeze_frame", [])
        opponents = [
            p["location"]
            for p in ff
            if not p.get("teammate", True) and p.get("location") is not None
        ]

        if not _passing_lane_open(ball_loc, loc, opponents, lane_buffer):
            continue

        count += 1

    return count / opportunities if opportunities > 0 else 0.0


def _passing_lane_open(
    ball_loc: list[float],
    player_loc: list[float],
    opponents: list[list[float]],
    buffer: float,
) -> bool:
    """Return True if no opponent is within ``buffer`` metres of the passing line."""
    bx, by = float(ball_loc[0]), float(ball_loc[1])
    px, py = float(player_loc[0]), float(player_loc[1])
    dx = px - bx
    dy = py - by
    seg_len = np.sqrt(dx**2 + dy**2)
    if seg_len == 0.0:
        return True

    for opp in opponents:
        ox, oy = float(opp[0]), float(opp[1])
        # Distance from point to line segment
        t = max(0.0, min(1.0, ((ox - bx) * dx + (oy - by) * dy) / (seg_len**2)))
        closest_x = bx + t * dx
        closest_y = by + t * dy
        dist = np.sqrt((ox - closest_x) ** 2 + (oy - closest_y) ** 2)
        if dist < buffer:
            return False
    return True

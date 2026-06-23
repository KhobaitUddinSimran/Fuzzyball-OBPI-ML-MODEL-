"""Receiving-quality metrics: M5 RBTL, M6 RUP, and M3 BRPC."""

from __future__ import annotations

from typing import Any

import pandas as pd
from shapely.geometry import Point

from obpi.data.preprocessor import nearest_opponent_distance
from obpi.utils import geometry

_BALL_RECEIPT_NAMES = {"Ball Receipt*", "Ball Receipt"}


def get_receipt_events(events: pd.DataFrame, player_id: int) -> pd.DataFrame:
    """Filter events to Ball Receipt events for a specific player.

    Args:
        events: StatsBomb-style events DataFrame.
        player_id: Target player identifier.

    Returns:
        Subset of ``events`` where the event type is a ball receipt and the
        player matches ``player_id``.
    """
    if {"type_name", "player_id"}.issubset(events.columns):
        player_ids = pd.to_numeric(events["player_id"], errors="coerce")
        receipts = events[
            events["type_name"].isin(_BALL_RECEIPT_NAMES) & (player_ids == player_id)
        ]
        return receipts.reset_index(drop=True)

    def _is_receipt(row: pd.Series) -> bool:
        ev_type = row.get("type", {})
        if isinstance(ev_type, dict):
            return ev_type.get("name", "") in _BALL_RECEIPT_NAMES
        if isinstance(ev_type, str):
            return ev_type in _BALL_RECEIPT_NAMES
        return False

    def _player_match(row: pd.Series) -> bool:
        player = row.get("player", {})
        if isinstance(player, dict):
            return player.get("id") == player_id
        return False

    mask = events.apply(lambda r: _is_receipt(r) and _player_match(r), axis=1)
    return events[mask].reset_index(drop=True)


def compute_rbtl(
    events: pd.DataFrame,
    player_id: int,
    def_line_x: float = 60.0,
    back_line_x: float = 120.0,
) -> float:
    """Compute Receipts Behind The Line (M5).

    Ratio of a player's ball-receipt events whose location falls inside the
    half-space polygon between ``def_line_x`` and ``back_line_x``.

    Args:
        events: StatsBomb-style events DataFrame.
        player_id: Target player identifier.
        def_line_x: x-coordinate of the defensive line.
        back_line_x: x-coordinate of the opponent goal line.

    Returns:
        RBTL in ``[0.0, 1.0]``. Returns ``0.0`` if no receipts found.
    """
    receipts = get_receipt_events(events, player_id)
    if receipts.empty:
        return 0.0

    poly = geometry.get_half_space_polygon(def_line_x, back_line_x)
    inside = 0
    for _, row in receipts.iterrows():
        loc = row.get("location")
        if loc is not None and len(loc) >= 2 and poly.contains(
            Point(float(loc[0]), float(loc[1]))
        ):
            inside += 1
    return inside / len(receipts)


def compute_rup(
    events: pd.DataFrame,
    player_id: int,
    frames: list[dict[str, Any]] | None = None,
    frames_by_event_id: dict[str, list[dict[str, Any]]] | None = None,
    proximity_threshold: float = 2.5,
) -> float:
    """Compute Receipt Under Pressure (M6).

    Ratio of ball receipts where the player is under pressure.

    Primary source: the ``under_pressure`` boolean flag on the event.
    Fallback (when flag is missing): check the corresponding 360 freeze frame
    for an opponent within ``proximity_threshold`` metres.

    Args:
        events: StatsBomb-style events DataFrame.
        player_id: Target player identifier.
        frames: Optional list of 360 freeze-frame dicts, one per receipt.
        proximity_threshold: Fallback distance threshold in metres.

    Returns:
        RUP in ``[0.0, 1.0]``. Returns ``0.0`` if no receipts found.
    """
    receipts = get_receipt_events(events, player_id)
    if receipts.empty:
        return 0.0

    under_pressure = 0
    for _, row in receipts.iterrows():
        up = row.get("under_pressure")
        if up is not None:
            if bool(up):
                under_pressure += 1
            continue

        # Fallback: check 360 frame proximity
        event_frames = _frames_for_event(row, frames_by_event_id)
        loc = row.get("location")
        if event_frames and loc is not None:
            dist = nearest_opponent_distance(event_frames[0], loc)
            if dist <= proximity_threshold:
                under_pressure += 1

    return under_pressure / len(receipts)


def compute_brpc(
    events: pd.DataFrame,
    frames: list[dict[str, Any]],
    player_id: int,
    frames_by_event_id: dict[str, list[dict[str, Any]]] | None = None,
    pressure_threshold: float = 5.0,
    cone_angle: float = 45.0,
    cone_length: float = 15.0,
) -> float:
    """Compute Ball Receipt Pressure Counter (M3).

    A receipt qualifies if:
    1. The nearest opponent is farther than ``pressure_threshold`` metres.
    2. The forward cone (``angle × length``) contains no opponents.

    Args:
        events: StatsBomb-style events DataFrame.
        frames: 360 freeze-frame dicts aligned 1-to-1 with receipts.
        player_id: Target player identifier.
        pressure_threshold: Minimum distance to nearest opponent in metres.
        cone_angle: Forward cone total opening angle in degrees.
        cone_length: Forward cone length in metres.

    Returns:
        BRPC in ``[0.0, 1.0]``. Returns ``0.0`` if no receipts found.
    """
    receipts = get_receipt_events(events, player_id)
    if receipts.empty:
        return 0.0

    count = 0
    for _, row in receipts.iterrows():
        event_frames = _frames_for_event(row, frames_by_event_id)
        if not event_frames:
            continue
        loc = row.get("location")
        if loc is None:
            continue

        frame = event_frames[0]
        dist = nearest_opponent_distance(frame, loc)
        if dist <= pressure_threshold:
            continue

        ff = frame.get("freeze_frame", [])
        opponents = [
            p["location"]
            for p in ff
            if not p.get("teammate", True) and p.get("location") is not None
        ]
        if not geometry.is_lane_open(
            loc, cone_angle=cone_angle, cone_length=cone_length, opponents=opponents
        ):
            continue

        count += 1

    return count / len(receipts)


def _frames_for_event(
    row: pd.Series,
    frames_by_event_id: dict[str, list[dict[str, Any]]] | None,
) -> list[dict[str, Any]]:
    """Return exact 360 frames matched to the event id."""
    if not frames_by_event_id:
        return []
    event_id = row.get("id")
    if not event_id:
        return []
    return frames_by_event_id.get(str(event_id), [])

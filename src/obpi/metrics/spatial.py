"""Spatial metrics: M7 SCI and M1 SC."""

from typing import Any

import numpy as np
from numpy.typing import NDArray

from obpi.utils import geometry


def _get_team_locations(
    frame: dict[str, Any], teammate: bool
) -> NDArray[np.float64]:
    """Extract locations of teammates or opponents from a freeze frame."""
    ff = frame.get("freeze_frame", [])
    locs: list[list[float]] = [
        p["location"]
        for p in ff
        if p.get("teammate", True) == teammate and p.get("location") is not None
    ]
    if locs:
        return np.array(locs, dtype=np.float64)
    return np.array([], dtype=np.float64).reshape(0, 2)


def compute_sci(
    frames_before: list[dict[str, Any]],
    frames_after: list[dict[str, Any]],
    player_id: int | None = None,
) -> float:
    """Compute Space Control Improvement (M7).

    Compares attacking-team Voronoi cell areas before and after an action.

    Args:
        frames_before: 360 freeze frames before the action.
        frames_after: 360 freeze frames after the action.
        player_id: Unused placeholder for API consistency.

    Returns:
        Mean area gain per teammate in square metres.
        Returns ``0.0`` if insufficient frames or no teammates.
    """
    if not frames_before or not frames_after:
        return 0.0

    gains: list[float] = []
    for before, after in zip(frames_before, frames_after):
        team_before = _get_team_locations(before, teammate=True)
        team_after = _get_team_locations(after, teammate=True)

        if len(team_before) < 3 or len(team_after) < 3:
            continue

        areas_before = geometry.voronoi_areas(team_before)
        areas_after = geometry.voronoi_areas(team_after)

        n = min(len(areas_before), len(areas_after))
        if n == 0:
            continue

        gains.extend([areas_after[i] - areas_before[i] for i in range(n)])

    return float(np.mean(gains)) if gains else 0.0


def compute_sc(
    frames_before: list[dict[str, Any]],
    frames_after: list[dict[str, Any]],
    player_location: list[float] | NDArray[np.float64],
    box_size: tuple[float, float] = (10.0, 10.0),
    threshold: float = 1.5,
) -> float:
    """Compute Screen Count (M1).

    Measures whether defenders near the player shift more than the global
    defender average between before/after frames. An adjusted shift greater
    than ``threshold`` metres counts as a successful screen.

    Args:
        frames_before: 360 freeze frames before the action.
        frames_after: 360 freeze frames after the action.
        player_location: ``[x, y]`` of the screening player.
        box_size: ``(width, height)`` of the local analysis box around the player.
        threshold: Minimum adjusted shift (m) to count as a screen.

    Returns:
        Mean screen count across all frame pairs (0.0–1.0).
    """
    if not frames_before or not frames_after:
        return 0.0

    px, py = float(player_location[0]), float(player_location[1])
    hw, hh = box_size[0] / 2.0, box_size[1] / 2.0

    def _in_box(points: NDArray[np.float64]) -> NDArray[np.float64]:
        if len(points) == 0:
            return points
        mask = (
            (points[:, 0] >= px - hw)
            & (points[:, 0] <= px + hw)
            & (points[:, 1] >= py - hh)
            & (points[:, 1] <= py + hh)
        )
        return points[mask]

    screens: list[float] = []
    for before, after in zip(frames_before, frames_after):
        opp_before = _get_team_locations(before, teammate=False)
        opp_after = _get_team_locations(after, teammate=False)

        if len(opp_before) == 0 or len(opp_after) == 0:
            continue

        local_before = _in_box(opp_before)
        local_after = _in_box(opp_after)

        local_before_mean = (
            np.mean(local_before, axis=0)
            if len(local_before) > 0
            else np.array([px, py])
        )
        local_after_mean = (
            np.mean(local_after, axis=0)
            if len(local_after) > 0
            else np.array([px, py])
        )
        global_before_mean = np.mean(opp_before, axis=0)
        global_after_mean = np.mean(opp_after, axis=0)

        local_shift = float(np.linalg.norm(local_after_mean - local_before_mean))
        global_shift = float(np.linalg.norm(global_after_mean - global_before_mean))
        adjusted_shift = local_shift - global_shift

        screens.append(1.0 if adjusted_shift > threshold else 0.0)

    return float(np.mean(screens)) if screens else 0.0

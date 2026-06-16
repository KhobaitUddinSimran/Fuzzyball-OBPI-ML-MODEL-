"""Preprocessing utilities: ConvexHull clipping and Δt gating."""

from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon


class ConvexHullClipper:
    """Clip geometric objects to the convex hull of visible players.

    This mitigates broadcast blind-spot bias: off-camera defenders are missing,
    so unclipped Voronoi diagrams overestimate space.
    """

    def __init__(
        self,
        pitch_bounds: tuple[float, float, float, float] = (0.0, 0.0, 120.0, 80.0),
    ) -> None:
        """Initialize with pitch bounds.

        Args:
            pitch_bounds: ``(xmin, ymin, xmax, ymax)`` in metres.
        """
        self.pitch_bounds = pitch_bounds
        self.pitch_polygon = Polygon(
            [
                (pitch_bounds[0], pitch_bounds[1]),
                (pitch_bounds[2], pitch_bounds[1]),
                (pitch_bounds[2], pitch_bounds[3]),
                (pitch_bounds[0], pitch_bounds[3]),
            ]
        )

    def visible_hull(self, points: NDArray[np.float64]) -> Polygon:
        """Compute the convex hull of visible player points, clipped to pitch.

        Args:
            points: ``(N, 2)`` array of ``[x, y]`` coordinates.

        Returns:
            Clipped convex-hull polygon.
        """
        if len(points) < 3:
            return self.pitch_polygon
        hull = ConvexHull(points)
        hull_points = points[hull.vertices]
        hull_poly = Polygon(hull_points)
        return hull_poly.intersection(self.pitch_polygon)

    def clip_voronoi_cells(
        self, cells: list[Polygon], player_points: NDArray[np.float64]
    ) -> list[Polygon]:
        """Clip a list of Voronoi cell polygons to the visible hull.

        Args:
            cells: List of Shapely polygons, one per player.
            player_points: ``(N, 2)`` array of player positions.

        Returns:
            Clipped cell polygons.
        """
        hull = self.visible_hull(player_points)
        return [cell.intersection(hull) for cell in cells]


class DeltaTGate:
    """Gate velocity-based calculations when inter-event spacing is too sparse.

    StatsBomb event data is discrete. If ``Δt > 1.5 s`` between frames,
    finite-difference velocity approximations become unreliable.
    """

    def __init__(self, max_dt: float = 1.5) -> None:
        """Initialize Δt gate.

        Args:
            max_dt: Maximum reliable ``Δt`` in seconds.
        """
        self.max_dt = max_dt

    def is_reliable(self, dt: float) -> bool:
        """Return True if the time gap is small enough for velocity inference."""
        return 0.0 < dt <= self.max_dt

    def filter_pairs(
        self, locations: NDArray[np.float64], timestamps: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """Filter location/timestamp pairs to those with reliable Δt.

        Args:
            locations: ``(N, 2)`` array of ``[x, y]`` positions.
            timestamps: ``(N,)`` array of timestamps in seconds.

        Returns:
            Tuple of ``(filtered_locations, filtered_timestamps, dts)``.
        """
        if len(locations) != len(timestamps):
            raise ValueError("locations and timestamps must have the same length")
        if len(timestamps) < 2:
            return locations, timestamps, np.array([])

        dts = np.diff(timestamps)
        mask = np.concatenate(([True], [self.is_reliable(dt) for dt in dts]))
        return locations[mask], timestamps[mask], dts[dts <= self.max_dt]


def nearest_opponent_distance(
    frame: dict[str, Any], player_location: list[float] | NDArray[np.float64]
) -> float:
    """Return Euclidean distance to the nearest opponent in a 360 freeze frame.

    Args:
        frame: A freeze-frame dict with a ``freeze_frame`` list of player dicts.
        player_location: ``[x, y]`` of the target player.

    Returns:
        Minimum distance to any non-teammate in the frame, or ``float("inf")``
        if no opponents are present.
    """
    ff = frame.get("freeze_frame", [])
    if not ff:
        return float("inf")
    px, py = float(player_location[0]), float(player_location[1])
    min_dist = float("inf")
    for p in ff:
        if p.get("teammate", True):
            continue
        loc = p.get("location")
        if loc is None or len(loc) < 2:
            continue
        ox, oy = float(loc[0]), float(loc[1])
        dist = ((px - ox) ** 2 + (py - oy) ** 2) ** 0.5
        if dist < min_dist:
            min_dist = dist
    return min_dist

"""Geometric primitives for OBPI metrics: vectors, angles, polygons."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import Voronoi
from shapely.geometry import Point, Polygon


def run_directness(v_run: NDArray[np.float64], v_goal: NDArray[np.float64]) -> float:
    """Compute run directness as cosine similarity clipped to [0, 1].

    Args:
        v_run: 2D run displacement vector ``[dx, dy]``.
        v_goal: 2D vector toward opponent goal ``[gx, gy]``.

    Returns:
        Directness in ``[0, 1]`` where 1 = perfectly toward goal,
        0 = perpendicular or backward.
    """
    norm_run = float(np.linalg.norm(v_run))
    norm_goal = float(np.linalg.norm(v_goal))
    if norm_run == 0.0 or norm_goal == 0.0:
        return 0.0
    cos_sim = float(np.dot(v_run, v_goal) / (norm_run * norm_goal))
    return max(0.0, min(1.0, cos_sim))


def to_goal_vector(
    location: NDArray[np.float64] | list[float],
    goal_center: tuple[float, float] = (120.0, 40.0),
) -> NDArray[np.float64]:
    """Return vector from a position to the opponent goal center.

    Args:
        location: Player ``[x, y]`` on a 120x80 metre pitch.
        goal_center: Opponent goal center coordinates.

    Returns:
        2D vector ``[goal_x - x, goal_y - y]``.
    """
    arr = np.asarray(location, dtype=np.float64)
    return np.array([goal_center[0] - arr[0], goal_center[1] - arr[1]], dtype=np.float64)


def get_half_space_polygon(
    def_line_x: float = 60.0,
    back_line_x: float = 120.0,
    pitch_width: float = 80.0,
) -> Polygon:
    """Return the polygon for the half-space between the defensive line and back line.

    Args:
        def_line_x: x-coordinate of the defensive line.
        back_line_x: x-coordinate of the opponent goal line.
        pitch_width: Total width of the pitch in metres.

    Returns:
        Shapely Polygon covering ``[def_line_x, back_line_x] × [0, pitch_width]``.
    """
    return Polygon(
        [
            (def_line_x, 0.0),
            (back_line_x, 0.0),
            (back_line_x, pitch_width),
            (def_line_x, pitch_width),
        ]
    )


def forward_cone(
    origin: list[float] | NDArray[np.float64],
    angle: float = 45.0,
    length: float = 15.0,
) -> Polygon:
    """Build a forward-facing cone polygon centred on the opponent goal.

    The cone opens toward positive x (attacking direction) with its apex at
    ``origin``.

    Args:
        origin: ``[x, y]`` apex coordinates.
        angle: Total opening angle in degrees.
        length: Cone length in metres.

    Returns:
        Shapely Polygon representing the cone.
    """
    x, y = float(origin[0]), float(origin[1])
    half = math.radians(angle / 2.0)
    p1 = (x + length * math.cos(half), y + length * math.sin(half))
    p2 = (x + length * math.cos(half), y - length * math.sin(half))
    return Polygon([(x, y), p1, p2])


def is_lane_open(
    origin: list[float] | NDArray[np.float64],
    cone_angle: float = 45.0,
    cone_length: float = 15.0,
    opponents: list[list[float]] | NDArray[np.float64] | None = None,
) -> bool:
    """Check whether any opponent lies inside the forward cone.

    Args:
        origin: Player ``[x, y]`` position.
        cone_angle: Total opening angle in degrees.
        cone_length: Cone length in metres.
        opponents: List or array of opponent ``[x, y]`` positions.

    Returns:
        ``True`` if no opponent is inside the cone (lane is open),
        ``False`` otherwise.
    """
    if opponents is None or len(opponents) == 0:  # noqa: PLC1901
        return True
    cone = forward_cone(origin, cone_angle, cone_length)
    return not any(
        cone.contains(Point(float(opp[0]), float(opp[1]))) for opp in opponents
    )


def voronoi_areas(
    points: NDArray[np.float64],
    clip_bounds: tuple[float, float, float, float] = (0.0, 0.0, 120.0, 80.0),
) -> list[float]:
    """Compute Voronoi cell areas for a set of 2D points, clipped to a bounding box.

    Args:
        points: ``(N, 2)`` array of player positions.
        clip_bounds: ``(xmin, ymin, xmax, ymax)`` pitch bounds.

    Returns:
        List of clipped cell areas, one per input point.
    """
    if len(points) < 3:
        return [0.0] * len(points)

    # Add buffer points to ensure finite regions for all players near edges
    xmin, ymin, xmax, ymax = clip_bounds
    buffer_scale = max(xmax - xmin, ymax - ymin) * 2.0
    buffered = np.vstack(
        [
            points,
            [
                [xmin - buffer_scale, ymin - buffer_scale],
                [xmin - buffer_scale, ymax + buffer_scale],
                [xmax + buffer_scale, ymin - buffer_scale],
                [xmax + buffer_scale, ymax + buffer_scale],
            ],
        ]
    )

    vor = Voronoi(buffered)
    bbox = Polygon(
        [
            (xmin, ymin),
            (xmax, ymin),
            (xmax, ymax),
            (xmin, ymax),
        ]
    )

    areas: list[float] = []
    for i in range(len(points)):
        region_idx = vor.point_region[i]
        vertices = [vor.vertices[v] for v in vor.regions[region_idx] if v >= 0]
        if len(vertices) < 3:
            areas.append(0.0)
            continue
        cell = Polygon(vertices)
        clipped = cell.intersection(bbox)
        areas.append(float(clipped.area))

    return areas

"""Placeholder adapter for tracking-data integration."""

from __future__ import annotations

from typing import Any

import pandas as pd


class TrackingAdapter:
    """Stub adapter for tracking data.

    When tracking data becomes available, this class will:
    1. Load raw tracking frames (25 Hz) for a match.
    2. Resample / align to event timestamps.
    3. Compute kinematics (velocity, acceleration) from
       high-resolution positions.
    4. Feed into :func:`~obpi.pipeline.compute_all_metrics`.
    """

    def __init__(self, match_id: int, data_dir: str = "data/raw/tracking") -> None:
        """Initialise adapter for *match_id*."""
        self.match_id = match_id
        self.data_dir = data_dir

    def load_frames(self) -> pd.DataFrame:
        """Return tracking frames as a DataFrame.

        Raises:
            NotImplementedError: until tracking data is wired up.
        """
        raise NotImplementedError(
            f"Tracking data not yet available for match {self.match_id}"
        )

    def infer_velocity(self, player_id: int) -> pd.DataFrame:
        """Infer per-frame velocity for *player_id* from tracking.

        Returns:
            DataFrame with columns ``[timestamp, x, y, vx, vy, speed]``.
        """
        _ = self.load_frames()  # placeholder
        # Would compute finite-difference velocities here
        return pd.DataFrame(columns=["timestamp", "x", "y", "vx", "vy", "speed"])

    def to_events(self) -> list[dict[str, Any]]:
        """Convert tracking-derived movements to pseudo-events.

        Returns:
            List of dicts compatible with the event-based pipeline.
        """
        return []

"""Preprocessing utilities for geometry helpers and StatsBomb open data."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_RAW_ROOT = _PROJECT_ROOT / "data" / "raw" / "statsbomb_open_data"
_DEFAULT_INTERIM_ROOT = _PROJECT_ROOT / "data" / "interim"


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


def _id_name_fields(value: Any, prefix: str) -> dict[str, Any]:
    """Flatten a StatsBomb ``{id, name}`` object into scalar columns."""
    if not isinstance(value, dict):
        return {f"{prefix}_id": None, f"{prefix}_name": None}

    id_value = value.get("id")
    name_value = value.get("name")
    if id_value is None:
        id_value = value.get(f"{prefix}_id")
    if name_value is None:
        name_value = value.get(f"{prefix}_name")

    return {
        f"{prefix}_id": id_value,
        f"{prefix}_name": name_value,
    }


def _location_fields(value: Any, prefix: str) -> dict[str, Any]:
    """Flatten a 2D or 3D location list into scalar coordinate columns."""
    if not isinstance(value, list):
        return {
            f"{prefix}_x": None,
            f"{prefix}_y": None,
            f"{prefix}_z": None,
        }
    return {
        f"{prefix}_x": value[0] if len(value) > 0 else None,
        f"{prefix}_y": value[1] if len(value) > 1 else None,
        f"{prefix}_z": value[2] if len(value) > 2 else None,
    }


def _json_string(value: Any) -> str | None:
    """Serialize nested values when keeping the raw structure is useful."""
    if value in (None, [], {}):
        return None
    return json.dumps(value, sort_keys=True)


class StatsBombOpenDataPreprocessor:
    """Normalize downloaded StatsBomb open data into analysis-ready parquet tables."""

    def __init__(
        self,
        raw_dir: str | Path = _DEFAULT_RAW_ROOT,
        output_dir: str | Path = _DEFAULT_INTERIM_ROOT,
    ) -> None:
        """Store input and output directories for preprocessing."""
        self.raw_dir = Path(raw_dir)
        self.output_dir = Path(output_dir)

    def preprocess_all(self) -> dict[str, pd.DataFrame]:
        """Run all available raw-data preprocessing steps."""
        competitions = self.preprocess_competitions()
        matches = self.preprocess_matches()
        player_matches = self.preprocess_lineups()
        event_manifest = self.preprocess_events()
        return {
            "competitions": competitions,
            "matches": matches,
            "player_matches": player_matches,
            "event_manifest": event_manifest,
        }

    def preprocess_competitions(self) -> pd.DataFrame:
        """Convert ``competitions.json`` into a flat parquet table."""
        source = self.raw_dir / "competitions.json"
        competitions = json.loads(source.read_text(encoding="utf-8"))
        df = pd.json_normalize(competitions, sep="_").sort_values(
            ["competition_id", "season_id"]
        )
        output = self.output_dir / "competitions.parquet"
        output.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output, index=False)
        return df.reset_index(drop=True)

    def preprocess_matches(self) -> pd.DataFrame:
        """Flatten all competition-season match index files into one table."""
        rows: list[dict[str, Any]] = []
        for path in sorted((self.raw_dir / "matches").glob("*/*.json")):
            competition_id = int(path.parent.name)
            season_id = int(path.stem)
            matches = json.loads(path.read_text(encoding="utf-8"))
            for match in matches:
                row: dict[str, Any] = {
                    "competition_id": competition_id,
                    "season_id": season_id,
                    "match_id": match.get("match_id"),
                    "match_date": match.get("match_date"),
                    "kick_off": match.get("kick_off"),
                    "match_week": match.get("match_week"),
                    "home_score": match.get("home_score"),
                    "away_score": match.get("away_score"),
                    "match_status": match.get("match_status"),
                    "match_status_360": match.get("match_status_360"),
                    "last_updated": match.get("last_updated"),
                    "last_updated_360": match.get("last_updated_360"),
                    "source_file": str(path.relative_to(self.raw_dir)),
                }
                row.update(_id_name_fields(match.get("competition"), "competition"))
                row.update(_id_name_fields(match.get("season"), "season"))
                row.update(_id_name_fields(match.get("home_team"), "home_team"))
                row.update(_id_name_fields(match.get("away_team"), "away_team"))
                row.update(
                    _id_name_fields(
                        match.get("competition_stage"),
                        "competition_stage",
                    )
                )
                row.update(_id_name_fields(match.get("stadium"), "stadium"))
                row.update(_id_name_fields(match.get("referee"), "referee"))

                metadata = match.get("metadata", {})
                if isinstance(metadata, dict):
                    row["metadata_data_version"] = metadata.get("data_version")
                    row["metadata_shot_fidelity_version"] = metadata.get(
                        "shot_fidelity_version"
                    )
                    row["metadata_xy_fidelity_version"] = metadata.get(
                        "xy_fidelity_version"
                    )
                rows.append(row)

        df = pd.DataFrame(rows).sort_values(["competition_id", "season_id", "match_id"])
        output = self.output_dir / "matches.parquet"
        output.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output, index=False)
        return df.reset_index(drop=True)

    def preprocess_lineups(self) -> pd.DataFrame:
        """Flatten all lineup files into a player-match table."""
        rows: list[dict[str, Any]] = []
        for path in sorted((self.raw_dir / "lineups").glob("*.json")):
            match_id = int(path.stem)
            lineups = json.loads(path.read_text(encoding="utf-8"))
            for team_entry in lineups:
                team_id = team_entry.get("team_id")
                team_name = team_entry.get("team_name")
                for player in team_entry.get("lineup", []):
                    positions = player.get("positions", [])
                    first_position = positions[0] if positions else {}
                    country = player.get("country", {})
                    rows.append(
                        {
                            "match_id": match_id,
                            "team_id": team_id,
                            "team_name": team_name,
                            "player_id": player.get("player_id"),
                            "player_name": player.get("player_name"),
                            "player_nickname": player.get("player_nickname"),
                            "jersey_number": player.get("jersey_number"),
                            "country_id": country.get("id")
                            if isinstance(country, dict)
                            else None,
                            "country_name": country.get("name")
                            if isinstance(country, dict)
                            else None,
                            "starting_position_id": first_position.get("position_id"),
                            "starting_position_name": first_position.get("position"),
                            "start_reason": first_position.get("start_reason"),
                            "end_reason": first_position.get("end_reason"),
                            "from_period": first_position.get("from_period"),
                            "to_period": first_position.get("to_period"),
                            "positions_json": _json_string(positions),
                            "cards_json": _json_string(player.get("cards")),
                            "source_file": str(path.relative_to(self.raw_dir)),
                        }
                    )

        df = pd.DataFrame(rows).sort_values(["match_id", "team_id", "player_id"])
        output = self.output_dir / "player_matches.parquet"
        output.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output, index=False)
        return df.reset_index(drop=True)

    def preprocess_events(self) -> pd.DataFrame:
        """Normalize event files into per-match parquet partitions plus a manifest."""
        events_output_dir = self.output_dir / "events_by_match"
        events_output_dir.mkdir(parents=True, exist_ok=True)
        manifest_rows: list[dict[str, Any]] = []

        for path in sorted((self.raw_dir / "events").glob("*.json")):
            match_id = int(path.stem)
            events = json.loads(path.read_text(encoding="utf-8"))
            flattened = [self._flatten_event(match_id, event) for event in events]
            df = pd.DataFrame(flattened).sort_values("index")
            df.to_parquet(events_output_dir / f"{match_id}.parquet", index=False)

            manifest_rows.append(
                {
                    "match_id": match_id,
                    "event_count": len(df),
                    "player_event_count": int(df["player_id"].notna().sum())
                    if not df.empty
                    else 0,
                    "team_count": int(df["team_id"].dropna().nunique())
                    if not df.empty
                    else 0,
                    "source_file": str(path.relative_to(self.raw_dir)),
                    "output_file": str(
                        (events_output_dir / f"{match_id}.parquet").relative_to(
                            self.output_dir
                        )
                    ),
                }
            )

        manifest = pd.DataFrame(manifest_rows).sort_values("match_id")
        manifest_output = self.output_dir / "events_manifest.parquet"
        manifest.to_parquet(manifest_output, index=False)
        return manifest.reset_index(drop=True)

    def _flatten_event(self, match_id: int, event: dict[str, Any]) -> dict[str, Any]:
        """Flatten one StatsBomb event payload into scalar columns."""
        row: dict[str, Any] = {
            "match_id": match_id,
            "event_id": event.get("id"),
            "index": event.get("index"),
            "period": event.get("period"),
            "timestamp": event.get("timestamp"),
            "minute": event.get("minute"),
            "second": event.get("second"),
            "duration": event.get("duration"),
            "possession": event.get("possession"),
            "under_pressure": event.get("under_pressure"),
            "counterpress": event.get("counterpress"),
            "off_camera": event.get("off_camera"),
            "out": event.get("out"),
            "related_events_json": _json_string(event.get("related_events")),
            "freeze_frame_json": _json_string(event.get("freeze_frame")),
            "visible_area_json": _json_string(event.get("visible_area")),
            "tactics_formation": (
                event.get("tactics", {}).get("formation")
                if isinstance(event.get("tactics"), dict)
                else None
            ),
            "source_event_type": event.get("type", {}).get("name")
            if isinstance(event.get("type"), dict)
            else None,
        }
        row.update(_id_name_fields(event.get("type"), "type"))
        row.update(_id_name_fields(event.get("team"), "team"))
        row.update(_id_name_fields(event.get("player"), "player"))
        row.update(_id_name_fields(event.get("position"), "position"))
        row.update(_id_name_fields(event.get("possession_team"), "possession_team"))
        row.update(_id_name_fields(event.get("play_pattern"), "play_pattern"))
        row.update(_location_fields(event.get("location"), "location"))

        pass_data = event.get("pass", {})
        if isinstance(pass_data, dict):
            row.update(_location_fields(pass_data.get("end_location"), "pass_end_location"))
            row["pass_length"] = pass_data.get("length")
            row["pass_angle"] = pass_data.get("angle")
            row.update(_id_name_fields(pass_data.get("recipient"), "pass_recipient"))
            row.update(_id_name_fields(pass_data.get("height"), "pass_height"))
            row.update(_id_name_fields(pass_data.get("type"), "pass_type"))
            row.update(_id_name_fields(pass_data.get("body_part"), "pass_body_part"))
            row.update(_id_name_fields(pass_data.get("outcome"), "pass_outcome"))
        else:
            row["pass_length"] = None
            row["pass_angle"] = None

        carry_data = event.get("carry", {})
        if isinstance(carry_data, dict):
            row.update(_location_fields(carry_data.get("end_location"), "carry_end_location"))

        shot_data = event.get("shot", {})
        if isinstance(shot_data, dict):
            row.update(_location_fields(shot_data.get("end_location"), "shot_end_location"))
            row["shot_statsbomb_xg"] = shot_data.get("statsbomb_xg")
            row["shot_first_time"] = shot_data.get("first_time")
            row.update(_id_name_fields(shot_data.get("outcome"), "shot_outcome"))
            row.update(_id_name_fields(shot_data.get("body_part"), "shot_body_part"))
            row.update(_id_name_fields(shot_data.get("type"), "shot_type"))
        else:
            row["shot_statsbomb_xg"] = None
            row["shot_first_time"] = None

        row.update(_id_name_fields(event.get("ball_receipt"), "ball_receipt"))
        row.update(_id_name_fields(event.get("dribble"), "dribble"))
        row.update(_id_name_fields(event.get("duel"), "duel"))
        row.update(_id_name_fields(event.get("foul_committed"), "foul_committed"))
        row.update(_id_name_fields(event.get("interception"), "interception"))
        row.update(_id_name_fields(event.get("clearance"), "clearance"))
        return row

"""Metric processing from interim parquet tables to OBPI-ready outputs."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd

from obpi.config.loader import Config, load_config
from obpi.metrics.movement import _exclude_set_pieces, compute_obr90, compute_oirc
from obpi.metrics.receiving import compute_brpc, compute_rbtl, compute_rup
from obpi.metrics.spatial import compute_sc, compute_sci
from obpi.metrics.temporal import compute_cbi, compute_lpc
from obpi.utils.xt_model import XTModel

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_INTERIM_ROOT = _PROJECT_ROOT / "data" / "interim"
_DEFAULT_PROCESSED_ROOT = _PROJECT_ROOT / "data" / "processed"

_METRIC_COLUMNS = [
    "player_id",
    "player_name",
    "team_id",
    "team_name",
    "match_id",
    "minutes",
    "starting_position_name",
    "has_360_data",
    "freeze_frame_count",
    "M1_SC",
    "M2_OIRC",
    "M3_BRPC",
    "M4_OBR90",
    "M5_RBTL",
    "M6_RUP",
    "M7_SCI",
    "M8_LPC",
    "M9_CBI",
]


def _parse_json(value: Any) -> Any:
    """Parse a JSON string field when present."""
    if value in (None, "", float("nan")):
        return None
    if isinstance(value, str):
        return json.loads(value)
    return value


def _period_minutes(time_str: str | None, period: int | None, fallback_end: bool) -> float:
    """Convert a lineup time marker into absolute match minutes."""
    base = 0.0
    if period and period > 1:
        base = 45.0 * float(period - 1)
    if time_str is None:
        return 90.0 if fallback_end else base
    minute_str, second_str = time_str.split(":")
    return base + float(minute_str) + float(second_str) / 60.0


def _minutes_from_positions(positions_json: str | None) -> float | None:
    """Estimate player minutes from lineup position segments."""
    positions = _parse_json(positions_json)
    if not positions:
        return None

    total = 0.0
    for segment in positions:
        if not isinstance(segment, dict):
            continue
        start = _period_minutes(segment.get("from"), segment.get("from_period"), False)
        end = _period_minutes(segment.get("to"), segment.get("to_period"), True)
        total += max(0.0, end - start)
    return total if total > 0 else None


class InterimMetricsProcessor:
    """Compute OBPI metrics from locally preprocessed interim parquet data."""

    def __init__(
        self,
        interim_dir: str | Path = _DEFAULT_INTERIM_ROOT,
        output_dir: str | Path = _DEFAULT_PROCESSED_ROOT,
        config: Config | None = None,
        xt_model: XTModel | None = None,
    ) -> None:
        """Store directories and shared metric dependencies."""
        self.interim_dir = Path(interim_dir)
        self.output_dir = Path(output_dir)
        self.config = config or load_config()
        self.xt_model = xt_model or XTModel()

    def process_matches(
        self,
        match_ids: Iterable[int] | None = None,
        require_360: bool = False,
        max_frames_per_match: int | None = None,
        position_keywords: list[str] | None = None,
    ) -> pd.DataFrame:
        """Compute player-match metrics and persist them to parquet."""
        player_matches = pd.read_parquet(self.interim_dir / "player_matches.parquet")
        manifest = pd.read_parquet(self.interim_dir / "events_manifest.parquet")

        if match_ids is None:
            target_manifest = manifest.copy()
            if require_360 and "freeze_frame_event_count" in target_manifest.columns:
                target_manifest = target_manifest[target_manifest["freeze_frame_event_count"] > 0]
            target_match_ids = target_manifest["match_id"].astype(int).tolist()
        else:
            target_match_ids = [int(match_id) for match_id in match_ids]
            if require_360 and "freeze_frame_event_count" in manifest.columns:
                frame_counts = manifest.set_index("match_id")["freeze_frame_event_count"]
                target_match_ids = [
                    match_id
                    for match_id in target_match_ids
                    if int(frame_counts.get(match_id, 0)) > 0
                ]

        rows: list[dict[str, Any]] = []
        for match_id in target_match_ids:
            events_path = self.interim_dir / "events_by_match" / f"{match_id}.parquet"
            if not events_path.exists():
                continue

            event_frame = pd.read_parquet(events_path)
            events = self._to_statsbomb_events(event_frame)
            movement_events = _exclude_set_pieces(events)
            frames = self._extract_frames(event_frame)
            frames = self._sample_frames(frames, max_frames_per_match)
            match_players = player_matches[player_matches["match_id"] == match_id]
            has_lineup_rows = not match_players.empty
            if position_keywords:
                position_pattern = "|".join(position_keywords)
                match_players = match_players[
                    match_players["starting_position_name"]
                    .fillna("")
                    .str.contains(position_pattern, case=False, regex=True)
                ]

            if match_players.empty and not has_lineup_rows:
                match_players = self._players_from_events(events, match_id)
            if match_players.empty:
                continue

            cached_sci = compute_sci(frames[:-1], frames[1:]) if len(frames) >= 2 else 0.0
            for _, player_row in match_players.iterrows():
                player_id = int(player_row["player_id"])
                minutes = _minutes_from_positions(player_row.get("positions_json"))
                if minutes is None:
                    minutes = self._estimate_minutes_from_events(events, player_id)

                avg_loc = self._average_location(events, player_id)
                rows.append(
                    {
                        "player_id": player_id,
                        "player_name": player_row.get("player_name"),
                        "team_id": player_row.get("team_id"),
                        "team_name": player_row.get("team_name"),
                        "match_id": match_id,
                        "minutes": minutes or 0.0,
                        "starting_position_name": player_row.get("starting_position_name"),
                        "has_360_data": bool(frames),
                        "freeze_frame_count": len(frames),
                        "M1_SC": (
                            compute_sc(frames[:-1], frames[1:], avg_loc)
                            if len(frames) >= 2 and avg_loc is not None
                            else 0.0
                        ),
                        "M2_OIRC": compute_oirc(
                            movement_events,
                            player_id,
                            exclude_set_pieces=False,
                        ),
                        "M3_BRPC": compute_brpc(
                            events,
                            frames,
                            player_id,
                            pressure_threshold=self.config.receiving.pressure_radius,
                            cone_angle=self.config.receiving.cone_angle,
                            cone_length=self.config.receiving.cone_length,
                        )
                        if frames
                        else 0.0,
                        "M4_OBR90": compute_obr90(
                            events,
                            player_id,
                            minutes_played=minutes,
                            v_threshold=self.config.movement.v_threshold,
                            duration_threshold=self.config.movement.duration_threshold,
                            max_dt=self.config.movement.max_dt,
                            exclude_set_pieces=False,
                        ),
                        "M5_RBTL": compute_rbtl(events, player_id),
                        "M6_RUP": compute_rup(
                            events,
                            player_id,
                            frames if frames else None,
                            proximity_threshold=self.config.receiving.proximity_threshold,
                        ),
                        "M7_SCI": cached_sci,
                        "M8_LPC": compute_lpc(
                            events,
                            player_id,
                            self.xt_model,
                            min_dt=self.config.temporal.min_dt,
                            max_vel=self.config.temporal.max_vel,
                        ),
                        "M9_CBI": compute_cbi(
                            events,
                            frames,
                            player_id,
                            angle_threshold=self.config.temporal.angle_threshold,
                            lane_buffer=self.config.temporal.lane_buffer,
                        )
                        if frames
                        else 0.0,
                    }
                )

        metrics = pd.DataFrame(rows, columns=_METRIC_COLUMNS)
        if not metrics.empty:
            metrics["has_360_data"] = metrics["has_360_data"].astype(bool)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        metrics.to_parquet(self.output_dir / "player_match_metrics.parquet", index=False)
        return metrics

    def aggregate_player_metrics(self, metrics_df: pd.DataFrame | None = None) -> pd.DataFrame:
        """Aggregate match-level metrics into player-level summary rows."""
        if metrics_df is None:
            metrics_df = pd.read_parquet(self.output_dir / "player_match_metrics.parquet")

        metric_cols = [column for column in metrics_df.columns if column.startswith("M")]
        aggregations: dict[str, Any] = dict.fromkeys(metric_cols, "mean")
        aggregations.update(
            {
                "player_name": "first",
                "team_id": "first",
                "team_name": "first",
                "minutes": "sum",
                "starting_position_name": "first",
                "has_360_data": "max",
                "freeze_frame_count": "sum",
                "match_id": "count",
            }
        )
        aggregated = (
            metrics_df.groupby("player_id", as_index=False)
            .agg(aggregations)
            .rename(columns={"match_id": "match_count"})
        )
        aggregated.to_parquet(self.output_dir / "player_aggregate_metrics.parquet", index=False)
        return aggregated

    def _to_statsbomb_events(self, event_frame: pd.DataFrame) -> pd.DataFrame:
        """Rebuild a minimal StatsBomb-style DataFrame for metric functions."""
        df = event_frame.copy()
        df["type"] = self._id_name_records(df, "type")
        df["team"] = self._id_name_records(df, "team")
        df["player"] = self._id_name_records(df, "player", integer_id=True)
        df["position"] = self._id_name_records(df, "position")
        df["possession_team"] = self._id_name_records(df, "possession_team")
        df["play_pattern"] = self._id_name_records(df, "play_pattern")
        location_x = df["location_x"].tolist()
        location_y = df["location_y"].tolist()
        df["location"] = [
            [location_x[index], location_y[index]]
            if pd.notna(location_x[index]) and pd.notna(location_y[index])
            else None
            for index in range(len(df))
        ]
        df["pass"] = df.apply(self._build_pass_dict, axis=1)
        return df.sort_values(["period", "timestamp", "index"]).reset_index(drop=True)

    def _id_name_records(
        self,
        df: pd.DataFrame,
        prefix: str,
        integer_id: bool = False,
    ) -> list[dict[str, Any]]:
        """Build StatsBomb-style id/name dicts from flat scalar columns."""
        id_column = f"{prefix}_id"
        name_column = f"{prefix}_name"
        records: list[dict[str, Any]] = []
        id_values = df[id_column].tolist()
        name_values = df[name_column].tolist()
        for index in range(len(df)):
            id_value = id_values[index]
            name_value = name_values[index]
            if pd.isna(id_value) and pd.isna(name_value):
                records.append({})
                continue
            if integer_id and pd.notna(id_value):
                id_value = int(id_value)
            records.append({"id": id_value, "name": name_value})
        return records

    def _build_pass_dict(self, row: pd.Series) -> dict[str, Any] | None:
        """Reconstruct the nested pass object used by metric code."""
        if pd.isna(row.get("pass_end_location_x")) and pd.isna(row.get("pass_recipient_id")):
            return None

        pass_payload: dict[str, Any] = {}
        if pd.notna(row.get("pass_recipient_id")):
            pass_payload["recipient"] = {
                "id": int(row["pass_recipient_id"]),
                "name": row.get("pass_recipient_name"),
            }
        if pd.notna(row.get("pass_length")):
            pass_payload["length"] = float(row["pass_length"])
        if pd.notna(row.get("pass_angle")):
            pass_payload["angle"] = float(row["pass_angle"])
        if pd.notna(row.get("pass_end_location_x")) and pd.notna(row.get("pass_end_location_y")):
            pass_payload["end_location"] = [
                float(row["pass_end_location_x"]),
                float(row["pass_end_location_y"]),
            ]
        if pd.notna(row.get("pass_height_id")) or pd.notna(row.get("pass_height_name")):
            pass_payload["height"] = {
                "id": row.get("pass_height_id"),
                "name": row.get("pass_height_name"),
            }
        if pd.notna(row.get("pass_type_id")) or pd.notna(row.get("pass_type_name")):
            pass_payload["type"] = {
                "id": row.get("pass_type_id"),
                "name": row.get("pass_type_name"),
            }
        if pd.notna(row.get("pass_body_part_id")) or pd.notna(row.get("pass_body_part_name")):
            pass_payload["body_part"] = {
                "id": row.get("pass_body_part_id"),
                "name": row.get("pass_body_part_name"),
            }
        if pd.notna(row.get("pass_outcome_id")) or pd.notna(row.get("pass_outcome_name")):
            pass_payload["outcome"] = {
                "id": row.get("pass_outcome_id"),
                "name": row.get("pass_outcome_name"),
            }
        return pass_payload

    def _extract_frames(self, event_frame: pd.DataFrame) -> list[dict[str, Any]]:
        """Extract any embedded freeze-frame events from the interim data."""
        frames: list[dict[str, Any]] = []
        for _, row in event_frame.iterrows():
            freeze_frame = _parse_json(row.get("freeze_frame_json"))
            if not freeze_frame:
                continue
            frame: dict[str, Any] = {"freeze_frame": freeze_frame}
            visible_area = _parse_json(row.get("visible_area_json"))
            if visible_area is not None:
                frame["visible_area"] = visible_area
            frames.append(frame)
        return frames

    def _sample_frames(
        self,
        frames: list[dict[str, Any]],
        max_frames: int | None,
    ) -> list[dict[str, Any]]:
        """Evenly sample long 360 frame sequences for tractable metric processing."""
        if max_frames is None or max_frames <= 0 or len(frames) <= max_frames:
            return frames
        if max_frames == 1:
            return [frames[0]]

        last_index = len(frames) - 1
        indices = {
            round(position * last_index / (max_frames - 1))
            for position in range(max_frames)
        }
        return [frames[index] for index in sorted(indices)]

    def _average_location(self, events: pd.DataFrame, player_id: int) -> list[float] | None:
        """Estimate a player's average event location within a match."""
        if "player_id" in events.columns:
            player_events = events[pd.to_numeric(events["player_id"], errors="coerce") == player_id]
        else:
            player_events = events[
                events["player"].apply(
                    lambda player: player.get("id") == player_id if isinstance(player, dict) else False
                )
            ]
        locations = [loc for loc in player_events["location"] if isinstance(loc, list)]
        if not locations:
            return None
        return [
            float(sum(location[0] for location in locations) / len(locations)),
            float(sum(location[1] for location in locations) / len(locations)),
        ]

    def _estimate_minutes_from_events(self, events: pd.DataFrame, player_id: int) -> float:
        """Fallback player-minute estimate based on first and last event times."""
        if "player_id" in events.columns:
            player_events = events[pd.to_numeric(events["player_id"], errors="coerce") == player_id]
        else:
            player_events = events[
                events["player"].apply(
                    lambda player: player.get("id") == player_id if isinstance(player, dict) else False
                )
            ]
        player_events = player_events.reset_index(drop=True)
        if player_events.empty:
            return 0.0

        def _to_seconds(row: pd.Series) -> float:
            hour, minute, second = row["timestamp"].split(":")
            base = float(hour) * 3600 + float(minute) * 60 + float(second)
            period_offset = 0.0 if int(row["period"]) == 1 else 45.0 * 60.0
            return base + period_offset

        start = _to_seconds(player_events.iloc[0])
        end = _to_seconds(player_events.iloc[-1])
        return max(0.0, min(90.0, (end - start) / 60.0))

    def _players_from_events(self, events: pd.DataFrame, match_id: int) -> pd.DataFrame:
        """Fallback player table when lineup data is missing for a match."""
        if "player_id" in events.columns:
            player_ids = pd.to_numeric(events["player_id"], errors="coerce")
            player_names = events["player"]
            team_names = events["team"]
            position_names = events["position"]
            players = pd.DataFrame({
                "player_id": player_ids,
                "player_name": player_names,
                "team_name": team_names,
                "position_name": position_names,
            }).dropna(subset=["player_id"]).drop_duplicates()
            rows = []
            for _, row in players.iterrows():
                rows.append(
                    {
                        "match_id": match_id,
                        "player_id": int(row["player_id"]),
                        "player_name": row["player_name"],
                        "team_id": None,
                        "team_name": row["team_name"],
                        "starting_position_name": row["position_name"],
                        "positions_json": None,
                    }
                )
        else:
            players = events[events["player"].apply(bool)][
                ["player", "team", "position"]
            ].drop_duplicates()
            rows = []
            for _, row in players.iterrows():
                player = row["player"]
                team = row["team"]
                position = row["position"]
                rows.append(
                    {
                        "match_id": match_id,
                        "player_id": player.get("id"),
                        "player_name": player.get("name"),
                        "team_id": team.get("id"),
                        "team_name": team.get("name"),
                        "starting_position_name": position.get("name"),
                        "positions_json": None,
                    }
            )
        return pd.DataFrame(rows)

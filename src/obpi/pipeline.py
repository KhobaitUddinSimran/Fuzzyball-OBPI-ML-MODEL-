"""Main orchestrator CLI entrypoint for the OBPI pipeline."""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from obpi.config.loader import Config, load_config
from obpi.data.loader import StatsBombLoader
from obpi.metrics.movement import compute_obr90, compute_oirc
from obpi.metrics.receiving import compute_brpc, compute_rbtl, compute_rup, get_receipt_events
from obpi.metrics.spatial import compute_sc, compute_sci
from obpi.metrics.temporal import compute_cbi, compute_lpc
from obpi.utils.logger import setup_logging
from obpi.utils.xt_model import XTModel

logger = logging.getLogger("obpi.pipeline")

# Increment when output schema changes to invalidate old caches
_SCHEMA_VERSION = 3

_IDENTIFIER_COLUMNS = [
    "player_id",
    "match_id",
]

_RAW_METRIC_COLUMNS = [
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

_STATUS_COLUMNS = [
    column
    for metric_name in _RAW_METRIC_COLUMNS
    for column in (f"{metric_name}_status", f"{metric_name}_reason")
]

_DATA_QUALITY_COLUMNS = [
    "has_360",
    "events_loaded",
    "frames_loaded",
    "joined_360_frames",
    "high_quality_frames",
    "minutes_available",
    "data_quality_warnings",
]

_METRIC_COLUMNS = _IDENTIFIER_COLUMNS + _RAW_METRIC_COLUMNS
_FUZZY_METRIC_COLUMNS = _RAW_METRIC_COLUMNS
_NORMALIZED_METRIC_COLUMNS = [f"M{i}" for i in range(1, 10)]


def _extract_unique_players(events: pd.DataFrame) -> list[int]:
    """Return sorted list of unique player IDs present in the events."""
    if events.empty:
        return []
    if "player_id" in events.columns:
        players = pd.to_numeric(events["player_id"], errors="coerce")
        return sorted({int(pid) for pid in players if pd.notna(pid)})
    if "player" not in events.columns:
        return []
    players = events["player"].apply(
        lambda p: p.get("id") if isinstance(p, dict) else None
    )
    return sorted({pid for pid in players if pid is not None})


def _get_player_events(events: pd.DataFrame, player_id: int) -> pd.DataFrame:
    """Return events for one player across flattened or nested StatsBomb schemas."""
    if events.empty:
        return events
    if "player_id" in events.columns:
        player_ids = pd.to_numeric(events["player_id"], errors="coerce")
        return events[player_ids == player_id].reset_index(drop=True)
    if "player" not in events.columns:
        return events.iloc[0:0].copy()
    return events[
        events["player"].apply(
            lambda p, pid=player_id: (
                p.get("id") == pid if isinstance(p, dict) else False
            )
        )
    ].reset_index(drop=True)


def _has_columns(events: pd.DataFrame, columns: set[str]) -> bool:
    """Return whether all required columns are present."""
    return columns.issubset(events.columns)


def _is_high_quality_frame(frame: dict[str, Any]) -> bool:
    """Return whether a 360 frame has enough player locations to be useful."""
    freeze_frame = frame.get("freeze_frame", [])
    locations = [
        player.get("location")
        for player in freeze_frame
        if isinstance(player, dict) and player.get("location") is not None
    ]
    return len(locations) >= 10


def _build_frames_by_event_id(
    frame_rows: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Group StatsBomb 360 rows by event_uuid/id into freeze-frame payloads."""
    frames_by_event_id: dict[str, list[dict[str, Any]]] = {}
    for row in frame_rows:
        frame_id = row.get("event_uuid") or row.get("id")
        if not frame_id:
            continue
        frame_key = str(frame_id)
        event_frames = frames_by_event_id.setdefault(
            frame_key,
            [
                {
                    "event_uuid": frame_key,
                    "visible_area": row.get("visible_area"),
                    "freeze_frame": [],
                }
            ],
        )
        frame_payload = event_frames[0]
        if not frame_payload.get("visible_area") and row.get("visible_area"):
            frame_payload["visible_area"] = row.get("visible_area")
        existing_freeze_frame = row.get("freeze_frame")
        if isinstance(existing_freeze_frame, list):
            frame_payload["freeze_frame"].extend(existing_freeze_frame)
        elif row.get("location") is not None:
            frame_payload["freeze_frame"].append(
                {
                    "location": row.get("location"),
                    "teammate": bool(row.get("teammate")),
                    "actor": bool(row.get("actor")),
                    "keeper": bool(row.get("keeper")),
                }
            )
    return {
        event_id: frames
        for event_id, frames in frames_by_event_id.items()
        if frames and frames[0].get("freeze_frame")
    }


def _event_id(row: pd.Series) -> str | None:
    """Return an event id from a DataFrame row."""
    event_id = row.get("id")
    if not event_id:
        return None
    return str(event_id)


def _ordered_frames_for_events(
    events: pd.DataFrame,
    frames_by_event_id: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Return one matched frame per event, ordered by event order."""
    ordered: list[dict[str, Any]] = []
    for _, event in events.iterrows():
        event_id = _event_id(event)
        if not event_id:
            continue
        event_frames = frames_by_event_id.get(event_id, [])
        if event_frames:
            ordered.append(event_frames[0])
    return ordered


def _count_events_with_frames(
    events: pd.DataFrame,
    frames_by_event_id: dict[str, list[dict[str, Any]]],
) -> int:
    """Count events that have exact matched 360 freeze frames."""
    count = 0
    for _, event in events.iterrows():
        event_id = _event_id(event)
        if event_id and frames_by_event_id.get(event_id):
            count += 1
    return count


def _set_metric(
    row: dict[str, Any],
    metric_name: str,
    value: float | None,
    status: str = "available",
    reason: str | None = None,
) -> None:
    """Set a metric value plus status metadata on a row."""
    row[metric_name] = None if value is None else float(value)
    row[f"{metric_name}_status"] = status
    row[f"{metric_name}_reason"] = reason


def _metric_unavailable(row: dict[str, Any], metric_name: str, reason: str) -> None:
    """Mark one metric as unavailable with a reason."""
    _set_metric(row, metric_name, None, "unavailable", reason)


def _metric_warning(row: dict[str, Any], message: str) -> None:
    """Append a data-quality warning to a metric row."""
    row.setdefault("data_quality_warnings", [])
    if message not in row["data_quality_warnings"]:
        row["data_quality_warnings"].append(message)


def _sci_from_frames(frames: list[dict[str, Any]]) -> float:
    """Compute average SCI across consecutive frame pairs."""
    if len(frames) < 2:
        return 0.0
    values = []
    for before, after in zip(frames, frames[1:]):
        values.append(compute_sci([before], [after]))
    return float(sum(values) / len(values))


def _sc_from_frames(
    frames: list[dict[str, Any]], player_location: list[float]
) -> float:
    """Compute average SC across consecutive frame pairs for a player."""
    if len(frames) < 2:
        return 0.0
    values = []
    for before, after in zip(frames, frames[1:]):
        values.append(compute_sc([before], [after], player_location))
    return float(sum(values) / len(values))


# Cache for team-level spatial metrics computed once per match
_match_cache: dict[int, dict[str, Any]] = {}


def _cached_sci(match_id: int, frames: list[dict[str, Any]]) -> float:
    key = f"sci_{match_id}"
    if key not in _match_cache:
        _match_cache[key] = _sci_from_frames(frames)
    return _match_cache[key]


def _cached_sc(
    match_id: int, frames: list[dict[str, Any]], player_location: list[float]
) -> float:
    key = f"sc_{match_id}_{player_location[0]:.1f}_{player_location[1]:.1f}"
    if key not in _match_cache:
        _match_cache[key] = _sc_from_frames(frames, player_location)
    return _match_cache[key]


def _compute_all_metrics_legacy(
    match_id: int,
    tier: str = "open",
    xt_model: XTModel | None = None,
    config: Config | None = None,
) -> pd.DataFrame:
    """Run the full OBPI metric pipeline for a single match.

    Loads events and 360 freeze frames, then computes M1–M9 for every
    player who appears in the event data.

    Args:
        match_id: StatsBomb match identifier.
        tier: Data tier (``"open"`` or ``"api"``).
        xt_model: Optional :class:`~obpi.utils.xt_model.XTModel` instance.
            If ``None``, a default model is created.
        config: Optional configuration object with threshold overrides.

    Returns:
        DataFrame with one row per player and columns
        ``[player_id, match_id, M1_SC, M2_OIRC, M3_BRPC,
        M4_OBR90, M5_RBTL, M6_RUP, M7_SCI, M8_LPC, M9_CBI]``.
    """
    cfg = config or load_config()
    loader = StatsBombLoader(tier=tier)
    events = loader.get_events(match_id)
    frames = loader.get_freeze_frames(match_id)

    logger.info(
        "match=%s events=%d frames=%d players=%d",
        match_id,
        len(events),
        len(frames),
        len(_extract_unique_players(events)),
    )

    if xt_model is None:
        xt_model = XTModel()

    players = _extract_unique_players(events)

    # Pre-compute team-level spatial metrics once per match
    cached_sci: float | None = None
    if frames:
        cached_sci = _cached_sci(match_id, frames)

    rows: list[dict[str, Any]] = []
    for player_id in players:
        row: dict[str, Any] = {
            "player_id": player_id,
            "match_id": match_id,
            "M1_SC": 0.0,
            "M2_OIRC": 0.0,
            "M3_BRPC": 0.0,
            "M4_OBR90": 0.0,
            "M5_RBTL": 0.0,
            "M6_RUP": 0.0,
            "M7_SCI": 0.0,
            "M8_LPC": 0.0,
            "M9_CBI": 0.0,
        }

        # Movement metrics (M4, M2)
        row["M4_OBR90"] = compute_obr90(
            events,
            player_id,
            v_threshold=cfg.movement.v_threshold,
            duration_threshold=cfg.movement.duration_threshold,
            max_dt=cfg.movement.max_dt,
        )
        row["M2_OIRC"] = compute_oirc(events, player_id)

        # Receiving metrics (M5, M6, M3)
        row["M5_RBTL"] = compute_rbtl(events, player_id)
        row["M6_RUP"] = compute_rup(
            events, player_id, frames if frames else None,
            proximity_threshold=cfg.receiving.proximity_threshold,
        )
        row["M3_BRPC"] = (
            compute_brpc(
                events,
                frames,
                player_id,
                pressure_threshold=cfg.receiving.pressure_radius,
                cone_angle=cfg.receiving.cone_angle,
                cone_length=cfg.receiving.cone_length,
            )
            if frames
            else 0.0
        )

        # Temporal metrics (M8, M9)
        row["M8_LPC"] = compute_lpc(
            events, player_id, xt_model,
            min_dt=cfg.temporal.min_dt,
            max_vel=cfg.temporal.max_vel,
        )
        row["M9_CBI"] = (
            compute_cbi(
                events,
                frames,
                player_id,
                angle_threshold=cfg.temporal.angle_threshold,
                lane_buffer=cfg.temporal.lane_buffer,
            )
            if frames
            else 0.0
        )

        # Spatial metrics (M7, M1) — use cached team-level values
        if frames and cached_sci is not None:
            row["M7_SCI"] = cached_sci
            # Use player's average location as proxy for SC analysis
            if "player_id" in events.columns:
                player_events = events[pd.to_numeric(events["player_id"], errors="coerce") == player_id]
            else:
                player_events = events[
                    events["player"].apply(
                        lambda p, pid=player_id: (
                            p.get("id") == pid if isinstance(p, dict) else False
                        )
                    )
                ]
            locs = player_events["location"].dropna()
            if not locs.empty:
                avg_loc = [
                    float(locs.apply(lambda loc: loc[0]).mean()),
                    float(locs.apply(lambda loc: loc[1]).mean()),
                ]
                row["M1_SC"] = _cached_sc(match_id, frames, avg_loc)

        rows.append(row)

    return pd.DataFrame(rows, columns=_METRIC_COLUMNS)


def compute_all_metrics(
    match_id: int,
    tier: str = "open",
    xt_model: XTModel | None = None,
    config: Config | None = None,
) -> pd.DataFrame:
    """Run OBPI metric extraction with explicit metric availability metadata."""
    cfg = config or load_config()
    loader = StatsBombLoader(tier=tier)
    events = loader.get_events(match_id)
    frame_rows = loader.get_freeze_frames(match_id)
    frames_by_event_id = _build_frames_by_event_id(frame_rows)
    frames = _ordered_frames_for_events(events, frames_by_event_id)

    logger.info(
        "match=%s events=%d frame_rows=%d matched_frames=%d players=%d",
        match_id,
        len(events),
        len(frame_rows),
        len(frames),
        len(_extract_unique_players(events)),
    )

    if xt_model is None:
        xt_model = XTModel()

    players = _extract_unique_players(events)
    cached_sci: float | None = _cached_sci(match_id, frames) if frames else None

    rows: list[dict[str, Any]] = []
    for player_id in players:
        player_events = _get_player_events(events, player_id)
        receipts = get_receipt_events(events, player_id)
        receipt_frame_count = _count_events_with_frames(receipts, frames_by_event_id)
        player_frame_count = _count_events_with_frames(player_events, frames_by_event_id)
        high_quality_frames = sum(1 for frame in frames if _is_high_quality_frame(frame))
        minutes_available = (
            not player_events.empty
            and "minute" in player_events.columns
            and player_events["minute"].notna().any()
        )
        row: dict[str, Any] = {
            "player_id": player_id,
            "match_id": match_id,
            "has_360": bool(frames),
            "events_loaded": not events.empty,
            "frames_loaded": bool(frames),
            "joined_360_frames": player_frame_count,
            "high_quality_frames": high_quality_frames,
            "minutes_available": bool(minutes_available),
            "data_quality_warnings": [],
        }

        if not _has_columns(events, {"timestamp", "location"}):
            reason = "Events are missing timestamp or location columns"
            _metric_unavailable(row, "M4_OBR90", reason)
            _metric_unavailable(row, "M2_OIRC", reason)
        else:
            if not minutes_available:
                _metric_warning(
                    row,
                    "Minutes played is missing, so M4_OBR90 may be unreliable.",
                )
            _set_metric(
                row,
                "M4_OBR90",
                compute_obr90(
                    events,
                    player_id,
                    v_threshold=cfg.movement.v_threshold,
                    duration_threshold=cfg.movement.duration_threshold,
                    max_dt=cfg.movement.max_dt,
                ),
                "available_with_warning" if not minutes_available else "available",
                None if minutes_available else "Minutes played was estimated from events",
            )
            _set_metric(row, "M2_OIRC", compute_oirc(events, player_id))

        _set_metric(row, "M5_RBTL", compute_rbtl(events, player_id))
        has_pressure_flag = (
            "under_pressure" in receipts.columns
            and receipts["under_pressure"].notna().any()
        )
        if receipts.empty:
            _set_metric(row, "M6_RUP", 0.0, "available", "No receipt opportunities")
            _set_metric(row, "M3_BRPC", 0.0, "available", "No receipt opportunities")
        else:
            if has_pressure_flag or receipt_frame_count > 0:
                _set_metric(
                    row,
                    "M6_RUP",
                    compute_rup(
                        events,
                        player_id,
                        frames if frames else None,
                        frames_by_event_id=frames_by_event_id,
                        proximity_threshold=cfg.receiving.proximity_threshold,
                    ),
                    "available",
                    "Used event pressure flags" if has_pressure_flag else None,
                )
            else:
                _metric_unavailable(
                    row,
                    "M6_RUP",
                    "No pressure flags or 360 frames for receipt pressure calculation",
                )

            if not frames:
                _metric_unavailable(
                    row,
                    "M3_BRPC",
                    "No 360 frames for best receiving position calculation",
                )
            elif receipt_frame_count == 0:
                _metric_unavailable(
                    row,
                    "M3_BRPC",
                    "No valid joined 360 frames for this player",
                )
            else:
                partial_join = receipt_frame_count < len(receipts)
                if partial_join:
                    _metric_warning(row, "Some receipt events have no joined 360 frame.")
                _set_metric(
                    row,
                    "M3_BRPC",
                    compute_brpc(
                        events,
                        frames,
                        player_id,
                        frames_by_event_id=frames_by_event_id,
                        pressure_threshold=cfg.receiving.pressure_radius,
                        cone_angle=cfg.receiving.cone_angle,
                        cone_length=cfg.receiving.cone_length,
                    ),
                    "available_with_warning" if partial_join else "available",
                    "Only partial receipt-frame joins were available" if partial_join else None,
                )

        if not _has_columns(events, {"timestamp", "period", "location"}):
            _metric_unavailable(
                row,
                "M8_LPC",
                "Events are missing timestamp, period, or location columns",
            )
        else:
            _set_metric(
                row,
                "M8_LPC",
                compute_lpc(
                    events,
                    player_id,
                    xt_model,
                    min_dt=cfg.temporal.min_dt,
                    max_vel=cfg.temporal.max_vel,
                ),
            )

        if receipts.empty:
            _set_metric(row, "M9_CBI", 0.0, "available", "No receipt opportunities")
        elif not frames:
            _metric_unavailable(row, "M9_CBI", "No 360 frames for CBI calculation")
        elif receipt_frame_count == 0:
            _metric_unavailable(
                row,
                "M9_CBI",
                "No valid joined 360 frames for this player",
            )
        else:
            partial_join = receipt_frame_count < len(receipts)
            _set_metric(
                row,
                "M9_CBI",
                compute_cbi(
                    events,
                    frames,
                    player_id,
                    frames_by_event_id=frames_by_event_id,
                    angle_threshold=cfg.temporal.angle_threshold,
                    lane_buffer=cfg.temporal.lane_buffer,
                ),
                "available_with_warning" if partial_join else "available",
                "Only partial receipt-frame joins were available" if partial_join else None,
            )

        if not frames:
            _metric_unavailable(row, "M7_SCI", "No 360 frames for spatial calculation")
            _metric_unavailable(row, "M1_SC", "No 360 frames for spatial calculation")
        elif high_quality_frames < 2 or cached_sci is None:
            reason = "No valid joined 360 frames for spatial calculation"
            _metric_unavailable(row, "M7_SCI", reason)
            _metric_unavailable(row, "M1_SC", reason)
        else:
            _set_metric(row, "M7_SCI", cached_sci)
            locs = (
                player_events["location"].dropna()
                if "location" in player_events.columns
                else pd.Series(dtype=object)
            )
            if locs.empty:
                _metric_unavailable(
                    row,
                    "M1_SC",
                    "No valid player locations for screening calculation",
                )
            else:
                avg_loc = [
                    float(locs.apply(lambda loc: loc[0]).mean()),
                    float(locs.apply(lambda loc: loc[1]).mean()),
                ]
                _set_metric(row, "M1_SC", _cached_sc(match_id, frames, avg_loc))

        if any(row.get(f"{metric}_status") == "unavailable" for metric in _RAW_METRIC_COLUMNS):
            _metric_warning(row, "Some frame-dependent metrics were unavailable.")

        rows.append(row)

    return pd.DataFrame(
        rows,
        columns=_METRIC_COLUMNS + _STATUS_COLUMNS + _DATA_QUALITY_COLUMNS,
    )


def run_pipeline(
    match_id: int,
    tier: str = "open",
    output_dir: str = "data/processed",
    xt_model: XTModel | None = None,
    config: Config | None = None,
) -> pd.DataFrame:
    """Compute metrics and cache the result to a Parquet file.

    Args:
        match_id: StatsBomb match identifier.
        tier: Data tier.
        output_dir: Directory for cached parquet files.
        xt_model: Optional xT model instance.
        config: Optional configuration object.

    Returns:
        Metrics DataFrame (same schema as :func:`compute_all_metrics`).
    """
    out_path = Path(output_dir) / f"{match_id}_metrics.parquet"
    if out_path.exists():
        df = pd.read_parquet(out_path)
        # Schema-version guard: recompute if outdated
        if df.get("_schema_version", pd.Series([0])).iloc[0] == _SCHEMA_VERSION:
            logger.info("Cache hit: %s", out_path)
            return df.drop(columns=["_schema_version"], errors="ignore")
        logger.info("Cache stale (schema %s != %s), recomputing", out_path, _SCHEMA_VERSION)

    df = compute_all_metrics(match_id, tier=tier, xt_model=xt_model, config=config)
    df["_schema_version"] = _SCHEMA_VERSION
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    return df.drop(columns=["_schema_version"], errors="ignore")


def run_fuzzy_pipeline(
    metrics_df: pd.DataFrame,
    output_path: str | Path = "data/processed/obpi_scores.parquet",
    score_column: str = "obpi",
) -> pd.DataFrame:
    """Compute OBPI scores from an existing M1-M9 metrics DataFrame.

    This is intentionally additive: the raw M1-M9 metrics pipeline remains in
    :func:`compute_all_metrics` and :func:`run_pipeline`; fuzzy aggregation is a
    separate downstream step that consumes their output.
    """
    from obpi.fuzzy.processing import (
        NORMALIZED_METRICS,
        run_real_data_fuzzy_processing,
    )
    from obpi.fuzzy.scoring import fit_fuzzy_engine, score_metrics_dataframe

    if set(_FUZZY_METRIC_COLUMNS).issubset(metrics_df.columns):
        scored, _metadata = run_real_data_fuzzy_processing(
            metrics_df,
            id_columns=[
                column
                for column in metrics_df.columns
                if column not in _FUZZY_METRIC_COLUMNS
            ],
            score_column=score_column,
        )
    elif set(_NORMALIZED_METRIC_COLUMNS).issubset(metrics_df.columns):
        metric_names = _NORMALIZED_METRIC_COLUMNS
        engine = fit_fuzzy_engine(metrics_df, metric_names=metric_names)
        scored = score_metrics_dataframe(
            metrics_df,
            engine=engine,
            metric_names=metric_names,
            score_column=score_column,
        )
    else:
        expected = ", ".join(_FUZZY_METRIC_COLUMNS)
        normalized = ", ".join(NORMALIZED_METRICS)
        raise ValueError(
            "metrics_df must contain either pipeline metric columns "
            f"({expected}) or normalized columns ({normalized})"
        )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    scored.to_parquet(output, index=False)
    return scored


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="obpi.pipeline",
        description="Off-Ball Positional Intelligence (OBPI) pipeline",
    )
    parser.add_argument(
        "--match-id",
        type=int,
        help="StatsBomb match ID to analyze",
    )
    parser.add_argument(
        "--player-id",
        type=int,
        help="StatsBomb player ID to analyze (optional)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed",
        help="Directory to write processed metric parquet files",
    )
    parser.add_argument(
        "--tier",
        type=str,
        default="open",
        choices=["open", "api"],
        help="StatsBomb data tier to use",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to custom YAML config file (default: config/default.yaml)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the OBPI pipeline from CLI arguments."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    if not args.match_id:
        parser.error("--match-id is required")

    setup_logging(level=logging.DEBUG if args.verbose else logging.INFO)
    config = load_config(args.config) if args.config else None

    df = run_pipeline(
        match_id=args.match_id,
        tier=args.tier,
        output_dir=args.output,
        config=config,
    )

    if args.player_id:
        df = df[df["player_id"] == args.player_id]

    print(df.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())

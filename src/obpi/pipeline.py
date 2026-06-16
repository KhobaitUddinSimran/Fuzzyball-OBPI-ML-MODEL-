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
from obpi.metrics.receiving import compute_brpc, compute_rbtl, compute_rup
from obpi.metrics.spatial import compute_sc, compute_sci
from obpi.metrics.temporal import compute_cbi, compute_lpc
from obpi.utils.logger import setup_logging
from obpi.utils.xt_model import XTModel

logger = logging.getLogger("obpi.pipeline")

# Increment when output schema changes to invalidate old caches
_SCHEMA_VERSION = 2

_METRIC_COLUMNS = [
    "player_id",
    "match_id",
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

_FUZZY_METRIC_COLUMNS = _METRIC_COLUMNS[2:]
_NORMALIZED_METRIC_COLUMNS = [f"M{i}" for i in range(1, 10)]


def _extract_unique_players(events: pd.DataFrame) -> list[int]:
    """Return sorted list of unique player IDs present in the events."""
    if events.empty or "player" not in events.columns:
        return []
    players = events["player"].apply(
        lambda p: p.get("id") if isinstance(p, dict) else None
    )
    return sorted({pid for pid in players if pid is not None})


def _sci_from_frames(frames: list[dict[str, Any]]) -> float:
    """Compute average SCI across consecutive frame pairs."""
    if len(frames) < 2:
        return 0.0
    values = []
    for i in range(len(frames) - 1):
        values.append(compute_sci([frames[i]], [frames[i + 1]]))
    return float(sum(values) / len(values))


def _sc_from_frames(
    frames: list[dict[str, Any]], player_location: list[float]
) -> float:
    """Compute average SC across consecutive frame pairs for a player."""
    if len(frames) < 2:
        return 0.0
    values = []
    for i in range(len(frames) - 1):
        values.append(compute_sc([frames[i]], [frames[i + 1]], player_location))
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


def compute_all_metrics(
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
    from obpi.fuzzy.scoring import fit_fuzzy_engine, score_metrics_dataframe

    if set(_FUZZY_METRIC_COLUMNS).issubset(metrics_df.columns):
        metric_names = _FUZZY_METRIC_COLUMNS
    elif set(_NORMALIZED_METRIC_COLUMNS).issubset(metrics_df.columns):
        metric_names = _NORMALIZED_METRIC_COLUMNS
    else:
        expected = ", ".join(_FUZZY_METRIC_COLUMNS)
        normalized = ", ".join(_NORMALIZED_METRIC_COLUMNS)
        raise ValueError(
            "metrics_df must contain either pipeline metric columns "
            f"({expected}) or normalized columns ({normalized})"
        )

    engine = fit_fuzzy_engine(metrics_df, metric_names=metric_names)
    scored = score_metrics_dataframe(
        metrics_df,
        engine=engine,
        metric_names=metric_names,
        score_column=score_column,
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

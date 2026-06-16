"""Main orchestrator CLI entrypoint for the OBPI pipeline."""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from obpi.data.loader import StatsBombLoader
from obpi.metrics.movement import compute_obr90, compute_oirc
from obpi.metrics.receiving import compute_brpc, compute_rbtl, compute_rup
from obpi.metrics.spatial import compute_sc, compute_sci
from obpi.metrics.temporal import compute_cbi, compute_lpc
from obpi.utils.xt_model import XTModel

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


def compute_all_metrics(
    match_id: int,
    tier: str = "open",
    xt_model: XTModel | None = None,
) -> pd.DataFrame:
    """Run the full OBPI metric pipeline for a single match.

    Loads events and 360 freeze frames, then computes M1–M9 for every
    player who appears in the event data.

    Args:
        match_id: StatsBomb match identifier.
        tier: Data tier (``"open"`` or ``"api"``).
        xt_model: Optional :class:`~obpi.utils.xt_model.XTModel` instance.
            If ``None``, a default model is created.

    Returns:
        DataFrame with one row per player and columns
        ``[player_id, match_id, M1_SC, M2_OIRC, M3_BRPC,
        M4_OBR90, M5_RBTL, M6_RUP, M7_SCI, M8_LPC, M9_CBI]``.
    """
    loader = StatsBombLoader(tier=tier)
    events = loader.get_events(match_id)
    frames = loader.get_freeze_frames(match_id)

    if xt_model is None:
        xt_model = XTModel()

    players = _extract_unique_players(events)

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
        row["M4_OBR90"] = compute_obr90(events, player_id)
        row["M2_OIRC"] = compute_oirc(events, player_id)

        # Receiving metrics (M5, M6, M3)
        row["M5_RBTL"] = compute_rbtl(events, player_id)
        row["M6_RUP"] = compute_rup(events, player_id, frames if frames else None)
        row["M3_BRPC"] = (
            compute_brpc(events, frames, player_id) if frames else 0.0
        )

        # Temporal metrics (M8, M9)
        row["M8_LPC"] = compute_lpc(events, player_id, xt_model)
        row["M9_CBI"] = compute_cbi(events, frames, player_id) if frames else 0.0

        # Spatial metrics (M7, M1) — averaged over consecutive frame pairs
        if frames:
            row["M7_SCI"] = _sci_from_frames(frames)
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
                row["M1_SC"] = _sc_from_frames(frames, avg_loc)

        rows.append(row)

    return pd.DataFrame(rows, columns=_METRIC_COLUMNS)


def run_pipeline(
    match_id: int,
    tier: str = "open",
    output_dir: str = "data/processed",
    xt_model: XTModel | None = None,
) -> pd.DataFrame:
    """Compute metrics and cache the result to a Parquet file.

    Args:
        match_id: StatsBomb match identifier.
        tier: Data tier.
        output_dir: Directory for cached parquet files.
        xt_model: Optional xT model instance.

    Returns:
        Metrics DataFrame (same schema as :func:`compute_all_metrics`).
    """
    out_path = Path(output_dir) / f"{match_id}_metrics.parquet"
    if out_path.exists():
        return pd.read_parquet(out_path)

    df = compute_all_metrics(match_id, tier=tier, xt_model=xt_model)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    return df


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
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the OBPI pipeline from CLI arguments."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    if not args.match_id:
        parser.error("--match-id is required")

    df = run_pipeline(
        match_id=args.match_id,
        tier=args.tier,
        output_dir=args.output,
    )

    if args.player_id:
        df = df[df["player_id"] == args.player_id]

    print(df.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())

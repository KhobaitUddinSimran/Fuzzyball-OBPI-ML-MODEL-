"""Process interim parquet tables into player-level OBPI metric outputs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser."""
    parser = argparse.ArgumentParser(
        description="Compute M1-M9 metrics from interim StatsBomb parquet tables."
    )
    parser.add_argument(
        "--interim-dir",
        type=Path,
        default=Path("data/interim"),
        help="Directory containing normalized interim parquet tables.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed"),
        help="Directory where processed metric outputs should be written.",
    )
    parser.add_argument(
        "--limit-matches",
        type=int,
        default=None,
        help="Optional limit on the number of matches to process.",
    )
    return parser


def main() -> int:
    """Run metric processing for interim parquet data."""
    from obpi.data.metric_processing import InterimMetricsProcessor

    args = build_parser().parse_args()
    processor = InterimMetricsProcessor(
        interim_dir=args.interim_dir,
        output_dir=args.output_dir,
    )

    match_ids = None
    if args.limit_matches is not None:
        import pandas as pd

        manifest = pd.read_parquet(args.interim_dir / "events_manifest.parquet")
        match_ids = manifest["match_id"].astype(int).head(args.limit_matches).tolist()

    match_metrics = processor.process_matches(match_ids=match_ids)
    aggregate_metrics = processor.aggregate_player_metrics(match_metrics)
    print(f"player_match_metrics: {len(match_metrics)} rows")
    print(f"player_aggregate_metrics: {len(aggregate_metrics)} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

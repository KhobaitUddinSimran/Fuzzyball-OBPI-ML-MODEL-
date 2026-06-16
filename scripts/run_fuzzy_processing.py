"""Run fuzzy OBPI scoring on processed metric parquet outputs."""

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
        description="Normalize processed metrics and compute OBPI fuzzy scores."
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        default=Path("data/processed/player_match_metrics.parquet"),
        help="Processed metric parquet file to score.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("data/processed/player_obpi_scores.parquet"),
        help="Destination parquet path for scored OBPI outputs.",
    )
    parser.add_argument(
        "--metadata-path",
        type=Path,
        default=Path("results/week5_membership_report.json"),
        help="Destination JSON path for normalization and membership metadata.",
    )
    return parser


def main() -> int:
    """Run fuzzy processing for a processed metrics parquet file."""
    import pandas as pd

    from obpi.fuzzy.processing import run_real_data_fuzzy_processing, save_fuzzy_outputs

    args = build_parser().parse_args()
    metrics_df = pd.read_parquet(args.input_path)
    scored_df, metadata = run_real_data_fuzzy_processing(metrics_df)
    save_fuzzy_outputs(
        scored_df,
        metadata,
        output_path=args.output_path,
        metadata_path=args.metadata_path,
    )
    print(f"scored_rows: {len(scored_df)}")
    print(f"output_path: {args.output_path}")
    print(f"metadata_path: {args.metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

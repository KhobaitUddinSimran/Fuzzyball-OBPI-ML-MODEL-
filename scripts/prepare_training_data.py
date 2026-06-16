"""Prepare an ML-ready extreme-quartile training dataset from OBPI scores."""

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
        description="Prepare scaled extreme-quartile OBPI data for ML validation."
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        default=Path("data/processed/player_obpi_scores.parquet"),
        help="Scored OBPI parquet file containing normalized M1..M9 plus obpi.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("data/processed/training_prepared.parquet"),
        help="Destination parquet path for the prepared training dataset.",
    )
    parser.add_argument(
        "--metadata-path",
        type=Path,
        default=Path("results/training_preparation_report.json"),
        help="Destination JSON path for scaler and fold metadata.",
    )
    parser.add_argument(
        "--cv-splits",
        type=int,
        default=5,
        help="Requested number of stratified CV folds.",
    )
    return parser


def main() -> int:
    """Prepare a training frame from scored OBPI rows."""
    import pandas as pd

    from obpi.ml.validation import prepare_training_frame, save_training_preparation

    args = build_parser().parse_args()
    scored_df = pd.read_parquet(args.input_path)
    result = prepare_training_frame(scored_df, cv_splits=args.cv_splits)
    save_training_preparation(
        result,
        output_path=args.output_path,
        metadata_path=args.metadata_path,
    )
    print(f"prepared_rows: {len(result.prepared_df)}")
    print(f"output_path: {args.output_path}")
    print(f"metadata_path: {args.metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

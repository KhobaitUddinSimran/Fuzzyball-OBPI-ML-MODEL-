"""Preprocess downloaded StatsBomb open data into interim parquet tables."""

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
        description="Normalize raw StatsBomb open data into interim parquet tables."
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/raw/statsbomb_open_data"),
        help="Directory containing downloaded StatsBomb open data.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/interim"),
        help="Directory where interim parquet outputs should be written.",
    )
    return parser


def main() -> int:
    """Run preprocessing for the configured raw-data directory."""
    from obpi.data.preprocessor import StatsBombOpenDataPreprocessor

    args = build_parser().parse_args()
    preprocessor = StatsBombOpenDataPreprocessor(
        raw_dir=args.raw_dir,
        output_dir=args.output_dir,
    )
    outputs = preprocessor.preprocess_all()
    for name, df in outputs.items():
        print(f"{name}: {len(df)} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

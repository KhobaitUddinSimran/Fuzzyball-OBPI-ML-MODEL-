"""Train Week 6 validation models from a prepared OBPI training dataset."""

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
        description="Train and evaluate OBPI validation models from prepared data."
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        default=Path("data/processed/training_prepared.parquet"),
        help="Prepared training parquet containing M1..M9 and label.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("results/cv_results.json"),
        help="Destination JSON path for validation metrics.",
    )
    parser.add_argument(
        "--output-markdown",
        type=Path,
        default=Path("results/VALIDATION_REPORT.md"),
        help="Destination Markdown path for the validation summary.",
    )
    parser.add_argument(
        "--cv-splits",
        type=int,
        default=5,
        help="Requested number of stratified CV folds.",
    )
    parser.add_argument(
        "--include-xgboost",
        action="store_true",
        help="Attempt XGBoost training in addition to logistic and SVM.",
    )
    return parser


def main() -> int:
    """Run the Week 6 model-training suite."""
    import pandas as pd

    from obpi.ml.validation import save_validation_results, validate_prepared_data

    args = build_parser().parse_args()
    prepared_df = pd.read_parquet(args.input_path)
    report = validate_prepared_data(
        prepared_df,
        cv_splits=args.cv_splits,
        include_xgboost=args.include_xgboost,
    )
    save_validation_results(
        report,
        output_path=args.output_json,
        markdown_path=args.output_markdown,
    )
    print(f"validated_rows: {report['n_rows']}")
    print(f"n_samples: {report['n_samples']}")
    print(f"models: {sorted(report['models'].keys())}")
    print(f"output_json: {args.output_json}")
    print(f"output_markdown: {args.output_markdown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

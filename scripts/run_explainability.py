"""Run Week 7 explainability artifacts from a prepared OBPI training dataset."""

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
        description="Generate explainability artifacts from prepared OBPI data."
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        default=Path("data/processed/training_prepared.parquet"),
        help="Prepared training parquet containing M1..M9 and label.",
    )
    parser.add_argument(
        "--weights-output",
        type=Path,
        default=Path("results/metric_weights.json"),
        help="Destination JSON path for normalized metric weights.",
    )
    parser.add_argument(
        "--permutation-output",
        type=Path,
        default=Path("results/permutation_importance.csv"),
        help="Destination CSV path for permutation importance scores.",
    )
    parser.add_argument(
        "--report-output",
        type=Path,
        default=Path("results/explainability_report.json"),
        help="Destination JSON path for explainability metadata.",
    )
    parser.add_argument(
        "--shap-output",
        type=Path,
        default=Path("results/shap_values.csv"),
        help="Destination CSV path for SHAP values when XGBoost is used.",
    )
    parser.add_argument(
        "--cv-splits",
        type=int,
        default=5,
        help="Requested number of stratified CV folds.",
    )
    parser.add_argument(
        "--prefer-model",
        choices=["svm", "logistic"],
        default="svm",
        help="Fallback model to explain when XGBoost is not requested.",
    )
    parser.add_argument(
        "--include-xgboost",
        action="store_true",
        help="Attempt XGBoost explainability and SHAP if dependencies exist.",
    )
    parser.add_argument(
        "--permutation-repeats",
        type=int,
        default=10,
        help="Number of repeats for permutation importance.",
    )
    return parser


def main() -> int:
    """Run the Week 7 explainability suite."""
    import pandas as pd

    from obpi.ml.explainability import (
        run_explainability,
        save_explainability_results,
    )

    args = build_parser().parse_args()
    prepared_df = pd.read_parquet(args.input_path)
    result = run_explainability(
        prepared_df,
        cv_splits=args.cv_splits,
        prefer_model=args.prefer_model,
        include_xgboost=args.include_xgboost,
        permutation_repeats=args.permutation_repeats,
    )
    save_explainability_results(
        result,
        weights_path=args.weights_output,
        permutation_path=args.permutation_output,
        report_path=args.report_output,
        shap_path=args.shap_output,
    )
    print(f"model_name: {result.model_name}")
    print(f"top_metric: {result.permutation_importance.iloc[0]['metric']}")
    print(f"weights_output: {args.weights_output}")
    print(f"permutation_output: {args.permutation_output}")
    print(f"report_output: {args.report_output}")
    if result.shap_values is not None:
        print(f"shap_output: {args.shap_output}")
    else:
        print("shap_output: skipped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Generate the OBPI research-validation audit from current artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser."""
    parser = argparse.ArgumentParser(
        description="Create a consolidated OBPI research-validation audit."
    )
    parser.add_argument(
        "--metrics-path",
        type=Path,
        default=Path("data/processed/player_match_metrics.parquet"),
        help="Raw processed metric parquet with 360 coverage columns.",
    )
    parser.add_argument(
        "--scores-path",
        type=Path,
        default=Path("data/processed/player_obpi_scores.parquet"),
        help="Fuzzy-scored OBPI parquet containing normalized M1..M9.",
    )
    parser.add_argument(
        "--prepared-path",
        type=Path,
        default=Path("data/processed/training_prepared.parquet"),
        help="Training-prepared parquet with labels.",
    )
    parser.add_argument(
        "--cv-report",
        type=Path,
        default=Path("results/cv_results.json"),
        help="Cross-validation JSON report.",
    )
    parser.add_argument(
        "--explainability-report",
        type=Path,
        default=Path("results/explainability_report.json"),
        help="Explainability JSON report.",
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=Path("data/interim/events_manifest.parquet"),
        help="Optional interim event manifest with 360 coverage fields.",
    )
    parser.add_argument(
        "--shap-path",
        type=Path,
        default=Path("results/shap_values.csv"),
        help="Optional SHAP values CSV.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("results/research_validation_audit.json"),
        help="Destination JSON audit path.",
    )
    parser.add_argument(
        "--output-markdown",
        type=Path,
        default=Path("results/RESEARCH_VALIDATION.md"),
        help="Destination Markdown audit path.",
    )
    return parser


def main() -> int:
    """Run the validation-audit generation."""
    import pandas as pd

    from obpi.ml.research_validation import (
        build_validation_audit,
        save_validation_audit,
    )

    args = build_parser().parse_args()
    metrics_df = pd.read_parquet(args.metrics_path)
    scored_df = pd.read_parquet(args.scores_path)
    prepared_df = pd.read_parquet(args.prepared_path)
    cv_report = json.loads(args.cv_report.read_text(encoding="utf-8"))
    explainability_report = json.loads(
        args.explainability_report.read_text(encoding="utf-8")
    )
    manifest_df = (
        pd.read_parquet(args.manifest_path) if args.manifest_path.exists() else None
    )
    shap_values = pd.read_csv(args.shap_path) if args.shap_path.exists() else None

    audit = build_validation_audit(
        metrics_df=metrics_df,
        scored_df=scored_df,
        prepared_df=prepared_df,
        cv_report=cv_report,
        explainability_report=explainability_report,
        manifest_df=manifest_df,
        shap_values=shap_values,
    )
    save_validation_audit(
        audit,
        json_path=args.output_json,
        markdown_path=args.output_markdown,
    )
    print(f"pipeline_validation: {audit['validity_status']['pipeline_validation']}")
    print(f"external_validation: {audit['validity_status']['external_validation']}")
    print(f"output_json: {args.output_json}")
    print(f"output_markdown: {args.output_markdown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

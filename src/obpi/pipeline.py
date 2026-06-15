"""Command-line scoring pipeline for OBPI metric tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from obpi.fuzzy.engine import DEFAULT_METRICS
from obpi.fuzzy.membership import summarize_metric_memberships
from obpi.fuzzy.scoring import fit_fuzzy_engine, score_metrics_dataframe
from obpi.ml.validation import validate


def read_metrics_table(path: Path) -> pd.DataFrame:
    """Read a CSV or Parquet metrics table."""

    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".parquet":
        try:
            return pd.read_parquet(path)
        except ImportError as exc:
            raise RuntimeError(
                "Parquet input requires pyarrow or fastparquet. "
                "Install the project requirements and try again."
            ) from exc
    raise ValueError(f"unsupported input format: {path.suffix}")


def write_metrics_table(metrics_df: pd.DataFrame, path: Path) -> None:
    """Write a CSV or Parquet metrics table."""

    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        metrics_df.to_csv(path, index=False)
        return
    if suffix == ".parquet":
        try:
            metrics_df.to_parquet(path, index=False)
        except ImportError as exc:
            raise RuntimeError(
                "Parquet output requires pyarrow or fastparquet. "
                "Install the project requirements and try again."
            ) from exc
        return
    raise ValueError(f"unsupported output format: {path.suffix}")


def run_scoring_pipeline(
    input_path: Path,
    output_path: Path,
    metric_names: list[str] | None = None,
    score_column: str = "obpi",
    run_validation: bool = False,
    membership_report_path: Path | None = None,
) -> dict[str, Any]:
    """Score an input metrics table and optionally run ML validation."""

    metric_names = metric_names or list(DEFAULT_METRICS)
    metrics_df = read_metrics_table(input_path)
    engine = fit_fuzzy_engine(metrics_df, metric_names=metric_names)
    scored_df = score_metrics_dataframe(
        metrics_df,
        engine=engine,
        metric_names=metric_names,
        score_column=score_column,
    )
    write_metrics_table(scored_df, output_path)

    membership_report = summarize_metric_memberships(engine.membership_functions)
    if membership_report_path is not None:
        membership_report_path.parent.mkdir(parents=True, exist_ok=True)
        membership_report_path.write_text(
            json.dumps(membership_report, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    summary: dict[str, Any] = {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "rows_scored": int(len(scored_df)),
        "score_column": score_column,
        "score_min": float(scored_df[score_column].min()),
        "score_max": float(scored_df[score_column].max()),
        "membership_report_path": (
            str(membership_report_path) if membership_report_path is not None else None
        ),
    }
    if run_validation:
        summary["validation"] = validate(
            scored_df,
            score_column=score_column,
            metric_columns=metric_names,
        )
    return summary


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""

    parser = argparse.ArgumentParser(
        description="Fit the fuzzy OBPI engine and score a metrics table.",
    )
    parser.add_argument("input", type=Path, help="Input CSV or Parquet metrics table")
    parser.add_argument("output", type=Path, help="Output CSV or Parquet path")
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=list(DEFAULT_METRICS),
        help="Metric columns to score. Defaults to M1 M2 ... M9.",
    )
    parser.add_argument(
        "--score-column",
        default="obpi",
        help="Name of the output score column. Defaults to obpi.",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run SVM/logistic validation after scoring.",
    )
    parser.add_argument(
        "--membership-report",
        type=Path,
        help="Optional JSON path for fitted membership percentiles and trapezoids.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the CLI."""

    args = build_parser().parse_args(argv)
    summary = run_scoring_pipeline(
        input_path=args.input,
        output_path=args.output,
        metric_names=args.metrics,
        score_column=args.score_column,
        run_validation=args.validate,
        membership_report_path=args.membership_report,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

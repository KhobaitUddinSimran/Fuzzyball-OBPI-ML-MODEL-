"""Compare match-level and aggregate player-level OBPI validation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser."""
    parser = argparse.ArgumentParser(
        description="Compare OBPI validation at player-match and aggregate-player levels."
    )
    parser.add_argument(
        "--scores-path",
        type=Path,
        default=Path("data/processed/player_obpi_scores.parquet"),
        help="Player-match OBPI score parquet.",
    )
    parser.add_argument(
        "--match-cv-report",
        type=Path,
        default=Path("results/cv_results.json"),
        help="Existing match-level validation report.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("results/match_vs_aggregate_validation.json"),
        help="Destination JSON report.",
    )
    parser.add_argument(
        "--output-markdown",
        type=Path,
        default=Path("results/MATCH_VS_AGGREGATE_VALIDATION.md"),
        help="Destination Markdown report.",
    )
    parser.add_argument(
        "--include-xgboost",
        action="store_true",
        help="Include XGBoost in aggregate-level validation.",
    )
    return parser


def main() -> int:
    """Run match-level versus aggregate-player validation."""
    import pandas as pd

    from obpi.ml.validation import METRIC_COLUMNS, validate

    args = build_parser().parse_args()
    scored_df = pd.read_parquet(args.scores_path)
    match_report = json.loads(args.match_cv_report.read_text(encoding="utf-8"))
    aggregate_df = _aggregate_player_scores(scored_df, METRIC_COLUMNS)
    aggregate_report = validate(
        aggregate_df,
        cv_splits=5,
        include_xgboost=args.include_xgboost,
    )
    report = {
        "match_level": _summarize_report(match_report),
        "aggregate_player_level": _summarize_report(aggregate_report),
        "aggregate_rows": int(len(aggregate_df)),
        "interpretation": (
            "Aggregate validation tests whether the signal persists when repeated "
            "player-match rows are collapsed to one row per player."
        ),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    args.output_markdown.write_text(_markdown(report), encoding="utf-8")
    print(f"aggregate_rows: {len(aggregate_df)}")
    print(f"aggregate_samples: {aggregate_report['n_samples']}")
    print(f"output_json: {args.output_json}")
    print(f"output_markdown: {args.output_markdown}")
    return 0


def _aggregate_player_scores(scored_df: Any, metric_columns: list[str]) -> Any:
    grouped = scored_df.groupby(["player_id", "player_name"], dropna=False)
    aggregate = grouped[metric_columns + ["obpi"]].mean().reset_index()
    aggregate["match_count"] = grouped["match_id"].nunique().to_numpy()
    return aggregate


def _summarize_report(report: dict[str, Any]) -> dict[str, Any]:
    models = report.get("models", {})
    best_model = None
    if models:
        best_model = max(
            models,
            key=lambda name: float(models[name].get("accuracy_mean", 0.0)),
        )
    return {
        "n_samples": int(report.get("n_samples", 0)),
        "n_rows": int(report.get("n_rows", 0)),
        "class_counts": report.get("class_counts", {}),
        "best_accuracy_model": best_model,
        "models": models,
    }


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Match vs Aggregate Validation",
        "",
        report["interpretation"],
        "",
        "## Summary",
        "",
        "| Level | Samples | Best model | Best accuracy |",
        "|---|---:|---|---:|",
    ]
    for level_key, label in [
        ("match_level", "Player-match"),
        ("aggregate_player_level", "Aggregate player"),
    ]:
        level = report[level_key]
        best_model = level["best_accuracy_model"]
        best_accuracy = (
            level["models"][best_model]["accuracy_mean"] if best_model else None
        )
        lines.append(
            "| {label} | {samples} | {best_model} | {accuracy} |".format(
                label=label,
                samples=level["n_samples"],
                best_model=best_model,
                accuracy="n/a" if best_accuracy is None else f"{best_accuracy:.4f}",
            )
        )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())

"""Run external or expert-label validation for OBPI scores."""

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
        description=(
            "Validate aggregate OBPI scores against independent expert ratings "
            "or benchmark columns."
        )
    )
    parser.add_argument(
        "--scores-path",
        type=Path,
        default=Path("data/processed/player_obpi_scores.parquet"),
        help="Scored OBPI parquet file.",
    )
    parser.add_argument(
        "--external-path",
        type=Path,
        default=Path("data/external/expert_ratings.csv"),
        help="CSV containing player_id/player_name plus expert or benchmark scores.",
    )
    parser.add_argument(
        "--template-output",
        type=Path,
        default=Path("results/expert_ratings_template.csv"),
        help="Template CSV written when external labels are absent.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("results/external_validation.json"),
        help="Destination JSON report.",
    )
    parser.add_argument(
        "--output-markdown",
        type=Path,
        default=Path("results/EXTERNAL_VALIDATION.md"),
        help="Destination Markdown report.",
    )
    return parser


def main() -> int:
    """Run external validation or create a pending-label template."""
    import pandas as pd

    from obpi.ml.correlation import (
        compare_benchmarks,
        cronbach_alpha,
        expert_correlation,
    )

    args = build_parser().parse_args()
    scored_df = pd.read_parquet(args.scores_path)
    aggregate = _aggregate_scores(scored_df)

    if not args.external_path.exists():
        _write_template(aggregate, args.template_output)
        report = {
            "status": "pending_external_or_expert_labels",
            "reason": f"External label file not found: {args.external_path}",
            "template_output": str(args.template_output),
            "template_rows": int(min(len(aggregate), 50)),
            "expected_columns": [
                "player_id",
                "player_name",
                "expert_1",
                "expert_2",
                "expert_3",
                "expert_median",
                "benchmark_rating",
            ],
        }
        _write_report(report, args.output_json, args.output_markdown)
        print("external_validation: pending")
        print(f"template_output: {args.template_output}")
        return 0

    external_df = pd.read_csv(args.external_path)
    merged = _merge_external(aggregate, external_df)
    expert_columns = [
        column
        for column in merged.columns
        if column.startswith("expert_") and column != "expert_median"
    ]
    benchmark_columns = [
        column
        for column in external_df.columns
        if column
        not in {
            "player_id",
            "player_name",
            "expert_median",
            *expert_columns,
        }
    ]

    report: dict[str, Any] = {
        "status": "complete" if len(merged) >= 3 else "insufficient_overlap",
        "external_path": str(args.external_path),
        "overlap_rows": int(len(merged)),
        "matched_players": int(merged["player_id"].nunique())
        if "player_id" in merged
        else int(len(merged)),
        "expert_validation": None,
        "inter_rater_reliability": None,
        "benchmark_validation": [],
    }
    if len(merged) >= 3 and ("expert_median" in merged or expert_columns):
        expert_median = (
            merged["expert_median"]
            if "expert_median" in merged
            else merged[expert_columns].median(axis=1)
        )
        report["expert_validation"] = expert_correlation(
            merged["obpi_mean"],
            expert_median,
        )
    if len(merged) >= 2 and len(expert_columns) >= 2:
        report["inter_rater_reliability"] = {
            "cronbach_alpha": cronbach_alpha(merged[expert_columns])
        }
    if len(merged) >= 3 and benchmark_columns:
        benchmark_df = merged[benchmark_columns].apply(
            pd.to_numeric,
            errors="coerce",
        )
        report["benchmark_validation"] = compare_benchmarks(
            merged["obpi_mean"],
            benchmark_df,
        ).to_dict(orient="records")

    _write_report(report, args.output_json, args.output_markdown)
    print(f"external_validation: {report['status']}")
    print(f"overlap_rows: {report['overlap_rows']}")
    return 0


def _aggregate_scores(scored_df: Any) -> Any:
    grouped = scored_df.groupby(["player_id", "player_name"], dropna=False)
    aggregate = grouped.agg(
        obpi_mean=("obpi", "mean"),
        obpi_median=("obpi", "median"),
        obpi_matches=("match_id", "nunique"),
    ).reset_index()
    metric_means = grouped[[f"M{i}" for i in range(1, 10)]].mean().reset_index()
    aggregate = aggregate.merge(metric_means, on=["player_id", "player_name"])
    return aggregate.sort_values("obpi_mean", ascending=False, ignore_index=True)


def _write_template(aggregate: Any, output_path: Path) -> None:
    template = aggregate.head(50)[["player_id", "player_name", "obpi_mean"]].copy()
    for column in [
        "expert_1",
        "expert_2",
        "expert_3",
        "expert_median",
        "benchmark_rating",
    ]:
        template[column] = ""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    template.to_csv(output_path, index=False)


def _merge_external(aggregate: Any, external_df: Any) -> Any:
    if "player_id" in external_df.columns:
        return aggregate.merge(external_df, on="player_id", how="inner")
    if "player_name" in external_df.columns:
        return aggregate.merge(external_df, on="player_name", how="inner")
    raise ValueError("external CSV must include player_id or player_name")


def _write_report(report: dict[str, Any], json_path: Path, markdown_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(_markdown(report), encoding="utf-8")


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# External Validation",
        "",
        f"- status: {report['status']}",
    ]
    if report["status"] == "pending_external_or_expert_labels":
        lines.extend(
            [
                f"- reason: {report['reason']}",
                f"- template_output: {report['template_output']}",
                "",
                (
                    "Fill the template with independent expert ratings or benchmark "
                    "scores, then rerun `python3 scripts/run_external_validation.py`."
                ),
            ]
        )
        return "\n".join(lines) + "\n"

    lines.extend(
        [
            f"- overlap_rows: {report['overlap_rows']}",
            f"- matched_players: {report['matched_players']}",
            "",
            "## Expert Validation",
            "",
        ]
    )
    lines.append(f"- {report['expert_validation']}")
    lines.extend(["", "## Inter-Rater Reliability", ""])
    lines.append(f"- {report['inter_rater_reliability']}")
    lines.extend(["", "## Benchmark Validation", ""])
    for item in report["benchmark_validation"]:
        lines.append(f"- {item}")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())

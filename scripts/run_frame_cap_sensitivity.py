"""Run 360 frame-cap sensitivity validation for OBPI metrics."""

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
            "Recompute the 360-backed attacking-midfield validation subset at "
            "multiple freeze-frame caps and compare model stability."
        )
    )
    parser.add_argument(
        "--interim-dir",
        type=Path,
        default=Path("data/interim"),
        help="Directory containing normalized interim parquet tables.",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path("data/validation/frame_cap_sensitivity"),
        help="Local working directory for cap-specific parquet outputs.",
    )
    parser.add_argument(
        "--caps",
        type=int,
        nargs="+",
        default=[25, 50, 75],
        help="Frame caps to evaluate.",
    )
    parser.add_argument(
        "--position-keyword",
        action="append",
        default=["Attacking Midfield"],
        help="Position keyword filter. May be supplied multiple times.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("results/frame_cap_sensitivity.json"),
        help="Destination JSON summary.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("results/frame_cap_sensitivity.csv"),
        help="Destination CSV summary.",
    )
    parser.add_argument(
        "--output-markdown",
        type=Path,
        default=Path("results/FRAME_CAP_SENSITIVITY.md"),
        help="Destination Markdown report.",
    )
    parser.add_argument(
        "--include-xgboost",
        action="store_true",
        help="Include XGBoost in each cap-specific validation run.",
    )
    return parser


def main() -> int:
    """Run frame-cap sensitivity analysis."""
    import pandas as pd

    from obpi.data.metric_processing import InterimMetricsProcessor
    from obpi.fuzzy.processing import run_real_data_fuzzy_processing
    from obpi.ml.validation import prepare_training_frame, validate_prepared_data

    args = build_parser().parse_args()
    rows: list[dict[str, Any]] = []
    args.work_dir.mkdir(parents=True, exist_ok=True)

    for cap in args.caps:
        cap_dir = args.work_dir / f"cap_{cap}"
        processor = InterimMetricsProcessor(
            interim_dir=args.interim_dir,
            output_dir=cap_dir,
        )
        match_metrics = processor.process_matches(
            require_360=True,
            max_frames_per_match=cap,
            position_keywords=args.position_keyword,
        )
        scored_df, _ = run_real_data_fuzzy_processing(match_metrics)
        prepared = prepare_training_frame(scored_df)
        report = validate_prepared_data(
            prepared.prepared_df,
            include_xgboost=args.include_xgboost,
        )

        model_rows = {
            f"{model_name}_{metric_name}": float(model_report[metric_name])
            for model_name, model_report in report["models"].items()
            for metric_name in (
                "accuracy_mean",
                "accuracy_std",
                "roc_auc_mean",
                "recall_class_1_mean",
            )
        }
        rows.append(
            {
                "frame_cap": cap,
                "player_match_rows": int(len(match_metrics)),
                "players": int(match_metrics["player_id"].nunique()),
                "matches": int(match_metrics["match_id"].nunique()),
                "rows_with_360": int(match_metrics["has_360_data"].sum())
                if "has_360_data" in match_metrics
                else None,
                "frame_count_mean": float(match_metrics["freeze_frame_count"].mean())
                if "freeze_frame_count" in match_metrics
                else None,
                "prepared_rows": int(report["n_samples"]),
                "class_0": int(report["class_counts"].get("0", 0)),
                "class_1": int(report["class_counts"].get("1", 0)),
                **model_rows,
            }
        )
        print(f"cap={cap} rows={len(match_metrics)} samples={report['n_samples']}")

    summary_df = pd.DataFrame(rows)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(args.output_csv, index=False)
    args.output_json.write_text(
        json.dumps({"caps": rows}, indent=2) + "\n",
        encoding="utf-8",
    )
    args.output_markdown.write_text(_markdown(summary_df), encoding="utf-8")
    print(f"output_csv: {args.output_csv}")
    print(f"output_json: {args.output_json}")
    print(f"output_markdown: {args.output_markdown}")
    return 0


def _markdown(summary_df: Any) -> str:
    lines = [
        "# Frame-Cap Sensitivity",
        "",
        (
            "This report reruns the 360-backed attacking-midfield validation subset "
            "with different per-match freeze-frame caps."
        ),
        "",
        "| Frame cap | Rows | Players | Matches | Samples | XGBoost accuracy | SVM accuracy |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_df.to_dict(orient="records"):
        xgb_accuracy = row.get("xgboost_accuracy_mean")
        svm_accuracy = row.get("svm_accuracy_mean")
        lines.append(
            "| {frame_cap} | {player_match_rows} | {players} | {matches} | "
            "{prepared_rows} | {xgb} | {svm} |".format(
                frame_cap=row["frame_cap"],
                player_match_rows=row["player_match_rows"],
                players=row["players"],
                matches=row["matches"],
                prepared_rows=row["prepared_rows"],
                xgb=_format_optional(xgb_accuracy),
                svm=_format_optional(svm_accuracy),
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            (
                "Stable accuracy across caps supports robustness to 360 sampling. "
                "Large swings mean the metric processor should be rerun with a "
                "larger cap for final research reporting."
            ),
        ]
    )
    return "\n".join(lines) + "\n"


def _format_optional(value: object) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.4f}"


if __name__ == "__main__":
    raise SystemExit(main())

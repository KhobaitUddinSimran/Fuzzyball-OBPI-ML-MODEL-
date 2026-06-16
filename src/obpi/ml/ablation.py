"""Ablation study helpers for OBPI metrics."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pandas as pd

from obpi.ml.validation import METRIC_COLUMNS, validate


def run_ablation(
    metrics_df: pd.DataFrame,
    score_column: str = "obpi",
    metric_columns: list[str] | None = None,
    cv_splits: int = 5,
    model_name: str = "svm",
    validation_fn: Callable[..., dict[str, Any]] = validate,
) -> pd.DataFrame:
    """Run leave-one-metric-out validation and return accuracy drops.

    `delta_accuracy` is defined as `full_accuracy - ablated_accuracy`; positive
    values mean removing the metric reduced validation accuracy.
    """

    metric_columns = metric_columns or METRIC_COLUMNS
    full_report = validation_fn(
        metrics_df,
        score_column=score_column,
        metric_columns=metric_columns,
        cv_splits=cv_splits,
    )
    full_accuracy = _extract_accuracy(full_report, model_name)

    rows = []
    for removed_metric in metric_columns:
        remaining_metrics = [
            metric for metric in metric_columns if metric != removed_metric
        ]
        ablated_report = validation_fn(
            metrics_df,
            score_column=score_column,
            metric_columns=remaining_metrics,
            cv_splits=cv_splits,
        )
        ablated_accuracy = _extract_accuracy(ablated_report, model_name)
        rows.append(
            {
                "removed_metric": removed_metric,
                "remaining_metrics": len(remaining_metrics),
                "full_accuracy": full_accuracy,
                "ablated_accuracy": ablated_accuracy,
                "delta_accuracy": full_accuracy - ablated_accuracy,
                "redundancy_flag": (full_accuracy - ablated_accuracy) < 0.02,
            }
        )

    return pd.DataFrame(rows).sort_values(
        "delta_accuracy",
        ascending=False,
        ignore_index=True,
    )


def write_ablation_report(
    ablation_df: pd.DataFrame,
    output_path: str,
    synthetic: bool = True,
) -> None:
    """Write a compact Markdown ablation report."""

    warning = ""
    if synthetic:
        warning = (
            "\n> Synthetic smoke-test only. Do not interpret metric redundancy "
            "until real StatsBomb-derived metrics are used.\n"
        )

    lines = [
        "# Ablation Benchmark Report",
        warning,
        "| Removed metric | Full accuracy | Ablated accuracy | Delta accuracy | Redundant? |",
        "|---|---:|---:|---:|---|",
    ]
    for row in ablation_df.to_dict(orient="records"):
        lines.append(
            "| {removed_metric} | {full_accuracy:.3f} | {ablated_accuracy:.3f} | "
            "{delta_accuracy:.3f} | {redundancy_flag} |".format(**row)
        )

    with open(output_path, "w", encoding="utf-8") as file:
        file.write("\n".join(lines) + "\n")


def _extract_accuracy(report: dict[str, Any], model_name: str) -> float:
    try:
        model_report = report["models"][model_name]
    except KeyError as exc:
        raise ValueError(f"model {model_name!r} not found in validation report") from exc

    if "accuracy_mean" in model_report:
        return float(model_report["accuracy_mean"])
    if "accuracy" in model_report:
        return float(model_report["accuracy"])
    raise ValueError(f"model {model_name!r} report does not contain accuracy")


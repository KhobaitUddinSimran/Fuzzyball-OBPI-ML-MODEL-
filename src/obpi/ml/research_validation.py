"""Research-validation audit helpers for OBPI outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from obpi.ml.validation import METRIC_COLUMNS


def build_validation_audit(
    metrics_df: pd.DataFrame,
    scored_df: pd.DataFrame,
    prepared_df: pd.DataFrame,
    cv_report: dict[str, Any],
    explainability_report: dict[str, Any],
    manifest_df: pd.DataFrame | None = None,
    shap_values: pd.DataFrame | None = None,
    sensitivity_report: dict[str, Any] | None = None,
    external_report: dict[str, Any] | None = None,
    aggregate_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a compact research-validity audit from pipeline artifacts."""
    validity_status = _validity_status(
        metrics_df,
        cv_report,
        explainability_report,
        sensitivity_report,
        external_report,
    )
    return {
        "data_coverage": _data_coverage(metrics_df, manifest_df),
        "target_population": _target_population(metrics_df),
        "score_distribution": _score_distribution(scored_df),
        "training_labels": _training_labels(prepared_df),
        "metric_variation": _metric_variation(scored_df),
        "model_validation": _model_validation(cv_report),
        "explainability": _explainability(explainability_report, shap_values),
        "robustness_validation": _robustness_validation(sensitivity_report),
        "aggregate_validation": aggregate_report,
        "external_validation_report": external_report,
        "validity_status": validity_status,
        "next_validation_requirements": _next_validation_requirements(validity_status),
    }


def save_validation_audit(
    audit: dict[str, Any],
    json_path: str | Path,
    markdown_path: str | Path,
) -> None:
    """Persist the validation audit as JSON and Markdown."""
    json_output = Path(json_path)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")

    markdown_output = Path(markdown_path)
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.write_text(_audit_markdown(audit), encoding="utf-8")


def _data_coverage(
    metrics_df: pd.DataFrame,
    manifest_df: pd.DataFrame | None,
) -> dict[str, Any]:
    coverage: dict[str, Any] = {
        "processed_player_match_rows": int(len(metrics_df)),
        "processed_players": int(metrics_df["player_id"].nunique())
        if "player_id" in metrics_df
        else 0,
        "processed_matches": int(metrics_df["match_id"].nunique())
        if "match_id" in metrics_df
        else 0,
        "rows_with_360_data": int(metrics_df.get("has_360_data", pd.Series()).sum())
        if "has_360_data" in metrics_df
        else None,
        "frame_count_mean": _optional_float(metrics_df, "freeze_frame_count", "mean"),
        "frame_count_min": _optional_float(metrics_df, "freeze_frame_count", "min"),
        "frame_count_max": _optional_float(metrics_df, "freeze_frame_count", "max"),
    }
    if manifest_df is not None:
        coverage.update(
            {
                "manifest_matches": int(len(manifest_df)),
                "matches_with_three_sixty_file": int(
                    manifest_df["has_three_sixty_file"].sum()
                )
                if "has_three_sixty_file" in manifest_df
                else None,
                "matches_with_freeze_frames": int(
                    (manifest_df["freeze_frame_event_count"] > 0).sum()
                )
                if "freeze_frame_event_count" in manifest_df
                else None,
                "freeze_frame_events": int(manifest_df["freeze_frame_event_count"].sum())
                if "freeze_frame_event_count" in manifest_df
                else None,
            }
        )
    return coverage


def _target_population(metrics_df: pd.DataFrame) -> dict[str, Any]:
    positions = (
        metrics_df["starting_position_name"].value_counts().to_dict()
        if "starting_position_name" in metrics_df
        else {}
    )
    return {
        "positions": {str(key): int(value) for key, value in positions.items()},
        "note": "Current validation subset is attacking-midfield roles with 360 coverage.",
    }


def _score_distribution(scored_df: pd.DataFrame) -> dict[str, float]:
    scores = scored_df["obpi"].astype(float)
    return {
        "count": int(scores.count()),
        "mean": float(scores.mean()),
        "std": float(scores.std()),
        "min": float(scores.min()),
        "p25": float(scores.quantile(0.25)),
        "median": float(scores.median()),
        "p75": float(scores.quantile(0.75)),
        "max": float(scores.max()),
    }


def _training_labels(prepared_df: pd.DataFrame) -> dict[str, Any]:
    counts = prepared_df["label"].value_counts().sort_index()
    return {
        "prepared_rows": int(len(prepared_df)),
        "class_counts": {str(label): int(count) for label, count in counts.items()},
        "label_source": "Top and bottom OBPI quartiles; not independent external labels.",
    }


def _metric_variation(scored_df: pd.DataFrame) -> dict[str, dict[str, float]]:
    rows: dict[str, dict[str, float]] = {}
    for metric in METRIC_COLUMNS:
        series = scored_df[metric].astype(float)
        rows[metric] = {
            "mean": float(series.mean()),
            "std": float(series.std()),
            "min": float(series.min()),
            "max": float(series.max()),
            "nonzero_fraction": float((series != 0).mean()),
        }
    return rows


def _model_validation(cv_report: dict[str, Any]) -> dict[str, Any]:
    models = cv_report.get("models", {})
    best_model = None
    if models:
        best_model = max(
            models,
            key=lambda name: float(models[name].get("accuracy_mean", 0.0)),
        )
    return {
        "n_samples": int(cv_report.get("n_samples", 0)),
        "class_counts": cv_report.get("class_counts", {}),
        "models": models,
        "best_accuracy_model": best_model,
        "notes": cv_report.get("notes", []),
    }


def _explainability(
    explainability_report: dict[str, Any],
    shap_values: pd.DataFrame | None,
) -> dict[str, Any]:
    summary = {
        "model_name": explainability_report.get("model_name"),
        "metric_weights": explainability_report.get("metric_weights", {}),
        "permutation_importance_ranking": explainability_report.get(
            "permutation_importance_ranking",
            [],
        ),
        "shap_available": bool(explainability_report.get("shap_available", False)),
        "notes": explainability_report.get("notes", []),
    }
    if shap_values is not None and not shap_values.empty:
        mean_abs = shap_values.abs().mean().sort_values(ascending=False)
        summary["mean_abs_shap_ranking"] = {
            str(metric): float(value) for metric, value in mean_abs.items()
        }
    return summary


def _validity_status(
    metrics_df: pd.DataFrame,
    cv_report: dict[str, Any],
    explainability_report: dict[str, Any],
    sensitivity_report: dict[str, Any] | None = None,
    external_report: dict[str, Any] | None = None,
) -> dict[str, str]:
    all_rows_360 = (
        "has_360_data" in metrics_df
        and bool(metrics_df["has_360_data"].all())
        and len(metrics_df) > 0
    )
    has_xgboost = "xgboost" in cv_report.get("models", {})
    has_shap = bool(explainability_report.get("shap_available", False))
    has_sensitivity = bool(sensitivity_report and sensitivity_report.get("caps"))
    external_status = (
        str(external_report.get("status"))
        if external_report is not None
        else "pending_external_or_expert_labels"
    )
    return {
        "pipeline_validation": "complete"
        if all_rows_360 and has_xgboost and has_shap
        else "partial",
        "construct_validation": "internal_obpi_extreme_quartile_only",
        "robustness_validation": "complete" if has_sensitivity else "pending",
        "external_validation": external_status,
        "interpretation": (
            "Use current scores as 360-enriched internal validation evidence. "
            "Do not present them as final independent convergent validity unless "
            "external_validation is complete."
        ),
    }


def _robustness_validation(
    sensitivity_report: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if not sensitivity_report or not sensitivity_report.get("caps"):
        return None

    caps = sensitivity_report["caps"]
    rows = []
    for cap_report in caps:
        rows.append(
            {
                "frame_cap": int(cap_report["frame_cap"]),
                "player_match_rows": int(cap_report["player_match_rows"]),
                "prepared_rows": int(cap_report["prepared_rows"]),
                "xgboost_accuracy_mean": _maybe_float(
                    cap_report.get("xgboost_accuracy_mean")
                ),
                "svm_accuracy_mean": _maybe_float(cap_report.get("svm_accuracy_mean")),
                "logistic_accuracy_mean": _maybe_float(
                    cap_report.get("logistic_accuracy_mean")
                ),
            }
        )
    xgb_values = [
        row["xgboost_accuracy_mean"]
        for row in rows
        if row["xgboost_accuracy_mean"] is not None
    ]
    return {
        "status": "complete",
        "caps": rows,
        "xgboost_accuracy_range": (
            float(max(xgb_values) - min(xgb_values)) if xgb_values else None
        ),
    }


def _next_validation_requirements(validity_status: dict[str, str]) -> list[str]:
    requirements: list[str] = []
    if validity_status["external_validation"] != "complete":
        requirements.extend(
            [
                "Collect independent player-quality labels or expert ratings.",
                "Run Spearman correlation between OBPI and external/expert ratings.",
                "Compute inter-rater reliability if expert-panel ratings are used.",
            ]
        )
    return requirements


def _maybe_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _optional_float(df: pd.DataFrame, column: str, method: str) -> float | None:
    if column not in df:
        return None
    value = getattr(df[column].astype(float), method)()
    return float(value)


def _audit_markdown(audit: dict[str, Any]) -> str:
    lines = [
        "# Research Validation Audit",
        "",
        "## Data Coverage",
        "",
    ]
    for key, value in audit["data_coverage"].items():
        lines.append(f"- {key}: {value}")

    lines.extend(["", "## Target Population", ""])
    for position, count in audit["target_population"]["positions"].items():
        lines.append(f"- {position}: {count}")
    lines.append(f"- note: {audit['target_population']['note']}")

    lines.extend(["", "## Model Validation", ""])
    lines.append(f"- samples: {audit['model_validation']['n_samples']}")
    lines.append(f"- class_counts: {audit['model_validation']['class_counts']}")
    lines.append(f"- best_accuracy_model: {audit['model_validation']['best_accuracy_model']}")
    for model_name, metrics in audit["model_validation"]["models"].items():
        lines.extend(
            [
                f"- {model_name} accuracy: {metrics['accuracy_mean']:.4f}",
                f"- {model_name} roc_auc: {metrics['roc_auc_mean']:.4f}",
                f"- {model_name} recall_class_1: {metrics['recall_class_1_mean']:.4f}",
            ]
        )

    lines.extend(["", "## Explainability", ""])
    lines.append(f"- model_name: {audit['explainability']['model_name']}")
    lines.append(f"- shap_available: {audit['explainability']['shap_available']}")
    ranking = audit["explainability"].get("permutation_importance_ranking", [])
    lines.append(f"- permutation_top_5: {ranking[:5]}")
    if "mean_abs_shap_ranking" in audit["explainability"]:
        top_shap = list(audit["explainability"]["mean_abs_shap_ranking"].items())[:5]
        lines.append(f"- mean_abs_shap_top_5: {top_shap}")

    lines.extend(["", "## Validity Status", ""])
    for key, value in audit["validity_status"].items():
        lines.append(f"- {key}: {value}")

    lines.extend(["", "## Robustness Validation", ""])
    robustness = audit.get("robustness_validation")
    if robustness is None:
        lines.append("- status: pending")
    else:
        lines.append(f"- status: {robustness['status']}")
        lines.append(
            f"- xgboost_accuracy_range: {robustness['xgboost_accuracy_range']}"
        )
        for cap_report in robustness["caps"]:
            lines.append(
                "- cap {frame_cap}: rows={player_match_rows}, samples={prepared_rows}, "
                "xgboost_accuracy={xgboost_accuracy_mean}".format(**cap_report)
            )

    lines.extend(["", "## External Validation", ""])
    external = audit.get("external_validation_report")
    if external is None:
        lines.append("- status: pending_external_or_expert_labels")
    else:
        lines.append(f"- status: {external['status']}")
        if "template_output" in external:
            lines.append(f"- template_output: {external['template_output']}")
        if "expert_validation" in external:
            lines.append(f"- expert_validation: {external['expert_validation']}")
        if "inter_rater_reliability" in external:
            lines.append(
                f"- inter_rater_reliability: {external['inter_rater_reliability']}"
            )

    lines.extend(["", "## Match vs Aggregate Validation", ""])
    aggregate = audit.get("aggregate_validation")
    if aggregate is None:
        lines.append("- status: pending")
    else:
        match_level = aggregate["match_level"]
        aggregate_level = aggregate["aggregate_player_level"]
        lines.append(
            f"- match_level_best_model: {match_level['best_accuracy_model']}"
        )
        lines.append(
            "- match_level_samples: "
            f"{match_level['n_samples']}"
        )
        lines.append(
            "- aggregate_level_best_model: "
            f"{aggregate_level['best_accuracy_model']}"
        )
        lines.append(
            "- aggregate_level_samples: "
            f"{aggregate_level['n_samples']}"
        )

    lines.extend(["", "## Next Validation Requirements", ""])
    for item in audit["next_validation_requirements"]:
        lines.append(f"- {item}")

    return "\n".join(lines) + "\n"

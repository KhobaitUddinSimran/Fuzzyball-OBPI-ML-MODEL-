from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from obpi.ml.research_validation import (
    build_validation_audit,
    save_validation_audit,
)


def test_build_validation_audit_marks_pipeline_complete_with_xgb_and_shap() -> None:
    metrics_df = _make_metrics_frame()
    scored_df = _make_scored_frame()
    prepared_df = scored_df.head(4).copy()
    prepared_df["label"] = [0, 0, 1, 1]
    cv_report = {
        "n_samples": 4,
        "class_counts": {"0": 2, "1": 2},
        "models": {
            "xgboost": {
                "accuracy_mean": 0.9,
                "roc_auc_mean": 0.95,
                "recall_class_1_mean": 1.0,
            }
        },
        "notes": [],
    }
    explainability_report = {
        "model_name": "xgboost",
        "metric_weights": {"M9": 0.7, "M7": 0.3},
        "permutation_importance_ranking": ["M9", "M7"],
        "shap_available": True,
        "notes": [],
    }
    manifest_df = pd.DataFrame(
        {
            "has_three_sixty_file": [True, True],
            "freeze_frame_event_count": [10, 20],
        }
    )
    shap_values = pd.DataFrame({"M1": [0.0, 0.0], "M9": [1.0, -2.0]})

    audit = build_validation_audit(
        metrics_df=metrics_df,
        scored_df=scored_df,
        prepared_df=prepared_df,
        cv_report=cv_report,
        explainability_report=explainability_report,
        manifest_df=manifest_df,
        shap_values=shap_values,
    )

    assert audit["validity_status"]["pipeline_validation"] == "complete"
    assert audit["validity_status"]["external_validation"] == (
        "pending_external_or_expert_labels"
    )
    assert audit["data_coverage"]["processed_player_match_rows"] == 6
    assert audit["explainability"]["mean_abs_shap_ranking"]["M9"] == 1.5


def test_save_validation_audit_writes_json_and_markdown(tmp_path: Path) -> None:
    audit = {
        "data_coverage": {"processed_player_match_rows": 6},
        "target_population": {
            "positions": {"Center Attacking Midfield": 6},
            "note": "test population",
        },
        "model_validation": {
            "n_samples": 4,
            "class_counts": {"0": 2, "1": 2},
            "best_accuracy_model": "xgboost",
            "models": {
                "xgboost": {
                    "accuracy_mean": 0.9,
                    "roc_auc_mean": 0.95,
                    "recall_class_1_mean": 1.0,
                }
            },
        },
        "explainability": {
            "model_name": "xgboost",
            "shap_available": True,
            "permutation_importance_ranking": ["M9"],
        },
        "validity_status": {
            "pipeline_validation": "complete",
            "external_validation": "pending_external_or_expert_labels",
        },
        "next_validation_requirements": ["Collect expert ratings."],
    }
    json_path = tmp_path / "audit.json"
    markdown_path = tmp_path / "audit.md"

    save_validation_audit(audit, json_path, markdown_path)

    assert json.loads(json_path.read_text())["validity_status"][
        "pipeline_validation"
    ] == "complete"
    assert "# Research Validation Audit" in markdown_path.read_text()


def _make_metrics_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "player_id": [1, 2, 3, 4, 5, 6],
            "match_id": [10, 10, 11, 11, 12, 12],
            "starting_position_name": ["Center Attacking Midfield"] * 6,
            "has_360_data": [True] * 6,
            "freeze_frame_count": [25] * 6,
        }
    )


def _make_scored_frame() -> pd.DataFrame:
    rows = []
    for idx, value in enumerate(np.linspace(0.1, 0.9, 6), start=1):
        row = {
            "player_id": idx,
            "match_id": 10 + idx,
            "obpi": float(value),
        }
        for metric_idx in range(1, 10):
            row[f"M{metric_idx}"] = float(value)
        rows.append(row)
    return pd.DataFrame(rows)

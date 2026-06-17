"""Tests for training and reporting from prepared validation data."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from obpi.ml.validation import (
    METRIC_COLUMNS,
    extract_prepared_xy,
    save_validation_results,
    validate_prepared_data,
)


def _make_prepared_frame(n_rows: int = 40) -> pd.DataFrame:
    values = np.linspace(0.05, 0.95, n_rows)
    rows = []
    for idx, value in enumerate(values):
        row = {
            "player_id": idx,
            "player_name": f"Player {idx}",
            "team_id": 10,
            "team_name": "Team A",
            "match_id": 1000 + idx,
            "minutes": 90.0,
            "starting_position_name": "Center Attacking Midfield",
            "obpi": float(value),
        }
        label = 0 if idx < n_rows // 2 else 1
        row["label"] = label
        for metric_idx in range(1, 10):
            metric_value = float(np.clip(value + (metric_idx - 5) * 0.02, 0.0, 1.0))
            row[f"M{metric_idx}"] = metric_value
            row[f"M{metric_idx}_scaled"] = metric_value
        rows.append(row)
    return pd.DataFrame(rows)


def test_extract_prepared_xy_reads_features_and_labels() -> None:
    prepared = _make_prepared_frame()
    x, y = extract_prepared_xy(prepared)
    assert x.shape == (40, 9)
    assert len(y) == 40
    assert set(x.columns) == set(METRIC_COLUMNS)
    assert set(y.unique()) == {0, 1}


def test_validate_prepared_data_returns_models() -> None:
    prepared = _make_prepared_frame()
    report = validate_prepared_data(prepared, cv_splits=3)
    assert report["n_rows"] == 40
    assert report["n_samples"] == 40
    assert set(report["models"]) == {"logistic", "svm"}


def test_save_validation_results_writes_json_and_markdown(tmp_path: Path) -> None:
    report = validate_prepared_data(_make_prepared_frame(), cv_splits=3)
    output_json = tmp_path / "cv_results.json"
    output_markdown = tmp_path / "validation_report.md"

    save_validation_results(report, output_json, output_markdown)

    assert output_json.exists()
    assert output_markdown.exists()
    loaded = json.loads(output_json.read_text())
    assert loaded["n_rows"] == 40

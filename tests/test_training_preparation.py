"""Tests for ML training-data preparation."""

from pathlib import Path

import numpy as np
import pandas as pd

from obpi.ml.validation import (
    METRIC_COLUMNS,
    prepare_training_frame,
    save_training_preparation,
)


def _make_scored_frame(n_rows: int = 40) -> pd.DataFrame:
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
        for metric_idx in range(1, 10):
            offset = (metric_idx - 5) * 0.02
            row[f"M{metric_idx}"] = float(np.clip(value + offset, 0.0, 1.0))
        rows.append(row)
    return pd.DataFrame(rows)


def test_prepare_training_frame_adds_scaled_columns_and_labels() -> None:
    result = prepare_training_frame(_make_scored_frame(), cv_splits=4)

    prepared = result.prepared_df
    assert len(prepared) == 20
    assert set(prepared["label"].unique()) == {0, 1}
    for column in METRIC_COLUMNS:
        assert f"{column}_scaled" in prepared.columns
    assert result.metadata["n_splits"] == 4
    assert result.metadata["class_counts"] == {"0": 10, "1": 10}


def test_save_training_preparation_writes_outputs(tmp_path: Path) -> None:
    result = prepare_training_frame(_make_scored_frame(), cv_splits=3)
    output_path = tmp_path / "prepared.parquet"
    metadata_path = tmp_path / "metadata.json"

    save_training_preparation(result, output_path, metadata_path)

    assert output_path.exists()
    assert metadata_path.exists()

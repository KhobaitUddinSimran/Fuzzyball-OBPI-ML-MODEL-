"""Tests for real-data fuzzy normalization and scoring."""

from pathlib import Path

import pandas as pd

from obpi.fuzzy.processing import (
    NORMALIZED_METRICS,
    normalize_processed_metrics,
    run_real_data_fuzzy_processing,
    save_fuzzy_outputs,
)


def _sample_metrics_df() -> pd.DataFrame:
    rows = []
    for idx, base in enumerate([0.0, 0.5, 1.0], start=1):
        rows.append(
            {
                "player_id": idx,
                "player_name": f"Player {idx}",
                "team_id": 10,
                "team_name": "Team A",
                "match_id": idx,
                "minutes": 90.0,
                "starting_position_name": "Center Attacking Midfield",
                "M1_SC": base,
                "M2_OIRC": 5.0 * base,
                "M3_BRPC": base,
                "M4_OBR90": 10.0 * base,
                "M5_RBTL": base,
                "M6_RUP": base,
                "M7_SCI": 3.0 * base,
                "M8_LPC": base,
                "M9_CBI": base,
            }
        )
    return pd.DataFrame(rows)


def test_normalize_processed_metrics_creates_m1_to_m9() -> None:
    normalized_df, summary = normalize_processed_metrics(_sample_metrics_df())
    assert set(NORMALIZED_METRICS).issubset(normalized_df.columns)
    assert normalized_df["M1"].between(0.0, 1.0).all()
    assert summary["M4"]["source_metric"] == "M4_OBR90"


def test_run_real_data_fuzzy_processing_scores_rows() -> None:
    scored_df, metadata = run_real_data_fuzzy_processing(_sample_metrics_df())
    assert "obpi" in scored_df.columns
    assert scored_df["obpi"].between(0.0, 1.0).all()
    assert set(NORMALIZED_METRICS).issubset(scored_df.columns)
    assert "memberships" in metadata
    assert "normalization" in metadata


def test_save_fuzzy_outputs_writes_files(tmp_path: Path) -> None:
    scored_df, metadata = run_real_data_fuzzy_processing(_sample_metrics_df())
    output_path = tmp_path / "scores.parquet"
    metadata_path = tmp_path / "metadata.json"
    save_fuzzy_outputs(scored_df, metadata, output_path, metadata_path)
    assert output_path.exists()
    assert metadata_path.exists()

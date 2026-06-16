import json

import numpy as np
import pandas as pd

from obpi.pipeline import build_parser, main, read_metrics_table, run_scoring_pipeline


def test_run_scoring_pipeline_writes_scored_csv(tmp_path) -> None:
    input_path = tmp_path / "metrics.csv"
    output_path = tmp_path / "scored.csv"
    membership_report_path = tmp_path / "memberships.json"
    metrics_df = pd.DataFrame(
        [{f"M{i}": value for i in range(1, 10)} for value in np.linspace(0, 1, 12)]
    )
    metrics_df.insert(0, "player_id", range(len(metrics_df)))
    metrics_df.to_csv(input_path, index=False)

    summary = run_scoring_pipeline(
        input_path,
        output_path,
        membership_report_path=membership_report_path,
    )
    scored_df = read_metrics_table(output_path)
    membership_report = json.loads(membership_report_path.read_text())

    assert summary["rows_scored"] == 12
    assert summary["score_column"] == "obpi"
    assert summary["membership_report_path"] == str(membership_report_path)
    assert "obpi" in scored_df.columns
    assert scored_df["obpi"].between(0.0, 1.0).all()
    assert "M1" in membership_report
    assert "low_points" in membership_report["M1"]


def test_cli_scores_csv_and_prints_summary(tmp_path, capsys) -> None:
    input_path = tmp_path / "metrics.csv"
    output_path = tmp_path / "scored.csv"
    pd.DataFrame(
        [{f"M{i}": value for i in range(1, 10)} for value in np.linspace(0, 1, 10)]
    ).to_csv(input_path, index=False)

    exit_code = main([str(input_path), str(output_path)])
    summary = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert output_path.exists()
    assert summary["output_path"] == str(output_path)
    assert summary["rows_scored"] == 10


def test_cli_parser_defaults_to_obpi_score() -> None:
    args = build_parser().parse_args(["metrics.csv", "scored.csv"])

    assert args.input.name == "metrics.csv"
    assert args.output.name == "scored.csv"
    assert args.score_column == "obpi"
    assert args.membership_report is None

import numpy as np
import pandas as pd
import pytest

from obpi.fuzzy import (
    FuzzyEngine,
    build_membership_functions,
    fit_fuzzy_engine,
    score_metrics_dataframe,
    summarize_metric_memberships,
)


def test_fuzzy_engine_outputs_corrected_range() -> None:
    engine = FuzzyEngine()
    score = engine.compute({f"M{i}": 0.5 for i in range(1, 10)})
    assert 0.0 <= score <= 1.0
    assert score == pytest.approx(0.5)


def test_fuzzy_engine_edge_cases() -> None:
    engine = FuzzyEngine()
    assert engine.compute({f"M{i}": 0.0 for i in range(1, 10)}) == 0.0
    assert engine.compute({f"M{i}": 1.0 for i in range(1, 10)}) == 1.0


def test_membership_functions_are_percentile_calibrated() -> None:
    values = np.linspace(0.0, 1.0, 101)
    membership = build_membership_functions(values)

    assert membership.percentiles["p20"] == pytest.approx(0.2)
    assert membership.percentiles["p80"] == pytest.approx(0.8)
    assert membership.low_points == pytest.approx([0.0, 0.0, 0.2, 0.5])
    assert membership.high_points == pytest.approx([0.5, 0.8, 1.0, 1.0])
    assert membership.low(0.0) == pytest.approx(1.0)
    assert membership.medium(0.5) == pytest.approx(1.0)
    assert membership.high(1.0) == pytest.approx(1.0)


def test_fuzzy_engine_accepts_metric_weights() -> None:
    weights = {f"M{i}": 1.0 for i in range(1, 10)}
    weights["M1"] = 10.0
    engine = FuzzyEngine(metric_weights=weights)
    score = engine.compute({"M1": 1.0, **{f"M{i}": 0.0 for i in range(2, 10)}})
    assert score > 0.5


def test_score_metrics_dataframe_adds_obpi_column() -> None:
    df = pd.DataFrame(
        [{f"M{i}": value for i in range(1, 10)} for value in np.linspace(0, 1, 21)]
    )
    engine = fit_fuzzy_engine(df)

    scored = score_metrics_dataframe(df, engine=engine)

    assert "obpi" in scored.columns
    assert scored["obpi"].between(0.0, 1.0).all()
    assert scored["obpi"].is_monotonic_increasing


def test_score_metrics_dataframe_rejects_missing_metrics() -> None:
    df = pd.DataFrame({"M1": [0.5]})

    with pytest.raises(ValueError, match="missing metric columns"):
        score_metrics_dataframe(df)


def test_membership_summary_is_json_serializable() -> None:
    df = pd.DataFrame(
        [{f"M{i}": value for i in range(1, 10)} for value in np.linspace(0, 1, 11)]
    )
    engine = fit_fuzzy_engine(df)
    summary = summarize_metric_memberships(engine.membership_functions)

    assert set(summary) == {f"M{i}" for i in range(1, 10)}
    assert summary["M1"]["percentiles"]["p50"] == pytest.approx(0.5)
    assert summary["M1"]["medium_points"] == pytest.approx([0.2, 0.4, 0.6, 0.8])

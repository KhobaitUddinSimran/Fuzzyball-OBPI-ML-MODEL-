"""SHAP explainability service — wraps run_explainability for API consumption."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

from obpi.ml.explainability import run_explainability

logger = logging.getLogger("obpi.api.shap")


@dataclass
class MatchExplainabilityResult:
    """Explainability result scoped to a single match."""

    metric_weights: dict[str, float]
    shap_values: pd.DataFrame | None
    model_name: str


def _prepare_match_dataset(
    metrics_df: pd.DataFrame,
    fuzzy_df: pd.DataFrame,
) -> pd.DataFrame:
    """Build a binary-classification-ready DataFrame from match metrics + fuzzy scores.

    Labels are derived from OBPI quartiles:
    - top 25% → class 1 (high off-ball impact)
    - bottom 25% → class 0 (low off-ball impact)
    - middle 50% → discarded
    """
    fuzzy_metric_cols = [
        "M1_SC",
        "M2_OIRC",
        "M3_BRPC",
        "M4_OBR90",
        "M5_RBTL",
        "M6_RUP",
        "M7_SCI",
        "M8_LPC",
        "M9_CBI",
    ]

    merged = fuzzy_df[["player_id", "obpi"]].merge(
        metrics_df[["player_id"] + fuzzy_metric_cols],
        on="player_id",
    )
    if merged.empty:
        return merged

    obpi = merged["obpi"].to_numpy(dtype=float)
    q75 = float(pd.Series(obpi).quantile(0.75))
    q25 = float(pd.Series(obpi).quantile(0.25))

    merged["label"] = merged["obpi"].apply(
        lambda score: 1 if score >= q75 else (0 if score <= q25 else -1)
    )
    prepared = merged[merged["label"] != -1].reset_index(drop=True)
    return prepared


def run_match_explainability(
    metrics_df: pd.DataFrame,
    fuzzy_df: pd.DataFrame,
    prefer_model: str = "svm",
    include_xgboost: bool = False,
) -> MatchExplainabilityResult:
    """Run explainability on a single match's prepared dataset.

    Falls back to uniform weights and no SHAP if the pipeline fails.
    """
    prepared = _prepare_match_dataset(metrics_df, fuzzy_df)

    if prepared.empty or len(prepared) < 4:
        logger.warning(
            "Too few extreme-quartile samples (%d) for explainability; using uniform weights",
            len(prepared),
        )
        n = 9
        uniform = 1.0 / n
        return MatchExplainabilityResult(
            metric_weights={f"M{i}": uniform for i in range(1, 10)},
            shap_values=None,
            model_name="uniform",
        )

    try:
        result = run_explainability(
            prepared,
            metric_columns=[
                "M1_SC",
                "M2_OIRC",
                "M3_BRPC",
                "M4_OBR90",
                "M5_RBTL",
                "M6_RUP",
                "M7_SCI",
                "M8_LPC",
                "M9_CBI",
            ],
            prefer_model=prefer_model,
            include_xgboost=include_xgboost,
            permutation_repeats=10,
        )
    except Exception as exc:
        logger.warning("Explainability pipeline failed: %s", exc)
        n = 9
        uniform = 1.0 / n
        return MatchExplainabilityResult(
            metric_weights={f"M{i}": uniform for i in range(1, 10)},
            shap_values=None,
            model_name="uniform",
        )

    return MatchExplainabilityResult(
        metric_weights=result.metric_weights,
        shap_values=result.shap_values,
        model_name=result.model_name,
    )

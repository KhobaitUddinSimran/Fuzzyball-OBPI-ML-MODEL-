"""Pipeline orchestration service — wraps compute_all_metrics + run_fuzzy_pipeline."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from api.models import (
    DataQuality,
    DimensionScores,
    ExplainabilityPayload,
    MetricBreakdown,
    MetricStatus,
    MetricValues,
    MetricWeights,
    PlayerProfile,
    PlayerSummary,
    ShapBreakdown,
)
from api.services.cache_service import get_cached, set_cached
from obpi.pipeline import compute_all_metrics, run_fuzzy_pipeline

logger = logging.getLogger("obpi.api.pipeline")

# In-memory store for explainability artifacts keyed by match_id
_match_explainability: dict[int, dict[str, Any]] = {}

_ARCHETYPE_THRESHOLDS = [
    (0.90, "high_all_rounder"),
    (0.75, "elite_creator"),
    (0.60, "tempo_controller"),
    (0.50, "balanced"),
    (0.40, "safe_receiver"),
    (0.30, "runner"),
    (0.20, "raw_runner"),
    (0.00, "low_impact"),
]

_METRIC_NAMES = [
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

_NORMALIZED_TO_RAW = {
    "M1": "M1_SC",
    "M2": "M2_OIRC",
    "M3": "M3_BRPC",
    "M4": "M4_OBR90",
    "M5": "M5_RBTL",
    "M6": "M6_RUP",
    "M7": "M7_SCI",
    "M8": "M8_LPC",
    "M9": "M9_CBI",
}


def _assign_archetype(obpi_score: float) -> str:
    """Assign a tactical archetype label from an OBPI score."""
    for threshold, label in _ARCHETYPE_THRESHOLDS:
        if obpi_score >= threshold:
            return label
    return "low_impact"


def _compute_dimensions(metrics_row: pd.Series) -> DimensionScores:
    """Aggregate M1–M9 into 4 dimension scores."""
    return DimensionScores(
        spatial=(float(metrics_row["M1_SC"]) + float(metrics_row["M7_SCI"])) / 2.0,
        movement=(float(metrics_row["M2_OIRC"]) + float(metrics_row["M4_OBR90"])) / 2.0,
        receiving=(
            float(metrics_row["M3_BRPC"])
            + float(metrics_row["M5_RBTL"])
            + float(metrics_row["M6_RUP"])
        )
        / 3.0,
        temporal=(float(metrics_row["M8_LPC"]) + float(metrics_row["M9_CBI"])) / 2.0,
    )


def _build_metric_breakdown(row: pd.Series) -> MetricBreakdown:
    """Convert a DataFrame row into a MetricBreakdown model."""
    return MetricBreakdown(
        M1_SC=float(row["M1_SC"]),
        M2_OIRC=float(row["M2_OIRC"]),
        M3_BRPC=float(row["M3_BRPC"]),
        M4_OBR90=float(row["M4_OBR90"]),
        M5_RBTL=float(row["M5_RBTL"]),
        M6_RUP=float(row["M6_RUP"]),
        M7_SCI=float(row["M7_SCI"]),
        M8_LPC=float(row["M8_LPC"]),
        M9_CBI=float(row["M9_CBI"]),
    )


def _clean_number(value: Any) -> float | None:
    """Return finite floats and convert missing/NaN values to None."""
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    return float(value)


def _build_metric_values(values: dict[str, Any]) -> MetricValues:
    """Build nullable metric values from a raw dict."""
    return MetricValues(**{metric: _clean_number(values.get(metric)) for metric in _METRIC_NAMES})


def _build_legacy_metric_breakdown(values: dict[str, float | None]) -> MetricBreakdown:
    """Build the legacy non-null metrics object from staged metric values."""
    return MetricBreakdown(
        **{metric: float(values.get(metric) or 0.0) for metric in _METRIC_NAMES}
    )


def _metric_average(values: dict[str, float | None], metric_names: list[str]) -> float:
    """Average available values; return 0.0 when all values are unavailable."""
    available = [values[name] for name in metric_names if values.get(name) is not None]
    if not available:
        return 0.0
    return float(sum(available) / len(available))


def _compute_dimensions_from_values(values: dict[str, float | None]) -> DimensionScores:
    """Aggregate normalized/fuzzy M1-M9 into 4 dimension scores."""
    return DimensionScores(
        spatial=_metric_average(values, ["M1_SC", "M7_SCI"]),
        movement=_metric_average(values, ["M2_OIRC", "M4_OBR90"]),
        receiving=_metric_average(values, ["M3_BRPC", "M5_RBTL", "M6_RUP"]),
        temporal=_metric_average(values, ["M8_LPC", "M9_CBI"]),
    )


def _build_metric_statuses(row: pd.Series) -> dict[str, MetricStatus]:
    """Build per-metric status metadata."""
    statuses: dict[str, MetricStatus] = {}
    for metric in _METRIC_NAMES:
        reason = row.get(f"{metric}_reason")
        if not isinstance(reason, str):
            reason = None if _clean_number(reason) is None else str(reason)
        if reason == "":
            reason = None
        statuses[metric] = MetricStatus(
            status=str(row.get(f"{metric}_status") or "available"),
            reason=reason or None,
        )
    return statuses


def _build_data_quality(row: pd.Series) -> DataQuality:
    """Build response-level data quality metadata from a metrics row."""
    unavailable = [
        metric
        for metric in _METRIC_NAMES
        if row.get(f"{metric}_status") == "unavailable"
    ]
    warnings = row.get("data_quality_warnings")
    if not isinstance(warnings, list):
        warnings = []
    return DataQuality(
        has_360=bool(row.get("has_360", False)),
        events_loaded=bool(row.get("events_loaded", False)),
        frames_loaded=bool(row.get("frames_loaded", False)),
        joined_360_frames=int(row.get("joined_360_frames") or 0),
        high_quality_frames=int(row.get("high_quality_frames") or 0),
        minutes_available=bool(row.get("minutes_available", False)),
        unavailable_metrics=unavailable,
        warnings=warnings,
    )


def _build_normalized_values(
    raw_row: pd.Series,
    fuzzy_row: pd.Series,
) -> dict[str, float | None]:
    """Return normalized metrics, preserving null for unavailable raw metrics."""
    values: dict[str, float | None] = {}
    for normalized_name, raw_name in _NORMALIZED_TO_RAW.items():
        if raw_row.get(f"{raw_name}_status") == "unavailable":
            values[raw_name] = None
        else:
            values[raw_name] = _clean_number(fuzzy_row.get(normalized_name))
    return values


def _build_fuzzy_values(fuzzy_row: pd.Series) -> dict[str, float | None]:
    """Return per-metric fuzzy-adjusted values used for final OBPI scoring."""
    values: dict[str, float | None] = {}
    for normalized_name, raw_name in _NORMALIZED_TO_RAW.items():
        values[raw_name] = _clean_number(fuzzy_row.get(f"{normalized_name}_fuzzy"))
    return values


def _build_shap_breakdown(shap_row: dict[str, float] | None) -> ShapBreakdown:
    """Convert a SHAP dict into a ShapBreakdown model, defaulting to zeros."""
    defaults = {f"M{i}": 0.0 for i in range(1, 10)}
    if shap_row:
        # Map metric columns like M1_SC → M1
        for i in range(1, 10):
            col = f"M{i}"
            if col in shap_row:
                defaults[col] = float(shap_row[col])
            else:
                # Try fuzzy match, e.g. M1_SC or M1_SC_normalized
                for key in shap_row:
                    if key.startswith(col):
                        defaults[col] = float(shap_row[key])
                        break
    return ShapBreakdown(**defaults)


def _build_metric_weights(weights: dict[str, float] | None) -> MetricWeights:
    """Convert raw weights dict into MetricWeights model, falling back to defaults."""
    if not weights:
        return MetricWeights()
    mapped: dict[str, float] = {}
    for i in range(1, 10):
        col = f"M{i}"
        val = weights.get(col)
        if val is None:
            # fuzzy match on keys starting with M{i}
            for k, v in weights.items():
                if k.startswith(col):
                    val = float(v)
                    break
        if val is None:
            val = 0.0
        mapped[f"M{i}_{_METRIC_SUFFIXES[i]}"] = val  # type: ignore[literal-required]
    return MetricWeights(**mapped)


# Suffix mapping for MetricWeights fields
_METRIC_SUFFIXES: dict[int, str] = {
    1: "SC",
    2: "OIRC",
    3: "BRPC",
    4: "OBR90",
    5: "RBTL",
    6: "RUP",
    7: "SCI",
    8: "LPC",
    9: "CBI",
}


def _resolve_player_name(events_df: pd.DataFrame, player_id: int) -> str:
    """Extract player name from event DataFrame by player_id."""
    if events_df.empty or "player" not in events_df.columns:
        return f"Player {player_id}"
    mask = events_df["player"].apply(
        lambda p, pid=player_id: (
            p.get("id") == pid if isinstance(p, dict) else False
        )
    )
    matches = events_df[mask]
    if matches.empty:
        return f"Player {player_id}"
    first = matches.iloc[0]["player"]
    if isinstance(first, dict):
        return first.get("name", f"Player {player_id}")
    return f"Player {player_id}"


def _resolve_minutes(events_df: pd.DataFrame, player_id: int) -> float:
    """Estimate minutes played from event timestamps."""
    if events_df.empty or "minute" not in events_df.columns:
        return 0.0
    mask = events_df["player"].apply(
        lambda p, pid=player_id: (
            p.get("id") == pid if isinstance(p, dict) else False
        )
    )
    minutes = events_df.loc[mask, "minute"]
    if minutes.empty:
        return 0.0
    return float(minutes.max())


def _load_or_compute_metrics(match_id: int, tier: str = "open") -> pd.DataFrame:
    """Load metrics from cache or run the pipeline."""
    cache_key = f"metrics_df:v3:{match_id}:{tier}"
    cached = get_cached(cache_key)
    if cached is not None:
        logger.info("Cache hit for match %s metrics", match_id)
        return pd.DataFrame(cached)

    logger.info("Computing metrics for match %s (tier=%s)", match_id, tier)
    try:
        df = compute_all_metrics(match_id, tier=tier)
    except Exception as exc:
        logger.error("Pipeline error for match %s: %s", match_id, exc)
        raise PipelineUnavailableError(f"Data source unavailable for match {match_id}") from exc

    set_cached(cache_key, df.to_dict(orient="records"))
    return df


def _load_or_compute_fuzzy(
    match_id: int, metrics_df: pd.DataFrame, tier: str = "open"
) -> pd.DataFrame:
    """Load fuzzy scores from cache or run the fuzzy pipeline."""
    cache_key = f"fuzzy_df:v3:{match_id}:{tier}"
    cached = get_cached(cache_key)
    if cached is not None:
        logger.info("Cache hit for match %s fuzzy scores", match_id)
        return pd.DataFrame(cached)

    logger.info("Computing fuzzy scores for match %s", match_id)
    # run_fuzzy_pipeline writes to parquet; we read it back
    fuzzy_df = run_fuzzy_pipeline(metrics_df)
    set_cached(cache_key, fuzzy_df.to_dict(orient="records"))
    return fuzzy_df


class PipelineUnavailableError(Exception):
    """Raised when the StatsBomb / pipeline data is unavailable."""

    pass


def get_all_player_summaries(match_id: int, tier: str = "open") -> list[PlayerSummary]:
    """Return a list of PlayerSummary for every player in the match."""
    metrics_df = _load_or_compute_metrics(match_id, tier)
    fuzzy_df = _load_or_compute_fuzzy(match_id, metrics_df, tier)

    if fuzzy_df.empty:
        return []

    # Resolve player names and minutes once
    from obpi.data.loader import StatsBombLoader
    loader = StatsBombLoader(tier=tier)
    events = loader.get_events(match_id)

    scores = fuzzy_df["obpi"].to_numpy(dtype=float)
    percentiles = pd.Series(scores).rank(pct=True).to_numpy() * 100.0

    summaries: list[PlayerSummary] = []
    for idx, row in fuzzy_df.iterrows():
        pid = int(row["player_id"])
        obpi = float(row["obpi"])
        metrics_row = metrics_df[metrics_df["player_id"] == pid]
        if metrics_row.empty:
            continue
        metrics_row = metrics_row.iloc[0]

        summaries.append(
            PlayerSummary(
                player_id=pid,
                player_name=_resolve_player_name(events, pid),
                match_id=match_id,
                minutes=_resolve_minutes(events, pid),
                obpi_score=obpi,
                percentile=round(float(percentiles[idx]), 2),
                archetype=_assign_archetype(obpi),
                dimensions=_compute_dimensions_from_values(_build_fuzzy_values(row)),
            )
        )

    # Sort by OBPI descending
    summaries.sort(key=lambda p: p.obpi_score, reverse=True)
    return summaries


def get_player_profile(
    match_id: int,
    player_id: int,
    tier: str = "open",
) -> PlayerProfile:
    """Run the full pipeline and return a single PlayerProfile."""
    metrics_df = _load_or_compute_metrics(match_id, tier)
    fuzzy_df = _load_or_compute_fuzzy(match_id, metrics_df, tier)

    player_fuzzy = fuzzy_df[fuzzy_df["player_id"] == player_id]
    if player_fuzzy.empty:
        raise ValueError(f"player_id {player_id} not found in match {match_id}")

    player_metrics = metrics_df[metrics_df["player_id"] == player_id]
    if player_metrics.empty:
        raise ValueError(f"player_id {player_id} not found in match {match_id}")

    from obpi.data.loader import StatsBombLoader
    loader = StatsBombLoader(tier=tier)
    events = loader.get_events(match_id)

    obpi = float(player_fuzzy.iloc[0]["obpi"])
    scores = fuzzy_df["obpi"].to_numpy(dtype=float)
    percentile = round(
        float(pd.Series(scores).rank(pct=True).loc[player_fuzzy.index[0]] * 100.0), 2
    )

    metrics_row = player_metrics.iloc[0]
    fuzzy_row = player_fuzzy.iloc[0]
    raw_metric_values = {
        metric: _clean_number(metrics_row.get(metric)) for metric in _METRIC_NAMES
    }
    normalized_metric_values = _build_normalized_values(metrics_row, fuzzy_row)
    fuzzy_metric_values = _build_fuzzy_values(fuzzy_row)

    # Try to get SHAP values
    shap_row = _get_cached_shap(match_id, player_id)
    if shap_row is None:
        shap_row = _compute_and_cache_shap(match_id, metrics_df, fuzzy_df)
        shap_row = shap_row.get(player_id, {})

    # Try to get metric weights
    weights = _get_cached_weights(match_id)
    if weights is None:
        weights = _compute_and_cache_weights(match_id, metrics_df, fuzzy_df)
    shap_values = shap_row if isinstance(shap_row, dict) else {}
    explainability_model = _get_cached_explainability_model(match_id)

    return PlayerProfile(
        player_id=player_id,
        player_name=_resolve_player_name(events, player_id),
        match_id=match_id,
        minutes=_resolve_minutes(events, player_id),
        obpi_score=obpi,
        percentile=percentile,
        archetype=_assign_archetype(obpi),
        dimensions=_compute_dimensions_from_values(fuzzy_metric_values),
        raw_metrics=_build_metric_values(raw_metric_values),
        normalized_metrics=_build_metric_values(normalized_metric_values),
        fuzzy_metrics=_build_metric_values(fuzzy_metric_values),
        metric_status=_build_metric_statuses(metrics_row),
        data_quality=_build_data_quality(metrics_row),
        explainability=ExplainabilityPayload(
            model=explainability_model,
            metric_weights=weights or {},
            shap_values=shap_values,
        ),
        metrics=_build_legacy_metric_breakdown(fuzzy_metric_values),
        shap=_build_shap_breakdown(shap_row if isinstance(shap_row, dict) else None),
        metric_weights=_build_metric_weights(weights),
    )


def _compute_and_cache_shap(
    match_id: int, metrics_df: pd.DataFrame, fuzzy_df: pd.DataFrame
) -> dict[int, dict[str, float]]:
    """Run explainability and cache per-player SHAP rows."""
    try:
        from api.services.shap_service import run_match_explainability
        result = run_match_explainability(metrics_df, fuzzy_df)
    except Exception as exc:
        logger.warning("SHAP computation failed for match %s: %s", match_id, exc)
        _match_explainability[match_id] = {"shap": {}, "weights": {}, "model": "unavailable"}
        return {}

    shap_map: dict[int, dict[str, float]] = {}
    if result.shap_values is not None and not result.shap_values.empty:
        for pid, row in result.shap_values.iterrows():
            shap_map[int(pid)] = row.to_dict()

    _match_explainability[match_id] = {
        "shap": shap_map,
        "weights": result.metric_weights,
        "model": result.model_name,
    }
    return shap_map


def _get_cached_shap(match_id: int, player_id: int) -> dict[str, float] | None:
    """Retrieve cached SHAP row for a player."""
    data = _match_explainability.get(match_id)
    if data is None:
        return None
    return data.get("shap", {}).get(player_id)


def _compute_and_cache_weights(
    match_id: int, metrics_df: pd.DataFrame, fuzzy_df: pd.DataFrame
) -> dict[str, float]:
    """Run explainability and cache metric weights."""
    try:
        from api.services.shap_service import run_match_explainability
        result = run_match_explainability(metrics_df, fuzzy_df)
    except Exception as exc:
        logger.warning("Weights computation failed for match %s: %s", match_id, exc)
        _match_explainability[match_id] = {"shap": {}, "weights": {}, "model": "unavailable"}
        return {}
    _match_explainability[match_id] = {
        "shap": result.shap_values.to_dict() if result.shap_values is not None else {},
        "weights": result.metric_weights,
        "model": result.model_name,
    }
    return result.metric_weights


def _get_cached_weights(match_id: int) -> dict[str, float] | None:
    """Retrieve cached metric weights for a match."""
    data = _match_explainability.get(match_id)
    if data is None:
        return None
    return data.get("weights")


def _get_cached_explainability_model(match_id: int) -> str:
    """Retrieve the cached explainability model name."""
    data = _match_explainability.get(match_id)
    if data is None:
        return "unavailable"
    return str(data.get("model") or "unavailable")


def generate_insight(a: PlayerProfile, b: PlayerProfile) -> str:
    """Generate a natural-language comparison insight between two players."""
    deltas = {
        "spatial": round(a.dimensions.spatial - b.dimensions.spatial, 2),
        "movement": round(a.dimensions.movement - b.dimensions.movement, 2),
        "receiving": round(a.dimensions.receiving - b.dimensions.receiving, 2),
        "temporal": round(a.dimensions.temporal - b.dimensions.temporal, 2),
    }

    a_leads = [dim for dim, val in deltas.items() if val > 0.05]
    b_leads = [dim for dim, val in deltas.items() if val < -0.05]

    parts: list[str] = []
    if a_leads:
        parts.append(
            f"{a.player_name} outperforms {b.player_name} on "
            + ", ".join(f"{d.title()} (+{deltas[d]:.2f})" for d in a_leads)
            + "."
        )
    if b_leads:
        parts.append(
            f"{b.player_name} leads on "
            + ", ".join(f"{d.title()} ({-deltas[d]:.2f})" for d in b_leads)
            + "."
        )
    if not parts:
        parts.append(
            f"{a.player_name} and {b.player_name} are closely matched across all dimensions."
        )

    return " ".join(parts)

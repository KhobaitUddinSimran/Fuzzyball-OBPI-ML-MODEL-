"""Pydantic request/response models for the OBPI API."""

from __future__ import annotations

from pydantic import BaseModel, Field

# ─── Shared ──────────────────────────────────────────────────────────────


class MetricBreakdown(BaseModel):
    """M1–M9 metric values for a single player."""

    M1_SC: float = Field(..., ge=0.0, description="Screening Coefficient")
    M2_OIRC: float = Field(..., ge=0.0, description="Off-Ball Impact Run Coefficient")
    M3_BRPC: float = Field(..., ge=0.0, description="Best Receiving Position Coefficient")
    M4_OBR90: float = Field(..., ge=0.0, description="Off-Ball Runs per 90")
    M5_RBTL: float = Field(..., ge=0.0, description="Receipts Between the Lines")
    M6_RUP: float = Field(..., ge=0.0, description="Receipts Under Pressure")
    M7_SCI: float = Field(..., ge=0.0, description="Space Creation Index")
    M8_LPC: float = Field(..., ge=0.0, description="La Pausa Coefficient")
    M9_CBI: float = Field(..., ge=0.0, description="Call-for-Ball Index")


class MetricValues(BaseModel):
    """Nullable M1-M9 values for one metric stage."""

    M1_SC: float | None = None
    M2_OIRC: float | None = None
    M3_BRPC: float | None = None
    M4_OBR90: float | None = None
    M5_RBTL: float | None = None
    M6_RUP: float | None = None
    M7_SCI: float | None = None
    M8_LPC: float | None = None
    M9_CBI: float | None = None


class MetricStatus(BaseModel):
    """Availability status and reason for one metric."""

    status: str
    reason: str | None = None


class ShapBreakdown(BaseModel):
    """SHAP value breakdown per metric."""

    M1: float
    M2: float
    M3: float
    M4: float
    M5: float
    M6: float
    M7: float
    M8: float
    M9: float


class MetricWeights(BaseModel):
    """Normalized metric importance weights (sum to 1.0)."""

    M9_CBI: float = 0.489
    M7_SCI: float = 0.231
    M6_RUP: float = 0.150
    M2_OIRC: float = 0.097
    M5_RBTL: float = 0.030
    M4_OBR90: float = 0.004
    M3_BRPC: float = 0.000
    M1_SC: float = 0.000
    M8_LPC: float = 0.000


class DimensionScores(BaseModel):
    """Aggregated 4-dimension scores."""

    spatial: float = Field(..., description="Average of M1_SC and M7_SCI")
    movement: float = Field(..., description="Average of M2_OIRC and M4_OBR90")
    receiving: float = Field(..., description="Average of M3_BRPC, M5_RBTL, M6_RUP")
    temporal: float = Field(..., description="Average of M8_LPC and M9_CBI")


class DataQuality(BaseModel):
    """Data availability summary for a player analysis response."""

    has_360: bool = False
    events_loaded: bool = False
    frames_loaded: bool = False
    joined_360_frames: int = 0
    high_quality_frames: int = 0
    minutes_available: bool = False
    unavailable_metrics: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class ExplainabilityPayload(BaseModel):
    """Optional model explainability output kept separate from OBPI scoring."""

    model: str = "unavailable"
    metric_weights: dict[str, float] = Field(default_factory=dict)
    shap_values: dict[str, float] = Field(default_factory=dict)


# ─── Player ────────────────────────────────────────────────────────────────


class PlayerSummary(BaseModel):
    """Compact player summary returned in list views."""

    player_id: int
    player_name: str
    match_id: int
    minutes: float
    obpi_score: float = Field(..., ge=0.0, le=1.0)
    percentile: float = Field(..., ge=0.0, le=100.0)
    archetype: str
    dimensions: DimensionScores


class PlayerProfile(PlayerSummary):
    """Full player profile with metrics, SHAP, and weights."""

    raw_metrics: MetricValues
    normalized_metrics: MetricValues
    fuzzy_metrics: MetricValues
    metric_status: dict[str, MetricStatus] = Field(default_factory=dict)
    data_quality: DataQuality = Field(default_factory=DataQuality)
    explainability: ExplainabilityPayload = Field(default_factory=ExplainabilityPayload)
    metrics: MetricBreakdown
    shap: ShapBreakdown
    metric_weights: MetricWeights


# ─── Requests ──────────────────────────────────────────────────────────────


class AnalyzeRequest(BaseModel):
    """Request body for POST /analyze."""

    match_id: int = Field(..., example=3794686)
    player_id: int = Field(..., example=1001)
    tier: str = Field(default="open", pattern="^(open|api)$")
    config: str | None = None


class CompareRequest(BaseModel):
    """Request body for POST /compare."""

    match_id: int
    player_ids: list[int] = Field(..., min_length=2, max_length=2)


# ─── Responses ─────────────────────────────────────────────────────────────


class PlayersResponse(BaseModel):
    """Response for GET /players."""

    match_id: int
    count: int
    players: list[PlayerSummary]


class DimensionDelta(BaseModel):
    """Delta scores between two players across dimensions."""

    spatial: float
    movement: float
    receiving: float
    temporal: float


class CompareResponse(BaseModel):
    """Response for POST /compare."""

    players: list[PlayerProfile]
    dimension_deltas: DimensionDelta
    auto_insight: str


class LeaderboardEntry(PlayerSummary):
    """Single leaderboard row with rank."""

    rank: int


class LeaderboardResponse(BaseModel):
    """Response for GET /leaderboard."""

    match_id: int
    count: int
    entries: list[LeaderboardEntry]


class HealthResponse(BaseModel):
    """Response for GET /health."""

    status: str = "ok"
    model_version: str = "1.0.0"
    schema_version: int = 3
    uptime_seconds: float
    cache_connected: bool


# ─── Error ─────────────────────────────────────────────────────────────────


class ErrorDetail(BaseModel):
    """Standard error response body."""

    detail: str

"""POST /compare endpoint — side-by-side player comparison."""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from api.models import CompareRequest, CompareResponse, DimensionDelta
from api.services.cache_service import get_cached, set_cached
from api.services.pipeline_service import (
    PipelineUnavailableError,
    generate_insight,
    get_player_profile,
)

router = APIRouter(tags=["compare"])
logger = logging.getLogger("obpi.api.routes.compare")


@router.post("/compare", response_model=CompareResponse)
def compare_players(request: CompareRequest) -> CompareResponse:
    """Compare two players from the same match."""
    id_a, id_b = request.player_ids
    cache_key = f"compare:{request.match_id}:{id_a}:{id_b}"
    cached = get_cached(cache_key)
    if cached is not None:
        return CompareResponse(**cached)

    try:
        profile_a = get_player_profile(request.match_id, id_a)
        profile_b = get_player_profile(request.match_id, id_b)
    except PipelineUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    delta = DimensionDelta(
        spatial=round(profile_a.dimensions.spatial - profile_b.dimensions.spatial, 2),
        movement=round(profile_a.dimensions.movement - profile_b.dimensions.movement, 2),
        receiving=round(profile_a.dimensions.receiving - profile_b.dimensions.receiving, 2),
        temporal=round(profile_a.dimensions.temporal - profile_b.dimensions.temporal, 2),
    )

    insight = generate_insight(profile_a, profile_b)

    response = CompareResponse(
        players=[profile_a, profile_b],
        dimension_deltas=delta,
        auto_insight=insight,
    )
    set_cached(cache_key, response.model_dump(mode="json"))
    return response

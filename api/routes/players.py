"""Player listing and detail endpoints."""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Query

from api.models import PlayerProfile, PlayersResponse
from api.services.cache_service import get_cached, set_cached
from api.services.pipeline_service import (
    PipelineUnavailableError,
    get_all_player_summaries,
    get_player_profile,
)

router = APIRouter(tags=["players"])
logger = logging.getLogger("obpi.api.routes.players")


@router.get("/players", response_model=PlayersResponse)
def list_players(
    match_id: int = Query(..., description="StatsBomb match identifier"),
) -> PlayersResponse:
    """Return all players with OBPI summaries for a given match."""
    cache_key = f"players:{match_id}"
    cached = get_cached(cache_key)
    if cached is not None:
        return PlayersResponse(**cached)

    try:
        summaries = get_all_player_summaries(match_id)
    except PipelineUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    if not summaries:
        raise HTTPException(status_code=404, detail=f"No players found for match_id {match_id}")

    response = PlayersResponse(
        match_id=match_id,
        count=len(summaries),
        players=summaries,
    )
    set_cached(cache_key, response.model_dump(mode="json"))
    return response


@router.get("/players/{player_id}", response_model=PlayerProfile)
def get_player(
    player_id: int,
    match_id: int = Query(..., description="StatsBomb match identifier"),
) -> PlayerProfile:
    """Return full player profile for a given match and player."""
    cache_key = f"player:{match_id}:{player_id}"
    cached = get_cached(cache_key)
    if cached is not None:
        return PlayerProfile(**cached)

    try:
        profile = get_player_profile(match_id, player_id)
    except PipelineUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    set_cached(cache_key, profile.model_dump(mode="json"))
    return profile

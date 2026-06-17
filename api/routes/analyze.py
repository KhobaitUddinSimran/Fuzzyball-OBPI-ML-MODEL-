"""POST /analyze endpoint — run full pipeline for a single player."""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from api.models import AnalyzeRequest, PlayerProfile
from api.services.cache_service import get_cached, set_cached
from api.services.pipeline_service import PipelineUnavailableError, get_player_profile

router = APIRouter(tags=["analyze"])
logger = logging.getLogger("obpi.api.routes.analyze")


@router.post("/analyze", response_model=PlayerProfile)
def analyze_player(request: AnalyzeRequest) -> PlayerProfile:
    """Run the full OBPI pipeline for a match/player and return the profile."""
    cache_key = f"analyze:{request.match_id}:{request.player_id}"
    cached = get_cached(cache_key)
    if cached is not None:
        return PlayerProfile(**cached)

    try:
        profile = get_player_profile(
            match_id=request.match_id,
            player_id=request.player_id,
            tier=request.tier,
        )
    except PipelineUnavailableError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Pipeline error: {exc}",
        ) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception(
            "Unhandled pipeline error for match %s player %s",
            request.match_id,
            request.player_id,
        )
        raise HTTPException(
            status_code=500,
            detail="Internal pipeline error",
        ) from exc

    set_cached(cache_key, profile.model_dump(mode="json"))
    return profile

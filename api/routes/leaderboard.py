"""GET /leaderboard endpoint — ranked player list for a match."""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Query

from api.models import LeaderboardEntry, LeaderboardResponse
from api.services.cache_service import get_cached, set_cached
from api.services.pipeline_service import PipelineUnavailableError, get_all_player_summaries

router = APIRouter(tags=["leaderboard"])
logger = logging.getLogger("obpi.api.routes.leaderboard")


@router.get("/leaderboard", response_model=LeaderboardResponse)
def get_leaderboard(
    match_id: int = Query(..., description="StatsBomb match identifier"),
    limit: int = Query(default=10, ge=1, le=50, description="Max entries to return"),
    archetype: str | None = Query(default=None, description="Filter by archetype label"),
) -> LeaderboardResponse:
    """Return a ranked leaderboard of all players in the match."""
    cache_key = f"leaderboard:{match_id}"
    cached = get_cached(cache_key)
    if cached is not None:
        response = LeaderboardResponse(**cached)
    else:
        try:
            summaries = get_all_player_summaries(match_id)
        except PipelineUnavailableError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

        if not summaries:
            raise HTTPException(
                status_code=404, detail=f"No players found for match_id {match_id}"
            )

        entries = [
            LeaderboardEntry(rank=idx + 1, **summary.model_dump())
            for idx, summary in enumerate(summaries)
        ]
        response = LeaderboardResponse(
            match_id=match_id,
            count=len(entries),
            entries=entries,
        )
        set_cached(cache_key, response.model_dump(mode="json"))

    # Apply filters post-cache
    filtered = response.entries
    if archetype:
        filtered = [e for e in filtered if e.archetype == archetype]

    # Apply limit post-cache
    limited = filtered[:limit]

    # Re-rank after filtering
    re_ranked = [
        LeaderboardEntry(rank=idx + 1, **entry.model_dump(exclude={"rank"}))
        for idx, entry in enumerate(limited)
    ]

    return LeaderboardResponse(
        match_id=match_id,
        count=len(re_ranked),
        entries=re_ranked,
    )

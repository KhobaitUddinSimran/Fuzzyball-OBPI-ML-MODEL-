"""StatsBomb browsing endpoints used by the dashboard."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query

from api.services.statsbomb_service import (
    get_eligible_players,
    get_match_360_frames,
    get_match_details,
    get_match_events,
    get_match_lineups,
    get_match_readiness,
    get_matches_by_year,
    get_statsbomb_360_match_ids,
    get_world_cup_dates,
    get_world_cup_years,
)

router = APIRouter(tags=["statsbomb"])


@router.get("/events/fifa-world-cup/years")
def fifa_world_cup_years() -> list[dict[str, Any]]:
    """Return available FIFA World Cup seasons from StatsBomb."""
    return get_world_cup_years()


@router.get("/events/fifa-world-cup/dates")
def fifa_world_cup_dates(
    year: str = Query(..., description="World Cup year"),
) -> list[dict[str, Any]]:
    """Return match dates for a FIFA World Cup year."""
    return get_world_cup_dates(year)


@router.get("/statsbomb-360/match-ids")
def statsbomb_360_match_ids(
    refresh: bool = Query(default=False),
) -> dict[str, Any]:
    """Return StatsBomb match IDs with 360 freeze-frame files."""
    try:
        return get_statsbomb_360_match_ids(refresh=refresh)
    except RuntimeError as error:
        raise HTTPException(status_code=502, detail=str(error)) from error


@router.get("/matches")
def matches(
    event: str = Query(default="fifa-world-cup"),
    year: str = Query(..., description="World Cup year"),
) -> list[dict[str, Any]]:
    """Return FIFA World Cup matches for a selected year."""
    if event != "fifa-world-cup":
        raise HTTPException(
            status_code=400,
            detail="Only fifa-world-cup is supported for now.",
        )

    try:
        return get_matches_by_year(year)
    except ValueError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except RuntimeError as error:
        raise HTTPException(status_code=502, detail=str(error)) from error


@router.get("/matches/{match_id}")
def match_details(match_id: int) -> dict[str, Any]:
    """Return normalized match details for a StatsBomb match."""
    try:
        match = get_match_details(match_id)
    except RuntimeError as error:
        raise HTTPException(status_code=502, detail=str(error)) from error

    if not match:
        raise HTTPException(status_code=404, detail="Match not found.")
    return match


@router.get("/matches/{match_id}/eligible-players")
def eligible_players(match_id: int) -> list[dict[str, Any]]:
    """Return attacking-midfield players with OBPI data readiness."""
    if not get_match_details(match_id):
        raise HTTPException(status_code=404, detail="Match not found.")
    return get_eligible_players(match_id)


@router.get("/matches/{match_id}/readiness")
def match_readiness(match_id: int) -> dict[str, Any]:
    """Return match-level OBPI readiness counts."""
    readiness = get_match_readiness(match_id)
    if not readiness:
        raise HTTPException(status_code=404, detail="Match not found.")
    return readiness


@router.get("/matches/{match_id}/events")
def match_events(match_id: int) -> list[dict[str, Any]]:
    """Return raw StatsBomb event rows for a match."""
    if not get_match_details(match_id):
        raise HTTPException(status_code=404, detail="Match not found.")
    return get_match_events(match_id)


@router.get("/matches/{match_id}/frames")
def match_360_frames(match_id: int) -> dict[str, Any]:
    """Return grouped StatsBomb 360 freeze frames for pitch preview."""
    if not get_match_details(match_id):
        raise HTTPException(status_code=404, detail="Match not found.")

    try:
        return get_match_360_frames(match_id)
    except RuntimeError as error:
        raise HTTPException(status_code=502, detail=str(error)) from error


@router.get("/matches/{match_id}/lineups")
def match_lineups(match_id: int) -> list[dict[str, Any]]:
    """Return StatsBomb lineups for a match."""
    if not get_match_details(match_id):
        raise HTTPException(status_code=404, detail="Match not found.")
    return get_match_lineups(match_id)

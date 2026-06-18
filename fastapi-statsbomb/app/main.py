from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from .statsbomb_service import (
    get_eligible_players,
    get_match_details,
    get_match_events,
    get_match_lineups,
    get_matches_by_year,
    get_world_cup_dates,
    get_world_cup_years,
)


app = FastAPI(title="Fuzzyball StatsBomb API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok", "source": "statsbombpy"}


@app.get("/events/fifa-world-cup/dates")
def fifa_world_cup_dates(year: str = Query(..., description="World Cup year")):
    return get_world_cup_dates(year)


@app.get("/events/fifa-world-cup/years")
def fifa_world_cup_years():
    return get_world_cup_years()


@app.get("/matches")
def matches(
    event: str = Query(default="fifa-world-cup"),
    year: str = Query(..., description="World Cup year"),
):
    if event != "fifa-world-cup":
        raise HTTPException(status_code=400, detail="Only fifa-world-cup is supported for now.")
    try:
        return get_matches_by_year(year)
    except ValueError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


@app.get("/matches/{match_id}")
def match_details(match_id: int):
    match = get_match_details(match_id)
    if not match:
        raise HTTPException(status_code=404, detail="Match not found.")
    return match


@app.get("/matches/{match_id}/eligible-players")
def eligible_players(match_id: int):
    match = get_match_details(match_id)
    if not match:
        raise HTTPException(status_code=404, detail="Match not found.")
    return get_eligible_players(match_id)


@app.get("/matches/{match_id}/events")
def match_events(match_id: int):
    if not get_match_details(match_id):
        raise HTTPException(status_code=404, detail="Match not found.")
    return get_match_events(match_id)


@app.get("/matches/{match_id}/lineups")
def match_lineups(match_id: int):
    if not get_match_details(match_id):
        raise HTTPException(status_code=404, detail="Match not found.")
    return get_match_lineups(match_id)

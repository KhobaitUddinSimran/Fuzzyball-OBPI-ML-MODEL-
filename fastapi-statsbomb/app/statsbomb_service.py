from functools import lru_cache
from math import isnan
from typing import Any
from datetime import date, time
import re

import pandas as pd
from statsbombpy import sb


WORLD_CUP_COMPETITION_ID = 43

ELIGIBLE_POSITION_KEYWORDS = (
    "Attacking Midfield",
    "Center Attacking Midfield",
    "Left Attacking Midfield",
    "Right Attacking Midfield",
    "Center Midfield",
    "Left Midfield",
    "Right Midfield",
)


@lru_cache(maxsize=1)
def get_world_cup_years() -> list[dict[str, Any]]:
    competitions = dataframe_to_records(sb.competitions())
    world_cup_rows = [
        row
        for row in competitions
        if int(row.get("competition_id") or 0) == WORLD_CUP_COMPETITION_ID
    ]

    years = []
    for row in world_cup_rows:
        year = extract_year(row.get("season_name"))
        if not year:
            continue
        years.append(
            {
                "year": year,
                "season_id": int(row["season_id"]),
                "label": f"{year} FIFA World Cup",
            }
        )

    return sorted(years, key=lambda item: item["year"], reverse=True)


@lru_cache(maxsize=8)
def get_world_cup_matches(year: str) -> list[dict[str, Any]]:
    season_id = season_id_for_year(year)
    matches = sb.matches(
        competition_id=WORLD_CUP_COMPETITION_ID,
        season_id=season_id,
    )
    return dataframe_to_records(matches)


def get_matches_by_year(year: str) -> list[dict[str, Any]]:
    return [normalize_match_summary(match) for match in get_world_cup_matches(year)]


def get_world_cup_dates(year: str) -> list[dict[str, Any]]:
    counts: dict[str, int] = {}

    for match in get_world_cup_matches(year):
        match_date = match.get("match_date")
        if not match_date:
            continue
        counts[match_date] = counts.get(match_date, 0) + 1

    return [
        {
            "date": match_date,
            "label": pd.to_datetime(match_date).strftime("%d %b %Y"),
            "match_count": count,
        }
        for match_date, count in sorted(counts.items())
    ]


def get_match_details(match_id: int) -> dict[str, Any] | None:
    match = find_match(match_id)
    if not match:
        return None

    details = normalize_match_summary(match)
    details["teams"] = {
        "home": {
            "team_id": team_id_for_name(match, details["home_team"]),
            "team_name": details["home_team"],
        },
        "away": {
            "team_id": team_id_for_name(match, details["away_team"]),
            "team_name": details["away_team"],
        },
    }
    details["eligible_player_count"] = None
    details["raw_match"] = match
    return details


def get_match_events(match_id: int) -> list[dict[str, Any]]:
    events = sb.events(match_id=match_id)
    return dataframe_to_records(events)


def get_match_lineups(match_id: int) -> list[dict[str, Any]]:
    lineups = sb.lineups(match_id=match_id)
    if isinstance(lineups, dict):
        return [{"team_name": team_name, "players": dataframe_to_records(players)} for team_name, players in lineups.items()]
    return dataframe_to_records(lineups)


def get_eligible_players(match_id: int) -> list[dict[str, Any]]:
    match = get_match_details_without_players(match_id)
    if not match:
        return []

    players_by_id: dict[int, dict[str, Any]] = {}

    for event in get_match_events(match_id):
        player_id = event.get("player_id")
        player_name = event.get("player")
        position = event.get("position")
        team_name = event.get("team")

        if not player_id or not player_name or not position:
            continue

        if not is_project_position(position):
            continue

        player_id = int(player_id)
        players_by_id[player_id] = {
            "player_id": player_id,
            "player_name": player_name,
            "team_id": team_id_for_name(match, team_name),
            "team_name": team_name,
            "position": position,
            "minutes": None,
        }

    return sorted(players_by_id.values(), key=lambda player: (player["team_name"], player["player_name"]))


def normalize_match_summary(match: dict[str, Any]) -> dict[str, Any]:
    return {
        "match_id": int(match["match_id"]),
        "date": match.get("match_date"),
        "competition": match.get("competition") or "FIFA World Cup",
        "stage": match.get("competition_stage"),
        "home_team": match.get("home_team"),
        "away_team": match.get("away_team"),
        "home_score": match.get("home_score"),
        "away_score": match.get("away_score"),
        "kickoff_time": match.get("kick_off"),
        "stadium": match.get("stadium"),
    }


def find_match(match_id: int) -> dict[str, Any] | None:
    for year in get_world_cup_years():
        for match in get_world_cup_matches(year["year"]):
            if int(match["match_id"]) == int(match_id):
                return match
    return None


def season_id_for_year(year: str) -> int:
    for item in get_world_cup_years():
        if item["year"] == str(year):
            return item["season_id"]
    raise ValueError(f"FIFA World Cup year {year} is not available in StatsBomb.")


def extract_year(value: Any) -> str | None:
    match = re.search(r"\d{4}", str(value or ""))
    return match.group(0) if match else None


def get_match_details_without_players(match_id: int) -> dict[str, Any] | None:
    match = find_match(match_id)
    if not match:
        return None

    return {
        "home_team": match.get("home_team"),
        "away_team": match.get("away_team"),
        "home_team_id": match.get("home_team_id") or 0,
        "away_team_id": match.get("away_team_id") or 0,
    }


def team_id_for_name(match: dict[str, Any], team_name: str | None) -> int:
    if team_name == match.get("home_team"):
        return int(match.get("home_team_id") or stable_team_id(team_name))
    if team_name == match.get("away_team"):
        return int(match.get("away_team_id") or stable_team_id(team_name))
    return stable_team_id(team_name)


def stable_team_id(team_name: str | None) -> int:
    if not team_name:
        return 0
    return sum((index + 1) * ord(char) for index, char in enumerate(team_name))


def is_project_position(position: str) -> bool:
    return any(keyword.lower() in position.lower() for keyword in ELIGIBLE_POSITION_KEYWORDS)


def dataframe_to_records(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, pd.DataFrame):
        return [clean_record(record) for record in data.to_dict(orient="records")]
    if isinstance(data, list):
        return [clean_record(record) for record in data]
    return []


def clean_record(record: dict[str, Any]) -> dict[str, Any]:
    cleaned = {}
    for key, value in record.items():
        cleaned[key] = clean_value(value)
    return cleaned


def clean_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, (date, time)):
        return value.isoformat()
    if isinstance(value, float) and isnan(value):
        return None
    if value is pd.NaT:
        return None
    if isinstance(value, dict):
        return {key: clean_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clean_value(item) for item in value]
    return value

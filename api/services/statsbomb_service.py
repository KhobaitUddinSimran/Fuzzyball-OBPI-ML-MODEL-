import json
import re
from datetime import date, time
from functools import lru_cache
from math import isnan
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd
from statsbombpy import sb


WORLD_CUP_COMPETITION_ID = 43
STATSBOMB_360_CONTENTS_URL = (
    "https://api.github.com/repos/statsbomb/open-data/contents/data/three-sixty"
)
STATSBOMB_360_CACHE_PATH = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "cache"
    / "statsbomb_360_match_ids.json"
)

ELIGIBLE_POSITION_KEYWORDS = (
    "Attacking Midfield",
    "Center Attacking Midfield",
    "Left Attacking Midfield",
    "Right Attacking Midfield",
    "Center Midfield",
    "Left Midfield",
    "Right Midfield",
)

OBPI_METRIC_GROUPS = (
    {
        "key": "M1",
        "label": "Screening coefficient",
        "columns": (
            "id",
            "index",
            "timestamp",
            "player_id",
            "team_id",
            "location",
            "type",
            "related_events",
        ),
    },
    {
        "key": "M2",
        "label": "Off-ball impact run coefficient",
        "columns": (
            "id",
            "index",
            "timestamp",
            "player_id",
            "team_id",
            "location",
            "type",
            "duration",
        ),
    },
    {
        "key": "M3",
        "label": "Best receiving position coefficient",
        "columns": (
            "pass_recipient_id",
            "pass_end_location",
            "ball_receipt_outcome",
            "under_pressure",
            "location",
            "timestamp",
        ),
    },
    {
        "key": "M4",
        "label": "Off-ball runs per 90",
        "columns": (
            "player_id",
            "team_id",
            "timestamp",
            "minute",
            "second",
            "duration",
            "substitution_outcome",
            "substitution_replacement_id",
        ),
    },
    {
        "key": "M5",
        "label": "Receipts between the lines",
        "columns": (
            "type",
            "player_id",
            "position",
            "location",
            "ball_receipt_outcome",
            "pass_recipient_id",
        ),
    },
    {
        "key": "M6",
        "label": "Receipts under pressure",
        "columns": (
            "type",
            "player_id",
            "ball_receipt_outcome",
            "under_pressure",
            "related_events",
        ),
    },
    {
        "key": "M7",
        "label": "Space creation index",
        "columns": (
            "id",
            "timestamp",
            "player_id",
            "team_id",
            "location",
            "type",
            "related_events",
        ),
    },
    {
        "key": "M8",
        "label": "La Pausa coefficient",
        "columns": (
            "type",
            "player_id",
            "timestamp",
            "duration",
            "location",
            "pass_end_location",
            "carry_end_location",
            "under_pressure",
        ),
    },
    {
        "key": "M9",
        "label": "Call-for-ball index",
        "columns": (
            "player_id",
            "team_id",
            "timestamp",
            "location",
            "pass_recipient_id",
            "pass_end_location",
        ),
    },
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
    return enrich_matches_with_360(
        [normalize_match_summary(match) for match in get_world_cup_matches(year)]
    )


def get_statsbomb_360_match_ids(refresh: bool = False) -> dict[str, Any]:
    if not refresh:
        cached_match_ids = read_cached_360_match_ids()
        if cached_match_ids is not None:
            return {
                "match_ids": cached_match_ids,
                "count": len(cached_match_ids),
                "source": "cache",
            }

    try:
        contents = fetch_statsbomb_360_contents()
        match_ids = extract_360_match_ids(contents)
        write_cached_360_match_ids(match_ids)
        return {
            "match_ids": match_ids,
            "count": len(match_ids),
            "source": "github",
        }
    except RuntimeError:
        cached_match_ids = read_cached_360_match_ids()
        if cached_match_ids is not None:
            return {
                "match_ids": cached_match_ids,
                "count": len(cached_match_ids),
                "source": "cache",
                "warning": "GitHub fetch failed, so cached StatsBomb 360 match IDs were used.",
            }
        raise


def fetch_statsbomb_360_contents() -> list[dict[str, Any]]:
    request = Request(
        STATSBOMB_360_CONTENTS_URL,
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": "fuzzyball-statsbomb-dashboard",
        },
    )

    try:
        with urlopen(request, timeout=20) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as error:
        raise RuntimeError("Unable to fetch StatsBomb 360 availability from GitHub.") from error

    if not isinstance(payload, list):
        raise RuntimeError("GitHub returned an unexpected StatsBomb 360 response.")

    return payload


def extract_360_match_ids(contents: list[dict[str, Any]]) -> list[int]:
    match_ids = []

    for item in contents:
        filename = str(item.get("name") or "")
        match = re.fullmatch(r"(\d+)\.json", filename)
        if match:
            match_ids.append(int(match.group(1)))

    return sorted(set(match_ids))


def read_cached_360_match_ids() -> list[int] | None:
    if not STATSBOMB_360_CACHE_PATH.exists():
        return None

    try:
        with STATSBOMB_360_CACHE_PATH.open("r", encoding="utf-8") as cache_file:
            payload = json.load(cache_file)
    except (OSError, json.JSONDecodeError):
        return None

    match_ids = payload.get("match_ids") if isinstance(payload, dict) else payload
    if not isinstance(match_ids, list):
        return None

    clean_ids = []
    for match_id in match_ids:
        try:
            clean_ids.append(int(match_id))
        except (TypeError, ValueError):
            continue

    return sorted(set(clean_ids))


def write_cached_360_match_ids(match_ids: list[int]) -> None:
    STATSBOMB_360_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with STATSBOMB_360_CACHE_PATH.open("w", encoding="utf-8") as cache_file:
        json.dump({"match_ids": sorted(set(match_ids))}, cache_file, indent=2)


def enrich_matches_with_360(matches: list[dict[str, Any]]) -> list[dict[str, Any]]:
    match_ids_360 = set(get_statsbomb_360_match_ids()["match_ids"])

    return [
        {
            **match,
            "has_360": int(match["match_id"]) in match_ids_360,
        }
        for match in matches
    ]


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
    try:
        details["has_360"] = has_match_360(match_id)
    except RuntimeError:
        details["has_360"] = False
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


@lru_cache(maxsize=64)
def get_match_events(match_id: int) -> list[dict[str, Any]]:
    events = sb.events(match_id=match_id)
    return dataframe_to_records(events)


@lru_cache(maxsize=64)
def get_match_lineups(match_id: int) -> list[dict[str, Any]]:
    lineups = sb.lineups(match_id=match_id)
    if isinstance(lineups, dict):
        return [
            {"team_name": team_name, "players": dataframe_to_records(players)}
            for team_name, players in lineups.items()
        ]
    return dataframe_to_records(lineups)


@lru_cache(maxsize=32)
def get_match_360_frames(match_id: int) -> dict[str, Any]:
    if not has_match_360(match_id):
        return {
            "match_id": int(match_id),
            "frame_count": 0,
            "frames": [],
        }

    try:
        rows = dataframe_to_records(sb.frames(match_id=match_id, fmt="dataframe"))
    except Exception as error:
        raise RuntimeError("Unable to fetch StatsBomb 360 freeze-frame data.") from error

    match = find_match(match_id) or {}
    events_by_id = {
        str(event["id"]): event
        for event in get_match_events(match_id)
        if event.get("id")
    }
    frames = group_360_frame_rows(rows, events_by_id, match)

    return {
        "match_id": int(match_id),
        "frame_count": len(frames),
        "frames": frames,
    }


def group_360_frame_rows(
    rows: list[dict[str, Any]],
    events_by_id: dict[str, dict[str, Any]],
    match: dict[str, Any],
) -> list[dict[str, Any]]:
    frames_by_id: dict[str, dict[str, Any]] = {}

    for row in rows:
        frame_id = row.get("event_uuid") or row.get("id")
        if not frame_id:
            continue

        frame_key = str(frame_id)
        event = events_by_id.get(frame_key, {})
        actor_team = event.get("team")
        actor_team_id = event.get("team_id")
        opponent_team = opponent_team_name(match, actor_team)
        opponent_team_id = opponent_team_id_from_match(match, actor_team)
        is_actor = bool(row.get("actor"))
        is_teammate = bool(row.get("teammate"))
        frame = frames_by_id.setdefault(
            frame_key,
            {
                "id": frame_key,
                "event_uuid": frame_key,
                "visible_area": row.get("visible_area"),
                "actor_player_name": event.get("player"),
                "actor_player_id": event.get("player_id"),
                "actor_team_name": actor_team,
                "actor_team_id": actor_team_id,
                "opponent_team_name": opponent_team,
                "opponent_team_id": opponent_team_id,
                "actor_position": event.get("position"),
                "event_type": event.get("type"),
                "players": [],
            },
        )

        if not frame.get("visible_area") and row.get("visible_area"):
            frame["visible_area"] = row.get("visible_area")

        frame["players"].append(
            {
                "location": row.get("location"),
                "teammate": is_teammate,
                "actor": is_actor,
                "keeper": bool(row.get("keeper")),
                "player_name": event.get("player") if is_actor else None,
                "player_id": event.get("player_id") if is_actor else None,
                "position": event.get("position") if is_actor else None,
                "team_name": actor_team if is_actor or is_teammate else opponent_team,
                "team_id": actor_team_id if is_actor or is_teammate else opponent_team_id,
                "event_type": event.get("type") if is_actor else None,
            }
        )

    frames = list(frames_by_id.values())
    for index, frame in enumerate(frames):
        frame["frame_number"] = index + 1
        frame["visible_player_count"] = len(frame["players"])
        frame["quality_label"] = (
            "High-quality frame"
            if frame["visible_player_count"] >= 14
            else "Low-quality frame"
        )

    return frames


def has_match_360(match_id: int) -> bool:
    return int(match_id) in set(get_statsbomb_360_match_ids()["match_ids"])


def opponent_team_name(match: dict[str, Any], team_name: str | None) -> str | None:
    if not team_name:
        return None
    if team_name == match.get("home_team"):
        return match.get("away_team")
    if team_name == match.get("away_team"):
        return match.get("home_team")
    return None


def opponent_team_id_from_match(match: dict[str, Any], team_name: str | None) -> int | None:
    if not team_name:
        return None
    if team_name == match.get("home_team"):
        return safe_int(match.get("away_team_id"))
    if team_name == match.get("away_team"):
        return safe_int(match.get("home_team_id"))
    return None


def get_eligible_players(match_id: int) -> list[dict[str, Any]]:
    match = get_match_details_without_players(match_id)
    if not match:
        return []

    events = get_match_events(match_id)
    return build_eligible_players(match, events)


def build_eligible_players(
    match: dict[str, Any],
    events: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    players_by_id: dict[int, dict[str, Any]] = {}

    for event in events:
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

    for player in players_by_id.values():
        metrics = build_obpi_metric_groups(events, player["player_id"])
        player["obpi_metrics"] = metrics
        player["has_obpi_data"] = all(metric["has_required_data"] for metric in metrics)
        player["missing_obpi_metrics"] = [
            metric["key"] for metric in metrics if not metric["has_required_data"]
        ]

    return sorted(
        players_by_id.values(),
        key=lambda player: (player["team_name"], player["player_name"]),
    )


def normalize_match_summary_with_readiness(match: dict[str, Any]) -> dict[str, Any]:
    summary = normalize_match_summary(match)
    summary.update(get_match_obpi_readiness(match))
    return summary


def get_match_readiness(match_id: int) -> dict[str, Any] | None:
    match = find_match(match_id)
    if not match:
        return None
    readiness = get_match_obpi_readiness(match)
    readiness["match_id"] = int(match_id)
    return readiness


def get_match_obpi_readiness(match: dict[str, Any]) -> dict[str, Any]:
    try:
        events = get_match_events(int(match["match_id"]))
        players = build_eligible_players(match_context_from_row(match), events)
        ready_players = [player for player in players if player["has_obpi_data"]]

        return {
            "eligible_player_count": len(players),
            "analysis_ready_player_count": len(ready_players),
            "has_analysis_ready_player": len(ready_players) > 0,
        }
    except Exception:
        return {
            "eligible_player_count": None,
            "analysis_ready_player_count": 0,
            "has_analysis_ready_player": False,
        }


def build_obpi_metric_groups(events: list[dict[str, Any]], player_id: int) -> list[dict[str, Any]]:
    return [build_obpi_metric_group(events, player_id, group) for group in OBPI_METRIC_GROUPS]


def build_obpi_metric_group(
    events: list[dict[str, Any]],
    player_id: int,
    group: dict[str, Any],
) -> dict[str, Any]:
    rows = [
        extract_metric_row(event, group["columns"])
        for event in events
        if is_event_relevant_for_metric(event, player_id, group["key"])
    ]

    available_columns = [
        column
        for column in group["columns"]
        if any(has_value(row.get(column)) for row in rows)
    ]
    missing_columns = [column for column in group["columns"] if column not in available_columns]
    complete_row_count = sum(1 for row in rows if row_has_all_metric_values(row, group["columns"]))

    return {
        "key": group["key"],
        "label": group["label"],
        "columns": list(group["columns"]),
        "row_count": len(rows),
        "complete_row_count": complete_row_count,
        "rows": rows,
        "has_required_data": complete_row_count > 0,
        "missing_columns": missing_columns,
    }


def extract_metric_row(event: dict[str, Any], columns: tuple[str, ...]) -> dict[str, Any]:
    row = {column: event.get(column) for column in columns}
    row["minute"] = event.get("minute")
    row["second"] = event.get("second")
    return row


def is_event_relevant_for_metric(event: dict[str, Any], player_id: int, metric_key: str) -> bool:
    event_player_id = safe_int(event.get("player_id"))
    pass_recipient_id = safe_int(event.get("pass_recipient_id"))
    substitution_replacement_id = safe_int(event.get("substitution_replacement_id"))

    if metric_key in ("M3", "M5", "M9"):
        return event_player_id == player_id or pass_recipient_id == player_id

    if metric_key == "M4":
        return event_player_id == player_id or substitution_replacement_id == player_id

    return event_player_id == player_id


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


def row_has_all_metric_values(row: dict[str, Any], columns: tuple[str, ...]) -> bool:
    return all(has_value(row.get(column)) for column in columns)


def safe_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def has_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, float) and isnan(value):
        return False
    if isinstance(value, str):
        return value.strip() != ""
    if isinstance(value, (list, dict)):
        return len(value) > 0
    return True


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
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
        return clean_value(value.tolist())
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

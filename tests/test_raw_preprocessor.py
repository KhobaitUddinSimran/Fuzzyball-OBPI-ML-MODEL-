"""Tests for raw StatsBomb open-data preprocessing."""

import json
from pathlib import Path

import pandas as pd

from obpi.data.preprocessor import StatsBombOpenDataPreprocessor


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_preprocess_all_creates_expected_outputs(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw" / "statsbomb_open_data"
    output_dir = tmp_path / "interim"

    _write_json(
        raw_dir / "competitions.json",
        [
            {
                "competition_id": 11,
                "competition_name": "La Liga",
                "country_name": "Spain",
                "competition_gender": "male",
                "competition_international": False,
                "competition_youth": False,
                "season_id": 90,
                "season_name": "2020/2021",
                "match_available": "2025-01-01T00:00:00",
                "match_available_360": "2025-01-01T00:00:00",
            }
        ],
    )
    _write_json(
        raw_dir / "matches" / "11" / "90.json",
        [
            {
                "match_id": 1234,
                "match_date": "2021-01-01",
                "kick_off": "21:00:00.000",
                "match_week": 1,
                "home_score": 1,
                "away_score": 0,
                "match_status": "available",
                "match_status_360": "available",
                "last_updated": "2025-01-01T00:00:00",
                "last_updated_360": "2025-01-01T00:00:00",
                "competition": {
                    "competition_id": 11,
                    "country_name": "Spain",
                    "competition_name": "La Liga",
                },
                "season": {"season_id": 90, "season_name": "2020/2021"},
                "home_team": {"home_team_id": 1, "home_team_name": "Barcelona"},
                "away_team": {"away_team_id": 2, "away_team_name": "Real Madrid"},
                "competition_stage": {"id": 1, "name": "Regular Season"},
                "stadium": {"id": 10, "name": "Camp Nou"},
                "referee": {"id": 20, "name": "Ref A"},
                "metadata": {
                    "data_version": "1.0",
                    "shot_fidelity_version": "2",
                    "xy_fidelity_version": "2",
                },
            }
        ],
    )
    _write_json(
        raw_dir / "lineups" / "1234.json",
        [
            {
                "team_id": 1,
                "team_name": "Barcelona",
                "lineup": [
                    {
                        "player_id": 99,
                        "player_name": "Player One",
                        "player_nickname": "P1",
                        "jersey_number": 8,
                        "country": {"id": 34, "name": "Spain"},
                        "cards": [],
                        "positions": [
                            {
                                "position_id": 19,
                                "position": "Center Attacking Midfield",
                                "from": "00:00",
                                "to": None,
                                "from_period": 1,
                                "to_period": None,
                                "start_reason": "Starting XI",
                                "end_reason": "Final Whistle",
                            }
                        ],
                    }
                ],
            }
        ],
    )
    _write_json(
        raw_dir / "events" / "1234.json",
        [
            {
                "id": "evt-1",
                "index": 1,
                "period": 1,
                "timestamp": "00:01:00.000",
                "minute": 1,
                "second": 0,
                "duration": 1.2,
                "possession": 1,
                "possession_team": {"id": 1, "name": "Barcelona"},
                "play_pattern": {"id": 1, "name": "Regular Play"},
                "team": {"id": 1, "name": "Barcelona"},
                "player": {"id": 99, "name": "Player One"},
                "position": {"id": 19, "name": "Center Attacking Midfield"},
                "type": {"id": 30, "name": "Pass"},
                "location": [60.0, 40.0],
                "under_pressure": False,
                "pass": {
                    "recipient": {"id": 100, "name": "Player Two"},
                    "length": 5.0,
                    "angle": 0.2,
                    "height": {"id": 1, "name": "Ground Pass"},
                    "type": {"id": 65, "name": "Kick Off"},
                    "body_part": {"id": 40, "name": "Right Foot"},
                    "end_location": [65.0, 42.0],
                },
            }
        ],
    )
    _write_json(
        raw_dir / "three-sixty" / "1234.json",
        [
            {
                "event_uuid": "evt-1",
                "visible_area": [0, 0, 120, 80],
                "freeze_frame": [
                    {"teammate": True, "location": [60.0, 40.0]},
                    {"teammate": False, "location": [63.0, 41.0]},
                ],
            }
        ],
    )

    preprocessor = StatsBombOpenDataPreprocessor(raw_dir=raw_dir, output_dir=output_dir)
    outputs = preprocessor.preprocess_all()

    assert set(outputs) == {
        "competitions",
        "matches",
        "player_matches",
        "event_manifest",
    }
    assert len(outputs["competitions"]) == 1
    assert len(outputs["matches"]) == 1
    assert len(outputs["player_matches"]) == 1
    assert len(outputs["event_manifest"]) == 1

    competitions = pd.read_parquet(output_dir / "competitions.parquet")
    matches = pd.read_parquet(output_dir / "matches.parquet")
    player_matches = pd.read_parquet(output_dir / "player_matches.parquet")
    event_manifest = pd.read_parquet(output_dir / "events_manifest.parquet")
    events = pd.read_parquet(output_dir / "events_by_match" / "1234.parquet")

    assert competitions.iloc[0]["competition_name"] == "La Liga"
    assert matches.iloc[0]["match_id"] == 1234
    assert player_matches.iloc[0]["player_id"] == 99
    assert event_manifest.iloc[0]["event_count"] == 1
    assert bool(event_manifest.iloc[0]["has_three_sixty_file"]) is True
    assert event_manifest.iloc[0]["freeze_frame_event_count"] == 1
    assert events.iloc[0]["type_name"] == "Pass"
    assert events.iloc[0]["pass_recipient_name"] == "Player Two"
    assert bool(events.iloc[0]["has_freeze_frame"]) is True
    assert bool(events.iloc[0]["has_visible_area"]) is True
    assert "63.0" in events.iloc[0]["freeze_frame_json"]

"""Tests for interim-to-processed metric computation."""

from pathlib import Path

import pandas as pd

from obpi.data.metric_processing import InterimMetricsProcessor


def test_process_matches_and_aggregate(tmp_path: Path) -> None:
    interim_dir = tmp_path / "interim"
    events_dir = interim_dir / "events_by_match"
    output_dir = tmp_path / "processed"
    events_dir.mkdir(parents=True)

    event_rows = [
        {
            "match_id": 1,
            "event_id": "e1",
            "index": 1,
            "period": 1,
            "timestamp": "00:00:01.000",
            "minute": 0,
            "second": 1,
            "duration": 0.5,
            "possession": 1,
            "under_pressure": False,
            "counterpress": None,
            "off_camera": None,
            "out": None,
            "related_events_json": None,
            "freeze_frame_json": (
                '[{"teammate": true, "location": [60.0, 40.0]}, '
                '{"teammate": false, "location": [63.0, 41.0]}]'
            ),
            "visible_area_json": "[0, 0, 120, 80]",
            "has_freeze_frame": True,
            "has_visible_area": True,
            "tactics_formation": None,
            "source_event_type": "Pass",
            "type_id": 30,
            "type_name": "Pass",
            "team_id": 10,
            "team_name": "Team A",
            "player_id": 100,
            "player_name": "Player A",
            "position_id": 19,
            "position_name": "Center Attacking Midfield",
            "possession_team_id": 10,
            "possession_team_name": "Team A",
            "play_pattern_id": 1,
            "play_pattern_name": "Regular Play",
            "location_x": 60.0,
            "location_y": 40.0,
            "location_z": None,
            "pass_end_location_x": 65.0,
            "pass_end_location_y": 42.0,
            "pass_end_location_z": None,
            "pass_length": 5.0,
            "pass_angle": 0.1,
            "pass_recipient_id": 101,
            "pass_recipient_name": "Player B",
            "pass_height_id": None,
            "pass_height_name": None,
            "pass_type_id": None,
            "pass_type_name": None,
            "pass_body_part_id": None,
            "pass_body_part_name": None,
            "pass_outcome_id": None,
            "pass_outcome_name": None,
            "carry_end_location_x": None,
            "carry_end_location_y": None,
            "carry_end_location_z": None,
            "shot_end_location_x": None,
            "shot_end_location_y": None,
            "shot_end_location_z": None,
            "shot_statsbomb_xg": None,
            "shot_first_time": None,
            "shot_outcome_id": None,
            "shot_outcome_name": None,
            "shot_body_part_id": None,
            "shot_body_part_name": None,
            "shot_type_id": None,
            "shot_type_name": None,
            "ball_receipt_id": None,
            "ball_receipt_name": None,
            "dribble_id": None,
            "dribble_name": None,
            "duel_id": None,
            "duel_name": None,
            "foul_committed_id": None,
            "foul_committed_name": None,
            "interception_id": None,
            "interception_name": None,
            "clearance_id": None,
            "clearance_name": None,
        },
        {
            "match_id": 1,
            "event_id": "e2",
            "index": 2,
            "period": 1,
            "timestamp": "00:00:03.000",
            "minute": 0,
            "second": 3,
            "duration": 0.5,
            "possession": 1,
            "under_pressure": False,
            "counterpress": None,
            "off_camera": None,
            "out": None,
            "related_events_json": None,
            "freeze_frame_json": (
                '[{"teammate": true, "location": [70.0, 40.0]}, '
                '{"teammate": false, "location": [73.0, 41.0]}]'
            ),
            "visible_area_json": "[0, 0, 120, 80]",
            "has_freeze_frame": True,
            "has_visible_area": True,
            "tactics_formation": None,
            "source_event_type": "Ball Receipt*",
            "type_id": 42,
            "type_name": "Ball Receipt*",
            "team_id": 10,
            "team_name": "Team A",
            "player_id": 100,
            "player_name": "Player A",
            "position_id": 19,
            "position_name": "Center Attacking Midfield",
            "possession_team_id": 10,
            "possession_team_name": "Team A",
            "play_pattern_id": 1,
            "play_pattern_name": "Regular Play",
            "location_x": 70.0,
            "location_y": 40.0,
            "location_z": None,
            "pass_end_location_x": None,
            "pass_end_location_y": None,
            "pass_end_location_z": None,
            "pass_length": None,
            "pass_angle": None,
            "pass_recipient_id": None,
            "pass_recipient_name": None,
            "pass_height_id": None,
            "pass_height_name": None,
            "pass_type_id": None,
            "pass_type_name": None,
            "pass_body_part_id": None,
            "pass_body_part_name": None,
            "pass_outcome_id": None,
            "pass_outcome_name": None,
            "carry_end_location_x": None,
            "carry_end_location_y": None,
            "carry_end_location_z": None,
            "shot_end_location_x": None,
            "shot_end_location_y": None,
            "shot_end_location_z": None,
            "shot_statsbomb_xg": None,
            "shot_first_time": None,
            "shot_outcome_id": None,
            "shot_outcome_name": None,
            "shot_body_part_id": None,
            "shot_body_part_name": None,
            "shot_type_id": None,
            "shot_type_name": None,
            "ball_receipt_id": None,
            "ball_receipt_name": None,
            "dribble_id": None,
            "dribble_name": None,
            "duel_id": None,
            "duel_name": None,
            "foul_committed_id": None,
            "foul_committed_name": None,
            "interception_id": None,
            "interception_name": None,
            "clearance_id": None,
            "clearance_name": None,
        },
    ]
    pd.DataFrame(event_rows).to_parquet(events_dir / "1.parquet", index=False)
    pd.DataFrame(
        [
            {
                "match_id": 1,
                "event_count": 2,
                "player_event_count": 2,
                "team_count": 1,
                "has_three_sixty_file": True,
                "freeze_frame_event_count": 2,
                "visible_area_event_count": 2,
                "source_file": "events/1.json",
                "three_sixty_file": "three-sixty/1.json",
                "output_file": "events_by_match/1.parquet",
            }
        ]
    ).to_parquet(interim_dir / "events_manifest.parquet", index=False)
    pd.DataFrame(
        [
            {
                "match_id": 1,
                "team_id": 10,
                "team_name": "Team A",
                "player_id": 100,
                "player_name": "Player A",
                "player_nickname": None,
                "jersey_number": 8,
                "country_id": None,
                "country_name": None,
                "starting_position_id": 19,
                "starting_position_name": "Center Attacking Midfield",
                "start_reason": "Starting XI",
                "end_reason": "Final Whistle",
                "from_period": 1,
                "to_period": None,
                "positions_json": (
                    '[{"position_id":19,"position":"Center Attacking Midfield",'
                    '"from":"00:00","to":null,"from_period":1,"to_period":null,'
                    '"start_reason":"Starting XI","end_reason":"Final Whistle"}]'
                ),
                "cards_json": None,
                "source_file": "lineups/1.json",
            }
        ]
    ).to_parquet(interim_dir / "player_matches.parquet", index=False)

    processor = InterimMetricsProcessor(interim_dir=interim_dir, output_dir=output_dir)
    match_metrics = processor.process_matches()
    aggregate_metrics = processor.aggregate_player_metrics(match_metrics)

    assert len(match_metrics) == 1
    assert match_metrics.iloc[0]["player_id"] == 100
    assert match_metrics.iloc[0]["minutes"] == 90.0
    assert bool(match_metrics.iloc[0]["has_360_data"]) is True
    assert match_metrics.iloc[0]["freeze_frame_count"] == 2
    assert len(aggregate_metrics) == 1
    assert aggregate_metrics.iloc[0]["match_count"] == 1
    assert bool(aggregate_metrics.iloc[0]["has_360_data"]) is True
    assert (output_dir / "player_match_metrics.parquet").exists()
    assert (output_dir / "player_aggregate_metrics.parquet").exists()


def test_process_matches_require_360_filters_manifest(tmp_path: Path) -> None:
    interim_dir = tmp_path / "interim"
    events_dir = interim_dir / "events_by_match"
    output_dir = tmp_path / "processed"
    events_dir.mkdir(parents=True)

    base_event = {
        "index": 1,
        "period": 1,
        "timestamp": "00:00:01.000",
        "minute": 0,
        "second": 1,
        "duration": 0.5,
        "possession": 1,
        "under_pressure": False,
        "counterpress": None,
        "off_camera": None,
        "out": None,
        "related_events_json": None,
        "visible_area_json": None,
        "has_freeze_frame": False,
        "has_visible_area": False,
        "tactics_formation": None,
        "source_event_type": "Pass",
        "type_id": 30,
        "type_name": "Pass",
        "team_id": 10,
        "team_name": "Team A",
        "player_id": 100,
        "player_name": "Player A",
        "position_id": 19,
        "position_name": "Center Attacking Midfield",
        "possession_team_id": 10,
        "possession_team_name": "Team A",
        "play_pattern_id": 1,
        "play_pattern_name": "Regular Play",
        "location_x": 60.0,
        "location_y": 40.0,
        "location_z": None,
        "pass_end_location_x": 65.0,
        "pass_end_location_y": 42.0,
        "pass_end_location_z": None,
        "pass_length": 5.0,
        "pass_angle": 0.1,
        "pass_recipient_id": 101,
        "pass_recipient_name": "Player B",
        "pass_height_id": None,
        "pass_height_name": None,
        "pass_type_id": None,
        "pass_type_name": None,
        "pass_body_part_id": None,
        "pass_body_part_name": None,
        "pass_outcome_id": None,
        "pass_outcome_name": None,
        "carry_end_location_x": None,
        "carry_end_location_y": None,
        "carry_end_location_z": None,
        "shot_end_location_x": None,
        "shot_end_location_y": None,
        "shot_end_location_z": None,
        "shot_statsbomb_xg": None,
        "shot_first_time": None,
        "shot_outcome_id": None,
        "shot_outcome_name": None,
        "shot_body_part_id": None,
        "shot_body_part_name": None,
        "shot_type_id": None,
        "shot_type_name": None,
        "ball_receipt_id": None,
        "ball_receipt_name": None,
        "dribble_id": None,
        "dribble_name": None,
        "duel_id": None,
        "duel_name": None,
        "foul_committed_id": None,
        "foul_committed_name": None,
        "interception_id": None,
        "interception_name": None,
        "clearance_id": None,
        "clearance_name": None,
    }
    with_frames = base_event | {
        "match_id": 1,
        "event_id": "e1",
        "freeze_frame_json": '[{"teammate": false, "location": [63.0, 41.0]}]',
        "visible_area_json": "[0, 0, 120, 80]",
        "has_freeze_frame": True,
        "has_visible_area": True,
    }
    without_frames = base_event | {
        "match_id": 2,
        "event_id": "e2",
        "freeze_frame_json": None,
    }
    pd.DataFrame([with_frames]).to_parquet(events_dir / "1.parquet", index=False)
    pd.DataFrame([without_frames]).to_parquet(events_dir / "2.parquet", index=False)
    pd.DataFrame(
        [
            {
                "match_id": 1,
                "event_count": 1,
                "player_event_count": 1,
                "team_count": 1,
                "has_three_sixty_file": True,
                "freeze_frame_event_count": 1,
                "visible_area_event_count": 1,
                "source_file": "events/1.json",
                "three_sixty_file": "three-sixty/1.json",
                "output_file": "events_by_match/1.parquet",
            },
            {
                "match_id": 2,
                "event_count": 1,
                "player_event_count": 1,
                "team_count": 1,
                "has_three_sixty_file": False,
                "freeze_frame_event_count": 0,
                "visible_area_event_count": 0,
                "source_file": "events/2.json",
                "three_sixty_file": None,
                "output_file": "events_by_match/2.parquet",
            },
        ]
    ).to_parquet(interim_dir / "events_manifest.parquet", index=False)
    player_matches = pd.DataFrame(
        [
            {
                "match_id": 1,
                "team_id": 10,
                "team_name": "Team A",
                "player_id": 100,
                "player_name": "Player A",
                "player_nickname": None,
                "jersey_number": 8,
                "country_id": None,
                "country_name": None,
                "starting_position_id": 19,
                "starting_position_name": "Center Attacking Midfield",
                "start_reason": "Starting XI",
                "end_reason": "Final Whistle",
                "from_period": 1,
                "to_period": None,
                "positions_json": None,
                "cards_json": None,
                "source_file": "lineups/1.json",
            },
            {
                "match_id": 2,
                "team_id": 10,
                "team_name": "Team A",
                "player_id": 100,
                "player_name": "Player A",
                "player_nickname": None,
                "jersey_number": 8,
                "country_id": None,
                "country_name": None,
                "starting_position_id": 19,
                "starting_position_name": "Center Attacking Midfield",
                "start_reason": "Starting XI",
                "end_reason": "Final Whistle",
                "from_period": 1,
                "to_period": None,
                "positions_json": None,
                "cards_json": None,
                "source_file": "lineups/2.json",
            },
        ]
    )
    player_matches.to_parquet(interim_dir / "player_matches.parquet", index=False)

    processor = InterimMetricsProcessor(interim_dir=interim_dir, output_dir=output_dir)
    match_metrics = processor.process_matches(require_360=True)

    assert set(match_metrics["match_id"]) == {1}

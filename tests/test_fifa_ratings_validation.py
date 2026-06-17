from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_fifa_ratings_validation.py"
SPEC = importlib.util.spec_from_file_location("run_fifa_ratings_validation", SCRIPT_PATH)
assert SPEC is not None
fifa_validation = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(fifa_validation)


def test_normalize_name_removes_accents_and_suffixes() -> None:
    assert fifa_validation.normalize_name("Neymar da Silva Santos Júnior") == (
        "neymar da silva santos"
    )
    assert fifa_validation.token_sort_name("Raheem Sterling") == "raheem sterling"


def test_match_obpi_to_fifa_deduplicates_exact_lookup_keys() -> None:
    aggregate = pd.DataFrame(
        {
            "player_id": [1, 2],
            "player_name": ["Raheem Sterling", "Phil Foden"],
            "obpi_mean": [0.7, 0.8],
            "obpi_median": [0.7, 0.8],
            "obpi_matches": [2, 1],
            "obpi_team_names": ["Chelsea", "Manchester City"],
        }
    )
    fifa_df = pd.DataFrame(
        {
            "player_id": [10, 11],
            "fifa_version": [23, 23],
            "fifa_update": [9, 9],
            "fifa_update_date": ["2023-01-13", "2023-01-13"],
            "short_name": ["R. Sterling", "P. Foden"],
            "long_name": ["Raheem Sterling", "Philip Foden"],
            "player_positions": ["LW, RW", "LW, CAM"],
            "overall": [85, 85],
            "potential": [85, 92],
            "pace": [90, 82],
            "shooting": [80, 78],
            "passing": [79, 82],
            "dribbling": [86, 88],
            "defending": [45, 56],
            "physic": [67, 60],
            "mentality_vision": [78, 84],
            "movement_reactions": [84, 86],
            "club_name": ["Chelsea", "Manchester City"],
            "league_name": ["Premier League", "Premier League"],
            "nationality_name": ["England", "England"],
        }
    )

    matched = fifa_validation.match_obpi_to_fifa(
        aggregate,
        fifa_df,
        min_fuzzy_confidence=0.90,
    )

    assert set(matched["player_name"]) == {"Raheem Sterling", "Phil Foden"}
    sterling = matched[matched["player_name"].eq("Raheem Sterling")].iloc[0]
    assert sterling["match_type"] == "exact_name"
    foden = matched[matched["player_name"].eq("Phil Foden")].iloc[0]
    assert foden["match_type"] == "fuzzy_name"
    assert foden["match_confidence"] >= 0.90


def test_match_prefers_contextual_containment_over_wrong_exact_name() -> None:
    aggregate = pd.DataFrame(
        {
            "player_id": [1],
            "player_name": ["Aaron Ramsey"],
            "obpi_mean": [0.7],
            "obpi_median": [0.7],
            "obpi_matches": [1],
            "obpi_team_names": ["Wales"],
        }
    )
    fifa_df = pd.DataFrame(
        {
            "player_id": [10, 11],
            "fifa_version": [23, 23],
            "fifa_update": [9, 9],
            "fifa_update_date": ["2023-01-13", "2023-01-13"],
            "short_name": ["A. Ramsey", "A. Ramsey"],
            "long_name": ["Aaron Ramsey", "Aaron James Ramsey"],
            "player_positions": ["CM", "CM"],
            "overall": [66, 80],
            "potential": [82, 80],
            "pace": [72, 59],
            "shooting": [60, 74],
            "passing": [64, 79],
            "dribbling": [69, 79],
            "defending": [49, 70],
            "physic": [60, 69],
            "mentality_vision": [68, 82],
            "movement_reactions": [54, 80],
            "club_name": ["Aston Villa", "Nice"],
            "league_name": ["Premier League", "Ligue 1"],
            "nationality_name": ["England", "Wales"],
        }
    )

    matched = fifa_validation.match_obpi_to_fifa(
        aggregate,
        fifa_df,
        min_fuzzy_confidence=0.90,
    )

    assert matched.iloc[0]["fifa_long_name"] == "Aaron James Ramsey"
    assert matched.iloc[0]["match_type"] == "token_containment"

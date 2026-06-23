"""API integration tests for the OBPI FastAPI backend."""

from __future__ import annotations

import pandas as pd
from fastapi.testclient import TestClient

from api.main import app
from api.services.pipeline_service import _build_data_quality
from obpi.fuzzy.processing import run_real_data_fuzzy_processing
from obpi.pipeline import _build_frames_by_event_id, _count_events_with_frames

client = TestClient(app)


class TestHealth:
    """Tests for the /health endpoint."""

    def test_health_returns_200(self) -> None:
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["model_version"] == "1.0.0"
        assert data["schema_version"] == 3
        assert data["uptime_seconds"] >= 0
        assert data["cache_connected"] is True


class TestPlayers:
    """Tests for the /players and /players/{id} endpoints."""

    def test_players_404_or_503_bad_match_id(self) -> None:
        """Requesting a nonexistent match should return 404 or 503."""
        response = client.get("/players?match_id=9999999")
        # 503 if StatsBomb data source is unavailable, 404 if match not found
        assert response.status_code in (404, 503)

    def test_player_detail_404_bad_player_id(self) -> None:
        """Requesting a nonexistent player in a match should return 400."""
        response = client.get("/players/999999?match_id=3794686")
        # 400 because player not found in match, 503 if statsbomb fails
        assert response.status_code in (400, 503)


class TestAnalyze:
    """Tests for the POST /analyze endpoint."""

    def test_analyze_422_invalid_tier(self) -> None:
        response = client.post(
            "/analyze",
            json={"match_id": 3794686, "player_id": 1001, "tier": "invalid"},
        )
        assert response.status_code == 422

    def test_analyze_422_missing_match_id(self) -> None:
        response = client.post("/analyze", json={"player_id": 1001})
        assert response.status_code == 422


class TestCompare:
    """Tests for the POST /compare endpoint."""

    def test_compare_422_missing_player_ids(self) -> None:
        response = client.post("/compare", json={"match_id": 3794686})
        assert response.status_code == 422

    def test_compare_422_wrong_number_of_players(self) -> None:
        response = client.post(
            "/compare",
            json={"match_id": 3794686, "player_ids": [1001]},
        )
        assert response.status_code == 422


class TestLeaderboard:
    """Tests for the GET /leaderboard endpoint."""

    def test_leaderboard_404_or_503_bad_match_id(self) -> None:
        response = client.get("/leaderboard?match_id=9999999")
        assert response.status_code in (404, 503)

    def test_leaderboard_limit_validation(self) -> None:
        """Limit > 50 should be rejected."""
        response = client.get("/leaderboard?match_id=3794686&limit=100")
        assert response.status_code == 422


class TestCors:
    """Smoke test for CORS preflight."""

    def test_cors_preflight(self) -> None:
        response = client.options(
            "/health",
            headers={
                "Origin": "http://localhost:5173",
                "Access-Control-Request-Method": "GET",
            },
        )
        assert response.status_code == 200
        assert "access-control-allow-origin" in response.headers


class TestStatsBombRoutes:
    """Tests for dashboard StatsBomb browsing endpoints."""

    def test_world_cup_years(self, monkeypatch) -> None:
        monkeypatch.setattr(
            "api.routes.statsbomb.get_world_cup_years",
            lambda: [{"year": "2022", "season_id": 106, "label": "2022 FIFA World Cup"}],
        )

        response = client.get("/events/fifa-world-cup/years")

        assert response.status_code == 200
        assert response.json()[0]["year"] == "2022"

    def test_matches_by_year(self, monkeypatch) -> None:
        monkeypatch.setattr(
            "api.routes.statsbomb.get_matches_by_year",
            lambda year: [{"match_id": 3794686, "home_team": "Argentina"}],
        )

        response = client.get("/matches?event=fifa-world-cup&year=2022")

        assert response.status_code == 200
        assert response.json()[0]["match_id"] == 3794686

    def test_matches_rejects_unknown_event(self) -> None:
        response = client.get("/matches?event=euro&year=2022")

        assert response.status_code == 400

    def test_match_detail_404(self, monkeypatch) -> None:
        monkeypatch.setattr("api.routes.statsbomb.get_match_details", lambda match_id: None)

        response = client.get("/matches/1")

        assert response.status_code == 404

    def test_eligible_players(self, monkeypatch) -> None:
        monkeypatch.setattr(
            "api.routes.statsbomb.get_match_details",
            lambda match_id: {"match_id": match_id},
        )
        monkeypatch.setattr(
            "api.routes.statsbomb.get_eligible_players",
            lambda match_id: [{"player_id": 10, "player_name": "Test Player"}],
        )

        response = client.get("/matches/3794686/eligible-players")

        assert response.status_code == 200
        assert response.json()[0]["player_id"] == 10

    def test_match_frames(self, monkeypatch) -> None:
        monkeypatch.setattr(
            "api.routes.statsbomb.get_match_details",
            lambda match_id: {"match_id": match_id},
        )
        monkeypatch.setattr(
            "api.routes.statsbomb.get_match_360_frames",
            lambda match_id: {"match_id": match_id, "frame_count": 0, "frames": []},
        )

        response = client.get("/matches/3794686/frames")

        assert response.status_code == 200
        assert response.json()["frames"] == []


class TestMetricQualityHandling:
    """Tests for raw/normalized/fuzzy metric quality handling."""

    def test_unavailable_metric_uses_neutral_fuzzy_fallback(self) -> None:
        metrics_df = pd.DataFrame(
            [
                {
                    "player_id": 1,
                    "match_id": 10,
                    "M1_SC": None,
                    "M2_OIRC": 0.0,
                    "M3_BRPC": 0.2,
                    "M4_OBR90": 4.62,
                    "M5_RBTL": 0.0,
                    "M6_RUP": 0.5,
                    "M7_SCI": None,
                    "M8_LPC": 0.0,
                    "M9_CBI": 0.0,
                },
                {
                    "player_id": 2,
                    "match_id": 10,
                    "M1_SC": None,
                    "M2_OIRC": 1.0,
                    "M3_BRPC": 0.4,
                    "M4_OBR90": 9.0,
                    "M5_RBTL": 1.0,
                    "M6_RUP": 0.0,
                    "M7_SCI": None,
                    "M8_LPC": 1.0,
                    "M9_CBI": 1.0,
                },
            ]
        )

        scored, metadata = run_real_data_fuzzy_processing(
            metrics_df,
            id_columns=["player_id", "match_id"],
        )

        assert scored["M1"].tolist() == [0.5, 0.5]
        assert scored["M7"].tolist() == [0.5, 0.5]
        assert "M1_fuzzy" in scored.columns
        assert scored["obpi"].between(0, 1).all()
        assert metadata["normalization"]["M1"]["fallback"].startswith("unavailable")

    def test_data_quality_reports_unavailable_metrics_and_missing_minutes(self) -> None:
        row = pd.Series(
            {
                "has_360": False,
                "events_loaded": True,
                "frames_loaded": False,
                "joined_360_frames": 0,
                "high_quality_frames": 0,
                "minutes_available": False,
                "M1_SC_status": "unavailable",
                "M2_OIRC_status": "available",
                "M3_BRPC_status": "unavailable",
                "M4_OBR90_status": "available_with_warning",
                "M5_RBTL_status": "available",
                "M6_RUP_status": "available",
                "M7_SCI_status": "unavailable",
                "M8_LPC_status": "available",
                "M9_CBI_status": "unavailable",
                "data_quality_warnings": [
                    "Minutes played is missing, so M4_OBR90 may be unreliable.",
                    "Some frame-dependent metrics were unavailable.",
                ],
            }
        )

        quality = _build_data_quality(row)

        assert quality.has_360 is False
        assert quality.minutes_available is False
        assert quality.unavailable_metrics == ["M1_SC", "M3_BRPC", "M7_SCI", "M9_CBI"]
        assert "Some frame-dependent metrics were unavailable." in quality.warnings

    def test_freeze_frame_rows_are_grouped_by_event_uuid(self) -> None:
        rows = [
            {
                "event_uuid": "event-a",
                "location": [1, 2],
                "teammate": True,
                "actor": True,
            },
            {
                "event_uuid": "event-a",
                "location": [3, 4],
                "teammate": False,
                "actor": False,
            },
            {
                "id": "event-b",
                "location": [5, 6],
                "teammate": True,
                "actor": False,
            },
        ]

        frames_by_event_id = _build_frames_by_event_id(rows)

        assert sorted(frames_by_event_id) == ["event-a", "event-b"]
        assert len(frames_by_event_id["event-a"]) == 1
        assert len(frames_by_event_id["event-a"][0]["freeze_frame"]) == 2
        assert frames_by_event_id["event-b"][0]["freeze_frame"][0]["location"] == [5, 6]

    def test_joined_360_frames_counts_exact_matching_events(self) -> None:
        events = pd.DataFrame(
            [
                {"id": "event-a", "type_name": "Ball Receipt*", "player_id": 1},
                {"id": "event-b", "type_name": "Ball Receipt*", "player_id": 1},
                {"id": "event-c", "type_name": "Ball Receipt*", "player_id": 1},
            ]
        )
        frames_by_event_id = {
            "event-a": [{"freeze_frame": [{"location": [1, 2]}]}],
            "event-c": [{"freeze_frame": [{"location": [3, 4]}]}],
        }

        assert _count_events_with_frames(events, frames_by_event_id) == 2

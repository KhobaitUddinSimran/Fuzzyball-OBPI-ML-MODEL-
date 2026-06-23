"""API integration tests for the OBPI FastAPI backend."""

from __future__ import annotations

from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


class TestHealth:
    """Tests for the /health endpoint."""

    def test_health_returns_200(self) -> None:
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["model_version"] == "1.0.0"
        assert data["schema_version"] == 2
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

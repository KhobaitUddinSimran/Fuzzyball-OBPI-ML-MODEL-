"""Pytest fixtures and synthetic data generators for offline unit tests."""

from typing import Any

import pytest


class SyntheticMatchGenerator:
    """Generate fake but geometrically plausible match events and 360 frames."""

    def __init__(self, match_id: int = 999999) -> None:
        self.match_id = match_id
        self.home_team_id = 1
        self.away_team_id = 2
        self.pitch_length = 120.0
        self.pitch_width = 80.0

    def events(self, n_events: int = 50) -> list[dict[str, Any]]:
        """Return a list of synthetic event dicts mimicking StatsBomb schema."""
        events: list[dict[str, Any]] = []
        for i in range(n_events):
            event = {
                "id": i,
                "match_id": self.match_id,
                "index": i + 1,
                "period": 1,
                "timestamp": f"00:{i:02d}.000",
                "minute": i // 60,
                "second": i % 60,
                "type": {"id": 30, "name": "Pass"},
                "possession": 1,
                "possession_team": {"id": self.home_team_id, "name": "Home FC"},
                "team": {"id": self.home_team_id, "name": "Home FC"},
                "player": {"id": 1001, "name": "Test Player"},
                "position": {"id": 15, "name": "Left Center Midfield"},
                "location": [60.0, 40.0],
                "under_pressure": False,
            }
            events.append(event)
        return events

    def freeze_frames(self, n_frames: int = 10) -> list[dict[str, Any]]:
        """Return synthetic 360 freeze-frame dicts."""
        frames: list[dict[str, Any]] = []
        for i in range(n_frames):
            frame = {
                "event_uuid": f"evt-{i}",
                "match_id": self.match_id,
                "visible_area": [0, 0, self.pitch_length, self.pitch_width],
                "freeze_frame": [
                    {"teammate": True, "actor": True, "keeper": False, "location": [60.0, 40.0]},
                    {"teammate": True, "actor": False, "keeper": False, "location": [55.0, 35.0]},
                    {"teammate": False, "actor": False, "keeper": False, "location": [65.0, 42.0]},
                ],
            }
            frames.append(frame)
        return frames


@pytest.fixture
def synthetic_generator() -> SyntheticMatchGenerator:
    """Provide a default SyntheticMatchGenerator for tests."""
    return SyntheticMatchGenerator(match_id=999999)

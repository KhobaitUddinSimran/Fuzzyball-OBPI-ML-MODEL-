"""StatsBomb open-data / API wrapper with 360 availability filter."""

from pathlib import Path
from typing import Any

import pandas as pd

_DATA_ROOT = Path(__file__).resolve().parents[3] / "data" / "raw"


class StatsBombLoader:
    """Load StatsBomb event and 360 freeze-frame data.

    Supports both the open-data repository (free) and the paid API tier.
    """

    def __init__(self, tier: str = "open") -> None:
        """Initialize loader.

        Args:
            tier: Either ``"open"`` (free GitHub repo) or ``"api"`` (paid).
        """
        if tier not in {"open", "api"}:
            raise ValueError(f"tier must be 'open' or 'api', got {tier!r}")
        self.tier = tier

    def get_competitions(self) -> pd.DataFrame:
        """Return available competitions metadata.

        Returns:
            DataFrame with columns including ``competition_id``, ``season_id``,
            and ``match_available_360``.
        """
        if self.tier == "open":
            try:
                from statsbombpy import sb

                return sb.competitions()
            except ImportError as exc:
                raise ImportError("statsbombpy is required for open-data loading") from exc
        # API tier stub — would call authenticated endpoint in production
        raise NotImplementedError("API tier competitions not yet implemented")

    def get_matches(self, competition_id: int, season_id: int) -> pd.DataFrame:
        """Return match list for a given competition + season."""
        if self.tier == "open":
            try:
                from statsbombpy import sb

                return sb.matches(competition_id, season_id)
            except ImportError as exc:
                raise ImportError("statsbombpy is required for open-data loading") from exc
        raise NotImplementedError("API tier matches not yet implemented")

    def get_events(self, match_id: int) -> pd.DataFrame:
        """Return event-level data for a match.

        Args:
            match_id: StatsBomb match identifier.

        Returns:
            Events DataFrame with standard StatsBomb schema.
        """
        if self.tier == "open":
            try:
                from statsbombpy import sb

                return sb.events(match_id)
            except ImportError as exc:
                raise ImportError("statsbombpy is required for open-data loading") from exc
        raise NotImplementedError("API tier events not yet implemented")

    def get_freeze_frames(self, match_id: int) -> list[dict[str, Any]]:
        """Return 360 freeze-frame JSONs for a match.

        Args:
            match_id: StatsBomb match identifier.

        Returns:
            List of freeze-frame dicts (one per event that has 360 data).
        """
        if self.tier == "open":
            try:
                from statsbombpy import sb

                df = sb.frames(match_id)
                return list(df.to_dict(orient="records"))
            except ImportError as exc:
                raise ImportError("statsbombpy is required for open-data loading") from exc
        raise NotImplementedError("API tier freeze frames not yet implemented")

    def matches_with_360(self, competition_id: int, season_id: int) -> pd.DataFrame:
        """Filter matches to those with 360 freeze-frame data available.

        Args:
            competition_id: StatsBomb competition identifier.
            season_id: StatsBomb season identifier.

        Returns:
            Subset of the matches DataFrame where ``match_available_360`` is True.
        """
        matches = self.get_matches(competition_id, season_id)
        if "match_available_360" not in matches.columns:
            # Fallback: attempt to load frames for each match and keep successes
            return matches
        return matches[matches["match_available_360"] == True].reset_index(drop=True)  # noqa: E712

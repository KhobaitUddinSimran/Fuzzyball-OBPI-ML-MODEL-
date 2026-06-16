"""Tests for the StatsBomb data loader and preprocessing utilities."""

import numpy as np
import pandas as pd
import pytest

from obpi.data.loader import StatsBombLoader
from obpi.data.preprocessor import ConvexHullClipper, DeltaTGate


class TestStatsBombLoader:
    def test_init_defaults(self) -> None:
        loader = StatsBombLoader(tier="open")
        assert loader.tier == "open"

    def test_init_invalid_tier(self) -> None:
        with pytest.raises(ValueError, match="tier must be 'open' or 'api'"):
            StatsBombLoader(tier="invalid")

    def test_get_competitions_returns_dataframe(self) -> None:
        loader = StatsBombLoader(tier="open")
        try:
            comps = loader.get_competitions()
        except ImportError:
            pytest.skip("statsbombpy not installed")
        assert isinstance(comps, pd.DataFrame)
        assert not comps.empty

    def test_get_events_returns_dataframe(self) -> None:
        loader = StatsBombLoader(tier="open")
        try:
            events = loader.get_events(match_id=3794687)
        except ImportError:
            pytest.skip("statsbombpy not installed")
        assert isinstance(events, pd.DataFrame)
        assert not events.empty
        assert "type" in events.columns


class TestConvexHullClipper:
    def test_visible_hull_with_three_points(self) -> None:
        clipper = ConvexHullClipper()
        points = np.array([[60.0, 40.0], [55.0, 35.0], [65.0, 42.0]])
        hull = clipper.visible_hull(points)
        assert isinstance(hull, type(clipper.pitch_polygon))
        assert hull.area > 0

    def test_visible_hull_fewer_than_three_returns_pitch(self) -> None:
        clipper = ConvexHullClipper()
        points = np.array([[60.0, 40.0]])
        hull = clipper.visible_hull(points)
        assert hull.equals(clipper.pitch_polygon)


class TestDeltaTGate:
    def test_is_reliable_within_threshold(self) -> None:
        gate = DeltaTGate(max_dt=1.5)
        assert gate.is_reliable(1.0) is True
        assert gate.is_reliable(1.5) is True
        assert gate.is_reliable(2.0) is False
        assert gate.is_reliable(0.0) is False

    def test_filter_pairs(self) -> None:
        gate = DeltaTGate(max_dt=1.5)
        locs = np.array([[0, 0], [1, 0], [2, 0], [4, 0]])
        times = np.array([0.0, 1.0, 2.5, 5.0])
        filtered_locs, filtered_times, dts = gate.filter_pairs(locs, times)
        assert len(filtered_locs) == 3
        assert len(dts) == 2
        assert np.all(dts <= 1.5)

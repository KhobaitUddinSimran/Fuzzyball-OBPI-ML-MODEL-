"""Train a data-driven Expected Threat (xT) model from StatsBomb shots."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
import pandas as pd
from scipy.ndimage import gaussian_filter
from sklearn.linear_model import LogisticRegression

from obpi.data.loader import StatsBombLoader

logger = logging.getLogger("obpi.xt_trainer")

PITCH_LENGTH = 120.0
PITCH_WIDTH = 80.0
N_COLS = 12
N_ROWS = 8


def _zone_centres(n_cols: int = N_COLS, n_rows: int = N_ROWS) -> NDArray[np.float64]:
    """Return the (x, y) centre of each grid cell."""
    cell_w = PITCH_LENGTH / n_cols
    cell_h = PITCH_WIDTH / n_rows
    xs = np.arange(cell_w / 2, PITCH_LENGTH, cell_w)
    ys = np.arange(cell_h / 2, PITCH_WIDTH, cell_h)
    centres: NDArray[np.float64] = np.array(
        [[x, y] for y in ys for x in xs]
    )
    return centres


def _load_shots(loader: StatsBombLoader, match_ids: list[int] | None = None) -> pd.DataFrame:
    """Load shot events from the specified matches (or all available 360 matches)."""
    if match_ids is None:
        # UEFA Euro 2020 (competition 55, season 43) has open 360 data
        try:
            df_360 = loader.matches_with_360(competition_id=55, season_id=43)
            match_ids = df_360["match_id"].tolist()
        except Exception:
            match_ids = []
        if not match_ids:
            match_ids = [3794687]  # fallback open-data match

    shots: list[dict[str, Any]] = []
    for mid in match_ids:
        try:
            events = loader.get_events(mid)
        except Exception as exc:
            logger.warning("Failed to load events for match %s: %s", mid, exc)
            continue

        if events.empty:
            continue

        shot_mask = events["type"].apply(
            lambda t: (t.get("name") == "Shot" if isinstance(t, dict) else t == "Shot")
        )
        for _, row in events[shot_mask].iterrows():
            loc = row.get("location")
            if not isinstance(loc, (list, tuple)) or len(loc) < 2:
                continue
            # Handle both nested-dict (API) and flat-column (statsbombpy) formats
            shot_outcome = ""
            if "shot" in row and isinstance(row["shot"], dict):
                shot_outcome = row["shot"].get("outcome", {}).get("name", "")
            elif "shot_outcome" in row and pd.notna(row["shot_outcome"]):
                shot_outcome = str(row["shot_outcome"])
            shots.append(
                {
                    "x": float(loc[0]),
                    "y": float(loc[1]),
                    "goal": 1.0 if shot_outcome == "Goal" else 0.0,
                }
            )

    logger.info("Loaded %d shots from %d matches", len(shots), len(match_ids))
    return pd.DataFrame(shots)


def _mirror_left_side(df: pd.DataFrame) -> pd.DataFrame:
    """Mirror shots from the left half to the right half for symmetry."""
    mirrored = df.copy()
    mirrored["x"] = PITCH_LENGTH - mirrored["x"]
    mirrored["y"] = PITCH_WIDTH - mirrored["y"]
    return pd.concat([df, mirrored], ignore_index=True)


def train_xt_grid(
    loader: StatsBombLoader | None = None,
    match_ids: list[int] | None = None,
    n_cols: int = N_COLS,
    n_rows: int = N_ROWS,
    sigma: float = 1.0,
    mirror: bool = True,
) -> NDArray[np.float64]:
    """Train a 12×8 xT grid from StatsBomb shot data.

    Uses logistic regression on (x, y) to predict goal probability,
    evaluates on a regular grid, and smooths with a Gaussian filter.

    Returns:
        ``(n_rows, n_cols)`` array of xT values in ``[0, 1]``.
    """
    if loader is None:
        loader = StatsBombLoader()

    shots = _load_shots(loader, match_ids=match_ids)
    if shots.empty:
        raise RuntimeError("No shots loaded; cannot train xT grid.")

    if mirror:
        shots = _mirror_left_side(shots)

    # Standard xG features: distance and angle to opponent goal
    goal_x, goal_y = PITCH_LENGTH, PITCH_WIDTH / 2.0
    shots["dist"] = np.sqrt((shots["x"] - goal_x) ** 2 + (shots["y"] - goal_y) ** 2)
    shots["angle"] = np.arctan2(
        np.abs(shots["y"] - goal_y), PITCH_LENGTH - shots["x"]
    )

    feature_cols = ["dist", "angle"]
    X = shots[feature_cols].values
    y = shots["goal"].values

    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(X, y)

    centres = _zone_centres(n_cols=n_cols, n_rows=n_rows)
    centre_df = pd.DataFrame(centres, columns=["x", "y"])
    centre_df["dist"] = np.sqrt(
        (centre_df["x"] - goal_x) ** 2 + (centre_df["y"] - goal_y) ** 2
    )
    centre_df["angle"] = np.arctan2(
        np.abs(centre_df["y"] - goal_y), PITCH_LENGTH - centre_df["x"]
    )
    probs = model.predict_proba(centre_df[feature_cols].values)[:, 1]
    grid = probs.reshape(n_rows, n_cols)

    # Smooth with 2D Gaussian (σ in grid-cell units)
    grid = gaussian_filter(grid, sigma=sigma)

    # Clip to non-negative and scale so max = 1.0 (relative threat)
    grid = np.clip(grid, 0.0, None)
    grid_max = float(grid.max())
    if grid_max > 0:
        grid = grid / grid_max
    else:
        grid = np.zeros_like(grid)

    logger.info(
        "Trained xT grid %dx%d — mean=%.4f, max=%.4f, min=%.4f",
        n_rows,
        n_cols,
        float(grid.mean()),
        float(grid.max()),
        float(grid.min()),
    )
    return grid


def save_xt_grid(
    grid: NDArray[np.float64],
    path: str | Path,
) -> None:
    """Save a trained xT grid as a NumPy ``.npy`` file."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, grid)
    logger.info("Saved xT grid to %s", out)


def load_xt_grid(path: str | Path) -> NDArray[np.float64]:
    """Load a trained xT grid from a NumPy ``.npy`` file."""
    grid: NDArray[np.float64] = np.load(path)
    return grid


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    grid = train_xt_grid()
    save_xt_grid(grid, "data/processed/xt_grid_12x8.npy")

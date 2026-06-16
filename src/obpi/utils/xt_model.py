"""Expected Threat (xT) model — Karun Singh 12×8 grid."""


import numpy as np
from numpy.typing import NDArray

_GRID_COLS = 12
_GRID_ROWS = 8
_PITCH_LENGTH = 120.0
_PITCH_WIDTH = 80.0
_CELL_W = _PITCH_LENGTH / _GRID_COLS  # 10.0
_CELL_H = _PITCH_WIDTH / _GRID_ROWS   # 10.0


def _zone_from_location(location: list[float] | NDArray[np.float64]) -> tuple[int, int]:
    """Return grid zone (col, row) for a pitch location."""
    x = max(0.0, min(float(location[0]), _PITCH_LENGTH - 1e-9))
    y = max(0.0, min(float(location[1]), _PITCH_WIDTH - 1e-9))
    col = int(x // _CELL_W)
    row = int(y // _CELL_H)
    return col, row


class XTModel:
    """Simple xT model backed by a synthetic 12×8 grid.

    The grid values increase monotonically toward the opponent goal
    (positive x direction) to reward progressive actions.
    """

    def __init__(self, grid: NDArray[np.float64] | None = None) -> None:
        """Initialize with a custom grid or the default synthetic grid.

        Args:
            grid: Optional 2D array of shape (8, 12). If ``None``, a
                synthetic grid is built where values rise from 0.01
                (own goal) to 0.30 (opponent goal).
        """
        if grid is not None:
            self.grid = grid
        else:
            self.grid = self._default_grid()
        assert self.grid.shape == (_GRID_ROWS, _GRID_COLS)

    @staticmethod
    def _default_grid() -> NDArray[np.float64]:
        """Build a synthetic xT grid."""
        # Linear ramp from left to right, slight vertical symmetry
        base = np.linspace(0.01, 0.30, _GRID_COLS)
        # Add slight vertical variation (centre is higher)
        vertical = 1.0 + 0.1 * np.sin(np.linspace(0, np.pi, _GRID_ROWS))
        grid = base * vertical[:, np.newaxis]
        return grid.astype(np.float64)  # type: ignore[no-any-return]

    def xt_from_location(self, location: list[float] | NDArray[np.float64]) -> float:
        """Return xT value for a pitch location.

        Args:
            location: ``[x, y]`` coordinates.

        Returns:
            xT value of the zone containing the location.
        """
        col, row = _zone_from_location(location)
        return float(self.grid[row, col])

    def delta_xt(
        self,
        origin: list[float] | NDArray[np.float64],
        destination: list[float] | NDArray[np.float64],
    ) -> float:
        """Compute xT difference for a pass or carry.

        Args:
            origin: Start ``[x, y]``.
            destination: End ``[x, y]``.

        Returns:
            ``xT(destination) - xT(origin)``.
        """
        return self.xt_from_location(destination) - self.xt_from_location(origin)

    def get_grid(self) -> NDArray[np.float64]:
        """Return the full xT grid."""
        return self.grid

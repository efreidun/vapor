"""This module provides a grid on R3."""

from typing import Iterable

import numpy as np


class R3Grid:
    """Grid on R3."""

    def __init__(self, resol: float, ranges: np.ndarray) -> None:
        """Construct the grid.

        Args:
            resol: width of cells
            ranges: minimum and maximum values of each dimension, shape (3, 2)
        """
        self._resol = resol
        self._ranges = ranges
        dim_widths = ranges[:, 1] - ranges[:, 0]
        num_cells = np.ceil(dim_widths / self._resol)
        self._bins = [ranges[i, 0] + np.arange(num_cell) * self._resol for i, num_cell in enumerate(num_cells)]

    def cell_vol(self) -> float:
        """Return the volume for one cell."""
        return self._cell_resol ** 3

    def num_cells(self) -> int:
        """Return the number of cells in the grid."""
        return np.prod(self.num_bins())

    def num_bins(self) -> Iterable[int]:
        """Return the number of bins along each dimensino."""
        return [len(bins) for bins in self._bins]

    def r3_to_index(self, point: np.ndarray):
        """Convert point in R3 to index.

        Args:
            point: coordinate in R3, shape (N, 3)

        Returns:
            index of cell in the grid, shape (N,)
        """
        indices = np.array(
            [
                np.digitize(point_dim, bins) - 1
                for point_dim, bins in zip(point.T, self._bins)
            ]
        ).T
        indices[indices < 0] = 0

        idcs = (
            indices[:, 2] * len(self._bins[0]) * len(self._bins[1])
            + indices[:, 1] * len(self._bins[0])
            + indices[:, 0]
        )
        return idcs

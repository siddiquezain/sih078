"""Earned-resolution engine.

Earned width = smallest window n (grid cells) with FSS >= useful_skill_line.
This is a single-event point estimate. Always label it that way.
"""

from __future__ import annotations
from typing import Optional
import numpy as np
from resolve.fss import useful_skill_line


def earned_width(
    fss_values: np.ndarray,
    windows_cells: list[int],
    f_obs: float,
) -> Optional[int]:
    """Find smallest window achieving useful skill.

    Returns None ("no skill") if no window clears the threshold.
    """
    threshold = useful_skill_line(f_obs)
    for n, fss in zip(windows_cells, fss_values):
        if np.isfinite(fss) and fss >= threshold:
            return n
    return None


def cells_to_km(n_cells: int, cell_deg: float = 0.25) -> float:
    """Convert window size in grid cells to approximate km."""
    return n_cells * cell_deg * 111.32

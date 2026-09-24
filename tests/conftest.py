"""Shared pytest fixtures for Resolve tests."""
import numpy as np
import pytest


@pytest.fixture
def tiny_grid():
    """5×5 lat/lon grid (degrees) for unit tests."""
    lat = np.linspace(10.0, 14.0, 5)
    lon = np.linspace(76.0, 80.0, 5)
    return lat, lon


@pytest.fixture
def land_mask_5x5():
    """All-land 5×5 mask (all True) for unit tests."""
    return np.ones((5, 5), dtype=bool)

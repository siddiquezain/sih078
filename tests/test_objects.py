import numpy as np
import pytest
from resolve.objects import (
    find_objects,
    rain_weighted_centroid,
    area_km2,
    RainObject,
)


def test_find_objects_single_blob():
    """Single connected blob → one object."""
    mask = np.zeros((8, 8), dtype=bool)
    mask[2:5, 2:5] = True  # 3×3 blob
    rain = mask.astype(float) * 10.0
    lat = np.linspace(10, 11.75, 8)
    lon = np.linspace(76, 77.75, 8)
    objs = find_objects(mask, rain, lat, lon, min_area_cells=1)
    assert len(objs) == 1


def test_find_objects_two_blobs():
    """Two disconnected blobs → two objects."""
    mask = np.zeros((10, 10), dtype=bool)
    mask[1:3, 1:3] = True
    mask[7:9, 7:9] = True
    rain = mask.astype(float) * 5.0
    lat = np.linspace(10, 12.25, 10)
    lon = np.linspace(76, 78.25, 10)
    objs = find_objects(mask, rain, lat, lon, min_area_cells=1)
    assert len(objs) == 2


def test_find_objects_min_area_filter():
    """Small blob below min_area_cells is dropped."""
    mask = np.zeros((10, 10), dtype=bool)
    mask[1:3, 1:3] = True   # 4 cells
    mask[7:9, 7:9] = True   # 4 cells
    rain = mask.astype(float)
    lat = np.linspace(10, 12.25, 10)
    lon = np.linspace(76, 78.25, 10)
    objs = find_objects(mask, rain, lat, lon, min_area_cells=5)
    assert len(objs) == 0


def test_rain_weighted_centroid_uniform():
    """Uniform rain: centroid is geometric centre of the blob."""
    mask = np.zeros((6, 6), dtype=bool)
    mask[2:4, 2:4] = True  # rows 2,3 cols 2,3
    rain = mask.astype(float)
    lat = np.array([10.0, 10.25, 10.5, 10.75, 11.0, 11.25])
    lon = np.array([76.0, 76.25, 76.5, 76.75, 77.0, 77.25])
    lat_c, lon_c = rain_weighted_centroid(rain, mask, lat, lon)
    np.testing.assert_allclose(lat_c, 10.625, atol=1e-3)
    np.testing.assert_allclose(lon_c, 76.625, atol=1e-3)


def test_area_km2_positive():
    """Area roughly correct for a 4×4 region at ~10°N."""
    mask = np.ones((4, 4), dtype=bool)
    lat = np.array([10.0, 10.25, 10.5, 10.75])
    area = area_km2(mask, lat, cell_deg=0.25)
    # 16 cells at 10°N: each ≈ (27.83)^2 * cos(10°) ≈ 764 km² → ~12,200 km²
    assert 10_000 < area < 15_000, f"Unexpected area: {area:.0f} km²"


def test_find_objects_sorted_by_sum_prob():
    """Objects are sorted by sum_prob descending, not area."""
    mask = np.zeros((10, 10), dtype=bool)
    rain = np.zeros((10, 10), dtype=float)
    # Blob A: 2 cells, high rain → high sum_prob
    mask[1:3, 1:3] = True
    rain[1:3, 1:3] = 200.0  # sum_prob = 4 × 200 = 800
    # Blob B: 9 cells, low rain → lower sum_prob
    mask[5:8, 5:8] = True
    rain[5:8, 5:8] = 50.0   # sum_prob = 9 × 50 = 450
    lat = np.linspace(10, 12.25, 10)
    lon = np.linspace(76, 78.25, 10)
    objs = find_objects(mask, rain, lat, lon, min_area_cells=1)
    assert len(objs) == 2
    # First object should have the highest sum_prob (blob A)
    assert objs[0].sum_prob > objs[1].sum_prob
    assert objs[0].n_cells == 4  # 2×2 blob


def test_find_objects_cells_field():
    """RainObject.cells contains flat indices covering the blob."""
    mask = np.zeros((6, 6), dtype=bool)
    mask[2:4, 2:4] = True  # 4 cells
    rain = mask.astype(float)
    lat = np.linspace(10, 11.25, 6)
    lon = np.linspace(76, 77.25, 6)
    objs = find_objects(mask, rain, lat, lon, min_area_cells=1)
    assert len(objs) == 1
    assert objs[0].cells is not None
    assert len(objs[0].cells) == 4
    # Verify all cells are within [0, 36)
    assert all(0 <= c < 36 for c in objs[0].cells)

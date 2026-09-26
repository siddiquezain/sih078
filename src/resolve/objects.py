"""8-connected component labelling and object feature extraction."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.ndimage import label as _label

_STRUCT8 = np.ones((3, 3), dtype=int)
EARTH_KM_PER_DEG = 111.32


@dataclass
class RainObject:
    label_id: int
    centroid_lat: float
    centroid_lon: float
    area_km2: float
    max_rain_mm: float
    n_cells: int
    bbox: tuple[int, int, int, int]  # row_min, row_max, col_min, col_max
    member: Optional[int] = None
    cells: Optional[list] = None   # flat indices into the (nlat, nlon) grid
    sum_prob: float = 0.0           # sum of rain/prob values over the object cells


def rain_weighted_centroid(
    rain: np.ndarray,
    mask: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
) -> tuple[float, float]:
    """Compute rain-weighted centroid. Falls back to geometric if rain is zero."""
    lon2d, lat2d = np.meshgrid(lon, lat)
    w = rain * mask
    total = w.sum()
    if total == 0.0:
        total = mask.sum()
        lat_c = float((lat2d * mask).sum() / total)
        lon_c = float((lon2d * mask).sum() / total)
    else:
        lat_c = float((lat2d * w).sum() / total)
        lon_c = float((lon2d * w).sum() / total)
    return lat_c, lon_c


def area_km2(
    mask: np.ndarray,
    lat: np.ndarray,
    cell_deg: float = 0.25,
) -> float:
    """Total area of mask=True cells with cos-lat correction."""
    cell_side = cell_deg * EARTH_KM_PER_DEG
    cos_lat = np.cos(np.radians(lat))
    cell_areas = cell_side ** 2 * cos_lat  # (n_lat,)
    return float((mask * cell_areas[:, np.newaxis]).sum())


def find_objects(
    binary_mask: np.ndarray,
    rain: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    min_area_cells: int = 5,
    cell_deg: float = 0.25,
    member: Optional[int] = None,
) -> list[RainObject]:
    """Label 8-connected components and extract features.

    Returns list of RainObject sorted by area_km2 descending.
    """
    labeled, n_features = _label(binary_mask, structure=_STRUCT8)
    if n_features == 0:
        return []

    nlon_grid = binary_mask.shape[1]
    objects = []
    for lbl in range(1, n_features + 1):
        obj_mask = labeled == lbl
        n_cells = int(obj_mask.sum())
        if n_cells < min_area_cells:
            continue

        rows, cols = np.where(obj_mask)
        bbox = (int(rows.min()), int(rows.max()), int(cols.min()), int(cols.max()))
        lat_c, lon_c = rain_weighted_centroid(rain, obj_mask, lat, lon)
        akm2 = area_km2(obj_mask, lat, cell_deg=cell_deg)
        max_rain = float(rain[obj_mask].max()) if n_cells > 0 else 0.0
        cells = [int(r * nlon_grid + c) for r, c in zip(rows.tolist(), cols.tolist())]
        sum_prob = float(rain[obj_mask].sum())

        objects.append(RainObject(
            label_id=lbl,
            centroid_lat=lat_c,
            centroid_lon=lon_c,
            area_km2=akm2,
            max_rain_mm=max_rain,
            n_cells=n_cells,
            bbox=bbox,
            member=member,
            cells=cells,
            sum_prob=sum_prob,
        ))

    objects.sort(key=lambda o: o.sum_prob, reverse=True)
    return objects

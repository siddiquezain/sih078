"""Run-to-run object matching using the Hungarian algorithm.

Cost = 1 − IoU, gated by centroid distance.
Lock-on = first run where drift ≤ threshold AND IoU ≥ threshold for all subsequent runs.
"""

from __future__ import annotations
import math
from typing import Optional

import numpy as np
from scipy.optimize import linear_sum_assignment

from resolve.objects import RainObject


def haversine_km(p1: tuple[float, float], p2: tuple[float, float]) -> float:
    """Great-circle distance (km) between two (lat, lon) points (degrees)."""
    R = 6371.0
    lat1, lon1 = math.radians(p1[0]), math.radians(p1[1])
    lat2, lon2 = math.radians(p2[0]), math.radians(p2[1])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2.0 * R * math.asin(math.sqrt(a))


def iou_masks(m1: np.ndarray, m2: np.ndarray) -> float:
    """IoU of two boolean masks of the same shape."""
    inter = np.logical_and(m1, m2).sum()
    union = np.logical_or(m1, m2).sum()
    return float(inter / union) if union > 0 else 0.0


def _bbox_iou(bbox_a: tuple, bbox_b: tuple) -> float:
    """Approximate IoU from bounding boxes (row_min, row_max, col_min, col_max)."""
    r0a, r1a, c0a, c1a = bbox_a
    r0b, r1b, c0b, c1b = bbox_b
    row_inter = max(0, min(r1a, r1b) - max(r0a, r0b) + 1)
    col_inter = max(0, min(c1a, c1b) - max(c0a, c0b) + 1)
    inter = row_inter * col_inter
    area_a = (r1a - r0a + 1) * (c1a - c0a + 1)
    area_b = (r1b - r0b + 1) * (c1b - c0b + 1)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def match_objects(
    objs_a: list[RainObject],
    objs_b: list[RainObject],
    masks_a: Optional[list[np.ndarray]],
    masks_b: Optional[list[np.ndarray]],
    max_centroid_km: float = 500.0,
) -> list[tuple[int, int]]:
    """Hungarian matching between two lists of RainObjects.

    Cost = 1 − IoU; pairs beyond max_centroid_km get cost = 1 (not matched).
    Returns list of (i, j) index pairs.
    """
    na, nb = len(objs_a), len(objs_b)
    if na == 0 or nb == 0:
        return []

    cost = np.ones((na, nb), dtype=float)
    for i, a in enumerate(objs_a):
        for j, b in enumerate(objs_b):
            d = haversine_km(
                (a.centroid_lat, a.centroid_lon),
                (b.centroid_lat, b.centroid_lon),
            )
            if d > max_centroid_km:
                continue
            if masks_a is not None and masks_b is not None:
                iou = iou_masks(masks_a[i], masks_b[j])
            else:
                iou = _bbox_iou(a.bbox, b.bbox)
            cost[i, j] = 1.0 - iou

    row_idx, col_idx = linear_sum_assignment(cost)
    return [
        (int(r), int(c))
        for r, c in zip(row_idx, col_idx)
        if cost[r, c] < 1.0
    ]


def detect_lockon(
    drifts_km: list[float],
    ious: list[float],
    drift_threshold_km: float = 100.0,
    iou_threshold: float = 0.5,
) -> Optional[int]:
    """Find first run index where lock-on is established.

    Lock-on = first index k such that for all m >= k:
      drifts_km[m] <= drift_threshold_km AND ious[m] >= iou_threshold.

    Returns None if never achieved.
    """
    n = len(drifts_km)
    for k in range(n):
        if all(
            drifts_km[m] <= drift_threshold_km and ious[m] >= iou_threshold
            for m in range(k, n)
        ):
            return k
    return None

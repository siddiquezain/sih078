import numpy as np
import pytest
from resolve.crossrun import haversine_km, iou_masks, match_objects, detect_lockon
from resolve.objects import RainObject


def _obj(lat, lon, area=1000.0, max_rain=200.0, n_cells=20,
         bbox=(0, 4, 0, 4), label=1, member=None):
    return RainObject(
        label_id=label,
        centroid_lat=lat,
        centroid_lon=lon,
        area_km2=area,
        max_rain_mm=max_rain,
        n_cells=n_cells,
        bbox=bbox,
        member=member,
    )


def test_haversine_known():
    """Mumbai → Chennai ≈ 1028 km."""
    d = haversine_km((19.07, 72.88), (13.08, 80.27))
    assert 1000 < d < 1100, f"got {d:.0f} km"


def test_haversine_zero():
    np.testing.assert_allclose(haversine_km((10.0, 80.0), (10.0, 80.0)), 0.0, atol=0.01)


def test_iou_perfect_overlap():
    m = np.zeros((6, 6), dtype=bool)
    m[1:4, 1:4] = True
    np.testing.assert_allclose(iou_masks(m, m), 1.0)


def test_iou_no_overlap():
    m1 = np.zeros((6, 6), dtype=bool)
    m1[0:2, 0:2] = True
    m2 = np.zeros((6, 6), dtype=bool)
    m2[4:6, 4:6] = True
    np.testing.assert_allclose(iou_masks(m1, m2), 0.0)


def test_match_objects_simple():
    """Two nearby objects → one match."""
    a = [_obj(15.0, 82.0, label=1)]
    b = [_obj(15.1, 82.1, label=1)]  # ~16 km away
    matches = match_objects(a, b, masks_a=None, masks_b=None, max_centroid_km=500.0)
    assert len(matches) == 1
    assert matches[0] == (0, 0)


def test_match_objects_too_far():
    """Objects beyond max_centroid_km are not matched."""
    a = [_obj(15.0, 82.0)]
    b = [_obj(25.0, 92.0)]  # far away
    matches = match_objects(a, b, masks_a=None, masks_b=None, max_centroid_km=100.0)
    assert len(matches) == 0


def test_detect_lockon():
    """Lock-on at index 2 when drift≤100 and iou≥0.5 from index 2 onward."""
    drifts = [200.0, 150.0, 80.0, 60.0, 50.0]
    ious   = [0.3,   0.4,   0.6,  0.7,  0.8]
    idx = detect_lockon(drifts, ious, drift_threshold_km=100.0, iou_threshold=0.5)
    assert idx == 2


def test_detect_lockon_never():
    """No lock-on if condition drops back."""
    drifts = [200.0, 80.0, 200.0]
    ious   = [0.3,   0.7,  0.3]
    idx = detect_lockon(drifts, ious, drift_threshold_km=100.0, iou_threshold=0.5)
    assert idx is None

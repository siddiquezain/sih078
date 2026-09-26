"""Tests for export_ui: compute_nmep and _check_export."""

import math

import numpy as np
import pytest

from resolve.export_ui import compute_nmep, _check_export


# ---------------------------------------------------------------------------
# compute_nmep
# ---------------------------------------------------------------------------

def _masks(n_members, nlat, nlon, rows, cols):
    """Build boolean member_masks with member 0 having rain at (rows, cols)."""
    m = np.zeros((n_members, nlat, nlon), dtype=bool)
    for r, c in zip(rows, cols):
        m[0, r, c] = True
    return m


def test_nmep_single_member_single_cell():
    """One member, one cell: tile containing it has nmep=1, others 0."""
    m = _masks(4, 4, 4, [1], [1])
    result = compute_nmep(m, 4, 4, 2)  # 2×2 tiles → 2×2 = 4 tiles
    assert len(result) == 4
    assert result[0] == 1   # tile (0,0) contains row 1, col 1
    assert result[1] == 0
    assert result[2] == 0
    assert result[3] == 0


def test_nmep_all_members_same_tile():
    """All 4 members have rain in tile (0,0) → nmep=4 for that tile."""
    m = np.zeros((4, 4, 4), dtype=bool)
    m[:, 0, 0] = True   # all members, cell (0,0)
    result = compute_nmep(m, 4, 4, 2)
    assert result[0] == 4
    assert result[1] == 0


def test_nmep_max_ge_max_counts():
    """max(nmep) >= max(counts) for any tile size."""
    rng = np.random.default_rng(42)
    n_members, nlat, nlon = 10, 8, 8
    m = rng.random((n_members, nlat, nlon)) > 0.7
    counts = m.sum(axis=0).astype(int)
    for n in [2, 3, 4]:
        nmep = compute_nmep(m, nlat, nlon, n)
        assert max(nmep) >= int(counts.max()), (
            f"nmep max {max(nmep)} < counts max {counts.max()} at n={n}"
        )


def test_nmep_length():
    """NMEP array length = ceil(nlat/n) * ceil(nlon/n)."""
    m = np.zeros((5, 7, 9), dtype=bool)
    for n in [2, 3, 5]:
        result = compute_nmep(m, 7, 9, n)
        expected = math.ceil(7 / n) * math.ceil(9 / n)
        assert len(result) == expected, f"n={n}: got {len(result)}, expected {expected}"


# ---------------------------------------------------------------------------
# _check_export
# ---------------------------------------------------------------------------

def _minimal_data(nlat=4, nlon=4, n_members=10):
    """Build a minimal valid export data structure."""
    n_cells = nlat * nlon
    windows = [{"cells": 1, "km": 28}, {"cells": 3, "km": 83}]
    counts = [2] * n_cells
    nmep_3 = compute_nmep(
        (np.arange(n_members)[:, None, None] < 2) * np.ones((n_members, nlat, nlon), dtype=bool),
        nlat, nlon, 3,
    )
    return {
        "schema": "resolve-ui/1",
        "meta": {"preview": False, "label": "replay"},
        "grid": {"lat0": 8.0, "lon0": 74.0, "step": 0.25, "nlat": nlat, "nlon": nlon},
        "land": [1] * n_cells,
        "observed": {"cells": [0], "n": 1, "f_obs": 0.0625, "max_mm": 200.0},
        "windows": windows,
        "runs": [{
            "init": "2025-10-25T00:00Z",
            "lead_days": 3,
            "lead_hours": 75,
            "window_offset_hours": 0,
            "members_valid": n_members,
            "peak_count": 2,
            "counts": counts,
            "nmep": {"3": nmep_3},
            "fss": [0.4, 0.7],
            "useful": 0.55,
            "earned_cells": 3,
            "object": None,
            "drift_km": None,
            "iou": None,
            "locked": False,
        }],
        "lockon_init": None,
        "track": [],
    }


def test_check_export_valid():
    """Valid data passes all checks."""
    _check_export(_minimal_data(), n_members=10)


def test_check_export_preview_must_be_false():
    data = _minimal_data()
    data["meta"]["preview"] = True
    with pytest.raises(AssertionError, match="preview"):
        _check_export(data, n_members=10)


def test_check_export_counts_length():
    data = _minimal_data(nlat=4, nlon=4)
    data["runs"][0]["counts"] = [2] * 10  # wrong length (should be 16)
    with pytest.raises(AssertionError, match="counts length"):
        _check_export(data, n_members=10)


def test_check_export_peak_count_mismatch():
    data = _minimal_data()
    data["runs"][0]["peak_count"] = 99
    with pytest.raises(AssertionError, match="peak_count"):
        _check_export(data, n_members=10)

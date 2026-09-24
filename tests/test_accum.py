# tests/test_accum.py
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from resolve.accum import rate_to_mm, accumulate_window, window_bounds_utc


def _make_ds(rate_value=1e-4, lat=None, lon=None, members=3):
    """Tiny IFS-like dataset with REAL schema:
    - lead_time is timedelta64 (not absolute timestamps)
    - ensemble dim is 'member' (already renamed, as ingest does)
    """
    if lat is None:
        lat = [10.0, 10.25]
    if lon is None:
        lon = [80.0, 80.25]
    # 3-hourly steps matching real IFS spacing; include 27h so the 24h
    # window (init+3h, init+27h] is fully covered (86400 s)
    lead_times = pd.to_timedelta([0, 3, 6, 9, 12, 24, 27], unit="h")
    rate = np.full((len(lead_times), members, len(lat), len(lon)), rate_value)
    ds = xr.Dataset(
        {"precipitation_surface": (
            ["lead_time", "member", "latitude", "longitude"], rate,
        )},
        coords={
            "lead_time": lead_times,
            "member": np.arange(members),
            "latitude": lat,
            "longitude": lon,
        },
    )
    return ds


def test_rate_to_mm_uniform_step():
    """1 mm/h for 3 h = 3 mm."""
    np.testing.assert_allclose(rate_to_mm(1.0 / 3600.0, 3 * 3600), 3.0, rtol=1e-6)


def test_rate_to_mm_non_negative():
    assert rate_to_mm(-1e-5, 3600) >= 0.0


def test_accumulate_window_24h():
    """Steps in (t0, t1] each contribute rate × Δt mm."""
    init = pd.Timestamp("2025-10-25T00:00:00")
    ds = _make_ds(rate_value=1.0 / 3600.0)  # 1 mm/h
    t0 = pd.Timestamp("2025-10-25T03:00:00")
    t1 = pd.Timestamp("2025-10-26T03:00:00")
    acc = accumulate_window(ds, init, t0, t1)
    # Steps in (t0, t1]: valid_times 06 h, 09 h, 12 h, 24 h, 27 h
    # Δt: 3h, 3h, 3h, 12h, 3h → 24 h at 1 mm/h = 24 mm
    assert acc.dims == ("member", "latitude", "longitude")
    np.testing.assert_allclose(acc.values, 24.0, rtol=1e-5)


def test_accumulate_window_empty():
    """Window with no steps returns zero."""
    init = pd.Timestamp("2025-10-25T00:00:00")
    ds = _make_ds(rate_value=1e-4)
    t0 = pd.Timestamp("2025-10-24T00:00:00")
    t1 = pd.Timestamp("2025-10-24T01:00:00")
    acc = accumulate_window(ds, init, t0, t1)
    np.testing.assert_allclose(acc.values, 0.0)


def test_leakage_guard():
    """Result must not depend on data outside the window."""
    init = pd.Timestamp("2025-10-25T00:00:00")
    ds = _make_ds(rate_value=1.0 / 3600.0)
    t0 = pd.Timestamp("2025-10-25T03:00:00")
    t1 = pd.Timestamp("2025-10-26T03:00:00")
    ref = accumulate_window(ds, init, t0, t1).values.copy()

    ds2 = ds.copy(deep=True)
    ds2["precipitation_surface"].values[:] = np.nan
    lead_td = pd.to_timedelta(ds2.lead_time.values)
    valid_times = pd.DatetimeIndex([init + td for td in lead_td])
    in_win = (valid_times > t0) & (valid_times <= t1)
    ds2["precipitation_surface"].values[in_win] = 1.0 / 3600.0
    result = accumulate_window(ds2, init, t0, t1).values
    np.testing.assert_allclose(result, ref, rtol=1e-5)
    # Both should equal 24 mm (24 h of steps at 1 mm/h)
    np.testing.assert_allclose(result, 24.0, rtol=1e-5)


def test_window_bounds_utc_ending():
    """Convention='ending': window ends 03 UTC on target date."""
    t0, t1 = window_bounds_utc("2025-10-28", convention="ending")
    assert t1 == pd.Timestamp("2025-10-28T03:00:00")
    assert t0 == pd.Timestamp("2025-10-27T03:00:00")


def test_window_bounds_utc_ending_offset():
    """offset_hours=-3 → window ends 00 UTC (for leads > 144 h)."""
    t0, t1 = window_bounds_utc("2025-10-28", convention="ending", offset_hours=-3)
    assert t1 == pd.Timestamp("2025-10-28T00:00:00")
    assert t0 == pd.Timestamp("2025-10-27T00:00:00")


def test_sanity_gate_86400s():
    """Window that doesn't sum to 86400 s raises ValueError."""
    # Build a dataset with a gap: skip the 12h step → only 18h coverage
    lead_times = pd.to_timedelta([0, 3, 6, 9, 24], unit="h")  # no 12h step → gap
    rate = np.full((len(lead_times), 2, 2, 2), 1.0 / 3600.0)
    ds = xr.Dataset(
        {"precipitation_surface": (["lead_time", "member", "latitude", "longitude"], rate)},
        coords={"lead_time": lead_times, "member": [0, 1],
                "latitude": [10.0, 10.25], "longitude": [80.0, 80.25]},
    )
    init = pd.Timestamp("2025-10-25T00:00:00")
    t0 = pd.Timestamp("2025-10-25T03:00:00")
    t1 = pd.Timestamp("2025-10-26T03:00:00")
    with pytest.raises(ValueError, match="86400"):
        accumulate_window(ds, init, t0, t1)


def test_sanity_gate_negative_rain():
    """Negative accumulated rain raises ValueError."""
    # Include 27h so window coverage = 86400 s; gate checks rain sign, not coverage
    lead_times = pd.to_timedelta([0, 3, 6, 9, 12, 24, 27], unit="h")
    rate = np.full((len(lead_times), 2, 2, 2), -1.0)  # negative rate
    ds = xr.Dataset(
        {"precipitation_surface": (["lead_time", "member", "latitude", "longitude"], rate)},
        coords={"lead_time": lead_times, "member": [0, 1],
                "latitude": [10.0, 10.25], "longitude": [80.0, 80.25]},
    )
    init = pd.Timestamp("2025-10-25T00:00:00")
    t0 = pd.Timestamp("2025-10-25T03:00:00")
    t1 = pd.Timestamp("2025-10-26T03:00:00")
    with pytest.raises(ValueError, match="negative"):
        accumulate_window(ds, init, t0, t1)

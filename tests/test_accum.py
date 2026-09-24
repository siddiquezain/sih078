# tests/test_accum.py
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from resolve.accum import rate_to_mm, accumulate_window, window_bounds_utc


def _make_ds(rate_value=1e-4, lat=None, lon=None, members=3):
    """Tiny synthetic IFS-like dataset."""
    if lat is None:
        lat = [10.0, 10.25]
    if lon is None:
        lon = [80.0, 80.25]
    steps = pd.to_timedelta([0, 3, 6, 9, 12, 24], unit="h")
    init = pd.Timestamp("2025-10-25T00:00:00")
    times = [init + s for s in steps]
    rate = np.full((len(steps), members, len(lat), len(lon)), rate_value)
    ds = xr.Dataset(
        {"precipitation_surface": (
            ["lead_time", "member", "latitude", "longitude"], rate
        )},
        coords={
            "lead_time": times,
            "member": np.arange(members),
            "latitude": lat,
            "longitude": lon,
        },
    )
    return ds, init


def test_rate_to_mm_uniform_step():
    """rate × Δt = mm; 1 mm/h for 3 hours = 3 mm."""
    rate = 1.0 / 3600.0  # kg/m²/s
    dt = 3 * 3600  # seconds
    result = rate_to_mm(rate, dt)
    np.testing.assert_allclose(result, 3.0, rtol=1e-6)


def test_rate_to_mm_non_negative():
    """Accumulated rain is never negative."""
    result = rate_to_mm(-1e-5, 3600)
    assert result >= 0.0


def test_accumulate_window_24h():
    """24 h window: steps inside (t0, t1] are included."""
    ds, init = _make_ds(rate_value=1.0 / 3600.0)  # 1 mm/h
    t0 = pd.Timestamp("2025-10-25T03:00:00")
    t1 = pd.Timestamp("2025-10-26T03:00:00")
    acc = accumulate_window(ds, t0, t1)
    # Steps with lead_time in (t0, t1]: 6h, 9h, 12h, 24h
    # Δt: 3h, 3h, 3h, 12h → 21 h total at 1 mm/h = 21 mm
    assert acc.dims == ("member", "latitude", "longitude")
    np.testing.assert_allclose(acc.values, 21.0, rtol=1e-5)


def test_accumulate_window_all_steps_zero():
    """Window with no steps returns zero rain."""
    ds, init = _make_ds(rate_value=1e-4)
    t0 = pd.Timestamp("2025-10-24T00:00:00")
    t1 = pd.Timestamp("2025-10-24T01:00:00")
    acc = accumulate_window(ds, t0, t1)
    np.testing.assert_allclose(acc.values, 0.0)


def test_leakage_guard():
    """Result must not depend on data outside the window."""
    ds, init = _make_ds(rate_value=1.0 / 3600.0)
    t0 = pd.Timestamp("2025-10-25T03:00:00")
    t1 = pd.Timestamp("2025-10-26T03:00:00")
    ref = accumulate_window(ds, t0, t1).values.copy()
    # NaN all steps, restore only the ones inside the window
    ds2 = ds.copy(deep=True)
    ds2["precipitation_surface"].values[:] = np.nan
    times = pd.DatetimeIndex(ds2.lead_time.values)
    mask = (times > t0) & (times <= t1)
    ds2["precipitation_surface"].values[mask] = 1.0 / 3600.0
    result = accumulate_window(ds2, t0, t1).values
    np.testing.assert_allclose(result, ref, rtol=1e-5)


def test_window_bounds_utc_offset_zero():
    """24 h ending 03 UTC on target date."""
    t0, t1 = window_bounds_utc("2025-10-28", offset_hours=0)
    assert t1 == pd.Timestamp("2025-10-28T03:00:00")
    assert t0 == pd.Timestamp("2025-10-27T03:00:00")


def test_window_bounds_utc_offset_minus3():
    """With offset -3: window ends 00 UTC (for leads > 144 h)."""
    t0, t1 = window_bounds_utc("2025-10-28", offset_hours=-3)
    assert t1 == pd.Timestamp("2025-10-28T00:00:00")
    assert t0 == pd.Timestamp("2025-10-27T00:00:00")

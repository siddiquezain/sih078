"""Rain accumulation: IFS ENS precipitation_surface rate → mm per window.

IFS precipitation_surface: mean rate since the previous step, kg m⁻² s⁻¹ = mm/s.
Rain for step k = rate_k × (t_k − t_{k−1}) seconds.
Include step k if t_k is in (t0, t1] (exclusive t0, inclusive t1).
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import xarray as xr


def rate_to_mm(rate: float, dt_seconds: float) -> float:
    """Convert rate (kg m⁻² s⁻¹) × Δt (s) to mm. Clamps to zero."""
    return float(max(0.0, rate * dt_seconds))


def accumulate_window(
    ds: xr.Dataset,
    t0: pd.Timestamp,
    t1: pd.Timestamp,
    rate_var: str = "precipitation_surface",
) -> xr.DataArray:
    """Accumulate member rain (mm) over the half-open window (t0, t1].

    Args:
        ds: Dataset with dims (lead_time, member, latitude, longitude).
            lead_time values are absolute timestamps.
        t0: Window start (exclusive).
        t1: Window end (inclusive).
        rate_var: Name of the rate variable.

    Returns:
        DataArray with dims (member, latitude, longitude), units mm.
        All-zero if no steps fall inside the window.
    """
    times = pd.DatetimeIndex(ds.lead_time.values)
    in_window = (times > t0) & (times <= t1)

    if not in_window.any():
        da = ds[rate_var].isel(lead_time=0) * 0.0
        return da.drop_vars("lead_time", errors="ignore")

    selected_indices = np.where(in_window)[0]
    accum = None

    for idx in selected_indices:
        t_k = times[idx]
        if idx == 0:
            # No predecessor → Δt undefined. This only fires if lead_time[0]
            # (= init time, a zero-length interval) falls inside the window,
            # which cannot happen for windows starting at 03 UTC D−1 (all
            # lead_time[0] = init time = 00 UTC D is excluded by t > t0).
            continue
        t_prev = times[idx - 1]
        dt_seconds = (t_k - t_prev).total_seconds()
        rate = ds[rate_var].isel(lead_time=idx)  # (member, lat, lon)
        rain_mm = rate * dt_seconds
        rain_mm = rain_mm.clip(min=0.0)
        if accum is None:
            accum = rain_mm
        else:
            accum = accum + rain_mm

    if accum is None:
        da = ds[rate_var].isel(lead_time=0) * 0.0
        return da.drop_vars("lead_time", errors="ignore")

    return accum.drop_vars("lead_time", errors="ignore")


def window_bounds_utc(
    target_date_str: str,
    offset_hours: int = 0,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Return (t0, t1) for a 24-hour IMD rain day ending 03 UTC.

    IMD rain day D: ends at 03 UTC on date D.
    Window = (D-1 03 UTC, D 03 UTC].

    For lead > 144 h (6-hourly), use offset_hours = -3 → window ends 00 UTC.
    """
    d = pd.Timestamp(target_date_str)
    t1 = d + pd.Timedelta(hours=3 + offset_hours)
    t0 = t1 - pd.Timedelta(hours=24)
    return t0, t1

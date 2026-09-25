"""Rain accumulation: IFS ENS precipitation_surface rate → mm per window.

IFS precipitation_surface: mean rate since the previous step, kg m⁻² s⁻¹ = mm/s.
lead_time is timedelta64 in the real dataset; pass init_time to get absolute times.
Rain for step k = rate_k × (t_k − t_{k−1}) seconds.
Include step k if valid_time_k is in (t0, t1] (exclusive t0, inclusive t1).
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
    init_time: pd.Timestamp,
    t0: pd.Timestamp,
    t1: pd.Timestamp,
    rate_var: str = "precipitation_surface",
) -> xr.DataArray:
    """Accumulate member rain (mm) over the half-open window (t0, t1].

    Args:
        ds: Dataset with lead_time as timedelta64 and dim 'member'.
        init_time: Forecast initialisation time (00 UTC).
        t0: Window start (exclusive).
        t1: Window end (inclusive).
        rate_var: Variable name for precipitation rate.

    Returns:
        DataArray (member, latitude, longitude), mm.

    Raises:
        ValueError: If steps inside the window don't sum to exactly 86 400 s,
                    or if any cell has negative accumulated rain.
    """
    lead_td = pd.to_timedelta(ds.lead_time.values)
    valid_times = pd.DatetimeIndex([init_time + td for td in lead_td])
    in_window = (valid_times > t0) & (valid_times <= t1)

    if not in_window.any():
        da = ds[rate_var].isel(lead_time=0) * 0.0
        return da.drop_vars("lead_time", errors="ignore")

    # Sanity gate: window must be covered by exactly 86 400 s of steps
    total_seconds = 0.0
    for idx in np.where(in_window)[0]:
        if idx == 0:
            # lead_time[0] = 0h → zero-length interval; cannot be in window
            # (window start t0 ≥ init_time, and t > t0 is strict)
            continue
        dt = (valid_times[idx] - valid_times[idx - 1]).total_seconds()
        total_seconds += dt

    if abs(total_seconds - 86400.0) > 1.0:
        raise ValueError(
            f"Window ({t0}, {t1}] sums to {total_seconds:.0f} s of forecast steps; "
            f"expected 86400 s. Check lead_time coverage for init {init_time.isoformat()}."
        )

    accum = None
    for idx in np.where(in_window)[0]:
        if idx == 0:
            continue
        dt_seconds = (valid_times[idx] - valid_times[idx - 1]).total_seconds()
        rate = ds[rate_var].isel(lead_time=idx)  # (member, lat, lon)
        rain_mm = rate * dt_seconds
        if accum is None:
            accum = rain_mm
        else:
            accum = accum + rain_mm

    if accum is None:
        da = ds[rate_var].isel(lead_time=0) * 0.0
        return da.drop_vars("lead_time", errors="ignore")

    # Sanity gate: rain must be ≥ 0 everywhere
    accum_vals = accum.values
    if np.any(accum_vals < 0):
        neg_min = float(accum_vals.min())
        raise ValueError(
            f"Accumulated rain has negative values (min={neg_min:.4f} mm). "
            "Check precipitation_surface data."
        )

    # Sanity gate: rain must be < 2000 mm (above world 24 h record of 1825 mm).
    # IFS ENS extreme members for landfalling cyclones can exceed 1000 mm, so
    # the threshold is set above the physical maximum to catch unit errors only
    # (e.g. mm/hr rates not converted to mm, which would give >86 000 mm).
    # ponytail: 2000 mm threshold; tighten if non-cyclone events are ever mislabelled
    if np.any(accum_vals >= 2000.0):
        pos_max = float(accum_vals.max())
        raise ValueError(
            f"Accumulated rain exceeds 2000 mm (max={pos_max:.1f} mm). "
            "Check units — precipitation_surface must be in kg m⁻² s⁻¹."
        )

    return accum.drop_vars("lead_time", errors="ignore")


def window_bounds_utc(
    target_date_str: str,
    convention: str = "ending",
    offset_hours: int = 0,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Return (t0, t1) for a 24-hour IMD rain day.

    Args:
        target_date_str: ISO date string (e.g. '2025-10-28').
        convention: 'ending' — IMD day D ends at 03 UTC on D (default).
                    'starting' — IMD day D starts at 00 UTC on D (unusual).
        offset_hours: Applied to t1; use -3 for leads > 144 h where the nearest
                      6-hourly step ends at 00 UTC instead of 03 UTC.

    Returns:
        (t0, t1): Window is (t0, t1] (exclusive start, inclusive end).
    """
    d = pd.Timestamp(target_date_str)
    if convention == "ending":
        t1 = d + pd.Timedelta(hours=3 + offset_hours)
    elif convention == "starting":
        t1 = d + pd.Timedelta(hours=24 + offset_hours)
    else:
        raise ValueError(f"convention must be 'ending' or 'starting', got {convention!r}")
    t0 = t1 - pd.Timedelta(hours=24)
    return t0, t1

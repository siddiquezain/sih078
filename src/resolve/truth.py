"""Truth data: IMD 0.25° daily rainfall and GPM IMERG V07.

IMD rain day D: 24 h ending 08:30 IST (03 UTC) on date D.
  imd_day_convention = 'ending'

GPM_3IMERGDF V07 'precipitation' variable: already mm/day (daily accumulation).
  IMERG day D: 00 UTC D to 00 UTC D+1 — 3 h earlier than IMD day D.

Alignment logic:
  IMD day D covers (03 UTC D−1, 03 UTC D].
  IMERG day D−1 covers (00 UTC D−1, 00 UTC D) → 21 h overlap with IMD day D.
  IMERG day D   covers (00 UTC D,   00 UTC D+1) → 3 h overlap with IMD day D.
  Empirically, IMD day D correlates better with IMERG day D−1.
  convention = 'ending' maps to this (IMERG_best = D−1).
"""

from __future__ import annotations
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import xarray as xr

logger = logging.getLogger(__name__)


def fetch_imd(
    start_year: int,
    end_year: int,
    file_dir: str | Path,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> xr.DataArray:
    """Fetch IMD 0.25° daily gridded rainfall via imdlib.

    Tries the yearwise archive first; if that raises, falls back to the
    real-time grid (requires start_date and end_date).

    Returns DataArray (time, latitude, longitude), mm/day. Land only; NaN over sea.
    """
    try:
        import imdlib
    except ImportError as e:
        raise ImportError("imdlib required for IMD truth. pip install imdlib") from e

    file_dir = str(file_dir)

    try:
        logger.info("Fetching IMD archive: %d–%d", start_year, end_year)
        data = imdlib.get_data(
            "rain", start_year, end_year,
            fn_format="yearwise",
            file_dir=file_dir,
        )
        logger.info("IMD archive fetched successfully")
    except Exception as archive_err:
        logger.warning("IMD archive failed (%s); trying real-time grid", archive_err)
        if start_date is None or end_date is None:
            raise RuntimeError(
                "IMD archive failed and no start_date/end_date provided for "
                "real-time fallback."
            ) from archive_err
        try:
            data = imdlib.get_real_data("rain", start_date, end_date, file_dir=file_dir)
            logger.info("IMD real-time grid fetched: %s to %s", start_date, end_date)
        except Exception as rt_err:
            raise RuntimeError(
                f"Both IMD archive ({archive_err}) and real-time ({rt_err}) failed."
            ) from rt_err

    result = data.get_xarray()
    # imdlib >= 0.1.22 returns a Dataset; extract the 'rain' DataArray.
    if hasattr(result, "data_vars"):
        da = result["rain"]
    else:
        da = result
    logger.info(
        "IMD shape: %s, %s to %s",
        da.shape, str(da.time.values[0])[:10], str(da.time.values[-1])[:10],
    )
    return da


def imd_subset(
    da: xr.DataArray,
    lat_min: float, lat_max: float,
    lon_min: float, lon_max: float,
    start_date: str, end_date: str,
) -> xr.DataArray:
    """Spatial and temporal subset of the IMD DataArray."""
    return da.sel(
        latitude=slice(lat_min, lat_max),
        longitude=slice(lon_min, lon_max),
        time=slice(start_date, end_date),
    )


def fetch_imerg_daily(
    start_date: str,
    end_date: str,
    cache_dir: str | Path,
    lat_min: float = 8.0, lat_max: float = 24.0,
    lon_min: float = 74.0, lon_max: float = 90.0,
) -> xr.DataArray:
    """Fetch GPM IMERG V07 daily Final (GPM_3IMERGDF) via earthaccess.

    GPM_3IMERGDF V07 'precipitation' is a daily accumulation in mm/day.
    Do NOT multiply by 24.

    IMERG day D covers 00 UTC D to 00 UTC D+1, which is 3 h earlier than
    IMD day D (03 UTC D−1 to 03 UTC D). This 3 h offset is resolved in
    check_alignment.

    Returns DataArray (time, lat, lon), mm/day.

    Raises:
        RuntimeError: If any cell exceeds 1 000 mm/day (likely a units error).
    """
    try:
        import earthaccess
    except ImportError as e:
        raise ImportError("earthaccess required for IMERG. pip install earthaccess") from e

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # login() auto-detects ~/.netrc, EARTHDATA_USERNAME/PASSWORD env vars
    earthaccess.login()

    results = earthaccess.search_data(
        short_name="GPM_3IMERGDF",
        version="07",
        temporal=(start_date, end_date),
        bounding_box=(lon_min, lat_min, lon_max, lat_max),
    )
    logger.info("Found %d IMERG files for %s to %s", len(results), start_date, end_date)

    files = earthaccess.download(results, local_path=str(cache_dir))
    logger.info("Downloaded %d files", len(files))

    arrays = []
    for fpath in sorted(files):
        try:
            ds = xr.open_dataset(fpath, group="Grid", engine="netcdf4")
            precip = ds["precipitation"]

            # V07: precipitation is already mm/day (daily accumulation).
            units = precip.attrs.get("units", "unknown")
            logger.debug("IMERG precipitation units from file: %s", units)
            # Do not multiply by 24.
            daily = precip.squeeze()

            daily_max = float(daily.max())
            if daily_max > 1000.0:
                raise RuntimeError(
                    f"IMERG daily max = {daily_max:.0f} mm in {fpath}. "
                    f"Units attribute says '{units}'. "
                    "Expected mm/day with max < 1000. "
                    "Check whether the file format or units have changed."
                )

            # Subset spatially; IMERG uses 'lat'/'lon' coord names
            daily = daily.sel(
                lat=slice(lat_min, lat_max),
                lon=slice(lon_min, lon_max),
            )
            arrays.append(daily)
        except RuntimeError:
            raise  # propagate units errors
        except Exception as e:
            logger.warning("Could not open %s: %s", fpath, e)

    if not arrays:
        raise RuntimeError(f"No IMERG data loaded for {start_date} to {end_date}")

    da = xr.concat(arrays, dim="time").sortby("time")
    logger.info("IMERG shape: %s", da.shape)
    return da


def check_alignment(
    imd_da: xr.DataArray,
    imerg_da: xr.DataArray,
    expected_convention: str = "ending",
) -> dict:
    """Verify IMD rain-day alignment by correlating with IMERG.

    IMD day D covers (03 UTC D−1, 03 UTC D].
    IMERG day D covers (00 UTC D, 00 UTC D+1).

    We compare:
      (a) IMD day D vs IMERG day D   — 3 h overlap  → matches 'starting' convention
      (b) IMD day D vs IMERG day D−1 — 21 h overlap → matches 'ending' convention

    The better correlation determines the actual convention. We then check it
    against expected_convention and raise if they disagree.

    Steps:
      1. Transpose IMERG to (time, lat, lon).
      2. Regrid IMERG to the IMD 0.25° grid via nearest-neighbour interpolation.
      3. Correlate on common dates (land cells only).

    Returns dict with: corr_imd_vs_imerg_same, corr_imd_vs_imerg_prev,
    best_match ('D' or 'D-1'), inferred_convention ('ending' or 'starting'),
    window_offset_hours, n_days.

    Raises:
        ValueError: If inferred convention contradicts expected_convention.
    """
    # 1. Transpose IMERG so dims are (time, lat, lon)
    imerg_da = imerg_da.transpose("time", "lat", "lon")

    # 2. Regrid IMERG to IMD 0.25° grid
    imd_lat = imd_da.latitude.values
    imd_lon = imd_da.longitude.values
    imerg_on_imd = imerg_da.interp(
        lat=imd_lat,
        lon=imd_lon,
        method="nearest",
    )
    # Rename IMERG lat/lon to match IMD coordinate names
    imerg_on_imd = imerg_on_imd.rename({"lat": "latitude", "lon": "longitude"})

    # 3. Find common dates
    common_dates = pd.DatetimeIndex(
        np.intersect1d(imd_da.time.values, imerg_on_imd.time.values)
    )
    if len(common_dates) == 0:
        logger.warning("No common dates between IMD and IMERG for alignment check")
        return {
            "corr_imd_vs_imerg_same": np.nan,
            "corr_imd_vs_imerg_prev": np.nan,
            "best_match": "unknown",
            "inferred_convention": "unknown",
            "window_offset_hours": -3,
            "n_days": 0,
        }

    # Land mask: IMD is NaN over sea
    land_mask = np.isfinite(imd_da.sel(time=common_dates[0]).values)

    def _corr(imd_days, imerg_days):
        """Pearson r over land cells, pooling all days."""
        imd_vals = imd_da.sel(time=imd_days).values  # (n_days, lat, lon)
        imerg_vals = imerg_on_imd.sel(time=imerg_days).values
        # Apply land mask to each day
        flat_imd = imd_vals[:, land_mask].ravel()
        flat_imerg = imerg_vals[:, land_mask].ravel()
        valid = np.isfinite(flat_imd) & np.isfinite(flat_imerg)
        if valid.sum() < 10:
            return np.nan
        return float(np.corrcoef(flat_imd[valid], flat_imerg[valid])[0, 1])

    # (a) IMD day D vs IMERG day D (same calendar date)
    corr_same = _corr(common_dates, common_dates)

    # (b) IMD day D vs IMERG day D−1
    prev_dates = common_dates - pd.Timedelta(days=1)
    imerg_times = pd.DatetimeIndex(imerg_on_imd.time.values)
    valid_prev = prev_dates[prev_dates.isin(imerg_times)]
    if len(valid_prev) > 0:
        imd_for_prev = common_dates[prev_dates.isin(imerg_times)]
        corr_prev = _corr(imd_for_prev, valid_prev)
    else:
        corr_prev = np.nan

    # Determine best match
    if np.isnan(corr_prev) or (not np.isnan(corr_same) and corr_same >= corr_prev):
        best_match = "D"
        inferred = "starting"   # IMERG D ~ IMD D → day starts at 00 UTC
        window_offset = 0
    else:
        best_match = "D-1"
        inferred = "ending"     # IMERG D-1 ~ IMD D → day ends at 03 UTC
        window_offset = -3

    logger.info(
        "Alignment: corr_same=%.3f corr_prev=%.3f → best=%s, convention=%s",
        corr_same, corr_prev, best_match, inferred,
    )

    if inferred != "unknown" and inferred != expected_convention:
        raise ValueError(
            f"IMERG alignment check inferred imd_day_convention='{inferred}' "
            f"but config says '{expected_convention}'. "
            "Update imd_day_convention in the config YAML or investigate the data."
        )

    return {
        "corr_imd_vs_imerg_same": corr_same,
        "corr_imd_vs_imerg_prev": corr_prev,
        "best_match": best_match,
        "inferred_convention": inferred,
        "window_offset_hours": window_offset,
        "n_days": len(common_dates),
    }

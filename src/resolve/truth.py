"""Truth data: IMD 0.25° daily rainfall and GPM IMERG V07.

IMD rain day D: 24 h ending 08:30 IST (03 UTC) on date D.
IMERG day D: 00–00 UTC on date D — 3 h earlier than IMD day D.
Verify alignment empirically and log in provenance.
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
    use_realtime: bool = False,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> xr.DataArray:
    """Fetch IMD 0.25° daily gridded rainfall via imdlib.

    Returns DataArray (time, latitude, longitude), mm/day. Land only; NaN over sea.
    """
    try:
        import imdlib
    except ImportError as e:
        raise ImportError("imdlib is required for IMD truth. pip install imdlib") from e

    file_dir = str(file_dir)

    if use_realtime:
        if start_date is None or end_date is None:
            raise ValueError("start_date and end_date required for realtime fetch")
        logger.info("Fetching IMD real-time: %s to %s", start_date, end_date)
        data = imdlib.get_real_data("rain", start_date, end_date, file_dir=file_dir)
    else:
        logger.info("Fetching IMD archive: %d to %d", start_year, end_year)
        data = imdlib.get_data(
            "rain", start_year, end_year,
            fn_format="yearwise",
            file_dir=file_dir,
        )

    da = data.get_xarray()
    logger.info("IMD shape: %s, %s to %s",
                da.shape, str(da.time.values[0])[:10], str(da.time.values[-1])[:10])
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

    IMERG day D covers 00–00 UTC — 3 h earlier than IMD day D.
    Record window_offset_hours = -3 in provenance when comparing to IMD.

    Returns DataArray (time, lat, lon), mm/day.
    """
    try:
        import earthaccess
    except ImportError as e:
        raise ImportError("earthaccess required for IMERG. pip install earthaccess") from e

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    earthaccess.login(strategy="netrc")

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
            daily = (precip * 24.0).squeeze()  # mm/hr → mm/day
            # GPM_3IMERGDF V07: precipitation is in mm/hr; * 24 → mm/day.
            # Verify: if any cell exceeds 1500 mm/day, something is wrong.
            daily_max = float(daily.max())
            if daily_max > 1500:
                logger.warning(
                    "IMERG daily max = %.0f mm; expected < 1500 mm. "
                    "Check units in %s", daily_max, fpath
                )
            daily = daily.sel(
                lat=slice(lat_min, lat_max),
                lon=slice(lon_min, lon_max),
            )
            arrays.append(daily)
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
) -> dict:
    """Verify IMD rain-day alignment by correlating with IMERG.

    Correlates IMD day D against:
      (a) IMERG day D  (same calendar day, window_offset = -3 h)
      (b) IMERG day D-1 (shifted by one day, window_offset = -27 h)

    Returns dict with corr_same_day, corr_shifted, best_match, window_offset_hours, n_days.
    """
    common_dates = pd.DatetimeIndex(
        np.intersect1d(imd_da.time.values, imerg_da.time.values)
    )
    if len(common_dates) == 0:
        logger.warning("No common dates for alignment check")
        return {
            "corr_same_day": np.nan,
            "corr_shifted": np.nan,
            "best_match": "unknown",
            "window_offset_hours": 0,
            "n_days": 0,
        }

    imd_common = imd_da.sel(time=common_dates)
    imerg_same = imerg_da.sel(time=common_dates)

    flat_imd = imd_common.values.ravel()
    flat_imerg = imerg_same.values.ravel()
    valid = np.isfinite(flat_imd) & np.isfinite(flat_imerg)
    corr_same = float(np.corrcoef(flat_imd[valid], flat_imerg[valid])[0, 1]) if valid.sum() > 10 else np.nan

    # Shifted: IMERG D corresponds to IMD D+1
    shifted_dates = common_dates - pd.Timedelta(days=1)
    imerg_times = pd.DatetimeIndex(imerg_da.time.values)
    valid_shifted = shifted_dates[shifted_dates.isin(imerg_times)]
    if len(valid_shifted) > 0:
        imd_shifted = imd_da.sel(time=valid_shifted + pd.Timedelta(days=1))
        imerg_shifted = imerg_da.sel(time=valid_shifted)
        flat_imd_s = imd_shifted.values.ravel()
        flat_imerg_s = imerg_shifted.values.ravel()
        valid_s = np.isfinite(flat_imd_s) & np.isfinite(flat_imerg_s)
        corr_shifted = float(np.corrcoef(flat_imd_s[valid_s], flat_imerg_s[valid_s])[0, 1]) if valid_s.sum() > 10 else np.nan
    else:
        corr_shifted = np.nan

    if np.isnan(corr_shifted) or corr_same >= corr_shifted:
        best, offset = "D", -3
    else:
        best, offset = "D-1", -27

    logger.info("Alignment: corr_same=%.3f corr_shifted=%.3f → best=%s (offset %+d h)",
                corr_same, corr_shifted, best, offset)
    return {
        "corr_same_day": corr_same,
        "corr_shifted": corr_shifted,
        "best_match": best,
        "window_offset_hours": offset,
        "n_days": len(common_dates),
    }

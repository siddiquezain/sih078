"""Ingest IFS ENS forecasts to a local Zarr cache.

Uses dynamical_catalog to open ECMWF IFS ENS 0.25° forecasts.
Subsets to region and variables before writing to Zarr.

IMPORTANT: Never use data.dynamical.org URLs directly.
Always go through the catalog API.
"""

from __future__ import annotations
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

logger = logging.getLogger(__name__)

CATALOG_ID = "ecmwf-ifs-ens-forecast-15-day-0-25-degree"

VARIABLES = [
    "precipitation_surface",
    "pressure_reduced_to_mean_sea_level",
    "wind_u_10m",
    "wind_v_10m",
]


def _catalog():
    """Lazy import of dynamical_catalog."""
    try:
        import dynamical_catalog
        return dynamical_catalog
    except ImportError as e:
        raise ImportError(
            "dynamical-catalog is required for ingest. "
            "Install with: pip install dynamical-catalog"
        ) from e


def subset_lat_lon(ds: xr.Dataset, lat_min, lat_max, lon_min, lon_max) -> xr.Dataset:
    """Subset dataset to a lat/lon bounding box.

    Handles both ascending and descending latitude orders.
    """
    lat = ds.latitude.values
    if lat[0] > lat[-1]:  # descending
        ds = ds.sel(
            latitude=slice(lat_max, lat_min),
            longitude=slice(lon_min, lon_max),
        )
    else:
        ds = ds.sel(
            latitude=slice(lat_min, lat_max),
            longitude=slice(lon_min, lon_max),
        )
    return ds


def ingest_one_init(
    init_time: pd.Timestamp,
    lat_min: float, lat_max: float,
    lon_min: float, lon_max: float,
    cache_dir: str | Path,
    variables: list[str] = VARIABLES,
) -> Path:
    """Download and cache one init time to Zarr.

    Returns path to the Zarr store. Cache hit returns immediately.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    zarr_path = cache_dir / f"{init_time.strftime('%Y%m%d_%H%M')}.zarr"

    if zarr_path.exists():
        logger.info("Cache hit: %s", zarr_path)
        return zarr_path

    logger.info("Fetching %s …", init_time.isoformat())
    t0 = time.perf_counter()

    dc = _catalog()
    cat = dc.open(CATALOG_ID, chunks=None)

    ds = cat.sel(init_time=init_time)
    ds = subset_lat_lon(ds, lat_min, lat_max, lon_min, lon_max)
    ds = ds[variables]
    ds = ds.compute()

    elapsed = time.perf_counter() - t0
    size_mb = sum(ds[v].nbytes for v in ds.data_vars) / 1e6
    logger.info("Downloaded %.1f MB in %.1f s for %s", size_mb, elapsed, init_time.isoformat())

    ds.to_zarr(str(zarr_path), mode="w")
    logger.info("Cached to %s", zarr_path)
    return zarr_path


def open_cached(zarr_path: str | Path) -> xr.Dataset:
    """Open a cached Zarr store."""
    return xr.open_zarr(str(zarr_path), chunks=None)


def ingest_all(
    start_date: str,
    end_date: str,
    lat_min: float, lat_max: float,
    lon_min: float, lon_max: float,
    cache_dir: str | Path,
) -> dict[pd.Timestamp, Path]:
    """Ingest all 00 UTC runs from start_date to end_date (inclusive).

    Returns dict: init_time → zarr_path.
    """
    inits = pd.date_range(start=start_date, end=end_date, freq="D")
    results = {}
    for init in inits:
        ts = pd.Timestamp(init).replace(hour=0, minute=0, second=0)
        try:
            p = ingest_one_init(ts, lat_min, lat_max, lon_min, lon_max, cache_dir)
            results[ts] = p
        except Exception as e:
            logger.error("Failed to ingest %s: %s", ts.isoformat(), e)
            raise
    return results

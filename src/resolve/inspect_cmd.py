"""inspect subcommand — probe catalog schema and print key facts.

Usage: python -m resolve inspect --config configs/montha.yaml

Does NOT do a full ingest. Downloads one init (2025-10-27 00 UTC) into a
temporary directory, computes the 24 h member rain for the target-day window,
and reports the regional max. Useful for verifying the environment before a
full run.
"""

from __future__ import annotations
import logging
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

from resolve.config import load_config

logger = logging.getLogger(__name__)


def run_inspect(config_path: str) -> None:
    """Open catalog, print schema, compute one 24 h accumulation."""
    cfg = load_config(config_path)

    # ── 1. Open catalog (no data download yet) ────────────────────────────
    try:
        import dynamical_catalog
    except ImportError:
        print("ERROR: dynamical-catalog not installed. pip install dynamical-catalog")
        return

    from resolve.ingest import CATALOG_ID
    print(f"\n{'='*60}")
    print(f"Catalog: {CATALOG_ID}")
    t_open = time.perf_counter()
    cat = dynamical_catalog.open(CATALOG_ID, chunks=None)
    print(f"Opened in {time.perf_counter() - t_open:.1f} s")

    # ── 2. Print schema ───────────────────────────────────────────────────
    print("\n--- Dimensions ---")
    for dim, size in cat.dims.items():
        print(f"  {dim}: {size}")

    print("\n--- Data variables ---")
    for v in cat.data_vars:
        arr = cat[v]
        print(f"  {v}: dtype={arr.dtype}, dims={arr.dims}")

    print("\n--- Coordinates ---")
    for c in cat.coords:
        arr = cat[c]
        vals = arr.values
        if vals.ndim == 1 and len(vals) <= 6:
            print(f"  {c}: {vals}")
        elif vals.ndim == 1:
            print(f"  {c}: dtype={arr.dtype}, n={len(vals)}, "
                  f"range=[{vals.min()}, {vals.max()}]")
        else:
            print(f"  {c}: dtype={arr.dtype}, shape={vals.shape}")

    # ── 3. Check lead_time dtype and latitude order ───────────────────────
    if "lead_time" in cat.coords:
        lt = cat.lead_time.values
        print(f"\n--- lead_time ---")
        print(f"  dtype: {lt.dtype}")
        print(f"  first 4: {lt[:4]}")
        print(f"  is timedelta: {np.issubdtype(lt.dtype, np.timedelta64)}")

    if "latitude" in cat.coords:
        lats = cat.latitude.values
        print(f"\n--- latitude ---")
        print(f"  range: {lats.min():.2f} to {lats.max():.2f}, n={len(lats)}")
        print(f"  ascending: {lats[0] < lats[-1]}")

    # ── 4. Check init_time range ──────────────────────────────────────────
    if "init_time" in cat.coords:
        its = pd.DatetimeIndex(cat.init_time.values)
        print(f"\n--- init_time ---")
        print(f"  range: {its.min().date()} to {its.max().date()}, n={len(its)}")

    # ── 5. Download one init and compute 24 h max member rain ─────────────
    probe_init = pd.Timestamp("2025-10-27T00:00:00")
    print(f"\n--- Probing init {probe_init.date()} ---")

    with tempfile.TemporaryDirectory() as tmpdir:
        from resolve.ingest import ingest_one_init, open_cached
        from resolve.accum import accumulate_window, window_bounds_utc

        zarr_path = ingest_one_init(
            probe_init,
            cfg.region.lat_min, cfg.region.lat_max,
            cfg.region.lon_min, cfg.region.lon_max,
            tmpdir,
        )
        ds = open_cached(zarr_path)

        print(f"  lead_time dtype after normalise: {ds.lead_time.dtype}")
        print(f"  dims: {dict(ds.dims)}")
        print(f"  member values: {ds.member.values[:5]} …")
        print(f"  latitude ascending: {ds.latitude.values[0] < ds.latitude.values[-1]}")

        # Pick target-day window (use 2025-10-28 as probe date)
        probe_target = "2025-10-28"
        lead_h = (pd.Timestamp(probe_target + "T03:00:00") - probe_init).total_seconds() / 3600
        offset = -3 if lead_h > 144 else 0
        t0, t1 = window_bounds_utc(
            probe_target,
            convention=cfg.imd_day_convention,
            offset_hours=offset,
        )
        print(f"  Accumulating ({t0}, {t1}] …")

        t_acc = time.perf_counter()
        accum = accumulate_window(ds, probe_init, t0, t1)
        elapsed_acc = time.perf_counter() - t_acc

        max_rain = float(accum.max())
        mean_rain = float(accum.mean())
        print(f"  Accumulation time: {elapsed_acc:.1f} s")
        print(f"  Regional max 24 h member rain: {max_rain:.1f} mm")
        print(f"  Regional mean 24 h member rain: {mean_rain:.2f} mm")

    print(f"\n{'='*60}")
    print("Inspect complete. Environment looks OK.")

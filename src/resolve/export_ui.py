"""Export pipeline results to ui/data/<event>.js per ui/SCHEMA.md (resolve-ui/1).

Entry point: python -m resolve export-ui --config <path>
"""

from __future__ import annotations
import json
import logging
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from resolve.config import load_config, ResolveConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# NMEP computation
# ---------------------------------------------------------------------------

def compute_nmep(member_masks: np.ndarray, nlat: int, nlon: int, n: int) -> list:
    """NMEP tiles at size n. Returns flat int list of length ceil(nlat/n)*ceil(nlon/n).

    Each value = number of members with ≥ 1 exceedance cell in that tile.
    Compute from member_masks (bool, n_members × nlat × nlon).
    """
    n_members = member_masks.shape[0]
    nr = math.ceil(nlat / n)
    nc = math.ceil(nlon / n)
    result = []
    for r in range(nr):
        for c in range(nc):
            tile = member_masks[:, r * n: min((r + 1) * n, nlat),
                                   c * n: min((c + 1) * n, nlon)]
            result.append(int(tile.reshape(n_members, -1).any(axis=1).sum()))
    return result


# ---------------------------------------------------------------------------
# Track loading
# ---------------------------------------------------------------------------

def _load_track_ibtracs(cache_dir: Path) -> list:
    """Load Montha 2025 track from IBTrACS CSV if present, else use IMD bulletin.

    Returns list of {time, lat, lon, label, source}.
    """
    candidates = list(cache_dir.glob("*ibtracs*.csv")) + list(cache_dir.glob("IBTrACS*.csv"))
    if candidates:
        import csv
        rows = []
        with open(candidates[0], newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                name = row.get("NAME", "").upper()
                if "MONTHA" not in name:
                    continue
                try:
                    rows.append({
                        "time": row["ISO_TIME"].strip(),
                        "lat": float(row["LAT"]),
                        "lon": float(row["LON"]),
                        "label": name.title(),
                        "source": f"IBTrACS v04r01 ({candidates[0].name})",
                    })
                except (KeyError, ValueError):
                    continue
        if rows:
            logger.info("Loaded %d IBTrACS track points for Montha", len(rows))
            return rows

    # IBTrACS not available — use IMD bulletin position
    logger.info("IBTrACS not found; using IMD bulletin position for Montha track")
    return [
        {
            "time": "2025-10-28T03:00Z",
            "lat": 14.9,
            "lon": 82.9,
            "label": "Montha landfall",
            "source": "IMD bulletin (08:30 IST 28 Oct 2025 position)",
        }
    ]


# ---------------------------------------------------------------------------
# Schema checks
# ---------------------------------------------------------------------------

def _check_export(data: dict, n_members: int) -> None:
    """Run all schema checks from ui/SCHEMA.md. Raises ValueError on any failure."""
    assert data["schema"] == "resolve-ui/1", "schema field wrong"
    assert data["meta"]["preview"] is False, "meta.preview must be false"
    assert data["meta"]["label"] in ("replay", "live"), "label must be replay or live"

    grid = data["grid"]
    nlat, nlon = grid["nlat"], grid["nlon"]
    n_cells = nlat * nlon
    land = data["land"]
    assert len(land) == n_cells, f"land length {len(land)} != {n_cells}"

    windows = data["windows"]
    for run in data["runs"]:
        counts = run["counts"]
        assert len(counts) == n_cells, f"counts length {len(counts)} != {n_cells} for {run['init']}"
        assert run["peak_count"] == max(counts), (
            f"peak_count {run['peak_count']} != max(counts) {max(counts)} for {run['init']}"
        )
        assert all(0 <= c <= n_members for c in counts), f"counts out of range for {run['init']}"
        for w in windows:
            n = w["cells"]
            if n <= 1:
                continue
            key = str(n)
            assert key in run["nmep"], f"nmep missing window {n} for {run['init']}"
            nmep_arr = run["nmep"][key]
            nr = math.ceil(nlat / n)
            nc = math.ceil(nlon / n)
            assert len(nmep_arr) == nr * nc, (
                f"nmep[{n}] length {len(nmep_arr)} != {nr * nc} for {run['init']}"
            )
            assert all(0 <= v <= n_members for v in nmep_arr), (
                f"nmep[{n}] out of range for {run['init']}"
            )
            assert max(nmep_arr) >= max(counts), (
                f"max(nmep[{n}])={max(nmep_arr)} < max(counts)={max(counts)} "
                f"for {run['init']}: NMEP tile can never be less than its wettest cell"
            )

    obs = data["observed"]
    n_obs = obs["n"]
    assert n_obs == len(obs["cells"]), "observed.n != len(observed.cells)"


# ---------------------------------------------------------------------------
# Main export function
# ---------------------------------------------------------------------------

def run_export_ui(config_path: str) -> Path:
    """Build ui/data/<event>.js for the given config."""
    cfg = load_config(config_path)
    output_dir = Path(cfg.output_dir)
    cache_dir = Path(cfg.cache_dir)

    # --- Truth ---
    from resolve.truth import fetch_imd, imd_subset
    from resolve.exceedance import obs_exceedance_mask
    imd_dir = cache_dir / "imd"
    imd_dir.mkdir(parents=True, exist_ok=True)
    try:
        imd_da_full = fetch_imd(2025, 2025, file_dir=imd_dir)
    except Exception as e:
        logger.error("IMD fetch failed: %s", e)
        sys.exit(1)
    imd_da = imd_subset(
        imd_da_full,
        cfg.region.lat_min, cfg.region.lat_max,
        cfg.region.lon_min, cfg.region.lon_max,
        "2025-10-20", "2025-11-01",
    )

    # --- Target day and land mask ---
    from resolve.hero import find_target_day
    target_date = find_target_day(imd_da, cfg)
    target_imd = imd_da.sel(time=target_date, method="nearest")
    lat = imd_da.latitude.values
    lon = imd_da.longitude.values
    land_mask = np.isfinite(target_imd.values)  # (nlat, nlon) bool
    obs_exc = obs_exceedance_mask(target_imd, cfg.rain_thresholds.very_heavy).values
    obs_cells = [int(i) for i in np.where(obs_exc.ravel())[0]]
    n_land = int(land_mask.sum())
    f_obs_global = float(len(obs_cells)) / n_land if n_land > 0 else 0.0
    max_mm_val = float(target_imd.values[land_mask].max()) if land_mask.any() else None
    nlat, nlon = len(lat), len(lon)

    # --- Per-run processing ---
    from resolve.hero import process_one_init
    from resolve.ingest import ingest_all

    zarr_paths = ingest_all(
        cfg.init_dates.start, cfg.init_dates.end,
        cfg.region.lat_min, cfg.region.lat_max,
        cfg.region.lon_min, cfg.region.lon_max,
        cache_dir,
    )

    run_results = {}
    for init_time, zarr_path in sorted(zarr_paths.items()):
        try:
            res = process_one_init(
                init_time, zarr_path, target_date, cfg,
                land_mask, lat, lon, obs_exc,
            )
            run_results[init_time] = res
        except ValueError as e:
            if "86400 s" in str(e):
                logger.warning("Init %s: window not covered — skipping. (%s)",
                               init_time.isoformat(), e)
            else:
                raise

    # --- Crossrun (mask IoU) ---
    from resolve.crossrun import haversine_km, iou_masks, detect_lockon
    inits_sorted = sorted(run_results.keys())
    crossrun_extra: dict[pd.Timestamp, dict] = {}
    prev_obj = None
    prev_mask = None
    drifts_c = []
    ious_c = []

    for init in inits_sorted:
        res = run_results[init]
        objs = res["objects"]
        tmask = res["tracked_mask"]
        drift_val = None
        iou_val = None

        if prev_obj is not None and objs:
            obj_c = objs[0]
            drift = haversine_km(
                (prev_obj.centroid_lat, prev_obj.centroid_lon),
                (obj_c.centroid_lat, obj_c.centroid_lon),
            )
            if drift <= cfg.crossrun.max_centroid_km:
                drift_val = round(float(drift), 2)
                if prev_mask is not None and tmask is not None:
                    iou_val = round(float(iou_masks(prev_mask, tmask)), 4)
            drifts_c.append(drift_val if drift_val is not None else np.nan)
            ious_c.append(iou_val if iou_val is not None else np.nan)
        else:
            drifts_c.append(np.nan)
            ious_c.append(np.nan)

        crossrun_extra[init] = {"drift_km": drift_val, "iou": iou_val}
        if objs:
            prev_obj = objs[0]
            prev_mask = tmask

    valid_drifts = [d if np.isfinite(d) else 1e9 for d in drifts_c]
    valid_ious = [i if np.isfinite(i) else 0.0 for i in ious_c]
    lockon_idx = detect_lockon(
        valid_drifts, valid_ious,
        cfg.crossrun.lockon_drift_km, cfg.crossrun.lockon_iou,
    )
    lockon_init_ts = None
    if lockon_idx is not None and lockon_idx + 1 < len(inits_sorted):
        lockon_init_ts = inits_sorted[lockon_idx + 1]
    lockon_init_str = (
        lockon_init_ts.strftime("%Y-%m-%dT%H:%MZ") if lockon_init_ts else None
    )

    # Mark locked flag per run
    locked_from = lockon_init_ts
    locked_set = set()
    if locked_from is not None:
        for t in inits_sorted:
            if t >= locked_from:
                locked_set.add(t)

    # --- Git commit ---
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(config_path).parent,
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        commit = "unknown"

    # --- Track ---
    track = _load_track_ibtracs(cache_dir)

    # --- Build runs ---
    target_dt = pd.Timestamp(target_date)
    windows_list = [
        {"cells": n, "km": round(n * 0.25 * 111.32)}
        for n in cfg.fss.windows_cells
    ]
    n_members = 51  # IFS ENS

    runs_out = []
    for init in inits_sorted:
        res = run_results[init]
        mbr_masks = res["member_masks"]   # (n_members, nlat, nlon) bool
        n_members_actual = mbr_masks.shape[0]
        counts_2d = mbr_masks.sum(axis=0).astype(int)   # (nlat, nlon)
        counts_flat = counts_2d.ravel().tolist()
        peak = int(counts_2d.max())

        nmep_by_n = {}
        for w in windows_list:
            n = w["cells"]
            if n <= 1:
                continue
            nmep_by_n[str(n)] = compute_nmep(mbr_masks, nlat, nlon, n)

        objs = res["objects"]
        obj_out = None
        if objs:
            top = objs[0]
            # count-weighted centroid from counts_2d
            cnt_weights = counts_2d.ravel()
            total_w = cnt_weights.sum()
            if total_w > 0 and top.cells:
                lats_c = np.array([lat[c // nlon] for c in top.cells], dtype=float)
                lons_c = np.array([lon[c % nlon] for c in top.cells], dtype=float)
                w_cells = np.array([counts_2d.ravel()[c] for c in top.cells], dtype=float)
                w_total = w_cells.sum()
                c_lat = float((lats_c * w_cells).sum() / w_total) if w_total > 0 else top.centroid_lat
                c_lon = float((lons_c * w_cells).sum() / w_total) if w_total > 0 else top.centroid_lon
            else:
                c_lat, c_lon = top.centroid_lat, top.centroid_lon
            obj_out = {
                "lat": round(c_lat, 4),
                "lon": round(c_lon, 4),
                "n_cells": top.n_cells,
                "area_km2": round(top.area_km2, 1),
            }

        cr = crossrun_extra[init]
        lead_days = int((target_dt - init).days)
        lead_hours = int((target_dt + pd.Timedelta(hours=res["window_offset"] + 3) - init).total_seconds() / 3600)

        runs_out.append({
            "init": init.strftime("%Y-%m-%dT%H:%MZ"),
            "lead_days": lead_days,
            "lead_hours": lead_hours,
            "window_offset_hours": res["window_offset"],
            "members_valid": n_members_actual,
            "peak_count": peak,
            "counts": counts_flat,
            "nmep": nmep_by_n,
            "fss": [round(float(v), 4) for v in res["fss_table"]],
            "useful": round(float(0.5 + res["f_obs"] / 2), 4),
            "earned_cells": res["earned_width_cells"],
            "object": obj_out,
            "drift_km": cr["drift_km"],
            "iou": cr["iou"],
            "locked": init in locked_set,
        })

    # --- Build top-level ---
    convention = cfg.imd_day_convention
    if convention == "ending":
        rain_day_str = f"24 h to 08:30 IST on {pd.Timestamp(target_date).strftime('%-d %b %Y')}"
    else:
        rain_day_str = f"24 h from 00 UTC on {pd.Timestamp(target_date).strftime('%-d %b %Y')}"

    data = {
        "schema": "resolve-ui/1",
        "meta": {
            "preview": False,
            "label": cfg.label,
            "event": "Cyclone Montha",
            "hazard": "Very heavy rain",
            "threshold_mm": cfg.rain_thresholds.very_heavy,
            "target_date": target_date,
            "imd_day_convention": convention,
            "rain_day": rain_day_str,
            "forecast": "ECMWF IFS ENS, 0.25°, 51 members",
            "forecast_credit": "ECMWF open data (CC BY 4.0), processed by dynamical.org",
            "truth": "IMD 0.25° gridded daily rainfall",
            "members": n_members,
            "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "commit": commit,
        },
        "grid": {
            "lat0": float(lat[0]),
            "lon0": float(lon[0]),
            "step": 0.25,
            "nlat": nlat,
            "nlon": nlon,
        },
        "land": land_mask.ravel().astype(int).tolist(),
        "observed": {
            "cells": obs_cells,
            "n": len(obs_cells),
            "f_obs": round(f_obs_global, 6),
            "max_mm": round(max_mm_val, 1) if max_mm_val is not None else None,
        },
        "windows": windows_list,
        "runs": runs_out,
        "lockon_init": lockon_init_str,
        "track": track,
    }

    # --- Schema checks ---
    _check_export(data, n_members)

    # --- Write ---
    ui_data_dir = Path("ui/data")
    ui_data_dir.mkdir(parents=True, exist_ok=True)
    out_path = ui_data_dir / "montha.js"
    js_body = json.dumps(data, separators=(",", ":"))
    out_path.write_text(f"window.RESOLVE_DATA = {js_body};\n")
    logger.info("Wrote %s (%.1f KB)", out_path, out_path.stat().st_size / 1024)
    return out_path

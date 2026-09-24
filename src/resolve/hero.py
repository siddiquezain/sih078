"""Hero orchestrator for Phase 1 — Montha hero figure.

Entry point: python -m resolve hero --config configs/montha.yaml
"""

from __future__ import annotations
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from resolve.config import load_config, ResolveConfig
from resolve.provenance import make_provenance, save_sidecar

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)


def find_target_day(imd_da, cfg: ResolveConfig) -> str:
    """Return the IMD date with the largest land area >= threshold_mm.

    Searches within [target_day_search.start, target_day_search.end].
    Raises RuntimeError if no qualifying day found.
    """
    threshold = cfg.target_day_search.threshold_mm
    start = pd.Timestamp(cfg.target_day_search.start)
    end = pd.Timestamp(cfg.target_day_search.end)

    subset = imd_da.sel(time=slice(start, end))
    best_date = None
    best_count = 0
    for t in subset.time.values:
        day = subset.sel(time=t)
        n_cells = int((day >= threshold).sum())
        logger.info("IMD %s: %d cells >= %.1f mm", str(t)[:10], n_cells, threshold)
        if n_cells > best_count:
            best_count = n_cells
            best_date = str(t)[:10]

    if best_date is None or best_count < cfg.target_day_search.min_cells:
        raise RuntimeError(
            f"No IMD day found with >= {cfg.target_day_search.min_cells} cells "
            f">= {threshold} mm in {start.date()}–{end.date()}"
        )

    logger.info("Target day: %s (%d cells >= %.1f mm)", best_date, best_count, threshold)
    return best_date


def process_one_init(
    init_time: pd.Timestamp,
    zarr_path: Path,
    target_date: str,
    cfg: ResolveConfig,
    land_mask: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    obs_exceedance: np.ndarray,
) -> dict:
    """Run the full pipeline for one init time.

    Returns dict with: fss_table, earned_width_cells, prob_field, objects,
    f_obs, window_offset, lead_hours.
    """
    from resolve.ingest import open_cached
    from resolve.accum import accumulate_window, window_bounds_utc
    from resolve.exceedance import member_exceedance_mask, ensemble_probability
    from resolve.objects import find_objects
    from resolve.fss import fss_table as compute_fss_table
    from resolve.earned import earned_width

    ds = open_cached(zarr_path)

    lead_hours = (pd.Timestamp(target_date + "T03:00:00") - init_time).total_seconds() / 3600
    if lead_hours > 144:
        t0, t1 = window_bounds_utc(
            target_date, convention=cfg.imd_day_convention, offset_hours=-3
        )
        window_offset = -3
    else:
        t0, t1 = window_bounds_utc(
            target_date, convention=cfg.imd_day_convention, offset_hours=0
        )
        window_offset = 0

    accum = accumulate_window(ds, init_time, t0, t1)

    # Align forecast grid onto IMD 0.25° grid by coordinates
    accum = accum.sortby(["latitude", "longitude"])
    accum = accum.reindex(
        latitude=lat,
        longitude=lon,
        method="nearest",
        tolerance=0.01,
    )
    if not np.allclose(accum.latitude.values, lat, atol=0.01):
        raise ValueError(
            "Forecast latitude coordinates don't match IMD grid after reindex. "
            "Check that both grids are on 0.25° spacing."
        )

    threshold = cfg.rain_thresholds.very_heavy
    mbr_mask = member_exceedance_mask(accum, threshold)
    prob = ensemble_probability(mbr_mask)
    prob_np = prob.values

    f_obs = float(obs_exceedance.sum()) / float(land_mask.sum()) if land_mask.sum() > 0 else 0.0

    if prob_np.shape != obs_exceedance.shape:
        raise ValueError(
            f"Forecast grid {prob_np.shape} != observed grid {obs_exceedance.shape} "
            "after reindex. Grids must be on the same 0.25° IMD grid."
        )
    fss_vals = compute_fss_table(
        prob_np, obs_exceedance.astype(float),
        cfg.fss.windows_cells, land_mask,
    )

    ew = earned_width(fss_vals, cfg.fss.windows_cells, f_obs)

    prob_binary = (prob_np >= cfg.objects.p_min)
    objs = find_objects(
        prob_binary, prob_np, lat, lon,
        min_area_cells=cfg.objects.min_area_cells,
    )

    return {
        "fss_table": fss_vals,
        "earned_width_cells": ew,
        "prob_field": prob_np,
        "objects": objs,
        "f_obs": f_obs,
        "window_offset": window_offset,
        "lead_hours": lead_hours,
    }


def make_hero_figure(
    run_results: dict,
    target_date: str,
    obs_exceedance: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    cfg: ResolveConfig,
    output_dir: Path,
    crossrun_stats: list[dict],
    track_lats: list[float] | None,
    track_lons: list[float] | None,
) -> tuple[Path, Path]:
    """Produce the 3-panel hero figure (PNG 300 dpi + SVG)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from scipy.ndimage import uniform_filter
    from resolve.fss import useful_skill_line
    from resolve.earned import cells_to_km

    try:
        import cartopy.crs as ccrs
        HAS_CARTOPY = True
    except ImportError:
        HAS_CARTOPY = False
        logger.warning("cartopy not found; plotting without coastlines")

    inits = sorted(run_results.keys())
    target_dt = pd.Timestamp(target_date)
    lead_colors = {7: "#4e9af1", 4: "#f1a34e", 2: "#e05c5c", 1: "#5ce05c"}

    fig = plt.figure(figsize=(18, 6))

    # Panel A: footprints at earned width for D-7, D-4, D-2, D-1
    if HAS_CARTOPY:
        import cartopy.feature as cfeature
        ax_a = fig.add_subplot(1, 3, 1, projection=ccrs.PlateCarree())
        ax_a.set_extent([lon.min(), lon.max(), lat.min(), lat.max()], ccrs.PlateCarree())
        ax_a.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax_a.add_feature(cfeature.BORDERS, linewidth=0.3)
        transform = ccrs.PlateCarree()
    else:
        ax_a = fig.add_subplot(1, 3, 1)
        transform = None

    lon2d, lat2d = np.meshgrid(lon, lat)
    panel_a_inits = {}
    for lead in [7, 4, 2, 1]:
        t = (target_dt - pd.Timedelta(days=lead)).replace(hour=0)
        if t in run_results:
            panel_a_inits[lead] = t

    for lead, init in sorted(panel_a_inits.items()):
        res = run_results[init]
        prob = res["prob_field"]
        ew_cells = res["earned_width_cells"]
        smooth_prob = uniform_filter(prob, size=ew_cells, mode="constant", cval=0.0) if ew_cells else prob
        color = lead_colors.get(lead, "gray")
        kwargs = dict(levels=[0.5, 1.01], colors=[color], alpha=0.4)
        if transform:
            ax_a.contourf(lon2d, lat2d, smooth_prob, transform=transform, **kwargs)
            ax_a.contour(lon2d, lat2d, smooth_prob, levels=[0.5], colors=[color], linewidths=1.0, transform=transform)
        else:
            ax_a.contourf(lon2d, lat2d, smooth_prob, **kwargs)
            ax_a.contour(lon2d, lat2d, smooth_prob, levels=[0.5], colors=[color], linewidths=1.0)

    obs_kw = dict(levels=[0.5], colors=["black"], linewidths=1.5)
    if transform:
        ax_a.contour(lon2d, lat2d, obs_exceedance.astype(float), transform=transform, **obs_kw)
    else:
        ax_a.contour(lon2d, lat2d, obs_exceedance.astype(float), **obs_kw)

    if track_lats and track_lons:
        track_kw = dict(color="purple", linewidth=1.5, marker="o", markersize=3)
        if transform:
            ax_a.plot(track_lons, track_lats, transform=transform, **track_kw)
        else:
            ax_a.plot(track_lons, track_lats, **track_kw)

    legend_handles = [
        Patch(facecolor=lead_colors.get(l, "gray"), alpha=0.5, label=f"D−{l}")
        for l in sorted(panel_a_inits.keys())
    ]
    legend_handles.append(Patch(facecolor="none", edgecolor="black", label="Obs ≥ 115.6 mm"))
    ax_a.legend(handles=legend_handles, loc="lower left", fontsize=7)
    ax_a.set_title(f"Panel A — Footprints at earned width\nTarget: {target_date}", fontsize=9)

    # Panel B: FSS vs width curves
    ax_b = fig.add_subplot(1, 3, 2)
    windows_km = [cells_to_km(n) for n in cfg.fss.windows_cells]
    for init in inits:
        res = run_results[init]
        lead_d = (target_dt - init).days
        ax_b.plot(windows_km, res["fss_table"], marker="o", markersize=4, linewidth=1,
                  label=f"D−{lead_d}", alpha=0.7)
    mean_f_obs = np.mean([run_results[i]["f_obs"] for i in inits])
    usl_val = useful_skill_line(mean_f_obs)
    ax_b.axhline(usl_val, color="black", linestyle="--", linewidth=1.2,
                 label=f"Useful skill ({usl_val:.2f})")
    ax_b.set_xlabel("Neighbourhood width (km)")
    ax_b.set_ylabel("FSS")
    ax_b.set_ylim(0, 1.05)
    ax_b.set_title("Panel B — FSS vs neighbourhood width", fontsize=9)
    ax_b.legend(fontsize=6, ncol=2)

    # Panel C: drift and probability by run
    ax_c1 = fig.add_subplot(1, 3, 3)
    ax_c2 = ax_c1.twinx()
    init_dates = [s["init"] for s in crossrun_stats]
    drifts = [s.get("drift_km", np.nan) for s in crossrun_stats]
    probs = [s.get("max_prob", np.nan) for s in crossrun_stats]
    lockon_init = next((s["init"] for s in crossrun_stats if s.get("lockon")), None)

    ax_c1.plot(init_dates, drifts, "b-o", markersize=4, label="Drift (km)")
    ax_c2.plot(init_dates, probs, "r-s", markersize=4, label="Max prob")
    if lockon_init is not None:
        ax_c1.axvline(lockon_init, color="green", linestyle=":", linewidth=1.5,
                      label=f"Lock-on: {str(lockon_init)[:10]}")
    ax_c1.set_ylabel("Run-to-run drift (km)", color="blue")
    ax_c2.set_ylabel("Max probability", color="red")
    ax_c1.set_xlabel("Init date")
    ax_c1.set_title("Panel C — Drift and probability by run", fontsize=9)
    ax_c1.tick_params(axis="x", rotation=45, labelsize=7)
    lines1, labels1 = ax_c1.get_legend_handles_labels()
    lines2, labels2 = ax_c2.get_legend_handles_labels()
    ax_c1.legend(lines1 + lines2, labels1 + labels2, fontsize=7)

    caption = (
        "Single event (Montha, Oct 2025); full event library before the finale.\n"
        f"Label: {cfg.label}."
    )
    fig.text(0.5, -0.02, caption, ha="center", fontsize=8, style="italic")
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / "hero.png"
    svg_path = output_dir / "hero.svg"
    fig.savefig(str(png_path), dpi=300, bbox_inches="tight")
    fig.savefig(str(svg_path), bbox_inches="tight")
    plt.close(fig)
    logger.info("Hero figure saved: %s, %s", png_path, svg_path)
    return png_path, svg_path


def write_summary(
    run_results: dict,
    target_date: str,
    crossrun_stats: list[dict],
    alignment: dict,
    cfg: ResolveConfig,
    output_dir: Path,
) -> Path:
    """Write outputs/montha/summary.json."""
    from resolve.earned import cells_to_km

    inits = sorted(run_results.keys())
    target_dt = pd.Timestamp(target_date)

    earned = {}
    fss_by_lead = {}
    for init in inits:
        lead_d = (target_dt - init).days
        res = run_results[init]
        ew = res["earned_width_cells"]
        earned[f"D-{lead_d}"] = {
            "earned_width_cells": ew,
            "earned_width_km": cells_to_km(ew) if ew else None,
        }
        fss_by_lead[f"D-{lead_d}"] = {
            str(n): float(v)
            for n, v in zip(cfg.fss.windows_cells, res["fss_table"])
        }

    lockon = next(
        (str(s["init"])[:10] for s in crossrun_stats if s.get("lockon")),
        "none"
    )

    summary = {
        "label": cfg.label,
        "target_date": target_date,
        "config": {
            "region": cfg.region.__dict__,
            "threshold_mm": cfg.rain_thresholds.very_heavy,
            "fss_windows_cells": cfg.fss.windows_cells,
        },
        "alignment": alignment,
        "earned_width_per_lead": earned,
        "fss_table": fss_by_lead,
        "lockon_run": lockon,
        "crossrun_stats": [
            {**s, "init": str(s["init"])[:10]}
            for s in crossrun_stats
        ],
        "note": (
            "Single-event point estimate. "
            "Full event library needed before drawing general conclusions."
        ),
    }

    out = output_dir / "summary.json"
    out.write_text(json.dumps(summary, indent=2, default=str))
    logger.info("Summary written: %s", out)
    return out


def run_hero(config_path: str, dry_run: bool = False) -> None:
    """Full Phase 1 pipeline."""
    cfg = load_config(config_path)
    output_dir = Path(cfg.output_dir)
    cache_dir = Path(cfg.cache_dir)

    if dry_run:
        logger.info("Dry run — config loaded:")
        logger.info("  label: %s", cfg.label)
        logger.info("  region: lat %.1f–%.1f, lon %.1f–%.1f",
                    cfg.region.lat_min, cfg.region.lat_max,
                    cfg.region.lon_min, cfg.region.lon_max)
        logger.info("  inits: %s to %s", cfg.init_dates.start, cfg.init_dates.end)
        logger.info("  output_dir: %s", output_dir)
        return

    # Step 1: Ingest
    from resolve.ingest import ingest_all
    logger.info("=== Step 1: Ingest ===")
    zarr_paths = ingest_all(
        cfg.init_dates.start, cfg.init_dates.end,
        cfg.region.lat_min, cfg.region.lat_max,
        cfg.region.lon_min, cfg.region.lon_max,
        cache_dir,
    )

    # Step 2: Truth
    from resolve.truth import fetch_imd, imd_subset, fetch_imerg_daily, check_alignment
    logger.info("=== Step 2: Truth ===")
    imd_dir = cache_dir / "imd"
    imd_dir.mkdir(parents=True, exist_ok=True)

    try:
        imd_da_full = fetch_imd(2025, 2025, file_dir=imd_dir)
    except Exception as e:
        logger.error("IMD fetch failed: %s", e)
        logger.error("Stopping. Propose fallback: use IMERG as truth or check imdlib setup.")
        sys.exit(1)

    truth_start = "2025-10-20"
    truth_end = "2025-11-01"
    imd_da = imd_subset(
        imd_da_full,
        cfg.region.lat_min, cfg.region.lat_max,
        cfg.region.lon_min, cfg.region.lon_max,
        truth_start, truth_end,
    )

    alignment = {"best_match": "D", "window_offset_hours": -3, "n_days": 0,
                 "note": "IMERG not attempted"}
    try:
        imerg_dir = cache_dir / "imerg"
        imerg_da = fetch_imerg_daily(
            truth_start, truth_end, imerg_dir,
            cfg.region.lat_min, cfg.region.lat_max,
            cfg.region.lon_min, cfg.region.lon_max,
        )
        alignment = check_alignment(
            imd_da, imerg_da,
            expected_convention=cfg.imd_day_convention,
        )
    except Exception as e:
        logger.warning("IMERG unavailable: %s — proceeding with IMD only (no 10 km rung)", e)
        alignment["note"] = f"IMERG unavailable: {e}. 10 km rung omitted."

    # Step 3: Target day
    logger.info("=== Step 3: Target day ===")
    target_date = find_target_day(imd_da, cfg)

    target_imd = imd_da.sel(time=target_date, method="nearest")
    land_mask = np.isfinite(target_imd.values)
    lat = imd_da.latitude.values
    lon = imd_da.longitude.values

    from resolve.exceedance import obs_exceedance_mask
    obs_exc = obs_exceedance_mask(target_imd, cfg.rain_thresholds.very_heavy).values

    # Steps 4–8: Per-run processing
    logger.info("=== Steps 4–8: Per-run processing ===")
    run_results = {}
    for init_time, zarr_path in sorted(zarr_paths.items()):
        logger.info("Processing init: %s", init_time.isoformat())
        res = process_one_init(
            init_time, zarr_path, target_date, cfg,
            land_mask, lat, lon, obs_exc,
        )
        run_results[init_time] = res

    # Step 6: Crossrun matching
    from resolve.crossrun import match_objects, haversine_km, detect_lockon, _bbox_iou
    logger.info("=== Step 6: Crossrun ===")
    crossrun_stats = []
    inits_sorted = sorted(run_results.keys())
    drifts = []
    ious_list = []

    for k, init in enumerate(inits_sorted):
        stat = {
            "init": init,
            "drift_km": np.nan,
            "iou": np.nan,
            "max_prob": float(run_results[init]["prob_field"].max()),
            "lockon": False,
        }
        if k > 0:
            prev = inits_sorted[k - 1]
            objs_prev = run_results[prev]["objects"]
            objs_curr = run_results[init]["objects"]
            matches = match_objects(
                objs_prev, objs_curr,
                masks_a=None, masks_b=None,
                max_centroid_km=cfg.crossrun.max_centroid_km,
            )
            if matches:
                i0, j0 = matches[0]
                drift = haversine_km(
                    (objs_prev[i0].centroid_lat, objs_prev[i0].centroid_lon),
                    (objs_curr[j0].centroid_lat, objs_curr[j0].centroid_lon),
                )
                stat["drift_km"] = drift
                drifts.append(drift)
                iou = _bbox_iou(objs_prev[i0].bbox, objs_curr[j0].bbox)
                stat["iou"] = iou
                ious_list.append(iou)
            else:
                drifts.append(np.nan)
                ious_list.append(np.nan)
        crossrun_stats.append(stat)

    valid_drifts = [d if np.isfinite(d) else 1e9 for d in drifts]
    valid_ious = [i if np.isfinite(i) else 0.0 for i in ious_list]
    lockon_idx = detect_lockon(
        valid_drifts, valid_ious,
        cfg.crossrun.lockon_drift_km, cfg.crossrun.lockon_iou,
    )
    if lockon_idx is not None:
        # drifts[m] = drift arriving at inits_sorted[m+1].
        # detect_lockon returns k = first index in drifts where all m>=k satisfy the condition.
        # The run that *arrives* at a stable position is inits_sorted[k+1].
        lockon_init = inits_sorted[lockon_idx + 1] if lockon_idx + 1 < len(inits_sorted) else None
        if lockon_init:
            for s in crossrun_stats:
                if s["init"] == lockon_init:
                    s["lockon"] = True
        logger.info("Lock-on at: %s", str(lockon_init)[:10] if lockon_init else "none")
    else:
        logger.info("Lock-on: none")

    # Step 9: Hero figure
    logger.info("=== Step 9: Hero figure ===")
    png_path, svg_path = make_hero_figure(
        run_results, target_date, obs_exc,
        lat, lon, cfg, output_dir,
        crossrun_stats,
        track_lats=[14.9], track_lons=[82.9],
    )

    # Step 10: Summary + provenance
    logger.info("=== Step 10: Summary + provenance ===")
    summary_path = write_summary(
        run_results, target_date, crossrun_stats, alignment, cfg, output_dir
    )

    prov = make_provenance(
        datasets=["ecmwf-ifs-ens-forecast-15-day-0-25-degree", "IMD-0.25deg-daily", "GPM_3IMERGDF"],
        init_times=[str(t) for t in sorted(zarr_paths.keys())],
        thresholds={
            "heavy_mm": cfg.rain_thresholds.heavy,
            "very_heavy_mm": cfg.rain_thresholds.very_heavy,
            "extremely_heavy_mm": cfg.rain_thresholds.extremely_heavy,
        },
        window_offsets={"window_offset_hours": alignment.get("window_offset_hours", -3)},
        label=cfg.label,
        extra={"target_date": target_date, "alignment": alignment},
    )
    save_sidecar(summary_path, prov)
    save_sidecar(png_path, prov)

    logger.info("=== Phase 1 complete ===")
    logger.info("PNG:     %s", png_path)
    logger.info("SVG:     %s", svg_path)
    logger.info("Summary: %s", summary_path)

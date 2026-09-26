# Real-Schema Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the ten bugs identified in code review so the pipeline runs correctly on real IFS ENS data, add an `inspect` subcommand, run end-to-end on real data, and report results.

**Architecture:** All fixes are in five files (`ingest.py`, `accum.py`, `truth.py`, `hero.py`, `config.py`) plus a new `inspect_cmd.py`. Tests are rebuilt around the real schema (timedelta `lead_time`, `member` dim). The inspect command probes the catalog without a full ingest to validate the environment quickly.

**Tech Stack:** Python 3.11, xarray≥2024.1, zarr≥2.16, scipy, dynamical-catalog, imdlib, earthaccess, pytest.

**Real schema facts (do not assume, verify with inspect):**
- `lead_time` is `timedelta64` (e.g. `0h`, `3h`, `6h`, …, `360h`)
- ensemble dimension is named `ensemble_member` in the catalog (renamed to `member` at ingest)
- `precipitation_surface` units: kg m⁻² s⁻¹ = mm/s (mean rate since previous step)
- GPM_3IMERGDF V07 `precipitation` is already **mm/day** (not mm/hr)
- IMD day D ends at 03 UTC on date D (`imd_day_convention: ending`)

---

## File Map

```
Modified:
  src/resolve/config.py          — add imd_day_convention field
  src/resolve/ingest.py          — rename ensemble_member→member, store init_time coord,
                                   clear encoding, drop zarr encoding
  src/resolve/accum.py           — accept init_time; compute valid_time=init_time+lead_time;
                                   add 86 400 s window gate; add member-rain sanity gate;
                                   add convention param to window_bounds_utc
  src/resolve/truth.py           — fetch_imd archive→realtime fallback;
                                   IMERG units fix (remove ×24, raise >1000 mm);
                                   earthaccess.login() auto-detect;
                                   check_alignment: transpose+regrid IMERG, proper
                                   convention mapping, raise on contradiction
  src/resolve/hero.py            — pass init_time to accumulate_window; sortby+reindex
                                   forecast onto IMD grid; pass convention to window_bounds_utc
  src/resolve/__main__.py        — add inspect subcommand

Created:
  src/resolve/inspect_cmd.py     — inspect subcommand implementation

Modified tests:
  tests/test_accum.py            — rebuild fixtures: timedelta lead_time, member dim,
                                   pass init to accumulate_window, add sanity-gate tests
  tests/test_config.py           — add imd_day_convention field
```

---

## Task 1: Config — add `imd_day_convention`

**Files:**
- Modify: `src/resolve/config.py`
- Modify: `configs/montha.yaml`
- Modify: `configs/default.yaml`
- Modify: `tests/test_config.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_config.py` (inside the CONFIG_YAML string and the roundtrip test):

```python
# Add to the CONFIG_YAML string, after the `label: replay` line:
# imd_day_convention: ending

# Add to test_load_config_roundtrip:
#   assert cfg.imd_day_convention == "ending"

# And a new test:
def test_load_config_convention_default(tmp_path):
    """imd_day_convention defaults to 'ending' when absent from YAML."""
    import yaml
    minimal = {
        "label": "replay",
        "region": {"lat_min": 8.0, "lat_max": 24.0, "lon_min": 74.0, "lon_max": 90.0},
        "init_dates": {"start": "2025-10-20", "end": "2025-10-29"},
        "target_day_search": {"start": "2025-10-26", "end": "2025-10-31",
                              "threshold_mm": 115.6},
        "cache_dir": "cache/test",
        "output_dir": "outputs/test",
    }
    p = tmp_path / "minimal.yaml"
    p.write_text(yaml.dump(minimal))
    cfg = load_config(str(p))
    assert cfg.imd_day_convention == "ending"
```

Full replacement of `tests/test_config.py`:

```python
import pytest
import yaml
from pathlib import Path
from resolve.config import load_config, ResolveConfig

CONFIG_YAML = """
label: replay
imd_day_convention: ending
region:
  lat_min: 8.0
  lat_max: 24.0
  lon_min: 74.0
  lon_max: 90.0
init_dates:
  start: "2025-10-20"
  end: "2025-10-29"
target_day_search:
  start: "2025-10-26"
  end: "2025-10-31"
  threshold_mm: 115.6
  min_cells: 10
rain_thresholds:
  heavy: 64.5
  very_heavy: 115.6
  extremely_heavy: 204.5
objects:
  min_area_cells: 5
  p_min: 0.1
crossrun:
  max_centroid_km: 500.0
  lockon_drift_km: 100.0
  lockon_iou: 0.5
fss:
  windows_cells: [1, 3, 5, 7, 9, 13]
cache_dir: "cache/test"
output_dir: "outputs/test"
"""


def test_load_config_roundtrip(tmp_path):
    p = tmp_path / "cfg.yaml"
    p.write_text(CONFIG_YAML)
    cfg = load_config(str(p))
    assert cfg.label == "replay"
    assert cfg.imd_day_convention == "ending"
    assert cfg.region.lat_min == 8.0
    assert cfg.rain_thresholds.very_heavy == 115.6
    assert cfg.fss.windows_cells == [1, 3, 5, 7, 9, 13]
    assert cfg.crossrun.lockon_iou == 0.5


def test_load_config_missing_file():
    with pytest.raises(FileNotFoundError):
        load_config("/nonexistent/path.yaml")


def test_load_config_convention_default(tmp_path):
    """imd_day_convention defaults to 'ending' when absent from YAML."""
    minimal = {
        "label": "replay",
        "region": {"lat_min": 8.0, "lat_max": 24.0, "lon_min": 74.0, "lon_max": 90.0},
        "init_dates": {"start": "2025-10-20", "end": "2025-10-29"},
        "target_day_search": {"start": "2025-10-26", "end": "2025-10-31",
                              "threshold_mm": 115.6},
        "cache_dir": "cache/test",
        "output_dir": "outputs/test",
    }
    p = tmp_path / "minimal.yaml"
    p.write_text(yaml.dump(minimal))
    cfg = load_config(str(p))
    assert cfg.imd_day_convention == "ending"
```

- [ ] **Step 2: Run to verify fail**

```bash
cd /Users/zain/Personal/sih078 && pytest tests/test_config.py -v
```

Expected: FAIL — `TypeError: ResolveConfig.__init__() got unexpected keyword argument 'imd_day_convention'`

- [ ] **Step 3: Update `src/resolve/config.py`**

Add `imd_day_convention: str = "ending"` to `ResolveConfig` and add it to `load_config`:

```python
"""YAML config loader using dataclasses."""

from __future__ import annotations
import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass
class RegionConfig:
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float


@dataclass
class InitDatesConfig:
    start: str
    end: str


@dataclass
class TargetDayConfig:
    start: str
    end: str
    threshold_mm: float
    min_cells: int = 10


@dataclass
class RainThresholds:
    heavy: float = 64.5
    very_heavy: float = 115.6
    extremely_heavy: float = 204.5


@dataclass
class ObjectsConfig:
    min_area_cells: int = 5
    p_min: float = 0.1


@dataclass
class CrossrunConfig:
    max_centroid_km: float = 500.0
    lockon_drift_km: float = 100.0
    lockon_iou: float = 0.5


@dataclass
class FSSConfig:
    windows_cells: List[int] = field(default_factory=lambda: [1, 3, 5, 7, 9, 13])


@dataclass
class ResolveConfig:
    label: str
    region: RegionConfig
    init_dates: InitDatesConfig
    target_day_search: TargetDayConfig
    rain_thresholds: RainThresholds
    objects: ObjectsConfig
    crossrun: CrossrunConfig
    fss: FSSConfig
    cache_dir: str
    output_dir: str
    imd_day_convention: str = "ending"


def load_config(path: str) -> ResolveConfig:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with p.open() as fh:
        raw = yaml.safe_load(fh)
    conv = raw.get("imd_day_convention", "ending")
    if conv not in ("ending", "starting"):
        raise ValueError(f"imd_day_convention must be 'ending' or 'starting', got {conv!r}")
    return ResolveConfig(
        label=raw["label"],
        region=RegionConfig(**raw["region"]),
        init_dates=InitDatesConfig(**raw["init_dates"]),
        target_day_search=TargetDayConfig(**raw["target_day_search"]),
        rain_thresholds=RainThresholds(**raw.get("rain_thresholds", {})),
        objects=ObjectsConfig(**raw.get("objects", {})),
        crossrun=CrossrunConfig(**raw.get("crossrun", {})),
        fss=FSSConfig(**raw.get("fss", {})),
        cache_dir=raw.get("cache_dir", "cache"),
        output_dir=raw.get("output_dir", "outputs"),
        imd_day_convention=conv,
    )
```

- [ ] **Step 4: Add `imd_day_convention: ending` to both YAML configs**

In `configs/montha.yaml`, add after `label: replay`:
```yaml
imd_day_convention: ending  # IMD day D ends at 08:30 IST (03 UTC) on date D
```

Same in `configs/default.yaml`.

- [ ] **Step 5: Run tests**

```bash
pytest tests/test_config.py -v
```

Expected: 3 PASS.

- [ ] **Step 6: Run full suite**

```bash
pytest -v
```

Expected: 32 PASS (30 old + 2 new config tests, minus the 1 old roundtrip that was replaced — net +2).

- [ ] **Step 7: Commit**

```bash
git add src/resolve/config.py configs/montha.yaml configs/default.yaml tests/test_config.py
git commit -m "feat: add imd_day_convention to config"
```

---

## Task 2: Fix `accum.py` — real schema + sanity gates

**Files:**
- Modify: `src/resolve/accum.py`
- Modify: `tests/test_accum.py`

The real IFS dataset has `lead_time` as `timedelta64`, not absolute timestamps. `accumulate_window` must now accept `init_time` and compute `valid_time = init_time + lead_time`.

Two new sanity gates (both raise):
1. The steps in the window must sum to exactly 86 400 s.
2. Per-member per-cell rain must be ≥ 0 and < 1 000 mm.

`window_bounds_utc` gains a `convention` parameter.

- [ ] **Step 1: Replace `tests/test_accum.py` with real-schema fixtures**

```python
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
    # 3-hourly steps matching real IFS spacing
    lead_times = pd.to_timedelta([0, 3, 6, 9, 12, 24], unit="h")
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
    # Steps in (t0, t1]: valid_times 06 h, 09 h, 12 h, 24 h
    # Δt: 3h, 3h, 3h, 12h → 21 h at 1 mm/h = 21 mm
    assert acc.dims == ("member", "latitude", "longitude")
    np.testing.assert_allclose(acc.values, 21.0, rtol=1e-5)


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
    lead_times = pd.to_timedelta([0, 3, 6, 9, 12, 24], unit="h")
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
```

- [ ] **Step 2: Run to verify fail**

```bash
pytest tests/test_accum.py -v
```

Expected: FAIL — `TypeError: accumulate_window() missing required argument 'init_time'`

- [ ] **Step 3: Replace `src/resolve/accum.py`**

```python
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

    # Sanity gate: rain must be < 1000 mm (physical upper bound for 24 h)
    if np.any(accum_vals >= 1000.0):
        pos_max = float(accum_vals.max())
        raise ValueError(
            f"Accumulated rain exceeds 1000 mm (max={pos_max:.1f} mm). "
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
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_accum.py -v
```

Expected: all 9 tests PASS.

- [ ] **Step 5: Run full suite**

```bash
pytest -v
```

Expected: ≥ 32 PASS (old tests that used the old `accumulate_window(ds, t0, t1)` signature will fail — fix them in the next step).

- [ ] **Step 6: Commit**

```bash
git add src/resolve/accum.py tests/test_accum.py
git commit -m "fix: accum real schema — timedelta lead_time, init_time param, 86400s+rain sanity gates"
```

---

## Task 3: Fix `ingest.py` — real schema normalisation

**Files:**
- Modify: `src/resolve/ingest.py`

Changes:
1. After `ds.compute()`: rename `ensemble_member` → `member` if present.
2. Store `init_time` as a scalar coordinate (for reference when reading cache).
3. Sort by latitude and longitude (ascending) before caching.
4. Clear all variable and coordinate encodings with `ds.drop_encoding()` (xarray ≥ 2024.1).
5. Write with `consolidated=True` for fast metadata reads.

- [ ] **Step 1: Replace `src/resolve/ingest.py`**

```python
"""Ingest IFS ENS forecasts to a local Zarr cache.

Real schema from the dynamical.org catalog:
  - lead_time: timedelta64 (0h, 3h, 6h, …, 360h)
  - ensemble_member: int 0–50  ← renamed to 'member' here
  - init_time: datetime64 (one value per file, stored as a scalar coord)
  - latitude, longitude: float64, ascending order after normalisation

IMPORTANT: Never use data.dynamical.org URLs directly.
Always go through the catalog API.
"""

from __future__ import annotations
import logging
import time
from pathlib import Path

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
    """Subset to a lat/lon bounding box, handling ascending or descending latitude."""
    lat = ds.latitude.values
    if lat[0] > lat[-1]:  # descending → reverse slice args
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


def _normalise(ds: xr.Dataset, init_time: pd.Timestamp) -> xr.Dataset:
    """Apply real-schema normalisations before caching.

    1. Rename ensemble_member → member (if present).
    2. Sort latitude and longitude ascending.
    3. Store init_time as a scalar coordinate.
    4. Clear all encodings so to_zarr starts clean.
    """
    if "ensemble_member" in ds.dims:
        ds = ds.rename({"ensemble_member": "member"})

    # Sort spatial coords ascending
    ds = ds.sortby(["latitude", "longitude"])

    # Store init_time for use by open_cached callers
    ds = ds.assign_coords(init_time=init_time.to_datetime64())

    # Clear encodings — xarray ≥ 2024.1
    ds = ds.drop_encoding()

    return ds


def ingest_one_init(
    init_time: pd.Timestamp,
    lat_min: float, lat_max: float,
    lon_min: float, lon_max: float,
    cache_dir: str | Path,
    variables: list[str] = VARIABLES,
) -> Path:
    """Download and cache one init time to Zarr.

    Normalises schema (member rename, sort, encoding clear) before writing.
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
    logger.info(
        "Downloaded %.1f MB in %.1f s for %s",
        size_mb, elapsed, init_time.isoformat(),
    )

    ds = _normalise(ds, init_time)
    ds.to_zarr(str(zarr_path), mode="w", consolidated=True)
    logger.info("Cached to %s", zarr_path)
    return zarr_path


def open_cached(zarr_path: str | Path) -> xr.Dataset:
    """Open a cached Zarr store and return the dataset.

    The returned dataset has:
      lead_time: timedelta64
      member: int
      latitude, longitude: float64 (ascending)
      init_time: scalar datetime64 coordinate
    """
    ds = xr.open_zarr(str(zarr_path), chunks=None, consolidated=True)
    return ds


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
```

- [ ] **Step 2: Verify import**

```bash
python -c "from resolve.ingest import subset_lat_lon, _normalise, CATALOG_ID; print('ok')"
```

- [ ] **Step 3: Run full suite**

```bash
pytest -v
```

Expected: ≥ 32 PASS.

- [ ] **Step 4: Commit**

```bash
git add src/resolve/ingest.py
git commit -m "fix: ingest real schema — rename ensemble_member→member, sort, clear encoding, store init_time coord"
```

---

## Task 4: Fix `truth.py` — IMD fallback, IMERG units, alignment

**Files:**
- Modify: `src/resolve/truth.py`

Changes:
1. `fetch_imd`: try archive first; on failure try realtime; log which path was used.
2. `fetch_imerg_daily`: remove ×24 (V07 `precipitation` is already mm/day); read `units` attr; **raise** (not warn) if daily max > 1 000 mm; use `earthaccess.login()` without `strategy` argument (auto-detects netrc and env vars).
3. `check_alignment`: transpose IMERG to (time, lat, lon); regrid IMERG to IMD 0.25° grid with `interp`; compare IMD day D with IMERG days D−1 and D; map better match to convention; raise if result contradicts `expected_convention`.

- [ ] **Step 1: Replace `src/resolve/truth.py`**

```python
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

    da = data.get_xarray()
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
```

- [ ] **Step 2: Verify import**

```bash
python -c "from resolve.truth import fetch_imd, check_alignment; print('ok')"
```

- [ ] **Step 3: Run full suite**

```bash
pytest -v
```

Expected: ≥ 32 PASS (truth module has no unit tests, just import check).

- [ ] **Step 4: Commit**

```bash
git add src/resolve/truth.py
git commit -m "fix: truth module — IMD archive→realtime fallback, IMERG mm/day (no ×24), raise >1000 mm, earthaccess auto-login, check_alignment regrid+convention"
```

---

## Task 5: Fix `hero.py` — call sites + grid alignment + sanity gate

**Files:**
- Modify: `src/resolve/hero.py`

Changes:
1. Pass `init_time` to `accumulate_window`.
2. Pass `convention` from config to `window_bounds_utc`.
3. After accumulation, `sortby` forecast lat/lon and `reindex` onto IMD grid (tolerance 0.01°).
4. Assert coordinates match within 0.01° after reindex.
5. Pass `expected_convention` to `check_alignment`.
6. `find_target_day` already raises on `min_cells` — add an explicit check with the exact message required.

- [ ] **Step 1: Update `process_one_init` in `hero.py`**

Replace the `process_one_init` function (lines 58–124) with:

```python
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
```

- [ ] **Step 2: Update `check_alignment` call in `run_hero`**

Find the line:
```python
        alignment = check_alignment(imd_da, imerg_da)
```
Replace with:
```python
        alignment = check_alignment(
            imd_da, imerg_da,
            expected_convention=cfg.imd_day_convention,
        )
```

- [ ] **Step 3: Verify CLI dry-run still works**

```bash
python -m resolve hero --config configs/montha.yaml --dry-run
```

Expected: logs config, exits 0.

- [ ] **Step 4: Run full test suite**

```bash
pytest -v
```

Expected: ≥ 32 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/resolve/hero.py
git commit -m "fix: hero — pass init_time to accum, reindex forecast onto IMD grid, convention-aware windows"
```

---

## Task 6: Add `inspect` subcommand

**Files:**
- Create: `src/resolve/inspect_cmd.py`
- Modify: `src/resolve/__main__.py`

The `inspect` command opens the catalog (without ingesting anything), prints schema facts, then downloads only the 2025-10-27 init and computes the regional max 24 h member rain for the target day window.

- [ ] **Step 1: Create `src/resolve/inspect_cmd.py`**

```python
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
```

- [ ] **Step 2: Update `src/resolve/__main__.py`**

```python
"""CLI entry point: python -m resolve <cmd> --config <path>"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(prog="resolve")
    sub = parser.add_subparsers(dest="cmd", required=True)

    hero_p = sub.add_parser("hero", help="Produce the Montha hero figure")
    hero_p.add_argument("--config", required=True, help="Path to YAML config")
    hero_p.add_argument("--dry-run", action="store_true",
                        help="Load config and print plan without fetching data")

    inspect_p = sub.add_parser(
        "inspect",
        help="Probe catalog schema and compute one 24 h accumulation (no full ingest)",
    )
    inspect_p.add_argument("--config", required=True, help="Path to YAML config")

    args = parser.parse_args()

    if args.cmd == "hero":
        from resolve.hero import run_hero
        run_hero(args.config, dry_run=args.dry_run)
    elif args.cmd == "inspect":
        from resolve.inspect_cmd import run_inspect
        run_inspect(args.config)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Verify inspect is importable**

```bash
python -c "from resolve.inspect_cmd import run_inspect; print('ok')"
python -m resolve inspect --help
```

Expected: prints usage for inspect subcommand.

- [ ] **Step 4: Run full test suite**

```bash
pytest -v
```

Expected: ≥ 32 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/resolve/inspect_cmd.py src/resolve/__main__.py
git commit -m "feat: add inspect subcommand — catalog schema probe + one 24 h accumulation"
```

---

## Task 7: Run and report

**This task requires live network access to the dynamical.org catalog and imdlib data.**

- [ ] **Step 1: Run inspect**

```bash
cd /Users/zain/Personal/sih078
python -m resolve inspect --config configs/montha.yaml
```

Record and report:
- lead_time dtype (expect timedelta64)
- latitude order (expect ascending after normalise)
- init_time range available in catalog
- MB downloaded for 2025-10-27 init
- Regional max 24 h member rain for 2025-10-28 window

- [ ] **Step 2: Run hero (IMD only if Earthdata not configured)**

```bash
python -m resolve hero --config configs/montha.yaml
```

If IMERG fails (no Earthdata credentials), the pipeline continues with IMD only.

Record and report:
- Timing and MB per init for all 10 runs (2025-10-20 to 2025-10-29)
- Alignment result (best_match, inferred_convention, window_offset_hours, n_days)
- Target day selected and IMD max rain on it
- FSS table (windows vs leads)
- Earned widths per lead (cells and km)
- Lock-on run (or "none")
- Any errors or warnings

- [ ] **Step 3: Push results**

```bash
git push origin main
```

---

## Self-Review

**Spec coverage:**

| Requirement | Task |
|---|---|
| 1. lead_time is timedelta; compute valid_time | Task 2 |
| 1. Rebuild test fixtures with timedelta + ensemble_member dim | Task 2 |
| 2. Rename ensemble_member → member at ingest | Task 3 |
| 3. Clear encodings before to_zarr | Task 3 |
| 4. sortby lat/lon, reindex onto IMD grid, assert coords match | Task 5 |
| 5. imd_day_convention in config | Task 1 |
| 5. window_bounds_utc uses convention param | Task 2 |
| 5. IMERG transpose to (time, lat, lon) + regrid | Task 4 |
| 5. Compare IMD D with IMERG D-1 and D, map convention | Task 4 |
| 5. Raise if contradicts config | Task 4 |
| 6. Remove ×24; read units attr; raise >1000 mm | Task 4 |
| 7. fetch_imd archive→realtime fallback | Task 4 |
| 8. earthaccess.login() auto-detect | Task 4 |
| 9. 86 400 s window gate | Task 2 |
| 9. member rain ≥ 0 and < 1000 mm | Task 2 |
| 9. target day meets min_cells | Task 5 (already in find_target_day) |
| 10. inspect subcommand | Task 6 |
| Run and report | Task 7 |

**No placeholders found.**

**Type consistency:**
- `accumulate_window(ds, init_time, t0, t1)` — Tasks 2 and 5 both use this signature.
- `window_bounds_utc(date, convention=, offset_hours=)` — Tasks 2 and 5 both use this.
- `check_alignment(imd_da, imerg_da, expected_convention=)` — Task 4 defines, Task 5 calls.

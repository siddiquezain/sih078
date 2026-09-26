# Resolve UI data contract (`resolve-ui/1`)

`python -m resolve export-ui --config configs/montha.yaml` writes `ui/data/<event>.js`:

```js
window.RESOLVE_DATA = { ...object below... };
```

It is a plain script, not JSON loaded with `fetch()`, so `ui/index.html` works from `file://` with no server and no network.

The UI never computes science. Every number it shows comes from this file.

## Top level

| Key | Type | Meaning |
| --- | --- | --- |
| `schema` | `"resolve-ui/1"` | The UI refuses any other value. |
| `meta` | object | Labels and provenance (below). |
| `grid` | object | `{lat0, lon0, step, nlat, nlon}`. Cell centres on the IMD 0.25° grid. Row 0 is the southernmost row (`lat0`) and column 0 the westernmost (`lon0`). Every array below is row-major: `index = row * nlon + col`. |
| `land` | int[nlat·nlon] | 1 where the IMD grid has data (the verification land mask), else 0. |
| `observed` | object | `{cells, n, f_obs, max_mm}`. `cells` holds the flat indices of IMD cells ≥ threshold on the target day. `n = len(cells)`. `f_obs = n / land cells`. `max_mm` is the wettest IMD cell in the domain, or `null`. |
| `windows` | object[] | `{cells, km}` for every FSS window, ascending. `km = round(cells × 0.25 × 111.32)`. |
| `runs` | object[] | One per init, oldest first (below). Skip inits whose window is not fully covered, as `run_hero` already does. |
| `lockon_init` | string or null | ISO init of the lock-on run, e.g. `"2025-10-26T00:00Z"`. |
| `track` | object[] | Observed cyclone positions: `{time, lat, lon, label, source}`. Use IBTrACS or IMD best track. If only IMD bulletin positions are available, set `source` to say so. Leave it empty rather than guess. |

## `meta`

| Key | Example | Notes |
| --- | --- | --- |
| `preview` | `false` | **Must be `false` in exported files.** `true` shows the placeholder banner. |
| `label` | `"replay"` | `"replay"` or `"live"` (hard rule 5). |
| `event` | `"Cyclone Montha"` | Shown in the header. |
| `hazard` | `"Very heavy rain"` | |
| `threshold_mm` | `115.6` | |
| `target_date` | `"2025-10-29"` | IMD rain-day label. |
| `imd_day_convention` | `"ending"` | Must match the value verified in the config. |
| `rain_day` | `"24 h to 08:30 IST on 29 Oct 2025"` | Human string derived from the convention. |
| `forecast` | `"ECMWF IFS ENS, 0.25°, 51 members"` | |
| `forecast_credit` | `"ECMWF open data (CC BY 4.0), processed by dynamical.org"` | |
| `truth` | `"IMD 0.25° gridded daily rainfall"` | |
| `members` | `51` | Denominator for every count below. |
| `generated_utc` | `"2026-09-26T10:12:00Z"` | |
| `commit` | `"ba0a405"` | `git rev-parse --short HEAD`. |

## `runs[]`

| Key | Type | Meaning |
| --- | --- | --- |
| `init` | string | `"YYYY-MM-DDT00:00Z"`. |
| `lead_days` | int | `(target_date − init).days`. This is the D−n label. |
| `lead_hours` | int | Hours from init to the end of the rain-day window. |
| `window_offset_hours` | int | `0` or `-3`, as used in accumulation. |
| `members_valid` | int | Members actually read. Normally equals `meta.members`. |
| `peak_count` | int | `max(counts)`. |
| `counts` | int[nlat·nlon] | Members with 24 h rain ≥ threshold, per cell, on the IMD grid after reindexing. Probability = `counts / members`. |
| `nmep` | `{ "<n>": int[] }` | One array for **every** window `n > 1`. The domain is cut into n×n tiles anchored at row 0, col 0. Edge tiles are partial. Tiles are row-major, `ceil(nlat/n) × ceil(nlon/n)`. Each value is the number of members with ≥ threshold in **at least one** cell of that tile (neighbourhood maximum ensemble probability, Schwartz & Sobash 2017). Compute it from member masks, not from `counts`. |
| `fss` | float[] | FSS per window, aligned with `windows`. |
| `useful` | float | `0.5 + f_obs / 2` for this run. |
| `earned_cells` | int or null | Smallest window with `fss ≥ useful`, or null. |
| `object` | object or null | The **tracked event** footprint for this run: `{lat, lon, n_cells, area_km2}`. `lat, lon` is the count-weighted centroid of its cells. |
| `drift_km` | float or null | Centroid shift from the previous run's tracked object. |
| `iou` | float or null | Footprint (mask) IoU with the previous run's tracked object, not bounding-box IoU. |
| `locked` | bool | True from the lock-on run onward. |

## Size

Counts are small ints, so the whole file for 9 runs × 65×65 cells is about 300 KB. Do not round probabilities to floats. Keep integer member counts.

## Checks the exporter must run before writing

- `len(counts) == nlat*nlon` for every run.
- Every `nmep` array has length `ceil(nlat/n)*ceil(nlon/n)`.
- `max(counts) == peak_count`.
- `0 ≤ values ≤ members`.
- For every n: `max(nmep[n]) ≥ max(counts)`, since a tile can never be less likely than its wettest cell.
- `observed.n` matches the IMD count used by `find_target_day`.
- `meta.preview is False`.

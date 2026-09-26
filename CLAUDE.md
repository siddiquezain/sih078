# CLAUDE.md — Resolve (SIH 2026 · PS 26078 · MoES/NCMRWF)

## Mission
Resolve tracks extreme-rain and cyclone events in 3–10-day ensemble forecasts.
- It draws each event's footprint only as finely as past forecasts have been verifiably skilful at that lead time ("earned resolution").
- It follows each event across lead times, ensemble members and successive daily runs.
- It ends in an IMD-coloured CAP draft that a forecaster approves.

Context:
- The user is the duty forecaster at NCMRWF/IMD.
- We post-process forecasts. We never build a weather model.
- The judges are NCMRWF scientists: correctness and honesty beat features.

## Hard rules
1. Real data only in anything shown (figures, API, dashboard). Synthetic data only inside tests.
2. Never fabricate or estimate a number that should be measured. If a data source fails, stop, show the exact error and propose a fallback.
3. No leakage: output for init time I uses only data available at I. Never evaluate on training periods or events.
4. Every output gets a provenance JSON sidecar with:
   - dataset ids and variables
   - init times
   - thresholds and window offsets
   - git commit and access time
5. Labelling:
   - Every output is labelled "replay" or "live".
   - 5 km output is labelled "scenario", never "forecast".
6. Out of scope:
   - LLMs and chatbots
   - blockchain and hash chains
   - auth, microservices, Kubernetes or Kafka, mobile apps
   - our own forecast model
   - analog retrieval and cost-loss engines
7. Physics constraints must be testable: non-negative rain and exact block-mean conservation. No wind-divergence or moisture-convergence loss terms.
8. If anything here is scientifically wrong, say so before building it. Log every deviation in docs/DECISIONS.md.

## Environment
- Check where you are running (`uname -sm`, RAM).
  - My laptop is an Intel Ivy Bridge MacBook Air: no GPU, no AVX2, little RAM. There, write code and run tests on tiny arrays only. Do not install torch, CUDA or other heavy stacks.
  - On Linux with ≥ 8 GB RAM (e.g., a GitHub Codespace), run Phases 0–2 end to end.
  - GPU work (Phases 3 and 5) runs in Colab or Kaggle notebooks.
- Project layout:
  - Python 3.11, installed with `pip install -e .`.
  - Package in `src/resolve/`; CLI is `python -m resolve <cmd>`; configs in `configs/*.yaml`.
  - Every phase gets a notebook in `notebooks/` that installs the repo and runs top to bottom in Colab.
- Dependencies (ask before adding anything else):
  - Core: xarray, zarr, dask, numpy, scipy, pandas, matplotlib, pyyaml, dynamical-catalog>=0.8.0, imdlib, earthaccess, pytest.
  - Optional: cartopy. Plots must work without it.
  - Phase 3: torch, torch-geometric, scikit-learn, xgboost.
  - Phase 4: fastapi, uvicorn, lxml.
- Never commit data, outputs or credentials. Earthdata credentials come from env vars or ~/.netrc.

## Data (verified — use exactly these)

**Forecasts: ECMWF IFS ENS via dynamical.org**
- Open with `dynamical_catalog.open("ecmwf-ifs-ens-forecast-15-day-0-25-degree", chunks=None)`.
- Dims: init_time × lead_time × ensemble_member (0–50) × latitude × longitude.
- Grid and timing: 0.25°; 00 UTC inits from 2024-04-01; lead 0–360 h, 3-hourly to 144 h, then 6-hourly.
- Variables:
  - precipitation_surface: mean rate since the previous step, kg m-2 s-1 = mm/s
  - pressure_reduced_to_mean_sea_level (Pa)
  - wind_u_10m, wind_v_10m (m/s)
  - wind_gust_10m
  - temperature_2m (°C)
- Handling:
  - Check latitude order before slicing.
  - Subset the region before loading, and process one init at a time.
  - Never use data.dynamical.org URLs; they stop working on 30 Sep 2026.
- Rain accumulation: rain over (t_{k−1}, t_k] = rate_k × (t_k − t_{k−1}) seconds → mm.

**Rain truth: IMD 0.25° daily gridded rainfall via imdlib**
- Archive: `get_data('rain', yr, yr, fn_format='yearwise', file_dir=...)`.
- If the archive is missing, use real-time: `get_real_data('rain', start, end, file_dir)`.
- Convert with `.get_xarray()`.
- Land only (NaN over sea). IMD's rain day is the 24 h ending 08:30 IST (03 UTC).

**Satellite truth: GPM IMERG V07 via earthaccess**
- Phase 1 uses daily Final `GPM_3IMERGDF`. Its days run 00–00 UTC, a 3 h offset from IMD; record it.
- Later phases use half-hourly `GPM_3IMERGHH` for exact windows.
- Resolution 0.1°. IMERG is weaker over complex terrain and coasts; say so on figures.

**Cyclone tracks**
- IBTrACS v04r01 north Indian Ocean CSV (NCEI).
- If Montha 2025 is missing, use IMD bulletin positions. IMD placed the centre near 14.9°N, 82.9°E at 08:30 IST on 28 Oct 2025. It crossed the Andhra coast near Kakinada that night.

**IMD 24 h rain thresholds**
- heavy ≥ 64.5 mm
- very heavy ≥ 115.6 mm
- extremely heavy ≥ 204.5 mm

**Districts (Phase 4)**
- Use current boundaries. Andhra Pradesh was reorganised into 26 districts in 2022, and Telangana's were redrawn from 2016.
- Census-2011 polygons (e.g., datameet) give wrong names for 2025 alerts.
- If no current source with a clear licence exists, fall back to IMD meteorological subdivisions and log it.

**Later phases**
- GEFSv12 reforecast 2000–2019 on AWS (model climate)
- GEFS (second model)
- CHIRPS v2 0.05° daily (Phase 5 target)
- NEPS-G GRIB2 through an adapter, only if NCMRWF shares data

**README attributions**
- ECMWF open data (CC BY 4.0 + ECMWF terms), processed by dynamical.org
- IMD
- NASA GES DISC (IMERG)
- IBTrACS

## Rain-day alignment
- Forecast 24 h windows end at 03 UTC to match IMD. This is exact up to 144 h.
- Beyond 144 h (6-hourly steps), use windows ending 00 UTC and record window_offset_hours = −3 in provenance.
- Verify IMD's date label empirically: correlate IMD day D against IMERG days D and D−1, use the better match and log it.

## Phase 0 — Setup
- Build the repo skeleton: pyproject, .gitignore, config loader, provenance helper, logging, pytest.
- Write PLAN.md with your file list for Phases 1–4 and any objections to this brief.
- Done when `pytest` passes locally with core deps only.

## Phase 1 — Montha hero figure (TOP PRIORITY: needed for the idea PDF within days)
Scope:
- Runs: 00 UTC inits 2025-10-20 → 2025-10-29.
- Region: 8–24°N, 74–90°E.
- Target day(s): pick from the truth data the IMD day(s) between 26 and 31 Oct with the largest area ≥ 115.6 mm in the region. Don't assume.

Steps:
1. **ingest:** subset the runs (rain, MSLP, 10 m wind) into a local Zarr cache. Log sizes and timings.
2. **truth:** fetch the IMD grid and IMERG daily for 20 Oct–1 Nov 2025. Run the alignment check.
3. **accumulate:** 24 h member rain, following the alignment rule.
4. **exceedance:** member masks at the three thresholds. Raw probability = member fraction.
5. **objects:** 8-connected components (min area in config) on the member masks and on the probability field (P ≥ p_min, default 0.1). Features: rain-weighted centroid, area in km² (cos-lat), max rain, bbox.
6. **crossrun:** for each target day, match the event footprint across successive runs (Hungarian on 1 − IoU, gated by centroid distance).
   - Per run, record: drift from the previous run (km, haversine), IoU with the previous run, and max and area-mean probability.
   - Lock-on = the first run from which drift ≤ 100 km and IoU ≥ 0.5 hold for every later run (thresholds in config). Report "none" if never.
7. **fss:** land-masked neighbourhood FSS.
   - Fractions = uniform_filter(x·mask) / uniform_filter(mask), using the same mask on forecast and truth.
   - Forecast fraction = neighbourhood mean of ensemble probability. Observed fraction = neighbourhood mean of binary truth.
   - FSS = 1 − mean((Ff − Fo)²) / (mean(Ff²) + mean(Fo²)).
   - Windows: n ∈ {1, 3, 5, 7, 9, 13} cells on the 0.25° grid (report km). Use n = 1 on IMERG 0.1° for the ~10 km rung.
   - Useful-skill line = 0.5 + f_obs/2.
8. **earned:** earned width per lead = smallest n with FSS ≥ the useful line, or "no skill" if none. This is a single-event point estimate; label it that way.
9. **hero figure** (PNG 300 dpi + SVG):
   - Panel A: footprints from the D-7, D-4, D-2 and D-1 runs, each at its earned width, with the observed ≥ 115.6 mm contour and the track.
   - Panel B: FSS vs width, one line per lead, with the useful-skill line.
   - Panel C: run-to-run drift (km) and event probability by init date, with lock-on marked.
   - Caption: "Single event (Montha, Oct 2025); full event library before the finale."
10. **summary and reproducibility:**
    - Write `outputs/montha/summary.json` with earned width per lead, lock-on run, the FSS table, cell counts and offsets.
    - Both `python -m resolve hero --config configs/montha.yaml` and `notebooks/01_montha_hero.ipynb` must reproduce everything.

Tests:
- FSS: identical fields → 1; displaced blobs → FSS rises with n; mask normalisation.
- Rate→mm accumulation and window alignment.
- Haversine, IoU and matching.
- Leakage guard: outputs for init I are unchanged when all other inits are set to NaN.

If IMERG blocks progress, ship the figure on IMD truth alone (no 10 km rung) and say so.

STOP after Phase 1 and report: timings, data volumes, offsets, any suspicious numbers, and what you would change.

## Phase 2 — Event library, skill tables, earned-resolution engine
- **Library rule** (fixed before viewing results):
  - Every IMD-named cyclone since 2024-04-01.
  - Every IMD day with ≥ N land cells ≥ 204.5 mm (N = 5 default).
  - Merge events within 2 days and 500 km. Output event_library.csv.
- **Skill tables:**
  - Replay runs D-7…D-0 for each event. Pool FSS(n, lead) into lead bands (days 1–3, 4–6, 7–10).
  - Bootstrap by event (1,000 resamples).
  - Earned width = smallest n whose lower 90% bound clears the useful line.
  - Hold one rung coarser while run-to-run drift exceeds the width, until two consecutive runs agree.
  - Add a random-forecast reference line beside 0.5 + f/2.
- **Raw-ensemble verification:** Brier, BSS vs climatology, reliability diagram, ROC area.
- **EFI/SOT against a pseudo model-climate:** archived runs within ±15 days of the calendar date, from other years only, per lead and cell. Report the sample size. The GEFSv12 path is optional.
- **Cyclone tracker per member:** MSLP minimum near the previous position, with closed-contour and wind checks. Report track and landfall error vs best track.

## Phase 3 — Calibrator (Colab/Kaggle GPU)
- **Target:** P(obs ≥ 64.5 / 115.6 / 204.5 mm) per IMD land cell, leads 1–10 days.
- **Baselines first:** raw member fraction; logistic regression per lead; XGBoost + isotonic.
- **Then a GNN (PyTorch Geometric):**
  - Nodes: IMD land cells.
  - Edges: neighbours ≤ 100 km, plus pooled super-nodes for 300–500 km context.
  - Layers: attention.
  - Features: ensemble mean, spread, member exceedance fractions, p90 and max member, EFI/SOT, elevation, slope, distance to coast, day of year, lead embedding.
  - Outputs: three monotone probabilities.
  - Training: class-weighted BCE, then isotonic recalibration on validation.
- **Splits:**
  - train 2024-04 → 2025-06
  - validate 2025-07 → 2025-09
  - test 2025-10 onward (Montha is test data)
- **Reporting:** Brier, BSS, ECE, reliability and ROC area with bootstrap CIs for every model. If the GNN loses, say so and ship the winner.

## Phase 4 — API, CAP drafts, replay dashboard
- **FastAPI endpoints:**
  - /events
  - /events/{id}
  - /events/{id}/runs/{init}
  - /skill
  - /cap/{event_id}
- **CAP 1.2** (namespace urn:oasis:names:tc:emergency:cap:1.2):
  - status "Draft" (never "Actual"), msgType "Alert", scope "Public", category "Met".
  - IMD colour as a parameter.
  - Areas = current districts touched by the footprint at its earned width.
  - Validate against the official OASIS CAP 1.2 XSD in tests.
  - "Approve" only flips a local flag.
- **Dashboard:** static HTML/JS served by FastAPI (MapLibre GL or Leaflet, no build step).
  - Controls: issue-date slider, lead selector, truth overlay toggle.
  - Map: footprint at earned width with a km ladder legend.
  - Panels: lock-on strip; event card (probability, earned width, lock-on, track record).
  - Replay/live badge. Works offline from cached outputs. "Live" processes the latest 00 UTC run and shows elapsed time.

## Phase 5 (optional) — 5 km rain scenarios (GPU)
- **Model:** start from the CorrDiff-Mini configs in NVIDIA PhysicsNeMo's CorrDiff recipe.
- **Target:** CHIRPS 0.05° daily over 12–22°N, 76–86°E.
- **Splits:** train 1981–2019, validate 2020–21, test 2022+.
- **Inputs:** quantile-mapped coarse rain, 0.05° elevation, land mask, day of year; 128×128 patches.
- **Conservation:** exact, using a multiplicative or softmax constraint layer (Harder et al., JMLR 2023).
- **Comparison:** against bicubic and UNet-only on CRPS, power-spectrum ratio, 99th/99.9th percentile ratios and conservation error.
- **Use:** run only where earned width ≤ 25 km, and label the output "scenario".

## Working style
- Make small commits, with tests for every module.
- After each phase, update the README status table (Implemented / Prototype / Planned) and report measured results with n.
- Prefer boring, readable code.
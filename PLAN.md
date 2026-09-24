# Resolve — Phase File List and Objections

Generated: 2026-09-24

## Phase 0 (complete)
- pyproject.toml, .gitignore
- src/resolve/__init__.py, __main__.py
- src/resolve/config.py (YAML → dataclasses)
- src/resolve/provenance.py (sidecar writer)
- configs/default.yaml, configs/montha.yaml
- tests/conftest.py, test_config.py, test_provenance.py

## Phase 1
- src/resolve/ingest.py   — IFS ENS → Zarr cache
- src/resolve/truth.py    — IMD + IMERG, alignment check
- src/resolve/accum.py    — rate → mm per window
- src/resolve/exceedance.py — member masks + probabilities
- src/resolve/objects.py  — 8-conn components, features
- src/resolve/crossrun.py — Hungarian run-to-run matching
- src/resolve/fss.py      — land-masked neighbourhood FSS
- src/resolve/earned.py   — earned width from FSS table
- src/resolve/hero.py     — orchestrator, figure, summary.json
- notebooks/01_montha_hero.ipynb
- tests/test_accum.py, test_fss.py, test_objects.py, test_crossrun.py
- outputs/montha/hero.png, hero.svg, summary.json

## Phase 2 (separate plan)
Event library, skill tables, bootstrap CI, earned-resolution engine over the full archive.

## Phase 3 (Colab/Kaggle)
GNN calibrator, XGBoost baselines, Brier/BSS/ECE metrics.

## Phase 4 (separate plan)
FastAPI endpoints, CAP 1.2 XML, replay dashboard (MapLibre/Leaflet).

## Phase 5 (optional, GPU)
CorrDiff-Mini downscaling to 5 km. Exact conservation layer.

## Objections and concerns

1. **dynamical-catalog API stability:** Warn against data.dynamical.org URLs
   after 30 Sep 2026. If catalog breaks, fallback: ECMWF open-data S3 bucket.

2. **IMD archive availability:** imdlib requires a local copy or live download.
   Firewall/auth issues must be resolved before Phase 1 can fetch truth data.

3. **IBTrACS Montha 2025 gap:** Storm may be missing from v04r01. IMD bulletin
   positions will be used (14.9°N, 82.9°E at 08:30 IST 28 Oct 2025).

4. **5 km label:** Phase 5 output labelled "scenario", never "forecast" (hard rule §5).

5. **District boundaries:** Census-2011 polygons give wrong AP names (post-2022
   reorganisation). Need 2022+ source; fallback: IMD meteorological subdivisions.

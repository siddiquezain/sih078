# Architecture Decisions

## D-001 — dataclasses over pydantic for config
**Date:** 2026-09-24
**Decision:** Use stdlib `dataclasses` for config, not pydantic.
**Reason:** Pydantic is not in the required dependency list; avoids adding a
heavy dependency for a simple config schema.

## D-002 — Zarr v2 for local cache
**Date:** 2026-09-24
**Decision:** Cache subsetted IFS ENS data in Zarr v2 format (zarr>=2.16).
**Reason:** zarr v3 is not yet stable in all environments. v2 is in the
required dependency list and widely supported.

## D-003 — No torch in Phase 0/1
**Date:** 2026-09-24
**Decision:** torch and torch-geometric are not installed in Phase 0/1.
**Reason:** MacBook Air Intel has no GPU and limited RAM.
Phase 3 runs in Colab/Kaggle.

## D-004 — Provenance sidecar beside every output
**Date:** 2026-09-24
**Decision:** Every output file (PNG, SVG, JSON) gets a .provenance.json
sidecar written by save_sidecar().
**Reason:** Hard rule §4 in CLAUDE.md.

## D-005 — imdlib ≥0.1.22 returns Dataset from get_xarray()
**Date:** 2026-09-25
**Decision:** In `fetch_imd`, extract `result["rain"]` if `get_xarray()` returns a Dataset.
Also rename lat/lon → latitude/longitude for consistency with IFS ENS coordinate names.
**Reason:** imdlib 0.1.22 changed the return type from DataArray to Dataset.

## D-006 — Rain sanity gate at 2000 mm, not 1000 mm
**Date:** 2026-09-25
**Decision:** `accumulate_window` raises if any cell ≥ 2000 mm in 24 h.
**Reason:** IFS ENS extreme members for Cyclone Montha produced 1136 mm/24 h over
the Andhra coast (D-5 run, 2025-10-24 init). World record 24 h rainfall is 1825 mm
(La Réunion, 1966). Gate purpose is to catch unit errors (mm/hr unconverted → >86 000 mm).

## D-007 — D-0 same-day inits skipped when 24 h window predates init
**Date:** 2026-09-25
**Decision:** `run_hero` logs a warning and skips inits where `accumulate_window` raises
the 86400 s coverage gate.
**Reason:** A D-0 init at 00 UTC on the target date covers only the last 3 h of the
(D-1 03 UTC, D 03 UTC] window. The forecast adds no skill for the preceding 21 h.

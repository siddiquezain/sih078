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

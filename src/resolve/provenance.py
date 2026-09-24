"""Provenance JSON sidecar helpers.

Every output file gets a sidecar at <output>.provenance.json containing:
  - dataset ids
  - init times used
  - thresholds and window offsets
  - git commit at time of writing
  - wall-clock access time
  - replay/live label
"""

from __future__ import annotations
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _git_commit() -> str:
    """Return HEAD commit hash, or 'unknown' if git is unavailable."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            cwd=str(Path(__file__).resolve().parents[2]),
        ).decode().strip()
    except Exception:
        return "unknown"


def make_provenance(
    *,
    datasets: list[str],
    init_times: list[str],
    thresholds: dict[str, Any],
    window_offsets: dict[str, Any],
    label: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a provenance dict."""
    assert label in ("replay", "live"), f"label must be 'replay' or 'live', got {label!r}"
    prov: dict[str, Any] = {
        "label": label,
        "datasets": datasets,
        "init_times": init_times,
        "thresholds": thresholds,
        "window_offsets": window_offsets,
        "git_commit": _git_commit(),
        "access_time": datetime.now(timezone.utc).isoformat(),
    }
    if extra:
        prov.update(extra)
    return prov


def save_sidecar(output_path: str | Path, provenance: dict[str, Any]) -> Path:
    """Write <output_path>.provenance.json next to the output file.

    If output_path ends in '.json', the sidecar is named <stem>.provenance.json.
    """
    p = Path(output_path)
    if p.suffix == ".json":
        sidecar = p.with_name(p.stem + ".provenance.json")
    else:
        sidecar = p.with_suffix(p.suffix + ".provenance.json")
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    sidecar.write_text(json.dumps(provenance, indent=2, default=str))
    return sidecar

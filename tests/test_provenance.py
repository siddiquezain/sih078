import json
import pytest
from pathlib import Path
from resolve.provenance import make_provenance, save_sidecar


def test_make_provenance_has_required_keys():
    prov = make_provenance(
        datasets=["ecmwf-ifs-ens-forecast-15-day-0-25-degree"],
        init_times=["2025-10-20T00:00:00"],
        thresholds={"very_heavy_mm": 115.6},
        window_offsets={"window_offset_hours": 0},
        label="replay",
    )
    assert "datasets" in prov
    assert "init_times" in prov
    assert "thresholds" in prov
    assert "window_offsets" in prov
    assert "git_commit" in prov
    assert "access_time" in prov
    assert "label" in prov
    assert prov["label"] == "replay"


def test_save_sidecar_roundtrip(tmp_path):
    prov = make_provenance(
        datasets=["test"],
        init_times=["2025-10-20"],
        thresholds={},
        window_offsets={},
        label="replay",
    )
    out = tmp_path / "result.json"
    sidecar_path = save_sidecar(out, prov)
    assert sidecar_path.name == "result.provenance.json"
    loaded = json.loads(sidecar_path.read_text())
    assert loaded["datasets"] == ["test"]
    assert loaded["label"] == "replay"

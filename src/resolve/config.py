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


def load_config(path: str) -> ResolveConfig:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with p.open() as fh:
        raw = yaml.safe_load(fh)
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
    )

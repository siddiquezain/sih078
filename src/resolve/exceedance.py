"""Member exceedance masks and raw ensemble probability fields."""

from __future__ import annotations
import xarray as xr


THRESHOLDS_MM = {
    "heavy": 64.5,
    "very_heavy": 115.6,
    "extremely_heavy": 204.5,
}


def member_exceedance_mask(
    accum_da: xr.DataArray,
    threshold_mm: float,
) -> xr.DataArray:
    """Boolean mask: True where member rain >= threshold.

    Args:
        accum_da: DataArray with dims (member, latitude, longitude), mm.
        threshold_mm: Threshold in mm.

    Returns:
        Boolean DataArray with same dims.
    """
    return (accum_da >= threshold_mm)


def ensemble_probability(
    member_mask: xr.DataArray,
) -> xr.DataArray:
    """Raw ensemble probability = fraction of True members.

    Returns float DataArray with 'member' dim reduced; values in [0, 1].
    """
    return member_mask.mean(dim="member").astype(float)


def obs_exceedance_mask(
    obs_da: xr.DataArray,
    threshold_mm: float,
) -> xr.DataArray:
    """Boolean mask for observed exceedance."""
    return (obs_da >= threshold_mm)

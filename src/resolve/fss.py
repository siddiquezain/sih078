"""Land-masked neighbourhood Fraction Skill Score (FSS).

Roberts & Lean (2008) with land-mask normalisation.

  Ff(n) = uniform_filter(fc_prob × mask, n) / uniform_filter(mask, n)
  Fo(n) = uniform_filter(obs_binary × mask, n) / uniform_filter(mask, n)

Cells where uniform_filter(mask, n) == 0 are excluded (NaN).

  FSS(n) = 1 − mean((Ff − Fo)²) / (mean(Ff²) + mean(Fo²))
         = 1.0 if both fields are zero everywhere.
"""

from __future__ import annotations
import numpy as np
from scipy.ndimage import uniform_filter


def _fractions(field: np.ndarray, mask: np.ndarray, n: int) -> np.ndarray:
    """Compute neighbourhood fractions with mask normalisation."""
    mask_f = mask.astype(float)
    numerator = uniform_filter(field * mask_f, size=n, mode="constant", cval=0.0)
    denominator = uniform_filter(mask_f, size=n, mode="constant", cval=0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        frac = np.where(denominator > 0, numerator / denominator, np.nan)
    return frac


def fss_score(
    fc_prob: np.ndarray,
    obs_binary: np.ndarray,
    n: int,
    mask: np.ndarray,
) -> float:
    """Compute FSS for a single window size n.

    Returns 1.0 if both forecast and observation are zero everywhere.
    Returns nan if no valid land cells remain after masking.
    """
    Ff = _fractions(fc_prob, mask, n)
    Fo = _fractions(obs_binary, mask, n)

    valid = mask & np.isfinite(Ff) & np.isfinite(Fo)
    if not valid.any():
        return np.nan

    Ff_v = Ff[valid]
    Fo_v = Fo[valid]

    num = np.mean((Ff_v - Fo_v) ** 2)
    denom = np.mean(Ff_v ** 2) + np.mean(Fo_v ** 2)

    if denom == 0.0:
        return 1.0
    return float(1.0 - num / denom)


def fss_table(
    fc_prob: np.ndarray,
    obs_binary: np.ndarray,
    windows: list[int],
    mask: np.ndarray,
) -> np.ndarray:
    """Compute FSS for each window size. Returns 1-D float array."""
    return np.array([fss_score(fc_prob, obs_binary, n, mask) for n in windows])


def useful_skill_line(f_obs: float) -> float:
    """Useful-skill threshold: 0.5 + f_obs / 2."""
    return 0.5 + f_obs / 2.0

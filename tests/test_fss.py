import numpy as np
import pytest
from resolve.fss import fss_score, fss_table, useful_skill_line


def test_fss_identical_fields():
    """FSS = 1 when forecast equals observation."""
    prob = np.array([
        [0.0, 0.8, 0.8],
        [0.0, 0.9, 0.9],
        [0.0, 0.0, 0.0],
    ], dtype=float)
    obs = prob.copy()
    mask = np.ones((3, 3), dtype=bool)
    score = fss_score(prob, obs, n=1, mask=mask)
    np.testing.assert_allclose(score, 1.0, atol=1e-6)


def test_fss_displaced_blobs_rises_with_n():
    """Displaced forecasts: FSS at n=5 > FSS at n=1."""
    obs = np.zeros((10, 10))
    obs[1:4, 1:4] = 1.0
    fc = np.zeros((10, 10))
    fc[5:8, 5:8] = 1.0
    mask = np.ones((10, 10), dtype=bool)
    fss1 = fss_score(fc, obs, n=1, mask=mask)
    fss5 = fss_score(fc, obs, n=5, mask=mask)
    assert fss5 > fss1, f"Expected FSS5={fss5:.3f} > FSS1={fss1:.3f}"


def test_fss_mask_normalisation():
    """Cells outside mask must not contribute."""
    obs = np.zeros((6, 6))
    obs[2, 2] = 1.0
    fc = np.zeros((6, 6))
    fc[2, 2] = 1.0
    mask = np.ones((6, 6), dtype=bool)
    mask[0:2, 0:2] = False
    score_masked = fss_score(fc, obs, n=1, mask=mask)
    assert score_masked > 0.5


def test_fss_zero_obs_zero_fc():
    """All-zero fields → FSS = 1 by convention."""
    obs = np.zeros((5, 5))
    fc = np.zeros((5, 5))
    mask = np.ones((5, 5), dtype=bool)
    score = fss_score(fc, obs, n=1, mask=mask)
    np.testing.assert_allclose(score, 1.0, atol=1e-6)


def test_useful_skill_line():
    np.testing.assert_allclose(useful_skill_line(0.2), 0.6)
    np.testing.assert_allclose(useful_skill_line(0.0), 0.5)
    np.testing.assert_allclose(useful_skill_line(1.0), 1.0)


def test_fss_table_shape():
    prob = np.random.rand(8, 8)
    obs = (np.random.rand(8, 8) > 0.7).astype(float)
    mask = np.ones((8, 8), dtype=bool)
    windows = [1, 3, 5]
    table = fss_table(prob, obs, windows, mask)
    assert table.shape == (3,)
    assert np.all((table >= 0.0) & (table <= 1.0))

import numpy as np
import pytest

from tfm.calibrate import apply_temperature, ece, fit_temperature, reliability_table


def test_reliability_table_bins_predictions_and_reports_observed_rate():
    p = np.array([0.05, 0.15, 0.15, 0.95, 0.95, 0.95])
    y = np.array([0, 0, 1, 1, 1, 0])
    t = reliability_table(y, p, n_bins=10)
    assert list(t.columns) == ["bin", "n", "mean_pred", "obs_rate"]
    row = t[t["bin"] == 1].iloc[0]                      # [0.1, 0.2): two rows, one fraud
    assert row["n"] == 2 and row["mean_pred"] == pytest.approx(0.15) and row["obs_rate"] == pytest.approx(0.5)
    assert t["n"].sum() == 6


def test_ece_is_zero_when_each_bin_is_perfectly_calibrated_and_grows_with_miscalibration():
    rng = np.random.default_rng(0)
    p = np.repeat([0.1, 0.5, 0.9], 1000)
    y = (rng.random(3000) < p).astype(int)
    assert ece(y, p, n_bins=10) < 0.03
    assert ece(y, np.clip(p * 0.5, 0, 1), n_bins=10) > 0.15


def test_fit_temperature_recovers_the_scale_that_generated_the_labels():
    rng = np.random.default_rng(1)
    logit = rng.normal(0, 3, 20000)
    p_over = 1 / (1 + np.exp(-logit))                   # overconfident scores
    y = (rng.random(20000) < 1 / (1 + np.exp(-logit / 2))).astype(int)   # truth used logit / 2
    T = fit_temperature(y, p_over)
    assert 1.7 < T < 2.3
    assert np.allclose(apply_temperature(p_over, 1.0), p_over)
    assert ece(y, apply_temperature(p_over, T)) < ece(y, p_over)

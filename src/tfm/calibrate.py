"""Step 6d: calibration tools for the fraud scores (no training).

A score is used two ways: as a ranking (alert the top 0.6%) and sometimes as a probability ("this
transaction is 30% likely fraud"). Ranking metrics do not care about calibration; a probability does.
Three versions of the score are compared on the test rows:

    raw          softmax output of a model trained on all frauds + 5% of normals (inflated, metrics.correct_for_sampling)
    corrected    odds multiplied by the keep rate (the sampling undone analytically)
    temperature  corrected log-odds divided by a scalar T fitted on the validation rows (Guo et al. 2017)

reliability_table  predicted-probability bins vs observed fraud rate, the numbers behind a reliability diagram
ece                expected calibration error: bin-weighted |mean_pred - obs_rate|
fit_temperature    T that minimises the negative log-likelihood on validation rows
"""
from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-7


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-z))


def reliability_table(y, p, n_bins: int = 10) -> pd.DataFrame:
    y, p = np.asarray(y), np.asarray(p, dtype=np.float64)
    b = np.minimum((p * n_bins).astype(int), n_bins - 1)
    rows = []
    for k in range(n_bins):
        m = b == k
        if m.any():
            rows.append({"bin": k, "n": int(m.sum()), "mean_pred": float(p[m].mean()), "obs_rate": float(y[m].mean())})
    return pd.DataFrame(rows, columns=["bin", "n", "mean_pred", "obs_rate"])


def ece(y, p, n_bins: int = 10) -> float:
    t = reliability_table(y, p, n_bins)
    if not len(t):
        return float("nan")
    return float((t["n"] / t["n"].sum() * (t["mean_pred"] - t["obs_rate"]).abs()).sum())


def apply_temperature(p, T: float) -> np.ndarray:
    return _sigmoid(_logit(p) / T)


def fit_temperature(y, p, grid: tuple[float, float] = (0.05, 20.0)) -> float:
    """Scalar T minimising the binary NLL of sigmoid(logit(p) / T) on (y, p); golden-section search in log T."""
    from scipy.optimize import minimize_scalar

    y = np.asarray(y, dtype=np.float64); z = _logit(p)

    def nll(log_t):
        q = np.clip(_sigmoid(z / np.exp(log_t)), EPS, 1 - EPS)
        return float(-(y * np.log(q) + (1 - y) * np.log(1 - q)).mean())

    res = minimize_scalar(nll, bounds=(np.log(grid[0]), np.log(grid[1])), method="bounded")
    return float(np.exp(res.x))

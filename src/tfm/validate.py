"""Step 6a: the validator's checks. Nothing here trains; every function reads saved test scores.

    bootstrap_diff     95% CI for metric(A) - metric(B) on the same test rows; resamples whole cards by default,
                       because a card's transactions are not independent (row bootstrap would be too narrow)
    monthly_metrics    the test period month by month, with ONE cutoff fixed on the whole test set, as a deployed
                       threshold would be: shows whether alerts and recall drift over time
    psi, psi_by_field  population stability index of each token field, train period vs test period
    subgroup_metrics   metrics per group (category, history length, cardholder attributes), groups with too few
                       frauds dropped rather than reported on noise
    disagreements      frauds flagged by exactly one of two models, for the 20 + 20 error analysis

PSI reading used in the notebook: < 0.1 stable, 0.1 to 0.25 moderate shift, > 0.25 major shift.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from tfm.metrics import evaluate_scores

HISTORY_EDGES = [0, 1, 5, 12, 32]
HISTORY_LABELS = ["0", "1-4", "5-11", "12-31", "32+"]


# ---- population stability ---------------------------------------------------------------------------------
def psi(expected_counts, actual_counts, eps: float = 1e-6) -> float:
    """sum over bins of (a - e) * ln(a / e) with a and e the bin shares; eps keeps empty bins finite."""
    e = np.asarray(expected_counts, dtype=float); a = np.asarray(actual_counts, dtype=float)
    e = np.maximum(e / e.sum(), eps); a = np.maximum(a / a.sum(), eps)     # eps only touches empty bins
    return float(((a - e) * np.log(a / e)).sum())


def psi_by_field(ids_ref: np.ndarray, ids_new: np.ndarray, field_names: list[str]) -> pd.DataFrame:
    """ids are (n, tokens_per_tx) token matrices; one PSI per column over the union of token ids seen."""
    rows = []
    for j, name in enumerate(field_names):
        toks = np.union1d(ids_ref[:, j], ids_new[:, j])
        e = np.array([(ids_ref[:, j] == t).sum() for t in toks]); a = np.array([(ids_new[:, j] == t).sum() for t in toks])
        rows.append({"field": name, "n_tokens": len(toks), "psi": psi(e, a)})
    return pd.DataFrame(rows)


# ---- cutoff and history ----------------------------------------------------------------------------------
def global_cutoff(score, alert_rate: float) -> float:
    """The score at which the top alert_rate share of ALL rows is flagged (score >= cutoff)."""
    s = np.sort(np.asarray(score))[::-1]
    k = max(1, int(round(alert_rate * len(s))))
    return float(s[k - 1])


def history_length(card_id) -> np.ndarray:
    """Rows before this one on the same card (data in card, time order)."""
    card = np.asarray(card_id)
    n = len(card)
    starts = np.flatnonzero(np.r_[True, card[1:] != card[:-1]])
    return np.arange(n) - np.repeat(starts, np.diff(np.r_[starts, n]))


def history_bucket(pos) -> np.ndarray:
    idx = np.digitize(np.asarray(pos), HISTORY_EDGES[1:], right=False)
    return np.asarray(HISTORY_LABELS, dtype=object)[idx]


# ---- slices -----------------------------------------------------------------------------------------------
def _slice_metrics(y, score, card, flag, mask) -> dict:
    y, s, c, f = y[mask], score[mask], card[mask], flag[mask]
    from sklearn.metrics import average_precision_score
    tp = int(((y == 1) & f).sum())
    return {"n": int(mask.sum()), "n_fraud": int((y == 1).sum()), "fraud_rate": float((y == 1).mean()),
            "pr_auc": float(average_precision_score(y, s)) if 0 < (y == 1).sum() < len(y) else float("nan"),
            "alerts": int(f.sum()), "alert_rate": float(f.mean()),
            "recall_at_cutoff": float(tp / max(1, (y == 1).sum())),
            "fp_ratio": float(((y == 0) & f).sum() / tp) if tp else float("inf")}


def monthly_metrics(y, score, card_id, ts, alert_rate: float = 0.006) -> pd.DataFrame:
    y, score, card = np.asarray(y), np.asarray(score), np.asarray(card_id)
    flag = score >= global_cutoff(score, alert_rate)
    month = pd.DatetimeIndex(pd.to_datetime(np.asarray(ts))).to_period("M").astype(str)
    rows = [{"month": m, **_slice_metrics(y, score, card, flag, month == m)} for m in sorted(set(month))]
    return pd.DataFrame(rows)


def subgroup_metrics(y, score, card_id, group, alert_rate: float = 0.006, min_fraud: int = 20) -> pd.DataFrame:
    """One row per group value; groups with fewer than min_fraud frauds are dropped (their recall is noise)."""
    y, score, card, g = np.asarray(y), np.asarray(score), np.asarray(card_id), np.asarray(group)
    flag = score >= global_cutoff(score, alert_rate)
    rows = []
    for v in sorted(set(g.tolist()), key=str):
        r = _slice_metrics(y, score, card, flag, g == v)
        if r["n_fraud"] >= min_fraud:
            rows.append({"group": v, **r})
    cols = ["group", "n", "n_fraud", "fraud_rate", "pr_auc", "alerts", "alert_rate", "recall_at_cutoff", "fp_ratio"]
    return pd.DataFrame(rows, columns=cols)


# ---- paired bootstrap ------------------------------------------------------------------------------------
def bootstrap_diff(y, score_a, score_b, card_id, alert_rate: float = 0.006, n_boot: int = 1000, seed: int = 0,
                   keys=("pr_auc", "recall_at_alert", "fp_ratio"), unit: str = "card") -> dict:
    """metric(A) - metric(B) with a percentile 95% CI. unit='card' resamples cards with replacement and takes all
    their rows (the honest choice: rows of one card are not independent); unit='row' resamples rows. Samples
    without both classes are skipped, so n_boot in the output can be below the request."""
    y, a, b, card = map(np.asarray, (y, score_a, score_b, card_id))
    rng = np.random.default_rng(seed)
    point = {k: evaluate_scores(y, a, card, alert_rate)[k] - evaluate_scores(y, b, card, alert_rate)[k] for k in keys}
    if unit == "card":
        cards, inv = np.unique(card, return_inverse=True)
        order = np.argsort(inv, kind="stable"); bounds = np.r_[0, np.cumsum(np.bincount(inv))]
        members = [order[bounds[i]:bounds[i + 1]] for i in range(len(cards))]
    diffs = {k: [] for k in keys}
    for _ in range(n_boot):
        if unit == "card":
            idx = np.concatenate([members[i] for i in rng.integers(0, len(members), len(members))])
        else:
            idx = rng.integers(0, len(y), len(y))
        ys = y[idx]
        if ys.min() == ys.max():
            continue
        ma, mb = evaluate_scores(ys, a[idx], card[idx], alert_rate), evaluate_scores(ys, b[idx], card[idx], alert_rate)
        for k in keys:
            diffs[k].append(ma[k] - mb[k])
    out = {}
    for k in keys:
        d = np.array(diffs[k], dtype=float)
        d = d[np.isfinite(d)]
        out[k] = {"diff": float(point[k]), "ci_low": float(np.percentile(d, 2.5)) if len(d) else float("nan"),
                  "ci_high": float(np.percentile(d, 97.5)) if len(d) else float("nan"), "n_boot": int(len(d)), "unit": unit}
    return out


# ---- error analysis --------------------------------------------------------------------------------------
def disagreements(y, score_a, score_b, alert_rate: float = 0.006, k: int = 20) -> tuple[np.ndarray, np.ndarray]:
    """Row indices of frauds flagged by A only and by B only (each at its own global cutoff), the k with the
    largest score gap first."""
    y, a, b = map(np.asarray, (y, score_a, score_b))
    fa, fb = a >= global_cutoff(a, alert_rate), b >= global_cutoff(b, alert_rate)
    only_a = np.flatnonzero((y == 1) & fa & ~fb); only_b = np.flatnonzero((y == 1) & fb & ~fa)
    only_a = only_a[np.argsort(-(a[only_a] - b[only_a]), kind="stable")][:k]
    only_b = only_b[np.argsort(-(b[only_b] - a[only_b]), kind="stable")][:k]
    return only_a, only_b


def agreement_table(y, score_a, score_b, alert_rate: float = 0.006) -> dict:
    """2 x 2 agreement table over the frauds: how many both models flag, A only, B only, neither, each model at
    its own global cutoff. The full counts, not the k-row lists disagreements() keeps for display; the four
    cells sum to the number of frauds."""
    y, a, b = map(np.asarray, (y, score_a, score_b))
    fr = y == 1
    fa, fb = a >= global_cutoff(a, alert_rate), b >= global_cutoff(b, alert_rate)
    return {"both": int((fr & fa & fb).sum()), "only_a": int((fr & fa & ~fb).sum()),
            "only_b": int((fr & ~fa & fb).sum()), "neither": int((fr & ~fa & ~fb).sum())}


def mcnemar_exact(n_only_a: int, n_only_b: int) -> float:
    """Exact two-sided McNemar p-value: under equal performance each discordant fraud is equally likely to be
    caught by A only or by B only, so the smaller count is Binomial(n_only_a + n_only_b, 0.5). Treats the
    discordant rows as independent, which frauds inside one burst are not, so on bursty data it is optimistic."""
    n = n_only_a + n_only_b
    if n == 0:
        return 1.0
    lo = min(n_only_a, n_only_b)
    tail = sum(math.comb(n, i) for i in range(0, lo + 1)) / 2 ** n
    return float(min(1.0, 2 * tail))

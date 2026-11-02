"""Evaluation used by steps 4, 5 and 6, aligned with the Wells Fargo business metrics.

    PR-AUC (main) and ROC-AUC     ranking quality; ROC-AUC alone is flattering at 0.5% fraud
    recall at a fixed alert rate  flag the top 0.6% of scores (WF's outsort rate), share of fraud caught
    card discovery rate           cards with fraud whose fraud was flagged at least once (WF unit-discovery rate)
    false-positive ratio          normals flagged / frauds flagged

Every model is scored on the same test rows with the same functions; seeds are aggregated with mean_sd.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def _flagged(score: np.ndarray, alert_rate: float) -> np.ndarray:
    k = max(1, int(round(alert_rate * len(score))))
    flag = np.zeros(len(score), dtype=bool)
    flag[np.argsort(-np.asarray(score), kind="stable")[:k]] = True
    return flag


def recall_at_alert_rate(y, score, alert_rate: float = 0.006) -> float:
    y = np.asarray(y); f = _flagged(score, alert_rate)
    return float(((y == 1) & f).sum() / max(1, (y == 1).sum()))


def false_positive_ratio(y, score, alert_rate: float = 0.006) -> float:
    y = np.asarray(y); f = _flagged(score, alert_rate)
    tp = ((y == 1) & f).sum()
    return float(((y == 0) & f).sum() / tp) if tp else float("inf")


def card_discovery_rate(y, score, card_id, alert_rate: float = 0.006) -> float:
    """Among cards with at least one fraud, the share with at least one fraud flagged."""
    y, card = np.asarray(y), np.asarray(card_id); f = _flagged(score, alert_rate)
    fraud_cards = np.unique(card[y == 1])
    hit = np.unique(card[(y == 1) & f])
    return float(len(hit) / max(1, len(fraud_cards)))


def evaluate_scores(y, score, card_id, alert_rate: float = 0.006) -> dict:
    y, score = np.asarray(y), np.asarray(score)
    return {
        "pr_auc": float(average_precision_score(y, score)),
        "roc_auc": float(roc_auc_score(y, score)),
        "recall_at_alert": recall_at_alert_rate(y, score, alert_rate),
        "card_discovery": card_discovery_rate(y, score, card_id, alert_rate),
        "fp_ratio": false_positive_ratio(y, score, alert_rate),
        "alert_rate": alert_rate,
        "n": int(len(y)),
        "n_fraud": int((y == 1).sum()),
    }


def correct_for_sampling(p, keep_rate: float) -> np.ndarray:
    """Undo negative downsampling: keeping a share keep_rate of normals multiplied the odds by 1 / keep_rate,
    so the probability a deployed model should report has odds * keep_rate."""
    p = np.clip(np.asarray(p, dtype=float), 1e-9, 1 - 1e-9)
    odds = p / (1 - p) * keep_rate
    return odds / (1 + odds)


def mean_sd(runs: list[dict]) -> dict:
    """Per numeric key: mean and sample sd (ddof 1) across seeds."""
    out = {}
    for k, v0 in runs[0].items():
        if isinstance(v0, bool) or not isinstance(v0, (int, float)):
            continue
        v = np.array([r[k] for r in runs], dtype=float)
        out[k] = {"mean": float(v.mean()), "sd": float(v.std(ddof=1)) if len(v) > 1 else 0.0, "n_seeds": len(v)}
    return out

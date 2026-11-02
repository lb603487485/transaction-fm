"""Step 5: the hand-built feature baseline, XGBoost on 29 features.

    raw row i  ->  current (6)   amount, category code, hour, weekday, distance
               ->  gaps (3)      seconds since the card's previous row, previous same-category row, previous same-merchant row
               ->  rolling (8)   count and amount sum over the last 1 h / 24 h / 7 d / 30 d of the same card
               ->  deviation (12) how row i compares with the card's whole past: mean, std, z-score, share of this
                                  category / merchant / hour period, first visit to this merchant

Leakage rules, each one a test in tests/test_baseline.py
-------------------------------------------------------
* every history feature is computed from rows STRICTLY before row i on the SAME card
  (``rolling(window, on="ts", closed="left")`` and ``cumsum - current``), so the row never sees itself
  and never sees a later row; a card's first row has NaN history, which XGBoost handles natively;
* the only fitted encoder (category -> code) is fit on train rows through ``assert_train_only``;
  an unseen category becomes -1;
* ``is_fraud`` and ``split`` never enter a feature; flipping every label leaves the matrix unchanged.

Fitting
-------
``fit_xgb`` trains on the train rows only, with early stopping on the validation rows, and selects the
number of trees by validation PR-AUC (xgboost's ``aucpr``). ``grid_search`` runs the guide's three knobs
(max_depth, learning_rate, scale_pos_weight) on validation. Test rows are scored once, in the notebook.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from tfm.data import assert_train_only, train_only

WINDOWS = ["1h", "24h", "7d", "30d"]
GROUPS = {
    "current": ["amt", "log_amt", "cat_code", "hour", "dow", "dist_km"],
    "gaps": ["gap_prev_s", "gap_prev_cat_s", "gap_prev_mer_s"],
    "rolling": [f"{k}_{w}" for w in WINDOWS for k in ("n", "amt")],
    "deviation": ["hist_n", "hist_mean_amt", "hist_std_amt", "amt_ratio_mean", "amt_z", "hist_cat_n", "hist_cat_share",
                  "hist_mer_n", "first_at_merchant", "hist_mean_dist", "dist_ratio_mean", "hist_hour_share"],
}
FEATURES = [c for cols in GROUPS.values() for c in cols]
CURRENT_ONLY = GROUPS["current"]                                   # the "current transaction only" variant
MIXED = ["amt_ratio_mean", "amt_z", "dist_ratio_mean"]            # current value divided by / standardised by history
DESCRIPTION = {
    "amt": "transaction amount", "log_amt": "log1p(amt)", "cat_code": "category as an integer code fit on train (-1 unseen)",
    "hour": "hour of day 0-23", "dow": "day of week 0=Monday", "dist_km": "cardholder home to merchant, km",
    "gap_prev_s": "seconds since the card's previous transaction", "gap_prev_cat_s": "seconds since the card's previous transaction in the same category",
    "gap_prev_mer_s": "seconds since the card's previous transaction at the same merchant",
    **{f"n_{w}": f"transactions of the card in the {w} before this one" for w in WINDOWS},
    **{f"amt_{w}": f"amount spent by the card in the {w} before this one" for w in WINDOWS},
    "hist_n": "number of earlier transactions of the card", "hist_mean_amt": "mean amount of those", "hist_std_amt": "std of those amounts",
    "amt_ratio_mean": "amt / hist_mean_amt", "amt_z": "(amt - hist_mean_amt) / hist_std_amt", "hist_cat_n": "earlier transactions in this category",
    "hist_cat_share": "hist_cat_n / hist_n", "hist_mer_n": "earlier transactions at this merchant", "first_at_merchant": "1 if hist_mer_n == 0",
    "hist_mean_dist": "mean dist_km of earlier transactions", "dist_ratio_mean": "dist_km / hist_mean_dist",
    "hist_hour_share": "share of earlier transactions in the same 4-hour period of the day",
}


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """The 29 features for every row of ``df`` (sorted by card then time, as step 1 writes it). Same index as df."""
    gaps = df.groupby("card_id")["ts"].diff()
    assert gaps.dropna().ge(pd.Timedelta(0)).all(), "rows must be sorted by card_id then ts"
    cats = sorted(train_only(df)["category"].unique())
    assert_train_only(train_only(df), "category codes")
    f = pd.DataFrame(index=df.index)
    g = df.groupby("card_id", sort=False)
    amt, dist = df["amt"], df["dist_km"]

    f["amt"] = amt
    f["log_amt"] = np.log1p(amt)
    f["cat_code"] = pd.Categorical(df["category"], categories=cats).codes.astype(np.int16)
    f["hour"] = df["ts"].dt.hour
    f["dow"] = df["ts"].dt.dayofweek
    f["dist_km"] = dist

    f["gap_prev_s"] = gaps.dt.total_seconds()
    f["gap_prev_cat_s"] = df.groupby(["card_id", "category"])["ts"].diff().dt.total_seconds()
    f["gap_prev_mer_s"] = df.groupby(["card_id", "merchant"])["ts"].diff().dt.total_seconds()

    for w in WINDOWS:                                              # [t - w, t): the row itself and same-time rows excluded
        r = g[["ts", "amt"]].rolling(w, on="ts", closed="left")["amt"].agg(["count", "sum"])
        assert len(r) == len(df)                                   # groups come back in df order (sorted by card, rows untouched)
        f[f"n_{w}"] = r["count"].fillna(0).to_numpy().astype(np.int32)
        f[f"amt_{w}"] = r["sum"].fillna(0.0).to_numpy()

    n = g.cumcount()                                               # earlier rows of the card
    s1 = g["amt"].cumsum() - amt                                   # sums over earlier rows only
    s2 = g["amt"].transform(lambda x: (x ** 2).cumsum()) - amt ** 2
    mean = s1 / n.replace(0, np.nan)
    var = (s2 - n * mean ** 2) / (n - 1).where(n >= 2)
    std = np.sqrt(var.clip(lower=0))
    f["hist_n"] = n.astype(np.int32)
    f["hist_mean_amt"] = mean
    f["hist_std_amt"] = std
    f["amt_ratio_mean"] = amt / mean
    f["amt_z"] = (amt - mean) / std.replace(0, np.nan)
    f["hist_cat_n"] = df.groupby(["card_id", "category"]).cumcount().astype(np.int32)
    f["hist_cat_share"] = f["hist_cat_n"] / n.replace(0, np.nan)
    f["hist_mer_n"] = df.groupby(["card_id", "merchant"]).cumcount().astype(np.int32)
    f["first_at_merchant"] = (f["hist_mer_n"] == 0).astype(np.int8)
    mean_dist = (g["dist_km"].cumsum() - dist) / n.replace(0, np.nan)
    f["hist_mean_dist"] = mean_dist
    f["dist_ratio_mean"] = dist / mean_dist
    f["hist_hour_share"] = df.groupby(["card_id", df["ts"].dt.hour // 4]).cumcount() / n.replace(0, np.nan)
    return f[FEATURES].replace([np.inf, -np.inf], np.nan)            # a zero history mean makes a ratio inf; XGBoost wants NaN


def fit_xgb(Xtr, ytr, Xv, yv, max_depth: int, learning_rate: float, scale_pos_weight: float, seed: int,
            n_estimators: int = 2000, early_stopping_rounds: int = 50, device: str = "cpu"):
    """Train on train rows, stop early on validation PR-AUC. Returns (model, info); predict_proba then uses the best iteration."""
    from sklearn.metrics import average_precision_score
    from xgboost import XGBClassifier

    model = XGBClassifier(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate, scale_pos_weight=scale_pos_weight,
                          subsample=0.8, colsample_bytree=0.8, tree_method="hist", device=device, eval_metric="aucpr",
                          early_stopping_rounds=early_stopping_rounds, random_state=seed, n_jobs=-1)
    t0 = time.time()
    model.fit(Xtr, ytr, eval_set=[(Xv, yv)], verbose=False)
    curve = model.evals_result()["validation_0"]["aucpr"]
    info = {"max_depth": max_depth, "learning_rate": learning_rate, "scale_pos_weight": scale_pos_weight, "seed": seed,
            "best_iteration": int(model.best_iteration), "n_trees_tried": len(curve),
            "val_pr_auc": float(average_precision_score(yv, model.predict_proba(Xv)[:, 1])), "fit_seconds": round(time.time() - t0, 1)}
    return model, info


def grid_search(Xtr, ytr, Xv, yv, grid: list[dict], seed: int = 0, device: str = "cpu", **fit_kw) -> pd.DataFrame:
    """One fit per config, all selected on validation; best first."""
    rows = [fit_xgb(Xtr, ytr, Xv, yv, seed=seed, device=device, **cfg, **fit_kw)[1] for cfg in grid]
    return pd.DataFrame(rows).sort_values("val_pr_auc", ascending=False, kind="stable").reset_index(drop=True)

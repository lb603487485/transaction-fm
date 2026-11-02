import numpy as np
import pandas as pd
import pytest

from tfm.baseline import FEATURES, CURRENT_ONLY, GROUPS, DESCRIPTION, MIXED, build_features, fit_xgb, grid_search


def small_frame():
    """Two cards. Card 0: six rows one hour apart on day 1 (one twice-equal timestamp), then one row 3 days later.
    Card 1: three rows. Categories and merchants chosen so same-category and same-merchant history is readable."""
    rows = [
        (0, "2020-01-01 00:00", "food", "m1", 10.0, 1.0, 0, "train"),
        (0, "2020-01-01 01:00", "food", "m2", 20.0, 2.0, 0, "train"),
        (0, "2020-01-01 02:00", "gas", "m1", 30.0, 3.0, 0, "train"),
        (0, "2020-01-01 02:00", "gas", "m3", 40.0, 4.0, 1, "train"),   # same timestamp as the row above
        (0, "2020-01-01 03:00", "food", "m1", 50.0, 5.0, 0, "val"),
        (0, "2020-01-04 03:00", "food", "m9", 60.0, 6.0, 1, "test"),
        (1, "2020-01-01 00:30", "gas", "m1", 100.0, 1.0, 0, "train"),
        (1, "2020-01-01 00:45", "gas", "m1", 200.0, 1.0, 0, "train"),
        (1, "2020-01-02 00:45", "food", "m2", 300.0, 1.0, 1, "test"),
    ]
    df = pd.DataFrame(rows, columns=["card_id", "ts", "category", "merchant", "amt", "dist_km", "is_fraud", "split"])
    df["ts"] = pd.to_datetime(df["ts"])
    df["unix_time"] = df["ts"].astype("int64") // 10**9
    return df


def brute(df, i, window):
    """Same-card rows with ts in [t - window, t), by hand."""
    t = df.loc[i, "ts"]
    m = (df["card_id"] == df.loc[i, "card_id"]) & (df["ts"] >= t - window) & (df["ts"] < t)
    return df[m]


def test_rolling_windows_hold_only_strictly_earlier_rows_of_the_same_card():
    df = small_frame()
    f = build_features(df)
    for i in range(len(df)):
        for w, name in ((pd.Timedelta("1h"), "1h"), (pd.Timedelta("24h"), "24h")):
            b = brute(df, i, w)
            assert f.loc[i, f"n_{name}"] == len(b), (i, name)
            assert f.loc[i, f"amt_{name}"] == pytest.approx(b["amt"].sum()), (i, name)
    assert f.loc[3, "n_1h"] == 1          # row 2 has the same timestamp as row 3 and is excluded (closed left)
    assert f.loc[5, "n_24h"] == 0         # three days after the previous row
    assert f.loc[5, "n_7d"] == 5


def test_history_features_exclude_the_current_row():
    df = small_frame()
    f = build_features(df)
    assert f["hist_n"].tolist() == [0, 1, 2, 3, 4, 5, 0, 1, 2]
    assert f.loc[2, "hist_mean_amt"] == pytest.approx(15.0)             # mean of 10 and 20, not of 10, 20, 30
    assert np.isnan(f.loc[0, "hist_mean_amt"]) and np.isnan(f.loc[0, "gap_prev_s"])
    assert f.loc[4, "amt_ratio_mean"] == pytest.approx(50.0 / 25.0)
    assert f.loc[4, "hist_cat_n"] == 2 and f.loc[4, "hist_mer_n"] == 2  # two earlier food rows, two earlier m1 rows
    assert f.loc[5, "first_at_merchant"] == 1 and f.loc[4, "first_at_merchant"] == 0
    assert f.loc[1, "gap_prev_s"] == 3600 and f.loc[1, "gap_prev_cat_s"] == 3600 and np.isnan(f.loc[2, "gap_prev_cat_s"])
    assert f.loc[4, "gap_prev_cat_s"] == 2 * 3600 and f.loc[4, "gap_prev_mer_s"] == 3600


def test_editing_a_later_row_never_changes_an_earlier_rows_features():
    df = small_frame()
    before = build_features(df)
    df2 = df.copy(); df2.loc[5, "amt"] = 9999.0; df2.loc[5, "category"] = "gas"
    after = build_features(df2)
    pd.testing.assert_frame_equal(before.drop(index=5), after.drop(index=5))


def test_editing_the_current_row_changes_only_its_current_features():
    df = small_frame()
    before = build_features(df)
    df2 = df.copy(); df2.loc[3, "amt"] = 9999.0
    after = build_features(df2)
    pure_history = [c for c in FEATURES if c not in CURRENT_ONLY and c not in MIXED]
    pd.testing.assert_series_equal(before.loc[3, pure_history], after.loc[3, pure_history])
    assert after.loc[3, "amt"] == 9999.0 and after.loc[3, "amt_ratio_mean"] != before.loc[3, "amt_ratio_mean"]


def test_labels_never_enter_the_features():
    df = small_frame()
    before = build_features(df)
    df2 = df.copy(); df2["is_fraud"] = 1 - df2["is_fraud"]
    pd.testing.assert_frame_equal(before, build_features(df2))
    assert "is_fraud" not in FEATURES and "split" not in FEATURES


def test_category_codes_come_from_train_rows_only():
    df = small_frame()
    df.loc[5, "category"] = "travel"                      # appears only in test
    f = build_features(df)
    assert f.loc[5, "cat_code"] == -1
    assert sorted(f.loc[df["split"] == "train", "cat_code"].unique().tolist()) == [0, 1]


def test_feature_list_is_in_the_guides_range_and_current_only_is_a_subset():
    assert 25 <= len(FEATURES) <= 40
    assert set(CURRENT_ONLY) < set(FEATURES)
    assert sum(len(v) for v in GROUPS.values()) == len(FEATURES)
    assert set(MIXED) < set(FEATURES) and not set(MIXED) & set(CURRENT_ONLY)
    assert [c for c in FEATURES if c not in DESCRIPTION] == []
    f = build_features(small_frame())
    assert list(f.columns) == FEATURES


def test_fit_xgb_early_stops_on_val_and_reports_the_best_iteration():
    pytest.importorskip("xgboost")
    rng = np.random.default_rng(0)
    X = rng.normal(size=(600, 4)); y = (X[:, 0] + 0.3 * rng.normal(size=600) > 1.2).astype(int)
    model, info = fit_xgb(X[:400], y[:400], X[400:], y[400:], max_depth=3, learning_rate=0.3, scale_pos_weight=1.0,
                          seed=0, n_estimators=300, early_stopping_rounds=10)
    assert 0 <= info["best_iteration"] < 300
    assert info["n_trees_tried"] <= 300 and info["n_trees_tried"] >= info["best_iteration"] + 1
    assert 0.5 < info["val_pr_auc"] <= 1.0
    p = model.predict_proba(X[400:])[:, 1]
    assert p.shape == (200,) and 0 <= p.min() and p.max() <= 1


def test_grid_search_ranks_configs_by_val_pr_auc_and_keeps_every_row():
    pytest.importorskip("xgboost")
    rng = np.random.default_rng(1)
    X = rng.normal(size=(400, 3)); y = (X[:, 1] > 1.0).astype(int)
    grid = [dict(max_depth=2, learning_rate=0.3, scale_pos_weight=1.0), dict(max_depth=4, learning_rate=0.1, scale_pos_weight=5.0)]
    table = grid_search(X[:300], y[:300], X[300:], y[300:], grid, seed=0, n_estimators=50, early_stopping_rounds=5)
    assert len(table) == 2 and list(table["val_pr_auc"]) == sorted(table["val_pr_auc"], reverse=True)
    assert {"max_depth", "learning_rate", "scale_pos_weight", "best_iteration", "val_pr_auc", "fit_seconds"} <= set(table.columns)


def test_ratios_over_a_zero_history_are_nan_not_inf():
    df = small_frame()
    df.loc[6, "dist_km"] = 0.0                                       # card 1's only earlier distance is zero, row 7's is not
    f = build_features(df)
    assert np.isnan(f.loc[7, "dist_ratio_mean"])
    assert not np.isinf(f.to_numpy(dtype=float)).any()

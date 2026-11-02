import numpy as np
import pandas as pd
import pytest

from tfm.validate import (bootstrap_diff, disagreements, global_cutoff, history_bucket, history_length,
                          monthly_metrics, psi, psi_by_field, subgroup_metrics)


def test_psi_is_zero_for_the_same_distribution_and_grows_with_shift():
    e = np.array([50, 30, 20]); a = np.array([500, 300, 200])            # same shape, different size
    assert psi(e, a) == pytest.approx(0.0, abs=1e-9)
    shifted = np.array([20, 30, 50])
    assert psi(e, shifted) > 0.1
    # hand-computed two-bin case: p = (0.5, 0.5), q = (0.8, 0.2)
    expect = (0.8 - 0.5) * np.log(0.8 / 0.5) + (0.2 - 0.5) * np.log(0.2 / 0.5)
    assert psi(np.array([5, 5]), np.array([8, 2])) == pytest.approx(expect)


def test_psi_by_field_reports_one_row_per_token_column():
    ref = np.array([[0, 7], [0, 7], [1, 7], [1, 8]])
    new = np.array([[0, 7], [1, 8], [1, 8], [1, 8]])
    out = psi_by_field(ref, new, ["A", "B"])
    assert list(out["field"]) == ["A", "B"]
    assert (out["psi"] > 0).all()
    same = psi_by_field(ref, ref, ["A", "B"])
    assert (same["psi"].abs() < 1e-9).all()


def test_global_cutoff_flags_exactly_the_top_share():
    s = np.arange(1000) / 1000
    c = global_cutoff(s, 0.01)
    assert (s >= c).sum() == 10


def test_history_length_counts_earlier_rows_of_the_same_card():
    card = np.array([0, 0, 0, 1, 2, 2])
    assert history_length(card).tolist() == [0, 1, 2, 0, 0, 1]
    assert history_bucket(np.array([0, 1, 4, 5, 11, 12, 31, 32, 100])).tolist() == \
        ["0", "1-4", "1-4", "5-11", "5-11", "12-31", "12-31", "32+", "32+"]


def test_monthly_metrics_uses_one_global_cutoff():
    n = 2000
    rng = np.random.default_rng(0)
    y = (rng.random(n) < 0.05).astype(int)
    score = y + rng.normal(0, 0.3, n)
    ts = pd.date_range("2020-07-01", periods=n, freq="h")                  # July, August, part of September
    card = np.arange(n) % 50
    m = monthly_metrics(y, score, card, ts, alert_rate=0.05)
    assert list(m["month"]) == ["2020-07", "2020-08", "2020-09"]
    assert m["n"].sum() == n and m["n_fraud"].sum() == y.sum()
    assert m["alerts"].sum() == round(0.05 * n)                               # the cutoff is global, so alerts add up
    assert (m["pr_auc"].between(0, 1)).all()


def test_subgroup_metrics_drops_groups_with_too_few_frauds():
    n = 3000
    rng = np.random.default_rng(1)
    y = (rng.random(n) < 0.05).astype(int)
    score = y + rng.normal(0, 0.3, n)
    group = np.where(np.arange(n) < 2900, "big", "tiny")                     # tiny has about 5 frauds
    card = np.arange(n) % 30
    out = subgroup_metrics(y, score, card, group, alert_rate=0.05, min_fraud=20)
    assert list(out["group"]) == ["big"]
    assert out.iloc[0]["n"] == 2900
    none = subgroup_metrics(y, score, card, group, alert_rate=0.05, min_fraud=10_000)
    assert len(none) == 0 and "recall_at_cutoff" in none.columns                  # empty but with the columns
    both = subgroup_metrics(y, score, card, group, alert_rate=0.05, min_fraud=1)
    assert sorted(both["group"]) == ["big", "tiny"] and both["n"].sum() == n


def test_bootstrap_diff_is_zero_for_identical_scores_and_positive_for_a_better_model():
    n = 4000
    rng = np.random.default_rng(2)
    y = (rng.random(n) < 0.05).astype(int)
    good = y + rng.normal(0, 0.3, n)
    bad = y + rng.normal(0, 1.5, n)
    card = np.arange(n) % 80
    same = bootstrap_diff(y, good, good, card, alert_rate=0.05, n_boot=50, seed=0)
    assert same["pr_auc"]["diff"] == 0 and same["pr_auc"]["ci_low"] == 0 and same["pr_auc"]["ci_high"] == 0
    better = bootstrap_diff(y, good, bad, card, alert_rate=0.05, n_boot=100, seed=0)
    assert better["pr_auc"]["ci_low"] > 0 and better["recall_at_alert"]["ci_low"] > 0
    assert better["pr_auc"]["ci_low"] <= better["pr_auc"]["diff"] <= better["pr_auc"]["ci_high"]
    again = bootstrap_diff(y, good, bad, card, alert_rate=0.05, n_boot=100, seed=0)
    assert again == better                                                     # seeded, reproducible


def test_bootstrap_diff_resamples_whole_cards():
    # two cards: card 0 has every fraud, card 1 has none. A card bootstrap sample that drops card 0 has no
    # positives, so PR-AUC is undefined there; the function must skip such samples and still return a CI.
    y = np.r_[np.ones(20), np.zeros(80)].astype(int)
    card = np.r_[np.zeros(20), np.ones(80)].astype(int)
    score = y + 0.0
    out = bootstrap_diff(y, score, score, card, alert_rate=0.2, n_boot=20, seed=0, unit="card")
    assert out["pr_auc"]["n_boot"] <= 20 and out["pr_auc"]["n_boot"] > 0


def test_disagreements_returns_frauds_flagged_by_exactly_one_model():
    n = 1000
    y = np.zeros(n, dtype=int); y[:40] = 1
    a = np.zeros(n); b = np.zeros(n)
    a[:20] = 1.0; a[30:40] = 1.0                   # A flags frauds 0-19 and 30-39
    b[10:30] = 1.0; b[500:510] = 1.0               # B flags frauds 10-29 and ten normals
    only_a, only_b = disagreements(y, a, b, alert_rate=0.03, k=20)
    assert set(only_a) == set(range(0, 10)) | set(range(30, 40))
    assert set(only_b) == set(range(20, 30))
    assert not set(only_a) & set(only_b)
    short_a, _ = disagreements(y, a, b, alert_rate=0.03, k=5)
    assert len(short_a) == 5

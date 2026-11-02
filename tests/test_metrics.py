import numpy as np
import pytest

from tfm.metrics import (card_discovery_rate, correct_for_sampling, evaluate_scores, false_positive_ratio,
                         mean_sd, recall_at_alert_rate)


def ranked_case():
    """1,000 transactions, 10 frauds. Scores are the rank, so alerting the top 1% flags rows 990..999.
    Frauds sit at 994..999 (6 flagged) and at 10, 20, 30, 40 (missed)."""
    y = np.zeros(1000, dtype=int); y[[10, 20, 30, 40, 994, 995, 996, 997, 998, 999]] = 1
    score = np.arange(1000) / 1000
    return y, score


def test_recall_at_alert_rate_flags_the_top_share():
    y, s = ranked_case()
    assert recall_at_alert_rate(y, s, 0.01) == pytest.approx(0.6)
    assert false_positive_ratio(y, s, 0.01) == pytest.approx(4 / 6)       # 4 normals among the 10 alerts, 6 frauds


def test_card_discovery_rate_counts_cards_with_a_flagged_fraud():
    y, s = ranked_case()
    card = np.zeros(1000, dtype=int)
    card[[10, 20]] = 1; card[[30, 40]] = 2; card[994:] = 3           # three fraud cards; only card 3 has a flagged fraud
    assert card_discovery_rate(y, s, card, 0.01) == pytest.approx(1 / 3)


def test_evaluate_scores_is_perfect_for_a_perfect_ranking():
    y = np.array([0] * 990 + [1] * 10); s = np.linspace(0, 1, 1000); card = np.arange(1000) // 100
    m = evaluate_scores(y, s, card, alert_rate=0.01)
    assert set(m) == {"pr_auc", "roc_auc", "recall_at_alert", "card_discovery", "fp_ratio", "alert_rate", "n", "n_fraud"}
    assert m["pr_auc"] == 1.0 and m["roc_auc"] == 1.0 and m["recall_at_alert"] == 1.0 and m["card_discovery"] == 1.0
    assert m["fp_ratio"] == 0.0 and m["n"] == 1000 and m["n_fraud"] == 10


def test_correct_for_sampling_undoes_negative_downsampling():
    # true odds 0.01 (p = 0.0099); keeping 5% of negatives makes the training odds 0.2 (p = 1/6)
    p_train = 0.2 / 1.2
    assert correct_for_sampling(np.array([p_train]), keep_rate=0.05)[0] == pytest.approx(0.01 / 1.01)
    assert correct_for_sampling(np.array([0.3]), keep_rate=1.0)[0] == pytest.approx(0.3)


def test_mean_sd_across_seeds_uses_sample_sd():
    runs = [{"pr_auc": 1.0, "n": 5}, {"pr_auc": 2.0, "n": 5}, {"pr_auc": 3.0, "n": 5}]
    agg = mean_sd(runs)
    assert agg["pr_auc"] == {"mean": 2.0, "sd": 1.0, "n_seeds": 3} and agg["n"]["mean"] == 5 and agg["n"]["sd"] == 0.0

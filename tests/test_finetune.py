import numpy as np
import pytest

from tfm.finetune import build_examples

PAD, BOS = 0, 1
T = 2   # tokens per transaction in these tests, so sequences are easy to read


def small_case():
    """Two cards. Card 0 has 5 rows (train, train, train, val, test), card 1 has 3 train rows.
    Token row i is [10 + i, 20 + i] so a history can be read off by eye."""
    ids = np.array([[10 + i, 20 + i] for i in range(8)], dtype=np.int32)
    card = np.array([0, 0, 0, 0, 0, 1, 1, 1])
    split = np.array(["train", "train", "train", "val", "test", "train", "train", "train"])
    fraud = np.array([0, 0, 1, 0, 1, 0, 1, 0])
    return ids, card, split, fraud


def test_example_is_bos_history_current_then_right_padding():
    ids, card, split, fraud = small_case()
    X, y, idx = build_examples(ids, card, split, fraud, target_split="train", history=2, pad_id=PAD, bos_id=BOS)
    assert X.shape[1] == 1 + 3 * T
    row = {i: X[k] for k, i in enumerate(idx)}
    assert row[2].tolist() == [BOS, 10, 20, 11, 21, 12, 22]            # two history rows then the current one
    assert row[1].tolist() == [BOS, 10, 20, 11, 21, PAD, PAD]          # one history row, padded on the right
    assert row[0].tolist() == [BOS, 10, 20, PAD, PAD, PAD, PAD]        # first row of the card: no history
    assert y.tolist() == [int(fraud[i]) for i in idx]


def test_history_only_ever_holds_the_same_cards_earlier_rows():
    ids, card, split, fraud = small_case()
    X, y, idx = build_examples(ids, card, split, fraud, target_split="train", history=2, pad_id=PAD, bos_id=BOS)
    row = {i: X[k] for k, i in enumerate(idx)}
    assert row[5].tolist() == [BOS, 15, 25, PAD, PAD, PAD, PAD]        # row 4 belongs to card 0, so card 1 starts empty
    assert row[7].tolist() == [BOS, 15, 25, 16, 26, 17, 27]


def test_val_and_test_examples_look_back_into_earlier_splits_but_target_their_own_rows():
    ids, card, split, fraud = small_case()
    Xv, yv, iv = build_examples(ids, card, split, fraud, target_split="val", history=2, pad_id=PAD, bos_id=BOS)
    Xt, yt, it = build_examples(ids, card, split, fraud, target_split="test", history=2, pad_id=PAD, bos_id=BOS)
    assert iv.tolist() == [3] and Xv[0].tolist() == [BOS, 11, 21, 12, 22, 13, 23]   # history is train rows: allowed direction
    assert it.tolist() == [4] and Xt[0].tolist() == [BOS, 12, 22, 13, 23, 14, 24] and yt.tolist() == [1]


def test_train_sampling_keeps_every_fraud_and_about_keep_rate_of_normals():
    n = 4000
    ids = np.stack([np.arange(n) + 10, np.arange(n) + 10], axis=1).astype(np.int32)
    card = np.zeros(n, dtype=int); split = np.array(["train"] * n)
    fraud = np.zeros(n, dtype=int); fraud[::40] = 1                    # 100 frauds
    X, y, idx = build_examples(ids, card, split, fraud, "train", history=3, pad_id=PAD, bos_id=BOS, neg_keep_rate=0.05, seed=0)
    assert y.sum() == 100 and set(idx[y == 1]) == set(np.flatnonzero(fraud))
    n_neg = int((y == 0).sum())
    assert 150 < n_neg < 250                                             # 3,900 normals x 5% = 195 expected
    assert np.array_equal(idx, np.sort(idx))                             # original time order kept


def test_changing_the_label_changes_y_but_never_the_tokens():
    ids, card, split, fraud = small_case()
    X1, y1, _ = build_examples(ids, card, split, fraud, "train", history=2, pad_id=PAD, bos_id=BOS)
    X2, y2, _ = build_examples(ids, card, split, 1 - fraud, "train", history=2, pad_id=PAD, bos_id=BOS)
    assert np.array_equal(X1, X2) and not np.array_equal(y1, y2)
    assert X1.dtype == np.int16


def test_best_eval_picks_the_highest_pr_auc_entry():
    from tfm.finetune import best_eval, softmax_fraud_prob
    log = [{"step": 500, "eval_pr_auc": 0.5}, {"step": 1000, "eval_pr_auc": 0.7}, {"step": 1500, "eval_pr_auc": 0.6}, {"step": 1600, "loss": 1}]
    assert best_eval(log)["step"] == 1000
    p = softmax_fraud_prob(np.array([[0.0, 0.0], [0.0, 10.0]]))
    assert p[0] == pytest.approx(0.5) and p[1] > 0.99

import json

import numpy as np
import pandas as pd
import pytest

from tfm.data import assign_splits, clean, train_only
from tfm.tokenizer import TransactionTokenizer, make_windows
from test_data import raw_frame


def frame():
    df = clean(raw_frame(n_per_card=60, step_hours=7))
    return assign_splits(df, test_start=pd.Timestamp("2019-01-12"), val_days=3)   # 60 rows at 7 h: ~27 train, ~10 val, rest test per card


def test_fit_refuses_non_train_rows():
    df = frame()
    with pytest.raises(ValueError):
        TransactionTokenizer().fit(df)


def test_encode_shape_and_layout():
    df = frame(); tok = TransactionTokenizer().fit(train_only(df))
    ids = tok.encode(df)
    assert ids.shape == (len(df), tok.tokens_per_tx) and tok.tokens_per_tx == 7
    assert (ids[:, 0] == tok.vocab["<TX>"]).all()
    assert ids.min() >= 0 and ids.max() < len(tok.vocab)
    assert ids.dtype == np.int32


def test_decode_round_trip_is_readable():
    df = frame(); tok = TransactionTokenizer().fit(train_only(df))
    row = df.iloc[5]
    toks = tok.decode(tok.encode(df.iloc[[5]])[0])
    assert toks[0] == "<TX>" and toks[1].startswith("DT_") and toks[2].startswith("HOUR_")
    assert toks[3] == f"CAT_{row['category']}" and toks[4] == f"MER_{row['merchant']}"
    assert toks[5].startswith("AMT_") and toks[6].startswith("DIST_")


def test_dt_first_row_na_and_monotone_in_gap():
    df = frame(); tok = TransactionTokenizer().fit(train_only(df))
    toks = [tok.decode(r) for r in tok.encode(df)]
    first_rows = df.groupby("card_id").head(1).index
    assert all(toks[i][1] == "DT_NA" for i in first_rows)
    # fixed edges: 7 h gap lands in the 6 to 12 h bin, index 6 of the 12 named bins
    assert toks[1][1] == "DT_06"
    assert tok.dt_bin(np.array([0.5, 5.0, 60 * 24 * 40]))[[0, 1, 2]].tolist() == [0, 1, 11]


def test_amt_bins_monotone_and_fit_on_train_only():
    df = frame(); tr = train_only(df)
    tok = TransactionTokenizer(n_amt_bins=20).fit(tr)
    assert len(tok.amt_edges) == 19
    assert tok.amt_edges[0] >= tr["amt"].min() and tok.amt_edges[-1] <= tr["amt"].max()
    a = tok.amt_bin(np.array([1.0, 50.0, 300.0, 10_000.0]))
    assert (np.diff(a) >= 0).all() and a[0] == 0 and a[-1] == 19
    lin = TransactionTokenizer(amt_binning="linear").fit(tr)
    assert not np.allclose(lin.amt_edges, tok.amt_edges)


def test_rare_merchant_and_unseen_category():
    df = frame(); tr = train_only(df)
    tok = TransactionTokenizer(top_merchants=2).fit(tr)
    assert sum(t.startswith("MER_") for t in tok.vocab) == 3            # 2 kept + MER_OTHER
    other = df.copy(); other["category"] = "never_seen"; other["merchant"] = "fraud_NewShop"
    toks = tok.decode(tok.encode(other.iloc[[0]])[0])
    assert toks[3] == "<UNK>" and toks[4] == "MER_OTHER"


def test_switches_change_layout():
    tr = train_only(frame())
    assert TransactionTokenizer(use_dt=False).fit(tr).tokens_per_tx == 6
    assert TransactionTokenizer(use_dist=False).fit(tr).tokens_per_tx == 6
    assert TransactionTokenizer(use_dt=False, use_dist=False).fit(tr).tokens_per_tx == 5


def test_save_load_round_trip(tmp_path):
    df = frame(); tok = TransactionTokenizer().fit(train_only(df))
    tok.save(tmp_path / "tokenizer.json")
    back = TransactionTokenizer.load(tmp_path / "tokenizer.json")
    assert back.vocab == tok.vocab and np.array_equal(back.encode(df), tok.encode(df))
    j = json.loads((tmp_path / "tokenizer.json").read_text())
    assert set(j) >= {"config", "vocab", "amt_edges", "dist_edges", "dt_edges_min"}


def test_make_windows_per_card_and_split():
    df = frame(); tok = TransactionTokenizer().fit(train_only(df))
    ids = tok.encode(df)
    # small seq_len so the fixture yields several windows per card
    w, meta = make_windows(ids, df["card_id"].to_numpy(), df["split"].to_numpy(), "train", seq_len=64, stride=32, pad_id=tok.vocab["<PAD>"])
    assert w.shape[1] == 64 and w.dtype == np.int32 and len(meta) == len(w)
    n_train = (df["split"] == "train").groupby(df["card_id"]).sum() * tok.tokens_per_tx
    expected = sum((int(n) - 64) // 32 + 1 for n in n_train)
    assert len(w) == expected
    assert set(meta["card_id"]) == {0, 1, 2}
    assert (w != tok.vocab["<PAD>"]).all()                                 # all windows full here
    # a card shorter than seq_len gets one padded window
    short, m2 = make_windows(ids[:3], np.array([0, 0, 0]), np.array(["train"] * 3), "train", seq_len=64, stride=32, pad_id=0)
    assert short.shape == (1, 64) and (short[0, 21:] == 0).all() and m2["n_real"].iloc[0] == 21
    # val windows contain val rows only
    wv, mv = make_windows(ids, df["card_id"].to_numpy(), df["split"].to_numpy(), "val", seq_len=64, stride=32, pad_id=0)
    assert (mv["split"] == "val").all()

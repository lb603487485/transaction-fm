import numpy as np
import pandas as pd
import pytest

from tfm.data import PII_COLUMNS, assign_splits, clean, eda_stats, run

RAW_COLUMNS = [
    "Unnamed: 0", "trans_date_trans_time", "cc_num", "merchant", "category", "amt", "first", "last",
    "gender", "street", "city", "state", "zip", "lat", "long", "city_pop", "job", "dob", "trans_num",
    "unix_time", "merch_lat", "merch_long", "is_fraud",
]


def raw_frame(n_per_card=40, cards=(1111, 2222, 3333), start="2019-01-01", step_hours=100):
    rows = []
    t0 = pd.Timestamp(start)
    for ci, card in enumerate(cards):
        for i in range(n_per_card):
            ts = t0 + pd.Timedelta(hours=step_hours * i + ci)
            rows.append({
                "Unnamed: 0": len(rows), "trans_date_trans_time": ts, "cc_num": card,
                "merchant": f"fraud_Shop{i % 3}", "category": ["grocery_pos", "gas_transport"][i % 2],
                "amt": 10.0 * (i + 1), "first": "A", "last": "B", "gender": "F", "street": "1 Main",
                "city": "X", "state": "ON", "zip": 1, "lat": 43.0, "long": -79.0, "city_pop": 100,
                "job": "dev", "dob": "1990-01-01", "trans_num": f"t{len(rows)}",
                "unix_time": int(ts.timestamp()), "merch_lat": 43.0 + 0.01 * i, "merch_long": -79.0,
                "is_fraud": int(i == n_per_card - 1),
            })
    df = pd.DataFrame(rows)[RAW_COLUMNS]
    return df.sample(frac=1.0, random_state=0).reset_index(drop=True)  # shuffle to test sorting


def test_clean_drops_pii_and_sorts_by_card_then_time():
    df = clean(raw_frame())
    assert not set(PII_COLUMNS) & set(df.columns)
    assert "cc_num" not in df.columns
    assert list(df.columns) == ["card_id", "ts", "unix_time", "category", "merchant", "amt", "dist_km", "is_fraud"]
    key = list(zip(df["card_id"], df["ts"]))
    assert key == sorted(key)
    assert df["card_id"].nunique() == 3 and df["card_id"].min() == 0


def test_card_id_is_deterministic_across_calls():
    a = clean(raw_frame())
    b = clean(raw_frame())
    assert a["card_id"].tolist() == b["card_id"].tolist()


def test_dist_km_is_haversine():
    df = clean(raw_frame(n_per_card=2))
    first = df[df["ts"] == df["ts"].min()].iloc[0]
    assert first["dist_km"] == pytest.approx(0.0, abs=1e-6)
    second = df.sort_values("ts").iloc[3]  # merch_lat offset 0.01 deg ~ 1.11 km
    assert second["dist_km"] == pytest.approx(1.11, abs=0.02)


def test_assign_splits_val_is_last_days_before_test_start():
    df = clean(raw_frame())
    test_start = pd.Timestamp("2019-05-01")
    out = assign_splits(df, test_start=test_start, val_days=30)
    assert set(out["split"].unique()) == {"train", "val", "test"}
    assert (out.loc[out["split"] == "test", "ts"] >= test_start).all()
    val = out.loc[out["split"] == "val", "ts"]
    assert (val < test_start).all() and (val >= test_start - pd.Timedelta(days=30)).all()
    assert (out.loc[out["split"] == "train", "ts"] < test_start - pd.Timedelta(days=30)).all()


def test_eda_stats_keys_and_fraud_rate():
    df = assign_splits(clean(raw_frame()), test_start=pd.Timestamp("2019-05-01"))
    s = eda_stats(df)
    assert s["n_transactions"] == 120 and s["n_cards"] == 3
    assert s["fraud_rate"] == pytest.approx(3 / 120)
    assert set(s["rows_per_split"]) == {"train", "val", "test"}
    assert "tx_per_card" in s and "gap_hours" in s and "fraud_rate_by_month" in s
    assert "fraud_burst_share" in s


def test_run_writes_parquet_and_json(tmp_path):
    raw = tmp_path / "raw"; raw.mkdir()
    df = raw_frame()
    df.iloc[:60].to_csv(raw / "fraudTrain.csv", index=False)
    df.iloc[60:].to_csv(raw / "fraudTest.csv", index=False)
    out = tmp_path / "processed"; res = tmp_path / "results"
    stats = run(raw_dir=raw, out_dir=out, results_dir=res)
    assert (out / "transactions.parquet").exists()
    assert (res / "01_data.json").exists()
    back = pd.read_parquet(out / "transactions.parquet")
    assert len(back) == 120 and "split" in back.columns
    assert stats["test_start"] == str(pd.read_csv(raw / "fraudTest.csv")["trans_date_trans_time"].min())

"""Step 1: load Sparkov, strip PII, sort by card and time, split by time, basic EDA.

Output is one parquet with a ``split`` column (train / val / test) so later steps
can read a card's full history while training only on train-period rows.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

# Columns that identify a person or a single transaction. Removed before anything else runs.
PII_COLUMNS = ["first", "last", "gender", "street", "city", "state", "zip", "city_pop", "job", "dob", "trans_num"]
KEEP = ["card_id", "ts", "unix_time", "category", "merchant", "amt", "dist_km", "is_fraud"]


def load_raw(raw_dir: Path) -> tuple[pd.DataFrame, pd.Timestamp]:
    """Concatenate fraudTrain.csv and fraudTest.csv; return the frame and the test period start."""
    raw_dir = Path(raw_dir)
    tr = pd.read_csv(raw_dir / "fraudTrain.csv", parse_dates=["trans_date_trans_time"])
    te = pd.read_csv(raw_dir / "fraudTest.csv", parse_dates=["trans_date_trans_time"])
    return pd.concat([tr, te], ignore_index=True), te["trans_date_trans_time"].min()


def _haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Drop PII, replace cc_num with a dense card_id, add cardholder-to-merchant distance, sort."""
    df = df.drop(columns=[c for c in PII_COLUMNS + ["Unnamed: 0"] if c in df.columns])
    df = df.rename(columns={"trans_date_trans_time": "ts"})
    df["ts"] = pd.to_datetime(df["ts"])
    df["card_id"] = pd.factorize(df["cc_num"].astype(np.int64), sort=True)[0]
    df["dist_km"] = _haversine_km(df["lat"].values, df["long"].values, df["merch_lat"].values, df["merch_long"].values)
    df = df.sort_values(["card_id", "ts"], kind="mergesort").reset_index(drop=True)
    return df[KEEP]


def assign_splits(df: pd.DataFrame, test_start: pd.Timestamp, val_days: int = 30) -> pd.DataFrame:
    """test = rows at or after test_start; val = the val_days before it; train = everything earlier."""
    val_start = test_start - pd.Timedelta(days=val_days)
    df = df.copy()
    df["split"] = np.where(df["ts"] >= test_start, "test", np.where(df["ts"] >= val_start, "val", "train"))
    return df


def eda_stats(df: pd.DataFrame) -> dict:
    """Numbers the report needs before any model is trained."""
    gaps = df.groupby("card_id")["ts"].diff().dt.total_seconds().div(3600).dropna()
    per_card = df.groupby("card_id").size()
    by_month = df.groupby(df["ts"].dt.to_period("M"))["is_fraud"].mean()
    fraud = df[df["is_fraud"] == 1]
    fraud_gap = fraud.groupby("card_id")["ts"].diff().dt.total_seconds().div(3600)
    q = lambda s: {f"p{p}": float(np.percentile(s, p)) for p in (5, 25, 50, 75, 95)}
    return {
        "n_transactions": int(len(df)),
        "n_cards": int(df["card_id"].nunique()),
        "fraud_rate": float(df["is_fraud"].mean()),
        "rows_per_split": {k: int(v) for k, v in df["split"].value_counts().items()},
        "fraud_rate_per_split": {k: float(v) for k, v in df.groupby("split")["is_fraud"].mean().items()},
        "tx_per_card": q(per_card),
        "gap_hours": q(gaps),
        "fraud_rate_by_month": {str(k): float(v) for k, v in by_month.items()},
        # share of fraud rows whose previous fraud on the same card was within 24 h: are frauds bursts?
        "fraud_burst_share": float((fraud_gap < 24).mean()) if len(fraud_gap.dropna()) else 0.0,
        "n_categories": int(df["category"].nunique()),
        "n_merchants": int(df["merchant"].nunique()),
        "amt": q(df["amt"]),
        "dist_km": q(df["dist_km"]),
    }


def run(raw_dir: Path, out_dir: Path, results_dir: Path, val_days: int = 30) -> dict:
    """Step 1 end to end: raw csv -> processed/transactions.parquet + results/01_data.json."""
    out_dir, results_dir = Path(out_dir), Path(results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    raw, test_start = load_raw(raw_dir)
    df = assign_splits(clean(raw), test_start=test_start, val_days=val_days)
    df.to_parquet(out_dir / "transactions.parquet", index=False)
    stats = eda_stats(df)
    stats["test_start"] = str(test_start)
    stats["val_start"] = str(test_start - pd.Timedelta(days=val_days))
    (results_dir / "01_data.json").write_text(json.dumps(stats, indent=2))
    return stats

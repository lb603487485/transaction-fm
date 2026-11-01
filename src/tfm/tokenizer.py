"""Step 2: one card transaction -> a fixed-order row of 5 to 7 tokens.

    <TX>  DT_k  HOUR_k  CAT_x  MER_x  AMT_k  DIST_k
     |     |      |       |      |      |      |
     |     |      |       |      |      |      distance home->merchant, log bins (switch use_dist)
     |     |      |       |      |      amount, log or linear bins, edges fit on train only
     |     |      |       |      merchant if in the train top-N, else MER_OTHER
     |     |      |       category, one token each (unseen -> <UNK>)
     |     |      hour of day in 6 periods of 4 h
     |     minutes since the card's previous transaction, 12 fixed log-spaced bins; DT_NA for a card's first row
     transaction separator

Everything that is *learned* from data (amount edges, distance edges, merchant top-N, category list)
is fit in ``fit`` which refuses any val/test row. The DT and HOUR bins are fixed by definition.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from tfm.data import assert_train_only

SPECIAL = ["<PAD>", "<BOS>", "<MASK>", "<UNK>", "<TX>"]
DT_EDGES_MIN = [1, 10, 30, 60, 180, 360, 720, 1440, 2880, 7200, 43200]   # <1m, 1-10m, ..., 5-30d, >30d  (12 bins)
HOUR_NAMES = ["LATE_NIGHT", "EARLY_MORNING", "MORNING", "AFTERNOON", "EVENING", "NIGHT"]   # hour // 4


def _log_edges(x: np.ndarray, n_bins: int, linear: bool = False) -> np.ndarray:
    """n_bins - 1 interior edges between the 1st and 99th percentile, evenly spaced in log1p (or raw) space."""
    lo, hi = np.percentile(x, 1), np.percentile(x, 99)
    if linear:
        return np.linspace(lo, hi, n_bins - 1)
    return np.expm1(np.linspace(np.log1p(lo), np.log1p(hi), n_bins - 1))


class TransactionTokenizer:
    def __init__(self, use_dt: bool = True, use_dist: bool = True, amt_binning: str = "log",
                 n_amt_bins: int = 20, n_dist_bins: int = 8, top_merchants: int = 500):
        assert amt_binning in ("log", "linear")
        self.config = dict(use_dt=use_dt, use_dist=use_dist, amt_binning=amt_binning,
                           n_amt_bins=n_amt_bins, n_dist_bins=n_dist_bins, top_merchants=top_merchants)
        self.vocab: dict[str, int] = {}
        self.amt_edges = self.dist_edges = None
        self.categories: list[str] = []
        self.merchants: list[str] = []

    # ---- fit -----------------------------------------------------------------------------------
    def fit(self, train_df: pd.DataFrame) -> "TransactionTokenizer":
        assert_train_only(train_df, "tokenizer.fit")
        c = self.config
        self.categories = sorted(train_df["category"].unique())
        counts = train_df["merchant"].value_counts()
        self.merchants = sorted(counts.index[: c["top_merchants"]].tolist())
        self.amt_edges = _log_edges(train_df["amt"].to_numpy(), c["n_amt_bins"], linear=c["amt_binning"] == "linear")
        self.dist_edges = _log_edges(train_df["dist_km"].to_numpy(), c["n_dist_bins"])
        self._build_vocab()
        return self

    def _build_vocab(self) -> None:
        c = self.config
        toks = list(SPECIAL)
        if c["use_dt"]:
            toks += ["DT_NA"] + [f"DT_{i:02d}" for i in range(len(DT_EDGES_MIN) + 1)]
        toks += [f"HOUR_{h}" for h in HOUR_NAMES]
        toks += [f"CAT_{x}" for x in self.categories]
        toks += [f"MER_{x}" for x in self.merchants] + ["MER_OTHER"]
        toks += [f"AMT_{i:02d}" for i in range(c["n_amt_bins"])]
        if c["use_dist"]:
            toks += [f"DIST_{i:02d}" for i in range(c["n_dist_bins"])]
        self.vocab = {t: i for i, t in enumerate(toks)}
        self.id2tok = toks

    @property
    def tokens_per_tx(self) -> int:
        return 5 + int(self.config["use_dt"]) + int(self.config["use_dist"])

    # ---- binning helpers (public so the notebook can show them) -----------------------------------
    def dt_bin(self, gap_minutes: np.ndarray) -> np.ndarray:
        return np.digitize(gap_minutes, DT_EDGES_MIN)

    def amt_bin(self, amt: np.ndarray) -> np.ndarray:
        return np.digitize(amt, self.amt_edges)

    def dist_bin(self, km: np.ndarray) -> np.ndarray:
        return np.digitize(km, self.dist_edges)

    # ---- encode / decode ------------------------------------------------------------------------
    def encode(self, df: pd.DataFrame) -> np.ndarray:
        """(n_rows, tokens_per_tx) int32 ids. df must be in card, time order (as transactions.parquet is);
        DT is the gap to the previous row of the same card inside df."""
        v, c = self.vocab, self.config
        cols = [np.full(len(df), v["<TX>"])]
        if c["use_dt"]:
            gap = df.groupby("card_id")["ts"].diff().dt.total_seconds().div(60).to_numpy()
            dt = np.where(np.isnan(gap), v["DT_NA"], v["DT_00"] + self.dt_bin(np.nan_to_num(gap)))
            cols.append(dt)
        cols.append(v["HOUR_LATE_NIGHT"] + df["ts"].dt.hour.to_numpy() // 4)
        cols.append(df["category"].map(lambda x: v.get(f"CAT_{x}", v["<UNK>"])).to_numpy())
        cols.append(df["merchant"].map(lambda x: v.get(f"MER_{x}", v["MER_OTHER"])).to_numpy())
        cols.append(v["AMT_00"] + self.amt_bin(df["amt"].to_numpy()))
        if c["use_dist"]:
            cols.append(v["DIST_00"] + self.dist_bin(df["dist_km"].to_numpy()))
        return np.stack(cols, axis=1).astype(np.int32)

    def decode(self, ids) -> list[str]:
        return [self.id2tok[int(i)] for i in np.asarray(ids).ravel()]

    # ---- persistence -------------------------------------------------------------------------------
    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps({
            "config": self.config, "vocab": self.id2tok, "categories": self.categories, "merchants": self.merchants,
            "amt_edges": self.amt_edges.tolist(), "dist_edges": self.dist_edges.tolist(), "dt_edges_min": DT_EDGES_MIN,
        }, indent=1))

    @classmethod
    def load(cls, path: Path) -> "TransactionTokenizer":
        j = json.loads(Path(path).read_text())
        tok = cls(**j["config"])
        tok.categories, tok.merchants = j["categories"], j["merchants"]
        tok.amt_edges, tok.dist_edges = np.array(j["amt_edges"]), np.array(j["dist_edges"])
        tok._build_vocab()
        assert tok.id2tok == j["vocab"]
        return tok

    def vocab_summary(self) -> pd.Series:
        """Token count per field prefix, for the notebook."""
        prefix = pd.Series([t.split("_")[0] if not t.startswith("<") else "special" for t in self.id2tok])
        return prefix.value_counts().reindex(["special", "DT", "HOUR", "CAT", "MER", "AMT", "DIST"]).dropna().astype(int)


def make_windows(ids: np.ndarray, card_id: np.ndarray, split: np.ndarray, target_split: str,
                 seq_len: int = 512, stride: int = 256, pad_id: int = 0) -> tuple[np.ndarray, pd.DataFrame]:
    """Concatenate one card's token rows (rows of `target_split` only) in time order and cut windows of
    seq_len tokens with the given stride. A card with fewer than seq_len tokens gives one right-padded
    window. Windows never cross cards and never mix splits, so train windows hold train rows only and
    val/test windows hold only their own rows. Returns (n_windows, seq_len) int32 and a meta frame."""
    keep = split == target_split
    out, meta = [], []
    for cid in pd.unique(card_id[keep]):
        flat = ids[keep & (card_id == cid)].ravel()
        n = len(flat)
        if n < seq_len:
            w = np.full(seq_len, pad_id, dtype=np.int32); w[:n] = flat
            out.append(w); meta.append((cid, target_split, 0, n))
            continue
        for s in range(0, n - seq_len + 1, stride):
            out.append(flat[s:s + seq_len]); meta.append((cid, target_split, s, seq_len))
    arr = np.stack(out).astype(np.int32) if out else np.zeros((0, seq_len), dtype=np.int32)
    return arr, pd.DataFrame(meta, columns=["card_id", "split", "start", "n_real"])

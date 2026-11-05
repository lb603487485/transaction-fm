"""Step 6e: a hybrid head. The transformer reads the 92-token window as before; the classification head
also receives four raw numbers about the judged transaction that the tokenizer threw away or capped.

6a's error analysis found the eight frauds only XGBoost caught were isolated large amounts (median amt_z 7.3):
the AMT token has 20 bins and its top bin starts at the train p99, so 600 and 6,000 dollars look the same to
the sequence model. XGBoost sees the raw amount and its z-score against the card's history. This arm gives
the head those numbers and asks whether the gap closes.

    features (per row, from baseline.build_features, standardised on train rows only, NaN -> 0 = train mean)
        log_amt          log1p of the amount
        amt_z            (amt - history mean) / history std of the same card
        amt_ratio_mean   amt / history mean
        gap_prev_s       log1p seconds since the card's previous transaction
    model   GPT2HybridForSequenceClassification: GPT2 body (from pretrain/best) -> last real token's 384 vector,
            concatenated with the 4 features -> Linear(388, 64) -> GELU -> Linear(64, 2)
    runs    3 seeds, otherwise the reference recipe (lr 5e-5, 3 epochs, keep 0.05), compared with 06b's H12 arm

Only one thing changes against the reference: the head's inputs. Pooling, body, data, sampling and schedule
are the reference's.
"""
from __future__ import annotations

import numpy as np

from tfm.baseline import build_features
from tfm.data import assert_train_only

HYBRID_FEATURES = ["log_amt", "amt_z", "amt_ratio_mean", "gap_prev_s"]
HEAD_HIDDEN = 64


def head_size(n_embd: int, n_feats: int, hidden: int = HEAD_HIDDEN, num_labels: int = 2) -> int:
    return (n_embd + n_feats) * hidden + hidden + hidden * num_labels + num_labels


def numeric_features(df, feats=None) -> tuple[np.ndarray, dict]:
    """(n_rows, 4) float32, standardised with train-row mean and std; NaN (no history) becomes 0 after standardising."""
    f = build_features(df) if feats is None else feats
    raw = f[HYBRID_FEATURES].copy()
    raw["gap_prev_s"] = np.log1p(raw["gap_prev_s"])
    train = raw[df["split"].to_numpy() == "train"]
    assert_train_only(df[df["split"] == "train"], "hybrid feature standardisation")
    stats = {c: {"mean": float(train[c].mean()), "std": float(train[c].std(ddof=0) or 1.0)} for c in HYBRID_FEATURES}
    out = np.stack([((raw[c] - stats[c]["mean"]) / stats[c]["std"]).to_numpy() for c in HYBRID_FEATURES], axis=1)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32), stats


def hybrid_spec(seed: int) -> dict:
    return {"exp": "hybrid", "arch": "hybrid", "history": 12, "pooling": "last", "init": "pretrained", "seed": seed,
            "lr": 5e-5, "keep": 0.05, "freeze": False, "name": f"06e_hybrid_seed{seed}", "out_rel": f"extras/hybrid/seed{seed}"}


# ---- torch side ----------------------------------------------------------------------------------------------
class HybridDataset:
    def __init__(self, X: np.ndarray, y: np.ndarray, pad_id: int, F: np.ndarray):
        self.X, self.y, self.pad_id, self.F = X, y, pad_id, F

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, i: int) -> dict:
        import torch

        ids = torch.as_tensor(self.X[i].astype(np.int64))
        return {"input_ids": ids, "attention_mask": (ids != self.pad_id).long(), "feats": torch.as_tensor(self.F[i], dtype=torch.float32),
                "labels": torch.as_tensor(int(self.y[i]))}


def _hybrid_class():
    import torch
    from torch import nn
    from transformers import GPT2Model, GPT2PreTrainedModel
    from transformers.modeling_outputs import SequenceClassifierOutput

    class GPT2HybridForSequenceClassification(GPT2PreTrainedModel):
        """Last-token pooling as in the reference, plus config.n_feats raw numbers concatenated before a 2-layer head."""

        def __init__(self, config):
            super().__init__(config)
            self.num_labels = config.num_labels
            self.transformer = GPT2Model(config)
            self.score = nn.Sequential(nn.Linear(config.n_embd + config.n_feats, HEAD_HIDDEN), nn.GELU(), nn.Linear(HEAD_HIDDEN, config.num_labels))
            self.post_init()

        def forward(self, input_ids=None, attention_mask=None, feats=None, labels=None, **kwargs):
            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids)
            h = self.transformer(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
            last = (attention_mask.sum(1) - 1).clamp(min=0)
            pooled = h[torch.arange(h.shape[0], device=h.device), last]
            logits = self.score(torch.cat([pooled, feats.to(pooled.dtype)], dim=1))
            loss = nn.functional.cross_entropy(logits, labels) if labels is not None else None
            return SequenceClassifierOutput(loss=loss, logits=logits)

    return GPT2HybridForSequenceClassification


def __getattr__(name):                                   # lazy, so importing tfm.hybrid needs no torch
    if name == "GPT2HybridForSequenceClassification":
        return _hybrid_class()
    raise AttributeError(name)


def load_hybrid(pretrained_dir: str, pad_id: int, n_feats: int):
    from transformers import GPT2Config

    cfg = GPT2Config.from_pretrained(pretrained_dir); cfg.num_labels = 2; cfg.n_feats = n_feats; cfg.pad_token_id = pad_id
    return _hybrid_class().from_pretrained(pretrained_dir, config=cfg)

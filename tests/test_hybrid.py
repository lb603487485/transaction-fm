import numpy as np
import pandas as pd
import pytest

from tfm.ablate import run_name, run_paths
from tfm.hybrid import HYBRID_FEATURES, HybridDataset, hybrid_spec, numeric_features


def _frame():
    rng = np.random.default_rng(0)
    rows = []
    for cid in range(3):
        t = pd.Timestamp("2020-01-01") + pd.to_timedelta(np.cumsum(rng.exponential(5, 60)), unit="h")
        rows.append(pd.DataFrame({"card_id": cid, "ts": t, "category": rng.choice(["a", "b"], 60), "merchant": rng.choice(["m1", "m2", "m3"], 60),
                                  "amt": rng.lognormal(3, 1, 60).round(2), "dist_km": rng.lognormal(3, 1, 60), "is_fraud": (rng.random(60) < 0.1).astype(int)}))
    df = pd.concat(rows, ignore_index=True).sort_values(["card_id", "ts"]).reset_index(drop=True)
    df["split"] = np.where(df["ts"] < df["ts"].quantile(0.7), "train", "test")
    return df


def test_numeric_features_are_standardised_on_train_rows_only_and_have_no_nan():
    df = _frame()
    F, stats = numeric_features(df)
    assert F.shape == (len(df), len(HYBRID_FEATURES)) and F.dtype == np.float32 and not np.isnan(F).any()
    tr = (df["split"] == "train").to_numpy()
    assert np.allclose(np.nanmean(F[tr], axis=0), 0, atol=0.05) and np.allclose(np.nanstd(F[tr], axis=0), 1, atol=0.15)
    assert list(stats) == HYBRID_FEATURES and all({"mean", "std"} <= set(v) for v in stats.values())
    first_rows = df.groupby("card_id").head(1).index.to_numpy()
    assert (F[first_rows, HYBRID_FEATURES.index("amt_z")] == 0).all()          # no history -> NaN -> 0 (the train mean)
    df2 = df.copy(); df2.loc[df2["split"] == "test", "amt"] *= 100
    assert np.allclose(numeric_features(df2)[1]["log_amt"]["mean"], stats["log_amt"]["mean"])   # test rows do not move the fit


def test_hybrid_dataset_returns_the_features_next_to_the_tokens():
    torch = pytest.importorskip("torch")
    X = np.array([[1, 5, 6, 0], [1, 7, 0, 0]], dtype=np.int16); y = np.array([1, 0], dtype=np.int8)
    F = np.array([[0.1, 0.2, 0.3, 0.4], [1, 2, 3, 4]], dtype=np.float32)
    item = HybridDataset(X, y, 0, F)[1]
    assert set(item) == {"input_ids", "attention_mask", "feats", "labels"}
    assert item["feats"].dtype == torch.float32 and torch.equal(item["feats"], torch.tensor([1, 2, 3, 4.0])) and item["attention_mask"].tolist() == [1, 1, 0, 0]


def test_hybrid_spec_names_and_paths():
    r = hybrid_spec(2)
    assert r["arch"] == "hybrid" and r["history"] == 12 and r["init"] == "pretrained" and r["seed"] == 2
    assert run_name(r) == "06e_hybrid_seed2" and run_paths(r, "/r", "/r/results")[0] == "/r/extras/hybrid/seed2"


@pytest.fixture
def tiny():
    torch = pytest.importorskip("torch")
    from tfm.pretrain import gpt2_config
    return torch, gpt2_config(vocab_size=20, pad_id=0, bos_id=1, n_positions=16, n_layer=1, n_embd=8, n_head=2)


def test_hybrid_model_reads_the_last_real_token_and_the_features(tiny):
    torch, cfg = tiny
    from tfm.hybrid import GPT2HybridForSequenceClassification
    cfg.num_labels, cfg.n_feats = 2, 4
    torch.manual_seed(0)
    m = GPT2HybridForSequenceClassification(cfg).eval()
    ids = torch.tensor([[1, 5, 6, 7, 0, 0]]); mask = (ids != 0).long(); f = torch.tensor([[0.0, 0.0, 0.0, 0.0]])
    other = ids.clone(); other[0, 4:] = 9
    with torch.no_grad():
        a = m(input_ids=ids, attention_mask=mask, feats=f).logits
        b = m(input_ids=other, attention_mask=mask, feats=f).logits
        c = m(input_ids=ids, attention_mask=mask, feats=f + 3.0).logits
    assert a.shape == (1, 2) and torch.allclose(a, b, atol=1e-6) and not torch.allclose(a, c, atol=1e-4)
    assert m(input_ids=ids, attention_mask=mask, feats=f, labels=torch.tensor([1])).loss.item() > 0


def test_hybrid_loads_the_pretrained_body_and_scores_with_features(tiny, tmp_path):
    torch, cfg = tiny
    from transformers import GPT2LMHeadModel
    from tfm.ablate import load_model, score_test, trainable_summary
    from tfm.hybrid import head_size
    torch.manual_seed(1)
    lm = GPT2LMHeadModel(cfg); lm.save_pretrained(tmp_path / "best")
    m = load_model("pretrained", "last", str(tmp_path / "best"), vocab_size=20, pad_id=0, bos_id=1, arch="hybrid", n_feats=4)
    assert torch.equal(m.transformer.h[0].attn.c_attn.weight, lm.transformer.h[0].attn.c_attn.weight)
    s = trainable_summary(m)
    assert s["score"] == head_size(8, 4) == (8 + 4) * 64 + 64 + 64 * 2 + 2
    X = np.array([[1, 5, 6, 0], [1, 7, 0, 0], [1, 2, 3, 4]], dtype=np.int16); F = np.zeros((3, 4), np.float32)
    p = score_test(m, X, pad_id=0, device="cpu", batch_size=2, feats=F)
    assert p.shape == (3,) and p.dtype == np.float32

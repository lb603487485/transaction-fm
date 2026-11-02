import json

import numpy as np
import pytest

from tfm.ablate import RUNS, best_checkpoint, run_name, run_paths


def test_runs_cover_the_history_and_pooling_grids_with_three_seeds_each():
    hist = [(r["history"], r["seed"]) for r in RUNS if r["pooling"] == "last"]
    assert sorted(hist) == [(h, s) for h in (0, 5, 12, 32) for s in (0, 1, 2)]
    mean = [(r["history"], r["seed"]) for r in RUNS if r["pooling"] == "mean"]
    assert sorted(mean) == [(12, s) for s in (0, 1, 2)]
    assert all(r["init"] == "pretrained" for r in RUNS)
    assert RUNS[0] == {"history": 12, "pooling": "last", "init": "pretrained", "seed": 0}   # the gate G3 run goes first
    assert len({run_name(r) for r in RUNS}) == len(RUNS) == 15


def test_run_paths_put_every_run_under_ablate_and_results():
    r = {"history": 5, "pooling": "mean", "init": "pretrained", "seed": 2}
    out, res = run_paths(r, root="/r", results_dir="/r/results")
    assert out == "/r/ablate/H5_mean/seed2" and res == "/r/results/06b_H5_mean_seed2.json"


def test_best_checkpoint_follows_trainer_state_even_when_it_was_written_on_another_machine(tmp_path):
    out = tmp_path / "run"
    for step in (2500, 3156):
        d = out / f"checkpoint-{step}"; d.mkdir(parents=True)
        for f in ("model.safetensors", "optimizer.pt", "scheduler.pt"):
            (d / f).write_bytes(b"")                                        # complete, as find_latest_checkpoint requires
        (d / "trainer_state.json").write_text(json.dumps({"global_step": step,
            "best_model_checkpoint": f"/content/drive/MyDrive/Colab Notebooks/tfm/finetune/x/seed0/checkpoint-2500"}))
    assert best_checkpoint(out) == out / "checkpoint-2500"              # remapped onto the local folder
    assert best_checkpoint(tmp_path / "nothing") is None


@pytest.fixture
def tiny():
    torch = pytest.importorskip("torch")
    from tfm.pretrain import gpt2_config
    cfg = gpt2_config(vocab_size=20, pad_id=0, bos_id=1, n_positions=16, n_layer=1, n_embd=8, n_head=2)
    return torch, cfg


def test_mean_pool_classifier_ignores_pad_positions(tiny):
    torch, cfg = tiny
    from tfm.ablate import GPT2MeanPoolForSequenceClassification
    cfg.num_labels = 2
    torch.manual_seed(0)
    m = GPT2MeanPoolForSequenceClassification(cfg).eval()
    ids = torch.tensor([[1, 5, 6, 7, 0, 0]]); mask = (ids != 0).long()
    other = torch.tensor([[1, 5, 6, 7, 0, 0]]); other[0, 4:] = 9             # different ids where the mask is 0
    with torch.no_grad():
        a = m(input_ids=ids, attention_mask=mask).logits
        b = m(input_ids=other, attention_mask=mask).logits
        c = m(input_ids=torch.tensor([[1, 5, 6, 0, 0, 0]]), attention_mask=torch.tensor([[1, 1, 1, 0, 0, 0]])).logits
    assert a.shape == (1, 2) and torch.allclose(a, b, atol=1e-6) and not torch.allclose(a, c, atol=1e-4)
    out = m(input_ids=ids, attention_mask=mask, labels=torch.tensor([1]))
    assert out.loss.ndim == 0 and out.loss.item() > 0


def test_mean_pool_classifier_loads_the_pretrained_body(tiny, tmp_path):
    torch, cfg = tiny
    from transformers import GPT2LMHeadModel
    from tfm.ablate import load_model
    torch.manual_seed(1)
    lm = GPT2LMHeadModel(cfg); lm.save_pretrained(tmp_path / "best")
    for pooling in ("last", "mean"):
        m = load_model("pretrained", pooling, str(tmp_path / "best"), vocab_size=20, pad_id=0, bos_id=1)
        assert torch.equal(m.transformer.h[0].attn.c_attn.weight, lm.transformer.h[0].attn.c_attn.weight)
        assert m.config.pad_token_id == 0 and m.config.num_labels == 2
    s = load_model("scratch", "mean", None, vocab_size=20, pad_id=0, bos_id=1)
    assert not torch.equal(s.transformer.h[0].attn.c_attn.weight, lm.transformer.h[0].attn.c_attn.weight)


def test_score_test_returns_one_fraud_probability_per_row(tiny):
    torch, cfg = tiny
    from tfm.ablate import GPT2MeanPoolForSequenceClassification, score_test
    cfg.num_labels = 2
    m = GPT2MeanPoolForSequenceClassification(cfg)
    X = np.array([[1, 5, 6, 0], [1, 7, 0, 0], [1, 2, 3, 4]], dtype=np.int16)
    p = score_test(m, X, pad_id=0, device="cpu", batch_size=2)
    assert p.shape == (3,) and p.dtype == np.float32 and ((p >= 0) & (p <= 1)).all()

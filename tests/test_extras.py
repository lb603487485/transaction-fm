import pytest

from tfm.ablate import run_name, run_paths
from tfm.extras import LRS, KEEPS, overrides_for, select_lr, spec


def test_specs_name_and_place_every_experiment_under_extras():
    r = spec("lr", lr=1e-4, seed=0)
    assert run_name(r) == "06d_lr1e-04_seed0" and run_paths(r, "/r", "/r/results")[0] == "/r/extras/lr_1e-04/seed0"
    assert r["history"] == 12 and r["pooling"] == "last" and r["init"] == "pretrained" and r["freeze"] is False and r["keep"] == 0.05
    p = spec("probe", init="scratch", seed=2)
    assert run_name(p) == "06d_probe_scratch_seed2" and p["freeze"] is True and p["lr"] == 1e-3
    k = spec("keep", keep=0.02, seed=1)
    assert run_name(k) == "06d_keep0.02_seed1" and run_paths(k, "/r", "/r/results")[0] == "/r/extras/keep_0.02/seed1" and k["keep"] == 0.02
    assert LRS == (1e-5, 2e-5, 1e-4, 2e-4) and KEEPS == (0.02, 0.10)


def test_overrides_only_carry_the_learning_rate():
    assert overrides_for(spec("lr", lr=2e-5, seed=0)) == {"learning_rate": 2e-5}
    assert overrides_for(spec("probe", init="pretrained", seed=0)) == {"learning_rate": 1e-3}


def test_select_lr_picks_the_highest_val_pr_auc_including_the_reference():
    sweep = [{"lr": 1e-5, "val_at_best": {"pr_auc": 0.97}}, {"lr": 1e-4, "val_at_best": {"pr_auc": 0.985}}]
    assert select_lr(sweep, ref_lr=5e-5, ref_val=0.98) == 1e-4
    assert select_lr(sweep, ref_lr=5e-5, ref_val=0.99) == 5e-5


@pytest.fixture
def tiny():
    torch = pytest.importorskip("torch")
    from tfm.pretrain import gpt2_config
    return torch, gpt2_config(vocab_size=20, pad_id=0, bos_id=1, n_positions=16, n_layer=1, n_embd=8, n_head=2)


def test_freeze_body_leaves_only_the_score_head_trainable(tiny):
    torch, cfg = tiny
    from transformers import GPT2ForSequenceClassification
    from tfm.ablate import freeze_body, trainable_summary
    cfg.num_labels = 2
    m = GPT2ForSequenceClassification(cfg)
    before = trainable_summary(m)
    assert before["trainable"] == before["total"] and before["score"] == 8 * 2
    freeze_body(m)
    after = trainable_summary(m)
    assert all(mod.p == 0.0 for mod in m.transformer.modules() if isinstance(mod, torch.nn.Dropout))   # a probe reads the representation without noise
    assert after["trainable"] == after["score"] == 16 and after["total"] == before["total"]
    assert set(after) >= {"wte", "wpe", "h", "ln_f", "score", "total", "trainable"}

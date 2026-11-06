import json

import numpy as np
import pytest

from tfm.compare import VARIANTS, finetune_table, ensemble_row


def _write(d, name, pr, machine="mac", accelerator="mps", trainable=100, total=100):
    (d / f"{name}.json").write_text(json.dumps({"name": name, "test": {"pr_auc": pr, "recall_at_alert": pr}, "trainable_params": trainable,
                                               "total_params": total, "runtime": {"machine": machine, "accelerator": accelerator}}))


@pytest.fixture
def results(tmp_path):
    for k, v in enumerate((0.98, 0.99, 0.97)): _write(tmp_path, f"06b_H12_last_seed{k}", v)              # reference, mean 0.98 sd 0.01
    for k, v in enumerate((0.97, 0.96, 0.95)): _write(tmp_path, f"06b_H5_last_seed{k}", v)
    for k, v in enumerate((0.985, 0.986, 0.984)): _write(tmp_path, f"04_pretrained_H12_seed{k}", v, machine="colab", accelerator="cuda")
    for k, v in enumerate((0.94, 0.95, 0.96)): _write(tmp_path, f"06d_probe_pretrained_seed{k}", v, trainable=2)
    for k, v in enumerate((0.975, 0.975, 0.975)): _write(tmp_path, f"05_xgb_full_seed{k}", v, machine="mac", accelerator="cpu")
    for k, v in enumerate((0.983, 0.984, 0.982)): _write(tmp_path, f"06c_tokenizer_A_ref_seed{k}", v)
    for k, v in enumerate((0.981, 0.980, 0.982)): _write(tmp_path, f"06c_tokenizer_B_linear_seed{k}", v)
    return tmp_path


def test_every_variant_is_compared_with_its_own_control_on_the_same_machine(results):
    rows = {r["variant"]: r for r in finetune_table(results)}
    ref = rows["reference: pretrained body, full fine-tuning, H12, last token, lr 5e-5, keep 0.05 (06b)"]
    assert ref["n_seeds"] == 3 and ref["pr_auc"] == pytest.approx(0.98) and ref["sd"] == pytest.approx(0.01) and ref["in_control_sd"] == 0.0
    h5 = rows["history 5 instead of 12 (06b)"]
    assert h5["pr_auc"] == pytest.approx(0.96) and h5["diff"] == pytest.approx(-0.02) and h5["in_control_sd"] == pytest.approx(-2.0)
    probe = rows["linear probe: body frozen, pretrained (06d)"]
    assert probe["trainable_params"] == 2 and probe["in_control_sd"] == pytest.approx(-3.0)
    t4 = rows["same configuration trained on the Colab T4 (04)"]
    assert t4["in_control_sd"] is None and t4["note"].startswith("other machine")          # never compared across machines
    xgb = rows["XGBoost, 29 hand-built features (05)"]
    assert xgb["in_control_sd"] is None and xgb["diff"] == pytest.approx(-0.005)             # other family: difference shown, no sd units
    b = rows["tokenizer: linear amount bins (06c)"]
    assert b["control"] == "06c_tokenizer_A_ref" and b["diff"] == pytest.approx(-0.002) and b["in_control_sd"] == pytest.approx(-2.0)


def test_missing_variants_are_skipped_not_invented(results):
    names = [r["variant"] for r in finetune_table(results)]
    assert "hybrid head: last token plus four raw numbers (06e)" not in names and len(names) == 7
    assert len(VARIANTS) > 7


def test_ensemble_row_averages_the_seed_scores_and_compares_in_control_sd():
    y = np.array([1, 0, 0, 1, 0, 0, 0, 0, 0, 0] * 20); card = np.arange(len(y))
    s0 = y + np.random.default_rng(0).normal(0, 0.3, len(y)); s1 = y + np.random.default_rng(1).normal(0, 0.3, len(y))
    row = ensemble_row("mean of two seeds", [s0, s1], y, card, control_mean=0.5, control_sd=0.1, alert=0.2)
    assert row["n_seeds"] == 2 and row["sd"] is None and 0 < row["pr_auc"] <= 1 and row["in_control_sd"] == pytest.approx((row["pr_auc"] - 0.5) / 0.1)

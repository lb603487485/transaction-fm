import json

import numpy as np
import pytest

from tfm.compare import VARIANTS, finetune_table, ensemble_row, log_history


def _write(d, name, pr, machine="mac", accelerator="mps", trainable=100, total=100, **extra):
    (d / f"{name}.json").write_text(json.dumps({"name": name, **extra, "test": {"pr_auc": pr, "roc_auc": 0.99, "recall_at_alert": pr, "card_discovery": 1.0, "fp_ratio": 1 - pr}, "trainable_params": trainable,
                                               "total_params": total, "runtime": {"machine": machine, "accelerator": accelerator}}))


@pytest.fixture
def results(tmp_path):
    for k, v in enumerate((0.98, 0.99, 0.97)): _write(tmp_path, f"06b_H12_last_seed{k}", v, wall_min=20 + k)   # reference, mean 0.98 sd 0.01
    for k, v in enumerate((0.97, 0.96, 0.95)): _write(tmp_path, f"06b_H5_last_seed{k}", v)
    for k, v in enumerate((0.985, 0.986, 0.984)): _write(tmp_path, f"04_pretrained_H12_seed{k}", v, machine="colab", accelerator="cuda")
    for k, v in enumerate((0.94, 0.95, 0.96)): _write(tmp_path, f"06d_probe_pretrained_seed{k}", v, trainable=2)
    for k, v in enumerate((0.975, 0.975, 0.975)): _write(tmp_path, f"05_xgb_full_seed{k}", v, machine="mac", accelerator="cpu", fit_seconds=30)
    for k, v in enumerate((0.983, 0.984, 0.982)): _write(tmp_path, f"06c_tokenizer_A_ref_seed{k}", v)
    for k, v in enumerate((0.981, 0.980, 0.982)): _write(tmp_path, f"06c_tokenizer_B_linear_seed{k}", v)
    return tmp_path


def test_every_variant_is_compared_with_its_own_control_on_the_same_machine(results):
    rows = {r["variant"]: r for r in finetune_table(results)}
    ref = rows["reference (06b)"]
    assert ref["n_seeds"] == 3 and ref["pr_auc"] == pytest.approx(0.98) and ref["sd"] == pytest.approx(0.01) and ref["in_control_sd"] == 0.0
    h5 = rows["history 5 (06b)"]
    assert h5["pr_auc"] == pytest.approx(0.96) and h5["diff"] == pytest.approx(-0.02) and h5["in_control_sd"] == pytest.approx(-2.0)
    probe = rows["linear probe, pretrained body (06d)"]
    assert probe["trainable_params"] == 2 and probe["in_control_sd"] == pytest.approx(-3.0)
    t4 = rows["same configuration on the Colab T4 (04)"]
    assert t4["in_control_sd"] is None and t4["note"].startswith("other machine")          # never compared across machines
    xgb = rows["XGBoost, 29 features (05)"]
    assert xgb["in_control_sd"] is None and xgb["diff"] == pytest.approx(-0.005)             # other family: difference shown, no sd units
    b = rows["linear amount bins (06c)"]
    assert b["control"] == "06c_tokenizer_A_ref" and b["diff"] == pytest.approx(-0.002) and b["in_control_sd"] == pytest.approx(-2.0)


def test_missing_variants_are_skipped_not_invented(results):
    names = [r["variant"] for r in finetune_table(results)]
    assert "hybrid head (06e)" not in names and len(names) == 7
    assert len(VARIANTS) > 7


def test_ensemble_row_averages_the_seed_scores_and_compares_in_control_sd():
    y = np.array([1, 0, 0, 1, 0, 0, 0, 0, 0, 0] * 20); card = np.arange(len(y))
    s0 = y + np.random.default_rng(0).normal(0, 0.3, len(y)); s1 = y + np.random.default_rng(1).normal(0, 0.3, len(y))
    row = ensemble_row("mean of two seeds", [s0, s1], y, card, control_mean=0.5, control_sd=0.1, alert=0.2)
    assert row["n_seeds"] == 2 and row["sd"] is None and 0 < row["pr_auc"] <= 1 and row["in_control_sd"] == pytest.approx((row["pr_auc"] - 0.5) / 0.1)


def test_every_row_carries_all_five_metrics_and_its_run_folder(results):
    rows = {r["variant"]: r for r in finetune_table(results)}
    h5 = rows["history 5 (06b)"]
    assert set(h5["metrics"]) == {"pr_auc", "roc_auc", "recall_at_alert", "card_discovery", "fp_ratio"}
    assert h5["metrics"]["fp_ratio"]["mean"] == pytest.approx(0.04) and h5["metrics"]["roc_auc"]["sd"] == pytest.approx(0.0)
    assert h5["dir"] == "ablate/H5_last/seed{k}" and rows["XGBoost, 29 features (05)"]["dir"] == "baseline/full/seed{k}"
    assert rows["reference (06b)"]["wall_min"] == pytest.approx(21.0) and rows["XGBoost, 29 features (05)"]["wall_min"] == pytest.approx(0.5)
    assert h5["wall_min"] is None                                                               # no timing recorded, no number invented


def test_log_history_reads_the_latest_checkpoint_and_splits_train_from_eval(tmp_path):
    for step, hist in ((500, [{"step": 50, "loss": 0.6}, {"step": 500, "eval_loss": 0.01, "eval_pr_auc": 0.9}]),
                       (1000, [{"step": 50, "loss": 0.6}, {"step": 500, "eval_loss": 0.01, "eval_pr_auc": 0.9}, {"step": 1000, "eval_loss": 0.005, "eval_pr_auc": 0.95}])):
        d = tmp_path / f"checkpoint-{step}"; d.mkdir(); (d / "trainer_state.json").write_text(json.dumps({"log_history": hist}))
    h = log_history(str(tmp_path))
    assert h["train"] == [(50, 0.6)] and h["eval"] == [(500, 0.01, 0.9), (1000, 0.005, 0.95)]
    assert log_history(str(tmp_path / "nowhere")) is None                                   # XGBoost folders have no checkpoints

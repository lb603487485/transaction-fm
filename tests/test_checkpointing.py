import json
from pathlib import Path

import pytest

from tfm.checkpointing import find_latest_checkpoint, training_args, verify_checkpoint

REQUIRED = ["model.safetensors", "optimizer.pt", "scheduler.pt", "trainer_state.json"]


def make_ckpt(out_dir: Path, step: int, complete: bool = True, state_step: int | None = None) -> Path:
    d = out_dir / f"checkpoint-{step}"
    d.mkdir(parents=True)
    files = REQUIRED if complete else REQUIRED[:1]
    for f in files:
        (d / f).write_bytes(b"x")
    if "trainer_state.json" in files:
        (d / "trainer_state.json").write_text(json.dumps({"global_step": state_step if state_step is not None else step}))
    return d


def test_latest_is_highest_step_not_lexical(tmp_path):
    make_ckpt(tmp_path, 500)
    make_ckpt(tmp_path, 1000)
    make_ckpt(tmp_path, 900)
    assert find_latest_checkpoint(tmp_path) == tmp_path / "checkpoint-1000"


def test_no_checkpoints_returns_none(tmp_path):
    assert find_latest_checkpoint(tmp_path) is None
    assert find_latest_checkpoint(tmp_path / "missing") is None


def test_incomplete_checkpoint_is_skipped(tmp_path):
    make_ckpt(tmp_path, 500)
    make_ckpt(tmp_path, 1000, complete=False)  # disconnected mid-save
    assert find_latest_checkpoint(tmp_path) == tmp_path / "checkpoint-500"


def test_verify_complete_checkpoint_has_no_problems(tmp_path):
    d = make_ckpt(tmp_path, 500)
    assert verify_checkpoint(d) == []


def test_verify_reports_missing_files_and_step_mismatch(tmp_path):
    d = make_ckpt(tmp_path, 500, complete=False)
    problems = verify_checkpoint(d)
    assert any("optimizer.pt" in p for p in problems)
    d2 = make_ckpt(tmp_path, 1000, state_step=900)
    problems2 = verify_checkpoint(d2)
    assert any("global_step" in p for p in problems2)


def test_training_args_warmup_is_five_percent_on_the_installed_transformers(tmp_path):
    pytest.importorskip("transformers")
    a = training_args(str(tmp_path), seed=0)
    assert getattr(a, "warmup_ratio", None) == 0.05 or a.warmup_steps == 0.05   # 4.x field or the 5.x float ratio
    assert a.save_steps == 500 and a.eval_steps == 500 and a.load_best_model_at_end and a.metric_for_best_model == "eval_loss"

"""Resumable training on free Colab.

Colab disconnects cannot be prevented, so every run must be able to lose at
most one save interval and restart with one call. Two rules:

1. ``output_dir`` is on Google Drive; the Trainer saves every ``save_steps``.
2. Before training, ``find_latest_checkpoint`` picks the newest *complete*
   checkpoint and the Trainer resumes from it (model, optimizer, LR schedule,
   RNG and global_step all come back). After training, ``verify_checkpoint``
   is run on what is left on Drive so a bad save is caught before the next step
   of the project builds on it.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

REQUIRED_FILES = ("model.safetensors", "optimizer.pt", "scheduler.pt", "trainer_state.json")
_CKPT_RE = re.compile(r"^checkpoint-(\d+)$")


def verify_checkpoint(path: Path) -> list[str]:
    """Return a list of problems; an empty list means the checkpoint is complete and consistent."""
    path = Path(path)
    problems = [f"{path.name}: missing {f}" for f in REQUIRED_FILES if not (path / f).exists()]
    state_file = path / "trainer_state.json"
    m = _CKPT_RE.match(path.name)
    if state_file.exists() and m:
        try:
            step = json.loads(state_file.read_text()).get("global_step")
        except json.JSONDecodeError:
            problems.append(f"{path.name}: trainer_state.json is not valid JSON")
            return problems
        if step != int(m.group(1)):
            problems.append(f"{path.name}: global_step {step} does not match directory name")
    return problems


def find_latest_checkpoint(out_dir: Path) -> Path | None:
    """Newest complete ``checkpoint-N`` under ``out_dir`` by step number, or None."""
    out_dir = Path(out_dir)
    if not out_dir.is_dir():
        return None
    candidates = []
    for p in out_dir.iterdir():
        m = _CKPT_RE.match(p.name)
        if m and p.is_dir() and not verify_checkpoint(p):
            candidates.append((int(m.group(1)), p))
    if not candidates:
        return None
    return max(candidates)[1]


def training_args(out_dir: str, seed: int = 0, **overrides):
    """TrainingArguments with the guide's values and the resume-safe settings.

    Save and eval share the same step so load_best_model_at_end can match them.
    save_total_limit=2 keeps the latest checkpoint and the best one; the best is
    never deleted by the limit.
    """
    import dataclasses

    from transformers import TrainingArguments  # imported here so tests run without torch

    # transformers 4.x: warmup_ratio=0.05; transformers 5.x removed it and takes a float warmup_steps in [0, 1) as the ratio
    names = {f.name for f in dataclasses.fields(TrainingArguments)}
    warmup = {"warmup_ratio": 0.05} if "warmup_ratio" in names else {"warmup_steps": 0.05}
    args = dict(
        output_dir=out_dir,
        seed=seed,
        learning_rate=5e-4,
        lr_scheduler_type="cosine",
        **warmup,
        num_train_epochs=8,
        eval_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=50,
        fp16=True,
        report_to="none",
    )
    args.update(overrides)
    return TrainingArguments(**args)

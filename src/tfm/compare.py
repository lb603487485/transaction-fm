"""Step 6f: every fine-tuning variant of the project in one table, read from results/metrics/*.json. No training.

Each variant is compared with a CONTROL trained on the same machine with the same code, in units of the control's
seed sd; the reference control is 06b's H12 last-token arm, the 06c arms use their own retrained A_ref. Variants
from another machine (the Colab T4 runs of step 4) or another model family (XGBoost) show the difference only,
never sd units: a cross-machine gap is a machine effect (06b gate G3), and a tree's seed sd says nothing about a
transformer's. A variant whose files are absent is skipped, not invented.
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np

from tfm.metrics import evaluate_scores, mean_sd

REF = "06b_H12_last"
# (json stem without _seed<k>, label, control stem, note when the control's sd units do not apply)
VARIANTS = [
    (REF, "reference: pretrained body, full fine-tuning, H12, last token, lr 5e-5, keep 0.05 (06b)", REF, ""),
    ("04_pretrained_H12", "same configuration trained on the Colab T4 (04)", REF, "other machine: difference only"),
    ("04_scratch_H12", "random body instead of pretrained, Colab T4 (04)", REF, "other machine: difference only"),
    ("06b_H0_last", "history 0 instead of 12 (06b)", REF, ""),
    ("06b_H5_last", "history 5 instead of 12 (06b)", REF, ""),
    ("06b_H32_last", "history 32 instead of 12 (06b)", REF, ""),
    ("06b_H12_mean", "mean pooling instead of last token (06b)", REF, ""),
    ("06d_lr1e-05", "learning rate 1e-5, one seed (06d)", REF, ""),
    ("06d_lr2e-05", "learning rate 2e-5, one seed (06d)", REF, ""),
    ("06d_lr1e-04", "learning rate 1e-4, one seed (06d)", REF, ""),
    ("06d_lr2e-04", "learning rate 2e-4, selected on val (06d)", REF, ""),
    ("06d_probe_pretrained", "linear probe: body frozen, pretrained (06d)", REF, ""),
    ("06d_probe_scratch", "linear probe: body frozen, random (06d)", REF, ""),
    ("06d_keep0.02", "negative sampling 2% instead of 5% (06d)", REF, ""),
    ("06d_keep0.10", "negative sampling 10% instead of 5% (06d)", REF, ""),
    ("06e_hybrid", "hybrid head: last token plus four raw numbers (06e)", REF, ""),
    ("06c_tokenizer_A_ref", "tokenizer group control: retrained reference (06c)", "06c_tokenizer_A_ref", ""),
    ("06c_tokenizer_B_linear", "tokenizer: linear amount bins (06c)", "06c_tokenizer_A_ref", ""),
    ("06c_tokenizer_C_no_dt", "tokenizer: no time-gap token (06c)", "06c_tokenizer_A_ref", ""),
    ("06c_objective_A_ref", "objective group control: retrained reference (06c)", "06c_objective_A_ref", ""),
    ("06c_objective_D_mlm", "objective: MLM encoder, same size (06c)", "06c_objective_A_ref", ""),
    ("05_xgb_full", "XGBoost, 29 hand-built features (05)", REF, "other model family: difference only"),
    ("05_xgb_current", "XGBoost, current transaction only (05)", REF, "other model family: difference only"),
]


def _runs(results_dir: str, stem: str) -> list[dict]:
    return [json.load(open(p)) for p in sorted(glob.glob(os.path.join(results_dir, f"{stem}_seed*.json")))]


def _machine(r: dict) -> str:
    rt = r.get("runtime", {})
    return f"{rt.get('machine', '?')} / {rt.get('accelerator', '?')}"


def finetune_table(results_dir: str) -> list[dict]:
    """One row per variant present in results_dir; diff and in_control_sd against its control."""
    groups = {stem: _runs(results_dir, stem) for stem, _, _, _ in VARIANTS}
    rows = []
    for stem, label, control, note in VARIANTS:
        runs = groups[stem]
        if not runs:
            continue
        agg = mean_sd([r["test"] for r in runs])["pr_auc"]
        ctrl = groups.get(control) or []
        c = mean_sd([r["test"] for r in ctrl])["pr_auc"] if ctrl else None
        if c and not note and _machine(runs[0]) != _machine(ctrl[0]):
            note = "other machine: difference only"                       # the same-machine rule, enforced from the jsons
        diff = agg["mean"] - c["mean"] if c else None
        in_sd = diff / max(c["sd"], 1e-9) if (c and not note) else None
        rows.append({"variant": label, "n_seeds": len(runs), "pr_auc": agg["mean"], "sd": agg["sd"] if len(runs) > 1 else None,
                     "recall_at_alert": mean_sd([r["test"] for r in runs])["recall_at_alert"]["mean"],
                     "trainable_params": runs[0].get("trainable_params"), "machine": _machine(runs[0]),
                     "control": control, "diff": diff, "in_control_sd": in_sd, "note": note})
    return rows


def ensemble_row(label: str, scores: list[np.ndarray], y, card, control_mean: float, control_sd: float, alert: float = 0.006) -> dict:
    """The mean of several seeds' test scores, evaluated once; one model, so no sd of its own."""
    m = evaluate_scores(y, np.mean(np.stack(scores, 1), 1), card, alert)
    d = m["pr_auc"] - control_mean
    return {"variant": label, "n_seeds": len(scores), "pr_auc": m["pr_auc"], "sd": None, "recall_at_alert": m["recall_at_alert"],
            "trainable_params": None, "machine": "", "control": "", "diff": d, "in_control_sd": d / max(control_sd, 1e-9), "note": "one model, no sd"}

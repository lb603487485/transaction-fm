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
METRICS = ["pr_auc", "roc_auc", "recall_at_alert", "card_discovery", "fp_ratio"]
PARAMS = ["body", "history", "pooling", "lr", "neg_keep", "trained", "head", "tokenizer", "objective"]
_BASE = {"body": "pretrained", "history": "12", "pooling": "last", "lr": "5e-5", "neg_keep": "0.05", "trained": "all 11.06M",
         "head": "linear 384x2", "tokenizer": "step 2 (log bins, DT)", "objective": "CLM decoder"}
_XGB = {"body": "boosted trees, depth 6", "history": "29 features", "pooling": "", "lr": "0.05 (grid)", "neg_keep": "spw 20",
        "trained": "trees to early stop", "head": "", "tokenizer": "none, raw numbers", "objective": "logloss"}
OTHER_MACHINE, OTHER_FAMILY = "other machine: difference only", "other model family: difference only"


def _v(stem, label, control=REF, note="", run_dir="", **changed):
    base = _XGB if stem.startswith("05_") else _BASE
    return (stem, label, control, note, run_dir, {**base, **changed})


# one tuple per variant: json stem without _seed<k>, label, control stem, note when the control's sd units do not apply,
# run folder under root, and the parameter cells (only the changed ones are passed; the rest come from the reference)
VARIANTS = [
    _v(REF, "reference (06b)", run_dir="ablate/H12_last/seed{k}"),
    _v("04_pretrained_H12", "same configuration on the Colab T4 (04)", note=OTHER_MACHINE, run_dir="finetune/pretrained_H12/seed{k}"),
    _v("04_scratch_H12", "random body, Colab T4 (04)", note=OTHER_MACHINE, run_dir="finetune/scratch_H12/seed{k}", body="random"),
    _v("06b_H0_last", "history 0 (06b)", run_dir="ablate/H0_last/seed{k}", history="0"),
    _v("06b_H5_last", "history 5 (06b)", run_dir="ablate/H5_last/seed{k}", history="5"),
    _v("06b_H32_last", "history 32 (06b)", run_dir="ablate/H32_last/seed{k}", history="32"),
    _v("06b_H12_mean", "mean pooling (06b)", run_dir="ablate/H12_mean/seed{k}", pooling="mean"),
    _v("06d_lr1e-05", "lr 1e-5, one seed (06d)", run_dir="extras/lr_1e-05/seed{k}", lr="1e-5"),
    _v("06d_lr2e-05", "lr 2e-5, one seed (06d)", run_dir="extras/lr_2e-05/seed{k}", lr="2e-5"),
    _v("06d_lr1e-04", "lr 1e-4, one seed (06d)", run_dir="extras/lr_1e-04/seed{k}", lr="1e-4"),
    _v("06d_lr2e-04", "lr 2e-4, selected on val (06d)", run_dir="extras/lr_2e-04/seed{k}", lr="2e-4"),
    _v("06d_probe_pretrained", "linear probe, pretrained body (06d)", run_dir="extras/probe_pretrained/seed{k}", lr="1e-3", trained="head only, 768"),
    _v("06d_probe_scratch", "linear probe, random body (06d)", run_dir="extras/probe_scratch/seed{k}", body="random", lr="1e-3", trained="head only, 768"),
    _v("06d_keep0.02", "negative sampling 2% (06d)", run_dir="extras/keep_0.02/seed{k}", neg_keep="0.02"),
    _v("06d_keep0.10", "negative sampling 10% (06d)", run_dir="extras/keep_0.10/seed{k}", neg_keep="0.10"),
    _v("06e_hybrid", "hybrid head (06e)", run_dir="extras/hybrid/seed{k}", trained="all 11.09M", head="MLP 388-64-2 + 4 raw numbers"),
    _v("06c_tokenizer_A_ref", "tokenizer group control (06c)", control="06c_tokenizer_A_ref", run_dir="ablate_pre/tokenizer/A_ref/finetune/seed{k}", body="pretrained again"),
    _v("06c_tokenizer_B_linear", "linear amount bins (06c)", control="06c_tokenizer_A_ref", run_dir="ablate_pre/tokenizer/B_linear/finetune/seed{k}", body="pretrained again", tokenizer="linear amount bins"),
    _v("06c_tokenizer_C_no_dt", "no time-gap token (06c)", control="06c_tokenizer_A_ref", run_dir="ablate_pre/tokenizer/C_no_dt/finetune/seed{k}", body="pretrained again", tokenizer="no DT token"),
    _v("06c_objective_A_ref", "objective group control (06c)", control="06c_objective_A_ref", run_dir="ablate_pre/objective/A_ref/finetune/seed{k}", body="pretrained again"),
    _v("06c_objective_D_mlm", "MLM encoder (06c)", control="06c_objective_A_ref", run_dir="ablate_pre/objective/D_mlm/finetune/seed{k}", body="pretrained again", objective="MLM encoder (BERT)"),
    _v("05_xgb_full", "XGBoost, 29 features (05)", note=OTHER_FAMILY, run_dir="baseline/full/seed{k}"),
    _v("05_xgb_current", "XGBoost, current row only (05)", note=OTHER_FAMILY, run_dir="baseline/current/seed{k}", history="6 features, current row"),
]


def _runs(results_dir: str, stem: str) -> list[dict]:
    return [json.load(open(p)) for p in sorted(glob.glob(os.path.join(results_dir, f"{stem}_seed*.json")))]


def _machine(r: dict) -> str:
    rt = r.get("runtime", {})
    return f"{rt.get('machine', '?')} / {rt.get('accelerator', '?')}"


def finetune_table(results_dir: str) -> list[dict]:
    """One row per variant present in results_dir; diff and in_control_sd against its control."""
    groups = {stem: _runs(results_dir, stem) for stem, *_ in VARIANTS}
    rows = []
    for stem, label, control, note, run_dir, params in VARIANTS:
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
        allm = mean_sd([r["test"] for r in runs])
        mins = [r["wall_min"] if "wall_min" in r else r["fit_seconds"] / 60 for r in runs if "wall_min" in r or "fit_seconds" in r]
        rows.append({"wall_min": float(np.mean(mins)) if mins else None,"variant": label, "params": params, "n_seeds": len(runs), "pr_auc": agg["mean"], "sd": agg["sd"] if len(runs) > 1 else None,
                     "recall_at_alert": allm["recall_at_alert"]["mean"],
                     "metrics": {k: {"mean": allm[k]["mean"], "sd": allm[k]["sd"]} for k in METRICS if k in allm}, "dir": run_dir,
                     "trainable_params": runs[0].get("trainable_params"), "machine": _machine(runs[0]),
                     "control": control, "diff": diff, "in_control_sd": in_sd, "note": note})
    return rows


def ensemble_row(label: str, scores: list[np.ndarray], y, card, control_mean: float, control_sd: float, alert: float = 0.006) -> dict:
    """The mean of several seeds' test scores, evaluated once; one model, so no sd of its own."""
    m = evaluate_scores(y, np.mean(np.stack(scores, 1), 1), card, alert)
    d = m["pr_auc"] - control_mean
    return {"variant": label, "params": {k: "" for k in PARAMS} | {"trained": f"mean of {len(scores)} seeds"}, "n_seeds": len(scores), "pr_auc": m["pr_auc"], "sd": None, "wall_min": None,
            "recall_at_alert": m["recall_at_alert"], "metrics": {k: {"mean": m[k], "sd": 0.0} for k in METRICS}, "dir": None,
            "trainable_params": None, "machine": "", "control": "", "diff": d, "in_control_sd": d / max(control_sd, 1e-9), "note": "one model, no sd"}


def log_history(run_dir: str) -> dict | None:
    """Train loss and the eval metrics logged while fine-tuning, from the latest checkpoint's trainer_state.json
    (save_total_limit keeps the best and the last; the last has the whole history). None when the folder has no
    checkpoints, which is the case for XGBoost."""
    cps = sorted(glob.glob(os.path.join(run_dir, "checkpoint-*", "trainer_state.json")), key=lambda p: int(p.split("checkpoint-")[1].split(os.sep)[0]))
    if not cps:
        return None
    hist = json.load(open(cps[-1]))["log_history"]
    return {"train": [(e["step"], e["loss"]) for e in hist if "loss" in e],
            "eval": [(e["step"], e["eval_loss"], e["eval_pr_auc"]) for e in hist if "eval_loss" in e]}

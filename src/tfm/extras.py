"""Step 6d: the fine-tuning follow-ups that need no new model code, all from pretrain/best on one machine.

    exp     what changes                            what it answers
    lr      learning rate 1e-5 / 2e-5 / 1e-4 / 2e-4   is the guide's 5e-5 a fair setting (XGBoost had a 12-point grid, the transformer none);
            one seed each, selected on val           the chosen rate gets three seeds if it is not the reference
    probe   body frozen, only the 384 x 2 head trains; pretrained body vs random body      how much the pretrained representation is worth
            lr 1e-3                                  by itself, without retraining the body
    keep    neg_keep_rate 0.02 / 0.10 (0.05 ref)     does the one sampling number the guide fixed matter for ranking and calibration
    ensemble, calibration: no training, computed in the notebook from the 06b H12 runs (calibrate.py)

The reference arm is 06b's H12 last-token seeds 0 to 2, trained on the same machine with the same code.
Folders under root: extras/<exp tag>/seed<k>; result files 06d_<name>.json.
"""
from __future__ import annotations

LRS = (1e-5, 2e-5, 1e-4, 2e-4)
KEEPS = (0.02, 0.10)
REF_LR, REF_KEEP, PROBE_LR = 5e-5, 0.05, 1e-3


def spec(exp: str, seed: int, lr: float | None = None, keep: float | None = None, init: str = "pretrained") -> dict:
    base = {"exp": exp, "history": 12, "pooling": "last", "arch": "gpt2", "seed": seed, "init": init,
            "lr": REF_LR, "keep": REF_KEEP, "freeze": False}
    if exp == "lr":
        base.update(lr=lr, name=f"06d_lr{lr:.0e}_seed{seed}", out_rel=f"extras/lr_{lr:.0e}/seed{seed}")
    elif exp == "probe":
        base.update(lr=PROBE_LR, freeze=True, name=f"06d_probe_{init}_seed{seed}", out_rel=f"extras/probe_{init}/seed{seed}")
    elif exp == "keep":
        base.update(keep=keep, name=f"06d_keep{keep:.2f}_seed{seed}", out_rel=f"extras/keep_{keep:.2f}/seed{seed}")
    else:
        raise ValueError(exp)
    return base


def overrides_for(r: dict) -> dict:
    return {"learning_rate": r["lr"]}


def select_lr(sweep: list[dict], ref_lr: float, ref_val: float) -> float:
    """The learning rate with the highest validation PR-AUC, the reference included; ties go to the reference."""
    best_lr, best_val = ref_lr, ref_val
    for s in sweep:
        if s["val_at_best"]["pr_auc"] > best_val:
            best_lr, best_val = s["lr"], s["val_at_best"]["pr_auc"]
    return best_lr

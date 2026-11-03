"""Step 6c: the ablations that need a new pretraining run, and the perplexity-vs-PR-AUC scatter.

    arm        tokenizer                  objective   what it tests
    A_ref      step 2's (log amt, DT on)  CLM         the control, retrained on the machine the other arms use (same-machine
                                                      rule); its training also leaves snapshots for the scatter
    B_linear   amount in 20 linear bins   CLM         WF: data shape beats data size (tokenizer ablation 1)
    C_no_dt    no time-gap token          CLM         tokenizer ablation 2
    D_mlm      step 2's                   MLM         WF (decoder) vs Wealthsimple (MLM for event-level anomalies)

Two groups, each a self-contained comparison with its own control arm A, so the two can run on two
machines at the same time (same-machine rule) without ever writing the same file:

    tokenizer   A_ref, B_linear, C_no_dt                      3 pretrains + 9 fine-tunes
    objective   A_ref, D_mlm; A leaves snapshots              2 pretrains + 6 fine-tunes + 5 snapshot fine-tunes

Each arm: fit its tokenizer on train rows, encode, cut windows, pretrain (resumable, json completion marker),
then fine-tune H12 last-token with three seeds through ablate.finetune_run. In the objective group arm A also
fine-tunes five log-spaced snapshots of its own pretraining (seed 0 each): perplexity at the snapshot against
test PR-AUC after fine-tuning is the Wealthsimple "intrinsic vs extrinsic" check.

Folders under root: ablate_pre/<group>/<arm>/{data,pretrain,snapshots,finetune/seed<k>,finetune_snap<step>/seed0};
result files 06c_<group>_... so the two groups' jsons never collide in results/metrics.
"""
from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path

import numpy as np

from tfm.checkpointing import find_latest_checkpoint, training_args, verify_checkpoint
from tfm.data import train_only
from tfm.pretrain import WindowDataset, argmax_logits, compute_metrics_fn, field_of, gpt2_config, prior_accuracy, summarize
from tfm.tokenizer import TransactionTokenizer, make_windows

BASE_TOKENIZER = dict(use_dt=True, use_dist=True, amt_binning="log", n_amt_bins=20, n_dist_bins=8, top_merchants=500)
ARMS = {
    "A_ref": {"tokenizer": {}, "objective": "clm"},
    "B_linear": {"tokenizer": {"amt_binning": "linear"}, "objective": "clm"},
    "C_no_dt": {"tokenizer": {"use_dt": False}, "objective": "clm"},
    "D_mlm": {"tokenizer": {}, "objective": "mlm"},
}
GROUPS = {
    "tokenizer": {"arms": ["A_ref", "B_linear", "C_no_dt"], "snapshots": None},
    "objective": {"arms": ["A_ref", "D_mlm"], "snapshots": "A_ref"},
}
SEQ_LEN, STRIDE, HISTORY, SEEDS = 512, 256, 12, (0, 1, 2)


def arm_tokenizer_config(arm: str) -> dict:
    return {**BASE_TOKENIZER, **ARMS[arm]["tokenizer"]}


def snapshot_steps(max_steps: int, save_steps: int, n: int = 5) -> list[int]:
    """n log-spaced multiples of save_steps from save_steps to the last multiple at or before max_steps."""
    last = (max_steps // save_steps) * save_steps
    if last < save_steps:
        return []
    raw = np.exp(np.linspace(np.log(save_steps), np.log(last), n))
    return sorted({int(round(s / save_steps)) * save_steps for s in raw})


def finetune_spec(group: str, arm: str, seed: int, snapshot: int | None = None) -> dict:
    tag = f"{arm}_snap{snapshot}" if snapshot is not None else arm
    sub = f"finetune_snap{snapshot}" if snapshot is not None else "finetune"
    return {"group": group, "arm": arm, "snapshot": snapshot, "history": HISTORY, "pooling": "last", "init": "pretrained", "seed": seed,
            "arch": "bert" if ARMS[arm]["objective"] == "mlm" else "gpt2",
            "name": f"06c_{group}_{tag}_seed{seed}", "out_rel": f"ablate_pre/{group}/{arm}/{sub}/seed{seed}"}


def pretrained_dir(r: dict, root: str) -> str:
    base = f"{root}/ablate_pre/{r['group']}/{r['arm']}"
    if r.get("snapshot") is not None:
        return f"{base}/snapshots/snap-{r['snapshot']}"
    return f"{base}/pretrain/best"


# ---- data per arm ---------------------------------------------------------------------------------------------
def arm_data(group: str, arm: str, df, root: str, log=print):
    """The arm's tokenizer (fit on train rows only) and its train / val windows, cached under ablate_pre/<group>/<arm>/data."""
    d = f"{root}/ablate_pre/{group}/{arm}/data"; os.makedirs(d, exist_ok=True)
    tp = f"{d}/tokenizer.json"
    if all(os.path.exists(f"{d}/{f}") for f in ("tokenizer.json", "tokens_train.npy", "tokens_val.npy")):
        tok = TransactionTokenizer.load(tp)
        return tok, np.load(f"{d}/tokens_train.npy"), np.load(f"{d}/tokens_val.npy")
    t0 = time.time()
    tok = TransactionTokenizer(**arm_tokenizer_config(arm)).fit(train_only(df))
    ids = tok.encode(df)
    card, split = df["card_id"].to_numpy(), df["split"].to_numpy()
    w = {s: make_windows(ids, card, split, s, seq_len=SEQ_LEN, stride=STRIDE, pad_id=tok.vocab["<PAD>"])[0] for s in ("train", "val")}
    tok.save(tp)
    for s, arr in w.items():
        np.save(f"{d}/tokens_{s}.npy", arr)
    log(f"{arm}: tokenizer fit on train, vocab {len(tok.vocab)}, {tok.tokens_per_tx} tokens per transaction, "
        f"{len(w['train']):,} train windows, {len(w['val']):,} val windows, {time.time() - t0:.0f} s")
    return tok, w["train"], w["val"]


# ---- pretraining per arm ---------------------------------------------------------------------------------------
def snapshot_callback(dest: str, tokenizer_json: str, n: int = 5):
    """Saves the weights (no optimizer) at log-spaced steps into dest/snap-<step>, for the scatter."""
    from transformers import TrainerCallback

    class Snapshot(TrainerCallback):
        steps: list[int] = []

        def on_train_begin(self, args, state, control, **kw):
            self.steps = snapshot_steps(state.max_steps, args.save_steps, n)
            print(f"snapshots for the perplexity scatter at steps {self.steps}", flush=True)

        def on_save(self, args, state, control, model=None, **kw):
            if state.global_step in self.steps and model is not None:
                p = f"{dest}/snap-{state.global_step}"
                model.save_pretrained(p); shutil.copy(tokenizer_json, f"{p}/tokenizer.json")
                print(f"snapshot saved: snap-{state.global_step}", flush=True)

    return Snapshot()


def pretrain_arm(group: str, arm: str, df, root: str, results_dir: str, epochs: int = 8, log=print,
                 overrides: dict | None = None, cfg_overrides: dict | None = None) -> dict:
    """Pretrain one arm of one group end to end, resumable; a finished arm is returned from its json without training."""
    import torch
    from transformers import PrinterCallback, Trainer
    from tfm.pretrain import print_progress_callback
    from tfm.runtime import describe

    name = f"06c_{group}_pretrain_{arm}"; base = f"{root}/ablate_pre/{group}/{arm}"; out, res = f"{base}/pretrain", f"{results_dir}/{name}.json"
    snapshots = GROUPS[group]["snapshots"] == arm
    tok, train_w, val_w = arm_data(group, arm, df, root, log)
    if os.path.exists(res):
        log(f"{name}: done earlier, skipped"); return json.load(open(res))
    v = tok.vocab; PAD, BOS, MASK, V = v["<PAD>"], v["<BOS>"], v["<MASK>"], len(v)
    fields = field_of(tok.id2tok)
    obj = ARMS[arm]["objective"]
    if obj == "clm":
        from transformers import GPT2LMHeadModel
        model = GPT2LMHeadModel(gpt2_config(V, PAD, BOS, **(cfg_overrides or {})))
        train_ds, val_ds, shift = WindowDataset(train_w, PAD, BOS), WindowDataset(val_w, PAD, BOS), True
    else:
        from transformers import BertForMaskedLM
        from tfm.mlm import MLMDataset, bert_config
        model = BertForMaskedLM(bert_config(V, PAD, **(cfg_overrides or {})))
        train_ds, val_ds, shift = MLMDataset(train_w, PAD, BOS, MASK, V, seed=0), MLMDataset(val_w, PAD, BOS, MASK, V, seed=1), False
    n_params = sum(p.numel() for p in model.parameters())
    settings = dict(num_train_epochs=epochs, per_device_train_batch_size=32, per_device_eval_batch_size=64,
                    eval_accumulation_steps=8, dataloader_num_workers=2, disable_tqdm=True)       # step 3's values
    settings.update(overrides or {})
    args = training_args(out, seed=0, **settings)
    callbacks = [print_progress_callback()] + ([snapshot_callback(f"{base}/snapshots", f"{base}/data/tokenizer.json")] if snapshots else [])
    trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds, compute_metrics=compute_metrics_fn(fields, shift=shift),
                      preprocess_logits_for_metrics=argmax_logits, callbacks=callbacks)
    trainer.remove_callback(PrinterCallback)
    last = find_latest_checkpoint(out)
    log(f"\n==== {name} ====  {obj.upper()}, {n_params / 1e6:.2f}M parameters, {len(train_ds):,} train windows, resuming from: {last.name if last else 'scratch'}")
    t0 = time.time()
    trainer.train(resume_from_checkpoint=str(last) if last else None)
    problems = {p.name: verify_checkpoint(p) for p in sorted(Path(out).glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))}
    for k, val in problems.items():
        log(f"  {k:18s}", val or "ok")
    trainer.save_model(f"{out}/best"); shutil.copy(f"{base}/data/tokenizer.json", f"{out}/best/tokenizer.json")
    s = summarize(trainer.state.log_history, prior_accuracy(train_w, fields, PAD), n_params, len(train_ds), train_w.shape[1], epochs)
    snaps = sorted(int(p.name.split("-")[1]) for p in Path(f"{base}/snapshots").glob("snap-*")) if snapshots else []
    summary = {"name": name, "group": group, "arm": arm, "objective": obj, "tokenizer_config": tok.config, "vocab_size": V, "tokens_per_tx": tok.tokens_per_tx,
               **s, "snapshots": snaps, "checkpoints_ok": all(not val for val in problems.values()),
               "wall_min": round((time.time() - t0) / 60, 1), "runtime": describe()}
    with open(res, "w") as f:
        json.dump(summary, f, indent=2)
    log(f"{name}: best step {s['best_step']}  val loss {s['best_eval_loss']:.4f}  perplexity {s['best_perplexity']:.2f}  |  {summary['wall_min']} min")
    del trainer, model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()
    return summary

"""Step 6b: the ablations that only need fine-tuning (history length, pooling), plus the gates that make a run on
a second machine trustworthy. Ablations that need a new pretraining run (tokenizer switches, CLM vs MLM, the
perplexity-vs-PR-AUC scatter) are step 6c and wait for the T4.

    RUNS            15 fine-tuning runs: history 0 / 5 / 12 / 32 with last-token pooling, and mean pooling at
                    history 12, three seeds each, all from pretrain/best. The H12-last seed-0 run is first on
                    purpose: on a new machine it is gate G3, compared with the T4 spread from step 4.
    best_checkpoint the checkpoint load_best_model_at_end restored, read from trainer_state.json and remapped
                    onto the local folder (the stored path is the machine that trained it)
    score_test      plain forward passes, no Trainer: used by gate G2 (re-score a T4 checkpoint here and match
                    the saved test_scores.npy) and after every run
    GPT2MeanPool... the pooling ablation: mean over non-pad positions instead of the last token
    finetune_run    one run end to end, resumable, finished runs skipped by their json (the step-4 loop as a function)

Same-machine rule: every arm of one ablation is trained on one kind of machine; each json records runtime.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np

from tfm.checkpointing import find_latest_checkpoint, verify_checkpoint
from tfm.finetune import ClassificationDataset, best_eval, build_examples, compute_metrics_fn, finetune_args, load_classifier
from tfm.metrics import correct_for_sampling, evaluate_scores
from tfm.pretrain import gpt2_config

HISTORIES = (0, 5, 12, 32)
RUNS = ([{"history": 12, "pooling": "last", "init": "pretrained", "seed": 0}]
        + [{"history": h, "pooling": "last", "init": "pretrained", "seed": s} for h in HISTORIES for s in (0, 1, 2) if not (h == 12 and s == 0)]
        + [{"history": 12, "pooling": "mean", "init": "pretrained", "seed": s} for s in (0, 1, 2)])


def run_name(r: dict) -> str:
    return r.get("name") or f"06b_H{r['history']}_{r['pooling']}_seed{r['seed']}"


def run_paths(r: dict, root: str, results_dir: str) -> tuple[str, str]:
    out = f"{root}/{r['out_rel']}" if r.get("out_rel") else f"{root}/ablate/H{r['history']}_{r['pooling']}/seed{r['seed']}"
    return out, f"{results_dir}/{run_name(r)}.json"


def best_checkpoint(out_dir: Path) -> Path | None:
    last = find_latest_checkpoint(out_dir)
    if last is None:
        return None
    state = json.loads((last / "trainer_state.json").read_text())
    best = state.get("best_model_checkpoint")
    if not best:
        return None
    local = Path(out_dir) / Path(best).name
    return local if local.is_dir() else None


# ---- torch side ----------------------------------------------------------------------------------------------
def _mean_pool_class():
    import torch
    from torch import nn
    from transformers import GPT2Model, GPT2PreTrainedModel
    from transformers.modeling_outputs import SequenceClassifierOutput

    class GPT2MeanPoolForSequenceClassification(GPT2PreTrainedModel):
        """Same body as GPT2ForSequenceClassification; the head reads the mean of the non-pad hidden states
        instead of the last real token's. The WF slide says last-token won every time; this is the check."""

        def __init__(self, config):
            super().__init__(config)
            self.num_labels = config.num_labels
            self.transformer = GPT2Model(config)
            self.score = nn.Linear(config.n_embd, config.num_labels, bias=False)
            self.post_init()

        def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
            h = self.transformer(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids)
            m = attention_mask.to(h.dtype).unsqueeze(-1)
            pooled = (h * m).sum(1) / m.sum(1).clamp(min=1.0)
            logits = self.score(pooled)
            loss = nn.functional.cross_entropy(logits, labels) if labels is not None else None
            return SequenceClassifierOutput(loss=loss, logits=logits)

    return GPT2MeanPoolForSequenceClassification


def __getattr__(name):                                   # lazy, so importing tfm.ablate needs no torch
    if name == "GPT2MeanPoolForSequenceClassification":
        return _mean_pool_class()
    raise AttributeError(name)


def load_model(init: str, pooling: str, pretrained_dir: str | None, vocab_size: int, pad_id: int, bos_id: int, arch: str = "gpt2"):
    if arch == "bert":                                       # step 6c, the MLM arm: encoder body, last-token head
        from tfm.mlm import load_bert_classifier
        assert init == "pretrained" and pooling == "last", (init, pooling)
        return load_bert_classifier(pretrained_dir, pad_id)
    if pooling == "last":
        return load_classifier(init, pretrained_dir, vocab_size, pad_id, bos_id)
    if pooling != "mean":
        raise ValueError(pooling)
    cls = _mean_pool_class()
    if init == "pretrained":
        model = cls.from_pretrained(pretrained_dir, num_labels=2)
    elif init == "scratch":
        cfg = gpt2_config(vocab_size, pad_id, bos_id); cfg.num_labels = 2
        model = cls(cfg)
    else:
        raise ValueError(init)
    model.config.pad_token_id = pad_id
    return model


def score_test(model, X: np.ndarray, pad_id: int, device: str, batch_size: int = 256) -> np.ndarray:
    """Fraud probability per row of X (int16 token matrix), plain forward passes in eval mode."""
    import torch

    model = model.to(device).eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            ids = torch.as_tensor(X[i:i + batch_size].astype(np.int64), device=device)
            logits = model(input_ids=ids, attention_mask=(ids != pad_id).long()).logits.float()
            out.append(torch.softmax(logits, dim=-1)[:, 1].cpu().numpy())
    return np.concatenate(out).astype(np.float32)


def finetune_run(r: dict, ids_all, card, split, fraud, pad_id: int, bos_id: int, vocab_size: int, pretrained_dir: str,
                 root: str, results_dir: str, examples: dict, keep: float = 0.05, alert: float = 0.006, log=print,
                 overrides: dict | None = None) -> dict:
    """One ablation run. examples caches (X, y, idx) for val and test per history length. Returns the summary
    (also written to the run's json); a run whose json exists is returned from that json without training."""
    import torch
    from transformers import PrinterCallback, Trainer
    from tfm.pretrain import print_progress_callback
    from tfm.runtime import accelerator, describe

    name = run_name(r); out, res = run_paths(r, root, results_dir)
    if os.path.exists(res):
        log(f"{name}: done earlier, skipped"); return json.load(open(res))
    H, seed = r["history"], r["seed"]
    for s in ("val", "test"):
        if (s, H) not in examples:
            examples[(s, H)] = build_examples(ids_all, card, split, fraud, s, H, pad_id, bos_id)
    Xv, yv, iv = examples[("val", H)]; Xt, yt, it = examples[("test", H)]
    Xtr, ytr, _ = build_examples(ids_all, card, split, fraud, "train", H, pad_id, bos_id, neg_keep_rate=keep, seed=seed)
    log(f"\n==== {name} ====  {len(ytr):,} train examples ({int(ytr.sum()):,} fraud), sequence length {Xtr.shape[1]}, accelerator {accelerator()}")
    model = load_model(r["init"], r["pooling"], pretrained_dir, vocab_size, pad_id, bos_id, arch=r.get("arch", "gpt2"))
    trainer = Trainer(model=model, args=finetune_args(out, seed=seed, **(overrides or {})), train_dataset=ClassificationDataset(Xtr, ytr, pad_id),
                      eval_dataset=ClassificationDataset(Xv, yv, pad_id), compute_metrics=compute_metrics_fn(card[iv], alert),
                      callbacks=[print_progress_callback()])
    trainer.remove_callback(PrinterCallback)
    last = find_latest_checkpoint(out)
    log(f"resuming from: {last.name if last else 'start'}")
    t0 = time.time()
    trainer.train(resume_from_checkpoint=str(last) if last else None)
    problems = {p.name: verify_checkpoint(p) for p in sorted(Path(out).glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))}
    for k, v in problems.items():
        log(f"  {k:18s}", v or "ok")
    best = best_eval(trainer.state.log_history)
    p_test = score_test(trainer.model, Xt, pad_id, str(trainer.model.device))
    np.save(f"{out}/test_scores.npy", p_test)
    test_m = evaluate_scores(yt, p_test, card[it], alert)
    summary = {"name": name, **r, "neg_keep_rate": keep, "alert_rate": alert, "n_train_examples": int(len(ytr)), "n_train_fraud": int(ytr.sum()),
               "seq_len": int(Xtr.shape[1]), "best_step": best["step"], "val_at_best": {k[len("eval_"):]: v for k, v in best.items() if k.startswith("eval_")},
               "test": test_m, "mean_prob_test_corrected": float(correct_for_sampling(p_test, keep).mean()),
               "checkpoints_ok": all(not v for v in problems.values()), "wall_min": round((time.time() - t0) / 60, 1), "runtime": describe()}
    with open(res, "w") as f:
        json.dump(summary, f, indent=2)
    log(f"{name}: best step {best['step']}  val PR-AUC {best['eval_pr_auc']:.4f}  |  test PR-AUC {test_m['pr_auc']:.4f}  "
        f"recall@{alert:.1%} {test_m['recall_at_alert']:.3f}  card discovery {test_m['card_discovery']:.3f}  FP ratio {test_m['fp_ratio']:.2f}  |  {summary['wall_min']} min")
    del trainer, model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()
    return summary

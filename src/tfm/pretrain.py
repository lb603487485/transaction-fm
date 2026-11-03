"""Step 3: pretrain a small GPT-2 decoder with next-token prediction (CLM) on the token windows.

    tokens_train.npy  (n_windows, 512)        from step 2, train rows only
         |  make_example: [<BOS>] + window[:-1]; <PAD> positions get attention 0 and label -100
         v
    GPT2LMHeadModel  6 layers, 384 hidden, 6 heads, 512 positions, about 11M parameters
         |  Hugging Face Trainer: fp16, cosine LR with warmup, eval + save every 500 steps on Drive,
         |  best checkpoint by validation loss (never the last step), resume from the newest complete checkpoint
         v
    results/03_pretrain.json  perplexity = exp(val loss), next-token accuracy per field vs the train prior

What trains on what: the model sees train windows only. Val windows (val rows only, step 2) are used
for the loss that picks the checkpoint and for the per-field accuracy. Test is not read here.

Per-field accuracy: every token has a field (DT, HOUR, CAT, MER, AMT, DIST, special). For each
field, how often the argmax prediction equals the next token. The comparison point is the
"prior": always predicting that field's most common train token. Wealthsimple calls the gap
"next-event accuracy over the prior" and finds it more informative than perplexity.
"""
from __future__ import annotations

import numpy as np

FIELDS = ["special", "DT", "HOUR", "CAT", "MER", "AMT", "DIST"]
IGNORE = -100


def make_example(window: np.ndarray, pad_id: int, bos_id: int) -> dict[str, np.ndarray]:
    """One training example from one step-2 window: shift right by one and start with <BOS>, so the
    first real token is also predicted. The window's last token is dropped to keep the length.
    <PAD> positions (short cards) get attention_mask 0 and label -100, so they add nothing to the loss."""
    ids = np.concatenate([[bos_id], window[:-1]]).astype(np.int64)
    real = ids != pad_id
    return {"input_ids": ids, "attention_mask": real.astype(np.int64), "labels": np.where(real, ids, IGNORE).astype(np.int64)}


def field_of(id2tok: list[str]) -> np.ndarray:
    """Field name per vocabulary id: 'special' for <...> tokens, else the text before the first underscore."""
    return np.array(["special" if t.startswith("<") else t.split("_")[0] for t in id2tok])


def per_field_accuracy(pred_ids: np.ndarray, label_ids: np.ndarray, fields: np.ndarray, shift: bool = True) -> dict[str, float]:
    """CLM (shift=True): pred_ids[:, t] is the model's guess for label_ids[:, t + 1]. MLM (shift=False): pred_ids[:, t]
    guesses label_ids[:, t], the masked token at that position. Positions with label -100 are skipped either way."""
    pred = np.asarray(pred_ids)[:, :-1].ravel() if shift else np.asarray(pred_ids).ravel()
    lab = np.asarray(label_ids)[:, 1:].ravel() if shift else np.asarray(label_ids).ravel()
    keep = lab != IGNORE
    pred, lab = pred[keep], lab[keep]
    hit = pred == lab
    out = {"all": float(hit.mean()) if len(lab) else float("nan")}
    f = fields[lab]
    for name in FIELDS:
        m = f == name
        if m.any():
            out[name] = float(hit[m].mean())
    return out


def prior_accuracy(train_windows: np.ndarray, fields: np.ndarray, pad_id: int) -> dict[str, float]:
    """Accuracy of always guessing each field's most common train token (pads excluded)."""
    ids = np.asarray(train_windows).ravel()
    ids = ids[ids != pad_id]
    counts = np.bincount(ids, minlength=len(fields))
    out, f = {}, fields
    for name in FIELDS:
        m = f == name
        if counts[m].sum():
            out[name] = float(counts[m].max() / counts[m].sum())
    out["all"] = float(sum(counts[f == n].max() for n in out) / counts.sum())
    return out


def summarize(log_history: list[dict], prior: dict[str, float], n_params: int, n_train_windows: int,
              seq_len: int, epochs: float) -> dict:
    """The numbers for results/03_pretrain.json, read from trainer.state.log_history."""
    evals = [e for e in log_history if "eval_loss" in e]
    best = min(evals, key=lambda e: e["eval_loss"])
    return {
        "best_step": best["step"],
        "best_eval_loss": best["eval_loss"],
        "best_perplexity": float(np.exp(best["eval_loss"])),
        "last_eval_loss": evals[-1]["eval_loss"],
        "per_field_accuracy_at_best": {k[len("eval_acc_"):]: v for k, v in best.items() if k.startswith("eval_acc_")},
        "prior_accuracy": prior,
        "eval_curve": [[e["step"], e["eval_loss"]] for e in evals],
        "train_curve": [[e["step"], e["loss"]] for e in log_history if "loss" in e and "eval_loss" not in e],
        "n_params": n_params,
        "n_train_windows": n_train_windows,
        "seq_len": seq_len,
        "train_tokens_per_epoch": n_train_windows * seq_len,
        "epochs": epochs,
    }


# ---- torch / transformers side (imported lazily so the tests above run on CPU without them) -------------
def gpt2_config(vocab_size: int, pad_id: int, bos_id: int, n_positions: int = 512,
                n_layer: int = 6, n_embd: int = 384, n_head: int = 6):
    """The guide's row of the WF-vs-this-project table: 6 layers, 384 hidden, 6 heads, 512 positions,
    GPT-2's learned absolute positions. <BOS> doubles as eos id because HF requires one; it is never generated."""
    from transformers import GPT2Config

    return GPT2Config(vocab_size=vocab_size, n_positions=n_positions, n_ctx=n_positions, n_embd=n_embd,
                      n_layer=n_layer, n_head=n_head, pad_token_id=pad_id, bos_token_id=bos_id, eos_token_id=bos_id)


class WindowDataset:
    """Map-style dataset over a (n_windows, seq_len) int array; each item is make_example as torch tensors."""

    def __init__(self, windows: np.ndarray, pad_id: int, bos_id: int):
        self.windows, self.pad_id, self.bos_id = windows, pad_id, bos_id

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, i: int) -> dict:
        import torch

        return {k: torch.as_tensor(v) for k, v in make_example(self.windows[i], self.pad_id, self.bos_id).items()}


def argmax_logits(logits, labels):
    """preprocess_logits_for_metrics: keep only the argmax so eval does not hold n x 512 x vocab floats."""
    return logits.argmax(dim=-1)


def compute_metrics_fn(fields: np.ndarray, shift: bool = True):
    """compute_metrics for the Trainer: per-field next-token (or masked-token) accuracy, keys eval_acc_<field>."""

    def compute_metrics(eval_pred):
        pred, lab = eval_pred.predictions, eval_pred.label_ids
        return {f"acc_{k}": v for k, v in per_field_accuracy(pred, lab, fields, shift=shift).items()}

    return compute_metrics


# ---- plain-text progress (the HF widget is lost on reconnect and does not render on GitHub) ----------------
def progress_line(step: int, max_steps: int, start_step: int, elapsed_s: float, loss: float, lr: float) -> str:
    done = step - start_step
    eta_s = elapsed_s / done * (max_steps - step) if done > 0 else float("nan")
    return (f"step {step:,}/{max_steps:,}  {100 * step / max_steps:.1f}%  loss {loss:.3f}  lr {lr:.1e}  "
            f"elapsed {elapsed_s / 60:.1f} min  eta {eta_s / 60:.0f} min")


def eval_line(step: int, logs: dict) -> str:
    accs = "  ".join(f"{k[len('eval_acc_'):]} {v:.2f}" for k, v in logs.items() if k.startswith("eval_acc_"))
    return f"eval @ {step:,}  val loss {logs['eval_loss']:.4f}  ppl {float(np.exp(logs['eval_loss'])):.2f}  acc: {accs}"


def print_progress_callback():
    """A TrainerCallback that prints one line per logging step, per eval and per checkpoint save."""
    import time

    from transformers import TrainerCallback

    class PrintProgress(TrainerCallback):
        def on_train_begin(self, args, state, control, **kw):
            self.t0, self.start_step = time.time(), state.global_step
            print(f"training: {state.max_steps:,} steps total, starting at step {state.global_step:,}, "
                  f"eval and save every {args.save_steps} steps", flush=True)

        def on_log(self, args, state, control, logs=None, **kw):
            if logs and "loss" in logs:
                print(progress_line(state.global_step, state.max_steps, self.start_step, time.time() - self.t0,
                                    logs["loss"], logs.get("learning_rate", float("nan"))), flush=True)
            if logs and "eval_loss" in logs:
                print(eval_line(state.global_step, logs), flush=True)

        def on_save(self, args, state, control, **kw):
            print(f"saved checkpoint-{state.global_step:,} to Drive", flush=True)

    return PrintProgress()

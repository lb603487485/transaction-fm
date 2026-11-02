"""Step 4: fine-tune the pretrained decoder to classify one transaction as fraud or not.

One example, exactly the Wells Fargo slide: the card's previous H transactions, then the transaction
to judge, right-padded; the label is that last transaction's is_fraud; GPT2ForSequenceClassification
reads the representation of the last non-pad token (last-token pooling).

    <BOS> | tx[i-H] ... tx[i-1] | tx[i] | <PAD> ...        length 1 + (H + 1) * tokens_per_tx
                history (fewer if the card is young)   ^ the row being judged; its last token is pooled

What trains on what
    train examples   target rows are train rows; all fraud kept, normals sampled to neg_keep_rate (5%)
    val / test       target rows are their own rows, NOT sampled, so metrics see the true 0.5% fraud rate
    history          may reach back into an earlier split (a test row sees its card's train-period rows):
                     that is the allowed direction, the same as a production scorer
    labels           never enter the tokens; the history rows' is_fraud is not used anywhere

Imbalance is handled by the sampling alone (no class weight, the guide says one or the other).
Probabilities from a model trained on sampled normals are inflated; metrics.correct_for_sampling undoes it
when calibration is reported. Ranking metrics (PR-AUC, recall at alert rate) are unaffected.
"""
from __future__ import annotations

import numpy as np

from tfm.pretrain import gpt2_config


def build_examples(ids: np.ndarray, card_id: np.ndarray, split: np.ndarray, is_fraud: np.ndarray,
                   target_split: str, history: int, pad_id: int, bos_id: int,
                   neg_keep_rate: float | None = None, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ids: (n_rows, tokens_per_tx) from the tokenizer, rows in card then time order. Returns
    X (m, 1 + (history + 1) * tokens_per_tx) int16, y (m,) int8, idx (m,) the row index of each example."""
    ids, card_id, split, is_fraud = map(np.asarray, (ids, card_id, split, is_fraud))
    n, t = ids.shape
    starts = np.flatnonzero(np.r_[True, card_id[1:] != card_id[:-1]])          # first row of each card
    card_start = np.repeat(starts, np.diff(np.r_[starts, n]))
    pos = np.arange(n) - card_start                                              # rows before this one on the same card

    sel = np.flatnonzero(split == target_split)
    if neg_keep_rate is not None:
        rng = np.random.default_rng(seed)
        sel = sel[(is_fraud[sel] == 1) | (rng.random(len(sel)) < neg_keep_rate)]

    h = np.minimum(pos[sel], history)                                            # history rows actually available
    slot = np.arange(history + 1)[None, :]
    valid = slot <= h[:, None]                                                   # slots 0..h hold rows, the rest pad
    src = np.where(valid, sel[:, None] - h[:, None] + slot, 0)
    seq = np.where(valid[:, :, None], ids[src], pad_id).reshape(len(sel), -1)
    X = np.concatenate([np.full((len(sel), 1), bos_id), seq], axis=1).astype(np.int16)
    return X, is_fraud[sel].astype(np.int8), sel


def best_eval(log_history: list[dict], key: str = "eval_pr_auc") -> dict:
    """The eval log entry with the highest key (the checkpoint load_best_model_at_end restored)."""
    return max((e for e in log_history if key in e), key=lambda e: e[key])


# ---- torch / transformers side -------------------------------------------------------------------------
class ClassificationDataset:
    def __init__(self, X: np.ndarray, y: np.ndarray, pad_id: int):
        self.X, self.y, self.pad_id = X, y, pad_id

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, i: int) -> dict:
        import torch

        ids = torch.as_tensor(self.X[i].astype(np.int64))
        return {"input_ids": ids, "attention_mask": (ids != self.pad_id).long(), "labels": torch.as_tensor(int(self.y[i]))}


def load_classifier(init: str, pretrained_dir: str | None, vocab_size: int, pad_id: int, bos_id: int):
    """init='pretrained': body from pretrain/best, new 2-class head. init='scratch': same config, random weights.
    pad_token_id tells GPT2ForSequenceClassification which token is the last real one (last-token pooling)."""
    from transformers import GPT2ForSequenceClassification

    if init == "pretrained":
        model = GPT2ForSequenceClassification.from_pretrained(pretrained_dir, num_labels=2)
    elif init == "scratch":
        cfg = gpt2_config(vocab_size, pad_id, bos_id); cfg.num_labels = 2
        model = GPT2ForSequenceClassification(cfg)
    else:
        raise ValueError(init)
    model.config.pad_token_id = pad_id
    return model


def finetune_args(out_dir: str, seed: int, **overrides):
    """The guide's fine-tuning settings on top of the resume-safe pretraining ones: lr one order of magnitude
    below pretraining, 3 epochs, checkpoint chosen by validation PR-AUC."""
    from tfm.checkpointing import training_args

    args = dict(learning_rate=5e-5, num_train_epochs=3, metric_for_best_model="eval_pr_auc", greater_is_better=True,
                per_device_train_batch_size=64, per_device_eval_batch_size=256, disable_tqdm=True)
    args.update(overrides)
    return training_args(out_dir, seed=seed, **args)


def softmax_fraud_prob(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e[:, 1] / e.sum(axis=1)


def compute_metrics_fn(card_id_eval: np.ndarray, alert_rate: float = 0.006):
    """compute_metrics for the Trainer; the eval loader keeps order, so card ids line up with the predictions."""
    from tfm.metrics import evaluate_scores

    def compute_metrics(eval_pred):
        p = softmax_fraud_prob(np.asarray(eval_pred.predictions))
        return evaluate_scores(eval_pred.label_ids, p, card_id_eval, alert_rate)

    return compute_metrics

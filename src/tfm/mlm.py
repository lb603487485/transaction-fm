"""Step 6c, the objective ablation: an encoder trained with masked-token prediction (MLM) next to the
step-3 decoder trained with next-token prediction (CLM). Wells Fargo found the decoder better for
downstream fraud; Wealthsimple argues MLM suits event-level anomaly detection. Same vocabulary, same
window, same size (6 layers, 384 hidden, 6 heads), same fine-tuning; only the objective and the
attention direction change.

    make_mlm_example   the step-3 example (<BOS> + window) with 15% of the real, non-<BOS> tokens chosen as
                       targets: 80% become <MASK>, 10% a random non-special token, 10% stay (BERT's recipe);
                       labels are -100 everywhere else, so the loss and the per-field accuracy see targets only
    MLMDataset         one fixed mask per window (seeded by the window index): static masking, as original BERT
    bert_config        BertConfig mirroring gpt2_config
    BertLastToken...   fine-tuning head reading the last real token, the same pooling as the decoder arm, so the
                       comparison changes the objective and not the pooling
"""
from __future__ import annotations

import numpy as np

from tfm.pretrain import IGNORE
from tfm.tokenizer import SPECIAL


def make_mlm_example(window: np.ndarray, pad_id: int, bos_id: int, mask_id: int, vocab_size: int,
                     rng: np.random.Generator, p: float = 0.15) -> dict[str, np.ndarray]:
    ids = np.concatenate([[bos_id], window[:-1]]).astype(np.int64)
    real = ids != pad_id
    cand = real.copy(); cand[0] = False                              # <BOS> is never a target
    target = cand & (rng.random(len(ids)) < p)
    u = rng.random(len(ids))
    inp = ids.copy()
    inp[target & (u < 0.8)] = mask_id
    rand = target & (u >= 0.8) & (u < 0.9)
    inp[rand] = rng.integers(len(SPECIAL), vocab_size, size=int(rand.sum()))
    return {"input_ids": inp, "attention_mask": real.astype(np.int64), "labels": np.where(target, ids, IGNORE).astype(np.int64)}


class MLMDataset:
    def __init__(self, windows: np.ndarray, pad_id: int, bos_id: int, mask_id: int, vocab_size: int, seed: int = 0):
        self.windows, self.pad_id, self.bos_id, self.mask_id, self.vocab_size, self.seed = windows, pad_id, bos_id, mask_id, vocab_size, seed

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, i: int) -> dict:
        import torch

        ex = make_mlm_example(self.windows[i], self.pad_id, self.bos_id, self.mask_id, self.vocab_size, np.random.default_rng([self.seed, i]))
        return {k: torch.as_tensor(v) for k, v in ex.items()}


def bert_config(vocab_size: int, pad_id: int, n_positions: int = 512, n_layer: int = 6, n_embd: int = 384, n_head: int = 6):
    from transformers import BertConfig

    return BertConfig(vocab_size=vocab_size, hidden_size=n_embd, num_hidden_layers=n_layer, num_attention_heads=n_head,
                      intermediate_size=4 * n_embd, max_position_embeddings=n_positions, pad_token_id=pad_id, type_vocab_size=1)


def _last_token_class():
    import torch
    from torch import nn
    from transformers import BertModel, BertPreTrainedModel
    from transformers.modeling_outputs import SequenceClassifierOutput

    class BertLastTokenForSequenceClassification(BertPreTrainedModel):
        """BERT body from the MLM checkpoint, head on the last non-pad position (the transaction being judged)."""

        def __init__(self, config):
            super().__init__(config)
            self.num_labels = config.num_labels
            self.bert = BertModel(config, add_pooling_layer=False)
            self.score = nn.Linear(config.hidden_size, config.num_labels, bias=False)
            self.post_init()

        def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids)
            h = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
            last = (attention_mask.sum(1) - 1).clamp(min=0)
            logits = self.score(h[torch.arange(h.shape[0], device=h.device), last])
            loss = nn.functional.cross_entropy(logits, labels) if labels is not None else None
            return SequenceClassifierOutput(loss=loss, logits=logits)

    return BertLastTokenForSequenceClassification


def __getattr__(name):                                   # lazy, so importing tfm.mlm needs no torch
    if name == "BertLastTokenForSequenceClassification":
        return _last_token_class()
    raise AttributeError(name)


def load_bert_classifier(pretrained_dir: str, pad_id: int):
    model = _last_token_class().from_pretrained(pretrained_dir, num_labels=2)
    model.config.pad_token_id = pad_id
    return model

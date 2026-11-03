import numpy as np
import pytest

from tfm.mlm import make_mlm_example
from tfm.pretrain import IGNORE, per_field_accuracy


def test_mlm_example_masks_real_non_bos_positions_only_and_labels_them():
    rng = np.random.default_rng(0)
    window = rng.integers(5, 40, size=2000).astype(np.int32); window[-300:] = 0      # 300 pads at the end
    ex = make_mlm_example(window, pad_id=0, bos_id=1, mask_id=2, vocab_size=40, rng=np.random.default_rng(1))
    ids = np.concatenate([[1], window[:-1]])
    assert ex["input_ids"].shape == ex["labels"].shape == ex["attention_mask"].shape == (2000,)
    assert ex["input_ids"][0] == 1 and ex["labels"][0] == IGNORE                       # <BOS> kept, never a target
    labeled = ex["labels"] != IGNORE
    assert (ex["attention_mask"] == (ids != 0)).all()
    assert not labeled[ids == 0].any()                                               # pads never targets
    assert (ex["labels"][labeled] == ids[labeled]).all()                             # label is the original token
    changed = ex["input_ids"] != ids
    assert changed[labeled].mean() > 0.8 and not changed[~labeled].any()             # 80% mask + 10% random; untouched elsewhere
    assert ((ex["input_ids"] == 2) <= labeled).all()                                 # <MASK> only at targets
    share = labeled.sum() / (ids != 0).sum()
    assert 0.12 < share < 0.18


def test_mlm_example_is_deterministic_for_a_seed():
    window = np.random.default_rng(3).integers(5, 40, size=500).astype(np.int32)
    a = make_mlm_example(window, 0, 1, 2, 40, np.random.default_rng(7))
    b = make_mlm_example(window, 0, 1, 2, 40, np.random.default_rng(7))
    assert (a["input_ids"] == b["input_ids"]).all() and (a["labels"] == b["labels"]).all()


def test_per_field_accuracy_unshifted_scores_masked_positions_in_place():
    fields = np.array(["special", "special", "AMT", "AMT", "CAT"])
    labels = np.array([[IGNORE, 2, IGNORE, 4], [3, IGNORE, IGNORE, IGNORE]])
    pred = np.array([[9, 2, 9, 3], [3, 9, 9, 9]])
    acc = per_field_accuracy(pred, labels, fields, shift=False)
    assert acc["all"] == pytest.approx(2 / 3) and acc["AMT"] == 1.0 and acc["CAT"] == 0.0 and "special" not in acc
    assert per_field_accuracy(pred, labels, fields, shift=True)["all"] != acc["all"]          # the shifted reading is a different comparison


@pytest.fixture
def tiny_bert():
    torch = pytest.importorskip("torch")
    from tfm.mlm import bert_config
    cfg = bert_config(vocab_size=20, pad_id=0, n_positions=16, n_layer=1, n_embd=8, n_head=2)
    return torch, cfg


def test_bert_config_mirrors_the_gpt2_sizes(tiny_bert):
    _, cfg = tiny_bert
    assert (cfg.vocab_size, cfg.hidden_size, cfg.num_hidden_layers, cfg.num_attention_heads, cfg.max_position_embeddings, cfg.pad_token_id) == (20, 8, 1, 2, 16, 0)
    assert cfg.intermediate_size == 4 * 8


def test_bert_last_token_classifier_pools_the_last_real_token(tiny_bert):
    torch, cfg = tiny_bert
    from tfm.mlm import BertLastTokenForSequenceClassification
    cfg.num_labels = 2
    torch.manual_seed(0)
    m = BertLastTokenForSequenceClassification(cfg).eval()
    ids = torch.tensor([[1, 5, 6, 7, 0, 0]]); mask = (ids != 0).long()
    other = ids.clone(); other[0, 4:] = 9                                            # pads replaced, mask unchanged
    last = ids.clone(); last[0, 3] = 8                                               # last real token changed
    with torch.no_grad():
        a = m(input_ids=ids, attention_mask=mask).logits
        b = m(input_ids=other, attention_mask=mask).logits
        c = m(input_ids=last, attention_mask=mask).logits
    assert a.shape == (1, 2) and torch.allclose(a, b, atol=1e-6) and not torch.allclose(a, c, atol=1e-4)
    assert m(input_ids=ids, attention_mask=mask, labels=torch.tensor([1])).loss.item() > 0


def test_bert_classifier_loads_the_masked_lm_body(tiny_bert, tmp_path):
    torch, cfg = tiny_bert
    from transformers import BertForMaskedLM
    from tfm.mlm import load_bert_classifier
    torch.manual_seed(1)
    lm = BertForMaskedLM(cfg); lm.save_pretrained(tmp_path / "best")
    m = load_bert_classifier(str(tmp_path / "best"), pad_id=0)
    assert torch.equal(m.bert.encoder.layer[0].attention.self.query.weight, lm.bert.encoder.layer[0].attention.self.query.weight)
    assert m.config.num_labels == 2 and m.config.pad_token_id == 0

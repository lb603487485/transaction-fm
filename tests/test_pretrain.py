import numpy as np
import pytest

from tfm.pretrain import field_of, make_example, per_field_accuracy, prior_accuracy, summarize

PAD, BOS = 0, 1
ID2TOK = ["<PAD>", "<BOS>", "<MASK>", "<UNK>", "<TX>", "DT_NA", "DT_00", "HOUR_MORNING", "CAT_a", "CAT_b", "MER_x", "MER_OTHER", "AMT_00", "AMT_01", "DIST_00"]


def test_make_example_prepends_bos_masks_pads_and_ignores_them_in_labels():
    window = np.array([4, 5, 7, 8, 10, 12, 14, 0, 0, 0], dtype=np.int32)   # one transaction then 3 pads
    ex = make_example(window, pad_id=PAD, bos_id=BOS)
    assert ex["input_ids"].tolist() == [1, 4, 5, 7, 8, 10, 12, 14, 0, 0]     # BOS first, last token dropped, same length
    assert ex["attention_mask"].tolist() == [1, 1, 1, 1, 1, 1, 1, 1, 0, 0]
    assert ex["labels"].tolist() == [1, 4, 5, 7, 8, 10, 12, 14, -100, -100]
    assert ex["input_ids"].dtype == np.int64 and ex["labels"].dtype == np.int64


def test_field_of_uses_token_prefix():
    f = field_of(ID2TOK)
    assert f.tolist()[:5] == ["special"] * 5
    assert f[5] == "DT" and f[7] == "HOUR" and f[9] == "CAT" and f[11] == "MER" and f[13] == "AMT" and f[14] == "DIST"


def test_per_field_accuracy_shifts_by_one_and_skips_ignored():
    fields = field_of(ID2TOK)
    # labels: position t is predicted from position t-1, so pred[:, :-1] is compared with label[:, 1:]
    labels = np.array([[1, 4, 5, 7, 8, 10, 12, 14, -100, -100]])
    preds = np.array([[4, 5, 7, 9, 10, 13, 14, 0, 0, 0]])          # pred at t predicts label at t+1
    #            label[1:]  = 4  5  7  8  10 12 14 -100 -100
    #            pred[:-1]  = 4  5  7  9  10 13 14   0    0      -> CAT wrong (9 vs 8), AMT wrong (13 vs 12)
    acc = per_field_accuracy(preds, labels, fields)
    assert acc["special"] == 1.0 and acc["DT"] == 1.0 and acc["HOUR"] == 1.0 and acc["MER"] == 1.0 and acc["DIST"] == 1.0
    assert acc["CAT"] == 0.0 and acc["AMT"] == 0.0
    assert acc["all"] == pytest.approx(5 / 7)


def test_prior_accuracy_is_majority_share_per_field():
    fields = field_of(ID2TOK)
    train = np.array([[4, 8, 12, 0], [4, 8, 13, 0], [4, 9, 13, 0], [4, 8, 13, 0]])   # pads excluded
    prior = prior_accuracy(train, fields, pad_id=PAD)
    assert prior["CAT"] == pytest.approx(3 / 4) and prior["AMT"] == pytest.approx(3 / 4)
    assert prior["special"] == 1.0 and "PAD" not in prior


def test_summarize_picks_lowest_eval_loss_not_last():
    log = [
        {"step": 50, "loss": 2.0},
        {"step": 500, "eval_loss": 1.0, "eval_acc_CAT": 0.5, "eval_acc_all": 0.4},
        {"step": 1000, "eval_loss": 0.5, "eval_acc_CAT": 0.7, "eval_acc_all": 0.6},
        {"step": 1500, "eval_loss": 0.6, "eval_acc_CAT": 0.65, "eval_acc_all": 0.55},
    ]
    s = summarize(log, prior={"CAT": 0.3, "all": 0.2}, n_params=11_000_000, n_train_windows=100, seq_len=512, epochs=8)
    assert s["best_step"] == 1000 and s["best_eval_loss"] == 0.5
    assert s["best_perplexity"] == pytest.approx(np.exp(0.5))
    assert s["last_eval_loss"] == 0.6
    assert s["per_field_accuracy_at_best"] == {"CAT": 0.7, "all": 0.6}
    assert s["prior_accuracy"] == {"CAT": 0.3, "all": 0.2}
    assert s["eval_curve"] == [[500, 1.0], [1000, 0.5], [1500, 0.6]] and s["train_curve"] == [[50, 2.0]]
    assert s["n_params"] == 11_000_000 and s["n_train_windows"] == 100 and s["train_tokens_per_epoch"] == 100 * 512


def test_gpt2_config_matches_guide():
    transformers = pytest.importorskip("transformers")
    from tfm.pretrain import gpt2_config
    cfg = gpt2_config(vocab_size=567, pad_id=PAD, bos_id=BOS)
    assert (cfg.n_layer, cfg.n_embd, cfg.n_head, cfg.n_positions, cfg.vocab_size) == (6, 384, 6, 512, 567)
    assert cfg.pad_token_id == PAD and cfg.bos_token_id == BOS and cfg.eos_token_id == BOS
    model = transformers.GPT2LMHeadModel(cfg)
    assert 10_000_000 < sum(p.numel() for p in model.parameters()) < 12_000_000


def test_progress_line_shows_step_share_loss_and_eta():
    from tfm.pretrain import progress_line
    line = progress_line(step=1000, max_steps=8000, start_step=0, elapsed_s=600, loss=2.1345, lr=4.9e-4)
    assert "1,000/8,000" in line and "12.5%" in line and "loss 2.134" in line and "lr 4.9e-04" in line
    assert "elapsed 10.0 min" in line and "eta 70 min" in line        # 600 s for 1000 steps -> 7000 steps left = 4200 s


def test_progress_line_eta_counts_only_this_session_after_resume():
    from tfm.pretrain import progress_line
    line = progress_line(step=4500, max_steps=8000, start_step=4000, elapsed_s=300, loss=1.0, lr=1e-4)
    assert "eta 35 min" in line                                        # 300 s for 500 steps -> 3500 left = 2100 s


def test_eval_line_shows_perplexity_and_per_field_accuracy():
    from tfm.pretrain import eval_line
    logs = {"eval_loss": 1.0, "eval_acc_all": 0.61, "eval_acc_CAT": 0.55, "eval_acc_MER": 0.3, "eval_runtime": 12.0}
    line = eval_line(step=500, logs=logs)
    assert line.startswith("eval @ 500") and "val loss 1.0000" in line and "ppl 2.72" in line
    assert "all 0.61" in line and "CAT 0.55" in line and "MER 0.30" in line and "runtime" not in line

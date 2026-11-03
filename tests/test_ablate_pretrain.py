from tfm.ablate import run_name, run_paths
from tfm.ablate_pretrain import ARMS, arm_tokenizer_config, finetune_spec, pretrained_dir, snapshot_steps


def test_arms_change_one_thing_each_against_the_step2_tokenizer():
    assert list(ARMS) == ["A_ref", "B_linear", "C_no_dt", "D_mlm"]
    base = arm_tokenizer_config("A_ref")
    assert base == {"use_dt": True, "use_dist": True, "amt_binning": "log", "n_amt_bins": 20, "n_dist_bins": 8, "top_merchants": 500}
    assert arm_tokenizer_config("B_linear") == {**base, "amt_binning": "linear"}
    assert arm_tokenizer_config("C_no_dt") == {**base, "use_dt": False}
    assert arm_tokenizer_config("D_mlm") == base
    assert [ARMS[a]["objective"] for a in ARMS] == ["clm", "clm", "clm", "mlm"]
    assert [ARMS[a]["snapshots"] for a in ARMS] == [True, False, False, False]


def test_snapshot_steps_are_log_spaced_multiples_of_the_save_interval():
    s = snapshot_steps(max_steps=7968, save_steps=500, n=5)
    assert s == [500, 1000, 2000, 4000, 7500]
    assert snapshot_steps(1200, 500, 5) == [500, 1000]                 # too short a run: no duplicates, no step past the end
    assert snapshot_steps(400, 500, 5) == []


def test_finetune_spec_names_and_paths_and_the_snapshot_variant():
    r = finetune_spec("B_linear", seed=2)
    assert r["history"] == 12 and r["pooling"] == "last" and r["init"] == "pretrained" and r["arch"] == "gpt2" and r["snapshot"] is None
    assert run_name(r) == "06c_B_linear_seed2"
    assert run_paths(r, "/r", "/r/results") == ("/r/ablate_pre/B_linear/finetune/seed2", "/r/results/06c_B_linear_seed2.json")
    assert pretrained_dir(r, "/r") == "/r/ablate_pre/B_linear/pretrain/best"
    d = finetune_spec("D_mlm", seed=0)
    assert d["arch"] == "bert"
    s = finetune_spec("A_ref", seed=0, snapshot=2000)
    assert run_name(s) == "06c_A_ref_snap2000_seed0"
    assert run_paths(s, "/r", "/r/results")[0] == "/r/ablate_pre/A_ref/finetune_snap2000/seed0"
    assert pretrained_dir(s, "/r") == "/r/ablate_pre/A_ref/snapshots/snap-2000"

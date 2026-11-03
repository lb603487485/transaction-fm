from tfm.ablate import run_name, run_paths
from tfm.ablate_pretrain import ARMS, GROUPS, arm_tokenizer_config, finetune_spec, pretrained_dir, snapshot_steps


def test_arms_change_one_thing_each_against_the_step2_tokenizer():
    assert list(ARMS) == ["A_ref", "B_linear", "C_no_dt", "D_mlm"]
    base = arm_tokenizer_config("A_ref")
    assert base == {"use_dt": True, "use_dist": True, "amt_binning": "log", "n_amt_bins": 20, "n_dist_bins": 8, "top_merchants": 500}
    assert arm_tokenizer_config("B_linear") == {**base, "amt_binning": "linear"}
    assert arm_tokenizer_config("C_no_dt") == {**base, "use_dt": False}
    assert arm_tokenizer_config("D_mlm") == base
    assert [ARMS[a]["objective"] for a in ARMS] == ["clm", "clm", "clm", "mlm"]


def test_groups_are_self_contained_and_each_carries_its_own_control():
    assert list(GROUPS) == ["tokenizer", "objective"]
    assert GROUPS["tokenizer"] == {"arms": ["A_ref", "B_linear", "C_no_dt"], "snapshots": None}
    assert GROUPS["objective"] == {"arms": ["A_ref", "D_mlm"], "snapshots": "A_ref"}


def test_snapshot_steps_are_log_spaced_multiples_of_the_save_interval():
    s = snapshot_steps(max_steps=7968, save_steps=500, n=5)
    assert s == [500, 1000, 2000, 4000, 7500]
    assert snapshot_steps(1200, 500, 5) == [500, 1000]                 # too short a run: no duplicates, no step past the end
    assert snapshot_steps(400, 500, 5) == []


def test_finetune_spec_names_and_paths_carry_the_group_so_two_machines_never_collide():
    r = finetune_spec("tokenizer", "B_linear", seed=2)
    assert r["history"] == 12 and r["pooling"] == "last" and r["init"] == "pretrained" and r["arch"] == "gpt2" and r["snapshot"] is None
    assert run_name(r) == "06c_tokenizer_B_linear_seed2"
    assert run_paths(r, "/r", "/r/results") == ("/r/ablate_pre/tokenizer/B_linear/finetune/seed2", "/r/results/06c_tokenizer_B_linear_seed2.json")
    assert pretrained_dir(r, "/r") == "/r/ablate_pre/tokenizer/B_linear/pretrain/best"
    assert finetune_spec("objective", "D_mlm", seed=0)["arch"] == "bert"
    s = finetune_spec("objective", "A_ref", seed=0, snapshot=2000)
    assert run_name(s) == "06c_objective_A_ref_snap2000_seed0"
    assert run_paths(s, "/r", "/r/results")[0] == "/r/ablate_pre/objective/A_ref/finetune_snap2000/seed0"
    assert pretrained_dir(s, "/r") == "/r/ablate_pre/objective/A_ref/snapshots/snap-2000"
    a_tok, a_obj = finetune_spec("tokenizer", "A_ref", 0), finetune_spec("objective", "A_ref", 0)
    assert run_name(a_tok) != run_name(a_obj) and run_paths(a_tok, "/r", "/r/results")[0] != run_paths(a_obj, "/r", "/r/results")[0]

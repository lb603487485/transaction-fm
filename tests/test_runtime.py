import pytest

from tfm import runtime


def test_root_is_the_drive_path_on_colab_and_tfm_root_on_a_laptop(monkeypatch, tmp_path):
    monkeypatch.setattr(runtime, "on_colab", lambda: True)
    assert runtime.root() == "/content/drive/MyDrive/Colab Notebooks/tfm"
    monkeypatch.setattr(runtime, "on_colab", lambda: False)
    monkeypatch.setenv("TFM_ROOT", str(tmp_path))
    assert runtime.root() == str(tmp_path)


def test_root_on_a_laptop_without_tfm_root_says_what_to_set(monkeypatch):
    monkeypatch.setattr(runtime, "on_colab", lambda: False)
    monkeypatch.delenv("TFM_ROOT", raising=False)
    with pytest.raises(RuntimeError, match="TFM_ROOT"):
        runtime.root()


def test_root_on_a_laptop_must_exist(monkeypatch, tmp_path):
    monkeypatch.setattr(runtime, "on_colab", lambda: False)
    monkeypatch.setenv("TFM_ROOT", str(tmp_path / "missing"))
    with pytest.raises(RuntimeError, match="missing"):
        runtime.root()


def test_devices_follow_gpu_availability(monkeypatch):
    monkeypatch.setattr(runtime, "gpu_name", lambda: "Tesla T4")
    assert runtime.gpu() is True and runtime.xgb_device() == "cuda" and runtime.fp16() is True
    monkeypatch.setattr(runtime, "gpu_name", lambda: None)
    assert runtime.gpu() is False and runtime.xgb_device() == "cpu" and runtime.fp16() is False


def test_describe_is_a_small_json_friendly_record(monkeypatch, tmp_path):
    monkeypatch.setattr(runtime, "on_colab", lambda: False)
    monkeypatch.setenv("TFM_ROOT", str(tmp_path))
    monkeypatch.setattr(runtime, "gpu_name", lambda: None)
    d = runtime.describe()
    assert d["where"] == "laptop" and d["gpu"] is None and d["device"] == "cpu" and d["fp16"] is False
    assert isinstance(d["machine"], str) and d["machine"]


def test_training_args_fp16_follows_gpu_availability(monkeypatch):
    pytest.importorskip("transformers")
    from tfm.checkpointing import training_args
    monkeypatch.setattr(runtime, "gpu_name", lambda: None)
    assert training_args("/tmp/x").fp16 is False
    monkeypatch.setattr(runtime, "gpu_name", lambda: "Tesla T4")
    assert training_args("/tmp/x").fp16 is True


def test_accelerator_prefers_cuda_then_mps_then_cpu(monkeypatch):
    monkeypatch.setattr(runtime, "gpu_name", lambda: "Tesla T4")
    monkeypatch.setattr(runtime, "mps_available", lambda: True)
    assert runtime.accelerator() == "cuda"
    monkeypatch.setattr(runtime, "gpu_name", lambda: None)
    assert runtime.accelerator() == "mps" and runtime.fp16() is False and runtime.xgb_device() == "cpu"
    monkeypatch.setattr(runtime, "mps_available", lambda: False)
    assert runtime.accelerator() == "cpu"
    monkeypatch.setattr(runtime, "on_colab", lambda: False)
    assert runtime.describe()["accelerator"] == "cpu"

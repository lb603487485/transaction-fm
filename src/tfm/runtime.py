"""Where am I running: Colab with Drive and a T4, or a laptop with a local folder and no CUDA.

    on_colab()   google.colab importable
    root()       Colab: MyDrive/Colab Notebooks/tfm   laptop: $TFM_ROOT (same sub-folders: data/, results/, pretrain/, ...)
    gpu()        nvidia-smi answers, so CUDA is there (torch-free, so step 5 needs no torch)
    xgb_device() "cuda" | "cpu"  for XGBoost
    fp16()       mixed precision only on CUDA; the Trainer refuses it on CPU and Apple MPS
    describe()   the small record every run json stores, so results from different machines can be told apart

Switching machine never touches saved weights: checkpoints are fp32 safetensors; fp16 is a compute setting
of the run that made them, recorded in that run's json.
"""
from __future__ import annotations

import os
import platform
import shutil
import subprocess

COLAB_ROOT = "/content/drive/MyDrive/Colab Notebooks/tfm"


def on_colab() -> bool:
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False


def root() -> str:
    if on_colab():
        return COLAB_ROOT
    r = os.environ.get("TFM_ROOT")
    if not r:
        raise RuntimeError("not on Colab: set TFM_ROOT to the local folder that mirrors Drive's tfm/ (data/processed/..., results/)")
    r = os.path.expanduser(r)
    if not os.path.isdir(r):
        raise RuntimeError(f"TFM_ROOT={r} does not exist")
    return r


def gpu_name() -> str | None:
    """The CUDA device name, or None. Asks nvidia-smi so this works before torch is installed."""
    if not shutil.which("nvidia-smi"):
        return None
    out = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True)
    return out.stdout.strip().splitlines()[0] if out.returncode == 0 and out.stdout.strip() else None


def gpu() -> bool:
    return gpu_name() is not None


def xgb_device() -> str:
    return "cuda" if gpu() else "cpu"


def fp16() -> bool:
    return gpu()


def describe() -> dict:
    name = gpu_name()
    return {"where": "colab" if on_colab() else "laptop", "gpu": name, "device": "cuda" if name else "cpu", "fp16": name is not None,
            "machine": name or f"{platform.system()} {platform.machine()} {os.cpu_count()} cores"}

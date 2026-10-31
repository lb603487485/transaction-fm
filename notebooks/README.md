# Colab notebooks

Notebooks stay thin: mount Drive, import `tfm`, call a function, plot. Any logic longer than about 20 lines moves into `src/tfm` so it can be tested and reviewed.

No GitHub credentials ever go to Colab. The repo stays private; code reaches Colab by hand through Drive, and results come back the same way.

## Getting code to Colab

Drive folder `MyDrive/tfm/code/tfm/` holds a copy of `src/tfm/*.py`. After any change on the laptop, drag the changed `.py` files into that Drive folder (web UI, overwrite). Nothing else from the repo is needed on Colab.

## Notebooks

| File | Step | What it does |
|---|---|---|
| `00_setup.ipynb` | setup | mount Drive, import `tfm`, create the Drive folders, check GPU and data |

Upload a notebook to Colab with File -> Upload notebook, or copy it to Drive and open it with Colab. Notebooks are committed with outputs cleared (Edit -> Clear all outputs before downloading back).

## Cell 1, every session: mount Drive, import the package

```python
import sys
from google.colab import drive
drive.mount("/content/drive")
sys.path.insert(0, "/content/drive/MyDrive/tfm/code")
%pip install -q datasets xgboost
import tfm
print("tfm ok")
```

## Cell 2, pretraining: resume if a checkpoint exists

```python
from pathlib import Path
from transformers import Trainer
from tfm.checkpointing import find_latest_checkpoint, verify_checkpoint, training_args

OUT = "/content/drive/MyDrive/tfm/pretrain"
last = find_latest_checkpoint(OUT)
print("resuming from", last or "scratch")

trainer = Trainer(model=model, args=training_args(OUT, seed=0),
                  train_dataset=train_ds, eval_dataset=val_ds, data_collator=collator)
trainer.train(resume_from_checkpoint=str(last) if last else None)

for p in sorted(Path(OUT).glob("checkpoint-*")):
    print(p.name, verify_checkpoint(p) or "ok")
trainer.save_model(f"{OUT}/best")
```

`model`, `train_ds`, `val_ds` and `collator` come from `tfm.pretrain` once step 3 exists.

## Getting results back

Every step writes its small outputs (metrics json, png) to `MyDrive/tfm/results/`. Download them from the Drive web UI into `results/metrics/` and `results/figures/` in the local repo, then commit and push from the laptop.

## End of session

```python
drive.flush_and_unmount()
```

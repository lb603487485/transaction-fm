# Colab notebooks

Notebooks stay thin: mount Drive, import `tfm`, call a function, plot. Any logic longer than about 20 lines moves into `src/tfm` so it can be tested and reviewed.

Colab pulls the repo with a **read-only, single-repo** fine-grained GitHub token (Contents: read only, 30-day expiry) stored in Colab Secrets as `GH_TOKEN`. Colab never has write access; results come back through Drive and are committed from the laptop.

## Notebooks

| File | Step | What it does |
|---|---|---|
| `00_setup.ipynb` | setup | mount Drive, clone or pull the repo, install, create the Drive folders, check GPU and data |

Upload a notebook to Colab with File -> Upload notebook, or copy it to Drive and open it with Colab. Notebooks are committed with outputs cleared (Edit -> Clear all outputs before downloading back).

## Cell 1, every session: mount Drive, pull the repo, install

```python
import os
from google.colab import drive, userdata
drive.mount("/content/drive")
tok = userdata.get("GH_TOKEN")
if not os.path.exists("/content/transaction-fm"):
    !git clone -q https://{tok}@github.com/lb603487485/transaction-fm.git /content/transaction-fm
else:
    !cd /content/transaction-fm && git pull -q
%pip install -q -e /content/transaction-fm
%cd /content/transaction-fm
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

# Colab notebooks

Notebooks stay thin: mount Drive, import `tfm`, call a function, plot. Any logic longer than about 20 lines moves into `src/tfm` so it can be tested and reviewed.

Colab pulls and pushes with a **single-repo** fine-grained GitHub token (Contents: read and write, 30-day expiry, only `transaction-fm`) stored in Colab Secrets as `GH_TOKEN`. Small result files are pushed from Colab; checkpoints and data stay on Drive.

## Notebooks

| File | Step | What it does |
|---|---|---|
| `00_setup.ipynb` | setup | mount Drive, clone or pull the repo, install, create the Drive folders, check GPU and data, push results and notebooks |
| `01_data.ipynb` | 1 | run `tfm.data.run`, print EDA stats, two charts, push |

Notebooks live in `MyDrive/Colab Notebooks/tfm/notebooks/` and are opened from Drive (right click -> Open with -> Google Colaboratory). Edits save to that Drive file. The push cell in `00_setup.ipynb` copies every notebook there back into this folder with outputs cleared, so the repo copy is always the Drive copy minus outputs. A new notebook written on the laptop reaches Drive automatically the next time any notebook's cell 2 runs.

## Cell 1, every session: mount Drive, pull the repo, sync new notebooks to Drive

```python
import os, sys, glob, shutil
from google.colab import drive, userdata
drive.mount("/content/drive")
ROOT = "/content/drive/MyDrive/Colab Notebooks/tfm"
tok = userdata.get("GH_TOKEN")
if not os.path.exists("/content/transaction-fm"):
    !git clone -q https://{tok}@github.com/lb603487485/transaction-fm.git /content/transaction-fm
else:
    !cd /content/transaction-fm && git pull -q
%pip install -q datasets xgboost
%cd /content/transaction-fm
sys.path.insert(0, "/content/transaction-fm/src")
os.makedirs(f"{ROOT}/notebooks", exist_ok=True)
for f in glob.glob("notebooks/*.ipynb"):          # new notebooks from the repo appear in Drive
    dst = f"{ROOT}/notebooks/{os.path.basename(f)}"
    if not os.path.exists(dst):
        shutil.copy(f, dst); print("new notebook in Drive:", os.path.basename(f))
import tfm
print("tfm ok")
```

The same cell is the second cell of every notebook, so whichever notebook is opened first pulls the latest code and drops any new notebook into `ROOT/notebooks/`. Existing Drive notebooks are never overwritten by this cell; they flow the other way, Drive -> repo, through the push cell.

## Cell 2, pretraining: resume if a checkpoint exists

```python
from pathlib import Path
from transformers import Trainer
from tfm.checkpointing import find_latest_checkpoint, verify_checkpoint, training_args

OUT = f"{ROOT}/pretrain"
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

Every step writes its small outputs (metrics json, png) to `MyDrive/Colab Notebooks/tfm/results/`. The last cell of `00_setup.ipynb` copies them into `results/` and pushes. Pull on the laptop afterwards before editing README or report.

## End of session

```python
drive.flush_and_unmount()
```

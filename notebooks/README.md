# Colab notebooks

Notebooks stay thin: mount, pull, install, call a function from `src/tfm`, plot. Any logic longer than about 20 lines moves into `src/tfm` so it can be tested and reviewed.

## Cell 1, every session: mount Drive, pull the repo, install

Store a GitHub token with repo scope in Colab's Secrets panel (left sidebar, key icon) under the name `GH_TOKEN`. Never paste it into a cell.

```python
import os
from google.colab import drive, userdata
drive.mount("/content/drive")
tok = userdata.get("GH_TOKEN")
if not os.path.exists("/content/transaction-fm"):
    !git clone https://{tok}@github.com/lb603487485/transaction-fm.git /content/transaction-fm
else:
    !cd /content/transaction-fm && git pull
%pip install -q -e /content/transaction-fm
%cd /content/transaction-fm
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

## Cell 3, push small results back

```python
!cp /content/drive/MyDrive/tfm/pretrain/best/trainer_state.json results/metrics/pretrain_seed0_state.json
!git config user.email "linboyy@hotmail.com" && git config user.name "Bo Lin"
!git add results && git commit -m "pretrain seed0 results" && git push
```

## End of session

```python
drive.flush_and_unmount()
```

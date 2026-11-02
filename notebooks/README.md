# Colab notebooks

Notebooks stay thin: mount Drive, import `tfm`, call a function, plot. Any logic longer than about 20 lines moves into `src/tfm` so it can be tested and reviewed.

Colab pulls and pushes with a **single-repo** fine-grained GitHub token (Contents: read and write, 30-day expiry, only `transaction-fm`) stored in Colab Secrets as `GH_TOKEN`. Small result files are pushed from Colab; checkpoints and data stay on Drive.

## Notebooks

| File | Step | What it does |
|---|---|---|
| `00_setup.ipynb` | setup | mount Drive, clone or pull the repo, install, create the Drive folders, check GPU and data, push results and notebooks |
| `01_data.ipynb` | 1 | one stage per cell: raw head, column fate table (constant-per-card evidence), dist_km worked example, after clean plus quality checks, card-level side table, after split with boundary rows, per-card table and one fraud burst, EDA stats, save, two charts, push |
| `02_tokenizer.ipynb` | 2 | fit on train rows, special tokens and vocab composition, bin edges in human units, six-row worked example, random spot checks plus a 5,000-row machine check of decode against raw values, coverage per split, the two switches (log vs linear, with and without DT) and the binning trade-off number, windows per split, save and the json contents |
| `03_pretrain.ipynb` | 3 | load the step-2 windows, show one example as the model sees it (`<BOS>` shift, pad mask, -100 labels), GPT-2 config and parameter count, the per-field prior, train or resume with eval and save every 500 steps on Drive, verify every checkpoint, save `best/` with `tokenizer.json`, loss curves, per-field next-token accuracy vs prior, perplexity, `03_pretrain.json` |
| `04_finetune.ipynb` | 4 | build history-plus-current examples (train: all fraud + 5% normals; val/test unsampled), one decoded example next to the raw rows plus a 2,000-example machine check per split, pretrained vs scratch classifier, six resumable runs (2 inits x 3 seeds, finished runs skipped), per-run and mean/sd test metrics, pretraining gain, calibration note, PR curves, `04_finetune.json` |
| `05_baseline.ipynb` | 5 | build the 29 features and time it, feature table with NaN shares, one card's rows next to their history features plus a 2,000-row brute-force check, 12-point grid on validation (saved, so reruns skip it), six fits (29 features and current-only x 3 seeds, finished fits skipped), per-fit and mean/sd tables, history gain, side-by-side with the step-4 transformers, top-10 feature importance, PR curves with the transformer, `05_baseline.json` |
| `06a_validate.ipynb` | 6a | no training: loads the saved test scores of all four models, paired bootstrap (cards resampled) on transformer minus XGBoost, month-by-month alert rate / recall / FP ratio at one fixed cutoff, PSI per token field train vs test, recall per category / history bucket / cardholder group (cards.parquet joined for the first time), 20 + 20 frauds caught by one model only with decoded histories and a medians table, the ceiling numbers; writes `06a_validate.json` and `06a_monthly.png`; CPU, minutes |

Notebooks live in `MyDrive/Colab Notebooks/tfm/notebooks/` and are opened from Drive (right click -> Open with -> Google Colaboratory). Edits save to that Drive file. The push cell copies every notebook there back into this folder with outputs kept, so GitHub shows the frames and charts each stage produced. A new notebook written on the laptop reaches Drive automatically the next time any notebook's cell 2 runs.

## Cell 1, every session: where am I, mount Drive on Colab, pull the repo, sync new notebooks to Drive

```python
import os, sys, glob, shutil
try:
    from google.colab import drive, userdata                      # Colab: mount Drive, pull the repo, sync new notebooks to Drive
    drive.mount("/content/drive")
    tok = userdata.get("GH_TOKEN")
    if not os.path.exists("/content/transaction-fm"):
        !git clone -q https://{tok}@github.com/lb603487485/transaction-fm.git /content/transaction-fm
    else:
        !cd /content/transaction-fm && git pull -q
    %pip install -q datasets xgboost
    %cd /content/transaction-fm
except ImportError:                                               # laptop: this notebook is the repo copy; data under $TFM_ROOT
    here = os.getcwd()
    while not os.path.isdir(os.path.join(here, "src", "tfm")):
        here = os.path.dirname(here)
    %cd {here}
sys.path.insert(0, os.path.abspath("src"))
from tfm.runtime import root, describe, on_colab
ROOT = root()
if on_colab():
    for f in glob.glob("notebooks/*.ipynb"):                      # new notebooks from the repo appear in Drive
        dst = f"{ROOT}/notebooks/{os.path.basename(f)}"
        if not os.path.exists(dst):
            shutil.copy(f, dst); print("new notebook in Drive:", os.path.basename(f))
print("tfm ok |", describe(), "| ROOT =", ROOT)
```

The same cell is the second cell of every notebook, so whichever notebook is opened first pulls the latest code and drops any new notebook into `ROOT/notebooks/`. Existing Drive notebooks are never overwritten by this cell; they flow the other way, Drive -> repo, through the push cell.

`tfm.runtime` decides the rest at run time: `root()` is the Drive folder on Colab and `$TFM_ROOT` on a laptop; `gpu()` asks `nvidia-smi`, so XGBoost gets `device="cuda"` or `"cpu"` and the Trainer gets `fp16` only on CUDA. `describe()` goes into every run's json so results from different machines can be told apart.

## Running a notebook on the laptop (no Colab quota, no GPU)

Steps 1, 2 and 5 need no GPU. Download from Drive's `tfm/` what the step reads, keeping the same sub-paths, straight into the repo root (for step 5: `data/processed/transactions.parquet` and `finetune/pretrained_H12/seed0/test_scores.npy`; `data/`, `pretrain/`, `finetune/`, `baseline/` are gitignored, and `04_finetune.json` is read from `results/metrics/` when `results/` has no copy). Then, from the repo:

```
pip install -r requirements.txt            # torch and transformers are only needed for steps 3, 4, 6b
TFM_ROOT=$PWD jupyter notebook notebooks/05_baseline.ipynb     # or notebooks/06a_validate.ipynb
```

Run top to bottom. The notebook file being run is the repo copy, so its outputs land in the repo directly; the push cell copies the result files into `results/` and stops, and the commit is made from the terminal. Before the next Colab session, delete the Drive copy of any notebook whose code changed on the laptop, so cell 2 brings the new one in.

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

The full version, with `compute_metrics` for per-field accuracy, is Stage 3c of `03_pretrain.ipynb`; `model` and the datasets come from `tfm.pretrain`.

## Getting results back

Every step writes its small outputs (metrics json, png) to `MyDrive/Colab Notebooks/tfm/results/`. The last cell of `00_setup.ipynb` copies them into `results/` and pushes. Pull on the laptop afterwards before editing README or report.

## End of session

```python
drive.flush_and_unmount()
```

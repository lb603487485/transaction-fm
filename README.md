# transaction-fm

Mini Transaction Foundation Model: a GPT-2 style decoder pretrained with next-token prediction on tokenized card transactions (Sparkov synthetic fraud data), fine-tuned for fraud detection, compared against an XGBoost baseline with engineered features, and checked the way a model validator would (drift, subgroups, calibration, leakage).

Scaled-down reproduction of the recipe in the Wells Fargo talk "What It Takes to Build Production-Grade Foundation Models in Finance" (1B parameters, 8 x H200) at about 11M parameters on one free Colab T4. The point is not the absolute score; it is testing which of the talk's claims survive at small scale.

Status: steps 1 to 3 written (2026-11-01), steps 4 to 7 to follow.

## Layout

```
src/tfm/
  checkpointing.py   resume-safe Trainer settings, latest-checkpoint finder, checkpoint verifier
  data.py            step 1: load, strip PII, sort by card and time, time-based split, EDA stats
  tokenizer.py       step 2: TransactionTokenizer (fit on train only, encode, decode, save/load), make_windows
  pretrain.py        step 3: GPT-2 config, <BOS>-shifted examples, per-field next-token accuracy vs prior, run summary
  finetune.py        step 4: GPT2ForSequenceClassification, last-token pooling           (todo)
  baseline.py        step 5: leak-free rolling features + XGBoost                        (todo)
  evaluate.py        step 6: ablations and validator checks                              (todo)
configs/             one yaml per run (seed, history length, bin counts)
results/metrics/     small json files pushed back from Colab
results/figures/     png charts used in README and report
notebooks/           thin Colab drivers: mount, pull, install, run, plot
tests/               pytest; runs on CPU without torch
report/              3 to 4 page technical report
```

Data, token arrays and checkpoints never enter this repo. They live on Google Drive. Code reaches Colab by `git clone` with a single-repo token; small result files are pushed back from Colab, checkpoints and data stay on Drive:

```
MyDrive/Colab Notebooks/tfm/          (ROOT in the notebooks)
  notebooks/           the .ipynb files, opened from Drive with Colab; synced back here by the push cell
  data/raw/            Sparkov csv
  data/processed/      transactions.parquet (split column); tokenizer.json (vocab, edges, config);
                       tokens_<split>.npy (n_windows x 512) and windows_<split>_meta.parquet
                       cards.parquet: gender, age_bucket, state, city_pop_bucket per card; read only by step 6 subgroup checks
  pretrain/            checkpoint-N/ (every 500 steps, newest 2 kept), best/ (lowest val loss, with its tokenizer.json)
  finetune/seed0..2/   checkpoint-N/ ..., best/
  xgb/                 saved boosters
  results/             metrics json and png, copied into results/ here by the notebook and pushed
```

## Train / val / test isolation

Split by time, written into every row as a `split` column: train is everything before the last 30 days of the Kaggle train period, val is those 30 days (checkpoint and threshold selection), test is the Kaggle test file (`fraudTest.csv`, scored once at the end). The boundaries are saved in `results/metrics/01_data.json`.

All three splits are kept in one parquet so that a test-period transaction can see the same card's earlier history, as a production scorer would. Leakage in the other direction is blocked in code rather than by file layout: every fit-style function (bin edges, vocabulary, merchant top-N, XGBoost) begins with `assert_train_only`, history windows only look backward, and model selection reads val only. See the docstring at the top of `src/tfm/data.py`.

## Training on free Colab without losing work

Colab disconnects (idle timeout, 12 h cap, GPU quota) cannot be prevented. The runs are built so a disconnect costs at most 500 steps:

1. `output_dir` is on Drive and the Trainer saves every 500 steps (`save_steps=500`, same cadence as `eval_steps` so `load_best_model_at_end` can pick the best by `eval_loss`).
2. Each Hugging Face checkpoint directory carries `model.safetensors`, `optimizer.pt`, `scheduler.pt`, `rng_state.pth` and `trainer_state.json` (global_step, best_metric, log_history), so resuming restores the weights, the optimizer moments, the position in the cosine schedule, the RNG and the best-so-far record.
3. Before `trainer.train(...)`, `find_latest_checkpoint(out_dir)` returns the newest *complete* checkpoint (a directory written half-way through a disconnect is skipped) and training resumes from it.
4. After training, `verify_checkpoint` is run on every checkpoint left on Drive; any missing file or a `global_step` that disagrees with the directory name is printed before the next step of the project uses it.
5. Keep each run under 2 hours and call `drive.flush_and_unmount()` at the end of a session so Drive finishes syncing.

See `notebooks/README.md` for the cells.

## Local development

```
pip install -e .
pytest
```

# transaction-fm

Mini Transaction Foundation Model: a GPT-2 style decoder pretrained with next-token prediction on tokenized card transactions (Sparkov synthetic fraud data), fine-tuned for fraud detection, compared against an XGBoost baseline with engineered features, and checked the way a model validator would (drift, subgroups, calibration, leakage).

Scaled-down reproduction of the recipe in the Wells Fargo talk "What It Takes to Build Production-Grade Foundation Models in Finance" (1B parameters, 8 x H200) at about 11M parameters on one free Colab T4. The point is not the absolute score; it is testing which of the talk's claims survive at small scale.

Status: skeleton only (2026-10-31). Steps 1 to 7 land in `src/tfm/` over the next week.

## Layout

```
src/tfm/
  checkpointing.py   resume-safe Trainer settings, latest-checkpoint finder, checkpoint verifier
  data.py            step 1: load, strip PII, sort by card and time, time-based split   (todo)
  tokenizer.py       step 2: per-field binning, vocabulary, windows                      (todo)
  pretrain.py        step 3: causal LM pretraining                                       (todo)
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

Data, token arrays and checkpoints never enter this repo. They live on Google Drive:

```
MyDrive/tfm/
  data/raw/            Sparkov csv
  data/processed/      token arrays (.npy), vocab.json, bin_edges.json
  pretrain/            checkpoint-N/ ..., best/
  finetune/seed0..2/   checkpoint-N/ ..., best/
  xgb/                 saved boosters
```

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

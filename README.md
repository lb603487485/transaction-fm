# transaction-fm

Mini Transaction Foundation Model: a GPT-2 style decoder pretrained with next-token prediction on tokenized card transactions (Sparkov synthetic fraud data), fine-tuned for fraud detection, compared against an XGBoost baseline with engineered features, and checked the way a model validator would (drift, subgroups, calibration, leakage).

Scaled-down reproduction of the recipe in the Wells Fargo talk "What It Takes to Build Production-Grade Foundation Models in Finance" (1B parameters, 8 x H200) at about 11M parameters on one free Colab T4. The point is not the absolute score; it is testing which of the talk's claims survive at small scale.

Status: steps 1 to 4 run on Colab; steps 5, 6a and 6b run on the laptop (5 and 6a on 2026-11-02 on CPU, 6b on 2026-11-02 to 11-03 on Apple MPS, 7.2 hours for 15 runs); 6c (tokenizer and objective ablations, perplexity vs PR-AUC) written and smoke-tested, to run on the T4; 7 to follow.

## Layout

```
src/tfm/
  checkpointing.py   resume-safe Trainer settings, latest-checkpoint finder, checkpoint verifier
  data.py            step 1: load, strip PII, sort by card and time, time-based split, EDA stats
  tokenizer.py       step 2: TransactionTokenizer (fit on train only, encode, decode, save/load), make_windows
  pretrain.py        step 3: GPT-2 config, <BOS>-shifted examples, per-field next-token accuracy vs prior, run summary
  finetune.py        step 4: history + current examples (train sampled, val/test not), GPT2ForSequenceClassification, last-token pooling
  metrics.py         PR-AUC, ROC-AUC, recall and card discovery at a fixed alert rate, FP ratio, sampling correction, mean/sd over seeds
  baseline.py        step 5: 29 leak-free features (rolling windows closed on the left, cumsum minus current), XGBoost with early stopping on val, grid
  validate.py        step 6a: validator checks on saved scores: paired bootstrap by card, monthly metrics at one fixed cutoff, PSI per token field, subgroups, 20 + 20 disagreements
  ablate.py          step 6b: history-length and pooling ablations from pretrain/best (15 resumable fine-tuning runs), mean-pooling head, the three gates for training on a second machine
  mlm.py             step 6c: masked-token examples (15%, BERT's 80/10/10), BertConfig mirroring the GPT-2 sizes, BERT body with a last-token head
  ablate_pretrain.py step 6c: four arms that pretrain from scratch (A step-2 tokenizer CLM, B linear amount bins, C no time-gap token, D MLM), snapshots of A for the perplexity-vs-PR-AUC scatter, resumable pretrain_arm
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
  finetune/<init>_H<H>/seed<k>/   checkpoint-N/ ..., test_scores.npy   (init = pretrained | scratch)
  baseline/<variant>/seed<k>/     test_scores.npy   (variant = full | current, the 29-feature and 6-feature XGBoost)
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

## Colab or laptop, GPU or CPU

`src/tfm/runtime.py` decides at run time: `root()` is the Drive folder on Colab and `$TFM_ROOT` on a laptop, normally the repo itself (`TFM_ROOT=$PWD`): the Drive sub-folders `data/`, `pretrain/`, `finetune/`, `baseline/` sit at the repo root and are gitignored, and run output lands in `results/` next to the tracked `metrics/` and `figures/`; `gpu()` asks `nvidia-smi`, so XGBoost runs on `cuda` or `cpu` and the Trainer uses `fp16` only on CUDA (CPU and Apple MPS run fp32). Every run json carries `runtime` (`describe()`: where, gpu, device, fp16, machine). Saved checkpoints are fp32 safetensors, so a model trained on the T4 loads unchanged on a laptop; fp16 is a property of the run that produced a result, not of the weights. Steps 1, 2, 5 and 6a are CPU work and run on a laptop in minutes; steps 3, 4, 6b and 6c train, want the T4, and can run on an Apple-silicon laptop through MPS (06c is four pretraining runs of about 37 min each on a T4 plus 17 fine-tuning runs of about 10 min; the arms are compared only with each other, so all of 06c runs on one machine) (`accelerator()` reports `cuda`, `mps` or `cpu`; the Trainer picks MPS by itself). Arms of one ablation run on one kind of machine, and notebook 06b opens with three gates (checkpoint integrity, re-scoring a T4 checkpoint locally, retraining the H12 arm and checking it lands in the T4 seed spread) before it trains anything. On the M4 Pro the first two gates passed (the re-scored PR-AUC matched the saved one to 6e-6) and the third did not: the retrained H12 arm landed 0.0012 PR-AUC below the T4 mean, 8 T4 standard deviations away but inside the laptop's own seed spread (`results/metrics/06b_ablations.json`, `gates`). Fine-tuning is therefore not reproducible across machines at the 1e-3 level, and no number in 06b is compared with a T4 number; the T4 reference is only reported next to it.

Anaconda note: pip's torch wheel ships its own `libomp.dylib`, while Anaconda's scikit-learn and xgboost use `/opt/anaconda3/lib/libomp.dylib`; two OpenMP runtimes in one process segfault on the first parallel kernel (seen in `layer_norm` and in `xgboost.train`). Fix used here: replace `site-packages/torch/lib/libomp.dylib` with a symlink to Anaconda's copy (the original is kept as `libomp.dylib.pip-original`). A conda-installed torch avoids the problem from the start.

## Feature engineering cost (step 5)

The guide asks for the cost of the hand-built features, because "no feature engineering" is one of the talk's selling points. Two numbers: compute time is measured and stored as `feature_seconds` in `results/metrics/05_baseline.json`; authoring time is from the commit log: `baseline.py` (29 features in about 60 lines) and its 10 tests were written in one sitting on 2026-11-02, under an hour, with an AI coding assistant. That number does not refute the talk's claim, which is about maintaining hundreds of features over years; it is recorded so the report can say what this baseline cost.

## Local development

```
pip install -e .
pytest
```

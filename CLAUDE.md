# CLAUDE.md

Mini Transaction Foundation Model for TD Bank interview prep (Applied ML Scientist I, Model Validation). The project guide is `interview-prep/companies/td-bank/source/Tabular-DL-credit-default-guide-2026-10-29.pdf` (title says credit default, content is this project); the prep plan with dates is `interview-prep/companies/td-bank/prep/prep-plan.md`.

## Rules

- Code lives in `src/tfm/` with tests in `tests/`. Notebooks are thin drivers only.
- Data, token arrays and checkpoints go to Google Drive (`MyDrive/tfm/`), never into git.
- Every training run must be resumable: `output_dir` on Drive, `save_steps=500`, resume from `find_latest_checkpoint`, verify with `verify_checkpoint`. See README.
- Numbers in README and report come from `results/metrics/*.json` only. Never write a metric that has not been measured.
- Repo is private until the report is done; then public, and the link goes on the resume.

## Coding guidelines

Follow the Karpathy principles in `/Users/evolin1/Desktop/AICoding/CLAUDE.md`: think before coding, simplicity first, surgical changes, goal-driven execution with tests.

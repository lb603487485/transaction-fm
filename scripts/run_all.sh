#!/usr/bin/env bash
# Run the whole pipeline on one machine, notebook by notebook, in order. Every notebook is executed in
# place with its outputs kept (the same command the laptop runs used), under TFM_ROOT = this repo, so the
# data, checkpoints and scores land in the gitignored mirror folders and the kept json/png in results/.
#
#   scripts/run_all.sh                 run 01 to 06f
#   scripts/run_all.sh --from 03       resume at 03_pretrain (01 and 02 already done)
#   scripts/run_all.sh --only 06f      run one notebook
#   scripts/run_all.sh --dry-run       print the commands only
#
# Stops at the first failure. Logs go to logs/<notebook>.log. Needs data/raw/fraudTrain.csv and
# fraudTest.csv (Kaggle, Sparkov) before 01. Steps 03, 04 and 06b to 06e train and want MPS or CUDA.
set -euo pipefail
cd "$(dirname "$0")/.."
ORDER=(01_data 02_tokenizer 03_pretrain 04_finetune 05_baseline 06a_validate 06b_ablations 06c_tokenizer 06c_objective 06d_finetune_extras 06e_hybrid_head 06f_compare_finetunes)
FROM=""; ONLY=""; DRY=0
while [ $# -gt 0 ]; do
  case "$1" in
    --from) FROM="$2"; shift 2 ;;
    --only) ONLY="$2"; shift 2 ;;
    --dry-run) DRY=1; shift ;;
    *) echo "unknown argument $1"; exit 2 ;;
  esac
done
[ -f data/raw/fraudTrain.csv ] && [ -f data/raw/fraudTest.csv ] || { echo "data/raw needs fraudTrain.csv and fraudTest.csv first"; [ $DRY -eq 1 ] || exit 1; }
mkdir -p logs
started=0
for nb in "${ORDER[@]}"; do
  [ -n "$ONLY" ] && [[ "$nb" != "$ONLY"* ]] && continue
  if [ -n "$FROM" ] && [ $started -eq 0 ]; then [[ "$nb" == "$FROM"* ]] && started=1 || continue; fi
  cmd=(env TFM_ROOT="$PWD" jupyter nbconvert --to notebook --execute --inplace "notebooks/$nb.ipynb" --ExecutePreprocessor.kernel_name=python3 --ExecutePreprocessor.timeout=-1)
  echo "$(date '+%Y-%m-%d %H:%M:%S')  $nb"
  if [ $DRY -eq 1 ]; then echo "  ${cmd[*]}"; continue; fi
  "${cmd[@]}" > "logs/$nb.log" 2>&1 || { echo "FAILED at $nb, see logs/$nb.log"; exit 1; }
done
echo "$(date '+%Y-%m-%d %H:%M:%S')  done"

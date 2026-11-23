#!/usr/bin/env bash
# Reproduce every table and figure of the paper from the raw data.
#   bash run_all.sh            -> full run (about 1-2 hours on a 2-core laptop)
#   bash run_all.sh --tables   -> only rebuild tables and figures from the committed results/ (seconds)
set -euo pipefail
cd "$(dirname "$0")"
J=.
if [ "${1:-}" != "--tables" ]; then
  python scripts/check_data.py $J
  # controlled label-shift benchmark: main configuration + robustness configurations
  rm -rf results/trials results/trials_*            # start clean (the scripts skip datasets already done)
  python scripts/benchmark_multi.py $J                         # LR, r=5 (main)
  python scripts/benchmark_multi.py $J --model xgb --tag xgb   # XGBoost
  for r in 1 2 10; do python scripts/benchmark_multi.py $J --r $r --tag r$r; done
  for n in 200 500 1000; do python scripts/benchmark_multi.py $J --nmax $n --minpos 5 --tag n$n; done
  # temporal case studies
  for c in fraud bank bank_nomacro diabetes; do python scripts/natural_shift.py $J $c; done
  # domain-classifier AUC under pure label shift
  python scripts/domain_auc_label_shift.py $J
else
  # the trial-level results are committed as one file; split it into results/trials*/ for the analysis scripts
  python scripts/expand_trials.py $J
fi
python scripts/natural_summary.py $J
python scripts/analyse.py $J            | tee results/analysis_output.txt
python scripts/compare_configs.py $J    | tee results/config_comparison.txt
python scripts/paper_tables.py $J $J > /dev/null
python scripts/paper_figures.py $J $J
python scripts/export_excel.py $J       # results/SAT_*.xlsx, results/SAT_trials_long.csv, data/dataset_catalogue.csv
echo "Done: results/SAT_*.xlsx (tables), paper_tables.json (all manuscript numbers), figs/ (figures)."

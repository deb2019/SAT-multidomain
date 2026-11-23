# Shift-Adaptive Decision Thresholds under Prevalence Shift — multi-domain study

Code, results and figures for the manuscript

> **Shift-Adaptive Decision Thresholds for Imbalanced Binary Classification under Prevalence Shift: A Multi-Domain Empirical Study.**
> Debashis Roy, Anandarup Roy, Utpal Roy (manuscript under submission).

It extends the conference paper *Shift-Adaptive Thresholding: Retraining-Free, Assumption-Aware Decision Thresholds for Imbalanced Binary Classification under Label Shift* (IMPACT 2026, accepted), whose code is at <https://github.com/deb2019/SAT-shift-adaptive-thresholding>.

## What the study does

When the prevalence of the minority class changes between training and deployment, a decision threshold tuned at training time stops being cost-optimal. **Shift-adaptive thresholding (SAT)** keeps the trained model and moves only its threshold:

1. estimate the target prevalence π̂ₜ from **unlabelled** target scores (EM, or BBSE);
2. threshold the calibrated source posterior at τ(π̂ₜ) = w₀ / (w₀ + r·w₁), with w₁ = π̂ₜ/πₛ, w₀ = (1−π̂ₜ)/(1−πₛ) and r = C_FN / C_FP.

The study compares eight decision policies (fixed 0.5, SMOTE, calibrated SMOTE with a tuned threshold, source-tuned threshold, BBSE-SAT, SMOTE + the same adaptation, Point-SAT, Robust-SAT with a goodness-of-fit gate) on **32 imbalanced datasets from four domains** (cyber security, finance and fraud, medical, industrial) under controlled prevalence shift, with robustness checks (XGBoost, r = 1/2/10, batch sizes 200–1,000), and in **three temporal case studies** (simulated card transactions, bank marketing, diabetes readmission).

## Quick start

```bash
pip install -r requirements.txt
bash run_all.sh --tables     # rebuild every table and figure from the committed results (seconds)

# full reproduction from raw data (about 1–2 hours on a 2-core laptop):
#   1. download the data files listed in data/README.md into data/
#   2. python scripts/check_data.py .
#   3. bash run_all.sh
```

All random seeds are fixed (data splits `random_state=0`, ten subsampling seeds per shift level; XGBoost uses `random_state=0, n_jobs=1`), so a full run reproduces the committed results.

## Repository layout

| Path | Content |
|---|---|
| `scripts/audit.py`, `scripts/select.py` | dataset audit (signal, leakage, duplicates, feasibility) and the pre-specified selection |
| `scripts/benchmark_multi.py` | controlled label-shift benchmark (all eight policies); options `--model xgb`, `--r`, `--nmax`, `--minpos`, `--tag` |
| `scripts/natural_shift.py` | temporal case studies (`fraud`, `bank`, `bank_nomacro`, `diabetes`) |
| `scripts/domain_auc_label_shift.py` | domain-classifier AUC under pure label shift |
| `scripts/analyse.py`, `scripts/compare_configs.py`, `scripts/natural_summary.py` | statistics (Friedman, Nemenyi, Wilcoxon + Holm) and summaries |
| `scripts/paper_tables.py`, `scripts/paper_figures.py` | every number, table and figure in the manuscript |
| `results/SAT_main_results.xlsx` | main benchmark: Tables 2, 4–7 and Appendix B1 (one sheet each) |
| `results/SAT_robustness_summary.xlsx` | XGBoost, cost ratio and batch-size configurations: Tables 8–9, ranks, mean excess cost |
| `results/SAT_temporal_summary.xlsx` | temporal case studies: Table 10 and the per-period results of each case |
| `results/SAT_diagnostics_summary.xlsx` | domain-classifier AUC (Table 11), gate fallback (Table 12), prior-estimation error per dataset |
| `results/SAT_trials_long.csv` | every target batch of every configuration, one row each (`config` column; achieved prevalence, batch size, EM/BBSE prior, thresholds, gate decision, absolute and excess cost of every policy) |
| `results/natural/` | per-period results of the temporal case studies (CSV) |
| `data/dataset_catalogue.csv` | all 75 candidate datasets: source, licence, audit statistics, removed columns, include/exclude decision and reason |
| `paper_tables.json` | all numbers quoted in the manuscript |
| `figs/` | manuscript figures |
| `audit_final.csv`, `sources.csv`, `drop_columns.json` | audit record, dataset sources and licences, columns removed before modelling |
| `data/` | place the data files here (see `data/README.md`; data are not redistributed) |

## Where each manuscript item comes from

Numbering follows the journal version (JCTA format, 5 tables, 4 figures, 1 algorithm).

| Manuscript | Produced by | Output |
|---|---|---|
| Table 1 (retained datasets), audit counts in Section 4.1 | `audit.py`, `select.py` | `audit_final.csv`, `data/dataset_catalogue.csv`, `results/SAT_main_results.xlsx` (Table2_Datasets) |
| Table 2 (policies and main results), Fig. 2 (CD diagram) | `benchmark_multi.py` → `analyse.py`, `paper_tables.py`, `paper_figures.py` | `results/SAT_main_results.xlsx` (Table4_Summary, Table5_Wilcoxon), `figs/fig4_cd.png` |
| Table 3 (by domain and by shift), Fig. 3 (excess cost vs shift) | same | `results/SAT_main_results.xlsx` (Table6_ByDomain, Table7_ByShift), `figs/fig5_shift.png` |
| Table 4 (robustness, including batch size) | `benchmark_multi.py --model/--r/--nmax` → `compare_configs.py` | `results/SAT_robustness_summary.xlsx` |
| Table 5, Fig. 4 (temporal cases) | `natural_shift.py` → `natural_summary.py` | `results/SAT_temporal_summary.xlsx`, `figs/fig9_natural.png` |
| Domain AUC under pure label shift and gate fallback (Section 5.4) | `domain_auc_label_shift.py`, `paper_tables.py` | `results/SAT_diagnostics_summary.xlsx` |
| Fig. 1 (study design) | `paper_figures.py` | `figs/fig2_design.png` |
| Supplementary: excluded datasets, removed columns, sources and licences | — | `data/dataset_catalogue.csv`, `drop_columns.json` |
| Supplementary: per-dataset results, by-domain, batch-size and rank figures | `paper_tables.py`, `paper_figures.py` | `results/SAT_main_results.xlsx` (B1 sheets), `figs/fig6_domain.png`, `figs/fig8_batch.png`, `figs/fig7_robustness.png` |
| Synthetic study | conference repository | <https://github.com/deb2019/SAT-shift-adaptive-thresholding> |

## Notes

* The Excel workbooks hold values only (no formulas) and are regenerated by `scripts/export_excel.py`. `bash run_all.sh --tables` first splits `results/SAT_trials_long.csv` into per-dataset files (`scripts/expand_trials.py`), then rebuilds everything.
* Columns marked for exclusion by the data preparer (identifiers, post-outcome variables, failure-mode flags) and row-index columns are removed automatically (`drop_columns.json`).
* The card-transaction data are simulated (Sparkov generator); that case tests a temporal change in prevalence, not real-world drift.
* The excess-cost reference is the best of 999 grid thresholds on the calibrated target scores; it is an empirical reference, not a Bayes bound.

## Citation

```bibtex
@article{roy_sat_multidomain,
  title   = {Shift-Adaptive Decision Thresholds for Imbalanced Binary Classification under
             Prevalence Shift: A Multi-Domain Empirical Study},
  author  = {Roy, Debashis and Roy, Anandarup and Roy, Utpal},
  note    = {Manuscript under submission},
  year    = {2026}
}
```

## License

Code: MIT (see `LICENSE`). Datasets remain under their original licences (`data/README.md`).

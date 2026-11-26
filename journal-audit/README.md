# Auditing Threshold Changes under Prevalence Shift with a Fixed Target Label Budget

The controlled benchmark has 1,790 batches from 37 datasets (17 finance and fraud, 11 medical, five cybersecurity and four industrial and anomaly). The 26 temporal batches come from bank marketing, diabetes readmission, simulated card transactions and office occupancy. Bank marketing appears in both designs. The combined study has 40 distinct datasets.

## Reproduce the audit

From this directory, install the Python dependencies in your environment. The simulation also requires a C++ compiler with C++17 support.

```bash
python -m pip install numpy pandas scipy matplotlib
cd comparison
g++ -O3 -std=c++17 simulate.cpp -o simulate
python run.py
python temporal_random.py
python summarise.py
cd ..
python figures.py
```

The comparison starts from the included finite-batch counts. It uses a false-positive cost of one and a false-negative cost of five. Monte Carlo results may differ slightly across C++ standard libraries. To inspect the published estimates immediately, use the committed CSV files below. Training a classifier again requires its prepared source data. See each cohort's README for its protocol and inputs.

| Manuscript item | File or computation |
|---|---|
| Table 1 | Literature comparison and descriptions of the earlier studies in the paper |
| Table 2 | Count unique datasets and batches by group and domain in `comparison/input_counts.csv` |
| Table 3 | `comparison/controlled_summary.csv` at budget 25. For the incumbent and unconditional SAT, average the per-dataset costs in `controlled-extension/combined_dataset_results.csv` at the same budget |
| Table 4 | Exact approval rates in `comparison/controlled_summary.csv`. For targeted and random costs, average per-dataset costs by budget in `controlled-extension/combined_dataset_results.csv` |
| Table 5 | Average per-dataset costs by domain at budget 25 in `controlled-extension/combined_dataset_results.csv` |
| Tables 6 and 7 | `comparison/all_temporal_gains.csv` at budget 25. Net gain equals remaining gain minus the unit label price times labels purchased |
| Figures 1–3 | `figures.py` generates the disagreement diagram, budget comparison and period-by-period savings in `figures/` |
| Appendix A | Audit catalogues in `controlled-original/` and `controlled-extension/`, restricted to the controlled IDs in `comparison/input_counts.csv` |

A positive full-batch saving favours the proposed threshold. Remaining gain excludes cases whose labels were purchased. The fixed-budget error guarantee applies to full-batch cost. Costs in the controlled benchmark give equal weight to each dataset. Approval probabilities condition on the sign of the full-batch saving. The original conference study and earlier 32-dataset benchmark are documented in the [repository overview](../README.md).

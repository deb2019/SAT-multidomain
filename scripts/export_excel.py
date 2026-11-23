"""Export all results to a few Excel workbooks + one long trial-level CSV + a dataset catalogue.
Usage: python scripts/export_excel.py .
Writes:
  results/SAT_trials_long.csv          every target batch of every configuration (one row each)
  results/SAT_main_results.xlsx        Tables 2, 4-7 and Appendix B1
  results/SAT_robustness_summary.xlsx  Tables 8-9 and per-configuration ranks / mean excess cost
  results/SAT_temporal_summary.xlsx    Table 10 and the per-period results of each temporal case
  results/SAT_diagnostics_summary.xlsx Tables 11-12, prior-estimation error, gate fallback per dataset
  data/dataset_catalogue.csv           every candidate dataset: source, licence, audit statistics, decision
All numbers are values computed by the scripts in scripts/ (no spreadsheet formulas)."""
import sys, os, glob, json
import pandas as pd
from openpyxl import load_workbook
from openpyxl.styles import Font, Alignment, PatternFill
from openpyxl.utils import get_column_letter

J = sys.argv[1] if len(sys.argv) > 1 else "."
RES = os.path.join(J, "results")
D = json.load(open(os.path.join(J, "paper_tables.json")))
METHODS = ["Fixed-0.5", "SMOTE@0.5", "SMOTE+cal+tuned", "Source-tuned", "BBSE-SAT", "SMOTE+cal+SAT", "Point-SAT", "Robust-SAT"]
CONFIGS = [("main (LR, r=5)", "trials"), ("XGBoost, r=5", "trials_xgb"), ("LR, r=1", "trials_r1"), ("LR, r=2", "trials_r2"),
           ("LR, r=10", "trials_r10"), ("LR, batch<=200", "trials_n200"), ("LR, batch<=500", "trials_n500"), ("LR, batch<=1000", "trials_n1000")]
NAMES = {d["id"]: d["name"] for d in D["datasets"]}

def style(path):
    """Arial everywhere, bold shaded header row, frozen header, sensible column widths."""
    wb = load_workbook(path)
    for ws in wb.worksheets:
        for row in ws.iter_rows():
            for c in row: c.font = Font(name="Arial", size=10, bold=c.font.bold)
        if ws.title != "README":
            for c in ws[1]:
                c.font = Font(name="Arial", size=10, bold=True); c.fill = PatternFill("solid", fgColor="DDE6F0")
                c.alignment = Alignment(wrap_text=True, vertical="top")
            ws.freeze_panes = "B2"
        else:
            ws["A1"].font = Font(name="Arial", size=12, bold=True)
        for i, col in enumerate(ws.columns, 1):
            width = max(len(str(c.value)) if c.value is not None else 0 for c in col)
            ws.column_dimensions[get_column_letter(i)].width = min(max(10, width + 2), 70 if ws.title == "README" else 28)
    wb.save(path)

def write(path, readme_lines, sheets):
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        pd.DataFrame({"README": readme_lines}).to_excel(xw, sheet_name="README", index=False, header=False)
        for name, df in sheets:
            df.to_excel(xw, sheet_name=name[:31], index=False)
    style(path); print("wrote", path)

# ---------------------------------------------------------------- long trial file
parts = []
for label, sub in CONFIGS:
    fs = sorted(glob.glob(os.path.join(RES, sub, "*.csv")))
    if fs:
        t = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True); t.insert(0, "config", label); t.insert(1, "config_dir", sub)
        parts.append(t)
LONG = pd.concat(parts, ignore_index=True)
LONG.to_csv(os.path.join(RES, "SAT_trials_long.csv"), index=False)
print("wrote SAT_trials_long.csv", LONG.shape)
main = LONG[LONG.config_dir == "trials"]

# ---------------------------------------------------------------- main results workbook
rows = D["main"]["rows"]
t4 = pd.DataFrame([{"Policy": r["method"], "Mean absolute cost": r["abs"], "Mean excess cost (x1e-3)": r["excess"],
                    "Median excess cost (x1e-3)": r["median"], "Average rank": r["rank"]} for r in rows])
t5 = pd.DataFrame([{"Point-SAT vs": r["vs"], "Point-SAT cheaper on (datasets)": f"{r['wins']}/{D['main']['N']}",
                    "Raw p (one-sided Wilcoxon)": r["raw"], "Holm-adjusted p": r["holm"]} for r in D["wilcoxon"]])
t6 = pd.DataFrame(D["domain"]).rename(columns={"ps_src": "Point-SAT cheaper than Source-tuned"})
t7 = pd.DataFrame(D["shift"]).rename(columns={"m": "Shift multiplier m", "cov": "Datasets supporting the level"})
t2 = pd.DataFrame([{"ID": d["id"], "Domain": d["domain"], "Dataset": d["name"], "File": d["file"], "N": d["N"], "Features": d["features"],
                    "Minority prevalence": d["prev"], "Calibrated logistic AUC": d["auc"]} for d in D["datasets"]])
b1 = (main.groupby("id")[METHODS].mean() * 1000).reset_index().rename(columns={"id": "ID"})
b1.insert(1, "Dataset", b1.ID.map(NAMES)); b1.insert(2, "Domain", b1.ID.map(main.groupby("id").domain.first()))
b1a = main.groupby("id")[["abs_" + k for k in METHODS]].mean().rename(columns=lambda c: c[4:]).reset_index().rename(columns={"id": "ID"})
b1a.insert(1, "Dataset", b1a.ID.map(NAMES))
write(os.path.join(RES, "SAT_main_results.xlsx"),
      ["SAT multi-domain study: main controlled label-shift benchmark (logistic scorer, isotonic calibration, r = 5)",
       f"{D['main']['N']} datasets, {D['main']['trials']} target batches. Friedman chi2 = {D['main']['friedman_chi2']:.2f}, p = {D['main']['friedman_p']:.2e}; Nemenyi CD (k = 8, alpha = 0.05) = {D['main']['CD']:.2f}.",
       "Excess cost = cost of a policy minus the cost of the best of 999 grid thresholds on the calibrated target scores (x1e-3). Absolute cost = (FP + r*FN)/n with C_FP = 1, C_FN = r.",
       "Sheets: Table2_Datasets, Table4_Summary, Table5_Wilcoxon, Table6_ByDomain, Table7_ByShift, B1_PerDataset_excess, B1_PerDataset_abs.",
       "Produced by scripts/benchmark_multi.py (results) and scripts/paper_tables.py + scripts/export_excel.py (tables). Trial-level rows: results/SAT_trials_long.csv (config = 'main (LR, r=5)')."],
      [("Table2_Datasets", t2), ("Table4_Summary", t4), ("Table5_Wilcoxon", t5), ("Table6_ByDomain", t6), ("Table7_ByShift", t7),
       ("B1_PerDataset_excess", b1), ("B1_PerDataset_abs", b1a)])

# ---------------------------------------------------------------- robustness workbook
t9 = pd.DataFrame(D["config"])
t8 = pd.DataFrame(D["prior_by_batch"]).rename(columns={"nmax": "Max target batch", "em": "EM |error|", "bbse": "BBSE |error|",
                                                       "ps": "Point-SAT mean excess (x1e-3)", "src": "Source-tuned mean excess (x1e-3)"})
rk = pd.DataFrame(D["config_ranks"]).T.reset_index().rename(columns={"index": "Configuration"})
mx = pd.DataFrame(D["config_excess"]).T.reset_index().rename(columns={"index": "Configuration"})
write(os.path.join(RES, "SAT_robustness_summary.xlsx"),
      ["SAT multi-domain study: robustness across scorer (XGBoost), cost ratio (r = 1, 2, 10) and target-batch size (<= 200, 500, 1000).",
       "Table9_Configs: per configuration, Friedman p, Point-SAT wins vs Source-tuned and Holm-adjusted p (7-comparison family), two-sided p vs SMOTE+cal+SAT and BBSE-SAT, gate fallback rate, prior-estimation errors.",
       "Table8_BatchSize: prior-estimation error and mean excess cost by maximum target-batch size.",
       "Ranks / MeanExcess: average rank and mean excess cost (x1e-3) of every policy in every configuration.",
       "Produced by scripts/benchmark_multi.py (options --model, --r, --nmax, --minpos) and scripts/compare_configs.py."],
      [("Table9_Configs", t9), ("Table8_BatchSize", t8), ("Ranks", rk), ("MeanExcess_x1e-3", mx)])

# ---------------------------------------------------------------- temporal workbook
t10 = pd.DataFrame(D["natural"])
sheets = [("Table10_Summary", t10)]
for c, nm in [("fraud", "Card_transactions"), ("bank", "Bank_marketing"), ("bank_nomacro", "Bank_no_macro"), ("diabetes", "Diabetes_readmission")]:
    d = pd.read_csv(os.path.join(RES, "natural", f"{c}_lr.csv"))
    d["PointSAT_minus_SourceTuned_x1e-3"] = (d["Point-SAT"] - d["Source-tuned"]) * 1000
    sheets.append((nm, d))
write(os.path.join(RES, "SAT_temporal_summary.xlsx"),
      ["SAT multi-domain study: temporal case studies (no subsampling; each target period keeps its own prevalence).",
       "Card transactions: simulated (Sparkov) data, source = fraudTrain.csv (1 Jan 2019 - 21 Jun 2020), targets = the seven calendar months of fraudTest.csv.",
       "Bank marketing: rows in calendar order; first half = source, second half = five consecutive periods (also without the five macro-economic columns).",
       "Diabetes readmission < 30 days: first encounter per patient in encounter order; first half = source, five later periods.",
       "Per-period sheets: true prevalence, EM and BBSE prior estimates, gate decision, domain-classifier AUC, excess cost (x1e-3 in the derived column) and absolute cost (abs_*) of every policy.",
       "Produced by scripts/natural_shift.py and scripts/natural_summary.py."], sheets)

# ---------------------------------------------------------------- diagnostics workbook
ls = pd.read_csv(os.path.join(RES, "domain_auc_label_shift.csv"))
t11 = pd.DataFrame(D["ls_auc"]["rows"]).rename(columns={"m": "Shift multiplier m", "n": "Datasets", "mean": "Mean AUC", "max": "Max AUC",
                                                        "ge06": "AUC >= 0.6", "ge08": "AUC >= 0.8"})
g = D["gate"]["by_shift"]
t12 = pd.DataFrame([{"Shift multiplier m": k, "Gate fallback rate": v} for k, v in
                    zip(["1/3", "1/2", "1", "2", "3"], [g["0.3333333333333333"], g["0.5"], g["1.0"], g["2.0"], g["3.0"]])]
                   + [{"Shift multiplier m": "All", "Gate fallback rate": D["gate"]["fallback"]}])
gate_ds = main.groupby("id").gate_reject.mean().reset_index().rename(columns={"id": "ID", "gate_reject": "Gate fallback rate"})
gate_ds.insert(1, "Dataset", gate_ds.ID.map(NAMES))
pe = main.assign(EM_abs_error=(main.em_prior - main.achieved_prev).abs(), BBSE_abs_error=(main.bbse_prior - main.achieved_prev).abs())
pe = pe.groupby("id")[["EM_abs_error", "BBSE_abs_error"]].mean().reset_index().rename(columns={"id": "ID"}); pe.insert(1, "Dataset", pe.ID.map(NAMES))
write(os.path.join(RES, "SAT_diagnostics_summary.xlsx"),
      ["SAT multi-domain study: diagnostics.",
       "Table11_DomainAUC: domain-classifier AUC (reference split vs one target batch) under pure label shift, by shift multiplier; DomainAUC_per_dataset gives every value.",
       "Table12_Gate: fallback rate of the Robust-SAT goodness-of-fit gate under pure label shift (nominal alpha = 0.05); Gate_per_dataset by dataset.",
       "PriorError_per_dataset: mean |estimated - achieved prevalence| of EM and BBSE (main configuration).",
       "Produced by scripts/domain_auc_label_shift.py, scripts/benchmark_multi.py and scripts/paper_tables.py."],
      [("Table11_DomainAUC", t11), ("DomainAUC_per_dataset", ls), ("Table12_Gate", t12), ("Gate_per_dataset", gate_ds),
       ("PriorError_per_dataset", pe)])

# ---------------------------------------------------------------- dataset catalogue
A = pd.read_csv(os.path.join(J, "audit_final.csv")); S = pd.read_csv(os.path.join(J, "sources.csv"))
drop = json.load(open(os.path.join(J, "drop_columns.json")))
cat = S[["id", "category", "prepared_file", "source_name", "source_host", "source_url", "licence", "redistribution", "notes"]].merge(
    A[["id", "status", "target", "N", "features", "prevalence", "n_pos", "lr_auc", "max_single_auc", "max_single_col",
       "dup_rows_frac", "feasible_levels", "domain", "decision", "reason"]], on="id", how="left")
cat["removed_columns"] = cat.id.map(lambda i: "; ".join(drop.get(i, [])))
cat["in_benchmark"] = cat.decision.eq("include")
cat.to_csv(os.path.join(J, "data", "dataset_catalogue.csv"), index=False)
print("wrote data/dataset_catalogue.csv", cat.shape, "| in benchmark:", int(cat.in_benchmark.sum()))

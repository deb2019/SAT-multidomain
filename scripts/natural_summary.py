"""Summarise the temporal case studies -> results/natural_summary.csv (Table 10)."""
import sys, os
import pandas as pd
J = sys.argv[1]
M = ["Fixed-0.5", "SMOTE+cal+tuned", "Source-tuned", "BBSE-SAT", "SMOTE+cal+SAT", "Point-SAT", "Robust-SAT"]
cases = [("fraud", "Finance: card fraud, monthly (2020)"), ("diabetes", "Medical: diabetes readmission, later periods"),
         ("bank", "Marketing: bank campaign, later periods"), ("bank_nomacro", "Marketing: same, macro columns removed")]
rows = []
for c, label in cases:
    d = pd.read_csv(os.path.join(J, "results", "natural", f"{c}_lr.csv"))
    r = {"case": label, "batches": len(d), "source prev": round(d.source_prev.iloc[0], 4),
         "target prev (range)": f"{d.true_prev.min():.4f}–{d.true_prev.max():.4f}",
         "domain AUC (mean)": round(d.domain_auc.mean(), 3), "EM |error| (mean)": round((d.em_prior - d.true_prev).abs().mean(), 4),
         "gate fired": f"{int(d.gate_reject.sum())}/{len(d)}"}
    for k in M: r[k] = round(d[k].mean() * 1000, 2)
    r["PS<Src"] = f"{int((d['Point-SAT'] < d['Source-tuned'] - 1e-12).sum())}/{len(d)}"
    rows.append(r)
S = pd.DataFrame(rows); S.to_csv(os.path.join(J, "results", "natural_summary.csv"), index=False)
pd.set_option("display.width", 250); print(S.to_string(index=False))

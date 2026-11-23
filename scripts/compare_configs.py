"""Summarise every benchmark configuration (LR r=5 main, XGB, r sweep, batch-size sweep) side by side."""
import sys, glob, os, json
import numpy as np, pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon
J = sys.argv[1]
METHODS = ["Fixed-0.5", "SMOTE@0.5", "SMOTE+cal+tuned", "Source-tuned", "BBSE-SAT", "SMOTE+cal+SAT", "Point-SAT", "Robust-SAT"]
CONFIGS = [("LR, r=5 (main)", "trials"), ("XGBoost, r=5", "trials_xgb"),
           ("LR, r=1", "trials_r1"), ("LR, r=2", "trials_r2"), ("LR, r=10", "trials_r10"),
           ("LR, batch<=200", "trials_n200"), ("LR, batch<=500", "trials_n500"), ("LR, batch<=1000", "trials_n1000")]
def holm(raw):
    it = sorted(raw.items(), key=lambda x: x[1]); m = len(it); run = 0; out = {}
    for i, (k, p) in enumerate(it): run = max(run, p*(m-i)); out[k] = min(1, run)
    return out
rows = []; ranks = {}; means = {}
for name, sub in CONFIGS:
    fs = glob.glob(os.path.join(J, "results", sub, "*.csv"))
    if not fs: print(f"{name}: not available yet"); continue
    T = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    M = T.groupby("id")[METHODS].mean(); N = len(M)
    st, p = friedmanchisquare(*[M[c].values for c in METHODS])
    rk = M.rank(axis=1).mean(); ranks[name] = rk.round(2); means[name] = (M.mean()*1000).round(1)
    raw = {b: wilcoxon(M["Point-SAT"], M[b], alternative="less").pvalue
           for b in METHODS if b != "Point-SAT"}          # same 7-comparison Holm family as Table 5
    adj = holm(raw)
    def two(b):  # two-sided for the near-equivalent adaptive comparators
        d = M["Point-SAT"] - M[b]; return wilcoxon(M["Point-SAT"], M[b]).pvalue if (d != 0).any() else np.nan
    rows.append({"config": name, "N": N, "trials": len(T), "Friedman p": f"{p:.1e}",
                 "PS<Src": f"{int((M['Point-SAT'] < M['Source-tuned']).sum())}/{N}", "Holm p vs Src": float(adj["Source-tuned"]),
                 "PS<SmCalTuned": f"{int((M['Point-SAT'] < M['SMOTE+cal+tuned']).sum())}/{N}",
                 "PS vs SmSAT (2-sided p)": round(two("SMOTE+cal+SAT"), 3), "PS vs BBSE (2-sided p)": round(two("BBSE-SAT"), 3),
                 "gate fallback": round(T.gate_reject.mean(), 3),
                 "Robust worse/better than PS": f"{int((M['Robust-SAT'] > M['Point-SAT'] + 1e-12).sum())}/{int((M['Robust-SAT'] < M['Point-SAT'] - 1e-12).sum())}",
                 "EM err": round((T.em_prior - T.achieved_prev).abs().mean(), 4),
                 "BBSE err": round((T.bbse_prior - T.achieved_prev).abs().mean(), 4)})
pd.set_option("display.width", 260); pd.set_option("display.max_columns", 30)
S = pd.DataFrame(rows); print(S.to_string(index=False)); print()
print("Average rank per method (lower is better):"); R = pd.DataFrame(ranks).T[METHODS]; print(R.to_string()); print()
print("Mean excess cost x1e3 per method:"); E = pd.DataFrame(means).T[METHODS]; print(E.to_string())
S.to_csv(os.path.join(J, "results", "config_summary.csv"), index=False)
R.to_csv(os.path.join(J, "results", "config_ranks.csv")); E.to_csv(os.path.join(J, "results", "config_mean_excess.csv"))

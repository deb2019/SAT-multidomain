import sys, glob, json
import numpy as np, pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon
J = sys.argv[1]
METHODS = ["Fixed-0.5", "SMOTE@0.5", "SMOTE+cal+tuned", "Source-tuned",
           "BBSE-SAT", "SMOTE+cal+SAT", "Point-SAT", "Robust-SAT"]
Q05 = {6: 2.850, 7: 2.949, 8: 3.031}
T = pd.concat([pd.read_csv(f) for f in glob.glob(J + "/results/trials/*.csv")], ignore_index=True)
pd.set_option("display.width", 250)

M = T.groupby("id")[METHODS].mean()               # dataset-level mean excess cost
A = T.groupby("id")[["abs_" + k for k in METHODS]].mean().rename(columns=lambda c: c[4:])
dom = T.groupby("id")["domain"].first()
N, k = len(M), len(METHODS)
rk = M.rank(axis=1).mean()
st, p = friedmanchisquare(*[M[c].values for c in METHODS])
CD = Q05[k] * np.sqrt(k * (k + 1) / (6 * N))
print(f"N={N} datasets, {len(T)} trials;  Friedman chi2={st:.2f}  p={p:.2e};  Nemenyi CD(k={k})={CD:.2f}\n")
S = pd.DataFrame({"mean abs cost": A.mean().round(4), "mean excess x1e3": (M.mean() * 1000).round(2),
                  "median excess x1e3": (M.median() * 1000).round(2), "avg rank": rk.round(2)})
print(S.sort_values("avg rank").to_string(), "\n")

def holm(raw):
    it = sorted(raw.items(), key=lambda x: x[1]); m = len(it); run = 0; out = {}
    for i, (kk, pp) in enumerate(it): run = max(run, pp * (m - i)); out[kk] = min(1, run)
    return out
def wil(ref, others, Mx):
    raw = {b: wilcoxon(Mx[ref], Mx[b], alternative="less").pvalue for b in others}
    adj = holm(raw)
    return pd.DataFrame({"raw p": raw, "Holm p": adj,
                         "wins": {b: int((Mx[ref] < Mx[b]).sum()) for b in others},
                         "ties": {b: int((np.isclose(Mx[ref], Mx[b])).sum()) for b in others},
                         "N": len(Mx)})
others = [m for m in METHODS if m != "Point-SAT"]
print("Point-SAT vs each method, dataset-level one-sided Wilcoxon (all 31), Holm:")
print(wil("Point-SAT", others, M).round(4).to_string(), "\n")

print("Per-domain mean excess cost x1e3 (and Point-SAT wins vs Source-tuned / vs SMOTE+cal+SAT):")
rows = []
for d, ids in dom.groupby(dom).groups.items():
    Md = M.loc[ids]
    r = {"domain": d, "N": len(ids)}; r.update((Md.mean() * 1000).round(1).to_dict())
    r["PS<Src"] = f"{int((Md['Point-SAT'] < Md['Source-tuned']).sum())}/{len(ids)}"
    r["PS<SmSAT"] = f"{int((Md['Point-SAT'] < Md['SMOTE+cal+SAT']).sum())}/{len(ids)}"
    rows.append(r)
print(pd.DataFrame(rows).to_string(index=False), "\n")

print("Per shift level, mean excess x1e3 over datasets that support it:")
P = T.groupby(["m", "id"])[METHODS].mean().groupby("m").mean() * 1000
P.insert(0, "cov", T.groupby("m")["id"].nunique())
print(P.round(2).to_string(), "\n")

T["err_em"] = (T.em_prior - T.achieved_prev).abs(); T["err_bbse"] = (T.bbse_prior - T.achieved_prev).abs()
T["err_smem"] = (T.smote_em_prior - T.achieved_prev).abs()
print("Prior-estimation error |pi_hat - true| (mean over trials):",
      T[["err_em", "err_bbse", "err_smem"]].mean().round(4).to_dict())
e = T.groupby("id")[["err_em", "err_bbse"]].mean()
print(f"EM more accurate than BBSE on {(e.err_em < e.err_bbse).sum()}/{len(e)} datasets; "
      f"Wilcoxon p={wilcoxon(e.err_em, e.err_bbse, alternative='less').pvalue:.2e}")
print("Robust-SAT fallback rate overall:", round(T.gate_reject.mean(), 3),
      "| datasets with any fallback:", int((T.groupby('id').gate_reject.mean() > 0).sum()))
print("Robust-SAT worse than Point-SAT on", int((M['Robust-SAT'] > M['Point-SAT'] + 1e-12).sum()), "datasets;",
      "better on", int((M['Robust-SAT'] < M['Point-SAT'] - 1e-12).sum()))
print("\nPer-dataset mean excess x1e3:")
out = (M * 1000).round(2); out.insert(0, "domain", dom); out.insert(1, "AUC", T.groupby("id").auc.first().round(3))
print(out.sort_values(["domain", "AUC"]).to_string())
json.dump({"N": N, "friedman": {"chi2": st, "p": p, "CD": CD}, "summary": S.to_dict(),
           "per_dataset_x1e3": out.drop(columns=["domain"]).to_dict(orient="index")},
          open(J + "/results/summary.json", "w"), indent=1, default=float)
out.to_csv(J + "/results/per_dataset_excess_x1e3.csv")

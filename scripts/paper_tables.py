"""Compute every table for the journal manuscript from the saved trial files -> paper_tables.json"""
import sys, glob, os, json
import numpy as np, pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon
J = sys.argv[1]; O = sys.argv[2]
RES = os.path.join(O, "results")
METHODS = ["Fixed-0.5", "SMOTE@0.5", "SMOTE+cal+tuned", "Source-tuned", "BBSE-SAT", "SMOTE+cal+SAT", "Point-SAT", "Robust-SAT"]
Q05 = {8: 3.031}
def load(sub): return pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(RES, sub, "*.csv"))], ignore_index=True)
def holm(raw):
    it = sorted(raw.items(), key=lambda x: x[1]); m = len(it); run = 0; out = {}
    for i, (k, p) in enumerate(it): run = max(run, p*(m-i)); out[k] = min(1, run)
    return out
out = {}
T = load("trials"); M = T.groupby("id")[METHODS].mean(); A = T.groupby("id")[["abs_"+k for k in METHODS]].mean().rename(columns=lambda c: c[4:])
dom = T.groupby("id").domain.first(); N = len(M)
rk = M.rank(axis=1).mean(); st, p = friedmanchisquare(*[M[c].values for c in METHODS])
CD = Q05[8]*np.sqrt(8*9/(6*N))
out["main"] = {"N": N, "trials": len(T), "friedman_chi2": st, "friedman_p": p, "CD": CD,
               "rows": [{"method": k, "abs": A[k].mean(), "excess": M[k].mean()*1000, "median": M[k].median()*1000, "rank": rk[k]} for k in METHODS]}
raw = {b: wilcoxon(M["Point-SAT"], M[b], alternative="less").pvalue for b in METHODS if b != "Point-SAT"}
adj = holm(raw)
out["wilcoxon"] = [{"vs": b, "wins": int((M["Point-SAT"] < M[b]).sum()), "raw": raw[b], "holm": adj[b]} for b in raw]
# two-sided for adaptive comparators
for b in ["SMOTE+cal+SAT", "BBSE-SAT"]:
    out.setdefault("two_sided", {})[b] = wilcoxon(M["Point-SAT"], M[b]).pvalue
# per domain
rows = []
for d in ["Cyber security", "Finance & fraud", "Medical", "Industrial & anomaly"]:
    ids = dom[dom == d].index; Md = M.loc[ids]
    r = {"domain": d, "N": len(ids)}
    for k in ["Fixed-0.5", "SMOTE+cal+tuned", "Source-tuned", "BBSE-SAT", "SMOTE+cal+SAT", "Point-SAT", "Robust-SAT"]: r[k] = Md[k].mean()*1000
    r["ps_src"] = f"{int((Md['Point-SAT'] < Md['Source-tuned']).sum())}/{len(ids)}"
    rows.append(r)
out["domain"] = rows
# per shift level
P = T.groupby(["m", "id"])[METHODS].mean().groupby("m").mean()*1000
cov = T.groupby("m").id.nunique()
out["shift"] = [{"m": float(m), "cov": int(cov[m]), **{k: float(P.loc[m, k]) for k in METHODS}} for m in P.index]
# prior estimation
T["e_em"] = (T.em_prior - T.achieved_prev).abs(); T["e_bb"] = (T.bbse_prior - T.achieved_prev).abs()
e = T.groupby("id")[["e_em", "e_bb"]].mean()
out["prior"] = {"em": T.e_em.mean(), "bbse": T.e_bb.mean(), "em_better": int((e.e_em < e.e_bb).sum()), "N": len(e),
                "p": wilcoxon(e.e_em, e.e_bb, alternative="less").pvalue, "bbse_nan": int(T.bbse_prior.isna().sum())}
pb = []
for sub, lab in [("trials_n200", 200), ("trials_n500", 500), ("trials_n1000", 1000), ("trials", 4000)]:
    t = load(sub); pb.append({"nmax": lab, "em": (t.em_prior - t.achieved_prev).abs().mean(),
                              "bbse": (t.bbse_prior - t.achieved_prev).abs().mean(),
                              "ps": t.groupby("id")["Point-SAT"].mean().mean()*1000,
                              "src": t.groupby("id")["Source-tuned"].mean().mean()*1000})
out["prior_by_batch"] = pb
# gate
g = T.groupby("id").gate_reject.mean()
out["gate"] = {"fallback": T.gate_reject.mean(), "datasets_any": int((g > 0).sum()), "max_rate": g.max(),
               "worse": int((M["Robust-SAT"] > M["Point-SAT"] + 1e-12).sum()), "better": int((M["Robust-SAT"] < M["Point-SAT"] - 1e-12).sum()),
               "by_shift": T.groupby("m").gate_reject.mean().round(3).to_dict()}
# robustness
out["config"] = pd.read_csv(os.path.join(RES, "config_summary.csv")).to_dict(orient="records")
out["config_excess"] = pd.read_csv(os.path.join(RES, "config_mean_excess.csv"), index_col=0).to_dict(orient="index")
out["config_ranks"] = pd.read_csv(os.path.join(RES, "config_ranks.csv"), index_col=0).to_dict(orient="index")
# natural
out["natural"] = pd.read_csv(os.path.join(RES, "natural_summary.csv")).to_dict(orient="records")
# datasets
Aud = pd.read_csv(os.path.join(J, "audit_final.csv"))
inc = Aud[Aud.decision == "include"].copy()
names = {"C01": "Malware (33-feature corpus)", "C03": "Phishing websites (large)", "C04": "Cyber intrusion (IDS)"}
inc["name"] = inc.apply(lambda r: names.get(r.id, r.get("source_name") if isinstance(r.get("source_name"), str) else r.file), axis=1)
src = pd.read_csv(os.path.join(J, "sources.csv")).set_index("id")
inc["name"] = [src.loc[i, "source_name"] if i in src.index and isinstance(src.loc[i, "source_name"], str) else names.get(i, f) for i, f in zip(inc.id, inc.file)]
inc["host"] = [src.loc[i, "source_host"] if i in src.index and isinstance(src.loc[i, "source_host"], str) else "" for i in inc.id]
auc_cal = T.groupby("id").auc.first()
out["datasets"] = [{"id": r.id, "domain": r.domain, "name": r.name, "file": r.file, "N": int(r.N), "features": int(r.features),
                    "prev": float(r.prevalence), "auc": float(auc_cal.get(r.id, np.nan)), "licence": r.licence, "host": r.host}
                   for r in inc.sort_values(["domain", "id"]).itertuples()]
out["excluded"] = Aud[Aud.decision == "exclude"][["id", "category", "file", "reason"]].fillna("").to_dict(orient="records")
out["per_dataset"] = {i: {k: float(M.loc[i, k]*1000) for k in METHODS} for i in M.index}
# domain-classifier AUC under pure label shift (Table 11, Fig. 9 band)
LS = pd.read_csv(os.path.join(RES, "domain_auc_label_shift.csv")); rows = []
for c, lab in [("m=0.33", "1/3"), ("m=1.00", "1"), ("m=3.00", "3")]:
    v = LS[c].dropna(); rows.append({"m": lab, "n": int(len(v)), "mean": float(v.mean()), "max": float(v.max()),
                                     "ge06": int((v >= 0.6).sum()), "ge08": int((v >= 0.8).sum())})
cols = ["m=0.33", "m=1.00", "m=3.00"]
out["ls_auc"] = {"rows": rows, "min": float(np.nanmin(LS[cols].values)), "max": float(np.nanmax(LS[cols].values))}
# removed columns (Table A2) and sources/licences (Table A3)
drop = json.load(open(os.path.join(J, "drop_columns.json"))); inc = [d["id"] for d in out["datasets"]]
out["dropped"] = {k: v for k, v in drop.items() if k in inc}
S2 = pd.read_csv(os.path.join(J, "sources.csv")).set_index("id")
txt = lambda v: v if isinstance(v, str) else ""
out["sources"] = [{"id": i, "host": txt(S2.loc[i, "source_host"]), "url": txt(S2.loc[i, "source_url"]),
                   "licence": txt(S2.loc[i, "licence"])} for i in inc]
json.dump(out, open(os.path.join(J, "paper_tables.json"), "w"), indent=1, default=float)
print(json.dumps({k: out[k] for k in ["main", "prior", "gate", "two_sided"]}, indent=1, default=float)[:3000])
print(pd.DataFrame(out["domain"]).round(1).to_string())
print(pd.DataFrame(out["prior_by_batch"]).round(4).to_string())
print(len(out["datasets"]), "datasets;", len(out["excluded"]), "excluded")

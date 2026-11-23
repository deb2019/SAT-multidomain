"""Domain-classifier AUC under PURE label shift (controlled subsampling keeps p(x|y) fixed).
For every retained dataset: reference split vs target batches at m in {1/3, 1, 3} (seed 0)."""
import sys, os, json, warnings
import numpy as np, pandas as pd
from numpy.random import default_rng
from sklearn.model_selection import train_test_split
from joblib import Parallel, delayed
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from benchmark_multi import build
from natural_shift import domain_auc
J = sys.argv[1]
A = pd.read_csv(os.path.join(J, "audit_final.csv")); A = A[A.decision == "include"]
DROP = json.load(open(os.path.join(J, "drop_columns.json")))
paths = {}
for root, _, fs in os.walk(os.path.join(J, "data")):
    for f in fs: paths.setdefault(f, os.path.join(root, f))
def one(r):
    df = pd.read_csv(paths[r.file], low_memory=False); df.columns = [str(c).strip() for c in df.columns]
    df = df.drop(columns=[c for c in DROP.get(r.id, []) if c in df.columns]); df = df[df[r.target].notna()]
    y = (pd.to_numeric(df[r.target], errors="coerce") == float(r.minority_value)).astype(int).values
    X = df.drop(columns=[r.target]).apply(pd.to_numeric, errors="coerce").values
    X = X[:, ~np.all(np.isnan(X), axis=0)]
    Xs, Xp, ys, yp = train_test_split(X, y, test_size=.30, stratify=y, random_state=0)
    Xtr, Xrest, ytr, yrest = train_test_split(Xs, ys, test_size=.40, stratify=ys, random_state=0)
    Xca, Xrf, yca, yrf = train_test_split(Xrest, yrest, test_size=.50, stratify=yrest, random_state=0)
    PO = np.where(yp == 1)[0]; NE = np.where(yp == 0)[0]; out = {"id": r.id, "domain": r.domain}
    for m in [1/3, 1.0, 3.0]:
        pit = m * y.mean()
        sel = build(PO, NE, pit, default_rng(97), mx=4000) if 0 < pit < 1 else None
        out[f"m={m:.2f}"] = domain_auc(Xrf, Xp[sel]) if sel is not None else np.nan
    return out
R = pd.DataFrame(Parallel(n_jobs=2)(delayed(one)(r) for r in A.itertuples()))
R.to_csv(os.path.join(J, "results", "domain_auc_label_shift.csv"), index=False)
pd.set_option("display.width", 200)
print(R.round(3).to_string(index=False))
print("\nmean:", R[[c for c in R.columns if c.startswith("m=")]].mean().round(3).to_dict())
print("max :", R[[c for c in R.columns if c.startswith("m=")]].max().round(3).to_dict())
print("datasets with AUC>=0.6 at m=3:", int((R["m=3.00"] >= 0.6).sum()), "of", int(R["m=3.00"].notna().sum()),
      "| >=0.8:", int((R["m=3.00"] >= 0.8).sum()))

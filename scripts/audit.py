"""Audit every prepared dataset for the multi-domain SAT study."""
import sys, os, re, warnings
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
warnings.filterwarnings("ignore")

D = sys.argv[1]
import json
DROP = json.load(open(D + "/drop_columns.json"))
src = pd.read_csv(D + "/sources.csv")
TARGET = {"D000": "Label", "D001": "class", "D002": "status", "D003": "data_stolen", "D004": "Result",
          "D005": "Label", "D006": "label", "D007": "label", "D008": "target", "D012": "Category",
          "D014": "ApprovalStatus", "D019": "Loan_Status", "D020": "good_bad_flag", "D021": "loan_status",
          "D023": "Loan_Status", "D038": "insuranceclaim", "D039": "fraud_reported", "D042": "CARAVAN",
          "D044": "Churn", "D046": "y", "D048": "Churn", "D049": "Revenue", "D050": "fraud", "D051": "IsFraud",
          "D052": "is_fraud", "D053": "Class", "D054": "fraud", "D056": "Response", "D058": "Diagnosis",
          "D059": "Status", "D060": "Diabetes_binary", "D061": "Label", "D062": "long_stay", "D063": "target",
          "D064": "Target_High_Admission", "D065": "RiskLabel_Binary", "D067": "status", "D068": "At_Risk",
          "D069": "Stunting_binary", "D070": "target", "D071": "NSP", "D073": None, "D074": "label",
          "D075": "Machine failure", "D076": "Label", "D077": "class", "D078": "outcome", "D079": "Class",
          "D080": "target", "D081": "Label", "D082": "PotentialFraud", "C01": "Label", "C02": "Label", "C03": "Label", "C04": "Label"}
BROKEN = {"D013": "file truncated inside the archive", "D057": "file truncated inside the archive"}
SHIFT_PAT = re.compile(r"(date|time|year|month|day|week|quarter|period|region|state|country|city|location|"
                       r"hospital|site|centre|center|zip|district|province|county|branch|store|cycle|wave)", re.I)

paths = {}
for root, _, fs in os.walk(D + "/data"):
    for f in fs:
        if f.lower().endswith(".csv"): paths[f] = os.path.join(root, f)

def feasible(npos_pool, nneg_pool, pi0):
    out = []
    for m in [1/3, 1/2, 1, 2, 3]:
        pi = m * pi0
        if not 0 < pi < 1: out.append(False); continue
        n = min(npos_pool / pi, nneg_pool / (1 - pi), 4000); n1 = int(round(pi * n)); n0 = int(round(n - n1))
        out.append(n1 >= 20 and n0 >= 20)
    return out

rows = []
for _, r in src.iterrows():
    rec = {"id": r.id, "category": r.category, "file": r.prepared_file, "licence": r.licence,
           "redistribution": r.redistribution, "author_min_auc": r.min_validation_auc}
    if r.id in BROKEN: rec["status"] = "EXCLUDE: " + BROKEN[r.id]; rows.append(rec); continue
    p = paths.get(r.prepared_file)
    if p is None: rec["status"] = "MISSING: not in uploaded archives"; rows.append(rec); continue
    t = TARGET.get(r.id)
    df = pd.read_csv(p, low_memory=False); df.columns = [str(c).strip() for c in df.columns]
    df = df.drop(columns=[c for c in DROP.get(r.id, []) if c in df.columns])   # author-listed / index columns
    if t is None or t not in df.columns:
        rec["status"] = "EXCLUDE: no usable binary target"; rec["N"] = len(df); rows.append(rec); continue
    df = df[df[t].notna()]
    vc = df[t].value_counts()
    rec["target"] = t; rec["N"] = len(df); rec["n_classes"] = len(vc)
    if len(vc) != 2:
        rec["status"] = f"EXCLUDE: target has {len(vc)} classes {list(vc.index[:5])}"; rows.append(rec); continue
    minority = vc.idxmin(); y = (df[t] == minority).astype(int).values
    rec["minority_value"] = minority; rec["prevalence"] = round(y.mean(), 4); rec["n_pos"] = int(y.sum())
    X = df.drop(columns=[t])
    obj = [c for c in X.columns if X[c].dtype == object]
    rec["non_numeric_cols"] = len(obj)
    Xn = X.apply(pd.to_numeric, errors="coerce")
    Xn = Xn.loc[:, Xn.notna().mean() > 0.5]
    rec["features"] = Xn.shape[1]
    rec["dup_rows_frac"] = round(df.duplicated().mean(), 3)
    rec["shift_cols"] = "; ".join([c for c in df.columns if SHIFT_PAT.search(c)][:8])
    # single-feature leakage screen
    Xi = Xn.fillna(Xn.median()); best, bestc = 0.5, ""
    for c in Xi.columns:
        v = Xi[c].values
        if np.nanstd(v) == 0: continue
        a = roc_auc_score(y, v); a = max(a, 1 - a)
        if a > best: best, bestc = a, c
    rec["max_single_auc"] = round(best, 4); rec["max_single_col"] = bestc[:40]
    # base logistic AUC on a 70/30 split
    try:
        Xtr, Xte, ytr, yte = train_test_split(Xn.values, y, test_size=.3, stratify=y, random_state=0)
        imp = SimpleImputer(strategy="median").fit(Xtr); sc = StandardScaler().fit(imp.transform(Xtr))
        lr = LogisticRegression(max_iter=3000).fit(sc.transform(imp.transform(Xtr)), ytr)
        rec["lr_auc"] = round(roc_auc_score(yte, lr.predict_proba(sc.transform(imp.transform(Xte)))[:, 1]), 4)
    except Exception as e:
        rec["lr_auc"] = np.nan; rec["note"] = str(e)[:60]
    npos, nneg = y.sum(), len(y) - y.sum()
    f = feasible(0.3 * npos, 0.3 * nneg, y.mean()); rec["feasible_levels"] = sum(f)
    rec["status"] = "audited"
    rows.append(rec); print(r.id, rec["prevalence"], rec.get("lr_auc"), rec["max_single_auc"], flush=True)

A = pd.DataFrame(rows)
A.to_csv(D + "/audit_raw.csv", index=False)
print("saved", len(A))

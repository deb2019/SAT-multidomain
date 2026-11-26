"""
Natural-shift case studies (no subsampling: target batches keep their real prevalence).
  fraud     simulated (Sparkov) card transactions: train on fraudTrain.csv (2019-01-01..2020-06-21), test each month of fraudTest.csv
  bank      bank telemarketing, rows in chronological order: first 50% = source, rest = 5 consecutive blocks
  bank_nomacro  same, without the five macro-economic columns (calendar-time proxies)
  diabetes  diabetes-130 readmission <30 days, first encounter per patient in encounter order:
            first 50% = source, rest = 5 consecutive blocks
Same source protocol as the benchmark (60/20/20 train/calibration/reference, isotonic, r=5).
Usage: python natural_shift.py <journal_dir> <case> [--model lr|xgb]
"""
import sys, os, json, argparse, warnings
import numpy as np, pandas as pd
from numpy.random import default_rng
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split, cross_val_score
from scipy.stats import wilcoxon
R = 5.0; GRID = np.linspace(0.001, 0.999, 999); BINS = np.linspace(0, 1, 21)
MULT = [1/3, 1/2, 1.0, 2.0, 3.0]; K = 10
METHODS = ["Fixed-0.5", "SMOTE@0.5", "SMOTE+cal+tuned", "Source-tuned",
           "BBSE-SAT", "SMOTE+cal+SAT", "Point-SAT", "Robust-SAT"]

def sat(ps, pt, r):
    w1 = pt/ps; w0 = (1-pt)/(1-ps); return w0/(w0 + r*w1)
def pc(p, ps, pt):
    w1 = pt/ps; w0 = (1-pt)/(1-ps); n = w1*p; return n/(n + w0*(1-p))
def em(p, ps, n=400, tol=1e-10):
    pt = ps
    for _ in range(n):
        nw = float(np.mean(pc(p, ps, pt)))
        if abs(nw-pt) < tol: break
        pt = nw
    return float(np.clip(pt, 1e-3, 1-1e-3))

def ecost(y, pr, r):
    fp = np.sum((pr == 1) & (y == 0)); fn = np.sum((pr == 0) & (y == 1)); return (fp + r*fn)/len(y)
def tuned(p, y, r):
    return GRID[int(np.argmin([ecost(y, (p >= t).astype(int), r) for t in GRID]))]

def make_model(kind, C):
    if kind == "xgb":
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
                             tree_method="hist", n_jobs=1, random_state=0, verbosity=0)
    return LogisticRegression(max_iter=3000, C=C)


# ------------------------------------------------------------------ loaders
def load_fraud(J):
    cats = None
    def prep(df):
        nonlocal cats
        dt = pd.to_datetime(df.trans_date_trans_time); dob = pd.to_datetime(df.dob)
        X = pd.DataFrame({"log_amt": np.log1p(df.amt), "hour": dt.dt.hour, "dow": dt.dt.dayofweek,
                          "age": (dt - dob).dt.days/365.25, "gender": (df.gender == "M").astype(int),
                          "log_pop": np.log1p(df.city_pop), "lat": df.lat, "long": df.long,
                          "dist": np.hypot(df.lat - df.merch_lat, df.long - df.merch_long)})
        X["night"] = ((X.hour >= 22) | (X.hour <= 3)).astype(int)
        if cats is None: cats = sorted(df.category.unique())
        for c in cats: X["cat_" + c] = (df.category == c).astype(int)
        return X.values.astype(float), df.is_fraud.values.astype(int), dt.dt.strftime("%Y-%m").values
    tr = pd.read_csv(find(J, "fraudTrain.csv")); te = pd.read_csv(find(J, "fraudTest.csv"))
    Xs, ys, _ = prep(tr); Xt, yt, mon = prep(te)
    batches = [(m, Xt[mon == m], yt[mon == m]) for m in sorted(np.unique(mon))]
    return Xs, ys, batches

def find(J, name):
    for root, _, fs in os.walk(os.path.join(J, "data")):
        if name in fs: return os.path.join(root, name)
    raise FileNotFoundError(f"{name} not found under {J}/data (see data/README.md)")

MACRO = ["emp.var.rate", "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed"]
def load_bank(J, drop_macro=False):
    b = pd.read_csv(find(J, "pre_bank_marketing.csv"))            # rows are in chronological order
    if drop_macro: b = b.drop(columns=MACRO)                       # remove calendar-time proxies
    y = b.pop("y").values.astype(int); X = b.apply(pd.to_numeric, errors="coerce").values
    h = len(b)//2; rest = np.array_split(np.arange(h, len(b)), 5)
    return X[:h], y[:h], [(f"block {i+1}", X[ix], y[ix]) for i, ix in enumerate(rest)]

def icd_group(s):
    s = str(s)
    if s.startswith(("V", "E")) or s == "?": return "other"
    try: v = float(s)
    except ValueError: return "other"
    if 390 <= v <= 459 or int(v) == 785: return "circulatory"
    if 460 <= v <= 519 or int(v) == 786: return "respiratory"
    if 520 <= v <= 579 or int(v) == 787: return "digestive"
    if int(v) == 250: return "diabetes"
    if 800 <= v <= 999: return "injury"
    if 710 <= v <= 739: return "musculoskeletal"
    if 580 <= v <= 629 or int(v) == 788: return "genitourinary"
    if 140 <= v <= 239: return "neoplasms"
    return "other"

def load_diabetes(J):
    d = pd.read_csv(find(J, "diabetic_data.csv"))
    d = d.sort_values("encounter_id").drop_duplicates("patient_nbr")
    d = d[~d.discharge_disposition_id.isin([11, 13, 14, 19, 20, 21])]          # expired / hospice
    y = (d.readmitted == "<30").astype(int).values
    num = ["time_in_hospital", "num_lab_procedures", "num_procedures", "num_medications", "number_outpatient",
           "number_emergency", "number_inpatient", "number_diagnoses"]
    X = d[num].copy()
    X["age"] = d.age.str.extract(r"\[(\d+)").astype(float)[0] + 5
    X["male"] = (d.gender == "Male").astype(int)
    X["change"] = (d.change == "Ch").astype(int); X["dm_med"] = (d.diabetesMed == "Yes").astype(int)
    cat = {"race": d.race, "adm_type": d.admission_type_id.astype(str), "disch": d.discharge_disposition_id.astype(str),
           "adm_src": d.admission_source_id.astype(str), "a1c": d.A1Cresult, "glu": d.max_glu_serum,
           "insulin": d.insulin, "metformin": d.metformin, "diag1": d.diag_1.map(icd_group),
           "spec": d.medical_specialty.where(d.medical_specialty.isin(d.medical_specialty.value_counts().index[:10]), "other")}
    for k, s in cat.items():
        top = s.value_counts().index[:12]
        for v in top[1:]: X[f"{k}_{v}"] = (s == v).astype(int)
    X = X.values.astype(float); h = len(y)//2; rest = np.array_split(np.arange(h, len(y)), 5)
    return X[:h], y[:h], [(f"block {i+1}", X[ix], y[ix]) for i, ix in enumerate(rest)]


def evaluate(Xs, ys, batches, model="lr"):
    Xtr, Xrest, ytr, yrest = train_test_split(Xs, ys, test_size=.40, stratify=ys, random_state=0)
    Xca, Xrf, yca, yrf = train_test_split(Xrest, yrest, test_size=.50, stratify=yrest, random_state=0)
    imp = SimpleImputer(strategy="median").fit(Xtr); sc = StandardScaler().fit(imp.transform(Xtr))
    tf = lambda A: sc.transform(imp.transform(A))
    base = make_model(model, 1.0).fit(tf(Xtr), ytr)
    iso = IsotonicRegression(out_of_bounds="clip").fit(base.predict_proba(tf(Xca))[:, 1], yca)
    cal = lambda A: np.clip(iso.transform(base.predict_proba(tf(A))[:, 1]), 1e-6, 1-1e-6)
    prf = cal(Xrf); tau_src = tuned(prf, yrf, R)

    out=[]
    for name,Xt,yt in batches:
        pt=cal(Xt)
        tau=sat(ytr.mean(),em(pt,ytr.mean()),5)
        a=pt>=tau_src
        b=pt>=tau
        changed=(a!=b)
        out.append([name,len(yt),int(changed.sum()),int(yt[changed].sum()),int(tau<tau_src),ecost(yt,a,5),ecost(yt,b,5)])
    return pd.DataFrame(out,columns=['batch','N','D','K','lower','src','sat'])
if __name__=='__main__':
    case=sys.argv[1]
    loader={'diabetes':load_diabetes,'bank':load_bank,'fraud':load_fraud}
    res=evaluate(*loader[case]('.'))
    print(res.to_string(index=False))
    ref=pd.read_csv(case+'.csv')
    print('max cost discrepancies',np.abs(res.src-ref['abs_Source-tuned']).max(),np.abs(res.sat-ref['abs_Point-SAT']).max())
    res.to_csv(case+'_pilot.csv',index=False)


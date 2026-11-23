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
from imblearn.over_sampling import SMOTE
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from benchmark_multi import sat, em, bbse, ecost, tuned, gate, make_model, GRID

R = 5.0
METHODS = ["Fixed-0.5", "SMOTE+cal+tuned", "Source-tuned", "BBSE-SAT", "SMOTE+cal+SAT", "Point-SAT", "Robust-SAT"]

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

# ------------------------------------------------------------------ evaluation
def domain_auc(A, B, seed=0):
    rng = default_rng(seed); n = min(len(A), len(B), 3000)
    a = A[rng.choice(len(A), n, False)]; b = B[rng.choice(len(B), n, False)]
    Z = np.vstack([a, b]); d = np.r_[np.zeros(n), np.ones(n)]
    Z = StandardScaler().fit_transform(SimpleImputer(strategy="median").fit_transform(Z))
    return float(cross_val_score(LogisticRegression(max_iter=1000), Z, d, cv=3, scoring="roc_auc").mean())

def evaluate(Xs, ys, batches, model="lr"):
    Xtr, Xrest, ytr, yrest = train_test_split(Xs, ys, test_size=.40, stratify=ys, random_state=0)
    Xca, Xrf, yca, yrf = train_test_split(Xrest, yrest, test_size=.50, stratify=yrest, random_state=0)
    imp = SimpleImputer(strategy="median").fit(Xtr); sc = StandardScaler().fit(imp.transform(Xtr))
    tf = lambda A: sc.transform(imp.transform(A))
    base = make_model(model, 1.0).fit(tf(Xtr), ytr)
    iso = IsotonicRegression(out_of_bounds="clip").fit(base.predict_proba(tf(Xca))[:, 1], yca)
    cal = lambda A: np.clip(iso.transform(base.predict_proba(tf(A))[:, 1]), 1e-6, 1-1e-6)
    prf = cal(Xrf); tau_src = tuned(prf, yrf, R)
    Xr, yr = SMOTE(random_state=0).fit_resample(tf(Xtr), ytr)
    sm = make_model(model, 1.0).fit(Xr, yr)
    iso_sm = IsotonicRegression(out_of_bounds="clip").fit(sm.predict_proba(tf(Xca))[:, 1], yca)
    cal_sm = lambda A: np.clip(iso_sm.transform(sm.predict_proba(tf(A))[:, 1]), 1e-6, 1-1e-6)
    tau_sm = tuned(cal_sm(Xrf), yrf, R); ps = ytr.mean(); ps_c = yca.mean()
    rows = []
    for name, Xt, yt in batches:
        pt = cal(Xt); pts = cal_sm(Xt)
        pi_em, rej, avail = gate(prf, yrf, pt, ps, seed=0)
        tau_pt = sat(ps, pi_em, R); tau_rb = tau_src if rej else tau_pt
        pi_bb = bbse(prf, yrf, pt, tau_src); tau_bb = sat(ps, pi_bb, R) if np.isfinite(pi_bb) else tau_src
        pi_sm = em(pts, ps_c); tau_smsat = sat(ps_c, pi_sm, R)
        c = {"Fixed-0.5": ecost(yt, (pt >= .5).astype(int), R),
             "SMOTE+cal+tuned": ecost(yt, (pts >= tau_sm).astype(int), R),
             "Source-tuned": ecost(yt, (pt >= tau_src).astype(int), R),
             "BBSE-SAT": ecost(yt, (pt >= tau_bb).astype(int), R),
             "SMOTE+cal+SAT": ecost(yt, (pts >= tau_smsat).astype(int), R),
             "Point-SAT": ecost(yt, (pt >= tau_pt).astype(int), R),
             "Robust-SAT": ecost(yt, (pt >= tau_rb).astype(int), R)}
        c_or = min(ecost(yt, (pt >= t).astype(int), R) for t in GRID)
        row = {"batch": name, "n": len(yt), "true_prev": yt.mean(), "source_prev": ps, "em_prior": pi_em,
               "bbse_prior": pi_bb, "gate_reject": int(rej), "auc": roc_auc_score(yt, pt) if 0 < yt.sum() < len(yt) else np.nan,
               "domain_auc": domain_auc(Xrf, Xt), "cost_reference": c_or}
        for k in METHODS: row[k] = c[k] - c_or; row["abs_" + k] = c[k]
        rows.append(row); print(f"  {name}: prev {yt.mean():.4f} EM {pi_em:.4f} domainAUC {row['domain_auc']:.3f}", flush=True)
    return pd.DataFrame(rows)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("J"); ap.add_argument("case"); ap.add_argument("--model", default="lr")
    a = ap.parse_args()
    Xs, ys, batches = {"fraud": lambda: load_fraud(a.J), "bank": lambda: load_bank(a.J), "bank_nomacro": lambda: load_bank(a.J, True), "diabetes": lambda: load_diabetes(a.J)}[a.case]()
    print(f"{a.case}: source n={len(ys)} prev={ys.mean():.4f} features={Xs.shape[1]}; {len(batches)} target batches")
    df = evaluate(Xs, ys, batches, a.model)
    os.makedirs(os.path.join(a.J, "results", "natural"), exist_ok=True)
    df.to_csv(os.path.join(a.J, "results", "natural", f"{a.case}_{a.model}.csv"), index=False)
    pd.set_option("display.width", 250)
    show = df[["batch", "n", "true_prev", "em_prior", "bbse_prior", "domain_auc", "gate_reject"] + METHODS].copy()
    for k in METHODS: show[k] = (show[k]*1000).round(2)
    print(show.round(4).to_string(index=False))
    print("mean excess x1e3:", (df[METHODS].mean()*1000).round(2).to_dict())
    for b in ["Source-tuned", "SMOTE+cal+tuned", "Fixed-0.5", "SMOTE+cal+SAT", "BBSE-SAT"]:
        d = df["Point-SAT"] - df[b]
        p = wilcoxon(df["Point-SAT"], df[b], alternative="less").pvalue if (d != 0).any() else np.nan
        print(f"  Point-SAT vs {b:16s}: cheaper in {(d < 0).sum()}/{len(d)} batches, one-sided Wilcoxon p={p:.4f}")

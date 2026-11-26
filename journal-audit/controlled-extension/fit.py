"""
Multi-domain controlled label-shift benchmark (journal extension of the CICPS SAT study).
Same protocol as src/cyber_benchmark.py, plus two methods:
  BBSE-SAT        Eq. (3) with the Black-Box Shift Estimation prior (hard labels at tau_src)
  SMOTE+cal+SAT   matched comparator: SMOTE model, isotonic calibration, SAME EM prior
                  adaptation as Point-SAT (isolates resampling from adaptation)
Usage: python benchmark_multi.py <journal_dir> [--jobs 2] [--model lr|xgb] [--r 5] [--nmax 4000] [--minpos 20] [--tag name]
Writes <journal_dir>/results/trials/<id>.csv (one per dataset, resumable) and meta.
"""
import sys, os, json, warnings
import numpy as np, pandas as pd
from numpy.random import default_rng
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from joblib import Parallel, delayed
warnings.filterwarnings("ignore")

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
def bbse(pref, yref, pt, tau):
    hr = (pref >= tau).astype(int)
    tpr = hr[yref == 1].mean(); fpr = hr[yref == 0].mean(); q = (pt >= tau).mean()
    if tpr - fpr < 1e-3: return np.nan
    return float(np.clip((q - fpr)/(tpr - fpr), 1e-3, 1-1e-3))
def ecost(y, pr, r):
    fp = np.sum((pr == 1) & (y == 0)); fn = np.sum((pr == 0) & (y == 1)); return (fp + r*fn)/len(y)
def tuned(p, y, r):
    return GRID[int(np.argmin([ecost(y, (p >= t).astype(int), r) for t in GRID]))]
def gate(pref, yref, pt, ps, B=100, alpha=.05, floor=.05, seed=0):
    pos = pref[yref == 1]; neg = pref[yref == 0]
    if len(pos) < 20 or len(neg) < 20: return em(pt, ps), True, False
    rng = default_rng(seed)
    h1, _ = np.histogram(pos, BINS); h1 = h1/max(h1.sum(), 1)
    h0, _ = np.histogram(neg, BINS); h0 = h0/max(h0.sum(), 1)
    pi = em(pt, ps); obs, _ = np.histogram(pt, BINS); obs = obs/obs.sum()
    tv = .5*np.abs(obs - (pi*h1 + (1-pi)*h0)).sum()
    n = len(pt); null = np.empty(B)
    for b in range(B):
        n1 = rng.binomial(n, pi)
        s = np.concatenate([rng.choice(pos, n1, True), rng.choice(neg, n-n1, True)])
        pib = em(s, ps); ob, _ = np.histogram(s, BINS); ob = ob/ob.sum()
        null[b] = .5*np.abs(ob - (pib*h1 + (1-pib)*h0)).sum()
    return pi, bool(tv > np.percentile(null, 100*(1-alpha)) and tv > floor), True
def build(pos, neg, pi, rng, mx=4000, mp=20, mn=20):
    n = min(len(pos)/pi, len(neg)/(1-pi), mx); n1 = int(round(pi*n)); n0 = int(round(n-n1))
    if n1 < mp or n0 < mn or n1 > len(pos) or n0 > len(neg): return None
    return np.concatenate([rng.choice(pos, n1, False), rng.choice(neg, n0, False)])

def make_model(kind, C):
    if kind == "xgb":
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
                             tree_method="hist", n_jobs=1, random_state=0, verbosity=0)
    return LogisticRegression(max_iter=3000, C=C)

def run(rec, out_dir, R=5.0, model="lr", nmax=4000, minpos=20):
    out = os.path.join(out_dir, f"{rec['id']}.csv")
    if os.path.exists(out): return rec["id"], "cached"
    df = pd.read_csv(rec["path"], low_memory=False); df.columns = [str(c).strip() for c in df.columns]
    df = df.drop(columns=[c for c in rec.get("drop", []) if c in df.columns])   # author-listed / index columns
    t = rec["target"]; df = df[df[t].notna()]
    y = (df[t].astype(str) == str(rec["minority_value"])).astype(int).values
    if y.mean() > 0.5 or y.sum() == 0:      # string formatting mismatch guard
        y = (pd.to_numeric(df[t], errors="coerce") == float(rec["minority_value"])).astype(int).values
    X = df.drop(columns=[t]).apply(pd.to_numeric, errors="coerce").values
    pi0 = y.mean()
    Xs, Xp, ys, yp = train_test_split(X, y, test_size=.30, stratify=y, random_state=0)
    Xtr, Xrest, ytr, yrest = train_test_split(Xs, ys, test_size=.40, stratify=ys, random_state=0)
    Xca, Xrf, yca, yrf = train_test_split(Xrest, yrest, test_size=.50, stratify=yrest, random_state=0)
    imp = SimpleImputer(strategy="median").fit(Xtr); sc = StandardScaler().fit(imp.transform(Xtr))
    tf = lambda A: sc.transform(imp.transform(A)); C = 0.05 if X.shape[1] > 150 else 1.0
    base = make_model(model, C).fit(tf(Xtr), ytr)
    iso = IsotonicRegression(out_of_bounds="clip").fit(base.predict_proba(tf(Xca))[:, 1], yca)
    cal = lambda A: np.clip(iso.transform(base.predict_proba(tf(A))[:, 1]), 1e-6, 1-1e-6)
    prf = cal(Xrf); tau_src = tuned(prf, yrf, R)
    ps=ytr.mean()
    pos=np.flatnonzero(yp); neg=np.flatnonzero(yp==0)
    pool=cal(Xp); rows=[]
    for m in MULT:
        if not 0<m*pi0<1: continue
        for k in range(K):
            sel=build(pos,neg,m*pi0,default_rng(1000*k+int(m*97)),mx=nmax,mp=minpos,mn=minpos)
            if sel is None: continue
            y=yp[sel]; p=pool[sel]
            tau=sat(ps,em(p,ps),R)
            a=p>=tau_src; b=p>=tau; changed=a!=b
            rows.append([rec["id"],rec["domain"],m,k,len(y),changed.sum(),y[changed].sum(),int(tau<tau_src),ecost(y,a,R),ecost(y,b,R)])
    pd.DataFrame(rows,columns=["id","domain","shift","seed","N","D","K","lower","source","sat"]).to_csv(out,index=False)

if __name__=="__main__":
    from pathlib import Path
    A=pd.read_csv("audit_final.csv").query("decision=='include'")
    drop=json.load(open("drop_columns.json"));Path("counts").mkdir(exist_ok=True)
    for r in A.itertuples():
        path=Path("raw")/r.file
        if path.exists():
            run(dict(id=r.id,path=path,domain=r.domain,target=r.target,minority_value=int(r.minority_value),drop=drop.get(r.id,[])),"counts")

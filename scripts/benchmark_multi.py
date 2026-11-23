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
from imblearn.over_sampling import SMOTE
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
    Xr, yr = SMOTE(random_state=0).fit_resample(tf(Xtr), ytr)
    sm = make_model(model, C).fit(Xr, yr)
    iso_sm = IsotonicRegression(out_of_bounds="clip").fit(sm.predict_proba(tf(Xca))[:, 1], yca)
    cal_sm = lambda A: np.clip(iso_sm.transform(sm.predict_proba(tf(A))[:, 1]), 1e-6, 1-1e-6)
    tau_sm = tuned(cal_sm(Xrf), yrf, R)
    ps = ytr.mean(); ps_c = yca.mean(); PO = np.where(yp == 1)[0]; NE = np.where(yp == 0)[0]
    auc = roc_auc_score(yp, cal(Xp))
    rows = []
    for m in MULT:
        pit = m*pi0
        if not (0 < pit < 1): continue
        for k in range(K):
            rng = default_rng(1000*k + int(m*97)); sel = build(PO, NE, pit, rng, mx=nmax, mp=minpos, mn=minpos)
            if sel is None: continue
            Xt = Xp[sel]; yt = yp[sel]; pt = cal(Xt); pts = cal_sm(Xt)
            pi_em, rej, avail = gate(prf, yrf, pt, ps, seed=k)
            tau_pt = sat(ps, pi_em, R); tau_rb = tau_src if rej else tau_pt
            pi_bb = bbse(prf, yrf, pt, tau_src)
            tau_bb = sat(ps, pi_bb, R) if np.isfinite(pi_bb) else tau_src
            pi_sm = em(pts, ps_c); tau_smsat = sat(ps_c, pi_sm, R)
            c = {"Fixed-0.5": ecost(yt, (pt >= 0.5).astype(int), R),
                 "SMOTE@0.5": ecost(yt, (sm.predict_proba(tf(Xt))[:, 1] >= 0.5).astype(int), R),
                 "SMOTE+cal+tuned": ecost(yt, (pts >= tau_sm).astype(int), R),
                 "Source-tuned": ecost(yt, (pt >= tau_src).astype(int), R),
                 "BBSE-SAT": ecost(yt, (pt >= tau_bb).astype(int), R),
                 "SMOTE+cal+SAT": ecost(yt, (pts >= tau_smsat).astype(int), R),
                 "Point-SAT": ecost(yt, (pt >= tau_pt).astype(int), R),
                 "Robust-SAT": ecost(yt, (pt >= tau_rb).astype(int), R)}
            c_or = min(ecost(yt, (pt >= t).astype(int), R) for t in GRID)
            row = {"id": rec["id"], "domain": rec["domain"], "m": m, "seed": k, "n": len(yt),
                   "requested_prev": pit, "achieved_prev": float(yt.mean()), "source_prev": ps,
                   "em_prior": pi_em, "bbse_prior": pi_bb, "smote_em_prior": pi_sm,
                   "tau_src": tau_src, "tau_sat": tau_pt, "gate_reject": int(rej), "gate_available": int(avail),
                   "cost_reference": c_or, "auc": auc, "N": len(y), "features": X.shape[1], "prev0": pi0}
            for k2 in METHODS: row[k2] = c[k2] - c_or; row["abs_" + k2] = c[k2]
            rows.append(row)
    pd.DataFrame(rows).to_csv(out, index=False)
    return rec["id"], f"{len(rows)} trials, AUC {auc:.3f}"

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("J"); ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--model", default="lr"); ap.add_argument("--r", type=float, default=5.0)
    ap.add_argument("--nmax", type=int, default=4000); ap.add_argument("--minpos", type=int, default=20)
    ap.add_argument("--tag", default="")
    a = ap.parse_args(); J = a.J; nj = a.jobs
    A = pd.read_csv(os.path.join(J, "audit_final.csv")); A = A[A.decision == "include"]
    paths = {}
    for root, _, fs in os.walk(os.path.join(J, "data")):
        for f in fs: paths.setdefault(f, os.path.join(root, f))
    DROP = json.load(open(os.path.join(J, "drop_columns.json")))
    recs = [dict(id=r.id, domain=r.domain, path=paths[r.file], target=r.target, drop=DROP.get(r.id, []),
                 minority_value=(int(r.minority_value) if float(r.minority_value).is_integer() else r.minority_value))
            for r in A.itertuples()]
    out_dir = os.path.join(J, "results", "trials" + (f"_{a.tag}" if a.tag else "")); os.makedirs(out_dir, exist_ok=True)
    # largest first so the long jobs start early
    recs.sort(key=lambda r: -os.path.getsize(r["path"]))
    for rid, msg in Parallel(n_jobs=nj, verbose=0)(delayed(run)(r, out_dir, a.r, a.model, a.nmax, a.minpos) for r in recs):
        print(rid, msg, flush=True)

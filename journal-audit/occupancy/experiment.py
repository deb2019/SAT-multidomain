import warnings
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
        prior=em(pt,ytr.mean())
        tau=sat(ytr.mean(),prior,5)
        a=pt>=tau_src
        b=pt>=tau
        changed=(a!=b)
        out.append([name,len(yt),int(changed.sum()),int(yt[changed].sum()),int(tau<tau_src),ecost(yt,a,5),ecost(yt,b,5),prior,yt.mean(),tau_src,tau])
    return pd.DataFrame(out,columns=['batch','N','D','K','lower','src','sat','estimated_prior','target_prevalence','source_cutoff','sat_cutoff'])

from scipy.stats import hypergeom
from functools import lru_cache

@lru_cache(None)
def switch_probability(D,K,m,lower):
    if m==0: return 0.
    x=np.arange(m+1)
    if lower:
        reject=hypergeom.sf(x-1,D,D//6,m)<=0.05
    else:
        reject=hypergeom.cdf(x,D,(D+5)//6,m)<=0.05
    return float(hypergeom.pmf(x[reject],D,K,m).sum())

from zipfile import ZipFile
from pathlib import Path
import sys
data_file=Path(sys.argv[1] if len(sys.argv)>1 else 'occupancy.zip')
with ZipFile(data_file) as archive:
    data=pd.concat([pd.read_csv(archive.open(name),index_col=0) for name in sorted(archive.namelist()) if name.endswith('.txt')],ignore_index=True)
data['date']=pd.to_datetime(data.date)
data=data.sort_values('date').reset_index(drop=True)
assert data.date.is_unique
features=['Temperature','Humidity','Light','CO2','HumidityRatio']
source=data.iloc[:len(data)//2]; target=data.iloc[len(source):]
batches=[(str(day),rows[features].values,rows.Occupancy.values) for day,rows in target.groupby(target.date.dt.date)]
counts=evaluate(source[features].values,source.Occupancy.values,batches)
counts.to_csv('period_counts.csv',index=False)
results=[]
for row in counts.itertuples(index=False):
    N,D,K=row.N,row.D,row.K
    gain=(6*K-D)*(1 if row.lower else -1)
    assert abs((row.src-row.sat)*N-gain)<1e-8
    for budget in (10,25,50):
        b=min(budget,N)
        pt=switch_probability(D,K,min(b,D),row.lower)
        hs=np.arange(max(0,b-(N-D)),min(b,D)+1)
        pr=sum(w*switch_probability(D,K,int(h),row.lower) for h,w in zip(hs,hypergeom.pmf(hs,N,D,b)))
        results.append(dict(date=row.batch,N=N,D=D,K=K,budget=budget,targeted_labels=min(b,D),random_labels=b if D else 0,available_gain=gain,prob_targeted=pt,prob_random=pr,targeted_gain=gain*pt,random_gain=gain*pr))
out=pd.DataFrame(results)
out.to_csv('audit_results.csv',index=False)
print(len(source),len(target),source.date.min(),source.date.max(),target.date.min(),target.date.max())
print(counts.to_string(index=False));print(out.groupby('budget')[['targeted_gain','random_gain','targeted_labels','random_labels']].sum())

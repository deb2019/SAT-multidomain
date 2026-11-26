from pathlib import Path
from functools import lru_cache
import numpy as np
import pandas as pd
from scipy.stats import hypergeom
@lru_cache(12000)
def critical(D,lower):
    m=np.arange(1,min(50,D)+1)
    K0=D//6 if lower else (D+5)//6
    # smallest upper-tail rejection count, or largest lower-tail rejection count
    if lower:
        c=hypergeom.ppf(.95,D,K0,m).astype(int)+1
        c-=hypergeom.sf(c-2,D,K0,m)<=.05
    else:
        c=hypergeom.ppf(.05,D,K0,m).astype(int)-1
        c+=hypergeom.cdf(c+1,D,K0,m)<=.05
    return m,c
def compare(row):
    D=row.D; K=row.K; N=row.N; low=row.lower
    delta=row.source-row.sat
    assert abs(delta*N-(6*K-D)*(1 if low else -1))<1e-6
    if D:
        m,c=critical(D,low)
        p=hypergeom.sf(c-1,D,K,m) if low else hypergeom.cdf(c,D,K,m)
        probs=np.r_[0.,p]
    else: probs=[0.]
    for b in [10,25,50]:
        cap=min(b,N);h=np.arange(max(0,cap-(N-D)),min(cap,D)+1)
        rand=np.dot(hypergeom.pmf(h,N,D,cap),np.asarray(probs)[h])
        tar=probs[min(D,cap)]
        yield [*row,b,tar,rand,row.source-delta*tar,row.source-delta*rand]
d=pd.concat(pd.read_csv(f) for f in sorted(Path('counts').glob('*.csv')))
o=pd.DataFrame([a for r in d.itertuples(index=False,name='Period') for a in compare(r)],columns=[*d.columns,'budget','prob_targeted','prob_random','targeted_cost','random_cost'])
o.to_csv('batch_results.csv',index=False)
g=o.groupby(['id','domain','budget'])[['source','sat','targeted_cost','random_cost']].mean().reset_index();g.to_csv('dataset_results.csv',index=False)
print(d.groupby('domain').id.nunique(),len(d));print(g.groupby('budget')[['source','sat','targeted_cost','random_cost']].mean())
print('targeted lower than random',g.assign(win=g.targeted_cost<g.random_cost-1e-12).groupby('budget').win.sum())

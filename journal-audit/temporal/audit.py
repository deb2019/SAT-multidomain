"""Fixed-budget, one-sided exact tests on the realised target batch (r=5)."""
import numpy as np
import pandas as pd
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

rows=[]
for case in ['diabetes','bank','fraud']:
    periods=pd.read_csv(case+'_pilot.csv').merge(pd.read_csv(case+'.csv')[['batch','abs_Source-tuned','abs_Point-SAT','abs_Robust-SAT']],on='batch',validate='one_to_one')
    for t in periods.itertuples(index=False,name=None):
        batch,N,D,K,lower,src,sat,pubsrc,pubsat,gate=t
        units=(6*K-D)*(1 if lower else -1)
        assert abs((src-sat)*N-units)<1e-7
        assert max(abs(src-pubsrc),abs(sat-pubsat))<1e-9
        for budget in [10,25,50]:
            m=min(budget,D)
            target=switch_probability(D,K,m,lower)
            h=np.arange(max(0,budget-(N-D)),min(budget,D)+1)
            whole=sum(p*switch_probability(D,K,int(i),lower) for i,p in zip(h,hypergeom.pmf(h,N,D,budget)))
            rows.append(dict(case=case,batch=batch,N=N,D=D,K=K,lower=lower,budget=budget,labels_targeted=m,labels_random=budget,p_targeted=target,p_random=whole,benefit=src-sat,source=src,sat=sat,old_gate=gate,expected_targeted=src-target*(src-sat),expected_random=src-whole*(src-sat),oracle=min(src,sat)))
d=pd.DataFrame(rows)
d.to_csv('pilot_results.csv',index=False)
for b,g in d.groupby('budget'):
    print(b,g.groupby('case')[['expected_targeted','expected_random','source','sat','old_gate']].mean().round(6))
    print('benefit capture',[(g[g.benefit>1e-12][k].mean(),g[g.benefit< -1e-12][k].mean()) for k in ['p_targeted','p_random']])
print(d[d.budget==50][['case','batch','D','K','benefit','p_targeted','p_random']].round(5).to_string(index=False))

import pandas as pd,numpy as np
from scipy.stats import hypergeom
d=pd.read_csv('all_results.csv').query("group=='temporal' and method=='exact'").copy()
for i,r in d.iterrows():
    N,D,K,L,b=map(int,[r.N,r.D,r.K,r.lower,r.budget])
    pfull=prem=ps=0.
    for m in range(1,min(D,b)+1):
        pm=hypergeom.pmf(m,N,D,b)
        x=np.arange(m+1)
        q=hypergeom.pmf(x,D,K,m)*pm
        yes=(hypergeom.sf(x-1,D,D//6,m) if L else hypergeom.cdf(x,D,(D+5)//6,m))<=.05
        g=(2*L-1)*(6*x-m)
        ps+=q@yes; pfull+=(q@yes)*r.true_gain
        prem+=q@(yes*(r.true_gain-g))
    d.loc[i,['switch','full_gain','remaining_gain','labels','classification_cost']]=[ps,pfull,prem,b if D else 0,r.source-pfull/N]
d['method']='random_exact'
d.to_csv('random_temporal.csv',index=False)

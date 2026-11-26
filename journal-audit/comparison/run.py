from pathlib import Path
from zipfile import ZipFile
import subprocess
import pandas as pd,numpy as np
from scipy.stats import hypergeom
d=pd.read_csv('input_counts.csv')
assert len(d)==1816
s=d[['N','D','K']].copy(); s['sign']=np.where(d.lower,1,-1)
rows=subprocess.run(['./simulate'],input=s.to_csv(index=False,header=False,sep=' '),text=True,capture_output=True,check=True).stdout.splitlines()
records=[]
for row,mc in zip(d.to_dict('records'),rows):
 n,D,K,L=[int(row[i]) for i in ['N','D','K','lower']]
 sign=2*L-1; gain=sign*(6*K-D); assert abs(gain-(row['source']-row['sat'])*n)<1e-6
 for b,(pc,remc,lc,rem2,seenD,pw,rw,rw2,pz,rz,rz2) in zip([10,25,50],np.array(mc.split(),float).reshape(3,11)):
  m=min(D,b);x=np.arange(m+1)
  if D:
   p=hypergeom.pmf(x,D,K,m)
   test=(hypergeom.sf(x-1,D,D//6,m) if L else hypergeom.cdf(x,D,(D+5)//6,m))<=.05
  else: p=np.ones(1);test=np.array([False])
  sample=sign*(6*x-m)
  for name,sw,rem,lab,mcse in [('exact',p@test,p@(test*(gain-sample)),m,0),('empirical_same_sample',p@(sample>0),p@((sample>0)*(gain-sample)),m,0),('AMC_Wald_limit',pz,rz,lc,(max(0,rz2-rz**2)/3999)**.5),('AMC_Wald_5pct',pw,rw,lc,(max(0,rw2-rw**2)/3999)**.5),('AMC_cost_hard',pc,remc,lc,(max(0,rem2-remc**2)/3999)**.5)]:
   records.append(dict(**row,budget=b,method=name,switch=sw,classification_cost=row['source']-sw*gain/n,full_gain=sw*gain,remaining_gain=rem,labels=lab,remaining_gain_mcse=mcse,true_gain=gain))
r=pd.DataFrame(records);r.to_csv('all_results.csv',index=False)
c=r[r.group=='controlled'].copy()
c['harm']=c.switch.where(c.true_gain<0);c['benefit']=c.switch.where(c.true_gain>0)
# Cost dataset-macro; approval probabilities batch-macro conditional on true sign.
g=c.groupby(['budget','method','id']).classification_cost.mean().groupby(level=[0,1]).mean().to_frame().join(c.groupby(['budget','method'])[['harm','benefit']].mean())
print(g);g.to_csv('controlled_summary.csv')
t=r[r.group=='temporal'].groupby(['budget','method','id'])[['full_gain','remaining_gain','labels']].sum();print(t.loc[25]);t.to_csv('temporal_summary.csv')

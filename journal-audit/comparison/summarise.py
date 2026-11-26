import pandas as pd
r=pd.read_csv('all_results.csv')
t=pd.concat([r.query("group=='temporal'"),pd.read_csv('random_temporal.csv')])
g=t.groupby(['budget','method','id'])[['full_gain','remaining_gain','labels']].sum()
g['remaining_gain_per_label']=g.remaining_gain/g.labels
g.to_csv('all_temporal_gains.csv')
assert len(r.query("budget==25 and method=='exact'"))==1816
assert r.query("method=='exact' and true_gain<=0").switch.max()<=.0500000001
assert abs(g.loc[(25,'exact','occupancy'),'remaining_gain']-14)<1e-7

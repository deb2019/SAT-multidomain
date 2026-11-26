from pathlib import Path
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
root = Path(__file__).resolve().parent
o = root / 'figures'
o.mkdir(exist_ok=True)
teal='#126457'; navy='#223c53'; red='#a4402c'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'text.color':navy,'axes.labelcolor':navy,'xtick.color':navy,'ytick.color':navy})
def save(f,n):
 f.savefig(o/n,dpi=320,bbox_inches='tight',pad_inches=.08,facecolor='white');plt.close(f)
f,ax=plt.subplots(figsize=(7.2,2.0));ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
for l,r,col,txt in [(0,.31,'#ebeff1','Both predict 0'),(.31,.69,'#d3eae5','Predictions differ'),(.69,1,'#ebeff1','Both predict 1')]:
 ax.add_patch(Rectangle((l,.48),r-l,.23,facecolor=col,edgecolor='white'));ax.text((r+l)/2,.593,txt,ha='center',va='center',weight='bold',fontsize=10)
for x,lab in [(.31,'Proposed cutoff'),(.69,'Incumbent cutoff')]:
 ax.plot([x,x],[.43,.78],color=navy,lw=1.2);ax.text(x,.82,lab,ha='center',size=10)
ax.annotate('',xy=(.995,.42),xytext=(.003,.42),arrowprops=dict(arrowstyle='->',color=navy))
ax.text(.995,.34,'Higher score',ha='right',fontsize=9);ax.text(.5,.29,'Sample labels from this interval',ha='center',color=teal,weight='bold');ax.text(.5,.10,'For each changed decision: a positive saves r units; a negative costs 1 unit.',ha='center',fontsize=10)
save(f,'interval.png')
t=pd.read_csv(root / 'comparison/input_counts.csv').query("group=='temporal'").copy()
t['gain']=(2*t.lower-1)*(6*t.K-t.D)
assert np.allclose(t.gain,(t.source-t.sat)*t.N)
f,ax=plt.subplots(figsize=(7.4,2.25));ax.set(xlim=(-2.7,9.05),ylim=(-.6,4.05));ax.axis('off')
for n in range(9):ax.text(n+.5,3.75,str(n+1),ha='center',size=9)
for i,(k,l) in enumerate([('bank','Bank marketing'),('diabetes','Diabetes readmission'),('fraud','Simulated cards'),('occupancy','Office occupancy')]):
 y=3.05-i*.85;ax.text(-.15,y,l,ha='right',va='center',size=10)
 for j,g in enumerate(t[t.id==k].gain.astype(int)):
  ax.add_patch(Rectangle((j+.025,y-.35),.95,.7,facecolor=('#dfede8' if g>0 else '#fae7e0' if g<0 else '#eceff0'),edgecolor='none'));ax.text(j+.5,y,format(g,'+d') if g else '0',ha='center',va='center',weight='bold',fontsize=9,color=teal if g>0 else red if g<0 else navy)
ax.text(0,-.24,'Period within each series (earliest first)',ha='left',size=9)
save(f,'periods.png')
g=pd.read_csv(root / 'comparison/controlled_summary.csv')
f,axes=plt.subplots(1,2,figsize=(7.25,2.65))
for name,lab,marker,ls,color in [('exact','Exact audit','o','-',teal),('empirical_same_sample','Disagreement empirical','D',':','#495e9b'),('AMC_cost_hard','AMC empirical','s','--',red),('AMC_Wald_5pct','AMC Wald','^','-.','#665276')]:
 v=g[g.method==name]
 for ax,key in zip(axes,['harm','benefit']):ax.plot(v.budget,v[key]*100,marker=marker,ls=ls,label=lab,color=color,ms=4)
for ax,title,limit in zip(axes,['Harmful proposals approved (%)','Beneficial proposals approved (%)'],[26,100]):
 ax.set_title(title,fontsize=10.5,weight='bold');ax.set(xlabel='Distinct-label cap',xticks=[10,25,50],ylim=(0,limit));ax.grid(axis='y',alpha=.22);ax.spines[['right','top']].set_visible(False)
f.subplots_adjust(left=.06,right=.99,top=.87,bottom=.31,wspace=.26);f.legend(*axes[0].get_legend_handles_labels(),loc='lower center',ncol=4,fontsize=8,frameon=False)
save(f,'budget.png')

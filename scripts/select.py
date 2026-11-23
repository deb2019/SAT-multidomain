import sys, pandas as pd, numpy as np
D=sys.argv[1]; A=pd.read_csv(D+"/audit_raw.csv")
DOMAIN={"Cyber-Security":"Cyber security","Finance and Marketing":"Finance & fraud","Fraud / insurance":"Finance & fraud",
        "Medical":"Medical","Maintenance / other":"Industrial & anomaly"}
A["domain"]=A.category.map(DOMAIN)
A.loc[A.id=="D081","domain"]="Medical"          # MIT-BIH ECG is a clinical signal
def decide(r):
    if r.status!="audited": return "exclude", r.status.split(": ",1)[-1]
    why=[]
    if r.lr_auc<0.70: why.append(f"weak signal (AUC {r.lr_auc:.2f})")
    if r.lr_auc>=0.998: why.append(f"near-perfect separation (AUC {r.lr_auc:.4f})")
    if r.max_single_auc>=0.99: why.append(f"single-feature leakage ({r.max_single_col}, AUC {r.max_single_auc:.3f})")
    if r.feasible_levels<4: why.append(f"too few positives for the shift protocol ({int(r.feasible_levels)}/5 levels)")
    if r.dup_rows_frac>=0.20: why.append(f"{r.dup_rows_frac:.0%} duplicate rows")
    return ("exclude","; ".join(why)) if why else ("include","")
A[["decision","reason"]]=A.apply(lambda r: pd.Series(decide(r)),axis=1)
A.to_csv(D+"/audit_final.csv",index=False)
inc=A[A.decision=="include"]
print(inc.groupby("domain").size()); print("total included:",len(inc))
pd.set_option("display.width",220); pd.set_option("display.max_colwidth",70)
print(inc[["id","domain","file","N","prevalence","lr_auc","feasible_levels"]].sort_values(["domain","id"]).to_string(index=False))
print(); print(A[(A.decision=="exclude")&(A.status=="audited")][["id","file","reason"]].to_string(index=False))

"""Split results/SAT_trials_long.csv back into results/trials*/<dataset>.csv (the layout the analysis scripts read).
Usage: python scripts/expand_trials.py ."""
import sys, os
import pandas as pd
J = sys.argv[1] if len(sys.argv) > 1 else "."
L = pd.read_csv(os.path.join(J, "results", "SAT_trials_long.csv"))
for (sub, did), g in L.groupby(["config_dir", "id"]):
    d = os.path.join(J, "results", sub); os.makedirs(d, exist_ok=True)
    g.drop(columns=["config", "config_dir"]).to_csv(os.path.join(d, f"{did}.csv"), index=False)
print("expanded", L.config_dir.nunique(), "configurations,", L.id.nunique(), "datasets")

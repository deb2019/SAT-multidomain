"""Figures for the journal manuscript (reads paper_tables.json and the natural-shift CSVs)."""
import sys, os, json
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
J = sys.argv[1]; O = sys.argv[2]; F = os.path.join(J, "figs"); os.makedirs(F, exist_ok=True)
D = json.load(open(os.path.join(J, "paper_tables.json")))
METHODS = ["Fixed-0.5", "SMOTE@0.5", "SMOTE+cal+tuned", "Source-tuned", "BBSE-SAT", "SMOTE+cal+SAT", "Point-SAT", "Robust-SAT"]
COL = {"Fixed-0.5": "#9e9e9e", "SMOTE@0.5": "#b5cf6b", "SMOTE+cal+tuned": "#637939", "Source-tuned": "#e6853a",
       "BBSE-SAT": "#6baed6", "SMOTE+cal+SAT": "#3182bd", "Point-SAT": "#08306b", "Robust-SAT": "#c44e52"}
ADAPT = {"BBSE-SAT", "SMOTE+cal+SAT", "Point-SAT", "Robust-SAT"}
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})

# ---- Fig. 2  study design --------------------------------------------------
def box(ax, x, y, w, h, text, fc):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.03", fc=fc, ec="#333", lw=.8))
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=8)
def arrow(ax, x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="->", lw=.9, color="#333"))
fig, ax = plt.subplots(figsize=(9, 3.6)); ax.set_xlim(0, 10); ax.set_ylim(0, 4); ax.axis("off")
box(ax, 0.1, 2.6, 1.7, 1.0, "Source data\n(prevalence $\\pi_s$)", "#eef3f8")
box(ax, 2.2, 3.05, 1.9, 0.75, "Training (60%)\nscorer fit", "#f7f7f7")
box(ax, 2.2, 2.2, 1.9, 0.75, "Calibration (20%)\nisotonic", "#f7f7f7")
box(ax, 2.2, 1.35, 1.9, 0.75, "Reference (20%)\n$\\tau_{src}$, histograms", "#f7f7f7")
box(ax, 0.1, 0.25, 1.7, 1.0, "Target batch\n(unlabelled, $\\pi_t$)", "#fdf0e6")
box(ax, 4.6, 2.2, 2.0, 1.2, "Calibrated source\nposterior $p_s(x)$", "#eef3f8")
box(ax, 4.6, 0.25, 2.0, 1.2, "Prior estimate $\\hat\\pi_t$\nEM  |  BBSE", "#e8f1fa")
box(ax, 7.1, 2.2, 2.7, 1.2, "Threshold\n$\\tau(\\hat\\pi_t)=w_0/(w_0+r\\,w_1)$\n(optional diagnostic: fall back to $\\tau_{src}$)", "#e8f1fa")
box(ax, 7.1, 0.25, 2.7, 1.2, "Evaluation on target labels\ncost (Eq. 1), excess over best grid\nthreshold, Friedman / Wilcoxon", "#f3f3f3")
for y in (3.42, 2.57, 1.72): arrow(ax, 1.8, 3.1, 2.2, y)
arrow(ax, 4.1, 3.42, 4.6, 3.0); arrow(ax, 4.1, 2.57, 4.6, 2.75)
arrow(ax, 1.8, 0.75, 4.6, 0.85); arrow(ax, 5.6, 2.2, 5.6, 1.45)
arrow(ax, 6.6, 0.85, 7.1, 2.4); arrow(ax, 6.6, 2.8, 7.1, 2.8); arrow(ax, 4.1, 1.72, 7.1, 2.3)
arrow(ax, 8.45, 2.2, 8.45, 1.45)
ax.text(0.95, 1.55, "controlled: subsample to $m\\cdot\\pi_0$\nnatural: later period / month", ha="center", fontsize=7, color="#555")
fig.tight_layout(); fig.savefig(os.path.join(F, "fig2_design.png"), dpi=220); plt.close(fig)

# ---- Fig. 4  Nemenyi CD diagram --------------------------------------------
rk = {r["method"]: r["rank"] for r in D["main"]["rows"]}; CD = D["main"]["CD"]
order = sorted(METHODS, key=lambda k: rk[k])
fig, ax = plt.subplots(figsize=(8, 2.8)); ax.set_xlim(0.6, 8.4); ax.set_ylim(-0.2, 1.6); ax.axis("off")
ax.plot([1, 8], [1.2, 1.2], color="k", lw=1)
for v in range(1, 9):
    ax.plot([v, v], [1.2, 1.26], color="k", lw=1); ax.text(v, 1.32, str(v), ha="center", fontsize=8)
ax.plot([1, 1 + CD], [1.5, 1.5], color="k", lw=1.6); ax.text(1 + CD/2, 1.55, f"CD = {CD:.2f}", ha="center", fontsize=8)
left = order[:4]; right = order[4:]
for i, k in enumerate(left):
    y = 1.0 - i*0.24; ax.plot([rk[k], rk[k], 0.95], [1.2, y, y], color=COL[k], lw=1.2)
    ax.text(0.9, y, f"{k} ({rk[k]:.2f})", ha="right", va="center", fontsize=8)
for i, k in enumerate(right[::-1]):
    y = 1.0 - i*0.24; ax.plot([rk[k], rk[k], 8.05], [1.2, y, y], color=COL[k], lw=1.2)
    ax.text(8.1, y, f"({rk[k]:.2f}) {k}", ha="left", va="center", fontsize=8)
# cliques: methods whose rank difference < CD
cl = []; s = sorted(METHODS, key=lambda k: rk[k])
for i in range(len(s)):
    j = max(jj for jj in range(i, len(s)) if rk[s[jj]] - rk[s[i]] < CD)
    if j > i and not any(a <= i and j <= b for a, b in cl): cl.append((i, j))
for n, (i, j) in enumerate(cl):
    y = 1.1 - n*0.06; ax.plot([rk[s[i]] - .03, rk[s[j]] + .03], [y, y], color="k", lw=2.4, solid_capstyle="round")
fig.tight_layout(); fig.savefig(os.path.join(F, "fig4_cd.png"), dpi=220); plt.close(fig)

# ---- Fig. 5  excess cost vs shift --------------------------------------------
S = pd.DataFrame(D["shift"])
fig, ax = plt.subplots(figsize=(6.6, 4))
for k in METHODS:
    ax.plot(S.m, S[k], "o-" if k in ADAPT else "s--", color=COL[k], label=k, lw=1.5 if k in ADAPT else 1.1, ms=4)
ax.set_xscale("log"); ax.set_yscale("log"); ax.minorticks_off()
ax.set_xticks(S.m); ax.set_xticklabels(["1/3", "1/2", "1", "2", "3"])
ax.set_xlabel("shift multiplier  $\\pi_t/\\pi_0$"); ax.set_ylabel("mean excess cost (×10⁻³, log scale)")
ax.set_title(f"Controlled label shift, {D['main']['N']} datasets (r = 5, logistic scorer)", fontsize=9)
ax.legend(frameon=False, fontsize=7, ncol=2); ax.grid(alpha=.25, which="both")
fig.tight_layout(); fig.savefig(os.path.join(F, "fig5_shift.png"), dpi=220); plt.close(fig)

# ---- Fig. 6  per-domain -----------------------------------------------------
Dm = pd.DataFrame(D["domain"]); show = ["Fixed-0.5", "SMOTE+cal+tuned", "Source-tuned", "BBSE-SAT", "SMOTE+cal+SAT", "Point-SAT", "Robust-SAT"]
fig, ax = plt.subplots(figsize=(8, 3.6)); w = 0.11; x = np.arange(len(Dm))
for i, k in enumerate(show):
    ax.bar(x + (i - 3)*w, Dm[k], w, color=COL[k], label=k)
ax.set_yscale("log"); ax.set_xticks(x); ax.set_xticklabels([f"{d}\n(N={n})" for d, n in zip(Dm.domain, Dm.N)])
ax.set_ylabel("mean excess cost (×10⁻³, log)")
ax.legend(frameon=False, fontsize=7, ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.0))
ax.grid(axis="y", alpha=.25); fig.tight_layout(); fig.savefig(os.path.join(F, "fig6_domain.png"), dpi=220); plt.close(fig)

# ---- Fig. 7  robustness: rank heat map -------------------------------------
Rk = pd.DataFrame(D["config_ranks"]).T[METHODS]
fig, ax = plt.subplots(figsize=(8, 3.4))
im = ax.imshow(Rk.values, cmap="Blues_r", vmin=1, vmax=8, aspect="auto")
ax.set_xticks(range(8)); ax.set_xticklabels(METHODS, rotation=30, ha="right", fontsize=8)
ax.set_yticks(range(len(Rk))); ax.set_yticklabels(Rk.index, fontsize=8)
for i in range(Rk.shape[0]):
    for j in range(8):
        v = Rk.values[i, j]; ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7, color="white" if v < 3.5 else "black")
cb = fig.colorbar(im, ax=ax, fraction=.03); cb.set_label("average rank (1 = best)", fontsize=8)
ax.spines[:].set_visible(False); fig.tight_layout(); fig.savefig(os.path.join(F, "fig7_robustness.png"), dpi=220); plt.close(fig)

# ---- Fig. 8  batch size -------------------------------------------------------
B = pd.DataFrame(D["prior_by_batch"])
fig, (a, b) = plt.subplots(1, 2, figsize=(8, 3.2))
a.plot(B.nmax, B.em, "o-", color=COL["Point-SAT"], label="EM"); a.plot(B.nmax, B.bbse, "s--", color=COL["BBSE-SAT"], label="BBSE")
a.set_xscale("log"); a.set_xticks(B.nmax); a.set_xticklabels(B.nmax); a.minorticks_off()
a.set_xlabel("maximum target batch size"); a.set_ylabel("mean |estimated − true prior|"); a.legend(frameon=False); a.grid(alpha=.25)
b.plot(B.nmax, B.ps, "o-", color=COL["Point-SAT"], label="Point-SAT"); b.plot(B.nmax, B.src, "s--", color=COL["Source-tuned"], label="Source-tuned")
b.set_xscale("log"); b.set_xticks(B.nmax); b.set_xticklabels(B.nmax); b.minorticks_off()
b.set_xlabel("maximum target batch size"); b.set_ylabel("mean excess cost (×10⁻³)"); b.legend(frameon=False); b.grid(alpha=.25)
fig.tight_layout(); fig.savefig(os.path.join(F, "fig8_batch.png"), dpi=220); plt.close(fig)

# ---- Fig. 9  natural shifts -------------------------------------------------
sty = {"fraud": ("#2a7f62", "o", "card transactions, simulated (finance)"), "diabetes": ("#c44e52", "s", "diabetes readmission (medical)"),
       "bank": ("#4c72b0", "^", "bank campaign (marketing)"), "bank_nomacro": ("#8172b3", "v", "bank, macro columns removed")}
fig, ax = plt.subplots(figsize=(7, 4.2))
LS = pd.read_csv(os.path.join(J, "results", "domain_auc_label_shift.csv"))
lo, hi = np.nanmin(LS[["m=0.33", "m=1.00", "m=3.00"]].values), np.nanmax(LS[["m=0.33", "m=1.00", "m=3.00"]].values)
ax.axvspan(lo, hi, color="#dddddd", alpha=.6, lw=0, label=f"range under pure label shift ({lo:.2f}–{hi:.2f})")
for c, (col, mk, lab) in sty.items():
    d = pd.read_csv(os.path.join(O, "results", "natural", f"{c}_lr.csv"))
    diff = (d["Point-SAT"] - d["Source-tuned"])*1000
    ax.scatter(d.domain_auc, diff, c=col, marker=mk, s=46, label=lab, edgecolor="white", linewidth=.6)
    if c == "bank":
        i = int(np.argmax(d.true_prev)); ax.annotate("final bank period: observed prevalence 0.46,\nEM estimate 0.999",
                                                    (d.domain_auc[i], diff[i]), xytext=(0.72, -300), fontsize=7,
                                                    arrowprops=dict(arrowstyle="->", lw=.6))
ax.axhline(0, color="k", lw=.8); ax.set_yscale("symlog", linthresh=1); ax.set_ylim(-4000, 3000)
ax.set_xlabel("domain-classifier AUC, reference vs target features (0.5 = indistinguishable)")
ax.set_ylabel("Point-SAT − source-tuned excess cost (×10⁻³)\nbelow 0: adaptation cheaper")
ax.legend(frameon=False, fontsize=8, loc="upper left"); ax.grid(alpha=.25)
fig.tight_layout(); fig.savefig(os.path.join(F, "fig9_natural.png"), dpi=220); plt.close(fig)
print("figures:", sorted(os.listdir(F)))

"""Post-hoc sensitivity analysis for the Provenance Density score D(T).

WHAT WE CAN SWEEP HERE
----------------------
The audit CSV stores aggregated per-row `p_int` and `density`, where
    density   = tanh( total_score / beta_orig )           with beta_orig = 5.0
    final     = (1 - p_int) * density

We can therefore sweep, post-hoc:
    1. The saturation scalar beta — by recovering total_score via atanh and
       re-saturating at beta'.
    2. The aggregation FORM combining (1 - p_int) and density:
         (a) multiplicative  D = (1 - p_int) * density            [paper]
         (b) gated           D = density if p_int < tau else 0
         (c) additive        D = w * (1 - p_int) + (1 - w) * density
         (d) min/max         D = min{(1 - p_int), density}
         (e) Veto-only       D = 1 - p_int
         (f) density-only    D = density

WHAT WE CANNOT SWEEP HERE
-------------------------
λ (diminishing returns), the cubic exponent in MatchRatio^3, and the reputation
weight scheme {high, medium, low} = {1.0, 0.5, 0.1} live inside the sum that is
saturated by tanh. Recovering them requires re-running the audit.

OUTPUT
------
- sensitivity_metrics.csv (full grid of AUC values)
- sensitivity_beta_curve.{png,pdf} (β-vs-AUC line plot, three splits)
- sensitivity_aggregation_table.csv (form vs AUC)
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
import matplotlib.pyplot as plt

ROOT = Path("/sessions/sleepy-zealous-goldberg/mnt/ijcai god")
df = pd.read_csv(ROOT / "tqa_judge_labels.csv")
print(f"Loaded {len(df)} judge-labeled rows.\n")

BETA_ORIG = 5.0  # paper's value (audit_pipeline.py uses tanh(total/5.0))

# Recover total_score = atanh(density) * beta_orig
# atanh saturates at density=1.0 — clip slightly to avoid inf
density_clip = np.clip(df["density"].values, 0.0, 1.0 - 1e-9)
total_score = np.arctanh(density_clip) * BETA_ORIG
df["total_score"] = total_score


def density_at_beta(total, beta):
    return np.tanh(total / beta)


def score_table(y, scores):
    auc = roc_auc_score(y, scores)
    ap = average_precision_score(y, scores)
    return auc, ap


SPLITS = {
    "All":        df,
    "TruthfulQA": df[df["type"] == "Static_TruthfulQA"].copy(),
    "FreshQA":    df[df["type"] == "Dynamic_FreshQA"].copy(),
}

# --------------------------------------------------------------------------- #
# 1. Saturation scalar β sweep
# --------------------------------------------------------------------------- #
betas = np.array([0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 25.0])

beta_rows = []
for name, sub in SPLITS.items():
    n_pos = int((sub["hallucinated"] == 1).sum())
    if n_pos < 2 or (sub["hallucinated"] == 0).sum() < 2:
        continue
    for beta in betas:
        d_b = density_at_beta(sub["total_score"].values, beta)
        # Detector for hallucination = 1 - D(T)
        full_dt = (1 - sub["p_int"].values) * d_b
        det_dt = 1 - full_dt
        det_dens = 1 - d_b
        det_pint = sub["p_int"].values
        y = sub["hallucinated"].values
        auc_dt, _ = score_table(y, det_dt)
        auc_dens, _ = score_table(y, det_dens)
        auc_pint, _ = score_table(y, det_pint)
        beta_rows.append({
            "split": name, "beta": beta,
            "AUC_D(T)": round(auc_dt, 3),
            "AUC_density": round(auc_dens, 3),
            "AUC_pint": round(auc_pint, 3),  # constant across β by definition
        })

beta_df = pd.DataFrame(beta_rows)
print("=== β saturation sweep ===")
print(beta_df.pivot(index="beta", columns="split", values="AUC_D(T)").round(3))
beta_df.to_csv(ROOT / "sensitivity_beta_sweep.csv", index=False)

# --------------------------------------------------------------------------- #
# 2. Aggregation form ablation (β fixed at paper value 5.0)
# --------------------------------------------------------------------------- #
def agg_multiplicative(p, d): return (1 - p) * d
def agg_min(p, d):            return np.minimum(1 - p, d)
def agg_density_only(p, d):   return d
def agg_pint_only(p, d):      return 1 - p

def agg_gated(tau):
    def fn(p, d):
        return np.where(p < tau, d, 0.0)
    return fn

def agg_additive(w_p):
    def fn(p, d):
        return w_p * (1 - p) + (1 - w_p) * d
    return fn

def agg_geometric(alpha):
    def fn(p, d):
        return ((1 - p) ** alpha) * (d ** (1 - alpha))
    return fn

AGGREGATIONS = {
    "Multiplicative (paper)":          agg_multiplicative,
    "Min{1-Pint, density}":            agg_min,
    "Density only":                    agg_density_only,
    "P_int (Veto) only":               agg_pint_only,
    "Gated, τ=0.10":                   agg_gated(0.10),
    "Gated, τ=0.30":                   agg_gated(0.30),
    "Gated, τ=0.50":                   agg_gated(0.50),
    "Additive 0.5/0.5":                agg_additive(0.5),
    "Additive 0.75 P / 0.25 D":        agg_additive(0.75),
    "Additive 0.25 P / 0.75 D":        agg_additive(0.25),
    "Geometric α=0.25":                agg_geometric(0.25),
    "Geometric α=0.50":                agg_geometric(0.50),
    "Geometric α=0.75":                agg_geometric(0.75),
}

agg_rows = []
for name, sub in SPLITS.items():
    if (sub["hallucinated"] == 1).sum() < 2 or (sub["hallucinated"] == 0).sum() < 2:
        continue
    p = sub["p_int"].values
    d = density_at_beta(sub["total_score"].values, BETA_ORIG)
    y = sub["hallucinated"].values
    for agg_name, fn in AGGREGATIONS.items():
        score = fn(p, d)
        # We're scoring "trust" — invert for hallucination detection
        det = -score  # higher score → lower hallucination probability
        auc, ap = score_table(y, det)
        agg_rows.append({
            "split": name, "form": agg_name,
            "AUC": round(auc, 3),
            "AP":  round(ap, 3),
        })

agg_df = pd.DataFrame(agg_rows)
print("\n=== Aggregation form ablation ===")
print(agg_df.pivot(index="form", columns="split", values="AUC").round(3).to_string())
agg_df.to_csv(ROOT / "sensitivity_aggregation_form.csv", index=False)

# --------------------------------------------------------------------------- #
# 3. Static-vs-Dynamic ecological separation as a function of β
#    (the paper's other key claim that β must preserve)
# --------------------------------------------------------------------------- #
eco_rows = []
tqa = df[df["type"] == "Static_TruthfulQA"]
fresh = df[df["type"] == "Dynamic_FreshQA"]
for beta in betas:
    d_tqa = density_at_beta(tqa["total_score"].values, beta)
    d_fresh = density_at_beta(fresh["total_score"].values, beta)
    full_tqa = (1 - tqa["p_int"].values) * d_tqa
    full_fresh = (1 - fresh["p_int"].values) * d_fresh
    eco_rows.append({
        "beta": beta,
        "M_TruthfulQA": float(np.mean(full_tqa)),
        "M_FreshQA": float(np.mean(full_fresh)),
        "ecological_gap": float(np.mean(full_tqa) - np.mean(full_fresh)),
    })
eco_df = pd.DataFrame(eco_rows)
print("\n=== Ecological Static-vs-Dynamic separation by β ===")
print(eco_df.round(3).to_string(index=False))
eco_df.to_csv(ROOT / "sensitivity_beta_ecological.csv", index=False)

# --------------------------------------------------------------------------- #
# 4. Plot β-vs-AUC + β-vs-ecological-gap, two-panel layout
# --------------------------------------------------------------------------- #
plt.rcParams.update({"font.family": "serif", "font.size": 9,
                     "axes.titlesize": 10, "legend.fontsize": 8,
                     "axes.spines.top": False, "axes.spines.right": False})

fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9))

# Panel A: AUC vs β (with all three splits + Veto baselines)
ax = axes[0]
SPLIT_ORDER = [("All", "All N=200", "#1f77b4", "o-"),
               ("TruthfulQA", "TruthfulQA N=150", "#2ca02c", "s-"),
               ("FreshQA", "FreshQA N=50", "#9467bd", "D-")]
for skey, stitle, color, marker in SPLIT_ORDER:
    sub = beta_df[beta_df["split"] == skey].sort_values("beta")
    ax.plot(sub["beta"], sub["AUC_D(T)"], marker, lw=1.4, ms=4.5,
            color=color, label=f"D(T) — {stitle}")
ax.axvline(BETA_ORIG, color="black", lw=0.6, ls="-", alpha=0.3)
ax.text(BETA_ORIG * 1.1, 0.55, "β=5\n(paper)", fontsize=7, alpha=0.6)
ax.axhline(0.5, color="black", lw=0.4, ls=":", alpha=0.4)
ax.set_xscale("log")
ax.set_xticks([0.5, 1, 2, 5, 10, 25])
ax.set_xticklabels(["0.5", "1", "2", "5", "10", "25"])
ax.set_xlabel("Saturation scalar β")
ax.set_ylabel("Hallucination-detection AUC")
ax.set_title("(a) Discriminative power vs. β")
ax.set_ylim(0.45, 0.95)
ax.legend(loc="upper right", frameon=False, fontsize=7)

# Panel B: Ecological gap (M_TQA - M_Fresh) vs β
ax = axes[1]
ax.plot(eco_df["beta"], eco_df["M_TruthfulQA"], "o-", lw=1.4, ms=4.5,
        color="#2ca02c", label="$\\overline{D(T)}$ TruthfulQA")
ax.plot(eco_df["beta"], eco_df["M_FreshQA"], "D-", lw=1.4, ms=4.5,
        color="#9467bd", label="$\\overline{D(T)}$ FreshQA")
ax.fill_between(eco_df["beta"], eco_df["M_FreshQA"], eco_df["M_TruthfulQA"],
                alpha=0.12, color="grey", label="Ecological gap")
ax.axvline(BETA_ORIG, color="black", lw=0.6, ls="-", alpha=0.3)
ax.text(BETA_ORIG * 1.1, 0.05, "β=5\n(paper)", fontsize=7, alpha=0.6)
ax.set_xscale("log")
ax.set_xticks([0.5, 1, 2, 5, 10, 25])
ax.set_xticklabels(["0.5", "1", "2", "5", "10", "25"])
ax.set_xlabel("Saturation scalar β")
ax.set_ylabel("Mean D(T)")
ax.set_title("(b) Static-vs-Dynamic separation vs. β")
ax.legend(loc="lower right", frameon=False, fontsize=7)
ax.set_ylim(0, 1.0)

plt.tight_layout()
fig.savefig(ROOT / "sensitivity_beta_curve.png", dpi=300, bbox_inches="tight")
fig.savefig(ROOT / "sensitivity_beta_curve.pdf", bbox_inches="tight")
print(f"\nSaved β sweep figure to {ROOT/'sensitivity_beta_curve.png'}")

# --------------------------------------------------------------------------- #
# Summary
# --------------------------------------------------------------------------- #
print("\n=== HEADLINE NUMBERS ===")
for skey in ["All", "TruthfulQA", "FreshQA"]:
    sub = beta_df[beta_df["split"] == skey]
    if len(sub) == 0: continue
    print(f"\n{skey}:")
    print(f"  AUC D(T) range over β [0.5..25]: [{sub['AUC_D(T)'].min():.3f}, {sub['AUC_D(T)'].max():.3f}]   variation = {(sub['AUC_D(T)'].max()-sub['AUC_D(T)'].min()):.3f}")
    print(f"  AUC P_int (constant): {sub['AUC_pint'].iloc[0]:.3f}")

print("\nBest aggregation form per split (by AUC):")
for skey in ["All", "TruthfulQA", "FreshQA"]:
    sub = agg_df[agg_df["split"] == skey].sort_values("AUC", ascending=False)
    if len(sub) == 0: continue
    print(f"\n  {skey}:")
    print(sub.head(5).to_string(index=False))

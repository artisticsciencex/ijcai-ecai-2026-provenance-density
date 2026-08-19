"""ROC / PR / AUC analysis on the LLM-as-judge labelled audit.

Outputs:
  - tqa_roc_metrics.csv          # numbers for the table
  - tqa_roc_pr_figure.{png,pdf}  # the figure for the paper
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA = REPO_ROOT / "data"
RESULTS = REPO_ROOT / "results"
FIGURES = RESULTS / "figures"
RESULTS.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA / "tqa_judge_labels.csv")
print(f"Loaded {len(df)} judge-labeled rows.")
if "label_status" not in df.columns:
    df["label_status"] = np.where(
        df["judge_has_refs"].astype(bool),
        "reference_grounded",
        "exploratory_closed_book",
    )

# Positive class for hallucination DETECTION = label==1.
# D(T), density: high values mean MORE grounded → expected detector = (1 - score).
# p_int: high = MORE uncertain → expected detector = p_int directly.

DETECTORS = {
    "D(T)":          ("final_score", lambda s: 1 - s),
    "Density only":  ("density",     lambda s: 1 - s),
    "P_int (Veto)":  ("p_int",       lambda s: s),
}

SPLITS = {
    "TruthfulQA (N=150, reference-grounded)": df[
        df["label_status"] == "reference_grounded"
    ].copy(),
    "All (N=200, mixed label basis; exploratory)": df,
    "Dynamic probes (N=50, closed-book; exploratory)": df[
        df["label_status"] == "exploratory_closed_book"
    ].copy(),
}


def bootstrap_auc(y, scores, n=2000, rng=None):
    rng = rng or np.random.RandomState(42)
    aucs = []
    n_samples = len(y)
    y = np.asarray(y); scores = np.asarray(scores)
    for _ in range(n):
        idx = rng.choice(n_samples, n_samples, replace=True)
        y2 = y[idx]; s2 = scores[idx]
        if len(set(y2)) < 2:
            continue
        aucs.append(roc_auc_score(y2, s2))
    if not aucs:
        return (np.nan, np.nan, np.nan)
    return tuple(np.percentile(aucs, [2.5, 50, 97.5]))


# ----- METRIC TABLE -----
rows = []
for split_name, sub in SPLITS.items():
    n_pos = int((sub["hallucinated"] == 1).sum())
    n_neg = int((sub["hallucinated"] == 0).sum())
    if n_pos < 2 or n_neg < 2:
        print(f"{split_name}: skipped (too few positives/negatives: pos={n_pos}, neg={n_neg})")
        continue
    for det_name, (col, transform) in DETECTORS.items():
        scores = transform(sub[col])
        y = sub["hallucinated"].values
        auc = roc_auc_score(y, scores)
        ap = average_precision_score(y, scores)
        ci_lo, ci_med, ci_hi = bootstrap_auc(y, scores)
        rows.append({
            "split": split_name,
            "evidence_status": (
                "confirmatory" if "reference-grounded" in split_name else "exploratory"
            ),
            "n": len(sub), "n_hallucinated": n_pos, "n_truthful": n_neg,
            "detector": det_name,
            "AUC": round(auc, 3),
            "AUC_95CI": f"[{ci_lo:.3f}, {ci_hi:.3f}]",
            "AP": round(ap, 3),
        })

metrics = pd.DataFrame(rows)
print("\n=== METRICS ===")
print(metrics.to_string(index=False))
metrics.to_csv(RESULTS / "tqa_roc_metrics.csv", index=False)
print(f"\nSaved metrics CSV to {RESULTS/'tqa_roc_metrics.csv'}")


# ----- TWO-PANEL FIGURE -----
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# IJCAI two-column width (~3.4 in per column → 7-in for two-column figure)
fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))

ALL = SPLITS["TruthfulQA (N=150, reference-grounded)"]

DET_COLORS = {
    "D(T)":         "#1f77b4",
    "Density only": "#999999",
    "P_int (Veto)": "#d62728",
}

# Panel A: ROC
ax = axes[0]
for det_name, (col, transform) in DETECTORS.items():
    scores = transform(ALL[col])
    y = ALL["hallucinated"].values
    fpr, tpr, _ = roc_curve(y, scores)
    auc = roc_auc_score(y, scores)
    ax.plot(fpr, tpr, lw=1.6, color=DET_COLORS[det_name],
            label=f"{det_name} (AUC = {auc:.2f})")
ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.4)
ax.set_xlabel("False positive rate")
ax.set_ylabel("True positive rate")
ax.set_title("(a) ROC — reference-grounded TruthfulQA")
ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
ax.legend(loc="lower right", frameon=False)

# Panel B: PR
ax = axes[1]
base = (ALL["hallucinated"] == 1).mean()
for det_name, (col, transform) in DETECTORS.items():
    scores = transform(ALL[col])
    y = ALL["hallucinated"].values
    p, r, _ = precision_recall_curve(y, scores)
    ap = average_precision_score(y, scores)
    ax.plot(r, p, lw=1.6, color=DET_COLORS[det_name],
            label=f"{det_name} (AP = {ap:.2f})")
ax.axhline(base, ls="--", color="k", lw=0.8, alpha=0.4,
           label=f"Prevalence = {base:.2f}")
ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.set_title("(b) Precision–Recall — reference-grounded")
ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
ax.legend(loc="upper right", frameon=False)

plt.tight_layout()
fig.savefig(FIGURES / "tqa_roc_pr_figure.png", dpi=300, bbox_inches="tight")
fig.savefig(FIGURES / "tqa_roc_pr_figure.pdf", bbox_inches="tight")
print(f"\nSaved figures to {FIGURES/'tqa_roc_pr_figure.png'} and .pdf")


# ----- DIAGNOSTIC: density-vs-Pint scatter coloured by judge label -----
fig2, ax = plt.subplots(figsize=(4.2, 3.4))
trues = ALL[ALL["hallucinated"] == 0]
hals = ALL[ALL["hallucinated"] == 1]
ax.scatter(trues["p_int"], trues["density"], s=18, alpha=0.55,
           color="#2ca02c", edgecolor="white", linewidth=0.4, label=f"Truthful (n={len(trues)})")
ax.scatter(hals["p_int"], hals["density"], s=24, alpha=0.85,
           color="#d62728", edgecolor="white", linewidth=0.4, label=f"Hallucinated (n={len(hals)})", marker="X")
ax.set_xlabel("Internal uncertainty $P_{int}$  →  more confused")
ax.set_ylabel("Evidence density  →  more grounded")
ax.set_title("Reference-grounded TruthfulQA in the (P_int, density) plane")
ax.set_xlim(-0.02, 0.85); ax.set_ylim(-0.02, 1.05)
ax.legend(loc="lower left", frameon=False)
plt.tight_layout()
fig2.savefig(FIGURES / "tqa_density_vs_pint_by_label.png", dpi=300, bbox_inches="tight")
fig2.savefig(FIGURES / "tqa_density_vs_pint_by_label.pdf", bbox_inches="tight")
print(f"Saved diagnostic scatter to {FIGURES/'tqa_density_vs_pint_by_label.png'}")

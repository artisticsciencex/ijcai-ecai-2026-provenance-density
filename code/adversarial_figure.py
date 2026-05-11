"""Two-panel figure for the adversarial robustness section."""
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path("/sessions/sleepy-zealous-goldberg/mnt/ijcai god")

plt.rcParams.update({"font.family": "serif", "font.size": 9,
                     "axes.titlesize": 10, "legend.fontsize": 8,
                     "axes.spines.top": False, "axes.spines.right": False})

# Panel A: rescue/false-alarm tradeoff at varying P_int thresholds
rescue = pd.read_csv(ROOT / "adv_natural_rescue.csv")

# Panel B: attack scenarios bar chart
attacks = pd.read_csv(ROOT / "adv_synthetic_attacks.csv")

fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2))

# Panel A: ROC-like tradeoff — Veto's hallucination recall vs false-alarm rate
ax = axes[0]
ax.plot(rescue["false_alarm"], rescue["attack_recall"], "o-", lw=1.6,
        color="#1f77b4", ms=6)
for _, r in rescue.iterrows():
    ax.annotate(f"τ={r['p_int_threshold']:.1f}",
                (r["false_alarm"], r["attack_recall"]),
                xytext=(6, 4), textcoords="offset points", fontsize=7,
                color="#444")
ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.4)
ax.set_xlabel("False-alarm rate on benign high-density rows")
ax.set_ylabel("Hallucination recall on natural attacks")
ax.set_title("(a) Veto rescue on natural adversarials\n(33 hallucinations with density ≥ 0.7)")
ax.set_xlim(-0.02, 0.5)
ax.set_ylim(-0.02, 1.0)

# Panel B: synthetic attack bar chart
ax = axes[1]
labels = [
    "Benign\n(true claim, aligned)",
    "Citation laundering\n(off-topic high-trust)",
    "Keyword stuffing\n(generic blogs)",
    "Empty-keyword\n(no proper nouns)",
    "Domain spoofing\n('.gov' substring)",
    "SEO inflation\n(perfect attack)",
]
vals = attacks["D_density"].tolist()
colors = ["#2ca02c"] + ["#d62728"] * 5
bars = ax.bar(range(len(vals)), vals, color=colors, edgecolor="black", linewidth=0.4)
ax.set_xticks(range(len(vals)))
ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=7)
ax.set_ylabel("D(T) density component")
ax.axhline(vals[0], color="#2ca02c", lw=0.8, ls=":", alpha=0.6)
ax.text(5.4, vals[0] + 0.02, f"benign = {vals[0]:.2f}", fontsize=7,
        ha="right", color="#2ca02c")
ax.set_title("(b) Synthetic attack inflation\n(fabricated claim, D=0 honest)")
ax.set_ylim(0, 0.75)

for bar, v in zip(bars, vals):
    ax.text(bar.get_x() + bar.get_width() / 2, v + 0.012, f"{v:.2f}",
            ha="center", va="bottom", fontsize=7)

plt.tight_layout()
fig.savefig(ROOT / "adversarial_robustness_figure.png", dpi=300, bbox_inches="tight")
fig.savefig(ROOT / "adversarial_robustness_figure.pdf", bbox_inches="tight")
print(f"Saved {ROOT/'adversarial_robustness_figure.png'}")

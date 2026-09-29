"""Recreate the six-configuration exploratory figure from comparison audit."""
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parents[2]  # thesis final/ (patched from original sibling-of-script layout)
audit = pd.read_csv(root / "results/comparison_audit.csv")
checks = pd.read_csv(root / "results/prompt_comparison_tests.csv")
rows = audit[audit.source == "LLM"]
wide = rows.groupby(["text_id", "config"]).tv.mean().unstack("config") * 100
assert wide.shape == (300, 6) and not wide.isna().any().any()
reported = checks.loc[checks.metric == "tv", "p_value"].iloc[0]
assert .19 < reported < .21
order = ["zero_shot/inline", "zero_shot/numbered", "one_shot/inline",
         "one_shot/numbered", "few_shot/inline", "few_shot/numbered"]
fig, ax = plt.subplots(figsize=(8.9, 5.4))
bp = ax.boxplot([wide[c] for c in order], tick_labels=[s.replace("_shot/", "-shot\n") for s in order],
                patch_artist=True, widths=.55, showfliers=False)
for box in bp["boxes"]:
    box.set(facecolor="#dbe5ee", edgecolor="#415e73")
for line in bp["medians"]:
    line.set(color="#b04a2f", linewidth=2)
ax.set(title="Model persona shift across prompt configurations (exploratory)",
       ylabel="Mean neutral-to-persona TV per text (percentage points)")
ax.text(.02,.97,f"Reported text-level Friedman test: p = {reported:.3f}; no effect detected",
        transform=ax.transAxes,ha="left",va="top",fontsize=10.5,color="#384d61",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=2))
ax.grid(axis="y", alpha=.2)
fig.tight_layout()
fig.savefig(root / "thesis_document/images/fig12_prompt_structure_exploratory.png", dpi=240)
plt.close(fig)

"""Rebuild the text-level H4 covariation figure from the supplied comparison audit."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[2]  # thesis final/ (patched from original sibling-of-script layout)
AUDIT = ROOT / "results" / "comparison_audit.csv"
audit = pd.read_csv(AUDIT)
means = audit.groupby(["text_id", "category", "source"], as_index=False).tv.mean()
wide = means.pivot(index=["text_id", "category"], columns="source", values="tv").reset_index()
assert len(wide) == 300 and not wide[["Human", "LLM"]].isna().any().any()

x, y = wide.Human.to_numpy(), wide.LLM.to_numpy()
rho = float(spearmanr(x, y).statistic)
rng = np.random.default_rng(42)
boot = np.empty(10_000)
for i in range(len(boot)):
    sample = rng.integers(0, len(wide), len(wide))
    boot[i] = spearmanr(x[sample], y[sample]).statistic
lo, hi = np.percentile(boot, [2.5, 97.5])
pd.DataFrame(
    [dict(unit="text", n_texts=300, rho=rho, ci_lo=lo, ci_hi=hi,
          bootstrap_resamples=10_000, seed=42)]
).to_csv(ROOT / "results" / "h4_text_level_covariation_verified.csv", index=False)

fig, ax = plt.subplots(figsize=(8.2, 5.7), layout="constrained")
colors = {
    "unambiguous": "#396caa",
    "author_independent": "#d38136",
    "author_relevant": "#b34f66",
}
for category, frame in wide.groupby("category"):
    ax.scatter(frame.Human * 100, frame.LLM * 100, label=category.replace("_", " ").title(),
               color=colors[category], s=30, alpha=0.65)
ax.set(xlabel="Mean human persona TV per text (%)", ylabel="Mean model persona TV per text (%)",
       title="Human and model framing shifts across 300 texts")
ax.text(0.02, 0.98, f"Spearman rho = {rho:.3f}; text bootstrap 95% CI [{lo:.3f}, {hi:.3f}]",
        transform=ax.transAxes, ha="left", va="top", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.95, edgecolor="#dddddd"))
ax.legend(title="Ambiguity type", loc="lower right", fontsize=8)
ax.grid(alpha=0.15)
fig.savefig(ROOT / "thesis_document" / "images" / "fig05_human_llm_magnitude_scatter.png", dpi=220)
plt.close(fig)
print(f"n=300 rho={rho:.6f} bootstrap CI=({lo:.6f}, {hi:.6f})")

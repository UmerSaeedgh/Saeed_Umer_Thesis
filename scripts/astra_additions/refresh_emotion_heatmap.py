"""Recreate human emotion heatmap with the corrected FDR annotation."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parents[2]  # thesis final/ (patched from original sibling-of-script layout)
rows = pd.read_csv(root / "results/exploratory_per_emotion_tests.csv")
human = rows[rows.source == "Human"]
assert len(human) == 33 and (human.p_BH < .05).sum() == 1
orders = ["Unambiguous", "Author-independent", "Author-relevant"]
emotions = sorted(human.emotion.unique())
grid = human.pivot(index="emotion", columns="category", values="mean_change_pp").loc[emotions,orders]
sig = human.pivot(index="emotion", columns="category", values="p_BH").loc[emotions,orders] < .05
fig, ax = plt.subplots(figsize=(7.8, 7.0))
vmax = np.abs(grid.to_numpy()).max()
im = ax.imshow(grid.to_numpy(), cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
ax.set_xticks(range(3),["Unambiguous", "Author-independent", "Author-relevant"],rotation=22,ha="right")
ax.set_yticks(range(11),emotions)
for i in range(11):
 for j in range(3):
    val = grid.iloc[i,j]
    ax.text(j,i,f"{val:+.1f}{'*' if sig.iloc[i,j] else ''}",ha="center",va="center",
            color="white" if abs(val)>vmax*.58 else "#182a37",fontsize=11)
ax.set_title("Mean human emotion change under persona framing",fontsize=15,pad=10)
fig.colorbar(im,ax=ax,label="Change from neutral (percentage points)")
fig.text(.08,.02,"* One human effect survives BH-FDR (guilt, author-independent); others descriptive.",fontsize=10)
fig.tight_layout(rect=(0,.035,1,1))
fig.savefig(root / "thesis_document/images/fig07_per_emotion_delta_heatmap.png",dpi=240)
plt.close(fig)

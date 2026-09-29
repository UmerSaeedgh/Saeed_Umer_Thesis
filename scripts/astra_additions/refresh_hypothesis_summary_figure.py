"""Render the thesis summary from the verified results tables."""
from pathlib import Path
import textwrap
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

root = Path(__file__).resolve().parents[2]  # thesis final/ (patched from original sibling-of-script layout)
results = root / "results"
h12 = pd.read_csv(results / "vector_H1_H2.csv")
h3 = pd.read_csv(results / "vector_H3_raw_and_referenced.csv")
direction = pd.read_csv(results / "vector_H4_direction.csv").iloc[0]
magnitude = pd.read_csv(results / "vector_H4_magnitude.csv").iloc[0]
count_gap = pd.read_csv(results / "count_standardised_magnitude_gap.csv")
ar = h12[(h12.hypothesis == "H1") & (h12.category == "author_relevant")]
un = h12[(h12.hypothesis == "H1") & (h12.category == "unambiguous")]
ai = h12[(h12.hypothesis == "H1") & (h12.category == "author_independent")]
model = h12[h12.hypothesis == "H2"]
specific = h3[h3.contrast == "Author-relevant minus Author-independent"]
raw = specific[specific.analysis == "Raw"].iloc[0]
adjusted = specific[specific.analysis == "Conditional-null referenced"].iloc[0]
standardised_gap = count_gap[count_gap.analysis == "tv_standardised"].iloc[0].mean_gap_pp
assert len(ar) == len(un) == len(ai) == len(model) == 2
assert ar.reject_05.all() and not un.reject_05.any() and model.reject_05.all()
ai_a = ai[ai.side == "a"].iloc[0]
ai_b = ai[ai.side == "b"].iloc[0]
assert ai_a.reject_05 and not ai_b.reject_05  # side a only, side b not -- must stay in sync with H1 detail text below
assert raw.p_holm < .05 and adjusted.p_holm >= .05

entries = [
 ("H1 · Humans", "Partial support", "Author-relevant: both sides significant; author-independent: side A only; unambiguous: neither side.", "#255b8c"),
 ("H2 · Models", "Supported (ensemble)", "Both persona contrasts significant (Holm p ≤ 0.0002 each).", "#196d62"),
 ("H3 · Ambiguity", "Inconclusive after adjustment", f"Raw relevant–independent: {raw.difference_pp:.2f}pp (Holm p = {raw.p_holm:.3f}); null-referenced: {adjusted.difference_pp:.2f}pp (Holm p = {adjusted.p_holm:.3f}).", "#a36616"),
 ("H4 · Direction", "Weak partial alignment", f"Mean cosine = {direction.mean_cosine:.3f}; permutation p ≤ {direction.p_permutation:.4f}.", "#54599b"),
 ("H4 · Magnitude", "Magnitudes differ", f"Human–model gap = {magnitude.human_minus_llm_pp:.2f}pp; {standardised_gap:.0f}pp after matching vote counts.", "#795590"),
]
# Two-line detail entries need a taller box; wrap first, then lay everything out in INCHES so
# the figure height always exactly matches the content regardless of how many lines wrap (the
# previous version used fixed axes-fraction offsets tuned for all-single-line detail text, which
# silently overflowed the axes -- and got clipped -- once two entries needed a second line).
wrapped = [(label, status, textwrap.wrap(detail, 62), color) for label, status, detail, color in entries]

WIDTH_IN = 7.5              # fixed: chosen so width=\textwidth (5.79in) gives a ~0.77 downscale
TITLE_H = 1.05
ROW_GAP = 0.14
HEADER_H = 0.34
LINE_H = 0.225
BOX_PAD_TOP = 0.10
BOX_PAD_BOTTOM = 0.14
BOTTOM_MARGIN = 0.10

row_heights = [HEADER_H + len(lines) * LINE_H + BOX_PAD_TOP + BOX_PAD_BOTTOM for _, _, lines, _ in wrapped]
total_h = TITLE_H + sum(row_heights) + ROW_GAP * (len(wrapped) - 1) + BOTTOM_MARGIN

fig = plt.figure(figsize=(WIDTH_IN, total_h))
ax = fig.add_axes([0, 0, 1, 1])  # axes fill the whole figure so 1 data unit == 1 inch exactly
fig.patch.set_facecolor("white")
ax.set(xlim=(0, WIDTH_IN), ylim=(0, total_h)); ax.axis("off")
ax.invert_yaxis()  # y grows downward, so "y" below reads top-to-bottom like the layout above

ax.text(.35, .12, "Hypothesis outcomes", fontsize=23, fontweight="bold", color="#162838", va="top")
ax.text(.35, .58, "Primary tests and key sensitivity results", fontsize=12, color="#465465", va="top")

y = TITLE_H
for (label, status, detail_lines, color), box_h in zip(wrapped, row_heights):
 ax.add_patch(FancyBboxPatch((.35, y), WIDTH_IN - .70, box_h, boxstyle="round,pad=0.03,rounding_size=.08",
                              facecolor="#f4f7f9", edgecolor="#e2e9ee", linewidth=1))
 ax.add_patch(FancyBboxPatch((.42, y + .07), .05, box_h - .14, boxstyle="round,pad=0.01", facecolor=color, edgecolor=color))
 ax.text(.65, y + .12, label, fontsize=13.7, fontweight="bold", color="#1b2d3a", va="top")
 ax.text(2.55, y + .12, status, fontsize=13, fontweight="bold", color=color, va="top")
 for j, line in enumerate(detail_lines):
  ax.text(.65, y + HEADER_H + LINE_H * j, line, fontsize=11.7, color="#334251", va="top")
 y += box_h + ROW_GAP

fig.savefig(root / "thesis_document/images/fig13_hypothesis_summary.png", dpi=240)
plt.close(fig)

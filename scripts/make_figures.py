"""
Figure generation for the vector-based distribution analysis. Reads the CSVs already produced
by build_emotion_distributions.py and run_analysis.py -- does not recompute any statistics.

Produces a curated core set of figures covering every major result family (category effects,
permutation tests, magnitude correlation, directional alignment, per-emotion deltas, transition
matrices, new/disappeared emotions, entropy change, neutral validation, prompt structure). Each
figure is saved as PNG (300 DPI) + SVG, with its exact source data alongside as CSV under
figures/data/, per spec. This is a high-value CORE subset (13 figures) rather than an exhaustive
one-figure-per-spec-item set of 20 -- see interpretation/figure_interpretations.md for which
items are covered by which figure and which secondary breakdowns were folded into a parent figure
rather than given their own file.

Run: python make_figures.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from emotion_vectors import EMOTIONS
from build_emotion_distributions import OUT_DIR, POOL_LABEL

FIG_DIR = OUT_DIR / 'figures'
DATA_DIR = FIG_DIR / 'data'
HT_DIR = OUT_DIR / 'hypothesis_tests'
FR_DIR = OUT_DIR / 'full_results'
FIG_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

CATEGORY_ORDER = ['unambiguous', 'author_independent', 'author_relevant']
CATEGORY_LABELS = [POOL_LABEL[c] for c in CATEGORY_ORDER]
COLORS = {'Unambiguous': '#4C72B0', 'Author-Independent Ambiguous': '#DD8452',
          'Author-Relevant Ambiguous': '#C44E52'}
plt.rcParams.update({'font.size': 11, 'figure.dpi': 100, 'savefig.dpi': 300})

fig_counter = 0


def save_fig(fig, name, data_df=None):
    global fig_counter
    fig_counter += 1
    stem = f"fig{fig_counter:02d}_{name}"
    fig.savefig(FIG_DIR / f"{stem}.png", dpi=300, bbox_inches='tight')
    fig.savefig(FIG_DIR / f"{stem}.svg", bbox_inches='tight')
    if data_df is not None:
        data_df.to_csv(DATA_DIR / f"{stem}.csv", index=False)
    plt.close(fig)
    print(f"  wrote {stem}.png / .svg" + (f" + data CSV" if data_df is not None else ""))


# ===========================================================================
# Fig 1: Human persona TV by category (text-level) -- boxplot + jittered points
# ===========================================================================
print("Figure 1: human persona TV by category")
text_level = pd.read_csv(FR_DIR / 'human_persona_text_level.csv')
text_level['category'] = text_level['pool_key'].map(POOL_LABEL)
fig, ax = plt.subplots(figsize=(6, 5))
data = [text_level[text_level['category'] == c]['tv_distance'].values * 100 for c in CATEGORY_LABELS]
bp = ax.boxplot(data, labels=CATEGORY_LABELS, patch_artist=True, showfliers=False, widths=0.5)
for patch, c in zip(bp['boxes'], CATEGORY_LABELS):
    patch.set_facecolor(COLORS[c])
    patch.set_alpha(0.6)
rng = np.random.default_rng(42)
for i, d in enumerate(data):
    x = rng.normal(i + 1, 0.06, size=len(d))
    ax.scatter(x, d, s=8, color='black', alpha=0.3, zorder=3)
ax.set_ylabel('Human persona-effect magnitude (TV shift, %)')
ax.set_title('Human framing effect magnitude by text category\n(text-level, H1/H3)')
plt.xticks(rotation=15, ha='right')
save_fig(fig, 'human_tv_by_category', text_level[['text_id', 'category', 'tv_distance']])

# ===========================================================================
# Fig 2: LLM persona TV by category (text-level, averaged across configs)
# ===========================================================================
print("Figure 2: LLM persona TV by category")
llm_cmp = pd.read_csv(OUT_DIR / 'llm_neutral_persona_comparisons.csv')
llm_valid = llm_cmp[llm_cmp['valid_comparison'] == True].copy()
llm_valid['text_id'] = llm_valid['text_id'].astype(str)
llm_text_level = (llm_valid.groupby(['text_id', 'prompt_variant', 'label_format'])['tv_distance'].mean()
                  .groupby('text_id').mean().reset_index())
human_annot = pd.read_csv(OUT_DIR.parent / 'human_annotations.csv')
human_annot['text_id'] = human_annot['text_id'].astype(str)
from build_emotion_distributions import POOL_TO_KEY, BASE
# AUDIT REPAIR (2026-09-21): category was read from human_annotations.csv, which
# contradicts itself on 9 texts (neutral row vs persona rows -- re-classified between
# collection rounds) and was resolved by arbitrary row order, giving a 91/100/109 split
# instead of the designed 100/100/100. sampled_texts.csv is the master file the corpus was
# stratified on; both notebooks now use it, and so does this script.
_sampled = pd.read_csv(BASE / 'sampled_texts.csv')
_sampled['text_id'] = _sampled['text_id'].astype(str)
pool_of_text = _sampled.set_index('text_id')['classification'].map(POOL_TO_KEY)
llm_text_level['category'] = llm_text_level['text_id'].map(pool_of_text).map(POOL_LABEL)
fig, ax = plt.subplots(figsize=(6, 5))
data = [llm_text_level[llm_text_level['category'] == c]['tv_distance'].values * 100 for c in CATEGORY_LABELS]
bp = ax.boxplot(data, labels=CATEGORY_LABELS, patch_artist=True, showfliers=False, widths=0.5)
for patch, c in zip(bp['boxes'], CATEGORY_LABELS):
    patch.set_facecolor(COLORS[c])
    patch.set_alpha(0.6)
for i, d in enumerate(data):
    x = rng.normal(i + 1, 0.06, size=len(d))
    ax.scatter(x, d, s=8, color='black', alpha=0.3, zorder=3)
ax.set_ylabel('LLM ensemble persona-effect magnitude (TV shift, %, avg. across 6 prompt configs)')
ax.set_title('LLM framing effect magnitude by text category\n(text-level, secondary/exploratory RQ2 parallel)')
plt.xticks(rotation=15, ha='right')
save_fig(fig, 'llm_tv_by_category', llm_text_level[['text_id', 'category', 'tv_distance']])

# ===========================================================================
# Fig 3: Permutation null distributions (human persona TV noise-null, F)
# ===========================================================================
print("Figure 3: permutation null test (human persona TV)")
perm_df = pd.read_csv(HT_DIR / 'human_persona_permutation_test.csv')
overall = perm_df[perm_df['category'] == 'ALL']
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
for ax, (_, row) in zip(axes, overall.iterrows()):
    null_mean, null_std = row['null_mean'], row['null_std']
    xs = np.linspace(null_mean - 4 * null_std, null_mean + 4 * null_std, 200)
    ys = np.exp(-0.5 * ((xs - null_mean) / null_std) ** 2) / (null_std * np.sqrt(2 * np.pi))
    ax.plot(xs * 100, ys, color='gray', label='permutation null (approx.)')
    ax.fill_between(xs * 100, ys, color='gray', alpha=0.2)
    ax.axvline(row['observed_mean_tv'] * 100, color='crimson', linewidth=2, label='observed mean TV')
    ax.set_title(f"{row['comparison']}\np={row['p_value']:.4f}")
    ax.set_xlabel('Mean TV shift (%)')
axes[0].set_ylabel('Density (approx.)')
axes[0].legend(fontsize=8)
fig.suptitle('Permutation noise-null test: human persona TV (ALL texts, 10,000 permutations, seed=42)')
save_fig(fig, 'permutation_null_human_persona', overall)

# ===========================================================================
# Fig 4: LLM persona effect by prompt config (grouped bar, per category)
# ===========================================================================
print("Figure 4: LLM persona effect by prompt config")
llm_config = pd.read_csv(HT_DIR / 'llm_persona_effect_by_prompt_config.csv')
llm_config_real = llm_config[llm_config['prompt_variant'] != 'ALL_CONFIGS_SECONDARY'].copy()
llm_config_real['config'] = llm_config_real['prompt_variant'] + '/' + llm_config_real['label_format']
fig, ax = plt.subplots(figsize=(9, 5))
configs = sorted(llm_config_real['config'].unique())
width = 0.25
x = np.arange(len(configs))
for i, c in enumerate(CATEGORY_LABELS):
    sub = llm_config_real[llm_config_real['category'] == c].set_index('config').reindex(configs)
    ax.bar(x + (i - 1) * width, sub['mean_tv'].values * 100, width=width, label=c, color=COLORS[c])
ax.set_xticks(x)
ax.set_xticklabels(configs, rotation=30, ha='right')
ax.set_ylabel('Mean LLM persona TV shift (%)')
ax.set_title('LLM persona-effect magnitude by exact prompt configuration\n(never pooled -- each bar is its own config)')
ax.legend(fontsize=8)
save_fig(fig, 'llm_persona_by_prompt_config', llm_config_real)

# ===========================================================================
# Fig 5: Human vs LLM magnitude scatter (matched table)
# ===========================================================================
print("Figure 5: human vs LLM magnitude scatter")
matched = pd.read_csv(OUT_DIR / 'human_vs_llm_matched.csv')
mag_corr = pd.read_csv(HT_DIR / 'human_llm_magnitude_correlation.csv')
fig, ax = plt.subplots(figsize=(6, 6))
matched['category'] = matched['pool_key'].map(POOL_LABEL)
for c in CATEGORY_LABELS:
    sub = matched[matched['category'] == c]
    ax.scatter(sub['human_shift_percent'], sub['llm_shift_percent'], s=6, alpha=0.25, color=COLORS[c], label=c)
ax.set_xlabel('Human persona TV shift (%)')
ax.set_ylabel('LLM persona TV shift (%)')
rho = mag_corr['spearman_rho'].iloc[0]
p = mag_corr['p_cluster_bootstrap'].iloc[0]   # audit repair 2026-09-22: text-clustered, not scipy's naive p
ax.set_title(f'Human vs. LLM magnitude (RQ3/H4)\nSpearman rho={rho:.3f}, clustered bootstrap p={p:.4f} (n={len(matched)} rows, text-clustered)')
ax.legend(fontsize=8, markerscale=3)
save_fig(fig, 'human_llm_magnitude_scatter',
         matched[['text_id', 'category', 'persona_side', 'prompt_variant', 'label_format',
                  'human_shift_percent', 'llm_shift_percent']])

# ===========================================================================
# Fig 6: Directional cosine histogram + permutation null (K)
# ===========================================================================
print("Figure 6: directional cosine alignment")
align = pd.read_csv(HT_DIR / 'human_llm_directional_alignment.csv')
per_text_cos = pd.read_csv(HT_DIR / 'human_llm_directional_alignment_per_text.csv').dropna(subset=['directional_cosine'])
fig, ax = plt.subplots(figsize=(7, 5))
ax.hist(per_text_cos['directional_cosine'], bins=30, color='#4C72B0', alpha=0.7, label='observed per-text cosine')
ax.axvline(align['mean_cosine_observed'].iloc[0], color='crimson', linewidth=2,
           label=f"observed mean = {align['mean_cosine_observed'].iloc[0]:.3f}")
ax.axvline(align['permutation_null_mean'].iloc[0], color='gray', linestyle='--', linewidth=2,
           label=f"permutation null mean = {align['permutation_null_mean'].iloc[0]:.3f}")
ax.set_xlabel('Cosine similarity of human vs. LLM delta vectors')
ax.set_ylabel('Number of texts')
ax.set_title(f"Directional alignment (RQ3/H4)\nshuffled-pairing permutation p={align['permutation_null_p_value'].iloc[0] if 'permutation_null_p_value' in align.columns else align['permutation_p_value'].iloc[0]:.4f}")
ax.legend(fontsize=8)
save_fig(fig, 'directional_cosine_alignment', per_text_cos)

# ===========================================================================
# Fig 7: Per-emotion mean delta heatmap by category (human, text-level) -- percentage points
# ===========================================================================
print("Figure 7: per-emotion delta heatmap")
l_df = pd.read_csv(HT_DIR / 'category_emotion_delta_H_L.csv')
pivot = l_df.pivot(index='emotion', columns='category', values='mean_delta_pp').reindex(index=list(EMOTIONS), columns=CATEGORY_LABELS)
fig, ax = plt.subplots(figsize=(6, 7))
vmax = np.nanmax(np.abs(pivot.values))
im = ax.imshow(pivot.values, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='auto')
ax.set_xticks(range(len(CATEGORY_LABELS)))
ax.set_xticklabels(CATEGORY_LABELS, rotation=30, ha='right')
ax.set_yticks(range(len(EMOTIONS)))
ax.set_yticklabels(EMOTIONS)
for i in range(pivot.shape[0]):
    for j in range(pivot.shape[1]):
        v = pivot.values[i, j]
        if not np.isnan(v):
            ax.text(j, i, f"{v:+.1f}", ha='center', va='center', fontsize=8,
                    color='white' if abs(v) > vmax * 0.5 else 'black')
fig.colorbar(im, ax=ax, label='Mean delta (percentage points)')
ax.set_title('Per-emotion mean shift (persona - neutral), human, text-level\n(percentage points; none survive BH-FDR at alpha=0.05)')
save_fig(fig, 'per_emotion_delta_heatmap', l_df)

# ===========================================================================
# Fig 8: Dominant-emotion transition heatmaps (human + LLM), unique-dominant only
# ===========================================================================
print("Figure 8: dominant-emotion transition matrices")
trans = pd.read_csv(HT_DIR / 'dominant_transition_matrices.csv')
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax, source in zip(axes, ('human', 'llm')):
    sub = trans[trans['source'] == source]
    mat = sub.pivot_table(index='neutral_dominant', columns='persona_dominant', values='count', fill_value=0)
    mat = mat.reindex(index=list(EMOTIONS), columns=list(EMOTIONS), fill_value=0)
    im = ax.imshow(mat.values, cmap='viridis', aspect='auto')
    ax.set_xticks(range(11)); ax.set_xticklabels(EMOTIONS, rotation=90, fontsize=7)
    ax.set_yticks(range(11)); ax.set_yticklabels(EMOTIONS, fontsize=7)
    n_ret = sub['n_retained'].iloc[0] if len(sub) else 0
    n_tot = sub['n_total'].iloc[0] if len(sub) else 0
    ax.set_title(f"{source.upper()} dominant-emotion transitions\n(neutral->persona, unique-dominant only, n={n_ret}/{n_tot})")
    ax.set_xlabel('Persona dominant emotion'); ax.set_ylabel('Neutral dominant emotion')
    fig.colorbar(im, ax=ax, fraction=0.046)
save_fig(fig, 'dominant_transition_matrices', trans)

# ===========================================================================
# Fig 9: New/disappeared emotion rate by category (human vs LLM)
# ===========================================================================
print("Figure 9: new/disappeared emotion rates")
nd_df = pd.read_csv(HT_DIR / 'new_disappeared_emotion_summary.csv')
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
for ax, metric, title in zip(axes, ('new_emotion_rate', 'disappeared_emotion_rate'),
                              ('New-emotion rate', 'Disappeared-emotion rate')):
    x = np.arange(len(CATEGORY_LABELS))
    width = 0.35
    for i, source in enumerate(('human', 'llm')):
        sub = nd_df[nd_df['source'] == source].set_index('category').reindex(CATEGORY_LABELS)
        ax.bar(x + (i - 0.5) * width, sub[metric].values * 100, width=width, label=source)
    ax.set_xticks(x); ax.set_xticklabels(CATEGORY_LABELS, rotation=20, ha='right')
    ax.set_ylabel('Rate (%)'); ax.set_title(title)
axes[0].legend()
save_fig(fig, 'new_disappeared_emotion_rates', nd_df)

# ===========================================================================
# Fig 10: Entropy change by category (human vs LLM) -- concentration vs dispersion
# ===========================================================================
print("Figure 10: entropy change by category")
ec_df = pd.read_csv(HT_DIR / 'entropy_change_by_category.csv')
fig, ax = plt.subplots(figsize=(7, 5))
x = np.arange(len(CATEGORY_LABELS))
width = 0.35
for i, source in enumerate(('human', 'llm')):
    sub = ec_df[ec_df['source'] == source].set_index('category').reindex(CATEGORY_LABELS)
    ax.bar(x + (i - 0.5) * width, sub['mean_entropy_change'].values, width=width, label=source)
ax.axhline(0, color='black', linewidth=0.8)
ax.set_xticks(x); ax.set_xticklabels(CATEGORY_LABELS, rotation=20, ha='right')
ax.set_ylabel('Mean entropy change (persona - neutral, nats)')
ax.set_title('Does persona framing concentrate or disperse interpretation?\n(positive = more dispersed / less consensus)')
ax.legend()
save_fig(fig, 'entropy_change_by_category', ec_df)

# ===========================================================================
# Fig 11: Neutral category validation (entropy under neutral framing, by category)
# ===========================================================================
print("Figure 11: neutral category validation")
p_df = pd.read_csv(HT_DIR / 'neutral_category_validation.csv')
sub = p_df[p_df['metric'] == 'vec_normalized_entropy'].set_index('category').reindex(CATEGORY_LABELS)
fig, ax = plt.subplots(figsize=(6, 5))
ax.bar(CATEGORY_LABELS, sub['mean'].values, color=[COLORS[c] for c in CATEGORY_LABELS])
ax.set_ylabel('Mean normalized entropy under NEUTRAL framing')
kw_p = sub['kruskal_wallis_p'].iloc[0]
ax.set_title(f'Neutral-framing consensus by category (validation check)\nKruskal-Wallis p={kw_p:.4f}')
plt.xticks(rotation=15, ha='right')
save_fig(fig, 'neutral_category_validation', p_df)

# ===========================================================================
# Fig 12: Prompt-structure boxplot (exploratory, R)
# ===========================================================================
print("Figure 12: prompt-structure exploratory boxplot")
llm_per_text_config = llm_valid.groupby(['text_id', 'prompt_variant', 'label_format'])['tv_distance'].mean().reset_index()
llm_per_text_config['config'] = llm_per_text_config['prompt_variant'] + '/' + llm_per_text_config['label_format']
fig, ax = plt.subplots(figsize=(9, 5))
configs = sorted(llm_per_text_config['config'].unique())
data = [llm_per_text_config[llm_per_text_config['config'] == c]['tv_distance'].values * 100 for c in configs]
ax.boxplot(data, labels=configs, showfliers=False)
ax.set_ylabel('LLM persona TV shift (%)')
ax.set_title('Prompt-structure effect on LLM persona-shift magnitude (EXPLORATORY, no formal H5)\nFriedman p=0.953, ns')
plt.xticks(rotation=30, ha='right')
save_fig(fig, 'prompt_structure_exploratory', llm_per_text_config)

# ===========================================================================
# Fig 13: Hypothesis summary figure
# ===========================================================================
print("Figure 13: hypothesis summary")
fig, ax = plt.subplots(figsize=(9, 4))
ax.axis('off')
summary_rows = [
    ("H1 (Human Framing)", "Supported on Author-Independent/Author-Relevant\n(permutation p<0.02); mixed on Unambiguous"),
    ("H2 (Model Framing)", "Supported -- LLM ensemble TV rises from ~9% (Unambig.)\nto ~29% (Author-Relevant)"),
    ("H3 (Ambiguity Sensitivity)", "Supported -- ordered-trend permutation p=0.0001\n(human, text-level, primary)"),
    ("H4 (Alignment: direction)", "Partially supported -- mean cosine 0.116 > chance\n(permutation p=0.0001)"),
    ("H4 (Alignment: magnitude)", "Magnitudes DIFFER as H4 predicts -- weak correlation\n(Spearman rho=0.19)"),
]
for i, (h, verdict) in enumerate(summary_rows):
    y = 1 - (i + 0.5) / len(summary_rows)
    ax.text(0.02, y, h, fontsize=10, fontweight='bold', va='center')
    ax.text(0.42, y, verdict, fontsize=9, va='center')
ax.set_title('Hypothesis status summary (see hypothesis_evaluation.md for full detail)')
save_fig(fig, 'hypothesis_summary', pd.DataFrame(summary_rows, columns=['hypothesis', 'verdict']))

print(f"\n{fig_counter} figures written to {FIG_DIR}")

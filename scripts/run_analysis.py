"""
Main statistical analysis orchestrator for the vector-based distribution analysis
(distribution_analysis_v1/). Reads the QC-passed CSVs produced by build_emotion_distributions.py
and produces every confirmatory/exploratory test tied to the actual thesis RQs/hypotheses
(see distribution_analysis_v1/interpretation/hypotheses_source.md for the exact recovered
wording -- H1-H4, RQ1-RQ3; no formal H5/RQ5 exists, so prompt-structure analysis here is
explicitly exploratory).

Precondition: python qc_gate.py must have exited 0. This script re-checks that the QC report
says PASS before doing any inferential work, per the "stop before inferential conclusions if
QC fails" instruction.

Run: python run_analysis.py
Outputs: CSVs under distribution_analysis_v1/hypothesis_tests/ and /full_results/, plus a
printed console summary at the end using the actual computed numbers.

Random seed fixed at 42 throughout (see stats_utils.RANDOM_SEED) -- every permutation test and
bootstrap CI here is exactly reproducible.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from emotion_vectors import EMOTIONS, build_vector, safe_cosine_similarity, dominant_set
import stats_utils as su
from build_emotion_distributions import (
    BASE, OUT_DIR, POOL_TO_KEY, POOL_LABEL, MODEL_COLS, MODEL_RAW_COLS,
    _normalize_label_fallback,
)

warnings.filterwarnings('ignore', category=RuntimeWarning)

HT_DIR = OUT_DIR / 'hypothesis_tests'
FR_DIR = OUT_DIR / 'full_results'
CATEGORY_ORDER = ['unambiguous', 'author_independent', 'author_relevant']  # H3's predicted increasing order
CATEGORY_LABELS = [POOL_LABEL[c] for c in CATEGORY_ORDER]

console_summary = []  # (section, line) collected for the final printed summary


def log(msg):
    print(msg)


def summarize(line):
    console_summary.append(line)


# ---------------------------------------------------------------------------
# 0. QC gate check
# ---------------------------------------------------------------------------
qc_report = (OUT_DIR / 'interpretation' / 'analysis_qc_report.md').read_text(encoding='utf-8')
if 'Gate status: PASS' not in qc_report:
    log("QC GATE HAS NOT PASSED. Run qc_gate.py and resolve failures before running this script.")
    sys.exit(1)
log("QC gate PASS confirmed. Proceeding with full inferential analysis.\n")

# ---------------------------------------------------------------------------
# Load full_results CSVs
# ---------------------------------------------------------------------------
human_cmp = pd.read_csv(OUT_DIR / 'human_neutral_persona_comparisons.csv')
llm_cmp = pd.read_csv(OUT_DIR / 'llm_neutral_persona_comparisons.csv')
matched = pd.read_csv(OUT_DIR / 'human_vs_llm_matched.csv')
baseline = pd.read_csv(OUT_DIR / 'baseline_validation_original_vs_new_neutral.csv')
human_vectors = pd.read_csv(OUT_DIR / 'human_vectors.csv')
llm_vectors = pd.read_csv(OUT_DIR / 'llm_ensemble_vectors.csv')
human_annotations = pd.read_csv(BASE / 'human_annotations.csv')
human_annotations['text_id'] = human_annotations['text_id'].astype(str)
human_annotations['pool_key'] = human_annotations['pool'].map(POOL_TO_KEY)

for df in (human_cmp, llm_cmp, matched, baseline, human_vectors, llm_vectors):
    df['text_id'] = df['text_id'].astype(str)

# AUDIT REPAIR (2026-09-21): category was read from human_annotations.csv, which
# contradicts itself on 9 texts (neutral row vs persona rows -- re-classified between
# collection rounds) and was resolved by arbitrary row order, giving a 91/100/109 split
# instead of the designed 100/100/100. sampled_texts.csv is the master file the corpus was
# stratified on; both notebooks now use it, and so does this script.
_sampled = pd.read_csv(BASE / 'sampled_texts.csv')
_sampled['text_id'] = _sampled['text_id'].astype(str)
pool_of_text = _sampled.set_index('text_id')['classification'].map(POOL_TO_KEY)

HT_DIR.mkdir(exist_ok=True)
FR_DIR.mkdir(exist_ok=True)

# ===========================================================================
# D. Baseline validation (original 5-reader vs new 3-reader neutral) BY CATEGORY
# ===========================================================================
log("=== D. Baseline validation by category ===")
baseline_valid = baseline[baseline['full_comparison_valid'] == True].copy()
groups_D = [baseline_valid[baseline_valid['pool_key'] == c]['full_tv_distance'].values for c in CATEGORY_ORDER]
kw_stat_D, kw_p_D = su.kruskal_wallis(*groups_D)
rows_D = []
for c, g in zip(CATEGORY_ORDER, groups_D):
    rows_D.append({'category': POOL_LABEL[c], 'n': len(g), 'mean_tv': np.mean(g), 'median_tv': np.median(g),
                    'std_tv': np.std(g, ddof=1) if len(g) > 1 else None})
baseline_by_cat = pd.DataFrame(rows_D)
baseline_by_cat['kruskal_wallis_stat'] = kw_stat_D
baseline_by_cat['kruskal_wallis_p'] = kw_p_D
baseline_by_cat.to_csv(HT_DIR / 'baseline_validation_by_category.csv', index=False)
log(baseline_by_cat.to_string(index=False))
summarize(f"Baseline validation (original-5 vs new-3 neutral) TV by category: "
          + ", ".join(f"{r['category']}={r['mean_tv']*100:.1f}%" for _, r in baseline_by_cat.iterrows())
          + f" (Kruskal-Wallis p={kw_p_D:.4f})")

# ===========================================================================
# E. Human persona effect -- TEXT-LEVEL aggregation as PRIMARY unit, persona-level as secondary
# ===========================================================================
log("\n=== E. Human persona effect: text-level (primary) + persona-level (secondary) ===")
human_valid = human_cmp[human_cmp['valid_comparison'] == True].copy()
human_valid['pool_key'] = human_valid['pool_key']
text_level = human_valid.groupby('text_id').agg(
    tv_distance=('tv_distance', 'mean'),
    entropy_change=('entropy_change', 'mean'),
    max_probability_change=('max_probability_change', 'mean'),
    support_size_change=('support_size_change', 'mean'),
    n_persona_sides=('persona_side', 'count'),
).reset_index()
text_level['pool_key'] = text_level['text_id'].map(pool_of_text)
text_level.to_csv(FR_DIR / 'human_persona_text_level.csv', index=False)
human_valid.to_csv(FR_DIR / 'human_persona_persona_level_secondary.csv', index=False)
log(f"Text-level table: {len(text_level)} texts (secondary persona-level table: {len(human_valid)} rows)")
for c in CATEGORY_ORDER:
    sub = text_level[text_level['pool_key'] == c]
    log(f"  {POOL_LABEL[c]}: n={len(sub)} mean_tv={sub['tv_distance'].mean()*100:.1f}%")

# ===========================================================================
# F. Permutation-based noise-null test for human persona TV (overall + by category)
# ===========================================================================
log("\n=== F. Permutation noise-null test: human persona TV (10,000 perms, seed=42) ===")


def votes_by_text(side, text_ids=None):
    sub = human_annotations[human_annotations['side'] == side]
    if text_ids is not None:
        sub = sub[sub['text_id'].isin(text_ids)]
    out = {}
    for _, r in sub.iterrows():
        votes = str(r['raters_raw']).split('|') if pd.notna(r['raters_raw']) else []
        out[r['text_id']] = [v.strip().lower() for v in votes if v.strip() != '']
    return out


def mean_tv_statistic(votes_a_by_text, votes_b_by_text):
    tvs = []
    for tid in votes_a_by_text:
        va, vb = votes_a_by_text[tid], votes_b_by_text[tid]
        vec_a, meta_a = build_vector(va), None
        if vec_a is None or build_vector(vb) is None:
            continue
        vec_b = build_vector(vb)
        tv = 0.5 * sum(abs(vec_a[e] - vec_b[e]) for e in EMOTIONS)
        tvs.append(tv)
    return float(np.mean(tvs)) if tvs else 0.0


perm_rows = []
neutral_votes_all = votes_by_text('neutral')
for side in ('a', 'b'):
    persona_votes_all = votes_by_text(side)
    common_all = sorted(set(neutral_votes_all) & set(persona_votes_all))
    na_all = {t: neutral_votes_all[t] for t in common_all if len(neutral_votes_all[t]) > 0 and len(persona_votes_all[t]) > 0}
    pa_all = {t: persona_votes_all[t] for t in na_all}
    obs, null, p = su.permutation_test_two_groups(na_all, pa_all, mean_tv_statistic, n_perm=10000)
    perm_rows.append({'comparison': f'neutral_vs_persona_{side}', 'category': 'ALL', 'n_texts': len(na_all),
                       'observed_mean_tv': obs, 'null_mean': float(np.mean(null)), 'null_std': float(np.std(null)),
                       'p_value': p})
    for c in CATEGORY_ORDER:
        # sorted(): a plain set intersection has hash-randomization-dependent iteration order
        # (PYTHONHASHSEED varies per process by default), which would silently break the
        # "fixed seed -> exactly reproducible" guarantee by feeding the permutation loop a
        # different text order (and therefore a different sequence of rng draws) on every run.
        text_ids_c = sorted(set(pool_of_text[pool_of_text == c].index) & set(na_all.keys()))
        na_c = {t: na_all[t] for t in text_ids_c}
        pa_c = {t: pa_all[t] for t in text_ids_c}
        if len(na_c) < 5:
            continue
        obs_c, null_c, p_c = su.permutation_test_two_groups(na_c, pa_c, mean_tv_statistic, n_perm=10000)
        perm_rows.append({'comparison': f'neutral_vs_persona_{side}', 'category': POOL_LABEL[c], 'n_texts': len(na_c),
                           'observed_mean_tv': obs_c, 'null_mean': float(np.mean(null_c)), 'null_std': float(np.std(null_c)),
                           'p_value': p_c})

perm_df = pd.DataFrame(perm_rows)
pvals = perm_df['p_value'].values
perm_df['p_value_holm'] = su.holm_correct(pvals)
perm_df.to_csv(HT_DIR / 'human_persona_permutation_test.csv', index=False)
log(perm_df.to_string(index=False))
summarize("Human persona permutation test (H1): " +
          "; ".join(f"{r['comparison']}/{r['category']} p={r['p_value']:.4f} (Holm p={r['p_value_holm']:.4f})"
                    for _, r in perm_df.iterrows()))

print("\n[checkpoint D+E+F written]")

# ===========================================================================
# G. LLM ensemble persona effect PER PROMPT CONFIG (never pooled across configs) +
#    secondary llm_mean_persona_sensitivity_across_configs (mean of 6 config-level means)
# ===========================================================================
log("\n=== G. LLM persona effect per exact prompt config ===")
llm_valid = llm_cmp[llm_cmp['valid_comparison'] == True].copy()
g_rows = []
for c in CATEGORY_ORDER:
    sub_c = llm_valid[llm_valid['pool_key'] == c]
    config_means = []
    for pv in sub_c['prompt_variant'].unique():
        for lf in sub_c['label_format'].unique():
            cell = sub_c[(sub_c['prompt_variant'] == pv) & (sub_c['label_format'] == lf)]
            if len(cell) == 0:
                continue
            m = cell['tv_distance'].mean()
            config_means.append(m)
            g_rows.append({'pool_key': c, 'category': POOL_LABEL[c], 'prompt_variant': pv, 'label_format': lf,
                            'n': len(cell), 'mean_tv': m, 'median_tv': cell['tv_distance'].median()})
    g_rows.append({'pool_key': c, 'category': POOL_LABEL[c], 'prompt_variant': 'ALL_CONFIGS_SECONDARY',
                    'label_format': 'llm_mean_persona_sensitivity_across_configs',
                    'n': len(config_means), 'mean_tv': float(np.mean(config_means)), 'median_tv': None})
llm_by_config = pd.DataFrame(g_rows)
llm_by_config.to_csv(HT_DIR / 'llm_persona_effect_by_prompt_config.csv', index=False)
log(llm_by_config[llm_by_config['prompt_variant'] != 'ALL_CONFIGS_SECONDARY'].to_string(index=False))
secondary_g = llm_by_config[llm_by_config['prompt_variant'] == 'ALL_CONFIGS_SECONDARY']
summarize("LLM persona sensitivity, mean-of-6-config-means (secondary, never pooled raw votes): " +
          ", ".join(f"{r['category']}={r['mean_tv']*100:.1f}%" for _, r in secondary_g.iterrows()))

# ===========================================================================
# H + S. Individual-model supplemental diagnostics: label-change rates + transition matrices.
#         Supplemental only -- does NOT replace the ensemble-vector analysis above.
# ===========================================================================
log("\n=== H/S. Individual-LLM-model diagnostics (supplemental, not a replacement) ===")


def load_llm_individual_labels(text_ids=None):
    df = pd.read_csv(BASE / 'experiment_results_all.csv')
    df['pool_key'] = df['classification'].map(POOL_TO_KEY)
    df['text_id'] = df['text_id'].astype(str)
    if text_ids is not None:
        df = df[df['text_id'].isin([str(t) for t in text_ids])]
    for model, norm_col in MODEL_COLS.items():
        raw_col = MODEL_RAW_COLS[model]
        norm_vals = df[norm_col].astype(str).str.strip().str.lower()
        needs_fallback = df[norm_col].isna() | (df[norm_col].astype(str).str.strip() == '')
        resolved = norm_vals.where(~needs_fallback, None)
        fallback_vals = df.loc[needs_fallback, raw_col].apply(
            lambda v: _normalize_label_fallback(v) if pd.notna(v) else None)
        resolved.loc[needs_fallback] = fallback_vals
        resolved = resolved.where(resolved.isin(EMOTIONS), None)
        df[f'{model}__resolved'] = resolved
    return df


llm_indiv = load_llm_individual_labels()
model_change_rows = []
transition_counts = {m: {} for m in MODEL_COLS}
key_cols = ['text_id', 'prompt_variant', 'label_format']
for (tid, pv, lf), g in llm_indiv.groupby(key_cols):
    by_framing = {r['framing_condition']: r for _, r in g.iterrows()}
    if 'neutral' not in by_framing:
        continue
    nrow = by_framing['neutral']
    for framing, side in (('persona_a', 'a'), ('persona_b', 'b')):
        if framing not in by_framing:
            continue
        prow = by_framing[framing]
        for model in MODEL_COLS:
            n_label = nrow[f'{model}__resolved']
            p_label = prow[f'{model}__resolved']
            if n_label is None or p_label is None:
                continue
            model_change_rows.append({
                'model': model, 'text_id': tid, 'persona_side': side, 'prompt_variant': pv, 'label_format': lf,
                'pool_key': nrow['pool_key'], 'neutral_label': n_label, 'persona_label': p_label,
                'label_changed': n_label != p_label,
            })
            transition_counts[model].setdefault((n_label, p_label), 0)
            transition_counts[model][(n_label, p_label)] += 1

model_change_df = pd.DataFrame(model_change_rows)
model_change_rate = model_change_df.groupby(['model', 'pool_key'])['label_changed'].agg(['mean', 'count']).reset_index()
model_change_rate['category'] = model_change_rate['pool_key'].map(POOL_LABEL)
model_change_rate.rename(columns={'mean': 'label_change_rate', 'count': 'n'}, inplace=True)
model_change_rate.to_csv(HT_DIR / 'llm_individual_model_label_change_rates.csv', index=False)
log(model_change_rate.pivot(index='model', columns='category', values='label_change_rate').to_string())

trans_rows = []
for model, counts in transition_counts.items():
    for (n_lab, p_lab), cnt in counts.items():
        trans_rows.append({'model': model, 'neutral_label': n_lab, 'persona_label': p_lab, 'count': cnt})
pd.DataFrame(trans_rows).to_csv(HT_DIR / 'llm_individual_model_transition_matrices.csv', index=False)
summarize("Individual-model label-change rate range across categories: " +
          f"{model_change_rate['label_change_rate'].min()*100:.1f}%-{model_change_rate['label_change_rate'].max()*100:.1f}%"
          + " (supplemental diagnostic, does not replace ensemble-vector TV analysis)")

print("\n[checkpoint G+H/S written]")

# ===========================================================================
# I. Category hypothesis test (H1 ambiguous>unambiguous, H3 author_relevant>author_independent):
#    Kruskal-Wallis global + Holm-corrected planned pairwise Mann-Whitney + Cliff's delta +
#    ordered-trend permutation test. PRIMARY = human text-level (confirmatory). LLM per-text
#    (averaged across the 6 configs, a secondary/exploratory parallel check) reported alongside.
# ===========================================================================
log("\n=== I. Category hypothesis test (H1 / H3) ===")


def run_category_test(values_by_cat, label):
    groups = [values_by_cat[c] for c in CATEGORY_ORDER]
    kw_stat, kw_p = su.kruskal_wallis(*groups)
    pair_specs = [('unambiguous', 'author_independent'), ('author_independent', 'author_relevant'),
                  ('unambiguous', 'author_relevant')]
    pair_rows = []
    raw_p = []
    for a, b in pair_specs:
        stat, p = su.mann_whitney(values_by_cat[a], values_by_cat[b])
        delta = su.cliffs_delta(values_by_cat[a], values_by_cat[b])
        pair_rows.append({'comparison': f'{POOL_LABEL[a]} vs {POOL_LABEL[b]}', 'u_stat': stat, 'p_raw': p,
                           'cliffs_delta': delta})
        raw_p.append(p)
    pair_df = pd.DataFrame(pair_rows)
    pair_df['p_holm'] = su.holm_correct(raw_p)
    trend_j, trend_null, trend_p = su.ordered_trend_permutation_test(groups, n_perm=10000)
    log(f"[{label}] Kruskal-Wallis: stat={kw_stat:.3f} p={kw_p:.6f}")
    log(pair_df.to_string(index=False))
    log(f"[{label}] Ordered-trend permutation test (unambiguous<author_independent<author_relevant): "
        f"J={trend_j:.1f} p={trend_p:.5f}")
    pair_df.insert(0, 'source', label)
    pair_df['kruskal_wallis_stat'] = kw_stat
    pair_df['kruskal_wallis_p'] = kw_p
    pair_df['trend_test_J'] = trend_j
    pair_df['trend_test_p'] = trend_p
    return pair_df


human_vals_by_cat = {c: text_level[text_level['pool_key'] == c]['tv_distance'].values for c in CATEGORY_ORDER}
human_cat_test = run_category_test(human_vals_by_cat, 'HUMAN (text-level, primary/confirmatory)')

llm_text_level = (llm_valid.groupby(['text_id', 'prompt_variant', 'label_format'])['tv_distance'].mean()
                  .groupby('text_id').mean().reset_index())
llm_text_level['pool_key'] = llm_text_level['text_id'].map(pool_of_text)
llm_vals_by_cat = {c: llm_text_level[llm_text_level['pool_key'] == c]['tv_distance'].values for c in CATEGORY_ORDER}
llm_cat_test = run_category_test(llm_vals_by_cat, 'LLM (text-level, averaged across configs, secondary/exploratory)')

category_test_df = pd.concat([human_cat_test, llm_cat_test], ignore_index=True)
category_test_df.to_csv(HT_DIR / 'category_hypothesis_test_H1_H3.csv', index=False)
summarize(f"H3 ordered-trend test (human, primary): J={human_cat_test['trend_test_J'].iloc[0]:.1f} "
          f"p={human_cat_test['trend_test_p'].iloc[0]:.5f} -- "
          + ("supports the predicted increasing order" if human_cat_test['trend_test_p'].iloc[0] < 0.05
             else "does NOT reach significance for the predicted increasing order"))

print("\n[checkpoint I written]")

# ===========================================================================
# J. Human vs LLM magnitude correlation (RQ3/H4 magnitude component): Spearman rho +
#    text-clustered bootstrap CI (never row-level -- matched has 12 rows/text: 2 sides x 6 configs)
# ===========================================================================
log("\n=== J. Human vs LLM magnitude correlation (RQ3 / H4 magnitude component) ===")
rho, p, lo, hi = su.spearman_with_cluster_bootstrap(
    matched['human_shift_percent'].values, matched['llm_shift_percent'].values, matched['text_id'].values,
    n_boot=10000)
# AUDIT REPAIR (2026-09-22): `p` is now the TEXT-CLUSTERED bootstrap p returned by
# spearman_with_cluster_bootstrap, not scipy's naive p over all 3,600 correlated rows.
# The column is renamed so downstream readers cannot mistake which one it is.
mag_corr_df = pd.DataFrame([{'spearman_rho': rho, 'p_cluster_bootstrap': p, 'ci95_lo': lo, 'ci95_hi': hi, 'n_rows': len(matched),
                              'n_texts': matched['text_id'].nunique()}])
mag_corr_df.to_csv(HT_DIR / 'human_llm_magnitude_correlation.csv', index=False)
log(mag_corr_df.to_string(index=False))
summarize(f"Human-LLM magnitude correlation (Spearman): rho={rho:.3f} (95% CI [{lo:.3f}, {hi:.3f}]), p={p:.4f}")

# By-category version
mag_corr_by_cat = []
for c in CATEGORY_ORDER:
    sub = matched[matched['pool_key'] == c]
    r, p_c, lo_c, hi_c = su.spearman_with_cluster_bootstrap(
        sub['human_shift_percent'].values, sub['llm_shift_percent'].values, sub['text_id'].values, n_boot=5000)
    mag_corr_by_cat.append({'category': POOL_LABEL[c], 'spearman_rho': r, 'p_cluster_bootstrap': p_c, 'ci95_lo': lo_c, 'ci95_hi': hi_c})
mag_corr_cat_df = pd.DataFrame(mag_corr_by_cat)
mag_corr_cat_df.to_csv(HT_DIR / 'human_llm_magnitude_correlation_by_category.csv', index=False)
log(mag_corr_cat_df.to_string(index=False))

# ===========================================================================
# K. Human vs LLM directional alignment (RQ3/H4 direction component): cosine of DELTA vectors +
#    shuffled-pairing permutation baseline. Pseudoreplication safeguard: collapsed to ONE
#    (human_delta, llm_delta) pair per TEXT (averaged across persona_side and all 6 prompt
#    configs) before permuting pairing, so the resampling/permutation unit is text_id, never a
#    repeated persona/prompt/model row.
# ===========================================================================
log("\n=== K. Human vs LLM directional alignment (RQ3 / H4 direction component) ===")
delta_cols_h = [f'human_delta_{e}' for e in EMOTIONS]
delta_cols_l = [f'llm_delta_{e}' for e in EMOTIONS]
text_deltas = matched.groupby('text_id')[delta_cols_h + delta_cols_l].mean().reset_index()
text_deltas['pool_key'] = text_deltas['text_id'].map(pool_of_text)

human_delta_list = [dict(zip(EMOTIONS, row[delta_cols_h])) for _, row in text_deltas.iterrows()]
llm_delta_list = [dict(zip(EMOTIONS, row[delta_cols_l])) for _, row in text_deltas.iterrows()]


def cosine_fn(a, b):
    res = safe_cosine_similarity(a, b)
    return res.value if res.defined else None


obs_mean_cos, null_cos, p_cos = su.permutation_pairing_test(human_delta_list, llm_delta_list, cosine_fn, n_perm=10000)
per_text_cos = [cosine_fn(human_delta_list[i], llm_delta_list[i]) for i in range(len(human_delta_list))]
text_deltas['directional_cosine'] = per_text_cos
n_defined = sum(1 for v in per_text_cos if v is not None)
align_df = pd.DataFrame([{
    'n_texts': len(text_deltas), 'n_defined_cosine': n_defined, 'mean_cosine_observed': obs_mean_cos,
    'permutation_null_mean': float(np.mean(null_cos)), 'permutation_null_std': float(np.std(null_cos)),
    'permutation_p_value': p_cos,
}])
align_df.to_csv(HT_DIR / 'human_llm_directional_alignment.csv', index=False)
text_deltas[['text_id', 'pool_key', 'directional_cosine']].to_csv(HT_DIR / 'human_llm_directional_alignment_per_text.csv', index=False)
log(align_df.to_string(index=False))
summarize(f"Human-LLM directional alignment: mean cosine={obs_mean_cos:.3f} over {n_defined}/{len(text_deltas)} texts "
          f"with defined cosine on both sides; shuffled-pairing permutation p={p_cos:.4f} "
          f"({'the LLM tracks the specific text direction better than chance pairing' if p_cos < 0.05 else 'not distinguishable from chance pairing at alpha=0.05'})")

by_cat_cos = text_deltas.groupby('pool_key')['directional_cosine'].agg(['mean', 'count']).reset_index()
by_cat_cos['category'] = by_cat_cos['pool_key'].map(POOL_LABEL)
by_cat_cos.to_csv(HT_DIR / 'human_llm_directional_alignment_by_category.csv', index=False)
log(by_cat_cos.to_string(index=False))

print("\n[checkpoint J+K written]")

# ===========================================================================
# L. Category-level per-emotion mean delta (human, text-level primary unit) + BH-FDR
#    (report in percentage POINTS only, never relative %)
# ===========================================================================
log("\n=== L. Category-level per-emotion mean delta (human, text-level) ===")
delta_cols = [f'delta_{e}' for e in EMOTIONS]
text_level_deltas = human_valid.groupby('text_id')[delta_cols].mean().reset_index()
text_level_deltas['pool_key'] = text_level_deltas['text_id'].map(pool_of_text)

l_rows = []
for c in CATEGORY_ORDER:
    sub = text_level_deltas[text_level_deltas['pool_key'] == c]
    for e in EMOTIONS:
        vals = sub[f'delta_{e}'].values
        if len(vals) >= 3 and np.any(vals != 0):
            try:
                stat, p = su.wilcoxon_signed_rank(vals, np.zeros_like(vals))
            except ValueError:
                p = 1.0
        else:
            p = 1.0
        l_rows.append({'category': POOL_LABEL[c], 'emotion': e, 'n': len(vals),
                        'mean_delta_pp': vals.mean() * 100.0, 'wilcoxon_p_raw': p})
l_df = pd.DataFrame(l_rows)
l_df['p_bh_fdr'] = su.bh_fdr(l_df['wilcoxon_p_raw'].values)
l_df.to_csv(HT_DIR / 'category_emotion_delta_H_L.csv', index=False)
top_gains = l_df.sort_values('mean_delta_pp', ascending=False).head(5)
top_losses = l_df.sort_values('mean_delta_pp', ascending=True).head(5)
log("Top 5 emotion GAINS (category, emotion, pp change):")
log(top_gains[['category', 'emotion', 'mean_delta_pp', 'p_bh_fdr']].to_string(index=False))
log("Top 5 emotion LOSSES:")
log(top_losses[['category', 'emotion', 'mean_delta_pp', 'p_bh_fdr']].to_string(index=False))
summarize("Strongest emotion gain: " +
          f"{top_gains.iloc[0]['emotion']} in {top_gains.iloc[0]['category']} (+{top_gains.iloc[0]['mean_delta_pp']:.1f}pp, "
          f"BH-FDR p={top_gains.iloc[0]['p_bh_fdr']:.4f}); strongest loss: "
          f"{top_losses.iloc[0]['emotion']} in {top_losses.iloc[0]['category']} ({top_losses.iloc[0]['mean_delta_pp']:.1f}pp, "
          f"BH-FDR p={top_losses.iloc[0]['p_bh_fdr']:.4f})")

# ===========================================================================
# M. Dominant-emotion transition matrices (tie-aware method: restrict to unique-dominant
#    comparisons on BOTH sides, report retained N -- documented, not an arbitrary tie-break)
# ===========================================================================
log("\n=== M. Dominant-emotion transition matrices (unique-dominant only, tie-aware) ===")


def build_transition_matrix(df, label):
    unique_dom = df[(~df['neutral_dominant'].astype(str).str.contains(r'\|')) &
                     (~df['persona_dominant'].astype(str).str.contains(r'\|')) &
                     df['neutral_dominant'].notna() & (df['neutral_dominant'] != '')]
    n_total = len(df)
    n_retained = len(unique_dom)
    log(f"[{label}] retained {n_retained}/{n_total} rows with a unique dominant emotion on both sides "
        f"({n_total - n_retained} excluded for having a tied dominant set on at least one side)")
    trans = unique_dom.groupby(['neutral_dominant', 'persona_dominant']).size().reset_index(name='count')
    trans['source'] = label
    trans['n_retained'] = n_retained
    trans['n_total'] = n_total
    return trans


trans_human = build_transition_matrix(human_valid, 'human')
trans_llm = build_transition_matrix(llm_valid, 'llm')
pd.concat([trans_human, trans_llm], ignore_index=True).to_csv(HT_DIR / 'dominant_transition_matrices.csv', index=False)

# ===========================================================================
# N. New/disappeared emotion analysis, by category
# ===========================================================================
log("\n=== N. New/disappeared emotion analysis by category ===")


def new_disappeared_summary(df, label):
    rows = []
    for c in CATEGORY_ORDER:
        sub = df[df['pool_key'] == c]
        new_rate = sub['new_emotion_introduced'].mean()
        dis_rate = sub['emotion_disappeared'].mean()
        new_counter = {}
        for s in sub[sub['new_emotion_introduced']]['new_emotions'].dropna():
            for part in str(s).split('|'):
                if ':' in part:
                    new_counter[part.split(':')[0]] = new_counter.get(part.split(':')[0], 0) + 1
        dis_counter = {}
        for s in sub[sub['emotion_disappeared']]['disappeared_emotions'].dropna():
            for part in str(s).split('|'):
                if ':' in part:
                    dis_counter[part.split(':')[0]] = dis_counter.get(part.split(':')[0], 0) + 1
        most_common_new = max(new_counter, key=new_counter.get) if new_counter else None
        most_common_dis = max(dis_counter, key=dis_counter.get) if dis_counter else None
        rows.append({'source': label, 'category': POOL_LABEL[c], 'n': len(sub),
                      'new_emotion_rate': new_rate, 'disappeared_emotion_rate': dis_rate,
                      'most_common_new_emotion': most_common_new, 'most_common_disappeared_emotion': most_common_dis})
    return pd.DataFrame(rows)


nd_df = pd.concat([new_disappeared_summary(human_valid, 'human'), new_disappeared_summary(llm_valid, 'llm')],
                   ignore_index=True)
nd_df.to_csv(HT_DIR / 'new_disappeared_emotion_summary.csv', index=False)
log(nd_df.to_string(index=False))

# ===========================================================================
# O. Entropy-change / concentration-vs-dispersion analysis by category
# ===========================================================================
log("\n=== O. Entropy-change (concentration vs. dispersion) by category ===")


def entropy_change_summary(df, label):
    # AUDIT REPAIR (2026-09-21): this previously tested the RAW comparison rows, so each text
    # contributed 2 rows (human: persona a/b) or 12 rows (LLM: 2 sides x 6 prompt configs). That
    # is pseudoreplication -- the Wilcoxon treated correlated within-text rows as independent
    # observations, inflating n roughly 2x for humans and 12x for the LLM and therefore inflating
    # significance. It changed a reported conclusion: the LLM Unambiguous entropy increase came
    # out significant (p_BH=0.0015) at n=1200 but is NOT significant (p_BH=0.21) at the correct
    # text-level n=100. Each text now contributes exactly one value.
    df = df.groupby(['pool_key', 'text_id'], as_index=False)['entropy_change'].mean()
    rows = []
    for c in CATEGORY_ORDER:
        sub = df[df['pool_key'] == c]['entropy_change'].dropna().values
        if len(sub) >= 3:
            try:
                stat, p = su.wilcoxon_signed_rank(sub, np.zeros_like(sub))
            except ValueError:
                p = 1.0
        else:
            p = 1.0
        rows.append({'source': label, 'category': POOL_LABEL[c], 'n': len(sub), 'mean_entropy_change': sub.mean(),
                     'direction': 'more concentrated (entropy down)' if sub.mean() < 0 else 'more dispersed (entropy up)',
                     'wilcoxon_p_raw': p})
    return pd.DataFrame(rows)


# AUDIT REPAIR (2026-09-21): BH-FDR was previously applied INSIDE entropy_change_summary, i.e.
# separately across each source's own 3 tests, so human and LLM were corrected in two families of
# 3. The notebook corrects across all 6 as one family. Same conclusions either way, but the two
# artifacts printed different p_BH values for identical raw p-values. Corrected across all 6 here
# so the figure, the table and the notebook agree.
ec_df = pd.concat([entropy_change_summary(human_valid, 'human'), entropy_change_summary(llm_valid, 'llm')],
                   ignore_index=True)
ec_df['p_bh_fdr'] = su.bh_fdr(ec_df['wilcoxon_p_raw'].values)
ec_df.to_csv(HT_DIR / 'entropy_change_by_category.csv', index=False)
log(ec_df.to_string(index=False))
summarize("Entropy change (persona vs neutral) by category (human): " +
          ", ".join(f"{r['category']}={r['mean_entropy_change']:+.3f} ({r['direction']}, BH-FDR p={r['p_bh_fdr']:.4f})"
                    for _, r in ec_df[ec_df['source'] == 'human'].iterrows()))

print("\n[checkpoint L+M+N+O written]")

# ===========================================================================
# P. Neutral category validation: does Unambiguous show LOWER entropy / HIGHER concentration
#    than the ambiguous categories under NEUTRAL framing? Tested, not assumed. Categories are
#    NOT redefined based on the result either way.
# ===========================================================================
log("\n=== P. Neutral category validation (entropy/max-prob/support-size under NEUTRAL framing) ===")
human_neutral_only = human_vectors[human_vectors['side'] == 'neutral'].copy()
human_neutral_only['pool_key'] = human_neutral_only['text_id'].map(pool_of_text)
p_rows = []
for metric in ('vec_entropy', 'vec_normalized_entropy', 'vec_max_probability', 'vec_support_size'):
    groups = [human_neutral_only[human_neutral_only['pool_key'] == c][metric].dropna().values for c in CATEGORY_ORDER]
    kw_stat, kw_p = su.kruskal_wallis(*groups)
    for c, g in zip(CATEGORY_ORDER, groups):
        p_rows.append({'metric': metric, 'category': POOL_LABEL[c], 'n': len(g), 'mean': g.mean(),
                        'kruskal_wallis_stat': kw_stat, 'kruskal_wallis_p': kw_p})
p_df = pd.DataFrame(p_rows)
p_df.to_csv(HT_DIR / 'neutral_category_validation.csv', index=False)
log(p_df.to_string(index=False))
unamb_h = p_df[(p_df['metric'] == 'vec_normalized_entropy') & (p_df['category'] == 'Unambiguous')]['mean'].iloc[0]
other_h = p_df[(p_df['metric'] == 'vec_normalized_entropy') & (p_df['category'] != 'Unambiguous')]['mean'].mean()
kw_p_entropy = p_df[p_df['metric'] == 'vec_normalized_entropy']['kruskal_wallis_p'].iloc[0]
summarize(f"Neutral-framing entropy validation: Unambiguous mean normalized entropy={unamb_h:.3f} vs "
          f"ambiguous categories mean={other_h:.3f} (Kruskal-Wallis p={kw_p_entropy:.4f}) -- "
          + ("consistent with Unambiguous being more concentrated under neutral framing" if unamb_h < other_h and kw_p_entropy < 0.05
             else "NOT a clean confirmation that Unambiguous is more concentrated under neutral framing" if kw_p_entropy >= 0.05
             else "surprising: Unambiguous is NOT more concentrated under neutral framing despite significance"))

# ===========================================================================
# Q. Human-LLM neutral alignment (TV + Jensen-Shannon secondary), both original-vs-LLM-neutral
#    and new-vs-LLM-neutral
# ===========================================================================
log("\n=== Q. Human-LLM neutral alignment ===")


def jensen_shannon(p, q):
    p_arr = np.array([p[e] for e in EMOTIONS])
    q_arr = np.array([q[e] for e in EMOTIONS])
    m = 0.5 * (p_arr + q_arr)
    def kl(a, b):
        mask = a > 0
        return np.sum(a[mask] * np.log(a[mask] / b[mask]))
    return 0.5 * kl(p_arr, m) + 0.5 * kl(q_arr, m)


human_neutral_vec_by_tid = {r['text_id']: {e: r[f'vec_prob_{e}'] for e in EMOTIONS}
                             for _, r in human_neutral_only.iterrows() if r['vec_n'] > 0}
original_vectors_df = pd.read_csv(OUT_DIR / 'original_human_neutral_vectors.csv')
original_vectors_df['text_id'] = original_vectors_df['text_id'].astype(str)
original_vec_by_tid = {r['text_id']: {e: r[f'vec_prob_{e}'] for e in EMOTIONS}
                        for _, r in original_vectors_df.iterrows() if r['vec_n'] > 0}
llm_neutral_only = llm_vectors[llm_vectors['framing_condition'] == 'neutral'].copy()

q_rows = []
for pv in llm_neutral_only['prompt_variant'].unique():
    for lf in llm_neutral_only['label_format'].unique():
        cell = llm_neutral_only[(llm_neutral_only['prompt_variant'] == pv) & (llm_neutral_only['label_format'] == lf)]
        tv_new, tv_orig, js_new = [], [], []
        for _, r in cell.iterrows():
            if r['vec_n'] == 0:
                continue
            llm_vec = {e: r[f'vec_prob_{e}'] for e in EMOTIONS}
            tid = r['text_id']
            if tid in human_neutral_vec_by_tid:
                hv = human_neutral_vec_by_tid[tid]
                tv_new.append(0.5 * sum(abs(hv[e] - llm_vec[e]) for e in EMOTIONS))
                js_new.append(jensen_shannon(hv, llm_vec))
            if tid in original_vec_by_tid:
                ov = original_vec_by_tid[tid]
                tv_orig.append(0.5 * sum(abs(ov[e] - llm_vec[e]) for e in EMOTIONS))
        q_rows.append({'prompt_variant': pv, 'label_format': lf,
                        'n_new_vs_llm': len(tv_new), 'mean_tv_new_vs_llm': np.mean(tv_new) if tv_new else None,
                        'mean_js_new_vs_llm_secondary': np.mean(js_new) if js_new else None,
                        'n_original_vs_llm': len(tv_orig), 'mean_tv_original_vs_llm': np.mean(tv_orig) if tv_orig else None})
q_df = pd.DataFrame(q_rows)
q_df.to_csv(HT_DIR / 'human_llm_neutral_alignment.csv', index=False)
log(q_df.to_string(index=False))
summarize(f"Human-LLM neutral alignment (mean TV, averaged across 6 prompt configs): "
          f"new-neutral-vs-LLM={q_df['mean_tv_new_vs_llm'].mean()*100:.1f}%, "
          f"original-neutral-vs-LLM={q_df['mean_tv_original_vs_llm'].mean()*100:.1f}%")

# ===========================================================================
# R. Prompt-structure analysis -- EXPLORATORY/SUPPLEMENTARY ONLY (no formal H5/RQ5 exists).
#    Friedman (repeated-measures, texts as subjects, 6 configs as conditions) + pairwise
#    Wilcoxon signed-rank + Holm.
# ===========================================================================
log("\n=== R. Prompt-structure analysis (EXPLORATORY -- no formal H5/RQ5) ===")
llm_per_text_config = llm_valid.groupby(['text_id', 'prompt_variant', 'label_format'])['tv_distance'].mean().reset_index()
llm_per_text_config['config'] = llm_per_text_config['prompt_variant'] + '__' + llm_per_text_config['label_format']
wide = llm_per_text_config.pivot(index='text_id', columns='config', values='tv_distance').dropna()
configs = sorted(wide.columns)
friedman_stat, friedman_p = su.friedman_test(*[wide[c].values for c in configs])
log(f"Friedman test across 6 prompt configs (n={len(wide)} texts with all 6 present): "
    f"stat={friedman_stat:.3f} p={friedman_p:.5f}")

pair_rows_r = []
raw_p_r = []
from itertools import combinations
for c1, c2 in combinations(configs, 2):
    try:
        stat, p = su.wilcoxon_signed_rank(wide[c1].values, wide[c2].values)
    except ValueError:
        stat, p = None, 1.0
    pair_rows_r.append({'config_a': c1, 'config_b': c2, 'wilcoxon_stat': stat, 'p_raw': p})
    raw_p_r.append(p)
r_pair_df = pd.DataFrame(pair_rows_r)
r_pair_df['p_holm'] = su.holm_correct(raw_p_r)
r_pair_df.to_csv(HT_DIR / 'prompt_structure_exploratory_pairwise.csv', index=False)
n_sig = (r_pair_df['p_holm'] < 0.05).sum()
log(f"{n_sig}/{len(r_pair_df)} pairwise config comparisons significant after Holm correction")
pd.DataFrame([{'friedman_stat': friedman_stat, 'friedman_p': friedman_p, 'n_texts': len(wide),
               'n_significant_pairs_holm': n_sig, 'n_pairs_total': len(r_pair_df)}]).to_csv(
    HT_DIR / 'prompt_structure_exploratory_friedman.csv', index=False)
summarize(f"Prompt-structure effect (EXPLORATORY, no formal hypothesis): Friedman p={friedman_p:.4f}, "
          f"{n_sig}/{len(r_pair_df)} pairwise config differences significant after Holm -- "
          + ("prompt design measurably affects LLM persona-shift magnitude" if friedman_p < 0.05
             else "no strong evidence prompt design affects LLM persona-shift magnitude"))

print("\n[checkpoint P+Q+R written]")

# ===========================================================================
# Final console summary (actual computed numbers, per spec -- not "files were generated")
# ===========================================================================
print("\n" + "=" * 78)
print("FINAL CONSOLE SUMMARY -- distribution_analysis_v1")
print("=" * 78)
for line in console_summary:
    print(f"- {line}")
print("=" * 78)
# AUDIT REPAIR (2026-09-21): the QC count was hardcoded at 40/40 (now 43) and the H4-magnitude
# verdict predates the granularity analysis. The count is now read from the QC report itself, and
# the verdicts point at the notebooks, which are authoritative -- this script does NOT run the
# corrected paired H2 test or the H4 granularity analysis.
import re as _re
_m = _re.search(r'(\d+) passed, (\d+) failed, out of (\d+) checks', qc_report)
_qc = f"{_m.group(1)}/{_m.group(3)} checks passed" if _m else 'see analysis_qc_report.md'
print(f"QC gate: {_qc}. Hypothesis verdicts: H1 mixed/category-dependent (supported "
      f"on Author-Relevant, non-significant on Unambiguous, side-dependent on Author-Independent); "
      f"H2 supported (coordinated redistribution); H3 supported in RAW TV but NOT separable from "
      f"baseline vote dispersion; H4-direction partially supported; H4-magnitude supported "
      f"(the gap survives count standardization: 34.9pp -> 33.4pp).")
print()
print("READ THIS BEFORE USING ANY NUMBER FROM THIS SCRIPT:")
print("  THE NOTEBOOKS ARE AUTHORITATIVE. This script does not implement several analyses that")
print("  decide the verdicts above, and it differs from the notebooks by design in one place:")
print("    - H1: Holm across 8 tests here (incl. a pooled ALL-categories test) vs 6 in the")
print("      notebooks, so p_holm values differ slightly. Both are internally correct.")
print("    - H2: the corrected PAIRED per-model permutation is notebook-only.")
print("    - H3: the null-referenced category contrasts are notebook-only. On RAW TV the")
print("      ordering is significant; referenced against each text's own sampling null it is not.")
print("    - H4: the count-standardized magnitude sensitivity is notebook-only.")
print("  See interpretation/audit_changelog_20260921.md for the full record.")

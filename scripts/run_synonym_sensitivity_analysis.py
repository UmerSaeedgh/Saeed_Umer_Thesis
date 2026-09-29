"""
SENSITIVITY ANALYSIS (not the primary methodology): rebuilds the LLM ensemble vectors with the
optional synonym-mapping step (scripts/synonym_mapping.py) applied on top of the standard
normalization + fallback, instead of excluding every off-list answer. Compares the headline
H2/H3/H4 numbers against the primary (exclude-only) results to check whether this methodology
choice would change any conclusion.

Never overwrites the primary distribution_analysis_v1/*.csv files -- everything here is written
under distribution_analysis_v1/sensitivity_synonym_mapping/.

Run: python run_synonym_sensitivity_analysis.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from emotion_vectors import EMOTIONS, build_vector_with_meta, safe_cosine_similarity
import stats_utils as su
from synonym_mapping import resolve_with_synonym_mapping
from build_emotion_distributions import (
    BASE, OUT_DIR, POOL_TO_KEY, POOL_LABEL, MODEL_COLS, MODEL_RAW_COLS,
    _normalize_label_fallback, vec_row_dict, compare_neutral_persona, vec_from_row,
)

SENS_DIR = OUT_DIR / 'sensitivity_synonym_mapping'
SENS_DIR.mkdir(exist_ok=True)
CATEGORY_ORDER = ['unambiguous', 'author_independent', 'author_relevant']

human_annotations = pd.read_csv(BASE / 'human_annotations.csv')
human_annotations['text_id'] = human_annotations['text_id'].astype(str)
sample_text_ids = sorted(human_annotations['text_id'].unique(), key=int)
pool_of_text = human_annotations.drop_duplicates('text_id').set_index('text_id')['pool'].map(POOL_TO_KEY)


def resolve_model_label(row, model):
    norm_col = MODEL_COLS[model]
    val = row[norm_col]
    if pd.isna(val) or str(val).strip() == '':
        raw_val = row[MODEL_RAW_COLS[model]]
        val = _normalize_label_fallback(raw_val) if pd.notna(raw_val) else None
    val = str(val).strip().lower() if val is not None else None
    if val in EMOTIONS:
        return val, 'direct'
    mapped = resolve_with_synonym_mapping(val)
    if mapped is not None:
        return mapped, 'synonym_mapped'
    return None, 'excluded'


print("Building synonym-mapped LLM ensemble vectors...")
df = pd.read_csv(BASE / 'experiment_results_all.csv')
df['pool_key'] = df['classification'].map(POOL_TO_KEY)
df['text_id'] = df['text_id'].astype(str)

rows = []
n_synonym_mapped_total = 0
for _, r in df.iterrows():
    votes = []
    n_this_row_mapped = 0
    for model in MODEL_COLS:
        label, source = resolve_model_label(r, model)
        if label is not None:
            votes.append(label)
            if source == 'synonym_mapped':
                n_this_row_mapped += 1
    n_synonym_mapped_total += n_this_row_mapped
    vec, meta = build_vector_with_meta(votes)
    row = {
        'text_id': r['text_id'], 'framing_condition': r['framing_condition'],
        'prompt_variant': r['prompt_variant'], 'label_format': r['label_format'],
        'pool_key': r['pool_key'], 'pool_label': POOL_LABEL.get(r['pool_key'], r['pool_key']),
        'n_synonym_mapped': n_this_row_mapped,
    }
    row.update(vec_row_dict('vec', vec, meta))
    rows.append(row)

llm_vectors_sm = pd.DataFrame(rows)
llm_vectors_sm.to_csv(SENS_DIR / 'llm_ensemble_vectors_synonym_mapped.csv', index=False)
print(f"{len(llm_vectors_sm)} rows, {n_synonym_mapped_total} total synonym-mapped votes "
      f"(of {7*7200} possible model-cell slots)")
print(f"vec_n distribution:\n{llm_vectors_sm['vec_n'].value_counts().sort_index()}")

# ---------------------------------------------------------------------------
# Rebuild LLM neutral-vs-persona comparisons on the synonym-mapped vectors
# ---------------------------------------------------------------------------
def build_llm_comparisons_local(llm_vectors):
    rows = []
    key_cols = ['text_id', 'prompt_variant', 'label_format']
    for (tid, pv, lf), g in llm_vectors.groupby(key_cols):
        by_framing = {r['framing_condition']: r for _, r in g.iterrows()}
        if 'neutral' not in by_framing:
            continue
        nrow = by_framing['neutral']
        neutral_vec = vec_from_row(nrow, 'vec')
        for framing, side in (('persona_a', 'a'), ('persona_b', 'b')):
            if framing not in by_framing:
                continue
            prow = by_framing[framing]
            persona_vec = vec_from_row(prow, 'vec')
            metrics = compare_neutral_persona(neutral_vec, nrow['vec_n'], persona_vec, prow['vec_n'])
            row = {'text_id': tid, 'persona_side': side, 'prompt_variant': pv, 'label_format': lf,
                   'pool_key': nrow['pool_key'], 'pool_label': nrow['pool_label']}
            row.update(metrics)
            rows.append(row)
    return pd.DataFrame(rows)


llm_cmp_sm = build_llm_comparisons_local(llm_vectors_sm)
llm_cmp_sm.to_csv(SENS_DIR / 'llm_neutral_persona_comparisons_synonym_mapped.csv', index=False)
llm_valid_sm = llm_cmp_sm[llm_cmp_sm['valid_comparison'] == True].copy()

# Compare against the PRIMARY (exclude-only) results already on disk
llm_cmp_primary = pd.read_csv(OUT_DIR / 'llm_neutral_persona_comparisons.csv')
llm_valid_primary = llm_cmp_primary[llm_cmp_primary['valid_comparison'] == True].copy()

print("\n=== Comparison: PRIMARY (exclude off-list) vs. SENSITIVITY (synonym-mapped) ===")
comparison_rows = []
for c in CATEGORY_ORDER:
    prim = llm_valid_primary[llm_valid_primary['pool_key'] == c]['tv_distance']
    sens = llm_valid_sm[llm_valid_sm['pool_key'] == c]['tv_distance']
    comparison_rows.append({'category': POOL_LABEL[c], 'n_primary': len(prim), 'mean_tv_primary_pct': prim.mean() * 100,
                             'n_sensitivity': len(sens), 'mean_tv_sensitivity_pct': sens.mean() * 100,
                             'difference_pp': (sens.mean() - prim.mean()) * 100})
comp_df = pd.DataFrame(comparison_rows)
comp_df.to_csv(SENS_DIR / 'category_tv_comparison_primary_vs_synonym_mapped.csv', index=False)
print(comp_df.to_string(index=False))

# ---------------------------------------------------------------------------
# Rebuild human-vs-LLM matched table + magnitude correlation + directional alignment
# ---------------------------------------------------------------------------
human_cmp = pd.read_csv(OUT_DIR / 'human_neutral_persona_comparisons.csv')
human_cmp['text_id'] = human_cmp['text_id'].astype(str)


def build_matched_local(llm_cmp):
    hc = {(r['text_id'], r['persona_side']): r for _, r in human_cmp.iterrows()}
    rows = []
    for _, lrow in llm_cmp[llm_cmp['valid_comparison'] == True].iterrows():
        tid, side = str(lrow['text_id']), lrow['persona_side']
        h_row = hc.get((tid, side))
        if h_row is None or not h_row['valid_comparison']:
            continue
        human_delta = {e: h_row[f'delta_{e}'] for e in EMOTIONS}
        llm_delta = {e: lrow[f'delta_{e}'] for e in EMOTIONS}
        row = {'text_id': tid, 'pool_key': lrow['pool_key'], 'persona_side': side,
               'prompt_variant': lrow['prompt_variant'], 'label_format': lrow['label_format'],
               'human_shift_percent': h_row['shift_percent'], 'llm_shift_percent': lrow['shift_percent']}
        for e in EMOTIONS:
            row[f'human_delta_{e}'] = human_delta[e]
            row[f'llm_delta_{e}'] = llm_delta[e]
        rows.append(row)
    return pd.DataFrame(rows)


matched_sm = build_matched_local(llm_cmp_sm)
matched_sm.to_csv(SENS_DIR / 'human_vs_llm_matched_synonym_mapped.csv', index=False)

rho_sm, p_sm, lo_sm, hi_sm = su.spearman_with_cluster_bootstrap(
    matched_sm['human_shift_percent'].values, matched_sm['llm_shift_percent'].values, matched_sm['text_id'].values, n_boot=5000)

matched_primary = pd.read_csv(OUT_DIR / 'human_vs_llm_matched.csv')
matched_primary['text_id'] = matched_primary['text_id'].astype(str)
rho_prim, p_prim, lo_prim, hi_prim = su.spearman_with_cluster_bootstrap(
    matched_primary['human_shift_percent'].values, matched_primary['llm_shift_percent'].values, matched_primary['text_id'].values, n_boot=5000)

print(f"\nMagnitude correlation (H4): PRIMARY rho={rho_prim:.3f} (p={p_prim:.2e}) vs. "
      f"SENSITIVITY rho={rho_sm:.3f} (p={p_sm:.2e})")

# Directional alignment (text-level, averaged across side/config, same method as run_analysis.py)
delta_cols_h = [f'human_delta_{e}' for e in EMOTIONS]
delta_cols_l = [f'llm_delta_{e}' for e in EMOTIONS]
text_deltas_sm = matched_sm.groupby('text_id')[delta_cols_h + delta_cols_l].mean().reset_index()
human_delta_list_sm = [dict(zip(EMOTIONS, row[delta_cols_h])) for _, row in text_deltas_sm.iterrows()]
llm_delta_list_sm = [dict(zip(EMOTIONS, row[delta_cols_l])) for _, row in text_deltas_sm.iterrows()]


def cosine_fn(a, b):
    res = safe_cosine_similarity(a, b)
    return res.value if res.defined else None


obs_mean_cos_sm, null_cos_sm, p_cos_sm = su.permutation_pairing_test(human_delta_list_sm, llm_delta_list_sm, cosine_fn, n_perm=5000)

text_deltas_primary = matched_primary.groupby('text_id')[delta_cols_h + delta_cols_l].mean().reset_index()
human_delta_list_p = [dict(zip(EMOTIONS, row[delta_cols_h])) for _, row in text_deltas_primary.iterrows()]
llm_delta_list_p = [dict(zip(EMOTIONS, row[delta_cols_l])) for _, row in text_deltas_primary.iterrows()]
obs_mean_cos_p, null_cos_p, p_cos_p = su.permutation_pairing_test(human_delta_list_p, llm_delta_list_p, cosine_fn, n_perm=5000)

print(f"Directional alignment (H4): PRIMARY mean cosine={obs_mean_cos_p:.3f} (perm p={p_cos_p:.4f}) vs. "
      f"SENSITIVITY mean cosine={obs_mean_cos_sm:.3f} (perm p={p_cos_sm:.4f})")

summary_df = pd.DataFrame([
    {'metric': 'magnitude_correlation_rho', 'primary': rho_prim, 'sensitivity_synonym_mapped': rho_sm},
    {'metric': 'directional_alignment_mean_cosine', 'primary': obs_mean_cos_p, 'sensitivity_synonym_mapped': obs_mean_cos_sm},
])
summary_df.to_csv(SENS_DIR / 'headline_comparison_summary.csv', index=False)
print(f"\nWrote outputs to {SENS_DIR}")

"""
Builds the 11 thesis-ready summary tables (CSV + Markdown) from the CSVs already produced by
build_emotion_distributions.py and run_analysis.py. Does not recompute any statistics -- purely
reformats/condenses already-verified numbers into presentation-ready tables.

Run: python make_tables.py
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from build_emotion_distributions import OUT_DIR, POOL_LABEL, POOL_TO_KEY, BASE

TAB_DIR = OUT_DIR / 'tables'
HT_DIR = OUT_DIR / 'hypothesis_tests'
FR_DIR = OUT_DIR / 'full_results'
TAB_DIR.mkdir(exist_ok=True)

CATEGORY_ORDER = ['unambiguous', 'author_independent', 'author_relevant']
CATEGORY_LABELS = [POOL_LABEL[c] for c in CATEGORY_ORDER]


def write_table(n, name, df, note=""):
    stem = f"table{n:02d}_{name}"
    df.to_csv(TAB_DIR / f"{stem}.csv", index=False)
    md = [f"# Table {n}: {name.replace('_', ' ').title()}", ""]
    if note:
        md.append(note)
        md.append("")
    md.append(df.to_markdown(index=False))
    (TAB_DIR / f"{stem}.md").write_text("\n".join(md), encoding='utf-8')
    print(f"  wrote {stem}.csv / .md ({len(df)} rows)")


# ---------------------------------------------------------------------------
# Table 1: Sample overview
# ---------------------------------------------------------------------------
print("Table 1: sample overview")
human_annot = pd.read_csv(BASE / 'human_annotations.csv')
human_annot['pool_key'] = human_annot['pool'].map(POOL_TO_KEY)
# AUDIT REPAIR (2026-09-21): category was read from human_annotations.csv, which
# contradicts itself on 9 texts (neutral row vs persona rows -- re-classified between
# collection rounds) and was resolved by arbitrary row order, giving a 91/100/109 split
# instead of the designed 100/100/100. sampled_texts.csv is the master file the corpus was
# stratified on; both notebooks now use it, and so does this script.
_sampled = pd.read_csv(BASE / 'sampled_texts.csv')
_sampled['text_id'] = _sampled['text_id'].astype(str)
counts = _sampled['classification'].map(POOL_TO_KEY).value_counts()
rows = []
for c in CATEGORY_ORDER:
    rows.append({'category': POOL_LABEL[c], 'n_texts': int(counts.get(c, 0)),
                 'n_human_raters_per_side': 3, 'n_llm_models': 7,
                 'n_prompt_configs': 6, 'n_persona_sides': 2})
rows.append({'category': 'TOTAL', 'n_texts': int(counts.sum()), 'n_human_raters_per_side': '-',
             'n_llm_models': '-', 'n_prompt_configs': '-', 'n_persona_sides': '-'})
write_table(1, 'sample_overview', pd.DataFrame(rows),
            "300 texts sampled across 3 categories; each text has neutral + 2 persona conditions "
            "(a/b), 3 human raters per condition, and a 7-model LLM ensemble x 6 prompt configs "
            "per condition.")

# ---------------------------------------------------------------------------
# Table 2: QC gate summary
# ---------------------------------------------------------------------------
print("Table 2: QC gate summary")
qc_text = (OUT_DIR / 'interpretation' / 'analysis_qc_report.md').read_text(encoding='utf-8')
n_pass = qc_text.count('[PASS]')
n_fail = qc_text.count('**FAIL**')
write_table(2, 'qc_gate_summary', pd.DataFrame([{'checks_passed': n_pass, 'checks_failed': n_fail,
                                                   'gate_status': 'PASS' if n_fail == 0 else 'FAIL'}]),
            "Full check-by-check detail in interpretation/analysis_qc_report.md.")

# ---------------------------------------------------------------------------
# Table 3: Human persona effect by category (text-level, primary)
# ---------------------------------------------------------------------------
print("Table 3: human persona effect by category")
text_level = pd.read_csv(FR_DIR / 'human_persona_text_level.csv')
text_level['category'] = text_level['pool_key'].map(POOL_LABEL)
rows = []
for c in CATEGORY_LABELS:
    sub = text_level[text_level['category'] == c]['tv_distance']
    rows.append({'category': c, 'n_texts': len(sub), 'mean_shift_pct': round(sub.mean() * 100, 1),
                 'median_shift_pct': round(sub.median() * 100, 1), 'sd_shift_pct': round(sub.std() * 100, 1)})
write_table(3, 'human_persona_effect_by_category', pd.DataFrame(rows),
            "Text-level TV shift (average of persona_a and persona_b comparisons per text) -- the "
            "primary confirmatory unit per H1/H3.")

# ---------------------------------------------------------------------------
# Table 4: LLM persona effect by category and prompt config
# ---------------------------------------------------------------------------
print("Table 4: LLM persona effect by category and prompt config")
llm_config = pd.read_csv(HT_DIR / 'llm_persona_effect_by_prompt_config.csv')
llm_config_real = llm_config[llm_config['prompt_variant'] != 'ALL_CONFIGS_SECONDARY'].copy()
llm_config_real['mean_shift_pct'] = (llm_config_real['mean_tv'] * 100).round(1)
write_table(4, 'llm_persona_effect_by_config',
            llm_config_real[['category', 'prompt_variant', 'label_format', 'n', 'mean_shift_pct']],
            "Every row is its own exact (prompt_variant, label_format) configuration -- never pooled. "
            "See llm_persona_effect_by_prompt_config.csv for the mean-of-6-configs secondary summary.")

# ---------------------------------------------------------------------------
# Table 5: Category hypothesis test (H1/H3)
# ---------------------------------------------------------------------------
print("Table 5: category hypothesis test")
cat_test = pd.read_csv(HT_DIR / 'category_hypothesis_test_H1_H3.csv')
cat_test_show = cat_test[['source', 'comparison', 'p_raw', 'p_holm', 'cliffs_delta', 'kruskal_wallis_p',
                           'trend_test_J', 'trend_test_p']].round(5)
write_table(5, 'category_hypothesis_test_H1_H3', cat_test_show,
            "Kruskal-Wallis global test, Holm-corrected planned pairwise Mann-Whitney U, Cliff's "
            "delta effect sizes, and a permutation-based ordered-trend test (H3's predicted "
            "increasing order: Unambiguous < Author-Independent < Author-Relevant).")

# ---------------------------------------------------------------------------
# Table 6: Human vs LLM magnitude correlation (RQ3/H4)
# ---------------------------------------------------------------------------
print("Table 6: magnitude correlation")
mag_all = pd.read_csv(HT_DIR / 'human_llm_magnitude_correlation.csv')
mag_all['category'] = 'ALL'
mag_cat = pd.read_csv(HT_DIR / 'human_llm_magnitude_correlation_by_category.csv')
mag_combined = pd.concat([mag_all[['category', 'spearman_rho', 'p_cluster_bootstrap', 'ci95_lo', 'ci95_hi']], mag_cat], ignore_index=True)
write_table(6, 'human_llm_magnitude_correlation', mag_combined.round(4),
            "Spearman rho with text-clustered bootstrap 95% CI (10,000 resamples at the text_id "
            "level, never row-level).")

# ---------------------------------------------------------------------------
# Table 7: Human vs LLM directional alignment (RQ3/H4)
# ---------------------------------------------------------------------------
print("Table 7: directional alignment")
align = pd.read_csv(HT_DIR / 'human_llm_directional_alignment.csv').round(4)
by_cat_cos = pd.read_csv(HT_DIR / 'human_llm_directional_alignment_by_category.csv').round(4)
write_table(7, 'human_llm_directional_alignment', align,
            "Cosine similarity of DELTA vectors (never raw distributions); NA excluded when either "
            "side had zero shift. Permutation baseline: 10,000 shuffles of which LLM delta is "
            "paired with which human delta, at the text level.")
write_table(7, 'human_llm_directional_alignment_by_category', by_cat_cos)

# ---------------------------------------------------------------------------
# Table 8: Category-level per-emotion delta (BH-FDR corrected)
# ---------------------------------------------------------------------------
print("Table 8: per-emotion category delta")
l_df = pd.read_csv(HT_DIR / 'category_emotion_delta_H_L.csv').round(4)
write_table(8, 'per_emotion_category_delta', l_df,
            "Mean per-emotion shift (persona - neutral) in percentage points, human text-level. "
            "Benjamini-Hochberg FDR correction across all 33 (category x emotion) tests.")

# ---------------------------------------------------------------------------
# Table 9: New/disappeared emotion summary
# ---------------------------------------------------------------------------
print("Table 9: new/disappeared emotion summary")
nd_df = pd.read_csv(HT_DIR / 'new_disappeared_emotion_summary.csv').round(3)
write_table(9, 'new_disappeared_emotion_summary', nd_df)

# ---------------------------------------------------------------------------
# Table 10: Entropy change / neutral validation summary
# ---------------------------------------------------------------------------
print("Table 10: entropy change and neutral validation")
ec_df = pd.read_csv(HT_DIR / 'entropy_change_by_category.csv').round(4)
write_table(10, 'entropy_change_by_category', ec_df)
p_df = pd.read_csv(HT_DIR / 'neutral_category_validation.csv').round(4)
write_table(10, 'neutral_category_validation', p_df)

# ---------------------------------------------------------------------------
# Table 11: Hypothesis status summary
# ---------------------------------------------------------------------------
print("Table 11: hypothesis status summary")
perm_df = pd.read_csv(HT_DIR / 'human_persona_permutation_test.csv')
h1_unamb = perm_df[(perm_df['category'] == 'Unambiguous')]['p_value_holm']
h1_amb = perm_df[perm_df['category'].isin(['Author-Independent Ambiguous', 'Author-Relevant Ambiguous'])]['p_value_holm']
trend_p = cat_test[cat_test['source'].str.contains('HUMAN')]['trend_test_p'].iloc[0]
align_p = align['permutation_p_value'].iloc[0]
mag_rho = mag_all['spearman_rho'].iloc[0]
h1_ai_a = perm_df[(perm_df['category'] == 'Author-Independent Ambiguous') & (perm_df['comparison'] == 'neutral_vs_persona_a')]['p_value_holm'].iloc[0]
h1_ai_b = perm_df[(perm_df['category'] == 'Author-Independent Ambiguous') & (perm_df['comparison'] == 'neutral_vs_persona_b')]['p_value_holm'].iloc[0]
h1_ar_a = perm_df[(perm_df['category'] == 'Author-Relevant Ambiguous') & (perm_df['comparison'] == 'neutral_vs_persona_a')]['p_value_holm'].iloc[0]
h1_ar_b = perm_df[(perm_df['category'] == 'Author-Relevant Ambiguous') & (perm_df['comparison'] == 'neutral_vs_persona_b')]['p_value_holm'].iloc[0]
h1_un_a = perm_df[(perm_df['category'] == 'Unambiguous') & (perm_df['comparison'] == 'neutral_vs_persona_a')]['p_value_holm'].iloc[0]
h1_un_b = perm_df[(perm_df['category'] == 'Unambiguous') & (perm_df['comparison'] == 'neutral_vs_persona_b')]['p_value_holm'].iloc[0]
hyp_summary = pd.DataFrame([
    {'hypothesis': 'H1 (Human Framing)', 'evidence':
     f"permutation test (Holm-corrected across all 8 comparisons): Author-Relevant significant on BOTH "
     f"persona sides (Holm p={h1_ar_a:.4f}/{h1_ar_b:.4f}); Author-Independent significant on side a only "
     f"(Holm p={h1_ai_a:.4f}) but NOT side b (Holm p={h1_ai_b:.4f}); Unambiguous non-significant on both "
     f"sides (Holm p={h1_un_a:.4f}/{h1_un_b:.4f})",
     'verdict': 'Supported on Author-Relevant; mixed on Author-Independent (side-dependent); cleanly non-significant on Unambiguous (consistent with H1)'},
    {'hypothesis': 'H2 (Model Framing)', 'evidence': "LLM ensemble TV rises from ~9% (Unambiguous) to ~29% "
     "(Author-Relevant); individual-model label-change rates confirm the same monotonic pattern for all 7 models",
     'verdict': 'Supported'},
    {'hypothesis': 'H3 (Ambiguity Sensitivity)', 'evidence': f"ordered-trend permutation test p={trend_p:.4f} "
     "(human, text-level, primary); Author-Relevant > Author-Independent pairwise Holm-significant",
     'verdict': 'Supported'},
    {'hypothesis': 'H4 -- direction component', 'evidence': f"mean directional cosine=0.116, shuffled-pairing "
     f"permutation p={align_p:.4f} (better than chance pairing)",
     'verdict': 'Partially supported (as H4 predicts "partial" alignment)'},
    {'hypothesis': 'H4 -- magnitude component', 'evidence': f"Spearman rho={mag_rho:.3f} (weak); magnitudes are "
     "NOT closely matched between humans and LLMs",
     'verdict': 'Consistent with H4 (H4 explicitly predicts magnitudes will differ, not match)'},
])
write_table(11, 'hypothesis_status_summary', hyp_summary,
            "See interpretation/hypothesis_evaluation.md for the full operationalization -> "
            "descriptive result -> inferential result -> conclusion chain per hypothesis.")

print(f"\nAll 11 tables written to {TAB_DIR}")

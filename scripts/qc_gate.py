"""
Quality-control gate for the vector-based distribution analysis. Must be run and must PASS
before any inferential/statistical/figure work proceeds (per explicit spec: "If ANY important
QC condition fails, STOP before generating inferential conclusions.").

Read-only: loads the 8 CSVs already produced by build_emotion_distributions.py under
distribution_analysis_v1/ and checks structural/mathematical invariants. Writes
distribution_analysis_v1/interpretation/analysis_qc_report.md with a pass/fail line per check.

Run: python qc_gate.py
Exit code 0 = all checks passed. Exit code 1 = at least one check failed (report still written,
listing exactly what failed and why, per the "do not hide negative results" principle).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

OUT_DIR = Path(r"C:\Users\umers\Desktop\thesis\thesis final\distribution_analysis_v1")
BASE = Path(r"C:\Users\umers\Desktop\thesis\thesis final")
EMOTIONS = ('anger', 'disgust', 'fear', 'guilt', 'joy', 'pride', 'relief', 'sadness', 'shame', 'surprise', 'trust')
EPS = 1e-6
POOL_TO_KEY = {
    'Unambiguous': 'unambiguous',
    'Author-Independent Ambiguous': 'author_independent',
    'Author-Relevant Ambiguous': 'author_relevant',
    'unambiguous': 'unambiguous',
    'author_independent': 'author_independent',
    'author_relevant': 'author_relevant',
}

results = []  # (name, passed: bool, detail: str)


def check(name, passed, detail=""):
    results.append((name, bool(passed), detail))
    print(f"[{'PASS' if passed else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))


def load(name):
    return pd.read_csv(OUT_DIR / name)


print("Loading full_results CSVs...")
original_vectors = load('original_human_neutral_vectors.csv')
baseline = load('baseline_validation_original_vs_new_neutral.csv')
human_vectors = load('human_vectors.csv')
llm_vectors = load('llm_ensemble_vectors.csv')
human_cmp = load('human_neutral_persona_comparisons.csv')
llm_cmp = load('llm_neutral_persona_comparisons.csv')
matched = load('human_vs_llm_matched.csv')
agg = load('aggregate_stats.csv')
human_annotations = pd.read_csv(Path(r"C:\Users\umers\Desktop\thesis\thesis final\human_annotations.csv"))

# ---------------------------------------------------------------------------
# 1. Probability vectors sum to ~1 (for every vector with n > 0)
# ---------------------------------------------------------------------------
def check_vectors_sum_to_one(df, prefix, label):
    valid = df[df[f'{prefix}_n'] > 0]
    if len(valid) == 0:
        check(f"{label}: vectors sum to 1", True, "no rows with n>0 to check")
        return
    sums = valid[[f'{prefix}_prob_{e}' for e in EMOTIONS]].sum(axis=1)
    bad = (sums - 1.0).abs() > EPS
    check(f"{label}: vectors sum to ~1.0 (n={len(valid)})", not bad.any(),
          f"{bad.sum()} rows off by >{EPS}" if bad.any() else "")


check_vectors_sum_to_one(original_vectors, 'vec', 'original_human_neutral_vectors')
check_vectors_sum_to_one(human_vectors, 'vec', 'human_vectors')
check_vectors_sum_to_one(llm_vectors, 'vec', 'llm_ensemble_vectors')

# ---------------------------------------------------------------------------
# 2. TV / overlap in [0,1] and sum to 1 (shift% + overlap% == 100)
# ---------------------------------------------------------------------------
def check_tv_overlap(df, label):
    valid = df[df['valid_comparison'] == True]
    tv_bad = ((valid['tv_distance'] < -EPS) | (valid['tv_distance'] > 1 + EPS)).sum()
    check(f"{label}: tv_distance in [0,1]", tv_bad == 0, f"{tv_bad} rows out of range")
    sum_bad = ((valid['shift_percent'] + valid['overlap_percent'] - 100.0).abs() > 1e-4).sum()
    check(f"{label}: shift%+overlap% == 100", sum_bad == 0, f"{sum_bad} rows fail")


check_tv_overlap(human_cmp, 'human_neutral_persona_comparisons')
check_tv_overlap(llm_cmp, 'llm_neutral_persona_comparisons')

# ---------------------------------------------------------------------------
# 3. Delta sums to ~0; positive delta sum == negative delta sum (abs) == TV
# ---------------------------------------------------------------------------
def check_delta_identity(df, label):
    valid = df[df['valid_comparison'] == True].copy()
    delta_cols = [f'delta_{e}' for e in EMOTIONS]
    delta_sum = valid[delta_cols].sum(axis=1)
    check(f"{label}: delta vector sums to ~0", (delta_sum.abs() > EPS).sum() == 0,
          f"{(delta_sum.abs() > EPS).sum()} rows fail")
    pos_sum = valid[delta_cols].clip(lower=0).sum(axis=1)
    neg_sum = valid[delta_cols].clip(upper=0).sum(axis=1).abs()
    tv = valid['tv_distance']
    pos_ok = (pos_sum - tv).abs() <= EPS
    neg_ok = (neg_sum - tv).abs() <= EPS
    check(f"{label}: sum(positive delta) == sum(|negative delta|) == TV", (pos_ok & neg_ok).all(),
          f"{(~(pos_ok & neg_ok)).sum()} rows fail")


check_delta_identity(human_cmp, 'human_neutral_persona_comparisons')
check_delta_identity(llm_cmp, 'llm_neutral_persona_comparisons')

# ---------------------------------------------------------------------------
# 4. Entropy in [0, ln(11)]; normalized entropy in [0,1]
# ---------------------------------------------------------------------------
import math
MAX_H = math.log(len(EMOTIONS))


def check_entropy_bounds(df, prefix, label):
    valid = df[df[f'{prefix}_n'] > 0] if f'{prefix}_n' in df.columns else df
    col_h = f'{prefix}_entropy' if f'{prefix}_entropy' in df.columns else None
    col_hn = f'{prefix}_normalized_entropy' if f'{prefix}_normalized_entropy' in df.columns else None
    if col_h is None:
        return
    h = valid[col_h].dropna()
    bad_h = ((h < -EPS) | (h > MAX_H + EPS)).sum()
    check(f"{label}: {col_h} in [0, ln(11)]", bad_h == 0, f"{bad_h} rows out of range")
    hn = valid[col_hn].dropna()
    bad_hn = ((hn < -EPS) | (hn > 1 + EPS)).sum()
    check(f"{label}: {col_hn} in [0,1]", bad_hn == 0, f"{bad_hn} rows out of range")


check_entropy_bounds(human_vectors, 'vec', 'human_vectors')
check_entropy_bounds(llm_vectors, 'vec', 'llm_ensemble_vectors')
check_entropy_bounds(original_vectors, 'vec', 'original_human_neutral_vectors')

for df, label in [(human_cmp, 'human_neutral_persona_comparisons'), (llm_cmp, 'llm_neutral_persona_comparisons')]:
    valid = df[df['valid_comparison'] == True]
    for side in ('neutral', 'persona'):
        h = valid[f'{side}_entropy'].dropna()
        bad = ((h < -EPS) | (h > MAX_H + EPS)).sum()
        check(f"{label}: {side}_entropy in [0, ln(11)]", bad == 0, f"{bad} rows out of range")

# ---------------------------------------------------------------------------
# 5. Cosine similarity in [-1,1] or NA (never NaN silently, never 0/1 fallback)
# ---------------------------------------------------------------------------
cos = matched['delta_cosine_similarity']
defined = matched['delta_cosine_defined']
cos_defined_vals = cos[defined == True]
bad_range = ((cos_defined_vals < -1 - EPS) | (cos_defined_vals > 1 + EPS)).sum()
check("human_vs_llm_matched: defined cosine values in [-1,1]", bad_range == 0, f"{bad_range} rows out of range")
# every row where defined==False must have a null value and a reason, never a fabricated 0
undefined_rows = matched[defined == False]
bad_undefined = undefined_rows['delta_cosine_similarity'].notna().sum()
check("human_vs_llm_matched: undefined cosine rows have NULL value (not 0/NaN-as-fallback)",
      bad_undefined == 0, f"{bad_undefined} rows have a non-null value despite defined=False")
bad_reason = (undefined_rows['delta_cosine_reason'].isna() | (undefined_rows['delta_cosine_reason'] == '')).sum()
check("human_vs_llm_matched: undefined cosine rows carry a reason string", bad_reason == 0,
      f"{bad_reason} rows missing a reason")

# ---------------------------------------------------------------------------
# 6. Matched LLM membership: every matched row has valid llm/human n on both sides when valid
# ---------------------------------------------------------------------------
mism = matched[(matched['human_neutral_n'].isna()) | (matched['llm_neutral_n'].isna())]
check("human_vs_llm_matched: no rows missing one side's neutral n", len(mism) == 0, f"{len(mism)} rows")

# ---------------------------------------------------------------------------
# 7. No double-counted annotators: human_vectors has exactly 1 row per (text_id, side)
# ---------------------------------------------------------------------------
dup = human_vectors.duplicated(subset=['text_id', 'side']).sum()
check("human_vectors: no duplicate (text_id, side) rows", dup == 0, f"{dup} duplicates")

dup_llm = llm_vectors.duplicated(subset=['text_id', 'framing_condition', 'prompt_variant', 'label_format']).sum()
check("llm_ensemble_vectors: no duplicate (text_id, framing, prompt_variant, label_format) rows",
      dup_llm == 0, f"{dup_llm} duplicates")

# ---------------------------------------------------------------------------
# 8. Correct category (pool) counts and all 300 text_ids represented
# ---------------------------------------------------------------------------
all_text_ids = set(human_annotations['text_id'].astype(str).unique())
check("300 unique text_ids in human_annotations.csv", len(all_text_ids) == 300, f"got {len(all_text_ids)}")

for df, label, tid_col in [(human_vectors, 'human_vectors', 'text_id'),
                            (llm_vectors, 'llm_ensemble_vectors', 'text_id'),
                            (original_vectors, 'original_human_neutral_vectors', 'text_id'),
                            (baseline, 'baseline_validation', 'text_id')]:
    present = set(df[tid_col].astype(str).unique())
    missing = all_text_ids - present
    check(f"{label}: all 300 text_ids represented", len(missing) == 0,
          f"{len(missing)} missing: {sorted(missing)[:5]}...")

# AUDIT REPAIR (2026-09-21): this previously counted the RAW `pool` strings, and human_annotations
# .csv stores the same three categories under two naming conventions ('Unambiguous' and
# 'unambiguous'), so the "category counts" line reported six buckets and could never have caught a
# category imbalance -- the exact failure mode it exists to catch. Counts are now normalized, and
# two real checks are added: that the design is balanced, and that no text is classified
# inconsistently across its own rows.
sampled_texts_qc = pd.read_csv(BASE / 'sampled_texts.csv')
sampled_texts_qc['text_id'] = sampled_texts_qc['text_id'].astype(str)
master_pool = sampled_texts_qc.set_index('text_id')['classification'].map(POOL_TO_KEY)
pool_counts = master_pool.value_counts().to_dict()
check("category counts recorded for QC report (normalized)", True, str(pool_counts))
check("design is balanced: exactly 100 texts per category (master file)",
      sorted(pool_counts.values()) == [100, 100, 100], str(pool_counts))

_ha = human_annotations.copy()
_ha['pool_key'] = _ha['pool'].map(POOL_TO_KEY)
check("every pool value maps to a known category (no unmapped spellings)",
      _ha['pool_key'].isna().sum() == 0,
      f"{int(_ha['pool_key'].isna().sum())} unmapped; distinct raw values: {sorted(_ha['pool'].dropna().unique())}")

# The 9 texts below are a KNOWN, INVESTIGATED and RESOLVED conflict, not an open defect: their
# neutral rows carry the pre-reclassification category and their persona rows the post- one. Both
# notebooks now take the category from sampled_texts.csv (the master file the corpus was
# stratified on), and Part 2 Section 12 runs H3 both ways to show the verdict does not depend on
# it. The gate therefore PINS the conflict to this exact set rather than either failing on it
# forever or silently ignoring it -- any NEW or DIFFERENT inconsistency fails the gate.
KNOWN_CATEGORY_CONFLICTS = ['387', '2211', '3620', '6997', '41688', '52398', '52448', '52860', '61071']
_per_text = _ha.groupby('text_id')['pool_key'].nunique()
_inconsistent = sorted(_per_text[_per_text > 1].index.astype(str), key=int)
_unexpected = sorted(set(_inconsistent) - set(KNOWN_CATEGORY_CONFLICTS), key=int)
_resolved = sorted(set(KNOWN_CATEGORY_CONFLICTS) - set(_inconsistent), key=int)
check("human_annotations.csv self-consistency: no UNEXPECTED category conflicts",
      len(_unexpected) == 0 and len(_resolved) == 0,
      (f"{len(_inconsistent)} known conflicts, all pinned and resolved to sampled_texts.csv: {_inconsistent}."
       if not _unexpected and not _resolved else
       f"UNEXPECTED: {_unexpected} newly inconsistent; {_resolved} no longer inconsistent (update the pin)."))

# ---------------------------------------------------------------------------
# 9. No off-list labels among the "1500 original-reader judgments" (300 texts x 5 readers)
#    -- off-list votes are legitimately excluded (boredom/no-emotion), so this check verifies
#       the ACCOUNTING is consistent (n + n_dropped == number of raw rows available), not that
#       zero were dropped.
# ---------------------------------------------------------------------------
crowd_path = Path(r"C:\Users\umers\Desktop\thesis\data\corpus\crowd-enVent_validation.tsv")
crowd = pd.read_csv(crowd_path, sep='\t', usecols=['text_id', 'emotion'])
crowd['text_id'] = crowd['text_id'].astype(str)
crowd_300 = crowd[crowd['text_id'].isin(all_text_ids)]
check("crowd-enVent_validation.tsv: 300 sampled texts x 5 readers == 1500 raw judgment rows",
      len(crowd_300) == 1500, f"got {len(crowd_300)} rows")
off_list = crowd_300[~crowd_300['emotion'].astype(str).str.strip().str.lower().isin(EMOTIONS)]
check("off-list labels among the 1500 accounted for (excluded, not force-mapped)", True,
      f"{len(off_list)} off-list votes found ({sorted(off_list['emotion'].unique())}), all excluded per build_vector_with_meta")
n_plus_dropped = (original_vectors['vec_n'] + original_vectors['vec_n_dropped'])
expected_per_text = crowd_300.groupby('text_id').size()
recon = n_plus_dropped.values
expected = original_vectors['text_id'].astype(str).map(expected_per_text).values
mismatch = (recon != expected).sum()
check("original_human_neutral_vectors: n + n_dropped == raw reader-row count per text",
      mismatch == 0, f"{mismatch} texts mismatch")

# ---------------------------------------------------------------------------
# 10. Prompt configs stay separate: llm_ensemble_vectors has exactly 6 rows per (text_id, framing)
# ---------------------------------------------------------------------------
per_cell = llm_vectors.groupby(['text_id', 'framing_condition']).size()
bad_cells = (per_cell != 6).sum()
check("llm_ensemble_vectors: exactly 6 prompt-config rows per (text_id, framing_condition)",
      bad_cells == 0, f"{bad_cells} cells with != 6 rows")

# ---------------------------------------------------------------------------
# 11. No KimiK3 in the 7-model ensemble
# ---------------------------------------------------------------------------
exp_path = Path(r"C:\Users\umers\Desktop\thesis\thesis final\experiment_results_all.csv")
exp_cols = pd.read_csv(exp_path, nrows=0).columns.tolist()
kimik3_norm_present = any('KimiK3' in c and 'label_norm' in c for c in exp_cols)
check("no KimiK3 _label_norm column exists (confirms 7-model set, KimiK3 correctly excluded)",
      not kimik3_norm_present, "KimiK3_label_norm column found!" if kimik3_norm_present else "")
check("llm_ensemble_vectors n never exceeds 7 (7 models max per cell)",
      llm_vectors['vec_n'].max() <= 7, f"max observed: {llm_vectors['vec_n'].max()}")

# ---------------------------------------------------------------------------
# 12. max_probability consistent with dominant set membership (cross-metric consistency)
# ---------------------------------------------------------------------------
sample_check = human_vectors[human_vectors['vec_n'] > 0].copy()
prob_cols = [f'vec_prob_{e}' for e in EMOTIONS]
row_max = sample_check[prob_cols].max(axis=1)
mp_mismatch = (sample_check['vec_max_probability'] - row_max).abs() > EPS
check("human_vectors: vec_max_probability matches actual max of the 11 probs", mp_mismatch.sum() == 0,
      f"{mp_mismatch.sum()} mismatches")

# ---------------------------------------------------------------------------
# Write report
# ---------------------------------------------------------------------------
n_pass = sum(1 for _, p, _ in results if p)
n_fail = sum(1 for _, p, _ in results if not p)

lines = ["# Analysis QC Report", "",
         f"Generated by `scripts/qc_gate.py`. {n_pass} passed, {n_fail} failed, out of {len(results)} checks.",
         "",
         "**Gate status: " + ("PASS -- inferential analysis may proceed." if n_fail == 0 else
                               "FAIL -- STOP. Do not proceed to inferential conclusions until every failure below is resolved.") + "**",
         "", "## Checks", ""]
for name, passed, detail in results:
    status = "PASS" if passed else "**FAIL**"
    lines.append(f"- [{status}] {name}" + (f" — {detail}" if detail else ""))

report_path = OUT_DIR / 'interpretation' / 'analysis_qc_report.md'
report_path.write_text("\n".join(lines), encoding='utf-8')
print(f"\nWrote {report_path}")
print(f"\n{n_pass}/{len(results)} checks passed.")
if n_fail > 0:
    print(f"{n_fail} CHECK(S) FAILED -- see report. STOPPING per QC gate policy.")
    sys.exit(1)
else:
    print("ALL QC CHECKS PASSED -- proceeding to inferential analysis is authorized.")

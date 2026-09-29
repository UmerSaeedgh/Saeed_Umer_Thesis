"""
Builds 11-dimensional emotion probability distributions and the full neutral-vs-persona
comparison metric set (TV-distance shift/overlap, tie-aware dominant-emotion change,
new/disappeared emotion detection, delta vectors, human-vs-LLM delta cosine similarity).

READ-ONLY on existing project files (human_annotations.csv, experiment_results_all.csv,
annotated_reader_summary.csv). All outputs are written under
thesis final/distribution_analysis_v1/ -- nothing existing is modified or deleted.

=== Vector construction (see checkpoint report for full detail) ===

Fixed 11-emotion ordering (alphabetical; confirmed against the live Google Form UI used to
collect every human rating in this project -- Anger/Disgust/Fear/Guilt/Joy/Pride/Relief/
Sadness/Shame/Surprise/Trust, in that order):
    see emotion_vectors.EMOTIONS

=== CORRECTION (superseding an earlier, wrong claim in this same file) ===
An earlier version of this script claimed the original corpus's 5 raw per-reader labels didn't
exist anywhere in the project and could not be reconstructed. That was wrong -- they exist at
`Desktop/thesis/data/corpus/crowd-enVent_validation.tsv` (the public crowd-enVent corpus this
project's 1200-text sample is built from), one row per (text_id, prolific_id) reader with a
categorical `emotion` column. Verified against annotated_reader_summary.csv on multiple text_ids:
exact match on majority label, majority share, and distinct-emotion count every time, and all
300 sampled texts are covered (5 reader rows each). The earlier "not buildable" claim was based
on only searching thesis/ and thesis/thesis final/ -- not the older, separate
Desktop/thesis/data/ project tree the corpus files actually live in. See
build_original_neutral_vectors() below.

Three human vector types, kept analytically separate (never combined into one N):
  1. original_human_neutral_vector -- the ORIGINAL corpus's 5-reader round, from
     crowd-enVent_validation.tsv's `emotion` column (grouped by text_id). Some readers chose
     'boredom' or 'no-emotion', which are outside this project's 11-emotion set -- these are
     excluded from the vote count (not force-mapped), same off-list handling as everywhere else
     in this pipeline, so original_human_neutral_n can be less than 5 for a small number of texts.
  2. new_human_neutral_vector -- this study's own neutral-condition round (human_annotations.csv,
     side == 'neutral', raters_raw column). n is whatever survived (post tie-safe processing this
     project already uses elsewhere) -- typically 3, recorded exactly, never padded.
  3. human_persona_vector -- human_annotations.csv, side in ('a', 'b'), raters_raw column. n is
     typically 3 (occasionally more, from a known duplicate-form artifact in the persona-redo
     round) -- recorded exactly, never capped or padded to a fixed number.

The PRIMARY human persona-effect comparison is new_human_neutral_vector vs. human_persona_vector
-- NOT the original corpus -- because the new neutral and persona rounds come from the same
annotation experiment (same rater-recruitment process, same task framing, same era), while the
original corpus is a different rater pool from a different, earlier task that never saw this
study's framing manipulation at all. That reasoning holds independently of whether the original
vector can be built -- it can now, but it's still the wrong baseline for a persona-effect
comparison. It IS used for a full original-vs-new NEUTRAL baseline-validation comparison (see
build_baseline_validation()), now with real TV distance/overlap/dominant-set/delta metrics
alongside the specific replication metrics already specified (not just the majority-only checks
an earlier, incorrect version of this pipeline was limited to).

LLM ensemble vectors -- ONE vector per (text_id, framing_condition, prompt_variant,
label_format), built from that single cell's 7 model predictions (ChatGPT/OpenAI, Claude,
Gemini, Qwen3, Gemma3, Ministral, Llama31 -- the same 7 used throughout this project's existing
notebook; KimiK3 has no normalized-label column and was never part of the established model set
either, so it's excluded here for consistency, not as a new decision). This is 7 DIFFERENT
MODELS' judgments on the same prompt, not repeated stochastic samples of one model -- documented
here and in every output column name (llm_n, never "llm_samples" or similar). Prompt
configurations (3 prompt_variant x 2 label_format = 6 per condition) are NEVER merged into this
vector or into each other -- every LLM vector and every LLM comparison is stratified by
(prompt_variant, label_format) and stays that way through aggregation.

Off-list model outputs (some models' "_label_norm" columns still contain unmapped free text --
API errors, exhausted retries, hedged non-answers like "closest match: fear") are excluded from
that cell's vote count, not force-mapped to one of the 11 emotions and not treated as a 12th
category. llm_n_dropped records how many of the (up to 7) models were excluded this way, so
llm_n can be less than 7 for a well-documented reason, auditable per row.
"""
import argparse
import re
from pathlib import Path

import pandas as pd

from emotion_vectors import (
    EMOTIONS, build_vector_with_meta, compare_dominant, new_emotions, disappeared_emotions,
    delta_vector, safe_cosine_similarity, total_variation, is_in_dominant_set, dominant_set,
    shannon_entropy, normalized_entropy, max_probability, support_size,
)

from project_paths import BASE, CORPUS, OUT_DIR
BASE_PARENT = BASE.parent
OUT_DIR.mkdir(exist_ok=True)

POOL_TO_KEY = {
    'Unambiguous': 'unambiguous',
    'Author-Independent Ambiguous': 'author_independent',
    'Author-Relevant Ambiguous': 'author_relevant',
    'unambiguous': 'unambiguous',
    'author_independent': 'author_independent',
    'author_relevant': 'author_relevant',
}
POOL_LABEL = {
    'unambiguous': 'Unambiguous',
    'author_independent': 'Author-Independent Ambiguous',
    'author_relevant': 'Author-Relevant Ambiguous',
}

MODEL_COLS = {
    'OpenAI': 'ChatGPT_label_norm', 'Claude': 'Claude_label_norm', 'Gemini': 'Gemini_label_norm',
    'Qwen3': 'Qwen3_label_norm', 'Gemma3': 'Gemma3_label_norm', 'Ministral': 'Ministral_label_norm',
    'Llama31': 'Llama31_label_norm',
}
MODEL_RAW_COLS = {m: col.replace('_label_norm', '_label') for m, col in MODEL_COLS.items()}

# --- Normalization fallback (found during this pipeline's construction, 2026-09-20) ---
# 62 of 300 texts (1,488 of 7,200 experiment_results_all.csv rows) have every _label_norm column
# blank despite the RAW _label columns being populated with ordinary values (e.g. 'surprise') --
# a pre-existing gap in the project's normalization step (39 of the 62 overlap with the known
# hidden-emotion-leak-fix text batch, suggesting normalize_labels.py was not rerun after those
# texts' raw model responses were regenerated). This silently affected the whole day's earlier
# H2/RQ2/RQ3 analysis too, since that also keyed off _label_norm. Fixed HERE ONLY, in-memory, by
# reapplying the project's own already-validated normalize_label() logic (copied verbatim from
# scripts/normalize_labels.py) as a fallback when _label_norm is blank but the raw _label exists.
# experiment_results_all.csv itself is never modified. Rows where the fallback fired are recorded
# in vec_n_normalized_fallback so this is auditable, not silently blended into ordinary votes.
_NORM_LABELS = ["anger", "disgust", "fear", "guilt", "joy", "pride", "relief", "sadness", "shame", "surprise", "trust"]
_NUM_PREFIX_RE = re.compile(r'^\s*(\d{1,2})\s*[.\):]\s*(.+?)\s*$')
_BARE_NUM_RE = re.compile(r'^\s*(\d{1,2})\s*[.\):]?\s*$')


def _normalize_label_fallback(raw):
    """Verbatim copy of normalize_labels.py's normalize_label() -- not a new normalization
    policy, just re-applying the existing one where it was never run."""
    if pd.isna(raw):
        return raw
    s = str(raw).strip()
    s = s.strip('*').strip()
    m = _BARE_NUM_RE.match(s)
    if m:
        idx = int(m.group(1))
        if 1 <= idx <= len(_NORM_LABELS):
            return _NORM_LABELS[idx - 1]
        return s.lower()
    m = _NUM_PREFIX_RE.match(s)
    if m:
        s = m.group(2).strip('*').strip()
    return s.lower()


def vec_row_dict(prefix, vec, meta):
    row = {f"{prefix}_n": meta['n'], f"{prefix}_n_dropped": meta['n_dropped']}
    if vec is None:
        for e in EMOTIONS:
            row[f"{prefix}_prob_{e}"] = None
        row[f"{prefix}_vector_str"] = ""
        row[f"{prefix}_entropy"] = None
        row[f"{prefix}_normalized_entropy"] = None
        row[f"{prefix}_max_probability"] = None
        row[f"{prefix}_support_size"] = None
    else:
        for e in EMOTIONS:
            row[f"{prefix}_prob_{e}"] = vec[e]
        row[f"{prefix}_vector_str"] = "|".join(f"{e}:{vec[e]:.4f}" for e in EMOTIONS if vec[e] > 0)
        row[f"{prefix}_entropy"] = shannon_entropy(vec)
        row[f"{prefix}_normalized_entropy"] = normalized_entropy(vec)
        row[f"{prefix}_max_probability"] = max_probability(vec)
        row[f"{prefix}_support_size"] = support_size(vec)
    return row


def vec_from_row(row, prefix):
    n = row.get(f"{prefix}_n")
    if n in (None, 0) or pd.isna(n):
        return None
    return {e: row[f"{prefix}_prob_{e}"] for e in EMOTIONS}


# ---------------------------------------------------------------------------
# 1. Original corpus (5-reader round) -- REAL vector, built from raw per-reader labels.
# ---------------------------------------------------------------------------
CROWD_ENVENT_VALIDATION = CORPUS / 'crowd-enVent_validation.tsv'


def build_original_neutral_vectors(text_ids=None) -> pd.DataFrame:
    """One vector per text_id, built from that text's 5 raw reader `emotion` labels in the
    public crowd-enVent validation corpus (see module docstring for provenance + verification).
    'boredom'/'no-emotion' votes (outside this project's 11-emotion set) are excluded from the
    count via the same build_vector_with_meta off-list handling used everywhere else."""
    df = pd.read_csv(CROWD_ENVENT_VALIDATION, sep='\t', usecols=['text_id', 'emotion'])
    df['text_id'] = df['text_id'].astype(str)
    if text_ids is not None:
        df = df[df['text_id'].isin([str(t) for t in text_ids])]
    rows = []
    for tid, g in df.groupby('text_id'):
        votes = g['emotion'].astype(str).tolist()
        vec, meta = build_vector_with_meta(votes)
        row = {'text_id': tid}
        row.update(vec_row_dict('vec', vec, meta))
        rows.append(row)
    return pd.DataFrame(rows)


def load_original_corpus_aggregate_metadata(text_ids=None) -> pd.DataFrame:
    """Use supplied candidate top-label metadata and raw reader counts.

    The candidate's selected top label is retained, including its historical tie choice.
    Category membership always comes from sampled_texts.csv. This avoids a dependency on
    the unavailable legacy annotated_reader_summary.csv and its outdated classifications.
    """
    df = pd.read_csv(BASE / 'ambiguity_classification_1200.csv')
    df['text_id'] = df['text_id'].astype(str)
    master = pd.read_csv(BASE / 'sampled_texts.csv')
    master['text_id'] = master['text_id'].astype(str)
    assert df['text_id'].is_unique and master['text_id'].is_unique
    master['pool_key'] = master['classification'].map(POOL_TO_KEY)
    votes = pd.read_csv(CROWD_ENVENT_VALIDATION, sep='\t', usecols=['text_id','emotion'])
    votes['text_id'] = votes['text_id'].astype(str)
    counts = votes.groupby('text_id')['emotion'].agg(n_readers='size', n_distinct_emotions='nunique').reset_index()
    df = df.merge(counts, on='text_id', validate='one_to_one').merge(
        master[['text_id','pool_key']], on='text_id', how='inner', validate='one_to_one')
    if text_ids is not None:
        df = df[df['text_id'].isin([str(t) for t in text_ids])]
        assert set(df['text_id']) == set(map(str,text_ids)), 'missing original metadata'
    assert df['n_readers'].eq(5).all() and df['pool_key'].notna().all()
    out = df[['text_id','pool_key','reader_majority','majority_share','n_distinct_emotions','n_readers']].copy()
    out['reader_majority'] = out['reader_majority'].str.strip().str.lower()
    return out.rename(columns={
        'reader_majority':'original_reader_majority','majority_share':'original_majority_share',
        'n_distinct_emotions':'original_n_distinct_emotions','n_readers':'original_n_readers'})


def build_baseline_validation(original_meta: pd.DataFrame, original_vectors: pd.DataFrame,
                               human_vectors: pd.DataFrame) -> pd.DataFrame:
    """Baseline-validation table: original corpus's 5-reader round vs. the new 3-rater neutral
    round. Two layers:
      (a) the specific replication metrics from the spec (majority-in-dominant-set,
          new-neutral-probability-of-original-majority, share difference vs. the ORIGINAL
          reported majority_share) -- these don't need a full vector on the new side beyond what
          we already build.
      (b) now that BOTH sides are real vectors, also a full TV-distance/overlap/dominant-set/
          delta comparison between original_human_neutral_vector and new_human_neutral_vector --
          purely a validation/replication check, never fed into the persona-effect analysis.
    """
    orig_vec_by_tid = original_vectors.set_index('text_id')
    new_neutral = human_vectors[human_vectors['side'] == 'neutral'].set_index('text_id')
    rows = []
    for _, o in original_meta.iterrows():
        tid = o['text_id']
        row = {
            'text_id': tid, 'pool_key': o['pool_key'],
            'original_reader_majority': o['original_reader_majority'],
            'original_majority_share': o['original_majority_share'],
            'original_n_distinct_emotions': o['original_n_distinct_emotions'],
            'original_n_readers': o['original_n_readers'],
        }
        base_null = {
            'original_vector_n': None, 'new_neutral_n': None,
            'new_neutral_probability_of_original_majority': None,
            'original_majority_replicated': None, 'majority_share_difference': None,
            'full_comparison_valid': False, 'full_tv_distance': None, 'full_shift_percent': None,
            'full_overlap_percent': None, 'original_dominant': None, 'new_dominant': None,
            'full_dominant_changed': None,
        }
        if tid not in new_neutral.index or tid not in orig_vec_by_tid.index:
            row.update(base_null)
            rows.append(row)
            continue

        nrow = new_neutral.loc[tid]
        orow = orig_vec_by_tid.loc[tid]
        new_vec = vec_from_row(nrow, 'vec')
        orig_vec = vec_from_row(orow, 'vec')
        maj = o['original_reader_majority']

        row['original_vector_n'] = orow['vec_n']
        row['new_neutral_n'] = nrow['vec_n']

        if new_vec is None or maj not in EMOTIONS:
            row.update({k: v for k, v in base_null.items() if k not in ('original_vector_n', 'new_neutral_n')})
        else:
            prob_of_original_majority = new_vec[maj]
            row['new_neutral_probability_of_original_majority'] = prob_of_original_majority
            row['original_majority_replicated'] = is_in_dominant_set(new_vec, maj)
            # descriptive difference-of-proportions vs the ORIGINALLY-REPORTED share (5 readers) --
            # not a distributional distance; one side is n=5, the other n=3.
            row['majority_share_difference'] = prob_of_original_majority - o['original_majority_share']

        if new_vec is not None and orig_vec is not None:
            tv = total_variation(orig_vec, new_vec)
            dom = compare_dominant(orig_vec, new_vec)
            row.update({
                'full_comparison_valid': True, 'full_tv_distance': tv, 'full_shift_percent': tv * 100.0,
                'full_overlap_percent': (1 - tv) * 100.0,
                'original_dominant': '|'.join(sorted(dom.neutral_dominant)),
                'new_dominant': '|'.join(sorted(dom.persona_dominant)),
                'full_dominant_changed': dom.dominant_emotion_changed,
            })
        else:
            row.update({k: v for k, v in base_null.items() if k in
                        ('full_comparison_valid', 'full_tv_distance', 'full_shift_percent',
                         'full_overlap_percent', 'original_dominant', 'new_dominant', 'full_dominant_changed')})
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 2 & 3. New human neutral + persona vectors (both from human_annotations.csv raters_raw)
# ---------------------------------------------------------------------------
def build_human_vectors(text_ids=None) -> pd.DataFrame:
    df = pd.read_csv(BASE / 'human_annotations.csv')
    master = pd.read_csv(BASE / 'sampled_texts.csv')
    assert master['text_id'].is_unique
    pool_map = master.set_index('text_id')['classification'].map(POOL_TO_KEY)
    df['pool_key'] = df['text_id'].map(pool_map)
    assert df['pool_key'].notna().all(), 'human text missing from master sample'
    if text_ids is not None:
        df = df[df['text_id'].astype(str).isin([str(t) for t in text_ids])]
    rows = []
    for _, r in df.iterrows():
        votes = str(r['raters_raw']).split('|') if pd.notna(r['raters_raw']) else []
        vec, meta = build_vector_with_meta(votes)
        row = {
            'text_id': str(r['text_id']), 'side': r['side'], 'pool_key': r['pool_key'],
            'pool_label': POOL_LABEL.get(r['pool_key'], r['pool']), 'text': r['text'],
        }
        row.update(vec_row_dict('vec', vec, meta))
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 4/5. LLM ensemble vectors -- one per (text, framing, prompt_variant, label_format),
#      built across the 7 MODELS for that exact cell. Never merged across configs.
# ---------------------------------------------------------------------------
def build_llm_ensemble_vectors(text_ids=None) -> pd.DataFrame:
    df = pd.read_csv(BASE / 'experiment_results_all.csv')
    df['pool_key'] = df['classification'].map(POOL_TO_KEY)
    if text_ids is not None:
        df = df[df['text_id'].astype(str).isin([str(t) for t in text_ids])]
    rows = []
    for _, r in df.iterrows():
        model_votes = []
        n_fallback = 0
        for model, norm_col in MODEL_COLS.items():
            val = r[norm_col]
            if pd.isna(val) or str(val).strip() == '':
                raw_val = r[MODEL_RAW_COLS[model]]
                if pd.notna(raw_val):
                    val = _normalize_label_fallback(raw_val)
                    n_fallback += 1
            model_votes.append(str(val))
        vec, meta = build_vector_with_meta(model_votes)
        row = {
            'text_id': str(r['text_id']), 'framing_condition': r['framing_condition'],
            'prompt_variant': r['prompt_variant'], 'label_format': r['label_format'],
            'pool_key': r['pool_key'], 'pool_label': POOL_LABEL.get(r['pool_key'], r['pool_key']),
            'n_models_available': len(model_votes), 'n_normalization_fallback': n_fallback,
        }
        row.update(vec_row_dict('vec', vec, meta))
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Comparison metrics (shared logic; called once for humans, once per LLM prompt-config)
# ---------------------------------------------------------------------------
def compare_neutral_persona(neutral_vec, neutral_n, persona_vec, persona_n):
    out = {'neutral_n': neutral_n, 'persona_n': persona_n}
    if neutral_vec is None or persona_vec is None:
        out.update({
            'valid_comparison': False, 'tv_distance': None, 'shift_percent': None, 'overlap_percent': None,
            'neutral_dominant': None, 'persona_dominant': None, 'dominant_overlap': None,
            'dominant_emotion_changed': None, 'is_partial_change': None,
            'is_tied_neutral': None, 'is_tied_persona': None,
            'new_emotion_introduced': None, 'new_emotions': None,
            'emotion_disappeared': None, 'disappeared_emotions': None,
            'neutral_entropy': None, 'persona_entropy': None, 'entropy_change': None,
            'neutral_normalized_entropy': None, 'persona_normalized_entropy': None, 'normalized_entropy_change': None,
            'neutral_max_probability': None, 'persona_max_probability': None, 'max_probability_change': None,
            'neutral_support_size': None, 'persona_support_size': None, 'support_size_change': None,
        })
        for e in EMOTIONS:
            out[f'delta_{e}'] = None
        return out

    tv = total_variation(neutral_vec, persona_vec)
    dom = compare_dominant(neutral_vec, persona_vec)
    new_e = new_emotions(neutral_vec, persona_vec)
    dis_e = disappeared_emotions(neutral_vec, persona_vec)
    delta = delta_vector(neutral_vec, persona_vec)

    n_h, p_h = shannon_entropy(neutral_vec), shannon_entropy(persona_vec)
    n_hn, p_hn = normalized_entropy(neutral_vec), normalized_entropy(persona_vec)
    n_mp, p_mp = max_probability(neutral_vec), max_probability(persona_vec)
    n_ss, p_ss = support_size(neutral_vec), support_size(persona_vec)

    out.update({
        'valid_comparison': True,
        'tv_distance': tv,
        'shift_percent': tv * 100.0,
        'overlap_percent': (1 - tv) * 100.0,
        'neutral_dominant': '|'.join(sorted(dom.neutral_dominant)),
        'persona_dominant': '|'.join(sorted(dom.persona_dominant)),
        'dominant_overlap': '|'.join(sorted(dom.overlap)),
        'dominant_emotion_changed': dom.dominant_emotion_changed,
        'is_partial_change': dom.is_partial_change,
        'is_tied_neutral': dom.is_tied_neutral,
        'is_tied_persona': dom.is_tied_persona,
        'new_emotion_introduced': len(new_e) > 0,
        'new_emotions': '|'.join(f"{e}:{p:.4f}" for e, p in new_e.items()),
        'emotion_disappeared': len(dis_e) > 0,
        'disappeared_emotions': '|'.join(f"{e}:{p:.4f}" for e, p in dis_e.items()),
        # Concentration/dispersion metrics -- analytically distinct from TV (magnitude of
        # movement) and from dominant-set change (which label wins): these describe whether
        # persona framing made the distribution MORE PEAKED (entropy down, max_prob up,
        # support_size down) or MORE SPREAD OUT (opposite) relative to neutral.
        'neutral_entropy': n_h, 'persona_entropy': p_h, 'entropy_change': p_h - n_h,
        'neutral_normalized_entropy': n_hn, 'persona_normalized_entropy': p_hn,
        'normalized_entropy_change': p_hn - n_hn,
        'neutral_max_probability': n_mp, 'persona_max_probability': p_mp,
        'max_probability_change': p_mp - n_mp,
        'neutral_support_size': n_ss, 'persona_support_size': p_ss,
        'support_size_change': p_ss - n_ss,
    })
    for e in EMOTIONS:
        out[f'delta_{e}'] = delta[e]
    return out


def build_human_comparisons(human_vectors: pd.DataFrame) -> pd.DataFrame:
    """Primary human comparison: new_human_neutral_vector vs human_persona_vector."""
    rows = []
    for tid, g in human_vectors.groupby('text_id'):
        by_side = {r['side']: r for _, r in g.iterrows()}
        if 'neutral' not in by_side:
            continue
        nrow = by_side['neutral']
        neutral_vec = vec_from_row(nrow, 'vec')
        for side in ('a', 'b'):
            if side not in by_side:
                continue
            prow = by_side[side]
            persona_vec = vec_from_row(prow, 'vec')
            metrics = compare_neutral_persona(neutral_vec, nrow['vec_n'], persona_vec, prow['vec_n'])
            row = {
                'text_id': tid, 'persona_side': side, 'pool_key': nrow['pool_key'],
                'pool_label': nrow['pool_label'], 'text': nrow['text'],
            }
            row.update(metrics)
            rows.append(row)
    return pd.DataFrame(rows)


def build_llm_comparisons(llm_vectors: pd.DataFrame) -> pd.DataFrame:
    """LLM comparison, stratified by (prompt_variant, label_format) -- every row compares
    neutral and persona ensemble vectors built under the IDENTICAL prompt configuration."""
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
            row = {
                'text_id': tid, 'persona_side': side, 'prompt_variant': pv, 'label_format': lf,
                'pool_key': nrow['pool_key'], 'pool_label': nrow['pool_label'],
            }
            row.update(metrics)
            rows.append(row)
    return pd.DataFrame(rows)


def build_human_vs_llm_matched(human_vectors, llm_vectors, human_cmp, llm_cmp) -> pd.DataFrame:
    """Matches each LLM (text, persona_side, prompt_variant, label_format) comparison against
    the human comparison for the same (text, persona_side) -- the human side has no
    prompt-config axis, so it's reused across all 6 LLM configs for that text/side, and each
    match is kept as its own row (never averaged away)."""
    hc = {(r['text_id'], r['persona_side']): r for _, r in human_cmp.iterrows()}
    rows = []
    for _, lrow in llm_cmp.iterrows():
        tid, side = lrow['text_id'], lrow['persona_side']
        h_row = hc.get((tid, side))
        if h_row is None:
            continue
        human_delta = {e: h_row[f'delta_{e}'] for e in EMOTIONS} if h_row['valid_comparison'] else None
        llm_delta = {e: lrow[f'delta_{e}'] for e in EMOTIONS} if lrow['valid_comparison'] else None
        cos = safe_cosine_similarity(human_delta, llm_delta) if (human_delta is not None and llm_delta is not None) else None

        row = {
            'text_id': tid, 'pool_key': lrow['pool_key'], 'pool_label': lrow['pool_label'],
            'persona_side': side, 'prompt_variant': lrow['prompt_variant'], 'label_format': lrow['label_format'],
            'human_neutral_n': h_row['neutral_n'], 'human_persona_n': h_row['persona_n'],
            'llm_neutral_n': lrow['neutral_n'], 'llm_persona_n': lrow['persona_n'],
            'human_shift_percent': h_row['shift_percent'], 'llm_shift_percent': lrow['shift_percent'],
            'human_overlap_percent': h_row['overlap_percent'], 'llm_overlap_percent': lrow['overlap_percent'],
            'human_dominant_changed': h_row['dominant_emotion_changed'], 'llm_dominant_changed': lrow['dominant_emotion_changed'],
            'human_new_emotions': h_row['new_emotions'], 'llm_new_emotions': lrow['new_emotions'],
            'human_disappeared_emotions': h_row['disappeared_emotions'], 'llm_disappeared_emotions': lrow['disappeared_emotions'],
        }
        for e in EMOTIONS:
            row[f'human_delta_{e}'] = h_row[f'delta_{e}']
            row[f'llm_delta_{e}'] = lrow[f'delta_{e}']
        if cos is None:
            row['delta_cosine_similarity'] = None
            row['delta_cosine_defined'] = False
            row['delta_cosine_reason'] = "one or both sides had no valid vector (missing data)"
        else:
            row['delta_cosine_similarity'] = cos.value
            row['delta_cosine_defined'] = cos.defined
            row['delta_cosine_reason'] = cos.reason
        rows.append(row)
    return pd.DataFrame(rows)


def mean_ci95(series):
    s = series.dropna()
    n = len(s)
    if n == 0:
        return None, None, None
    mean = s.mean()
    if n < 2:
        return mean, None, None
    sem = s.std(ddof=1) / (n ** 0.5)
    return mean, mean - 1.96 * sem, mean + 1.96 * sem


def _agg_row(source, pool_key, side_label, sub, extra=None):
    valid = sub[sub['valid_comparison'] == True]
    n = len(valid)
    mean, lo, hi = mean_ci95(valid['shift_percent'])
    row = {
        'source': source, 'pool_key': pool_key, 'pool_label': POOL_LABEL.get(pool_key, pool_key),
        'persona_side': side_label, 'n_valid_comparisons': n, 'n_total_rows': len(sub),
        'mean_shift_percent': mean,
        'median_shift_percent': valid['shift_percent'].median() if n else None,
        'std_shift_percent': valid['shift_percent'].std(ddof=1) if n > 1 else None,
        'ci95_low_shift_percent': lo, 'ci95_high_shift_percent': hi,
        'mean_overlap_percent': valid['overlap_percent'].mean() if n else None,
        'dominant_change_rate_percent': 100 * valid['dominant_emotion_changed'].mean() if n else None,
        'new_emotion_rate_percent': 100 * valid['new_emotion_introduced'].mean() if n else None,
        'disappearance_rate_percent': 100 * valid['emotion_disappeared'].mean() if n else None,
    }
    if extra:
        row.update(extra)
    for e in EMOTIONS:
        row[f'mean_delta_{e}'] = valid[f'delta_{e}'].mean() if n else None
    return row


def aggregate(human_cmp: pd.DataFrame, llm_cmp: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for pool_key in POOL_LABEL:
        for side in ('a', 'b', 'pooled'):
            sub = human_cmp[human_cmp['pool_key'] == pool_key]
            sub = sub if side == 'pooled' else sub[sub['persona_side'] == side]
            rows.append(_agg_row('human', pool_key, side, sub))

    # LLM: primary aggregation stratified by (prompt_variant, label_format) -- never pooled raw.
    variants = llm_cmp['prompt_variant'].unique()
    formats = llm_cmp['label_format'].unique()
    for pool_key in POOL_LABEL:
        for pv in variants:
            for lf in formats:
                for side in ('a', 'b', 'pooled'):
                    sub = llm_cmp[(llm_cmp['pool_key'] == pool_key) & (llm_cmp['prompt_variant'] == pv) & (llm_cmp['label_format'] == lf)]
                    sub = sub if side == 'pooled' else sub[sub['persona_side'] == side]
                    rows.append(_agg_row('llm', pool_key, side, sub, extra={'prompt_variant': pv, 'label_format': lf}))

    # Secondary summary ONLY: mean-of-the-6-per-config-means, explicitly labeled as such --
    # this is NOT pooling the raw 6-config votes together into one sample.
    for pool_key in POOL_LABEL:
        for side in ('a', 'b', 'pooled'):
            config_means = []
            for pv in variants:
                for lf in formats:
                    sub = llm_cmp[(llm_cmp['pool_key'] == pool_key) & (llm_cmp['prompt_variant'] == pv) & (llm_cmp['label_format'] == lf)]
                    sub = sub if side == 'pooled' else sub[sub['persona_side'] == side]
                    valid = sub[sub['valid_comparison'] == True]
                    if len(valid):
                        config_means.append(valid['shift_percent'].mean())
            row = {
                'source': 'llm_SECONDARY_mean_of_6_config_means_NOT_pooled_raw', 'pool_key': pool_key,
                'pool_label': POOL_LABEL.get(pool_key, pool_key), 'persona_side': side,
                'n_valid_comparisons': None, 'n_total_rows': None,
                'mean_shift_percent': (sum(config_means) / len(config_means)) if config_means else None,
                'median_shift_percent': None, 'std_shift_percent': None,
                'ci95_low_shift_percent': None, 'ci95_high_shift_percent': None,
                'mean_overlap_percent': None, 'dominant_change_rate_percent': None,
                'new_emotion_rate_percent': None, 'disappearance_rate_percent': None,
                'prompt_variant': 'ALL_6_AVERAGED', 'label_format': 'ALL_6_AVERAGED',
            }
            for e in EMOTIONS:
                row[f'mean_delta_{e}'] = None
            rows.append(row)

    return pd.DataFrame(rows)


def print_text_report(tid, baseline_validation, original_vectors, human_vectors, llm_vectors, human_cmp, llm_cmp,
                       prompt_variant='one_shot', label_format='numbered'):
    print(f"\n{'=' * 78}\nText {tid}\n{'=' * 78}")
    hv = human_vectors[human_vectors['text_id'] == str(tid)]
    text_val = hv['text'].iloc[0] if len(hv) else '?'
    print(f"  {text_val[:130]}")

    print("  -- Baseline validation (original 5-reader round vs. new 3-rater neutral; NOT used in any persona-effect metric) --")
    ov = original_vectors[original_vectors['text_id'] == str(tid)]
    if len(ov):
        o = ov.iloc[0]
        print(f"  Original human neutral (n={o['vec_n']}, from crowd-enVent_validation.tsv): {o['vec_vector_str']}")
    bv = baseline_validation[baseline_validation['text_id'] == str(tid)]
    if len(bv):
        b = bv.iloc[0]
        print(f"  Original corpus reported majority: {b['original_reader_majority']} "
              f"({b['original_majority_share']*100:.0f}% share, {b['original_n_distinct_emotions']} distinct among 5)")
        if pd.notna(b['new_neutral_probability_of_original_majority']):
            print(f"  New neutral (n={b['new_neutral_n']}) support for that majority emotion: "
                  f"{b['new_neutral_probability_of_original_majority']*100:.1f}%  "
                  f"(replicated in new dominant set: {b['original_majority_replicated']}; "
                  f"share difference vs. originally-reported share: {b['majority_share_difference']*100:+.1f}pp, "
                  f"5-rater vs 3-rater -- descriptive, not a distribution distance)")
        if b['full_comparison_valid']:
            print(f"  FULL vector comparison (original 5 vs new 3): shift={b['full_shift_percent']:.1f}%  "
                  f"overlap={b['full_overlap_percent']:.1f}%  dominant {b['original_dominant']} -> {b['new_dominant']} "
                  f"(changed={b['full_dominant_changed']})")

    for side, label in [('neutral', 'New Human Neutral'), ('a', 'Human Persona A'), ('b', 'Human Persona B')]:
        r = hv[hv['side'] == side]
        if len(r) == 0:
            continue
        r = r.iloc[0]
        print(f"  {label} (n={r['vec_n']}): {r['vec_vector_str']}")

    for side in ('a', 'b'):
        c = human_cmp[(human_cmp['text_id'] == str(tid)) & (human_cmp['persona_side'] == side)]
        if len(c) == 0:
            continue
        c = c.iloc[0]
        if not c['valid_comparison']:
            print(f"  [Human neutral->{side}] no valid comparison (missing data)")
            continue
        print(f"  [Human neutral->{side}] shift={c['shift_percent']:.1f}%  overlap={c['overlap_percent']:.1f}%  "
              f"dominant {c['neutral_dominant']} -> {c['persona_dominant']} (changed={c['dominant_emotion_changed']})  "
              f"new={c['new_emotions'] or '-'}  disappeared={c['disappeared_emotions'] or '-'}")

    print(f"  -- LLM ensemble (7 models), prompt config = {prompt_variant}/{label_format} --")
    lv = llm_vectors[(llm_vectors['text_id'] == str(tid)) & (llm_vectors['prompt_variant'] == prompt_variant) & (llm_vectors['label_format'] == label_format)]
    for framing, label in [('neutral', 'LLM Neutral'), ('persona_a', 'LLM Persona A'), ('persona_b', 'LLM Persona B')]:
        r = lv[lv['framing_condition'] == framing]
        if len(r) == 0:
            continue
        r = r.iloc[0]
        print(f"  {label} (n={r['vec_n']}/{r['n_models_available']} models valid): {r['vec_vector_str']}")

    for side in ('a', 'b'):
        c = llm_cmp[(llm_cmp['text_id'] == str(tid)) & (llm_cmp['persona_side'] == side) &
                     (llm_cmp['prompt_variant'] == prompt_variant) & (llm_cmp['label_format'] == label_format)]
        if len(c) == 0:
            continue
        c = c.iloc[0]
        if not c['valid_comparison']:
            print(f"  [LLM neutral->{side}] no valid comparison (missing data)")
            continue
        print(f"  [LLM neutral->{side}] shift={c['shift_percent']:.1f}%  overlap={c['overlap_percent']:.1f}%  "
              f"dominant {c['neutral_dominant']} -> {c['persona_dominant']} (changed={c['dominant_emotion_changed']})  "
              f"new={c['new_emotions'] or '-'}  disappeared={c['disappeared_emotions'] or '-'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=None, help='Only process this many text_ids (small-subset test mode)')
    args = parser.parse_args()

    # crowd-enVent_validation.tsv and annotated_reader_summary.csv cover the full 1200-text
    # corpus, but human_annotations.csv / experiment_results_all.csv are already scoped to this
    # project's 300-text sample -- so text_ids must ALWAYS be restricted to that 300 (not just in
    # --limit mode), or the original-corpus outputs pick up 900 irrelevant texts.
    human_all = pd.read_csv(BASE / 'human_annotations.csv')
    sample_text_ids = sorted(human_all['text_id'].astype(str).unique(), key=int)
    text_ids = sample_text_ids
    if args.limit:
        text_ids = sample_text_ids[:args.limit]
        print(f"SMALL-SUBSET TEST MODE: {len(text_ids)} texts -> {text_ids}")
    else:
        print(f"FULL RUN: {len(text_ids)} texts (this project's 300-text sample)")

    print("\n[1] Original corpus neutral -- REAL vector from crowd-enVent_validation.tsv (5 raw reader labels/text)...")
    original_vectors = build_original_neutral_vectors(text_ids)
    original_meta = load_original_corpus_aggregate_metadata(text_ids)
    print(f"    {len(original_vectors)} original-neutral vectors, {len(original_meta)} aggregate-metadata rows")

    print("[2/3] Building new-human-neutral + human-persona vectors...")
    human_vectors = build_human_vectors(text_ids)
    print(f"    {len(human_vectors)} (text_id, side) vectors")

    print("[4/5] Building LLM ensemble vectors (7 models per prompt-config cell)...")
    llm_vectors = build_llm_ensemble_vectors(text_ids)
    print(f"    {len(llm_vectors)} (text_id, framing, prompt_variant, label_format) vectors")
    zero_n = (llm_vectors['vec_n'] == 0).sum()
    print(f"    {zero_n} of those have ZERO valid model votes (all 7 off-list/error for that exact cell)")
    any_dropped = (llm_vectors['vec_n_dropped'] > 0).sum()
    print(f"    {any_dropped} of those had at least 1 of 7 models excluded as off-list/garbage")

    print("Building human neutral-vs-persona comparisons...")
    human_cmp = build_human_comparisons(human_vectors)
    print(f"    {len(human_cmp)} rows, {human_cmp['valid_comparison'].sum()} valid")

    print("Building LLM neutral-vs-persona comparisons (stratified by prompt config)...")
    llm_cmp = build_llm_comparisons(llm_vectors)
    print(f"    {len(llm_cmp)} rows, {llm_cmp['valid_comparison'].sum()} valid")

    print("Building human-vs-LLM matched comparison table...")
    matched = build_human_vs_llm_matched(human_vectors, llm_vectors, human_cmp, llm_cmp)
    print(f"    {len(matched)} matched rows")

    print("Building original-vs-new neutral baseline validation...")
    baseline_validation = build_baseline_validation(original_meta, original_vectors, human_vectors)
    print(f"    {len(baseline_validation)} rows, {baseline_validation['full_comparison_valid'].sum()} with a full vector comparison")

    print("Aggregating (stratified by pool x prompt_variant x label_format for LLM)...")
    agg = aggregate(human_cmp, llm_cmp)
    print(f"    {len(agg)} aggregate rows")

    suffix = f"_TESTSUBSET{args.limit}" if args.limit else ""
    original_vectors.to_csv(OUT_DIR / f'original_human_neutral_vectors{suffix}.csv', index=False)
    baseline_validation.to_csv(OUT_DIR / f'baseline_validation_original_vs_new_neutral{suffix}.csv', index=False)
    human_vectors.to_csv(OUT_DIR / f'human_vectors{suffix}.csv', index=False)
    llm_vectors.to_csv(OUT_DIR / f'llm_ensemble_vectors{suffix}.csv', index=False)
    human_cmp.to_csv(OUT_DIR / f'human_neutral_persona_comparisons{suffix}.csv', index=False)
    llm_cmp.to_csv(OUT_DIR / f'llm_neutral_persona_comparisons{suffix}.csv', index=False)
    matched.to_csv(OUT_DIR / f'human_vs_llm_matched{suffix}.csv', index=False)
    agg.to_csv(OUT_DIR / f'aggregate_stats{suffix}.csv', index=False)
    print(f"\nWrote 8 files to {OUT_DIR} with suffix '{suffix}'")

    if args.limit:
        print("\n" + "#" * 78)
        print("# PER-TEXT HUMAN-READABLE REPORTS (small-subset test -- 5 of the texts)")
        print("#" * 78)
        for tid in text_ids[:5]:
            print_text_report(tid, baseline_validation, original_vectors, human_vectors, llm_vectors, human_cmp, llm_cmp)


if __name__ == '__main__':
    main()

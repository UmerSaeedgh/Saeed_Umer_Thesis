"""
Derives ambiguity-pool membership (Unambiguous / Author-Independent Ambiguous / Author-Relevant
Ambiguous) for the 1,200-text corpus and produces the balanced 300-text sample (100/pool) used
throughout the experiment.

Classification rule, applied to the corpus after dropping texts whose author_emotion or
reader_majority is 'boredom' or 'no-emotion' (outside the 11-label set -- forcing these into the
taxonomy produces unstable persona-to-persona flips that are a labeling artifact, not a real
interpretive shift):

  1. THRESHOLD FIRST: a text is Unambiguous if the original corpus's reader agreement is strong
     enough (majority_share >= THRESHOLD) AND matches the author's own label
     (author_matches_readers). This is decided purely from the original reader-validation data and
     is not overridden by the persona/flip test -- a text with strong, author-matching reader
     consensus counts as Unambiguous even if a persona pair can be found that moves the judge.
  2. Among the remaining (non-Unambiguous) texts, the persona flip test (judge_a != judge_b)
     decides Author-Relevant Ambiguous vs. Author-Independent Ambiguous.
  3. Samples 100 texts per pool (reproducibly, via SEED), excluding the reserved few-shot/one-shot
     example texts, and writes sampled_texts.csv plus the three per-pool
     experiment_results_*.csv files (24 empty-label rows per text, in the shape
     build_prompts.build_rows expects) for run_classification.py to fill in.

THRESHOLD is a named constant, not a buried manual judgment call, so it's visible and reproducible.
"""

import pandas as pd
from pathlib import Path

import build_prompts as bp

from project_paths import BASE

OFF_LABELS = {'boredom', 'no-emotion'}
EXAMPLE_IDS = {468, 393, 417}
THRESHOLD = 0.8
SEED = 42

LABEL_ORDER = ['Unambiguous', 'Author-Independent Ambiguous', 'Author-Relevant Ambiguous']
POOL_SHORT_TO_FULL = {
    'Unambiguous': 'Unambiguous',
    'Author-Independent': 'Author-Independent Ambiguous',
    'Author-Relevant': 'Author-Relevant Ambiguous',
}


def classify(full):
    """threshold-first rule, on the full (uncleaned) corpus. Off-label rows get pool=None.

    Unambiguous is decided purely from the original corpus's reader-agreement data
    (majority_share >= THRESHOLD AND author_matches_readers) -- a persona flip does not override
    a text that already has strong, author-matching reader consensus. Only among the remaining
    (non-unambiguous) texts does the persona flip test decide Author-Relevant vs Author-Independent.
    """
    off = full['reader_majority'].isin(OFF_LABELS) | full['author_emotion'].isin(OFF_LABELS)
    pool = pd.Series(None, index=full.index, dtype=object)

    not_off = ~off
    is_unamb = not_off & (full['majority_share'] >= THRESHOLD) & (full['author_matches_readers'] == True)
    pool.loc[is_unamb] = 'Unambiguous'

    remaining = not_off & ~is_unamb
    flip_true = remaining & (full['flip'] == True)
    pool.loc[flip_true] = 'Author-Relevant'
    pool.loc[remaining & ~flip_true] = 'Author-Independent'

    return pool.map(POOL_SHORT_TO_FULL)


def main():
    full = pd.read_csv(BASE / 'ambiguity_classification_1200.csv')
    full['text_id'] = full['text_id'].astype(int)
    full['new_pool'] = classify(full)

    print("Cleaned-corpus pool sizes (candidates, all 1200 minus boredom/no-emotion):")
    print(full['new_pool'].value_counts())
    print()

    sample = pd.read_csv(BASE / 'sampled_texts.csv')
    sample['text_id'] = sample['text_id'].astype(int)
    already_used = set(sample['text_id']) | EXAMPLE_IDS

    # Pull every derived field (text, persona_a/b, rationale, judge_a/b, flip) fresh from `full`
    # rather than trusting sample's own copy -- `full` is the source of truth and a text can stay
    # in-sample across a rerun while its underlying fields (e.g. masked-text corrections) changed.
    refresh_cols = ['text_id', 'new_pool', 'text', 'persona_a', 'persona_b', 'rationale',
                     'judge_a', 'judge_b', 'flip', 'author_emotion', 'reader_majority',
                     'majority_share', 'author_matches_readers']
    merged = sample[['text_id', 'classification']].rename(columns={'classification': 'prior_classification'}) \
        .merge(full[refresh_cols], on='text_id', how='left')
    dropped = merged[merged['new_pool'].isna()]
    kept = merged[merged['new_pool'].notna()].copy()
    kept['classification'] = kept['new_pool']
    print(f"{len(kept)} texts already correctly classified, {len(dropped)} need replacement (off-label)")
    print("Replaced, by prior pool label:")
    print(dropped['prior_classification'].value_counts())
    print()

    # Trim any pool that now has MORE than 100 kept members (e.g. a re-typing rule change moved
    # texts INTO a pool, not just out of one) -- reproducibly drop the excess back to 100 so a
    # later pool can't end up over-represented while another needs replacements.
    trimmed_rows = []
    for label in LABEL_ORDER:
        label_rows = kept[kept['classification'] == label]
        excess = len(label_rows) - 100
        if excess > 0:
            to_drop = label_rows.sample(excess, random_state=SEED)
            trimmed_rows.append(to_drop)
            kept = kept.drop(to_drop.index)
            print(f"{label}: kept count was {len(label_rows)}, trimmed {excess} back to 100")
    if trimmed_rows:
        print()

    kept_counts = kept['classification'].value_counts()
    needed = {label: 100 - int(kept_counts.get(label, 0)) for label in LABEL_ORDER}
    print("Replacements needed to restore 100/pool:", needed)
    print()

    replacements = []
    for label, n in needed.items():
        if n <= 0:
            continue
        candidates = full[
            (full['new_pool'] == label) &
            (~full['text_id'].isin(already_used)) &
            (~full['reader_majority'].isin(OFF_LABELS)) &
            (~full['author_emotion'].isin(OFF_LABELS))
        ]
        chosen = candidates.sample(n, random_state=SEED)
        already_used |= set(chosen['text_id'])
        chosen = chosen.copy()
        chosen['classification'] = label
        replacements.append(chosen)
        print(f"  {label}: drew {len(chosen)} replacements from {len(candidates)} eligible")

    replacements = pd.concat(replacements, ignore_index=True) if replacements else pd.DataFrame()

    out_cols = ['text_id', 'text', 'author_emotion', 'reader_majority', 'majority_share',
                'author_matches_readers', 'classification', 'persona_a', 'persona_b',
                'rationale', 'judge_a', 'judge_b', 'flip']
    kept_out = kept[out_cols]
    repl_out = replacements[out_cols] if len(replacements) else replacements

    new_sample = pd.concat([kept_out, repl_out], ignore_index=True)
    assert len(new_sample) == 300, f"expected 300, got {len(new_sample)}"
    for label in LABEL_ORDER:
        n = (new_sample['classification'] == label).sum()
        assert n == 100, f"{label}: expected 100, got {n}"
    assert new_sample['text_id'].nunique() == 300, "duplicate text_id"
    still_bad = new_sample['reader_majority'].isin(OFF_LABELS).sum()
    assert still_bad == 0, f"{still_bad} texts still off-label"

    new_sample.to_csv(BASE / 'sampled_texts.csv', index=False)
    print(f"\nSaved sampled_texts.csv: 300 texts, 100/pool, 0 boredom/no-emotion")

    # Update the master experiment CSVs for any pool whose text set changed.
    for pool, path in bp.POOL_TO_FILE.items():
        old_ids_for_pool = set(sample.loc[sample['classification'] == pool, 'text_id'])
        new_ids_for_pool = set(new_sample.loc[new_sample['classification'] == pool, 'text_id'])
        dropped_ids = old_ids_for_pool - new_ids_for_pool
        added_ids = new_ids_for_pool - old_ids_for_pool
        if not dropped_ids and not added_ids:
            print(f"{path.name}: no change ({len(old_ids_for_pool)} texts, all kept)")
            continue

        master = pd.read_csv(path)
        master['text_id'] = master['text_id'].astype(int)
        n_before = len(master)
        master = master[~master['text_id'].isin(dropped_ids)]
        n_dropped_rows = n_before - len(master)

        new_rows_for_pool = new_sample[new_sample['text_id'].isin(added_ids)]
        if len(new_rows_for_pool):
            new_block = bp.build_rows(new_rows_for_pool)
            # build_rows only initializes the original 4 models' columns; add empty columns for
            # the 4 local models too so run_classification.py finds every label column it expects.
            for col in ['Qwen3_label', 'Qwen3_reason', 'Gemma3_label', 'Gemma3_reason',
                        'Ministral_label', 'Ministral_reason', 'Llama31_label', 'Llama31_reason']:
                if col not in new_block.columns:
                    new_block[col] = ''
            master = pd.concat([master, new_block], ignore_index=True)

        master.to_csv(path, index=False)
        print(f"{path.name}: dropped {len(dropped_ids)} texts ({n_dropped_rows} rows), "
              f"added {len(added_ids)} texts ({len(new_rows_for_pool) * 24} rows) -> {len(master)} total rows")


if __name__ == '__main__':
    main()

"""
Fixes the 49 texts (1176 rows: 49 x 24 conditions) whose framed_stimulus/full_prompt in the 3
pool CSVs were built from a STALE version of sampled_texts.csv -- before that text's
emotion-word redaction and persona rewrite were applied. Confirmed: sampled_texts.csv currently
holds the correct text (matches human_annotations.csv exactly, 0/49 mismatches) and the correct,
redesigned personas; the pool CSVs still have the old leaked text and old flawed personas for
these specific 49 texts only (every other one of the 300 texts is unaffected).

Safety:
- Backs up all 3 pool CSVs first (once; won't overwrite an existing backup from this fix).
- Only touches rows whose text_id is in the confirmed AFFECTED_TEXT_IDS list.
- Rebuilds text/persona_a/persona_b/framed_stimulus/full_prompt using the project's own
  make_framed_stimulus/make_full_prompt functions from build_prompts.py (imported, not
  reimplemented) so the rebuilt prompts are byte-identical in structure to every other row.
- Blanks all 7 models' label/reason/label_norm columns for these rows only, since their old
  answers were responses to the WRONG prompt and are no longer valid data.
- Also cleans matching entries out of every model's *.partial.csv checkpoint file for the
  affected pools, so run_classification.py's resume logic doesn't skip these rows as "already
  done" using stale, wrong-prompt answers (the same bug hit during the Gemini fix).

Run: python fix_leaked_prompts.py
"""
import shutil
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from build_prompts import make_framed_stimulus, make_full_prompt

BASE = Path(r"C:\Users\umers\Desktop\thesis\thesis final")
FILES = {
    'unambiguous': BASE / 'experiment_results_unambiguous.csv',
    'author_independent': BASE / 'experiment_results_author_independent.csv',
    'author_relevant': BASE / 'experiment_results_author_relevant.csv',
}
MODELS_ALL = ['openai', 'claude', 'gemini', 'qwen3', 'gemma3', 'ministral', 'llama31']
LABEL_COLS = {
    'openai': ('ChatGPT_label', 'ChatGPT_reason', 'ChatGPT_label_norm'),
    'claude': ('Claude_label', 'Claude_reason', 'Claude_label_norm'),
    'gemini': ('Gemini_label', 'Gemini_reason', 'Gemini_label_norm'),
    'qwen3': ('Qwen3_label', 'Qwen3_reason', 'Qwen3_label_norm'),
    'gemma3': ('Gemma3_label', 'Gemma3_reason', 'Gemma3_label_norm'),
    'ministral': ('Ministral_label', 'Ministral_reason', 'Ministral_label_norm'),
    'llama31': ('Llama31_label', 'Llama31_reason', 'Llama31_label_norm'),
}

AFFECTED_TEXT_IDS = [287, 315, 641, 2212, 2213, 3283, 3319, 3328, 3338, 3426, 3431, 3443, 3493,
                     3645, 3936, 4146, 4490, 4497, 4874, 5688, 5704, 5880, 5914, 6789, 6869, 7179,
                     7200, 31015, 31089, 41082, 41208, 41228, 41260, 41378, 41415, 41548, 41671,
                     41722, 51249, 51258, 51263, 51688, 51710, 51835, 52157, 52184, 52458, 52828,
                     52895]
AFFECTED_TEXT_IDS = set(AFFECTED_TEXT_IDS)

sample = pd.read_csv(BASE / 'sampled_texts.csv')
sample['text_id'] = sample['text_id'].astype(int)
sample_by_id = {int(r['text_id']): r for _, r in sample.iterrows()}

missing = AFFECTED_TEXT_IDS - set(sample_by_id.keys())
if missing:
    print(f"ABORT: {len(missing)} affected text_ids not found in sampled_texts.csv: {missing}")
    sys.exit(1)

total_rows_fixed = 0
affected_keys_by_pool = {}  # pool_key -> set of (text_id, framing, prompt_variant, label_format)

for pool_key, path in FILES.items():
    backup = path.with_suffix(path.suffix + '.bak_pre_leak_fix')
    if not backup.exists():
        shutil.copy(path, backup)
        print(f"Backed up {path.name} -> {backup.name}")
    else:
        print(f"Backup already exists: {backup.name} (not overwriting)")

    df = pd.read_csv(path)
    df['text_id'] = df['text_id'].astype(int)
    mask = df['text_id'].isin(AFFECTED_TEXT_IDS)
    n_affected = mask.sum()
    if n_affected == 0:
        print(f"{pool_key}: no affected rows")
        continue

    keys_this_pool = set()
    for idx in df[mask].index:
        tid = df.at[idx, 'text_id']
        src = sample_by_id[tid]
        framing = df.at[idx, 'framing_condition']
        prompt_variant = df.at[idx, 'prompt_variant']
        label_format = df.at[idx, 'label_format']

        framed_stimulus = make_framed_stimulus(src['text'], framing, src['persona_a'], src['persona_b'])
        full_prompt = make_full_prompt(framed_stimulus, prompt_variant, label_format)

        df.at[idx, 'text'] = src['text']
        df.at[idx, 'persona_a'] = src['persona_a']
        df.at[idx, 'persona_b'] = src['persona_b']
        df.at[idx, 'framed_stimulus'] = framed_stimulus
        df.at[idx, 'full_prompt'] = full_prompt

        for model in MODELS_ALL:
            label_col, reason_col, norm_col = LABEL_COLS[model]
            df.at[idx, label_col] = ''
            df.at[idx, reason_col] = ''
            df.at[idx, norm_col] = ''

        keys_this_pool.add((tid, framing, prompt_variant, label_format))

    df.to_csv(path, index=False)
    affected_keys_by_pool[pool_key] = keys_this_pool
    total_rows_fixed += n_affected
    print(f"{pool_key}: rebuilt prompts + blanked all 7 models for {n_affected} rows")

print(f"\nTotal rows fixed: {total_rows_fixed} (expect 1176 = 49 texts x 24 conditions)")

# ---------------------------------------------------------------------------
# Clean matching entries out of every model's partial checkpoint file
# ---------------------------------------------------------------------------
print("\nCleaning stale checkpoint entries across all 7 models' partial files...")
for pool_key, path in FILES.items():
    keys = affected_keys_by_pool.get(pool_key)
    if not keys:
        continue
    for model in MODELS_ALL:
        ppath = path.parent / f"{path.stem}.{model}.partial.csv"
        if not ppath.exists():
            continue
        p = pd.read_csv(ppath)
        if len(p) == 0:
            continue
        p['text_id'] = p['text_id'].astype(int)
        p_keys = list(zip(p['text_id'], p['framing_condition'], p['prompt_variant'], p['label_format']))
        drop_mask = [k in keys for k in p_keys]
        n_drop = sum(drop_mask)
        if n_drop > 0:
            p_clean = p[[not d for d in drop_mask]].copy()
            p_clean.to_csv(ppath, index=False)
            print(f"  {ppath.name}: removed {n_drop} stale entries")

print("\nDone. Next: run all 7 models against the now-pending 1176 rows, then:")
print("  1. python run_classification.py --merge --files unambiguous,author_independent,author_relevant --models openai,claude,gemini,qwen3,gemma3,ministral,llama31")
print("  2. python normalize_labels.py")
print("  3. python build_unified_dataset.py")
print("  4. Rerun build_emotion_distributions.py -> qc_gate.py -> run_analysis.py -> figures/tables -> notebooks")

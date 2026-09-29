"""
Rebuilds human_annotations.csv end to end (base persona a/b -> persona-redo merge -> neutral
append) with the tie-breaking fix applied: build_human_annotations.py and
01_build_redo_annotations.py used to resolve a 3-way rater split (e.g. one vote each for
disgust/fear/anger) by picking whichever answer happened to be first in insertion order,
silently turning "no consensus" into a fake majority. This mirrors that same pipeline but
sources the persona-redo first-response data from this folder's persistent copy instead of the
now-gone session scratch dir the original 01_ script pointed at, and applies the shared
tie-safe majority_vote() at both aggregation points. Neutral rows are untouched (1 rater/text,
so "majority" is trivially that one answer -- no tie is possible).
"""
import json, csv
from collections import Counter
from pathlib import Path
import pandas as pd

BASE = Path.home() / 'Desktop' / 'thesis' / 'thesis final'
REDO_DIR = BASE / 'persona_redo_20260917'
RAW = BASE / 'raw_form_responses'


def majority_vote(answers):
    counts = Counter(a.strip().lower() for a in answers)
    top_n = max(counts.values())
    winners = [lab for lab, c in counts.items() if c == top_n]
    majority = winners[0] if len(winners) == 1 else ''
    return majority, top_n


sample_ids = set(pd.read_csv(BASE / 'sampled_texts.csv')['text_id'].astype(str))
print(f"Current sample: {len(sample_ids)} texts")

# ---- Part 1: original 60-form study + Batch 2, persona a/b only (build_human_annotations.py) ----
SOURCES = [
    (RAW / 'form_assignment.json', RAW / 'raw_responses.csv'),
    (RAW / 'form_assignment_batch2.json', RAW / 'raw_responses_batch2.csv'),
]
by_pair = {}
for forms_path, raw_path in SOURCES:
    with open(forms_path, encoding='utf-8') as f:
        forms = json.load(f)
    assignment = {}
    for form_id, items in forms.items():
        for idx, it in enumerate(items, start=1):
            assignment[(form_id, idx)] = it
    raw = pd.read_csv(raw_path)
    for _, r in raw.iterrows():
        key = (r['form_id'], int(r['item_index']))
        if key not in assignment:
            continue
        it = assignment[key]
        if str(it['text_id']) not in sample_ids:
            continue
        pair_key = (str(it['text_id']), it['side'])
        by_pair.setdefault(pair_key, {'persona': it['persona'], 'text': it['text'], 'pool': it['pool'], 'answers': []})
        by_pair[pair_key]['answers'].append(str(r['emotion']).strip().lower())

base_rows = []
for (text_id, side), d in by_pair.items():
    majority, top_n = majority_vote(d['answers'])
    base_rows.append({
        'text_id': text_id, 'side': side, 'persona': d['persona'], 'text': d['text'], 'pool': d['pool'],
        'n_raters': len(d['answers']), 'raters_raw': '|'.join(d['answers']),
        'human_majority': majority, 'human_agree': round(top_n / len(d['answers']), 3),
    })
print(f"Base (original+batch2) persona pairs: {len(base_rows)}")

# ---- Part 2: persona redo, 2026-09-17 (01_build_redo_annotations.py) ----
with open(REDO_DIR / 'form_assignment_redo.json', encoding='utf-8') as f:
    redo_assignment = json.load(f)
first_responses = {}
with open(REDO_DIR / 'redo_first_response.csv', encoding='utf-8') as f:
    for line in f:
        line = line.rstrip('\n')
        if not line.strip():
            continue
        parts = line.split(',')
        first_responses[parts[0]] = parts[1:]
alias_source = {'form_redo_10_dup': 'form_redo_10'}

redo_by_pair, redo_meta = {}, {}
for form_key, emotions in first_responses.items():
    source_key = alias_source.get(form_key, form_key)
    items = redo_assignment[source_key]
    if len(items) != len(emotions):
        continue
    for it, emo in zip(items, emotions):
        key = (str(it['text_id']), it['side'])
        redo_by_pair.setdefault(key, []).append(emo.strip())
        redo_meta[key] = {'persona': it['persona'], 'text': it['text'], 'pool': it['pool']}

redo_rows = []
for (text_id, side), answers in redo_by_pair.items():
    majority, top_n = majority_vote(answers)
    meta = redo_meta[(text_id, side)]
    redo_rows.append({
        'text_id': text_id, 'side': side, 'persona': meta['persona'], 'text': meta['text'], 'pool': meta['pool'],
        'n_raters': len(answers), 'raters_raw': '|'.join(a.lower() for a in answers),
        'human_majority': majority, 'human_agree': round(top_n / len(answers), 3),
    })
print(f"Persona-redo pairs: {len(redo_rows)}")

# ---- Part 3: merge base + redo (03_merge_redo_into_annotations.py) ----
redo_text_ids = {r['text_id'] for r in redo_rows}
kept = [r for r in base_rows if r['text_id'] not in redo_text_ids and r['text_id'] in sample_ids]
merged_persona = kept + redo_rows
print(f"Merged persona a/b: {len(merged_persona)} rows (kept {len(kept)} untouched + {len(redo_rows)} redo)")

# ---- Part 4: append neutral, untouched -- 1 rater/text, no tie possible (04_merge_neutral_into_annotations.py) ----
with open(REDO_DIR / 'human_annotations_neutral.csv', encoding='utf-8') as f:
    neutral_rows = list(csv.DictReader(f))
print(f"Neutral rows: {len(neutral_rows)}")

final_rows = merged_persona + neutral_rows
final_rows.sort(key=lambda r: (int(r['text_id']), r['side']))

fieldnames = ['text_id', 'side', 'persona', 'text', 'pool', 'n_raters', 'raters_raw', 'human_majority', 'human_agree']
out_path = BASE / 'human_annotations.csv'
with open(out_path, 'w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    for r in final_rows:
        w.writerow({k: r.get(k, '') for k in fieldnames})

n_ties = sum(1 for r in merged_persona if r['human_majority'] == '')
print(f"\nWrote {len(final_rows)} rows -> {out_path}")
print(f"Persona a/b pairs with no true majority (tie, now blank instead of arbitrary): {n_ties} of {len(merged_persona)}")

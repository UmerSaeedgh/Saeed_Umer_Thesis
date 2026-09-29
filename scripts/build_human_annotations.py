"""
Joins the raw Google Forms responses (form_id, item_index, emotion; raw_form_responses/) collected
for the Phase 4 human annotation study -- both the original 60-form run and the 6-form Batch 2
extension covering the 29 rebalanced replacement texts -- against their known form assignments
(form_id, item_index -> text_id, side/persona), aggregates to majority vote per (text_id, side),
and writes human_annotations.csv with columns:

  text_id, side, persona, text, pool, n_raters, raters_raw (list), human_majority, human_agree

Results are filtered down to (text_id, side) pairs whose text_id is in the current sampled_texts.csv
(the 300-text rebalanced sample). This is what makes the merge correct: the original 60-form run
covered 300 texts, 29 of which were later swapped out during rebalancing, and Batch 2 covers exactly
those 29 replacements -- so filtering against the current sample keeps the right text_id from each
source without needing to track the swap explicitly.
"""
import json
import pandas as pd
from pathlib import Path
from collections import Counter

BASE = Path.home() / 'Desktop' / 'thesis' / 'thesis final'
RAW = BASE / 'raw_form_responses'

SOURCES = [
    (RAW / 'form_assignment.json', RAW / 'raw_responses.csv'),
    (RAW / 'form_assignment_batch2.json', RAW / 'raw_responses_batch2.csv'),
]

sample_ids = set(pd.read_csv(BASE / 'sampled_texts.csv')['text_id'])
print(f"Current sample: {len(sample_ids)} texts")

by_pair = {}  # (text_id, side) -> list of emotion answers
missing = 0
skipped_out_of_sample = 0

for forms_path, raw_path in SOURCES:
    with open(forms_path, encoding='utf-8') as f:
        forms = json.load(f)

    assignment = {}
    for form_id, items in forms.items():
        for idx, it in enumerate(items, start=1):
            assignment[(form_id, idx)] = it

    raw = pd.read_csv(raw_path)
    print(f"{raw_path.name}: {len(raw)} item-answers from {raw['form_id'].nunique()} forms")

    for _, r in raw.iterrows():
        key = (r['form_id'], int(r['item_index']))
        if key not in assignment:
            missing += 1
            continue
        it = assignment[key]
        if it['text_id'] not in sample_ids:
            skipped_out_of_sample += 1
            continue
        pair_key = (it['text_id'], it['side'])
        by_pair.setdefault(pair_key, {'persona': it['persona'], 'text': it['text'], 'pool': it['pool'], 'answers': []})
        by_pair[pair_key]['answers'].append(str(r['emotion']).strip().lower())

if missing:
    print(f"WARNING: {missing} raw responses had no matching (form_id, item_index) in an assignment map")
print(f"Skipped {skipped_out_of_sample} responses for text_ids no longer in the current sample (pre-rebalance texts)")

def majority_vote(answers):
    """Plurality vote. Returns ('', agree) when the top count is shared by more than one
    label -- e.g. 3 raters giving 3 different emotions -- instead of arbitrarily picking
    whichever answer happens to come first in Counter.most_common()'s insertion-order
    tie-break (that silently turned "no consensus" into a fake majority)."""
    counts = Counter(answers)
    top_n = max(counts.values())
    winners = [lab for lab, c in counts.items() if c == top_n]
    majority = winners[0] if len(winners) == 1 else ''
    return majority, top_n

rows = []
for (text_id, side), d in by_pair.items():
    answers = d['answers']
    majority, top_n = majority_vote(answers)
    agree = top_n / len(answers)
    rows.append({
        'text_id': text_id,
        'side': side,
        'persona': d['persona'],
        'text': d['text'],
        'pool': d['pool'],
        'n_raters': len(answers),
        'raters_raw': '|'.join(answers),
        'human_majority': majority,
        'human_agree': round(agree, 3),
    })

out = pd.DataFrame(rows).sort_values(['text_id', 'side'])
out_path = BASE / 'human_annotations.csv'
out.to_csv(out_path, index=False)

print(f"\nWrote {len(out)} (text_id, side) pairs with >=1 rater to {out_path.name}")
print(f"Coverage: {len(out)} of {len(sample_ids) * 2} possible (text_id, side) pairs have at least 1 response")
print(f"  n_raters distribution:")
print(out['n_raters'].value_counts().sort_index())
print(f"\n  by pool:")
print(out.groupby('pool').size())

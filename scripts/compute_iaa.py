"""
Computes inter-annotator agreement (Fleiss' kappa) for the Phase 4 human annotation study
(persona_a/persona_b, the completed 60-form + 6-form Batch 2 round), answering the supervisor's
question of how reliable the annotator pool is as a whole.

Mirrors the join logic in build_human_annotations.py (form_id, item_index -> text_id, side, pool
via the two form_assignment*.json files) but, instead of collapsing to a majority vote per
(text_id, side), keeps every individual rater's label so a Fleiss' kappa category-count table can
be built. Restricted to (text_id, side) pairs with exactly 3 raters (a few have more/fewer due to
partial completion) so every item contributes the same number of raters, as Fleiss' kappa requires.

Reports kappa overall, by pool, and by side (a vs b), plus raw agreement (% of items where all 3
raters agreed) as an easier-to-read companion statistic.
"""
import json
import pandas as pd
from pathlib import Path
from collections import Counter
from statsmodels.stats.inter_rater import fleiss_kappa

BASE = Path.home() / 'Desktop' / 'thesis' / 'thesis final'
RAW = BASE / 'raw_form_responses'

SOURCES = [
    (RAW / 'form_assignment.json', RAW / 'raw_responses.csv'),
    (RAW / 'form_assignment_batch2.json', RAW / 'raw_responses_batch2.csv'),
]

sample_ids = set(pd.read_csv(BASE / 'sampled_texts.csv')['text_id'])

by_pair = {}  # (text_id, side) -> {'pool':..., 'answers': [...]}
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
        if it['text_id'] not in sample_ids:
            continue
        pair_key = (it['text_id'], it['side'])
        by_pair.setdefault(pair_key, {'pool': it['pool'], 'answers': []})
        by_pair[pair_key]['answers'].append(str(r['emotion']).strip().lower())

rows = []
for (text_id, side), d in by_pair.items():
    rows.append({'text_id': text_id, 'side': side, 'pool': d['pool'],
                 'n_raters': len(d['answers']), 'answers': d['answers']})
df = pd.DataFrame(rows)

print(f"Total (text_id, side) pairs with >=1 response: {len(df)}")
print("n_raters distribution:")
print(df['n_raters'].value_counts().sort_index())
print()

df3 = df[df['n_raters'] == 3].reset_index(drop=True)
print(f"Restricting to exactly 3 raters/pair (Fleiss' kappa requires equal n): {len(df3)} pairs")
print()

LABELS = ['anger', 'boredom', 'disgust', 'fear', 'guilt', 'joy', 'pride', 'relief',
          'sadness', 'shame', 'surprise', 'trust', 'no-emotion']


def kappa_and_raw_agreement(sub):
    if len(sub) == 0:
        return None, None
    table = []
    n_full_agree = 0
    for _, r in sub.iterrows():
        counts = Counter(r['answers'])
        table.append([counts.get(l, 0) for l in LABELS])
        if len(counts) == 1:
            n_full_agree += 1
    table = pd.DataFrame(table, columns=LABELS).values
    k = fleiss_kappa(table, method='fleiss')
    raw_agree = n_full_agree / len(sub)
    return k, raw_agree


print("=== Overall (all pools, both sides pooled) ===")
k, raw = kappa_and_raw_agreement(df3)
print(f"Fleiss' kappa: {k:.3f}   |   raw 3/3 agreement: {raw:.1%}   |   n items: {len(df3)}")
print()

print("=== By pool ===")
for pool in ['unambiguous', 'author_independent', 'author_relevant']:
    sub = df3[df3['pool'] == pool]
    k, raw = kappa_and_raw_agreement(sub)
    print(f"{pool:20s} kappa={k:.3f}   raw_agree={raw:.1%}   n={len(sub)}")
print()

print("=== By side (persona_a vs persona_b) ===")
for side in ['a', 'b']:
    sub = df3[df3['side'] == side]
    k, raw = kappa_and_raw_agreement(sub)
    print(f"side {side}: kappa={k:.3f}   raw_agree={raw:.1%}   n={len(sub)}")
print()

print("=== By pool x side ===")
out_rows = []
for pool in ['unambiguous', 'author_independent', 'author_relevant']:
    for side in ['a', 'b']:
        sub = df3[(df3['pool'] == pool) & (df3['side'] == side)]
        k, raw = kappa_and_raw_agreement(sub)
        print(f"{pool:20s} side {side}: kappa={k:.3f}   raw_agree={raw:.1%}   n={len(sub)}")
        out_rows.append({'pool': pool, 'side': side, 'fleiss_kappa': k, 'raw_agreement': raw, 'n_items': len(sub)})

pd.DataFrame(out_rows).to_csv(BASE / 'iaa_results.csv', index=False)
print(f"\nSaved per-(pool,side) breakdown to iaa_results.csv")

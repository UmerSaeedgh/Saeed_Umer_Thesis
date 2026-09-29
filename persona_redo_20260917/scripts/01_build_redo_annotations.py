import json, csv, os
from collections import Counter, defaultdict

BASE = r"C:\Users\umers\Desktop\thesis\thesis final\persona_redo_20260917"
SCRATCH = r"C:\Users\umers\AppData\Local\Temp\claude\C--Users-umers-Desktop-thesis\3262364f-bd1c-4666-ace5-505b30cf2135\scratchpad"

with open(os.path.join(BASE, "form_assignment_redo.json"), encoding="utf-8") as f:
    assignment = json.load(f)

# read the compact first-response CSV: form_key,emotion1,emotion2,...
first_responses = {}
with open(os.path.join(SCRATCH, "redo_first_response.csv"), encoding="utf-8") as f:
    for line in f:
        line = line.rstrip("\n")
        if not line.strip():
            continue
        parts = line.split(",")
        form_key = parts[0]
        emotions = parts[1:]
        first_responses[form_key] = emotions

# form_redo_10_dup shares the exact same item list/order as form_redo_10
alias_source = {"form_redo_10_dup": "form_redo_10"}

by_pair = defaultdict(list)  # (text_id, side) -> list of emotions
pair_meta = {}

mismatches = []
for form_key, emotions in first_responses.items():
    source_key = alias_source.get(form_key, form_key)
    items = assignment[source_key]
    if len(items) != len(emotions):
        mismatches.append((form_key, len(items), len(emotions)))
        continue
    for it, emo in zip(items, emotions):
        key = (it["text_id"], it["side"])
        by_pair[key].append(emo.strip())
        pair_meta[key] = {"persona": it["persona"], "text": it["text"], "reason": it["reason"], "pool": it["pool"]}

print(f"Forms processed: {len(first_responses)}")
if mismatches:
    print("LENGTH MISMATCHES:", mismatches)
print(f"Distinct (text_id, side) pairs covered: {len(by_pair)} (of 204 possible)")

def majority_vote(answers):
    """Plurality vote. Returns ('', agree) when the top count is shared by more than one
    label -- e.g. 3 raters giving 3 different emotions -- instead of arbitrarily picking
    whichever answer happens to come first in Counter.most_common()'s insertion-order
    tie-break (that silently turned "no consensus" into a fake majority)."""
    counts = Counter(a.lower() for a in answers)
    top_n = max(counts.values())
    winners = [lab for lab, c in counts.items() if c == top_n]
    majority = winners[0] if len(winners) == 1 else ''
    return majority, top_n

rows = []
for (text_id, side), answers in by_pair.items():
    majority, top_n = majority_vote(answers)
    agree = top_n / len(answers)
    meta = pair_meta[(text_id, side)]
    rows.append({
        "text_id": text_id,
        "side": side,
        "persona": meta["persona"],
        "text": meta["text"],
        "pool": meta["pool"],
        "reason": meta["reason"],
        "n_raters": len(answers),
        "raters_raw": "|".join(a.lower() for a in answers),
        "human_majority": majority,
        "human_agree": round(agree, 3),
    })

rows.sort(key=lambda r: (int(r["text_id"]), r["side"]))

out_path = os.path.join(BASE, "human_annotations_persona_redo.csv")
with open(out_path, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["text_id", "side", "persona", "text", "pool", "reason",
                                       "n_raters", "raters_raw", "human_majority", "human_agree"])
    w.writeheader()
    w.writerows(rows)

print(f"Wrote {len(rows)} pairs -> {out_path}")

n_raters_dist = Counter(r["n_raters"] for r in rows)
print("n_raters distribution:", dict(sorted(n_raters_dist.items())))

# sanity: how many of the 204 target pairs got zero coverage
all_pairs = set()
for form_key, items in assignment.items():
    for it in items:
        all_pairs.add((it["text_id"], it["side"]))
missing = all_pairs - set(by_pair.keys())
print(f"Pairs with ZERO responses: {len(missing)} of {len(all_pairs)}")
if missing:
    print(sorted(missing)[:20])

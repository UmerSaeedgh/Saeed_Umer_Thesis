import csv, os
from collections import Counter

BASE = r"C:\Users\umers\Desktop\thesis\thesis final"
EDITPLAN = os.path.join(BASE, "backup_pre_hidden_emo_fix", "form_texts", "all_forms_editplan_final.csv")
SCRATCH = r"C:\Users\umers\AppData\Local\Temp\claude\C--Users-umers-Desktop-thesis\3262364f-bd1c-4666-ace5-505b30cf2135\scratchpad"
OUT_DIR = os.path.join(BASE, "persona_redo_20260917")  # reuse folder for outputs

# build form -> ordered list of current text_ids
form_items = {}
with open(EDITPLAN, encoding="utf-8") as f:
    for row in csv.DictReader(f):
        form = row["form"]
        idx = int(row["item_index"])
        if row["action"] == "REPLACE_DROPPED":
            tid = row["new_text_id"].split(".")[0]
        else:
            tid = row["old_text_id"]
        form_items.setdefault(form, {})[idx] = tid

for form, items in form_items.items():
    assert sorted(items.keys()) == list(range(1, len(items) + 1)), (form, sorted(items.keys()))

# sampled_texts.csv for text + pool (classification)
text_meta = {}
with open(os.path.join(BASE, "sampled_texts.csv"), encoding="utf-8") as f:
    for row in csv.DictReader(f):
        text_meta[row["text_id"]] = {"text": row["text"], "pool": row["classification"]}

# first-response emotions
first_responses = {}
with open(os.path.join(SCRATCH, "neutral_first_response.csv"), encoding="utf-8") as f:
    for line in f:
        line = line.rstrip("\n")
        if not line.strip():
            continue
        parts = line.split(",")
        first_responses[parts[0]] = parts[1:]

rows = []
mismatches = []
missing_meta = []
for form, emotions in first_responses.items():
    items = form_items[form]
    ordered_ids = [items[i] for i in range(1, len(items) + 1)]
    if len(ordered_ids) != len(emotions):
        mismatches.append((form, len(ordered_ids), len(emotions)))
        continue
    for tid, emo in zip(ordered_ids, emotions):
        meta = text_meta.get(tid)
        if meta is None:
            missing_meta.append((form, tid))
            continue
        rows.append({
            "text_id": tid,
            "side": "neutral",
            "persona": "",
            "text": meta["text"],
            "pool": meta["pool"],
            "n_raters": 1,
            "raters_raw": emo.strip().lower(),
            "human_majority": emo.strip().lower(),
            "human_agree": 1.0,
        })

print(f"Forms processed: {len(first_responses)}")
if mismatches:
    print("LENGTH MISMATCHES:", mismatches)
if missing_meta:
    print("MISSING TEXT_ID IN sampled_texts.csv:", missing_meta)

print(f"Total rows: {len(rows)} (expect 300)")
print(f"Distinct text_ids: {len(set(r['text_id'] for r in rows))}")

dupes = [tid for tid, c in Counter(r["text_id"] for r in rows).items() if c > 1]
if dupes:
    print("DUPLICATE text_ids across forms:", dupes)

rows.sort(key=lambda r: int(r["text_id"]))
out_path = os.path.join(OUT_DIR, "human_annotations_neutral.csv")
with open(out_path, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["text_id", "side", "persona", "text", "pool",
                                       "n_raters", "raters_raw", "human_majority", "human_agree"])
    w.writeheader()
    w.writerows(rows)
print(f"Wrote -> {out_path}")

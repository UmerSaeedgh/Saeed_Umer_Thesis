import csv, shutil, os
from datetime import datetime

BASE = r"C:\Users\umers\Desktop\thesis\thesis final"
MAIN = os.path.join(BASE, "human_annotations.csv")
REDO = os.path.join(BASE, "persona_redo_20260917", "human_annotations_persona_redo.csv")

backup = os.path.join(BASE, "persona_redo_20260917", "human_annotations_pre_redo_merge_backup.csv")
shutil.copy2(MAIN, backup)
print(f"Backed up current human_annotations.csv -> {backup}")

with open(MAIN, encoding="utf-8") as f:
    main_rows = list(csv.DictReader(f))
with open(REDO, encoding="utf-8") as f:
    redo_rows = list(csv.DictReader(f))

redo_text_ids = {r["text_id"] for r in redo_rows}
print(f"Main file: {len(main_rows)} rows, {len(set(r['text_id'] for r in main_rows))} texts")
print(f"Redo file: {len(redo_rows)} rows, {len(redo_text_ids)} texts")

current_sample_ids = set()
with open(os.path.join(BASE, "sampled_texts.csv"), encoding="utf-8") as f:
    for r in csv.DictReader(f):
        current_sample_ids.add(r["text_id"])

kept = [r for r in main_rows if r["text_id"] not in redo_text_ids and r["text_id"] in current_sample_ids]
dropped_stale = [r for r in main_rows if r["text_id"] not in redo_text_ids and r["text_id"] not in current_sample_ids]
replaced = [r for r in main_rows if r["text_id"] in redo_text_ids]

print(f"Kept untouched rows: {len(kept)}")
print(f"Dropped (old, no longer in current 300-sample): {len(dropped_stale)} rows, ids: {sorted(set(r['text_id'] for r in dropped_stale))}")
print(f"Replaced (stale, now superseded by redo data): {len(replaced)} rows")

for r in redo_rows:
    r.pop("reason", None)

merged = kept + redo_rows
merged.sort(key=lambda r: (int(r["text_id"]), r["side"]))

fieldnames = ["text_id", "side", "persona", "text", "pool", "n_raters", "raters_raw", "human_majority", "human_agree"]
with open(MAIN, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    for r in merged:
        w.writerow({k: r.get(k, "") for k in fieldnames})

print(f"\nWrote merged human_annotations.csv: {len(merged)} rows, {len(set(r['text_id'] for r in merged))} texts")
print(f"Expected: 600 rows, 300 texts (matches current sample: {len(current_sample_ids)} texts)")

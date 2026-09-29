import csv, shutil, os

BASE = r"C:\Users\umers\Desktop\thesis\thesis final"
MAIN = os.path.join(BASE, "human_annotations.csv")
NEUTRAL = os.path.join(BASE, "persona_redo_20260917", "human_annotations_neutral.csv")

backup = os.path.join(BASE, "persona_redo_20260917", "human_annotations_pre_neutral_merge_backup.csv")
shutil.copy2(MAIN, backup)
print(f"Backed up -> {backup}")

with open(MAIN, encoding="utf-8") as f:
    main_rows = list(csv.DictReader(f))
with open(NEUTRAL, encoding="utf-8") as f:
    neutral_rows = list(csv.DictReader(f))

existing_neutral = [r for r in main_rows if r["side"] == "neutral"]
if existing_neutral:
    print(f"WARNING: {len(existing_neutral)} existing 'neutral' rows already present, removing before re-adding")
    main_rows = [r for r in main_rows if r["side"] != "neutral"]

merged = main_rows + neutral_rows
merged.sort(key=lambda r: (int(r["text_id"]), r["side"]))

fieldnames = ["text_id", "side", "persona", "text", "pool", "n_raters", "raters_raw", "human_majority", "human_agree"]
with open(MAIN, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    for r in merged:
        w.writerow({k: r.get(k, "") for k in fieldnames})

print(f"Wrote merged human_annotations.csv: {len(merged)} rows")
print(f"  by side: a={sum(1 for r in merged if r['side']=='a')}, b={sum(1 for r in merged if r['side']=='b')}, neutral={sum(1 for r in merged if r['side']=='neutral')}")
print(f"Expected: 900 rows (300 texts x 3 conditions: a, b, neutral)")

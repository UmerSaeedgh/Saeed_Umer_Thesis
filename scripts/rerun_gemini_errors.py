"""
Re-runs the Gemini API calls for cells that previously only got a rate-limit error
("error after 4 retries: 429 client error...") instead of a real answer.

Safety:
- Aborts immediately if GOOGLE_API_KEY is not set -- never blanks data it can't replace.
- Backs up all 3 pool CSVs before touching them.
- Only blanks Gemini_label/Gemini_reason/Gemini_label_norm for rows where Gemini_label
  currently contains the literal word "error" (267 rows, verified) -- every other cell,
  every other model, is untouched.
- Delegates the actual API calls to the project's own run_classification.py (paced at a
  conservative rate to respect Gemini's free-tier ~20 req/min quota), which is itself
  resumable/checkpointed.

Run: python rerun_gemini_errors.py
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

BASE = Path(r"C:\Users\umers\Desktop\thesis\thesis final")
FILES = {
    'unambiguous': BASE / 'experiment_results_unambiguous.csv',
    'author_independent': BASE / 'experiment_results_author_independent.csv',
    'author_relevant': BASE / 'experiment_results_author_relevant.csv',
}

if not os.environ.get('GOOGLE_API_KEY'):
    print("GOOGLE_API_KEY is not set in this session. Set it first, e.g.:")
    print('  ! export GOOGLE_API_KEY="your-key-here"')
    print("Aborting -- no files touched.")
    sys.exit(1)

total_blanked = 0
for name, path in FILES.items():
    backup = path.with_suffix(path.suffix + '.bak_pre_gemini_rerun')
    if not backup.exists():
        shutil.copy(path, backup)
        print(f"Backed up {path.name} -> {backup.name}")
    else:
        print(f"Backup already exists: {backup.name} (not overwriting)")

    df = pd.read_csv(path)
    is_error = df['Gemini_label'].astype(str).str.contains('error', case=False, na=False)
    n = is_error.sum()
    if n > 0:
        df.loc[is_error, ['Gemini_label', 'Gemini_reason', 'Gemini_label_norm']] = ''
        df.to_csv(path, index=False)
    print(f"{name}: blanked {n} Gemini error rows for reprocessing")
    total_blanked += n

print(f"\nTotal blanked: {total_blanked} (expected 267)")
print("\nRunning run_classification.py --models gemini --all (paced, workers=2, min-interval=3.5s "
      "to respect Gemini's free-tier ~20 req/min quota)...")

result = subprocess.run(
    [sys.executable, 'run_classification.py', '--models', 'gemini', '--all',
     '--workers', '2', '--min-interval', '3.5'],
    cwd=str(BASE / 'scripts'),
)
if result.returncode != 0:
    print(f"\nrun_classification.py exited with code {result.returncode} -- check output above.")
    sys.exit(result.returncode)

print("\nDone. Next steps (run manually to verify before trusting the result):")
print("  1. python normalize_labels.py")
print("  2. python build_unified_dataset.py")
print("  3. Check remaining Gemini error count with the same diagnostic used before")
print("  4. Rerun build_emotion_distributions.py -> qc_gate.py -> run_analysis.py -> notebooks")

# Persona redo + neutral collection (2026-09-17) — COMPLETE

Re-collection of human annotations for the 102 texts affected by the hidden-emotion-leak fix
(78 mask-fixed texts whose content changed after the original Sep 8 study, + 24 texts that
replaced ones dropped from the 300-text sample), plus first-time collection of the neutral
condition. Both are now merged into the main `human_annotations.csv` in `thesis final\`.

## Persona redo (persona_a / persona_b)

- `form_assignment_redo.json` / `form_items_flat.csv` — the 21-form assignment (612 items =
  204 (text_id, side) pairs x 3 raters each, ~29-30 items/form, no form repeats a text_id on
  both sides).
- `per_form_create_scripts/form_redo_01.gs` ... `form_redo_21.gs` — one Apps Script per form;
  each creates a fresh Google Form via `FormApp.create` (same "Text: ... / Author: ..." item
  format as the original persona study, from `build_prompts.py`'s persona_a/persona_b framing).
- `redo_first_response.csv` — the first (oldest) submitted response per form, one row per form
  as `form_key,emotion_1,emotion_2,...`. Forms occasionally received more than one submission;
  only the first is used (design intent: 3 raters per pair come from 3 *different* forms, not
  from multiple people re-filling one form). One form (form_redo_10) was accidentally created
  twice under an identical title — both copies' first responses are included as independent
  extra raters (`form_redo_10` and `form_redo_10_dup`), which is why 29 of the 204 pairs ended
  up with 4 raters instead of 3.
- `scripts/01_build_redo_annotations.py` — joins `redo_first_response.csv` against
  `form_assignment_redo.json` by item position, aggregates per (text_id, side) pair, writes
  `human_annotations_persona_redo.csv` (204 rows: text_id, side, persona, text, pool, reason,
  n_raters, raters_raw, human_majority, human_agree).
- `scripts/03_merge_redo_into_annotations.py` — merges those 204 rows into the main
  `human_annotations.csv`: replaces the 78 stale mask-fixed texts' old (Sep 8, pre-fix) rows,
  adds the 24 new replacement texts, and drops the 24 old dropped texts that are no longer in
  the current 300-text sample. Backs up the pre-merge file to
  `human_annotations_pre_redo_merge_backup.csv` before writing.

**Result: all 175 non-duplicated pairs got 3 raters, the 29 duplicate-Form-10 pairs got 4 —
full coverage, zero gaps.**

## Neutral condition

- Item assignment for the 10 neutral forms lives in
  `backup_pre_hidden_emo_fix/form_texts/all_forms_editplan_final.csv` (form, item_index,
  text_id, action — built during the same hidden-emotion-leak fix).
- `neutral_first_response.csv` — first response per neutral form, same format as the redo file.
- `scripts/02_build_neutral_annotations.py` — joins against the editplan (resolving
  `REPLACE_DROPPED` items to their real replacement text_id via `new_text_id`), writes
  `human_annotations_neutral.csv` (300 rows, one per text, side="neutral", persona="").
- `scripts/04_merge_neutral_into_annotations.py` — appends those 300 rows into
  `human_annotations.csv`. Backs up to `human_annotations_pre_neutral_merge_backup.csv` first.

**Limitation to carry into the analysis:** each neutral form was sent out once (no 3x
redundancy like persona had), so every neutral row has `n_raters = 1`. That's a real limit on
statistical power, but it's a single consistent rater pool answering fresh — a genuine
improvement over the old approach of reusing Phase 2's `reader_majority` column (a different,
disjoint rater pool), which was flagged as a confound in `complete_analysis.ipynb`'s RQ1/H1
section.

## Current state of `thesis final/human_annotations.csv`

900 rows = 300 texts x 3 conditions (persona_a, persona_b, neutral). This is the first version
of the file where all three human conditions come from data collected specifically for the
current (post-fix) 300-text sample.

**`complete_analysis.ipynb` has not yet been rerun against this updated file** — its cached
RQ1/H1 outputs still reflect the older 600-row (no-neutral) version. Re-run it before citing
current numbers.

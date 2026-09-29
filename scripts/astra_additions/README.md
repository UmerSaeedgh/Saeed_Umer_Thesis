# Astra's corrections, integrated 2026-09-25

These five scripts were supplied by an external review (GPT-6 Astra) as
`thesis_review_corrected/verify_proposal_additions.py` and four `refresh_*.py` figure
regenerators, alongside a corrected copy of the thesis `.tex` files. Every number and factual
claim they produce was independently re-derived from the raw data by Claude before anything was
merged (see the session transcript); all of it checked out:

- **Fleiss' kappa is genuinely computable** here without retained rater identities — the
  standard multi-rater formula only needs vote *counts* per item with a constant number of raters,
  which this data has for 754 of 900 cells. An earlier session had wrongly told the user this was
  impossible and written that into `methodology.tex`; that claim is now corrected.
- The neutral majority-label benchmark (macro-F1 / agreement against the original corpus's
  241 texts with a strict 5-reader majority) reproduced exactly against an independent
  from-scratch recomputation.
- The H4 text-level covariation figure (Spearman rho) turned out to differ between the two of us
  (0.324 vs 0.331) because of **1-ULP floating-point noise flipping tie-breaks** in the rank
  correlation — both `study.h.groupby(...)` and the `comparison_audit.csv` pivot are
  mathematically identical, but summed in different orders. Rounding both to 10 decimals before
  ranking converges them to the same value (0.3303), confirming Astra's 0.331 was correct and the
  session's earlier 0.324 was not.
- The four regenerated figures (fig05, fig07, fig12, fig13) fixed a real, previously-unnoticed
  bug: images referenced by `thesis_document/images/` had never been regenerated after the
  Sep 23 2026 numeric reconciliation, so they still showed pre-reconciliation numbers (e.g.
  fig13 cited a nonexistent "H5" and rho=0.19; fig05 showed the same stale rho). The text and
  figures are now consistent.

**Paths were patched** (see the `# patched from original sibling-of-script layout` comments) to
resolve correctly from this location (`thesis final/scripts/astra_additions/`) against
`thesis final/data/`, `thesis final/results/` and the sibling `../../thesis_document/images/`.
Every script was re-run from here after patching and reproduced byte-identical CSVs and
visually identical figures to Astra's originals.

One deliberate deviation from Astra's suggested text: the abstract's closing paragraph restores
a sentence Astra had dropped (the "0 of 6 vs 3 of 6" selected-label-vs-vector comparison for H1)
since it is accurate, was not disputed, and Astra kept the identical claim in `conclusion.tex`.

A backup of the pre-merge `thesis_document/content/*.tex` and `images/*.png` is at
`../../backup_pre_astra_corrections_20260925/` (relative to this file, i.e.
`thesis-2/backup_pre_astra_corrections_20260925/`).

## Second round, same day (a newer zip from ChatGPT/Astra)

A second, larger corrected-thesis zip arrived a few hours later. It added one genuinely new,
independently-verified test — `verify_persona_side_contrast.py`, a direct paired test of
persona A vs persona B for author-independent human texts (mean A−B = 4.44pp, 95% CI
[−2.08, 11.06], exploratory paired sign-flip p = 0.186 — confirms the two persona sides are
**not** shown to differ from each other, which earlier sessions had only argued qualitatively)
— and restructured `result.tex` into an explicit two-pass shape (all four hypotheses under the
old selected-label method first, then the vector-representation loss, then all four hypotheses
again with full distributions). Verified line-for-line by section label: every section from the
previous version survived untouched or was accurately updated; nothing was silently dropped.

**One thing this round deliberately did NOT copy in:** the zip also bundled its own copies of
`distribution_analysis_v1/*.ipynb`, `scripts/`, and `data/`. The three notebooks in that bundle
have noticeably fewer cells (41/27 vs the canonical 57/40) than the canonical, fully-inlined
"readable" notebooks rewritten on 2026-09-23 — meaning Astra was working from an **older
snapshot of the project**, from before that inlining rewrite, despite its own README claiming
they were "copied without changing code or outputs." Only the new script, its output CSV, the
four `.tex` chapters, and the plain-English defence guide were taken from this second zip. The
guide (`THESIS_STORY_IN_PLAIN_ENGLISH.md`) is intentionally kept out of this project folder and
lives at `~/Desktop/THESIS_STORY_IN_PLAIN_ENGLISH.md` instead, since it is defence-prep material
for the author, not something to submit alongside the thesis. **Lesson: always diff notebook cell counts/content before
trusting a "no changes" claim about bundled analysis code, even from a source that has been
reliable so far.**

My abstract punchline sentence (the "0 of 6 vs 3 of 6" comparison) was dropped again in this
round's abstract.tex and was re-added again after merging, for the same reason as before.

## Third round, same day (in direct response to `NOTE_TO_ASTRA.md`)

A third zip arrived after the note above was sent. It fixed the exact thing the note flagged:
`distribution_analysis_v1/*.ipynb`, `scripts/`, and `data/` in this round are **byte-identical**
to the canonical project files (verified with `cmp`), so the "older snapshot" problem from round
two is resolved. Astra's own README now correctly states the real cell counts (26/57/40 total,
12/34/24 code cells) rather than a stale claim.

Substantive content this round was small and precise — three sentences, all removing exactly
the kind of "accepting the null" overclaim this project has been hunting for throughout:

- Abstract and conclusion: "a selected-label version of this study would have concluded that
  human judgments do not respond to author framing at all" → "...would not have detected a
  statistically significant human framing effect in those tests." (Not detecting an effect is
  not the same claim as concluding there is none — the earlier wording overclaimed.)
- `result.tex`, H3 verdict: "H3 is supported in raw TV, but..." → "H3 shows the predicted raw
  ordering, but the framing-specific contrast is inconclusive..." This fixed a real internal
  inconsistency: the H3 *summary table* had already dropped the word "supported" in round one,
  but this sentence in the H3 section body still used it. Missed until this round.

Four regenerated figures were also included but confirmed visually identical in content to the
ones already in place (just re-encoded with different file sizes) — no image files were
replaced this round.

## Fourth round (2026-09-26): a much deeper external review reversed a prior "approved" verdict

A markdown review document ("Detailed final review — Round 6") withdrew an earlier unconditional
approval after a much more thorough pass: all 70 notebook code cells re-executed fresh, every
cited bibliography entry checked against its primary source, a character-boundary scan of all 80
PDF pages, and closer inspection of several figures and tables. Every claim in it was independently
re-verified before anything was changed (via CrossRef, arXiv, ACL Anthology and direct recomputation
against `results/comparison_audit.csv`), not taken on trust — and every one of them turned out to
be correct:

- **Bibliography accuracy (highest priority).** This had never actually been checked against
  primary sources before, at any point in this project — every prior round's bibliography work
  only fixed the Acheampong duplicate and its author names. This round found the *rest* of the
  bibliography was substantially wrong: `Hong2024AERLLM` cited a real but completely unrelated
  arXiv paper under a fabricated author list; `Leonard2020AnnotatorDisagreement` turned out to be a
  fabricated citation with no real underlying paper (confirmed via a CrossRef search returning no
  match, and by resolving its DOI to an unrelated document-segmentation paper) — the one sentence
  citing it was redirected to `Davani2022Disagreement`, already used elsewhere for the same point;
  `Troiano2023CrowdEnVent` and `Henestrosa2023Authorship` both had placeholder `note = {...requires
  verification}` fields still sitting in the bib file with fabricated titles/venues; and six further
  entries (Zhang, Chou, Prabhakaran, Geva, Davani, Santurkar, Basile, Waseem) had wrong DOIs, wrong
  page ranges, wrong given names, a missing co-author, or truncated titles. All corrected against
  primary sources (CrossRef, arXiv, ACL Anthology, or the publisher directly), each with an inline
  `% CORRECTED` comment in `database.bib` recording what was wrong and how it was verified.
- **Figure 5.10 (dominant-emotion transition matrix) used stale data.** Same staleness class as the
  earlier fig05/07/12/13 bug: generated from an old, never-refreshed `distribution_analysis_v1/
  hypothesis_tests/` CSV. The LLM panel's n (3,321/3,600) didn't match a fresh recomputation
  (3,309/3,600); the human panel (338/600) was already correct. Fixed by
  `refresh_transition_matrix_figure.py`, sourced directly from `results/comparison_audit.csv`.
- **Table 5.15's entropy-test p-values were not reproducible from anything in this project.** The
  six-test calculation existed only in the superseded `scripts/run_analysis.py`
  (`entropy_change_summary()`) and was never carried into the canonical notebook pipeline. Running
  that exact, documented method (per-text mean entropy change, one-sample Wilcoxon signed-rank vs.
  zero, BH-FDR jointly across all six tests) against current data reproduces the table's *means*
  exactly but gives different p-values than what was printed, and no version of this file anywhere
  in the project (current or any backup) matches the printed p-values either. Table 5.15 was
  updated to the newly verified, reproducible numbers (same qualitative pattern: all three human
  categories significant, all three model categories not) via the new
  `verify_entropy_change_tests.py`, which is now the canonical source for this table.
- **The generic-author "predominantly attributable to identity" claim survived in four more places**
  after an earlier round had already fixed it in the abstract, conclusion and result.tex body text:
  a Results summary-table row (Table 5.18) and a Discussion paragraph both still asserted it. Fixed
  to report the two measured distances without an attribution/additivity claim, matching the
  already-corrected wording elsewhere.
- **Causal-sounding language around H3's excess-TV and H4's count-standardisation reduction**, e.g.
  "attributable to framing" / "attributable to baseline dispersion" / "vote-count granularity
  accounts for four percent," overstated what these sensitivity analyses establish. Reworded
  throughout result.tex, discussion.tex and conclusion.tex to describe them as distances above a
  reference point, not causal decompositions.
- **Exchangeability was asserted too strongly in two places** (implementation.tex's human-annotation
  section, methodology.tex's participant-count deviation note) compared to the correctly-qualified
  version already in the permutation-test methodology. Both now consistently state that disjoint
  raters remove within-text pairing but do not by themselves establish exchangeability.
- **Methodology's framing-manipulation claim ("any difference is attributable to the attribution")
  was too strong** — persona pairs can differ in substantive situational content, not just identity
  (confirmed against the actual data: text 2105's persona pair really does contrast "struggled all
  year" against "expected to pass easily"). Reworded to describe sensitivity to the framing as a
  whole rather than a pure identity mechanism.
- **A results-chapter passage described a single rater "changing their mind"** to explain
  new-emotion appearance rates, which is impossible under the disjoint-rater design (no individual
  ever rated both conditions). Reworded to describe a change in vote *support* instead of an
  individual transition, and to note that support detection is itself a function of sample size.
- **Three layout/editorial defects**: a file path (`results/model_explanation_coverage_verified.csv`)
  extended past the page margin on page 17 — fixed with `\allowbreak`; the Contents entry for
  "Bibliography" showed the wrong page number because `\addcontentsline` was placed after
  `\printbibliography` instead of before it — reordered; a subsection heading still said "Four
  Outcomes" when the body text and list had already been corrected to five — fixed. While visually
  scanning pages for these, also caught and fixed (not in the original review, found independently)
  an unrelated cosmetic bug: Figure 5.11's annotation text overlapped a gridline making it look
  crossed out — fixed with a white background box (`refresh_prompt_figure.py`).
- **Reproducibility-archive scope**: the archive's own README overstated what it could regenerate.
  Both new scripts above are now included in `Analysis_Code_and_Data/scripts/astra_additions/`,
  narrowing but not closing this gap — see that README for what is and is not actually covered.

Everything above was verified before merging: bibliography facts against CrossRef/arXiv/ACL
Anthology directly (not just against the review's claims), the entropy p-values by literally
running the documented method against current data, and the transition-matrix count by an
independent recomputation from `comparison_audit.csv`. `check_tex.py` reports 0 structural problems
after all changes; H1-H4 numbers were re-checked against `results/*.csv` and are unchanged.

## Fifth round (2026-09-26): a follow-up spot-check found one real transcription error and four more real issues

A follow-up review independently re-derived the entropy table itself and compared it to the printed
PDF, plus checked one bibliography entry's exact venue/pages and flagged remaining overclaim
language. All confirmed real:

- **A real transcription error, found by comparing my own already-correct CSV against what I'd
  actually typed into the LaTeX table.** `results/entropy_change_by_category_verified.csv` already
  had the correct Model p_BH values (0.226, 0.230, 0.350), but the previous round's edit to
  `result.tex` had typed different, wrong numbers (0.209, 0.261, 0.367) into Table 5.15. Fixed to
  match the CSV exactly. Does not change any conclusion (all three remain non-significant).
- **The Henestrosa literature claim was corrected with newly-supplied source content.** The actual
  finding is that authorship cues affected perceptions of the *author*, with no detected difference
  in the *article's* credibility or trustworthiness between AI/human conditions -- a real
  distinction the previous wording blurred. Reworded per the supplied replacement text.
- **The Basile bibliography entry mixed two publication versions.** It linked to the CEUR workshop
  paper but gave the venue/pages of a different, longer proceedings version. Verified directly
  against the CEUR-WS Vol-2776 index: the linked paper is in the AIxIA 2020 Discussion Papers
  Workshop, pages 31-40, not pages 441-453. Corrected.
- **The entropy-dispersion "human vs. model" contrast was overstated in four places** (result.tex,
  discussion.tex twice, conclusion.tex): a significant result on one side and a non-significant
  result on the other does not by itself establish a statistically significant difference between
  the two, since no direct test of that difference was performed. All four reworded to report the
  two findings separately and describe the apparent contrast cautiously, per the correction
  pattern this project has applied to every other "accepting the null" overclaim.
- **A Future Directions sentence assumed what it proposed to test** ("would directly test whether
  humans, like models, respond mostly to persona content" -- begging the question the generic-human
  extension is meant to answer). Reworded to propose the comparison without presupposing its
  result.
- **Model provenance, recovered where possible.** Configured default model identifiers were found
  in the collection scripts and Ollama run logs (not previously surfaced): `gpt-5-nano`,
  `claude-haiku-4-5-20251001`, `gemini-3.6-flash`, `qwen3:8b`, `gemma3:4b`, `ministral-3:8b`,
  `llama3.1:8b`. No decoding parameters (temperature, top-p) were ever explicitly set for any
  model, and only the open-weight models' collection dates could be recovered from timestamped
  logs (2026-09-03/04); this is stated honestly in `implementation.tex` rather than left
  unaddressed.
- **The archive's "every calculation is in the notebooks" claim was narrowed**: several
  supplementary results (IAA, the majority-label benchmark, the entropy tests, the transition-matrix
  figure) are produced by standalone scripts, not inlined into the three core notebooks.

Recompiled after all fixes: 82 pages, 0 errors, 0 broken references. H1-H4 numbers re-checked and
unchanged.

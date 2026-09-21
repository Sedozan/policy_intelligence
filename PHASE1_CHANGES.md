# Phase 1 changes — correctness fixes from the ChatGPT + Gemini reviews

All fixes are pure-Python and covered by the offline suite (**35 tests, all passing**;
7 new). No Spark/LLM/PDF access was required. SQL is still verified as strings, not
executed on a cluster — that remains a Phase-2/3 gap.

| # | Fix | File(s) | Reviewer(s) | Test |
|---|---|---|---|---|
| 1 | **P0 — temporal collapse.** `_edition_key` now includes the effective/end window, so different time-runs of the same code/edition never merge and windows are never widened. Only same-window editions collapse. | `rule_normalize.py` | ChatGPT (P0), Gemini (#6) | `test_collapse_never_merges_different_time_windows`, `test_collapse_different_editions_different_windows_stay_separate` |
| 2 | **PTP modifier bypass on both codes.** A modifier-indicator-1 pair is bypassed by a valid NCCI modifier on *either* the Column 1 or Column 2 code (was only `b`). | `detector_gen.py` | ChatGPT | `test_ptp_indicator1_uses_full_bypass_set_on_both_codes` |
| 3 | **NCCI MUEs are per-line.** CMS Medicaid guidance: MUEs are not summed across lines for a DOS. Ingest now always emits `max_units_per_line`; MAI is recorded for audit only, never routed to a per-DOS sum. | `ingest_ncci_tables.py` | ChatGPT (contra Gemini) | `test_mue_always_per_line_even_when_mai_present` |
| 4 | **Grounding tightened.** Drops a proposal unless *all* its codes are in the source (no more `some` pass); threshold must appear in the *quoted sentence*, not anywhere in the section. | `extract_rules.py` | ChatGPT, Gemini | `test_grounding_requires_all_codes_and_sentence_scoped_threshold` |
| 5 | **Effective-date parser broadened.** Handles "Beginning with dates of service on and after April 1st, 2018", "on or after January 1, 2020", month-name dates — not just `effective MM/DD/YYYY`. A date with no effective-date cue is ignored. | `extract_rules.py` | ChatGPT | `test_effective_date_phrasings` |
| 6 | **Per-rule period.** The extractor reads the period from the text; "contract year"/"benefit year" → Oct 1–Sep 30 bucket (verified in the manual), instead of assuming "month". | `extract_rules.py` | Gemini (fiscal year, corrected) | `test_stub_period_reads_contract_year_as_oct_sep` |
| 7 | **Review "Fix" corrects the predicate.** A "fix" now requires a valid `corrected_predicate` JSON to change the executable rule and be approved; a prose-only fix is parked as `needs_review` (it can no longer silently approve a wrong predicate). Workbook gains `corrected_predicate` + `extraction_key` columns. | `state_rules.py` | ChatGPT | `test_review_fix_requires_predicate_to_change_executable_rule` |
| 8 | **Audit join restored.** `extraction_key` is now a first-class field that survives the row round-trip and normalization and is carried onto the compiled detector and the review log, so an approval joins to the exact executable rule (rule_id is still content-re-derived). | `schema.py`, `state_rules.py`, `detector_gen.py`, `run_detectors.py` | ChatGPT | `test_extraction_key_survives_normalization_for_audit_join` |
| 9 | **Honesty / telemetry.** Scope query gains a `PAID_FILTER` hook and stops labelling the scope "paid" when nothing filters; `run_detectors` records `detectors_eligible / detectors_run / partial_run` and warns loudly on a capped run; the pre-filter records skipped sections (`sections_skipped_noncandidate`, `skipped_headings`) so recall is measurable; stale `route_note()` rewritten. | `notebooks/build_kb.py`, `notebooks/run_detectors.py`, `extract_rules.py`, `ingest_ncci_tables.py` | ChatGPT | (covered by extraction stats tests) |

## Deliberately NOT in Phase 1 (needs data or a decision)

- **NCCI edition collapse default** (most-permissive vs ambiguity-lead vs provider-type routing) — needs a frequency measurement on the real claims data. Phase 1 only made the collapse *safe*, not *decided*.
- Layout-aware PDF parsing (needs the real PDFs), category→code resolver (needs the fee schedule), composable/typed predicate grammar with conditions & carve-outs (changes table shapes), NLI advisory gate, PTP self-join → combinations rewrite. See `state-extraction-upgrade-plan.md` in the project.

## Verified against primary sources
- AHCCCS contract/benefit year = Oct 1–Sep 30 (FFS Ch10; benefit-changes memo).
- Medicaid NCCI MUEs are per-claim-line (CMS Medicaid NCCI Technical Guidance).
- NCCI editions apply by provider context, not code (same guidance) — hence the collapse is a documented workaround, flagged in `route_note()`.

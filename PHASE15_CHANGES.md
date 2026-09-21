# Phase 1.5 — correctness fixes from ChatGPT code review #2 (2026-09-20)

Scope: bugs in the **trusted path** (compiler, gate, review importer, extraction identity)
that had to be fixed before building the Phase 2 grammar on top of them. All offline-testable.
Tests: **63 passing** (35 original + 28 new in `tests/test_phase15.py`). ChatGPT's own probe
script now reproduces **4 of 15** defects, all four deliberately deferred (see bottom).

Primary sources verified this session (verbatim quotes in the adjudication doc):
CMS 2026 Medicaid NCCI Technical Guidance (Column Two denied; same-anatomic-modifier rule;
Column One payment eligibility; PA is not a bypass) and AHCCCS FFS Ch10 p.9 (CHW limits, PA
exception, three-code prohibition).

## Facts confirmed by SB (now in `config.py`)
| Fact | Config | Consequence |
|---|---|---|
| `all_data_C_A` is paid-only | `CLAIMS_PAID_ONLY = True` | F09 is a no-op; `build_kb` labels scope honestly. Run once: `SELECT COUNT(*) ... WHERE PMT_AMT < 0`. |
| NCCI tables from CMS public files | `NCCI_SOURCE = "cms_public"` | Findings are "per CMS public edit set; AZ deactivations not applied" (printed by `build_kb`). |
| `EVENT_PA_NO` = PA number, populated ⇒ PA exists | `CLAIM_COLS["pa_no"]` | Three-outcome model below. |

## Post-run fix (2026-09-20 21:52): the model never ran, and the pipeline hid it
SB's first real `extract_chapters` run finished in 16 s with 7 drafts, every section logging
`HYBRID: model backend failed (KeyError); using patterns only`. Root cause, in code Claude wrote:
`DBFMBackend._client_` swallowed the real error from the databricks-sdk route with a bare `except`,
then the fallback read `os.environ["DATABRICKS_HOST"]`, which notebooks do not set -> `KeyError`;
`HybridBackend` swallowed that too and printed only the type name; the run summary showed no model
statistics. Every draft ever produced by this pipeline has been regex-only.

Fixed:
- `DBFMBackend`: three client routes (databricks-sdk OpenAI client -> mlflow.deployments -> openai
  against `<workspace>/serving-endpoints` with env or notebook-context token). Every failure is
  recorded with its full message; if all fail, `ModelUnavailable` names all three. `finish_reason ==
  "length"` (truncated JSON) is a loud, retryable failure. `healthcheck()` makes one real call and
  returns the model's reply.
- `HybridBackend(require_model=True)` **by default raises** when the model fails; patterns-only output is
  opt-in (`make_backend(..., allow_patterns_only=True)`), labelled `degraded`, never cached, counted.
- `extract_chapter` stats: `model_failed`, `degraded_patterns_only`, `unparseable` separately, plus
  `model_calls / model_failures / model_route / last_model_error`.
- `notebooks/extract_chapters`: **health-check cell before any chapter**; hard stop if the model is dead
  unless `ALLOW_PATTERNS_ONLY=True`; per-chapter model-failure line; end-of-run summary prints model
  calls vs failures, drafts by source, and a `*** EVERY MODEL CALL FAILED ***` banner when applicable.
- `notebooks/diag_model.py` (new): tries each route on its own with full tracebacks, lists visible
  serving endpoints, checks the configured endpoint name.
- `config.LLM_MAX_TOKENS` 2048 -> 4096 (gpt-oss reasoning consumes output budget).

## HFLOCAL: the model path is now SB's proven loop (2026-09-21)
SB's `llm_summarization_notebook` does not use a serving endpoint: it loads `openai/gpt-oss-120b`
with `transformers` on Serverless GPU and parses output with `openai-harmony`. `DBFM` was the wrong
default for this workspace. `HFLocalBackend` is now a line-for-line mirror of `generate_with_harmony` +
`generate_final_with_retry`:
- `load_hf_runtime()`: `login(hf_token)` from `config.HF_TOKEN_PATH`, `from_pretrained(torch_dtype="auto",
  device_map="auto", trust_remote_code=True)`, `load_harmony_encoding(HARMONY_GPT_OSS)`; **once per
  process** (module-level singleton).
- `apply_chat_template(..., add_generation_prompt=True, return_dict=True, reasoning_effort=<level>)` ->
  `model.generate(max_new_tokens=<budget>, do_sample=False)` -> `parse_messages_from_completion_tokens`;
  keep `final`, drop `analysis`; no final + hit budget + no parse error -> one retry at
  `RETRY_BUDGET_FACTOR` (1.8x). Deliberate differences: no deterministic fallback (a failed call raises
  and is recorded as `model_failed`, never cached); the "reasoning leak" check is replaced by the JSON
  grammar check.
- `config`: `HYBRID_MODEL_BACKEND="HFLOCAL"`, `HF_MODEL="openai/gpt-oss-120b"`, `HF_TOKEN_PATH`,
  `LLM_REASONING="low"` (extraction is a reading task), `REASONING_TOKEN_BUDGET` (low 3000 / medium
  4500 / high 7000 - JSON output grows with rules per section), `CACHE_FLUSH_EVERY=25`.
- `extract_chapter(checkpoint_fn, checkpoint_every)`: flushes the model-output cache to Delta every N
  new model calls (SB's `FLUSH_EVERY` pattern) and once per chapter; previously written only at the very
  end of the run, so a crash at minute 55 of an hour-long run lost everything.
- `notebooks/extract_chapters`: `%pip install ... torch accelerate sentencepiece openai-harmony` +
  `dbutils.library.restartPython()` preamble, **Serverless GPU**, a model-load cell that mirrors the
  summarization notebook, `REASONING` setting, per-chapter timing and retry/failure counts.
- Tests: fake tokenizer/model/harmony objects shaped like the notebook's; assert the exact call
  arguments, final-only extraction, 1.8x retry, loud failure on no final channel, and the checkpoint
  firing through HYBRID. **59 passing.**
- Expect ~1 hour for the six MVP chapters (one generate per candidate section on a 120B model).

## GPU / CPU split + a read-safety fix (2026-09-21, after the CPU-cluster run)
SB's run on "Big Data Memory Opti…" failed every chapter with `PermissionError [Errno 13]` on the Volume and
ended with `[CLOUD_ACCESS_DENIED] … 403 … AuthorizationFailure` on the metastore storage: that cluster's
identity cannot reach Unity Catalog storage at all. Worse, the notebook's `_table_or_none` swallowed the
403 on `state_rules` and returned None, so the merge produced an empty frame and called
`write_state_rules(mode="overwrite")` - only the skip-on-empty guard (deferred F08) stopped it from
wiping the reviewed rules.
- `kb_io.table_or_none(spark, fqn)`: None ONLY for a missing table; any other failure raises
  `TableAccessError` naming the cluster-access cause. Used by every notebook that reads a table.
- `extract_io.guard_state_rules_merge`: refuses an overwrite that would drop a reviewed row.
- **Three-notebook split**, mirroring the summarization pipeline (GPU writes files, CPU writes Delta):
  `stage_inputs` (CPU, UC: PDFs + cache/state_rules/inventory snapshots -> Workspace files, stops on a
  read failure) -> `extract_chapters_gpu` (Serverless GPU, no Spark: reads staged files, runs the model,
  rewrites `RUN_ROOT/<run_id>/*.jsonl + cache.json + manifest.json` after every chapter, cache flushed every
  25 calls) -> `persist_extraction` (CPU, UC: merges drafts into the LIVE table with the guard, appends
  telemetry, overwrites cache, exports the workbook; refuses unfinished or all-failed runs).
- `extract_io.py`: the JSONL/JSON run-folder format (atomic writes, NaN -> null), snapshot export/load.
- `config`: `WORKSPACE_DATA_DIR`, `STAGE_DIR`, `STAGE_PDF_DIR`, `RUN_ROOT`.
- The single-cluster `extract_chapters` remains for compute that has BOTH a GPU and UC access; it now uses
  `table_or_none` and the guard.
- Tests: missing-vs-denied reads, the shrink guard (the exact near-miss), run-folder round trip, snapshot
  export stopping on a denied read. **63 passing.**
- gpt-oss-120b on CPU is NOT an option (days, not hours); the model stays on Serverless GPU.

## Changes by module

### `detector_gen.py` (F04, F05, F10, F17-evidence, PA)
- **Same-anatomic-modifier exception (F04).** Indicator-1 pairs: bypass on any NCCI-associated
  modifier on either line, *unless* both lines share the same anatomic modifier and neither
  carries 58/59/78/79/XE/XP/XS/XU. `LT`+`LT` now flags; `LT`+`LT`+`59` bypasses; `LT`+`RT` bypasses.
- **Column Two is the finding target (F05).** Primary `claim_id/line_no` = line `b` (the denied
  code). Column One travels as `col1_claim_id/col1_line_no`. Grain: one row per (b line, a line).
- **`predicate.scope` (F10)** ∈ `{member_provider_day, member_day}`. Explicit wins; default by
  `origin` (`ncci` → provider; anything else → member). `service_category` is a routing label and
  never decides scope. Emitted as `pair_scope` on findings. `pair_scope(rule)` is public.
- **PA three-outcome model.** For state (non-NCCI) unit / period / not-covered detectors, a
  populated PA number sets `pa_present = 1` and `finding_severity = 'verify_pa'`; otherwise the base
  severity. PA never suppresses; NCCI edits ignore PA entirely (CMS). Not applied to pairs.
- **Aggregate evidence (F17).** `max_units_per_day/dos/period` emit
  `evidence_lines` = `claim:line,...` of contributors (Spark `COLLECT_LIST`).

### `state_seed.py` (F02)
- CHW same-day prohibition seeded as **all three pairs** (was 1 of 3); all state pairs carry
  `scope: member_day`. CHW caps carry the PA-exception note from Ch10 p.9.
- Seeds still load as `legacy_hand_authored`. Flipping them to `draft` empties the state KB
  until a human re-signs 21 rules — **SB's call**, not made here.

### `extract_rules.py` (F06, F11, F12, F14, truncation)
- `semantic_key` now includes **population** and **effective_date**; for **uncoded** proposals
  it includes the normalised quote, so distinct leads no longer collapse in the within-run dedup.
  Statement wording stays out (model rephrasing must not fork identity).
- `HybridBackend._key` compares the full proposition (period, scope, modifier, reason,
  population); corroboration bump only when **sources differ**.
- Hybrid returns `degraded: true` when the model half failed; `propose()` catches backend
  exceptions as `backend error: ...`; `extract_chapter` **never caches** unparseable / backend-error /
  degraded outcomes (`stats.retryable_sections`).
- Grounding: a fuzzy match (≥0.85) is accepted only if quote and source sentence agree on
  negation/condition tokens and every number; otherwise `quote_found = "drift"` and dropped with
  the nearest sentence in `drop_reason`.
- Pattern extractor: "do not bill X **unless/except/if/when/only/prior authorization**" is a
  `not_checkable` lead (`predicate_not_in_grammar`, ambiguous) — never an unconditional
  `code_not_covered`.
- `predicate_valid`: threshold must be integral (`4.9` rejected; `4.0`, `"4"` accepted).

### `validate_gate.py` (F01)
- A machine-checkable rule with `population` ∉ {all, any, ""} is downgraded to a lead
  (`requires_member_attribute`), population preserved for the reviewer. Lifts when the Phase 2
  conditions grammar exists.

### `state_rules.py` (F07)
- Export carries `doc_hash`. Import: **reviewer required** (else parked, `no_reviewer`);
  **version check** (workbook `doc_hash` ≠ current → parked, `stale_version`); a **fix re-derives
  `codes` and `required_claim_fields`** from the corrected predicate and supersedes the grounding
  record. Review log gains `applied_outcome` (attempted vs. applied).

### `ingest_ncci_tables.py` (F15 residual)
- `quarter_gaps()`; `mue_history(..., allow_gaps=False)` refuses non-contiguous quarters and names
  the missing ones. `config.NCCI_ALLOW_MUE_GAPS=True` infers across a gap and flags every run that
  spans one as ambiguous.

### `notebooks/preflight_ncci.py` (new, read-only)
Run before the first `build_kb` after Phase 1.5 and whenever a new NCCI quarter lands. Answers, with
evidence, the four things the ingest otherwise assumes: (1) MUE quarter contiguity (the build now
refuses a gap), (2) whether older PTP quarters hold pairs absent from the newest (latest-only loading),
(3) loaded quarters vs the claims date range (claims before the earliest MUE quarter are unchecked),
(4) how many codes/pairs disagree across editions, weighted by paid lines and dollars (decides the
collapse policy and F03). Plus (5) negative-amount paid lines (closes F09). Prints GO / NO-GO.

### `chapter_profile.py` + `notebooks/profile_chapters.py` (new, read-only, no LLM)
Answers "is the framework overfit to Ch10/13/19/22?" with numbers: per chapter, what the parser sees
(pages, sections, blank/scanned pages), what the pre-filter keeps, which code families the text uses
(CPT/HCPCS the pipeline can see; revenue/DRG/NDC it cannot), which rule cues dominate (modifier,
per-diem, PA, rate, age, POS), and what the deterministic extractor yields. Heuristic profile per
chapter: `code_rich` (ready) / `category_driven` (resolver) / `modifier_driven` (`modifier_required`) /
`foreign_codes` (new CODE_RE families) / `process` (KB text only) / `scanned` (OCR). Thresholds are
visible in `classify()`; adjust after one real run. Tested on the fixture PDF + five synthetic profiles.

### `notebooks/run_detectors.py`, `notebooks/build_kb.py`
- Findings carry `pa_present, col1_claim_id, col1_line_no, pair_scope, evidence_lines`.
- Honest labels for paid-only scope and CMS-public NCCI source.

## Deliberately NOT fixed here (Phase 2, per adjudication)
| Probe still reproducing | Why deferred |
|---|---|
| Overlapping edition windows (F03) | Needs interval-split collapse; decide after measuring edition-disagreement prevalence (open decision #3). |
| Empty publication keeps stale tables (F08) | Needs explicit per-table schemas for zero-row writes + a build manifest. |
| Repeated body line stripped as header (F13) | Needs a body sentence on ≥50 % of pages — implausible on real chapters; cheap margin-only hardening queued. |
| Table attaches to last section on page (F13) | Real, but validation chapters are text-not-table; fix with the table-density gate. |

Also queued for Phase 2 grammar: mid-period effective dates (a benefit-year cap effective Apr 1
currently excludes Oct–Mar utilisation from the SUM → undercount); typed conditions/carve-outs;
`modifier_required` compiler; category→code resolver.

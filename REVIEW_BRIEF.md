# Policy Intelligence Engine (PIE) — MVP review brief

You are being asked to critically assess the **logic and execution quality** of this
codebase. Please be adversarial: find correctness bugs, unsound assumptions, scaling
problems, and places where the design claims more than the code delivers. A section at
the end lists the specific things the author is least sure about — scrutinize those hardest,
but don't limit yourself to them. Do not grade on a curve; assume this is going into a
government Medicaid program-integrity setting where a false positive wastes an
investigator's time and a false negative misses fraud.

## 1. What the system does

Turns Arizona Medicaid (AHCCCS) billing policy into **auditable SQL detectors** that run
against ~17M FFS claim lines. Two rule sources:
- **Federal NCCI edits** — structured quarterly tables (PTP code-pair edits + MUE unit
  limits) already loaded as Delta tables.
- **State AHCCCS manual** — 29 policy PDFs; rules extracted into the same schema.

Pipeline: **sources → `Rule` objects (closed predicate grammar) → validation gate →
normalize → compile to SQL detectors → run against claims → findings + coverage + gap
tables.** An LLM is used only to *propose* state rules from PDFs; it is never in the
detection path. Runs on Databricks/Spark (Unity Catalog), Python 3.11.

Design center of gravity: **the `Rule` object is the product.** Interpretation happens once,
is human-reviewable, and everything downstream is mechanical translation.

## 2. Modules (≈3,360 LOC Python + 5 notebooks; 28 offline tests, all passing)

| module | LOC | responsibility |
|---|---|---|
| `schema.py` | 133 | `Rule` dataclass; closed predicate grammar (compiled vs authorable); content-derived stable IDs |
| `config.py` | 134 | all table names, claims column map, NCCI routing config, 29-chapter registry (URLs), LLM backend settings |
| `validate_gate.py` | 118 | gate: clean / patched / downgraded / needs_review / rejected — with explicit reasons; downgrades rules needing absent claim fields to review leads |
| `ingest_ncci_tables.py` | 363 | NCCI PTP+MUE tables → `Rule`s: Spark-side scope filter, latest-PTP-quarter dedup, MUE run-length reconstruction across quarters, MAI, per-rule effective/deletion windows, provenance |
| `policy_docs.py` | 283 | Stages 0–2: acquire+hash PDF, PyMuPDF parse (pypdf fallback), strip repeated headers, ALL-CAPS heading segmentation, cheap rule-signal pre-filter |
| `extract_rules.py` | 599 | Stages 3–5: STUB/DBFM/HFLOCAL/**HYBRID** backends; prompt into closed grammar; deterministic **grounding** (quote-in-source, codes-in-source, predicate well-formed); draft rows |
| `rule_normalize.py` | 180 | plane derivation, provenance, content-dedup, **cross-edition NCCI collapse to most-permissive** |
| `detector_gen.py` | 287 | compile rules → Spark SQL; per-rule date windows; PTP modifier-bypass gated on indicator; MUE line vs DOS; routing; severity |
| `gap.py` | 93 | claims-weighted coverage + gap (paid-dollar exposure) |
| `state_rules.py` | 236 | reviewed state-rule Delta table; draft merge (never downgrade approved / resurrect rejected); Excel review round-trip |
| `state_seed.py` | 123 | 19 hand-authored, cited Ch10/Ch19 rules (bootstrap) |
| `kb_io.py` | 54 | pandas→Delta with explicit schemas (avoids all-None inference failures) |
| `policy_changes.py` | 64 | quarter-over-quarter diff by rule meaning; document-hash change log |
| notebooks | — | `migrate_state_rules`, `extract_chapters`, `build_kb`, `run_detectors`, `demo_kb` |

## 3. Design decisions worth judging (and the rationale)

1. **Closed predicate grammar; LLM proposes, never decides, never in the detection path.**
   Detectors compile deterministically from a fixed set of predicate types. The LLM only
   emits into that grammar as drafts.
2. **Deterministic grounding gate.** Every extracted proposal must (a) quote a sentence that
   actually exists in the source section (exact or ≥0.85 fuzzy) and (b) name codes that
   appear in that section, or it is refused and logged. This is the hallucination control.
3. **Hybrid extraction.** Deterministic regex patterns AND the model both propose into the
   same grammar; results are unioned; a rule both find is marked *corroborated* (confidence
   bump); if the model is unavailable the patterns still produce a KB.
4. **Human review lifecycle.** Extracted rules land as `draft`; only `approved` /
   `legacy_hand_authored` rows compile. Excel round-trip records reviewer + date.
5. **Content-derived stable rule IDs** (the PoC had ID collisions — 9 rules sharing one ID).
6. **NCCI correctness:** per-rule effective/deletion date windows; latest-quarter PTP dedup
   (files are cumulative); MUE validity reconstructed as run-length over the quarter
   sequence (removed values retired); full ~30-modifier NCCI bypass set gated on the
   edit's modifier indicator (0 ⇒ no bypass clause at all).
7. **Routing by code, not claim form.** A diagnostic join showed all three NCCI editions
   (practitioner/outpatient/DME) land on the same `FORM_TYP` value ('A'); form cannot
   separate them. So NCCI detectors route by procedure code, and a code appearing in
   multiple editions is collapsed to the **most-permissive** limit (max MUE threshold; PTP
   bypass allowed if any edition allows it) to avoid false-positiving a shared claim.
8. **Claims-scoped rule universe** — the KB is bounded to codes actually paid, so a rule
   only exists where it can fire.
9. **Idempotent build; explicit Delta schemas** (`kb_io`) to survive all-None columns and
   UC Volume FUSE write quirks (openpyxl seek-writes fail on Volumes → write local, copy).

## 4. Environment & testing honesty

- **No local Spark / no live LLM endpoint / no internet egress in the dev sandbox.** So the
  28 tests are **offline**: they exercise the pure-Python logic (ingest transforms, gate,
  normalize, compilers producing SQL *strings*, grounding, merge, hybrid union, change
  diff, xlsx round-trip) against synthetic frames and a **fixture PDF built to mimic the
  real Ch10 layout**. The SQL is asserted as text, **not executed against Spark**. The
  STUB extractor (not a real model) drives the extraction tests.
- Column names / date-int formats were verified once against the user's real tables and a
  live read of Ch10; not continuously.

## 5. Where the author is least confident — please attack these

1. **Grounding checks provenance, not semantics.** A proposal can quote a real sentence and
   name real codes yet attach the *wrong predicate or threshold* and still pass grounding
   (quote present, codes present). The only backstop is human review. Is that sufficient, or
   should grounding also verify the number/relation against the quoted sentence?
2. **Most-permissive cross-edition collapse** trades false negatives for false positives
   (a practitioner claim that violates the stricter practitioner MUE passes if a laxer DME
   MUE exists for the same code). Correct call for an MVP, or the wrong default in an FWA tool?
3. **PDF parser is heuristic and validated against ONE chapter's layout (Ch10).** Header
   stripping, ALL-CAPS heading detection, and paragraph re-flow may degrade on table-heavy
   chapters (13 DME, 22 Nursing Facility) or multi-column pages. How brittle is this, really?
4. **PTP detector is a self-join** on the claims table per code pair (`a JOIN b ON member,
   provider, DOS`). Across 17M lines and thousands of pairs, is this viable, and how should
   it be batched/rewritten?
5. **Benefit-year bucketing** uses `YEAR(ADD_MONTHS(dos, 3))` to model an Oct 1–Sep 30 year.
   Verify this is correct at the boundaries.
6. **Cross-edition collapse widens the date window to the union** of members' windows —
   could it keep a deleted edit active because another edition still had it?
7. **MUE MAI is absent from the Medicaid files**, so everything is modeled as per-line; the
   per-DOS summation path exists but is dormant and untested on real data.
8. **Findings are not de-duplicated across overlapping detectors**, so summing exposure over
   findings can double-count (flagged in code, not solved).
9. **`schema.Rule` is large and flat** (24 fields); is the closed-grammar approach the right
   abstraction, or will it ossify as more predicate types are needed?

## 6. Useful questions to ask the reviewer
- Is the LLM-proposes / deterministic-verifies / human-approves split actually sound, or is
  there a failure mode it misses?
- Where will this break first at production scale, and what's the cheapest fix?
- Is anything here over-engineered for an MVP — what would you cut?
- Rate correctness, maintainability, and testing separately, and name the single highest-risk file.

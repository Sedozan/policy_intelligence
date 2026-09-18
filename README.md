# Policy Knowledge Base — MVP package

This is the complete engine. Every module below **replaces** the PoC file of the
same name; the authoring modules you wrote by hand (`ingest_ahcccs.py`,
`demo_three_beats.py`, `demo_ch10_beats.py`, `ch10_full_extract.py`,
`policy_clarity_report.py`) are untouched and still work: `Rule` keeps every field
name, `assign_id()` and `to_dict()`, and `schema` still exports `PLANES`,
`BINDING`, `PREDICATE_TYPES`. `analysis.py` and `store.py` are retired.

```
config.py               claims-schema mapping, tables, routing maps      (replaces)
schema.py               Rule + closed predicate grammar + stable ids     (replaces)
validate_gate.py        the gate: clean / patched / downgraded / needs_review / rejected  (replaces)
ingest_ncci_tables.py   NCCI PTP + MUE -> rules, Spark-scoped            (replaces)
detector_gen.py         rules -> SQL, severity, routing                  (replaces)
rule_normalize.py       planes, provenance, de-dup                       (new)
gap.py                  claims-weighted coverage + gap                   (new, replaces analysis.py)
state_rules.py          reviewed state-rule table + Excel review loop    (new)
notebooks/
  migrate_state_rules   run ONCE  - Python-authored Ch10/Ch19 rules -> main.sedo.state_rules
  build_kb              the build - replaces run_pipeline_chap10_chap19
  run_detectors         executes detectors -> policy_findings + telemetry
  demo_kb               client walkthrough; reads published tables only
tests/test_offline.py   12 checks against the real modules, no Spark:  python -m pytest tests -q
```

## Run order

1. Copy the eight `.py` files into `/Workspace/Users/.../policy_intelligence/`, overwriting.
2. Import the four notebooks (File → Import → source file).
3. `migrate_state_rules`  (once)
4. `build_kb`
5. Read the **routing check** cell. If `dme` is unrouted, take the DME value from the
   `FORM_TYP` distribution printed above it, set `SERVICE_FORM_MAP = {"dme": ["<value>"]}`
   in `config.py`, rebuild.
6. `run_detectors`
7. `demo_kb`

## What changed from the PoC, and why

| PoC | MVP | Why |
|---|---|---|
| `POC_CODES` hand list | scope = distinct codes in the paid-claims table | KB bounded to what can actually fire; gap table means something |
| PTP kept if *either* code in scope | both codes in scope, filtered in Spark | a pair with a never-paid code can't fire |
| all PTP quarters loaded | latest PTP quarter per service | files are cumulative snapshots; several quarters duplicate every pair |
| MUE code vanishing from a quarter stays live forever | run closed on absence | removed MUEs no longer fire |
| `plane='enforcement'` + `edit_failure` second rule per indicator-0 pair | gone; detector emits `finding_severity='hard_denial'` | a claims outcome is not a policy authority (your settled principle) |
| bypass set of 6 modifiers, applied even to indicator 0 | full NCCI set (~30), only for indicator 1 | false positives on LT/RT/F1 bypasses; false negatives on hard denials |
| global `EFF` date floor | per-rule effective/deletion window, deletion inclusive | deleted edits stop flagging later dates of service |
| `rule_id` = hash(codes, type, source) → 9 Ch10 rules shared one id | content-derived id in `schema.stable_rule_id` | `policy_rules` is a keyed table again |
| gate forced/allowed by plane | gate reasons are explicit: `predicate_not_in_grammar`, `missing_claim_field`, self-pair, authorable-but-uncompiled | every lead says why |
| chapters = Python modules | `main.sedo.state_rules` with `review_status` | add a chapter = add reviewed rows |
| checkable-only rules written to `policy_rules` | every vetted rule written, leads included | the coverage findings are the OIG product |
| `risk = "STATE SILENT -- exploit surface"` + rule-id lists | `policy_coverage` + `policy_gap` weighted by lines / paid $ | no rule + $8M paid is a finding; no rule + $0 is noise |
| detectors never executed | `run_detectors` → `policy_findings`, `policy_detector_runs` | "has this rule ever fired?" answerable with data |
| 8× MAI warnings | one line | Medicaid MUE files carry no MAI |

New compiled predicates: `code_not_covered`, `max_units_per_period`
(month / year / benefit_year Oct–Sep), `max_units_per_dos`. The migration
upgrades five Chapter 10 leads (00938, 99070, 11975, 11977, CHW 24/month) into
detectors, exactly as their own notes asked.

## Tables written

| table | one row per |
|---|---|
| `policy_rules` | vetted rule, compiled or lead (`machine_checkable`, `not_checkable_reason`, `validation_status`, `gate_notes`) |
| `policy_detectors` | rule: SQL + `finding_severity` + `routing`, or `reason` there is no SQL |
| `policy_coverage` | paid procedure code: `coverage`, rule counts, lines, units, paid $ |
| `policy_gap` | paid code with no state rule, ranked by exposure |
| `policy_build_log` | build: scope and counts (append) |
| `policy_detector_runs` | detector × run: hits, seconds, error (append) |
| `policy_findings` | flagged line / member-day with `rule_id` (append) |
| `policy_review_log` | review decision (append, via `state_rules.ingest_review_decisions`) |

## Still needs your environment to confirm

* `PMT_AMT` is the paid-amount column (config; `gap.py` auto-detects if not).
* The DME form value (run order step 5) and whether DME is adjudicated on this FFS table at all.
* Ch 14 (Transportation) and Ch 22 (Nursing Facility) are the recommended next
  chapters; they enter through `state_rules`, not through code.

# Databricks notebook source
# MAGIC %md
# MAGIC # Extract state rules from AHCCCS chapter PDFs
# MAGIC
# MAGIC PDF → sections → proposals (model) → **deterministic grounding** → drafts in `main.sedo.state_rules`.
# MAGIC
# MAGIC | stage | module | writes |
# MAGIC |---|---|---|
# MAGIC | 0 acquire + hash | `policy_docs.acquire` | `policy_document_inventory` |
# MAGIC | 1-2 parse + segment | `policy_docs.parse_pdf / segment` | `policy_sections` |
# MAGIC | 3 pre-filter | `policy_docs.rule_signal` | (flag on sections) |
# MAGIC | 4 propose | `extract_rules.propose` (STUB / DBFM / HFLOCAL) | `policy_extraction_cache` |
# MAGIC | 5 ground | `extract_rules.ground` | `policy_extraction_dropped` (what was refused, and why) |
# MAGIC | 6 draft → review | `state_rules.merge_drafts` | `state_rules` (review_status = draft) |
# MAGIC | 7 change tracking | `policy_changes.diff_chapter` | `policy_change_log` |
# MAGIC
# MAGIC The model only ever **proposes**. A proposal is kept only if its verbatim quote and every code are found in
# MAGIC the section. Drafts never compile: `build_kb` loads `approved` / `legacy_hand_authored` rows only.
# MAGIC Re-running is safe: approved rows are never downgraded, rejected rows are never resurrected.

# MAGIC
# MAGIC **This is the single-cluster path: it needs compute that has BOTH a GPU and Unity Catalog access.**
# MAGIC If your GPU compute cannot write Delta (Serverless GPU usually cannot), use the three-notebook split
# MAGIC instead: `stage_inputs` (CPU) → `extract_chapters_gpu` (GPU, files only) → `persist_extraction` (CPU).
# MAGIC
# MAGIC **Compute: Serverless GPU** (same as `llm_summarization_notebook`). The model half is `openai/gpt-oss-120b`
# MAGIC loaded on the cluster with `transformers` and read through `openai-harmony` - the exact loop that works in
# MAGIC the summarization notebook. Expect **~1 hour for the six MVP chapters** (one generate per candidate
# MAGIC section); the model-output cache is flushed to Delta every `config.CACHE_FLUSH_EVERY` calls, so a crash
# MAGIC costs at most that many calls and a re-run resumes from cache.

# COMMAND ----------

# MAGIC %pip install -q pymupdf pypdf requests openpyxl torch accelerate sentencepiece openai-harmony

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import os, sys, json, importlib, time
import pandas as pd
# ---- find the package: this notebook's folder or its parent must contain config.py.
# No hard-coded workspace path; set PKB_PACKAGE_DIR if you keep the notebooks elsewhere.
_here = os.getcwd()
for _cand in (os.environ.get("PKB_PACKAGE_DIR"), _here, os.path.dirname(_here)):
    if _cand and os.path.exists(os.path.join(_cand, "config.py")):
        sys.path.insert(0, _cand); break
else:
    raise ImportError(f"config.py not found in {_here} or its parent; run this notebook from inside the pie_mvp folder "
                      f"or set PKB_PACKAGE_DIR")

import config, schema, kb_io, policy_docs, extract_rules, state_rules, policy_changes
for m in (config, schema, kb_io, policy_docs, extract_rules, state_rules, policy_changes):
    importlib.reload(m)

# ---- run settings -----------------------------------------------------------
CHAPTERS_TO_RUN = config.MVP_CHAPTERS            # ["4","10","13","14","19","22"]; use list(config.CHAPTERS) for all
LLM_BACKEND     = "HYBRID"                       # "HYBRID" (patterns + model; use this for real runs) | "HFLOCAL" | "DBFM" | "STUB" (patterns only: tests/smoke, NOT complete)
REASONING       = config.LLM_REASONING           # gpt-oss reasoning effort for extraction: "low" (default) | "medium" | "high"
DOWNLOAD        = True                           # False = read PDFs already uploaded to PDF_DIR
ONLY_CANDIDATES = True                           # skip sections with no rule signal (saves model calls)
# A dead model must STOP the run, not quietly turn it into a regex run (2026-09-20: every model
# call failed with KeyError, the run "succeeded" in 16 s with 7 pattern-only drafts). Set True only
# for a deliberate patterns-only smoke run; the output is then labelled degraded and never cached.
ALLOW_PATTERNS_ONLY = False

KB   = f"{config.OUTPUT_CATALOG}.{config.OUTPUT_SCHEMA}"
STATE_RULES_TBL = config.STATE_RULES_TABLE
T_INV, T_SEC, T_LOG = f"{KB}.policy_document_inventory", f"{KB}.policy_sections", f"{KB}.policy_extraction_log"
T_DROP, T_CHG, T_CACHE = f"{KB}.policy_extraction_dropped", f"{KB}.policy_change_log", f"{KB}.policy_extraction_cache"

# ---- where PDFs live: the staged Workspace folder next to the package (config.PDF_DIR).
# This notebook no longer creates or touches a Unity Catalog Volume. If you want one, set
# PKB_PDF_DIR to an existing Volume path before importing config.
PDF_DIR = config.PDF_DIR
os.makedirs(PDF_DIR, exist_ok=True)
print(f"PDFs: {PDF_DIR}")

# COMMAND ----------

# MAGIC %md ## Load model — once per session, exactly as in `llm_summarization_notebook`

# COMMAND ----------

USES_LOCAL_MODEL = LLM_BACKEND == "HFLOCAL" or (LLM_BACKEND == "HYBRID" and config.HYBRID_MODEL_BACKEND == "HFLOCAL")
if USES_LOCAL_MODEL:
    # login(hf_token) ; AutoTokenizer/AutoModelForCausalLM.from_pretrained(torch_dtype="auto",
    # device_map="auto", trust_remote_code=True) ; load_harmony_encoding(HARMONY_GPT_OSS).
    # Cached in-process: re-running this cell does not reload 120B parameters.
    rt = extract_rules.load_hf_runtime(config.HF_MODEL, config.HF_TOKEN_PATH)
    config.LLM_REASONING = REASONING
else:
    print(f"backend {LLM_BACKEND}: no local model to load")

# COMMAND ----------

run_id = pd.Timestamp.now().strftime("%Y%m%dT%H%M%S")
backend = extract_rules.make_backend(LLM_BACKEND, allow_patterns_only=ALLOW_PATTERNS_ONLY)
print(f"run {run_id} | backend {backend.name}:{backend.model} | prompt {extract_rules.PROMPT_VERSION} | chapters {CHAPTERS_TO_RUN}")

# COMMAND ----------

# MAGIC %md ## Model health check — one real call before any chapter is touched

# COMMAND ----------

t0 = time.time()
ok, detail = extract_rules.healthcheck(backend)
print(f"{'OK ' if ok else 'FAIL'} ({time.time() - t0:.1f}s) {detail}")
if not ok and not ALLOW_PATTERNS_ONLY:
    raise RuntimeError("Model backend is not reachable - see the routes and errors above. Fix the endpoint / auth, "
                       "or set ALLOW_PATTERNS_ONLY=True for a deliberate regex-only smoke run "
                       "(NOT a complete extraction).")
if LLM_BACKEND == "STUB" or (not ok and ALLOW_PATTERNS_ONLY):
    print("WARNING: this run is PATTERNS ONLY. The deterministic extractor matches a handful of Ch10/19 phrasings; "
          "zero drafts on other chapters means nothing about those chapters.")

# COMMAND ----------

# MAGIC %md ## Load prior state (cache, inventory, existing rules)

# COMMAND ----------

# kb_io.table_or_none returns None ONLY for a missing table. A cluster that cannot READ
# (CLOUD_ACCESS_DENIED / 403) raises here - it must never look like an empty table, because
# the merge below would then overwrite state_rules with nothing (2026-09-21 near-miss).
_table_or_none = lambda fqn: kb_io.table_or_none(spark, fqn)

cache_df = _table_or_none(T_CACHE)
cache = {} if cache_df is None else {r["cache_key"]: json.loads(r["proposals"]) for r in cache_df.to_dict("records")}
prev_inventory = _table_or_none(T_INV)
existing_rules = _table_or_none(STATE_RULES_TBL)
print(f"cache entries: {len(cache):,} | prior inventory rows: {0 if prev_inventory is None else len(prev_inventory)} "
      f"| existing state_rules: {0 if existing_rules is None else len(existing_rules)}")

# COMMAND ----------

# MAGIC %md ## Run — one chapter at a time, failures isolated

# COMMAND ----------

def flush_cache(cache_dict):
    """Persist the model-output cache (SB's FLUSH_EVERY pattern). Overwrite is fine: the dict is the
    full state and a re-run rebuilds drafts from it without a single model call."""
    kb_io.write_table(spark, [{"cache_key": k, "proposals": json.dumps(v)} for k, v in cache_dict.items()],
                      T_CACHE, mode="overwrite")

inventory_rows, section_rows, log_rows, dropped_rows, change_rows, all_drafts = [], [], [], [], [], []
for ch in CHAPTERS_TO_RUN:
    t0 = time.time()
    title = config.CHAPTERS[ch][0]
    print(f"\n=== Chapter {ch}: {title}")
    try:
        inv = policy_docs.acquire(ch, PDF_DIR, download=DOWNLOAD)
        pages = policy_docs.parse_pdf(inv["path"])
        secs, meta = policy_docs.segment(pages, ch, inv["doc_hash"])
        inv["revision_date"] = meta["revision_date"]
        inventory_rows.append(policy_docs.inventory_record(inv, meta))
        for s in secs:
            section_rows.append({k: (json.dumps(v) if isinstance(v, list) else v) for k, v in s.items()} | {"run_id": run_id})
        print(f"  {meta['pages']} pages, rev {meta['revision_date']}, {meta['sections']} sections, "
              f"{meta['candidate_sections']} candidates ({inv['source']})")

        drafts, dropped, stats = extract_rules.extract_chapter(
            secs, inv, backend, run_id, ONLY_CANDIDATES, cache,
            checkpoint_fn=flush_cache, checkpoint_every=config.CACHE_FLUSH_EVERY)
        flush_cache(cache)                                       # and once per chapter regardless
        changes = policy_changes.diff_chapter(existing_rules, drafts, inv, prev_inventory, run_id)
        all_drafts += drafts; dropped_rows += dropped; change_rows += changes
        stats["seconds"] = round(time.time() - t0, 1); stats["error"] = None
        log_rows.append(stats)
        print(f"  proposals {stats['proposals']} -> grounded {stats['grounded']} (dropped {stats['dropped']}, "
              f"cache hits {stats['sections_from_cache']}) | drafts {stats['drafts_unique']} | "
              f"changes {policy_changes.summarize(changes)}")
        print(f"  {stats['seconds']}s | model calls so far {stats['model_calls']} (retries {stats.get('model_retries', 0)}, "
              f"failures {stats['model_failures']})")
        if stats["model_failed"] or stats["degraded_patterns_only"] or stats["unparseable"]:
            print(f"  MODEL: {stats['model_failed']} section(s) failed, {stats['degraded_patterns_only']} degraded to "
                  f"patterns, {stats['unparseable']} unparseable | last error: {str(stats.get('last_model_error'))[:200]}")
    except Exception as e:
        log_rows.append({"chapter": ch, "run_id": run_id, "backend": f"{backend.name}:{backend.model}",
                         "prompt_version": extract_rules.PROMPT_VERSION, "seconds": round(time.time() - t0, 1),
                         "error": f"{type(e).__name__}: {str(e)[:400]}"})
        print(f"  FAILED: {type(e).__name__}: {e}")

mh = extract_rules.model_health(backend)
print(f"\n{len(all_drafts)} drafts across {len(CHAPTERS_TO_RUN)} chapters; {len(dropped_rows)} proposals refused by grounding")
print(f"model {mh['model_backend']} via {mh['model_route']}: {mh['model_calls']} calls, {mh['model_failures']} failures")
if mh["model_calls"] and mh["model_failures"] == mh["model_calls"]:
    print("*** EVERY MODEL CALL FAILED. These drafts are regex-only and are NOT an extraction. "
          f"Last error: {mh['last_model_error']} ***")
elif mh["model_failures"]:
    print(f"*** {mh['model_failures']}/{mh['model_calls']} model calls failed; those sections were NOT cached and "
          f"will retry next run. Last error: {mh['last_model_error']} ***")
by_src = {}
for d in all_drafts:
    src = "model" if "Found by: model" in (d.get("notes") or "") else ("pattern+model" if "pattern+model" in (d.get("notes") or "") else "pattern")
    by_src[src] = by_src.get(src, 0) + 1
print(f"drafts by source: {by_src}   (a healthy HYBRID run has model-found drafts; pattern-only means the model contributed nothing)")

# COMMAND ----------

# MAGIC %md ## Merge drafts into `state_rules` and persist everything

# COMMAND ----------

import extract_io
merged, counts = state_rules.merge_drafts(existing_rules, all_drafts)
print("merge:", counts)
extract_io.guard_state_rules_merge(existing_rules, merged, STATE_RULES_TBL)   # never shrink the reviewed set
state_rules.write_state_rules(spark, STATE_RULES_TBL, merged.to_dict("records"), mode="overwrite")

kb_io.write_table(spark, inventory_rows, T_INV, mode="append")
kb_io.write_table(spark, section_rows, T_SEC, mode="append")
kb_io.write_table(spark, log_rows, T_LOG, mode="append")
kb_io.write_table(spark, dropped_rows, T_DROP, mode="append")
kb_io.write_table(spark, change_rows, T_CHG, mode="append")
kb_io.write_table(spark, [{"cache_key": k, "proposals": json.dumps(v)} for k, v in cache.items()], T_CACHE, mode="overwrite")

# COMMAND ----------

# MAGIC %md ## What a reviewer sees next

# COMMAND ----------

display(spark.sql(f"""
  SELECT chapter, review_status, COUNT(*) AS rules, ROUND(AVG(extraction_confidence),2) AS avg_conf,
         SUM(CASE WHEN ambiguity_flag THEN 1 ELSE 0 END) AS ambiguous
  FROM {STATE_RULES_TBL} GROUP BY chapter, review_status ORDER BY chapter, review_status"""))

# COMMAND ----------

# MAGIC %md ## Export the review workbook (drafts, lowest confidence first)

# COMMAND ----------

wb = os.path.join(PDF_DIR, f"rule_review_{run_id}.xlsx")
state_rules.export_review_workbook(spark, STATE_RULES_TBL, wb, statuses=("draft", "needs_review"))
print(f"STEP 1 done. Next: a reviewer fills decision/reviewer in {wb}, then run notebook "
      "review_decisions (step 2), then build_kb (step 3) to publish approved rules to the KB.")

# Databricks notebook source
# MAGIC %md
# MAGIC # Step 1b — Extract state rules with the model (Serverless GPU, no Spark)
# MAGIC
# MAGIC Same loop as `llm_summarization_notebook`: `gpt-oss-120b` loaded with `transformers`, read through
# MAGIC `openai-harmony`. **No Delta, no Volumes, no Spark** in this notebook — it reads what `stage_inputs`
# MAGIC put in Workspace files and writes a run folder there:
# MAGIC
# MAGIC ```
# MAGIC RUN_ROOT/<run_id>/  drafts.jsonl · dropped.jsonl · sections.jsonl · inventory.jsonl · log.jsonl ·
# MAGIC                     changes.jsonl · cache.json · manifest.json
# MAGIC ```
# MAGIC
# MAGIC `cache.json` is flushed every `config.CACHE_FLUSH_EVERY` model calls and every file is rewritten after
# MAGIC each chapter, so a crash keeps the completed chapters. Then run `persist_extraction` on a UC-enabled CPU
# MAGIC cluster to load the run folder into tables. Expect **~1 hour for the six MVP chapters**.

# COMMAND ----------

# MAGIC %pip install -q pymupdf pypdf torch accelerate sentencepiece openai-harmony

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

import config, schema, policy_docs, extract_rules, extract_io, policy_changes, state_rules
for m in (config, schema, policy_docs, extract_rules, extract_io, policy_changes, state_rules):
    importlib.reload(m)

# ---- run settings -----------------------------------------------------------
CHAPTERS_TO_RUN = config.MVP_CHAPTERS            # ["4","10","13","14","19","22"]
LLM_BACKEND     = "HYBRID"                       # "HYBRID" (patterns + model) | "HFLOCAL" | "STUB" (patterns only, NOT complete)
REASONING       = config.LLM_REASONING           # "low" (default) | "medium" | "high"
ONLY_CANDIDATES = True                           # skip sections with no rule signal
ALLOW_PATTERNS_ONLY = False                      # a dead model STOPS the run unless this is True
ALLOW_DOWNLOAD  = False                          # True: try to download a PDF missing from STAGE_PDF_DIR

STAGE_DIR, PDF_DIR, RUN_ROOT = config.STAGE_DIR, config.STAGE_PDF_DIR, config.RUN_ROOT
run_id = pd.Timestamp.now().strftime("%Y%m%dT%H%M%S")
RUN_DIR = extract_io.new_run_dir(RUN_ROOT, run_id)
print(f"run {run_id} -> {RUN_DIR}")

# COMMAND ----------

# MAGIC %md ## Inputs from stage_inputs (files only)

# COMMAND ----------

missing = [ch for ch in CHAPTERS_TO_RUN
           if not os.path.exists(os.path.join(PDF_DIR, os.path.basename(config.CHAPTERS[ch][1])))]
if missing and not ALLOW_DOWNLOAD:
    raise RuntimeError(f"PDFs missing from {PDF_DIR} for chapters {missing}. Run stage_inputs on a UC-enabled "
                       f"CPU cluster first (or set ALLOW_DOWNLOAD=True if this compute has internet egress).")

cache, existing_rules, prev_inventory, snap_meta = extract_io.load_snapshots(STAGE_DIR)
print(f"snapshots exported {snap_meta.get('at', '?')}: cache entries {len(cache):,} | existing state_rules "
      f"{0 if existing_rules is None else len(existing_rules)} | prior inventory rows "
      f"{0 if prev_inventory is None else len(prev_inventory)}")
if not snap_meta:
    print("WARNING: no snapshots found - drafts will be produced without cache hits or change tracking; "
          "persist_extraction will still merge against the live table.")

# COMMAND ----------

# MAGIC %md ## Load model — once per session, exactly as in `llm_summarization_notebook`

# COMMAND ----------

USES_LOCAL_MODEL = LLM_BACKEND == "HFLOCAL" or (LLM_BACKEND == "HYBRID" and config.HYBRID_MODEL_BACKEND == "HFLOCAL")
if USES_LOCAL_MODEL:
    rt = extract_rules.load_hf_runtime(config.HF_MODEL, config.HF_TOKEN_PATH)
    config.LLM_REASONING = REASONING
backend = extract_rules.make_backend(LLM_BACKEND, allow_patterns_only=ALLOW_PATTERNS_ONLY)
print(f"backend {backend.name}:{backend.model} | prompt {extract_rules.PROMPT_VERSION} | reasoning {REASONING}")

# COMMAND ----------

# MAGIC %md ## Model health check — one real call before any chapter is touched

# COMMAND ----------

t0 = time.time()
ok, detail = extract_rules.healthcheck(backend)
print(f"{'OK ' if ok else 'FAIL'} ({time.time() - t0:.1f}s) {detail}")
if not ok and not ALLOW_PATTERNS_ONLY:
    raise RuntimeError("Model backend is not working - see above. Fix it, or set ALLOW_PATTERNS_ONLY=True for a "
                       "deliberate regex-only smoke run (NOT a complete extraction).")
if LLM_BACKEND == "STUB" or (not ok and ALLOW_PATTERNS_ONLY):
    print("WARNING: this run is PATTERNS ONLY. Zero drafts on a chapter means nothing about that chapter.")

# COMMAND ----------

# MAGIC %md ## Run — one chapter at a time; every file rewritten after each chapter

# COMMAND ----------

inventory_rows, section_rows, log_rows, dropped_rows, change_rows, all_drafts = [], [], [], [], [], []

def manifest(finished: bool) -> dict:
    mh = extract_rules.model_health(backend)
    return {"run_id": run_id, "backend": f"{backend.name}:{backend.model}", "prompt_version": extract_rules.PROMPT_VERSION,
            "reasoning": REASONING, "chapters": CHAPTERS_TO_RUN, "stage_dir": STAGE_DIR,
            "snapshots_exported_at": snap_meta.get("at"), "finished": finished, **mh}

def flush_cache(cache_dict):
    extract_io.write_cache(RUN_DIR, cache_dict)

for ch in CHAPTERS_TO_RUN:
    t0 = time.time()
    title = config.CHAPTERS[ch][0]
    print(f"\n=== Chapter {ch}: {title}")
    try:
        inv = policy_docs.acquire(ch, PDF_DIR, download=ALLOW_DOWNLOAD)
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
    # everything so far on disk: a crash in the next chapter loses nothing done
    extract_io.write_run(RUN_DIR, inventory_rows=inventory_rows, section_rows=section_rows, log_rows=log_rows,
                         dropped_rows=dropped_rows, change_rows=change_rows, drafts=all_drafts, cache=cache,
                         manifest=manifest(finished=False))

counts = extract_io.write_run(RUN_DIR, inventory_rows=inventory_rows, section_rows=section_rows, log_rows=log_rows,
                              dropped_rows=dropped_rows, change_rows=change_rows, drafts=all_drafts, cache=cache,
                              manifest=manifest(finished=True))

# COMMAND ----------

# MAGIC %md ## Summary — this is what the persist step and the deck read

# COMMAND ----------

mh = extract_rules.model_health(backend)
print(f"{len(all_drafts)} drafts across {len(CHAPTERS_TO_RUN)} chapters; {len(dropped_rows)} proposals refused by grounding")
print(f"model {mh['model_backend']} via {mh['model_route']}: {mh['model_calls']} calls, {mh['model_failures']} failures, "
      f"{mh.get('model_retries', 0)} retries")
if mh["model_calls"] and mh["model_failures"] == mh["model_calls"]:
    print("*** EVERY MODEL CALL FAILED. These drafts are regex-only and are NOT an extraction. "
          f"Last error: {mh['last_model_error']} ***")
elif mh["model_failures"]:
    print(f"*** {mh['model_failures']}/{mh['model_calls']} model calls failed; those sections were NOT cached and "
          f"will retry next run. Last error: {mh['last_model_error']} ***")
by_src = {}
for d in all_drafts:
    n = d.get("notes") or ""
    src = "pattern+model" if "pattern+model" in n else ("model" if "Found by: model" in n else "pattern")
    by_src[src] = by_src.get(src, 0) + 1
print(f"drafts by source: {by_src}   (a healthy HYBRID run has model-found drafts)")
print(f"\nrun folder: {RUN_DIR}  {counts}")
print("Next: run persist_extraction on a UC-enabled CPU cluster with RUN_ID =", run_id)

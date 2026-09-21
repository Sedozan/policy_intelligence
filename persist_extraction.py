# Databricks notebook source
# MAGIC %md
# MAGIC # Step 1c — Persist a GPU extraction run into Delta (CPU, Unity-Catalog-enabled cluster)
# MAGIC
# MAGIC Reads `RUN_ROOT/<run_id>/` written by `extract_chapters_gpu` and loads it:
# MAGIC
# MAGIC | file | table | mode |
# MAGIC |---|---|---|
# MAGIC | `drafts.jsonl` merged into the live table | `main.sedo.state_rules` | overwrite, **guarded** |
# MAGIC | `inventory.jsonl` / `sections.jsonl` / `log.jsonl` / `dropped.jsonl` / `changes.jsonl` | `policy_kb.*` | append |
# MAGIC | `cache.json` | `policy_kb.policy_extraction_cache` | overwrite |
# MAGIC
# MAGIC Guards: the merge reads the **live** `state_rules` table (not the snapshot); a read failure stops the run
# MAGIC instead of becoming an empty frame; the overwrite is refused if it would drop a single reviewed row.
# MAGIC Then exports the review workbook. Re-running the same run folder is safe: drafts merge by key.

# COMMAND ----------

# MAGIC %pip install -q openpyxl

# COMMAND ----------

import os, sys, json, importlib
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
import config, extract_io, kb_io, state_rules
for m in (config, extract_io, kb_io, state_rules):
    importlib.reload(m)

RUN_ID = None                                     # None = latest folder under config.RUN_ROOT
ALLOW_UNFINISHED = False                          # True: persist a run whose manifest says finished=False
REVIEW_XLSX = f"{config.WORKSPACE_DATA_DIR}/review/policy_rules_review.xlsx"

KB = f"{config.OUTPUT_CATALOG}.{config.OUTPUT_SCHEMA}"
STATE_RULES_TBL = config.STATE_RULES_TABLE
T_INV, T_SEC, T_LOG = f"{KB}.policy_document_inventory", f"{KB}.policy_sections", f"{KB}.policy_extraction_log"
T_DROP, T_CHG, T_CACHE = f"{KB}.policy_extraction_dropped", f"{KB}.policy_change_log", f"{KB}.policy_extraction_cache"

run_dir = os.path.join(config.RUN_ROOT, RUN_ID) if RUN_ID else extract_io.latest_run_dir(config.RUN_ROOT)
assert run_dir and os.path.isdir(run_dir), f"no run folder under {config.RUN_ROOT}"
run = extract_io.read_run(run_dir)
man = run["manifest"]
print(f"run folder: {run_dir}")
print(f"manifest: run {man.get('run_id')} | finished={man.get('finished')} | backend {man.get('model_backend')} | "
      f"model calls {man.get('model_calls')} failures {man.get('model_failures')} | counts {man.get('counts')}")
if not man.get("finished") and not ALLOW_UNFINISHED:
    raise RuntimeError("this run did not finish (manifest.finished=False). Re-run extract_chapters_gpu to completion, "
                       "or set ALLOW_UNFINISHED=True to persist the completed chapters only.")
if man.get("model_calls") and man.get("model_failures") == man.get("model_calls"):
    raise RuntimeError("every model call in this run failed - its drafts are regex-only. Not persisting.")

# COMMAND ----------

# MAGIC %md ## Merge drafts into the LIVE state_rules table (read failure = stop; shrink = stop)

# COMMAND ----------

existing_rules = kb_io.table_or_none(spark, STATE_RULES_TBL)      # raises TableAccessError on 403, never None-for-denied
print(f"live state_rules: {'(table does not exist yet)' if existing_rules is None else f'{len(existing_rules)} rows'}")

merged, counts = state_rules.merge_drafts(existing_rules, run["drafts"])
print("merge:", counts)
extract_io.guard_state_rules_merge(existing_rules, merged, STATE_RULES_TBL)
state_rules.write_state_rules(spark, STATE_RULES_TBL, merged.to_dict("records"), mode="overwrite")

# COMMAND ----------

# MAGIC %md ## Append the run's telemetry and overwrite the cache

# COMMAND ----------

kb_io.write_table(spark, run["inventory"], T_INV, mode="append")
kb_io.write_table(spark, run["sections"], T_SEC, mode="append")
kb_io.write_table(spark, run["log"], T_LOG, mode="append")
kb_io.write_table(spark, run["dropped"], T_DROP, mode="append")
kb_io.write_table(spark, run["changes"], T_CHG, mode="append")
# the run's cache already contains every snapshot entry it was given plus the new calls
kb_io.write_table(spark, [{"cache_key": k, "proposals": json.dumps(v)} for k, v in run["cache"].items()],
                  T_CACHE, mode="overwrite")

# COMMAND ----------

# MAGIC %md ## Export the review workbook (drafts, lowest confidence first)

# COMMAND ----------

os.makedirs(os.path.dirname(REVIEW_XLSX), exist_ok=True)
wb = state_rules.export_review_workbook(spark, STATE_RULES_TBL, REVIEW_XLSX)
print(f"STEP 1 done. Next: a reviewer fills decision/reviewer in {wb}, then run review_decisions (step 2), "
      f"then build_kb (step 3).")

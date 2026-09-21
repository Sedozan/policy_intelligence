# Databricks notebook source
# MAGIC %md
# MAGIC # Step 1a — Stage inputs for the GPU extraction (CPU, Unity-Catalog-enabled cluster)
# MAGIC
# MAGIC Serverless GPU runs the model but should not touch Delta or Volumes. This notebook puts everything the
# MAGIC GPU notebook needs into **Workspace files** it can read:
# MAGIC
# MAGIC | writes | what |
# MAGIC |---|---|
# MAGIC | `STAGE_DIR/policy_docs/*.pdf` | the chapter PDFs (downloaded here, where the cluster has egress) |
# MAGIC | `STAGE_DIR/snapshots/cache.json` | prior model outputs, so unchanged sections cost no model call |
# MAGIC | `STAGE_DIR/snapshots/state_rules.jsonl` | current reviewed rules, so drafts merge and changes diff correctly |
# MAGIC | `STAGE_DIR/snapshots/inventory_prev.jsonl` | prior document hashes, for change tracking |
# MAGIC
# MAGIC Run on the same UC-enabled CPU compute the summarization pipeline's CPU notebook uses. If a table read
# MAGIC fails with `CLOUD_ACCESS_DENIED`, this notebook **stops** — it never exports an empty snapshot in place
# MAGIC of a table it could not read.

# COMMAND ----------

# MAGIC %pip install -q requests

# COMMAND ----------

import os, sys, importlib, shutil
# ---- find the package: this notebook's folder or its parent must contain config.py.
# No hard-coded workspace path; set PKB_PACKAGE_DIR if you keep the notebooks elsewhere.
_here = os.getcwd()
for _cand in (os.environ.get("PKB_PACKAGE_DIR"), _here, os.path.dirname(_here)):
    if _cand and os.path.exists(os.path.join(_cand, "config.py")):
        sys.path.insert(0, _cand); break
else:
    raise ImportError(f"config.py not found in {_here} or its parent; run this notebook from inside the pie_mvp folder "
                      f"or set PKB_PACKAGE_DIR")
import config, policy_docs, extract_io, kb_io
for m in (config, policy_docs, extract_io, kb_io):
    importlib.reload(m)

CHAPTERS_TO_STAGE = config.MVP_CHAPTERS       # or list(config.CHAPTERS) for all 29
STAGE_DIR = config.STAGE_DIR
PDF_DIR   = config.STAGE_PDF_DIR
ALSO_COPY_TO_VOLUME = False                    # True: keep a copy in config.PDF_DIR as well (needs Volume write)

KB = f"{config.OUTPUT_CATALOG}.{config.OUTPUT_SCHEMA}"
T_INV, T_CACHE = f"{KB}.policy_document_inventory", f"{KB}.policy_extraction_cache"
os.makedirs(PDF_DIR, exist_ok=True)
print(f"stage dir: {STAGE_DIR}")

# COMMAND ----------

# MAGIC %md ## PDFs → Workspace files

# COMMAND ----------

staged = []
for ch in CHAPTERS_TO_STAGE:
    try:
        inv = policy_docs.acquire(ch, PDF_DIR, download=True)
        staged.append(inv)
        print(f"  ch {ch:>3}  {inv['bytes']:>9,} bytes  {inv['source']:8s}  {os.path.basename(inv['path'])}")
        if ALSO_COPY_TO_VOLUME:
            os.makedirs(config.PDF_DIR, exist_ok=True)
            shutil.copyfile(inv["path"], os.path.join(config.PDF_DIR, os.path.basename(inv["path"])))
    except Exception as e:
        print(f"  ch {ch:>3}  FAILED: {type(e).__name__}: {str(e)[:200]}")
print(f"{len(staged)}/{len(CHAPTERS_TO_STAGE)} PDFs in {PDF_DIR}")

# COMMAND ----------

# MAGIC %md ## Table snapshots → Workspace files (stops on a read failure)

# COMMAND ----------

summary = extract_io.export_snapshots(spark, STAGE_DIR, T_CACHE, config.STATE_RULES_TABLE, T_INV)
print(summary)
if not summary["state_rules_table_exists"]:
    print("NOTE: state_rules table does not exist yet - run migrate_state_rules first if you expect reviewed rules.")
print(f"\nNext: run extract_chapters_gpu on Serverless GPU. It reads {STAGE_DIR} and writes {config.RUN_ROOT}/<run_id>/.")

# Databricks notebook source
# MAGIC %md
# MAGIC # One-time: migrate hand-authored state rules → `main.sedo.state_rules`
# MAGIC
# MAGIC Moves the Chapter 10 / 19 rules that currently live in Python modules
# MAGIC (`ingest_ahcccs`, `demo_three_beats`, `demo_ch10_beats`, `ch10_full_extract`)
# MAGIC into the reviewed Delta table the build reads from. After this runs, chapters are
# MAGIC **data**: add rows (draft → review → approved), never edit the build notebook.
# MAGIC
# MAGIC Also applies the grammar upgrades the rules' own notes asked for
# MAGIC (`code_not_covered`, `max_units_per_period`) so those leads become detectors.
# MAGIC
# MAGIC Re-running overwrites the table with the same content (idempotent). It will NOT
# MAGIC preserve rows added later by hand — set `MODE = "append"` for that.

# COMMAND ----------

import os, sys, json, importlib
sys.path.insert(0, os.getcwd())
sys.path.append("/Workspace/Users/sedo.senou@azahcccs.gov/Projects/policy_intelligence")

import config, schema, state_rules, rule_normalize
for m in (config, schema, state_rules, rule_normalize):
    importlib.reload(m)

STATE_RULES_TBL = config.STATE_RULES_TABLE       # main.sedo.state_rules
MODE = "overwrite"          # "append" to keep rows already in the table

# COMMAND ----------

# MAGIC %md ## Collect the Python-authored rules

# COMMAND ----------

sources = []
def _try(modname, fn):
    try:
        m = importlib.import_module(modname); importlib.reload(m)
        rs = getattr(m, fn)()
        sources.append((f"{modname}.{fn}", rs))
        print(f"  {modname}.{fn:28s} {len(rs):4d} rules")
    except Exception as e:
        print(f"  {modname}.{fn:28s} SKIPPED ({e})")

_try("ingest_ahcccs",      "sample_rules")
_try("demo_three_beats",   "all_demo_rules")
_try("demo_ch10_beats",    "all_ch10_rules")
_try("ch10_full_extract",  "all_ch10_extended_rules")
raw = [r for _, rs in sources for r in rs]
print(f"\n{len(raw)} rule objects collected (duplicates across modules collapse below)")

# COMMAND ----------

# MAGIC %md ## Grammar upgrades
# MAGIC These rules were authored as `not_checkable` only because the predicate did not
# MAGIC exist yet. The compiler now supports both, so they become detectors. The original
# MAGIC statement and citation are untouched; the note records the upgrade.

# COMMAND ----------

NOT_COVERED = {"00938", "99070", "11975", "11977"}

def upgrade(r):
    p = r.predicate or {}
    if p.get("type") != "not_checkable":
        return r
    codes = [str(c).upper() for c in (r.codes or [])]
    if len(codes) == 1 and codes[0] in NOT_COVERED:
        r.predicate = {"type": "code_not_covered", "code": codes[0]}
        r.machine_checkable, r.not_checkable_reason = True, None
        r.required_claim_fields = ["proc_cd"]
        r.notes = (r.notes or "") + " [grammar upgrade: code_not_covered]"
    elif set(codes) == {"98960", "98961", "98962"} and "per month" in (r.statement or ""):
        r.predicate = {"type": "max_units_per_period", "code_set": codes, "threshold": 24,
                       "period": "month", "service_category": "practitioner"}
        r.machine_checkable, r.not_checkable_reason = True, None
        r.required_claim_fields = ["proc_cd", "units", "member_id", "srvc_bgn_dt"]
        r.notes = (r.notes or "") + " [grammar upgrade: max_units_per_period]"
    return r

raw = [upgrade(r) for r in raw]
print("upgraded:", sum(1 for r in raw if "[grammar upgrade" in (r.notes or "")))

# COMMAND ----------

# MAGIC %md ## Chapter tag, stable ids, de-duplicate, write

# COMMAND ----------

import re
def chapter_of(r):
    m = re.search(r"Chapter\s+(\d+)", r.source_doc or "")
    return m.group(1) if m else "unknown"

rows, seen = [], set()
for r in raw:
    row = state_rules.rule_to_row(r, chapter=chapter_of(r), review_status="legacy_hand_authored",
                                  reviewer=None, review_note="hand-authored during PoC; migrated")
    row["rule_id"] = schema.stable_rule_id(r.to_dict())   # content-derived: no more collisions
    if row["rule_id"] in seen:
        continue
    seen.add(row["rule_id"])
    rows.append(row)

import pandas as pd
summary = pd.DataFrame(rows).groupby(["chapter", "machine_checkable"]).size().reset_index(name="rules")
print(summary.to_string(index=False))
state_rules.write_state_rules(spark, STATE_RULES_TBL, rows, mode=MODE)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Adding a new chapter from here on
# MAGIC 1. Author rules as rows (predicate JSON, citation, page, URL) with `review_status = 'draft'`.
# MAGIC 2. `state_rules.export_review_workbook(spark, STATE_RULES_TBL, "/Volumes/.../review.xlsx")`
# MAGIC 3. Reviewer fills **decision / corrected_rule / reviewer** and saves.
# MAGIC 4. `state_rules.ingest_review_decisions(spark, STATE_RULES_TBL, path, "main.policy_kb.policy_review_log")`
# MAGIC 5. Run `build_kb`. Only `approved` / `legacy_hand_authored` rows enter the KB.

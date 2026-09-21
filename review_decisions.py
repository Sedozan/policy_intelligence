# Databricks notebook source
# MAGIC %md
# MAGIC # Step 2 of 3 — Apply the human review decisions
# MAGIC
# MAGIC The extraction notebook (`extract_chapters`) lands every extracted rule in
# MAGIC `main.sedo.state_rules` with `review_status = draft` and exports a workbook.
# MAGIC **Nothing reaches the KB until a named human decides.** This notebook applies those
# MAGIC decisions.
# MAGIC
# MAGIC | column the reviewer fills | meaning |
# MAGIC |---|---|
# MAGIC | `decision` | `Approve` — rule is correct as written · `Reject` — not a rule / wrong · `Fix` — right idea, wrong details |
# MAGIC | `corrected_rule` | (Fix) corrected plain-English statement |
# MAGIC | `corrected_predicate` | (Fix) corrected predicate **as JSON** — required; the predicate is what compiles |
# MAGIC | `reviewer`, `review_note` | who decided and why |
# MAGIC
# MAGIC Safety rules enforced here: an approval can never be silently downgraded later; a
# MAGIC rejected rule is never resurrected by re-extraction; a **Fix without a valid
# MAGIC corrected_predicate is parked as `needs_review`**, not approved.
# MAGIC Every decision is appended to `policy_review_log` (with `extraction_key`, so it joins
# MAGIC to the exact compiled detector).
# MAGIC
# MAGIC Next: run `build_kb` (step 3) — it loads **only** `approved` / `legacy_hand_authored` rows.

# COMMAND ----------

import os, sys
# ---- find the package: this notebook's folder or its parent must contain config.py.
# No hard-coded workspace path; set PKB_PACKAGE_DIR if you keep the notebooks elsewhere.
_here = os.getcwd()
for _cand in (os.environ.get("PKB_PACKAGE_DIR"), _here, os.path.dirname(_here)):
    if _cand and os.path.exists(os.path.join(_cand, "config.py")):
        sys.path.insert(0, _cand); break
else:
    raise ImportError(f"config.py not found in {_here} or its parent; run this notebook from inside the pie_mvp folder "
                      f"or set PKB_PACKAGE_DIR")
import importlib, config, state_rules
importlib.reload(config); importlib.reload(state_rules)

STATE_RULES_TBL = config.STATE_RULES_TABLE                      # main.sedo.state_rules
LOG_TBL         = f"{config.OUTPUT_CATALOG}.{config.OUTPUT_SCHEMA}.policy_review_log"
WORKBOOK        = ""      # <-- path of the filled-in rule_review_<run_id>.xlsx (on the Volume)

assert WORKBOOK, "set WORKBOOK to the reviewed .xlsx exported by extract_chapters"

# COMMAND ----------

# MAGIC %md ## Before — what is waiting for a decision

# COMMAND ----------

before = spark.table(STATE_RULES_TBL).toPandas()
print(before["review_status"].value_counts().to_string())

# COMMAND ----------

# MAGIC %md ## Apply decisions

# COMMAND ----------

state_rules.ingest_review_decisions(spark, STATE_RULES_TBL, WORKBOOK, LOG_TBL)

# COMMAND ----------

# MAGIC %md ## After — what will load into the KB

# COMMAND ----------

after = spark.table(STATE_RULES_TBL).toPandas()
print(after["review_status"].value_counts().to_string())
loadable = after[after["review_status"].isin(state_rules.LOADABLE_STATUS)]
print(f"\n{len(loadable)} rules will load into the KB on the next build_kb run "
      f"(by chapter: {loadable['chapter'].value_counts().to_dict()})")
parked = after[after["review_status"] == "needs_review"]
if len(parked):
    print(f"\n{len(parked)} parked as needs_review (Fix without a valid predicate) — see review_note:")
    print(parked[["rule_id", "chapter", "review_note"]].head(20).to_string(index=False))

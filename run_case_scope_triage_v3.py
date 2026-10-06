# Databricks notebook source
# MAGIC %md
# MAGIC # CaseScope v3 — provider risk triage run
# MAGIC Runs the rules engine on paid claims, checks the results, and writes case leads.
# MAGIC
# MAGIC **Before the first run:** put `case_scope_triage.py` (the engine, `casescope_triage_spark.py` renamed)
# MAGIC and `case_narratives.py` in the **same folder as this notebook**.
# MAGIC
# MAGIC Run the cells top to bottom (or **Run all**). If the checks cell fails, stop: don't share that run's outputs.

# COMMAND ----------

# 1. Load the scripts. reload() picks up edits to the .py files without restarting the cluster.
import importlib, os, datetime
from pyspark.sql import functions as F
import case_scope_triage as cs
import case_narratives as cn
importlib.reload(cs); importlib.reload(cn)
from case_scope_triage import *
print("rules registered:", len(REG))

# COMMAND ----------

# 2. Settings — the only cell you normally edit.
CFG["table"] = "main.prod_input.all_data_C_A"
CFG["role_reference_path"] = None      # R10: a table with provider_key, expected_role. None = off.
CFG["out_schema"] = "main.siu_fwa"     # only used if SAVE_TABLES = True; must be a schema you can write to
SAVE_TABLES = False
OUT_DIR = "/Workspace/Users/sedo.senou@azahcccs.gov/Projects/provider_risk_triage_engine/outputs"
RUN_LABEL = f"{CFG['table']} — run {datetime.date.today()}"

# COMMAND ----------

# 3. Schema check: fail fast if the table is missing columns the engine needs.
src = spark.table(CFG["table"])
required = ["ADJU_STA", "BILLING_PROVIDER_KEY", "SERVICING_PROVIDER_KEY", "MEMBER_KEY", "COS_DESC",
            "PROC_CD", "QUANTITY_PAID", "PMT_AMT", "UNIQUE_KEY", "Claim_Hdr_ID", "Svc_Begin_Dt",
            "PAY_TO_TAX_ID", "BProv_Name", "BProv_Type_Desc", "SProv_Name", "SProv_Type_Desc"]
optional = {"PROC_CD_DESC": "R1 (billed time over 24h) is skipped without it",
            "line_of_business": "R9 can't exclude MSP claims without it"}
missing = [c for c in required if c not in src.columns]
assert not missing, f"Missing required columns: {missing}"
for c, why in optional.items():
    if c not in src.columns:
        print(f"WARNING: {c} missing — {why}")
print("schema OK")

# COMMAND ----------

# 4. Prepare paid claims (approved lines, 24-month window).
paid = prep(src).cache()
print(f"paid lines in window: {paid.count():,}")
display(paid.agg(F.min("svc_dt").alias("window_start"), F.max("svc_dt").alias("window_end")))

# COMMAND ----------

# 5. Run the rules.
hits = run_rules(paid).cache()
display(hits.groupBy("rule_id", "family", "tier", "entity_type").count().orderBy("rule_id"))

# COMMAND ----------

# 6. Score providers and run the built-in checks. Raises an error if any check fails.
queue = score_providers(hits, paid).cache()
run_checks(queue, hits, paid)
display(queue.groupBy("entity_type", "tier_label").count().orderBy("entity_type", "tier_label"))
display(queue.filter("corroborated"))

# COMMAND ----------

# 7. Write the case leads and the shareable table (corroborated providers only).
q_pdf, h_pdf, n_pdf = cn.from_spark(queue, hits, paid)
leads = cn.build_case_leads(q_pdf, h_pdf, n_pdf, data_label=RUN_LABEL)
table = cn.queue_with_reasons(q_pdf, h_pdf, n_pdf, data_label=RUN_LABEL)
cn.check_leads(leads, q_pdf, table)
print(f"{len(leads)} leads")
print(leads.lead.iloc[0] if len(leads) else "No corroborated leads in this run.")
display(table)

# COMMAND ----------

# 8. Save. Files go to OUT_DIR; Delta tables only if SAVE_TABLES = True.
os.makedirs(OUT_DIR, exist_ok=True)
stamp = datetime.date.today().isoformat()
table.to_csv(f"{OUT_DIR}/corroborated_providers_{stamp}.csv", index=False)
with open(f"{OUT_DIR}/case_leads_{stamp}.txt", "w") as f:
    f.write("\n\n\n".join(leads.lead))
print("files saved to", OUT_DIR)
if SAVE_TABLES:
    s = CFG["out_schema"]
    queue.write.mode("overwrite").saveAsTable(f"{s}.casescope_entity_queue")
    hits.drop("line_keys").write.mode("overwrite").saveAsTable(f"{s}.casescope_rule_hits")
    flagged_lines(hits, paid).write.mode("overwrite").saveAsTable(f"{s}.casescope_flagged_lines")
    print("tables saved to", s)

# COMMAND ----------

# 9. Optional: claim-level evidence for one provider (every flagged line, with the rule behind it).
PROVIDER = int(q_pdf.entity_id.iloc[0]) if len(q_pdf) else None
if PROVIDER is not None:
    display(flagged_lines(hits, paid).filter(F.col("entity_id") == PROVIDER)
            .join(paid.select("UNIQUE_KEY", "PROC_CD", "COS_DESC", "QUANTITY_PAID", "Claim_Hdr_ID"), "UNIQUE_KEY")
            .orderBy("svc_dt", "rule_id"))

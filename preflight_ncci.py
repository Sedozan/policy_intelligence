# Databricks notebook source
# MAGIC %md
# MAGIC # NCCI pre-flight — answer the four questions the ingest currently ASSUMES
# MAGIC
# MAGIC Run once before the first `build_kb` after Phase 1.5, and again whenever a new NCCI quarter lands.
# MAGIC Read-only: no tables are written. Prints a go / no-go with the evidence.
# MAGIC
# MAGIC | # | question | why it matters |
# MAGIC |---|---|---|
# MAGIC | 1 | Are the MUE quarters **contiguous** per service? | `mue_history` now refuses a gap (it used to carry the old value across it silently). |
# MAGIC | 2 | Do **older PTP quarters hold pairs absent from the newest**? | The ingest loads only the latest PTP file per service, assuming it is cumulative. If CMS drops old deleted edits, historical claims lose them. |
# MAGIC | 3 | Do the loaded quarters **cover the claims date range**? | A MUE's effective date is the first quarter *you loaded*; claims before it are unchecked. |
# MAGIC | 4 | How often do **editions disagree** on the same code / pair? | Decides the collapse policy (open decision #3) and whether F03 interval-split is worth building. |
# MAGIC | 5 | Any **negative-amount paid lines**? | Closes F09 (paid-only table) fully. |

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

import importlib
import config, ingest_ncci_tables as ing
for m in (config, ing):
    importlib.reload(m)

CAT, SCH = config.CATALOG, config.SCHEMA
CLAIMS = config.CLAIMS_TABLE
DOS = config.CLAIM_COLS["dos"]
PAID = config.CLAIM_COLS["paid_amt"]
fc = ing._find_col

metas = ing.discover_tables(spark, CAT, SCH)
assert metas, f"no medicaid_ncci_edit_* tables in {CAT}.{SCH}"
by = {}
for m in metas:
    by.setdefault((m["service"], m["edit_type"]), []).append(m)
for k in by:
    by[k].sort(key=ing.quarter_sort_key)

verdict = {"go": True, "blockers": [], "warnings": [], "info": []}
def block(msg): verdict["go"] = False; verdict["blockers"].append(msg); print("  BLOCKER ", msg)
def warn(msg):  verdict["warnings"].append(msg); print("  WARN    ", msg)
def info(msg):  verdict["info"].append(msg); print("  info    ", msg)

print("loaded tables:")
for (svc, et), ms in sorted(by.items()):
    print(f"  {svc:12s} {et}  {[m['label'] for m in ms]}")

# COMMAND ----------

# MAGIC %md ## 1. Quarter contiguity (MUE is the one that matters; PTP is informational)

# COMMAND ----------

print("1. quarter contiguity")
for (svc, et), ms in sorted(by.items()):
    gaps = ing.quarter_gaps(ms)
    if not gaps:
        info(f"{svc} {et}: contiguous {ms[0]['label']}..{ms[-1]['label']}")
    elif et == "mue":
        block(f"{svc} mue: missing quarter(s) starting {gaps} - load them, or set config.NCCI_ALLOW_MUE_GAPS=True "
              f"(runs spanning a gap are then flagged ambiguous)")
    else:
        info(f"{svc} ptp: quarters {gaps} not loaded (harmless if #2 below is clean)")

# COMMAND ----------

# MAGIC %md ## 2. PTP: does the newest file still contain every pair the oldest one had?

# COMMAND ----------

print("2. PTP cumulative-file assumption")
ptp_dropout = {}
for (svc, et), ms in sorted(by.items()):
    if et != "ptp" or len(ms) < 2:
        if et == "ptp":
            info(f"{svc} ptp: only one quarter loaded - assumption untestable; load an older quarter to check")
        continue
    old, new = ms[0], ms[-1]
    def pairs(m):
        cols = spark.table(m["fqn"]).columns
        c1, c2 = fc(cols, "column1", "col1", "comprehensive code"), fc(cols, "column2", "col2", "component code")
        return spark.sql(f"SELECT DISTINCT UPPER(TRIM(`{c1}`)) AS a, UPPER(TRIM(`{c2}`)) AS b FROM {m['fqn']}")
    missing = pairs(old).subtract(pairs(new))
    n_old, n_new, n_miss = pairs(old).count(), pairs(new).count(), missing.count()
    ptp_dropout[svc] = n_miss
    if n_miss == 0:
        info(f"{svc} ptp: {old['label']} ({n_old:,} pairs) is fully contained in {new['label']} ({n_new:,}) - latest-only is safe")
    else:
        sample = [f"{r.a}/{r.b}" for r in missing.limit(5).collect()]
        warn(f"{svc} ptp: {n_miss:,} pairs in {old['label']} are ABSENT from {new['label']} (e.g. {sample}). "
             f"Historical claims lose these edits under latest-only loading; ingest should union all quarters by (pair, EffDt).")

# COMMAND ----------

# MAGIC %md ## 3. Date coverage: loaded quarters vs the claims table

# COMMAND ----------

print("3. date coverage")
rng = spark.sql(f"SELECT MIN(CAST({DOS} AS DATE)) AS lo, MAX(CAST({DOS} AS DATE)) AS hi, COUNT(*) AS n FROM {CLAIMS}").collect()[0]
info(f"claims: {rng.n:,} lines, DOS {rng.lo} .. {rng.hi}")
for (svc, et), ms in sorted(by.items()):
    first = ms[0]["starts"]
    if et == "mue":
        unchecked = spark.sql(f"SELECT COUNT(*) AS n FROM {CLAIMS} WHERE CAST({DOS} AS DATE) < DATE '{first}'").collect()[0].n
        if unchecked:
            warn(f"{svc} mue: earliest loaded quarter starts {first}; {unchecked:,} claim lines are dated before it "
                 f"and get NO MUE check (label this in the run, or load earlier quarters)")
        else:
            info(f"{svc} mue: earliest quarter {first} covers the whole claims range")
    else:
        m = ms[-1]; cols = spark.table(m["fqn"]).columns
        c_eff = fc(cols, "effective date", "effdt")
        if c_eff:
            e = spark.sql(f"SELECT MIN(`{c_eff}`) AS lo FROM {m['fqn']}").collect()[0].lo
            info(f"{svc} ptp: earliest EffDt in {m['label']} = {e} (per-pair dates; coverage is by pair, not by file)")

# COMMAND ----------

# MAGIC %md ## 4. Edition disagreement (decides the collapse policy and F03)

# COMMAND ----------

print("4. edition disagreement, latest quarter per service")
# MUE: same code, different values across editions
mue_latest = []
for (svc, et), ms in by.items():
    if et != "mue":
        continue
    m = ms[-1]; cols = spark.table(m["fqn"]).columns
    c_code = fc(cols, "hcpcs/cpt code", "hcpcs cpt code", "procedure code", "code")
    c_val = fc(cols, "mue value", "mue values")
    mue_latest.append(f"SELECT '{svc}' AS svc, UPPER(TRIM(`{c_code}`)) AS code, CAST(`{c_val}` AS INT) AS val FROM {m['fqn']}")
if len(mue_latest) > 1:
    u = " UNION ALL ".join(mue_latest)
    dis = spark.sql(f"""
        WITH u AS ({u}),
        multi AS (SELECT code, COUNT(DISTINCT svc) AS n_ed, COUNT(DISTINCT val) AS n_val,
                         MIN(val) AS lo, MAX(val) AS hi FROM u GROUP BY code HAVING COUNT(DISTINCT svc) > 1),
        paid AS (SELECT UPPER(TRIM({config.CLAIM_COLS['code']})) AS code, COUNT(*) AS lines, SUM({PAID}) AS dollars
                 FROM {CLAIMS} GROUP BY 1)
        SELECT COUNT(*) AS codes_in_multiple_editions,
               SUM(CASE WHEN n_val > 1 THEN 1 ELSE 0 END) AS codes_disagreeing,
               SUM(CASE WHEN n_val > 1 THEN COALESCE(p.lines, 0) ELSE 0 END) AS disagreeing_claim_lines,
               SUM(CASE WHEN n_val > 1 THEN COALESCE(p.dollars, 0) ELSE 0 END) AS disagreeing_paid_dollars
        FROM multi LEFT JOIN paid p USING (code)""").collect()[0]
    info(f"MUE: {dis.codes_in_multiple_editions:,} codes in >1 edition; {dis.codes_disagreeing:,} disagree on the limit; "
         f"those codes carry {dis.disagreeing_claim_lines or 0:,} paid lines / ${dis.disagreeing_paid_dollars or 0:,.0f}")
    top = spark.sql(f"""
        WITH u AS ({u})
        SELECT code, COLLECT_LIST(CONCAT(svc, '=', val)) AS by_edition FROM u GROUP BY code
        HAVING COUNT(DISTINCT val) > 1 ORDER BY code LIMIT 10""").collect()
    for r in top:
        print(f"      {r.code}: {r.by_edition}")
    if (dis.codes_disagreeing or 0) == 0:
        info("MUE editions never disagree -> collapse policy and F03 are moot for MUE")
    else:
        warn("MUE editions disagree: most-permissive collapse MISSES violations on those codes; "
             "the same codes are where F03 (unequal windows) can bite. Decide policy with these numbers.")
else:
    info("only one MUE edition loaded - no cross-edition disagreement possible")

# PTP: same pair, different modifier indicator across editions
ptp_latest = []
for (svc, et), ms in by.items():
    if et != "ptp":
        continue
    m = ms[-1]; cols = spark.table(m["fqn"]).columns
    c1, c2 = fc(cols, "column1", "col1", "comprehensive code"), fc(cols, "column2", "col2", "component code")
    c_mod = fc(cols, "modifier indicator", "modifier")
    ptp_latest.append(f"SELECT '{svc}' AS svc, UPPER(TRIM(`{c1}`)) AS a, UPPER(TRIM(`{c2}`)) AS b, "
                      f"SUBSTR(TRIM(`{c_mod}`),1,1) AS mi FROM {m['fqn']} WHERE SUBSTR(TRIM(`{c_mod}`),1,1) IN ('0','1')")
if len(ptp_latest) > 1:
    u = " UNION ALL ".join(ptp_latest)
    d = spark.sql(f"""WITH u AS ({u})
        SELECT COUNT(*) AS pairs_multi, SUM(CASE WHEN n_mi > 1 THEN 1 ELSE 0 END) AS pairs_disagree
        FROM (SELECT a, b, COUNT(DISTINCT svc) AS n_ed, COUNT(DISTINCT mi) AS n_mi FROM u GROUP BY a, b HAVING COUNT(DISTINCT svc) > 1)""").collect()[0]
    info(f"PTP: {d.pairs_multi:,} pairs in >1 edition; {d.pairs_disagree or 0:,} disagree on modifier indicator")

# COMMAND ----------

# MAGIC %md ## 5. Paid-only sanity (closes F09)

# COMMAND ----------

print("5. paid-only sanity")
neg = spark.sql(f"SELECT COUNT(*) AS n, COALESCE(SUM({PAID}),0) AS amt FROM {CLAIMS} WHERE {PAID} < 0").collect()[0]
zero = spark.sql(f"SELECT COUNT(*) AS n FROM {CLAIMS} WHERE {PAID} = 0 OR {PAID} IS NULL").collect()[0].n
if neg.n:
    warn(f"{neg.n:,} lines with {PAID} < 0 (sum ${neg.amt:,.0f}) - reversals ride as negative paid lines; "
         f"detectors should exclude or net them (config.CLAIMS_PAID_ONLY is then not the whole story)")
else:
    info(f"no negative-amount lines; {zero:,} zero/null-amount lines (fine for unit edits, note for dollar exposure)")

# COMMAND ----------

print()
print("=" * 70)
print("GO" if verdict["go"] else "NO-GO", "-", len(verdict["blockers"]), "blocker(s),", len(verdict["warnings"]), "warning(s)")
for b in verdict["blockers"]:
    print("  BLOCKER", b)
for w in verdict["warnings"]:
    print("  WARN   ", w)
print("=" * 70)

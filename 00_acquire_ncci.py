# Databricks notebook source
# MAGIC %md
# MAGIC # Acquire NCCI — CMS Medicaid NCCI quarterly files → `main.sedo` tables (no manual reformatting)
# MAGIC
# MAGIC Replaces: download 6 zips by hand → open → rename/reformat columns → load one table per file.
# MAGIC
# MAGIC | step | what happens | fails loudly when |
# MAGIC |---|---|---|
# MAGIC | 1 | probe egress to cms.gov | cluster cannot reach the internet → switch to the drop zone (below) |
# MAGIC | 2 | read the CMS page, **discover** zip links (slugs differ quarter to quarter, so URLs are never built); classify each by its **file name only** | a quarter lacks one of the 6 files; two different files claim the same quarter/edit/edition (**error**); CMS adds an edit-like link we cannot classify (**error**) |
# MAGIC | 3 | download → `NCCI_RAW_DIR/<YYYY>q<N>/` (kept forever, SHA-256 logged) | CMS returns an HTML page instead of a zip |
# MAGIC | 4 | parse text file (Excel fallback), locate header, map columns to the **same names as the hand-loaded tables**; every non-blank line must be an edit or a recognised footer/notes line | missing column, bad date, bad modifier indicator, non-numeric MUE, malformed edit row (e.g. blank Column Two), file quarter ≠ link quarter, **Category / MUE header names another edition than the file name** |
# MAGIC | 5 | write `medicaid_ncci_edit_<service>_<ptp\|mue>_q<N>_<YYYY>`; a CMS **revision** (`...-r1-...`) replaces the table loaded from the original (status `replaced`) | table already exists (skipped, never overwritten unless `OVERWRITE = True`); a revision next to a hand-loaded table (a human must compare) |
# MAGIC | 6 | log every file to `ncci_acquisition_log` (its own SHA-256; `src_*` provenance on every load); report missing quarters | any file failed → the notebook ends red (so a scheduled Job alerts) |
# MAGIC
# MAGIC **No egress?** Download the zips from the CMS page by hand into `NCCI_RAW_DIR` (any sub-folder — folder names are
# MAGIC ignored, only the file name is read), keep the CMS file names, set `DOWNLOAD = False`. Steps 4–6 are the same.
# MAGIC Backfilling history: the MUE archive (`acq.CMS_MUE_ARCHIVE`) keeps 2020Q1–2026Q2 under CMS's original names, which
# MAGIC this notebook classifies as they are (verified 2026-10-02). There is no Medicaid PTP archive (PTP files are cumulative).
# MAGIC
# MAGIC **Schedule:** a Databricks Job on this notebook, monthly (e.g. 5th of each month). CMS posts a quarter about a
# MAGIC month before it takes effect and keeps only two quarters on the page; a missed quarter is a permanent MUE gap,
# MAGIC and `mue_history` will then refuse to build. Re-runs are idempotent.
# MAGIC
# MAGIC **Then:** `01_preflight_ncci` → `07_build_kb`.

# COMMAND ----------

import os, sys
_here = os.getcwd()
for _cand in (os.environ.get("PKB_PACKAGE_DIR"), _here, os.path.dirname(_here)):
    if _cand and os.path.exists(os.path.join(_cand, "config.py")):
        sys.path.insert(0, _cand); break
else:
    raise ImportError(f"config.py not found in {_here} or its parent; set PKB_PACKAGE_DIR")

import importlib, re
import pandas as pd
import config, acquire_ncci as acq, ingest_ncci_tables as ing, kb_io
for m in (config, acq, ing, kb_io):
    importlib.reload(m)

# ---- parameters
DOWNLOAD         = True                    # False = read zips already in NCCI_RAW_DIR (drop zone)
OVERWRITE        = False                   # True replaces an existing quarter table - destructive, leave False
QUARTERS         = None                    # e.g. {"2026q4"}; None = every quarter listed on the page / in the drop zone
CROSS_CHECK_XLSX = False                   # also parse the Excel copy and require the same row count (slow for PTP)
CATEGORY_CHECK   = "strict"                # C2-16: "strict" refuses a Category value naming no known edition;
                                           # "lenient" loads it on the file name with a warning. A Category naming
                                           # ANOTHER edition is refused in both modes.
RAW_DIR          = config.NCCI_RAW_DIR
LOG_TABLE        = config.NCCI_ACQUISITION_LOG
CAT, SCH         = config.CATALOG, config.SCHEMA
os.makedirs(RAW_DIR, exist_ok=True)
print("raw archive:", RAW_DIR)

# COMMAND ----------

# MAGIC %md ## 1–2. Egress probe and link discovery

# COMMAND ----------


session = acq.make_session() if DOWNLOAD else None
if DOWNLOAD:
    ok, detail = acq.probe_egress(session)
    print(f"egress to cms.gov: {'OK' if ok else 'BLOCKED'} - {detail}")
    if not ok:
        raise RuntimeError("This cluster cannot reach cms.gov. Either run on compute with internet egress, or "
                           f"download the zips by hand into {RAW_DIR} and set DOWNLOAD = False.")
    html = acq._get(session, acq.CMS_PAGE).text
    links = acq.discover_links(html)
else:
    links = acq.links_from_drop_zone(RAW_DIR)

unknown = [l for l in links if l.kind == "unknown"]
edits = [l for l in links if l.kind == "edits"]
print(f"{len(edits)} edit files, {sum(l.kind == 'change_report' for l in links)} change reports, {len(unknown)} unclassified")
for l in sorted(edits, key=lambda l: (l.label, l.edit, l.service, l.revision)):
    print(f"  {l.label}  {l.edit}  {l.service:12s}  {'r' + str(l.revision) if l.revision else '  '}  {l.name}"
          + (f"   NOTE {l.note}" if l.note else ""))
for l in unknown:
    print(f"  UNCLASSIFIED (review - CMS may have changed naming): {l.url}" + (f"   ({l.note})" if l.note else ""))
# an unclassifiable file whose name looks like an edit file is a quarter we would silently not load
edit_like_unknown = [l for l in unknown if re.search(r"ptp|mue|edit", l.name, re.I)]

# C2-15: two different files for the same quarter/edit/edition/revision is an ERROR - nothing can tell which is
# right. Both are failed by acquire() below (the rest still loads) and the notebook ends red.
duplicates = acq.duplicate_links(links)
for k, urls in duplicates.items():
    print(f"  ERROR duplicate files for {k}: {urls} - keep exactly one")
incomplete = acq.check_quarter_complete(links, allow_duplicates=True)
for q, miss in incomplete.items():
    print(f"  WARNING {q} is missing {miss}")
plan = acq.plan_loads([l for l in links if not QUARTERS or l.label in QUARTERS])
for l in edits:
    if plan.get(l.url, ("load",))[0] == "superseded":
        print(f"  revision: {l.name} is {plan[l.url][1]}")
assert edits, "no NCCI edit files found - check the page / drop zone before going further"

# COMMAND ----------

# MAGIC %md ## 3–5. Download, parse, validate, load

# COMMAND ----------


existing = ing.discover_tables(spark, CAT, SCH)
prev_counts = {}
for m in existing:                           # latest existing table per (edit, service) = the row-count baseline
    prev_counts[f"{m['edit_type']}/{m['service']}"] = m["fqn"]
prev_counts = {k: spark.table(v).count() for k, v in prev_counts.items()}

results = acq.acquire(spark, links, RAW_DIR, CAT, SCH, download_files=DOWNLOAD, overwrite=OVERWRITE,
                      session=session, quarters=QUARTERS, cross_check_xlsx=CROSS_CHECK_XLSX,
                      prev_counts=prev_counts, category_check=CATEGORY_CHECK)
res = acq.results_frame(results)
# fixed column types whatever this run contained (a run with only failures used to write 'rows' as double)
acq.write_log(spark, results, LOG_TABLE)
display(res[["quarter", "edit", "service", "revision", "status", "rows", "tolerated_rows", "fmt", "warning", "error",
             "sha256"]])

# COMMAND ----------

# MAGIC %md ## 6. Completeness and go / no-go

# COMMAND ----------


after = ing.discover_tables(spark, CAT, SCH)
by = {}
for m in after:
    by.setdefault((m["service"], m["edit_type"]), []).append((m["year"], m["quarter"]))
print("quarters loaded (service, edit): missing between first loaded and the current quarter")
for k in sorted(by):
    miss = acq.missing_quarters(by[k])
    print(f"  {k[0]:12s} {k[1]}  {len(by[k])} loaded, latest {max(by[k])}  missing: {miss or 'none'}")

print("\nper file:")
for status in ("loaded", "replaced", "skipped_exists", "superseded", "archived_only", "failed"):
    sub = res[res["status"] == status]
    if len(sub):
        print(f"  {status:15s} {len(sub)}")
replaced = res[res["status"] == "replaced"]
for _, r in replaced.iterrows():
    print(f"  REPLACED {r['quarter']} {r['edit']}/{r['service']}: was {r['replaced_src_zip']}, now {r['src_zip']} "
          f"(Delta history keeps the old version; the old zip stays in {RAW_DIR})")
noted = res[res["note"].notna()]
for _, r in noted.iterrows():
    print(f"  note {r['quarter']} {r['edit']}/{r['service']} [{r['status']}]: {r['note']}")

failed = res[res["status"] == "failed"]
warned = res[res["warning"].notna()]
if len(warned):
    print(f"\n{len(warned)} file(s) WITH WARNINGS - read them before 07_build_kb:")
    for _, r in warned.iterrows():
        print(f"  {r['quarter']} {r['edit']}/{r['service']} [{r['status']}]: {r['warning']}")
problems = []
if len(failed):
    problems.append(f"{len(failed)} NCCI file(s) failed; nothing else was affected. "
                    + "; ".join(f"{r['quarter']} {r['edit']}/{r['service']}: {r['error'][:200]}" for _, r in failed.iterrows()))
if edit_like_unknown:
    problems.append(f"{len(edit_like_unknown)} edit-like file(s) could not be classified and were NOT loaded: "
                    + "; ".join(f"{l.name} ({l.note})" for l in edit_like_unknown))
if problems:
    raise RuntimeError(" | ".join(problems))
print("\nGO: next run 01_preflight_ncci, then 07_build_kb.")

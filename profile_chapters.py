# Databricks notebook source
# MAGIC %md
# MAGIC # Chapter profile — is the extraction framework overfit to the four validation chapters?
# MAGIC
# MAGIC Read-only, no LLM. Runs parse → segment → pre-filter → deterministic (STUB) extraction over every
# MAGIC chapter in `config.CHAPTERS` and reports, per chapter, what the pipeline can *see* and what it
# MAGIC *yields*, then a heuristic profile:
# MAGIC
# MAGIC | profile | means | needs |
# MAGIC |---|---|---|
# MAGIC | `code_rich` | CPT/HCPCS-dense, rules in prose | nothing — current grammar |
# MAGIC | `category_driven` | few codes, many rule cues / per-diem | category→code resolver |
# MAGIC | `modifier_driven` | modifier language dominates | `modifier_required` compiler |
# MAGIC | `foreign_codes` | revenue / DRG / NDC dominate | new code families + predicates |
# MAGIC | `process` | few codes, few cues | KB text only |
# MAGIC | `scanned` | pages with no extractable text | OCR |
# MAGIC
# MAGIC The numbers are the product; the label is a suggestion. Writes nothing to Delta unless `SAVE_CSV`.

# COMMAND ----------

import os, sys, json, time
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
import pandas as pd
import config, policy_docs, extract_rules, chapter_profile
for m in (config, policy_docs, extract_rules, chapter_profile):
    importlib.reload(m)

PDF_DIR   = config.PDF_DIR          # staged Workspace folder next to the package (data/policy_docs)
DOWNLOAD  = True                    # False = only profile PDFs already in PDF_DIR
CHAPTERS  = list(config.CHAPTERS)   # all 29; or e.g. ["10", "13", "19", "22"]
SAVE_CSV  = f"{PDF_DIR}/chapter_profile.csv"   # None = don't save

# COMMAND ----------

rows, failures = [], []
stub = extract_rules.StubBackend()
for ch in CHAPTERS:
    title = config.CHAPTERS[ch][0]
    t0 = time.time()
    try:
        inv = policy_docs.acquire(ch, PDF_DIR, download=DOWNLOAD)
        pages = policy_docs.parse_pdf(inv["path"])
        secs, meta = policy_docs.segment(pages, ch, inv["doc_hash"])
        inv["revision_date"] = meta["revision_date"]
        drafts, dropped, stats = extract_rules.extract_chapter(secs, inv, stub, "profile",
                                                               only_candidates=True, cache={}, log_fn=lambda x: None)
        row = chapter_profile.profile_chapter(ch, title, pages, secs, meta, drafts, dropped, stats)
        row["seconds"] = round(time.time() - t0, 1)
        rows.append(row)
        print(f"  ch {ch:>3}  {row['profile']:16s} pages={row['pages']:3d} sec={row['sections']:3d} "
              f"cand={row['candidate_sections']:3d} codes={row['codes_pipeline']:3d} cues={row['trigger_phrases']:3d} "
              f"stub={row['stub_drafts']:3d}  {title}")
    except Exception as e:
        failures.append({"chapter": ch, "title": title, "error": f"{type(e).__name__}: {str(e)[:200]}"})
        print(f"  ch {ch:>3}  FAILED  {type(e).__name__}: {str(e)[:120]}")

# COMMAND ----------

df = pd.DataFrame(rows)
cols = ["chapter", "profile", "title", "pages", "blank_pages", "sections", "candidate_sections", "candidate_ratio",
        "codes_pipeline", "codes_cpt", "codes_hcpcs", "codes_cdt", "mentions_revenue", "mentions_drg", "mentions_ndc",
        "trigger_phrases", "cue_modifier", "cue_per_diem", "cue_prior_auth", "cue_rate", "cue_age", "cue_pos",
        "stub_drafts", "stub_compiled_types", "stub_dropped", "profile_reason"]
display(df[[c for c in cols if c in df.columns]].sort_values(["profile", "chapter"]))

# COMMAND ----------

print("profile -> chapters")
for prof, chs in chapter_profile.summarize(rows).items():
    print(f"  {prof:16s} {chs}\n{'':18s}needs: {chapter_profile.PROFILE_NEEDS.get(prof, '?')}")
if failures:
    print("\nfailed chapters (fix acquisition / parsing before trusting coverage):")
    for f in failures:
        print(f"  ch {f['chapter']}: {f['error']}")

tot_pages = int(df["pages"].sum()) if len(df) else 0
ready = df[df["profile"] == "code_rich"]
print(f"\n{len(df)}/{len(CHAPTERS)} chapters profiled, {tot_pages:,} pages. "
      f"Ready for the current grammar: {len(ready)} chapter(s) = {int(ready['pages'].sum()) if len(ready) else 0:,} pages "
      f"({(ready['pages'].sum() / tot_pages * 100) if tot_pages else 0:.0f}% of the manual by page).")
print("Deterministic extraction is a floor on known phrasings only; the LLM half (HYBRID) has NOT been measured "
      "on any chapter yet - run extract_chapters on one code_rich chapter next and score it.")

# COMMAND ----------

if SAVE_CSV and len(df):
    df.drop(columns=["stub_by_type"], errors="ignore").to_csv(SAVE_CSV, index=False)
    print(f"saved {SAVE_CSV}")

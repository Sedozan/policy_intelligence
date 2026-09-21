"""
state_rules.py  -  the reviewed authoring source for AHCCCS (state) rules.

State rules live in a Delta table, not in Python modules. "Add a chapter" means
"add reviewed rows", never "edit the build notebook". Every row carries a
review_status; the build loads only rows a named human has approved (or the
hand-authored PoC rules migrated as 'legacy_hand_authored').

Table: main.sedo.state_rules  (see STATE_RULE_COLUMNS)

Review round-trip (the human gate, made auditable):
  export_review_workbook()  -> Excel of rows needing a decision
  ingest_review_decisions() <- the same file with decision columns filled;
                               appends to policy_review_log and updates
                               review_status on state_rules.
"""
from __future__ import annotations

import json
from datetime import datetime

import pandas as pd

from schema import Rule

RULE_FIELDS = [
    "rule_id", "origin", "plane", "binding_status", "statement", "predicate", "codes",
    "population", "machine_checkable", "not_checkable_reason", "required_claim_fields",
    "ambiguity_flag", "ambiguity_note", "alternates", "source_doc", "source_locator",
    "source_url", "doc_version", "doc_hash", "effective_date", "end_date", "deactivated",
    "supersedes", "notes",
]
_JSON_FIELDS = {"predicate", "codes", "required_claim_fields", "alternates", "supersedes"}
REVIEW_FIELDS = ["chapter", "review_status", "reviewer", "reviewed_at", "review_note"]
# filled only for rows produced by extract_rules (None for hand-authored rows)
EXTRACTION_FIELDS = ["extraction_key", "verbatim_quote", "section_id", "extraction_confidence",
                     "extraction_run_id", "extraction_backend", "grounding"]
STATE_RULE_COLUMNS = RULE_FIELDS + REVIEW_FIELDS + EXTRACTION_FIELDS

LOADABLE_STATUS = ("approved", "legacy_hand_authored")


# ------------------------------------------------------------ row <-> rule
def rule_to_row(rule, chapter: str, review_status: str = "draft",
                reviewer: str | None = None, review_note: str | None = None) -> dict:
    d = rule.to_dict() if hasattr(rule, "to_dict") else dict(rule)
    row = {}
    for f in RULE_FIELDS:
        v = d.get(f)
        if f in _JSON_FIELDS:
            v = json.dumps(v if v is not None else ([] if f != "predicate" else {}), default=str)
        row[f] = v
    row.update(chapter=chapter, review_status=review_status, reviewer=reviewer,
               reviewed_at=datetime.now().isoformat(timespec="seconds") if reviewer else None,
               review_note=review_note)
    if d.get("extraction_key"):
        row["extraction_key"] = d["extraction_key"]
    for f in EXTRACTION_FIELDS:
        row.setdefault(f, None)
    return row


def row_to_rule(row: dict) -> Rule:
    kw = {}
    for f in RULE_FIELDS:
        v = row.get(f)
        if f in _JSON_FIELDS and isinstance(v, str):
            v = json.loads(v) if v else ([] if f != "predicate" else {})
        if f in ("machine_checkable", "ambiguity_flag", "deactivated") and v is not None:
            v = bool(v)
        if isinstance(v, float) and pd.isna(v):
            v = None
        kw[f] = v
    if not kw.get("supersedes"):
        kw["supersedes"] = None
    kw.pop("rule_id", None)              # re-derived by rule_normalize
    ek = row.get("extraction_key")       # stable join key; survives normalization
    kw["extraction_key"] = None if (isinstance(ek, float) and pd.isna(ek)) else (ek or None)
    return Rule(**kw)


# ----------------------------------------------------------------- loading
def load_state_rules(spark, table: str, scope_codes: set[str] | None = None,
                     statuses=LOADABLE_STATUS) -> list[Rule]:
    df = spark.table(table).toPandas()
    if df.empty:
        print(f"  state: {table} is empty")
        return []
    df = df[df["review_status"].isin(statuses) & ~df["deactivated"].fillna(False).astype(bool)]
    out, skipped_scope = [], 0
    for row in df.to_dict("records"):
        codes = json.loads(row["codes"]) if isinstance(row.get("codes"), str) and row["codes"] else []
        codes = [str(c).strip().upper() for c in codes]
        # a coded rule must touch a paid code; a policy-level rule (no codes) always loads
        if scope_codes and codes and not any(c in scope_codes for c in codes):
            skipped_scope += 1
            continue
        out.append(row_to_rule(row))
    by_ch = df["chapter"].value_counts().to_dict()
    print(f"  state: {len(out)} reviewed rules loaded {by_ch}"
          + (f"; {skipped_scope} out of claims scope" if skipped_scope else ""))
    return out


# ---------------------------------------------------------------- writing
def write_state_rules(spark, table: str, rows: list[dict], mode: str = "overwrite"):
    from kb_io import write_table
    pdf = pd.DataFrame(rows, columns=STATE_RULE_COLUMNS)
    for b in ("machine_checkable", "ambiguity_flag", "deactivated"):
        pdf[b] = pdf[b].fillna(False).astype(bool)
    write_table(spark, pdf, table, mode)


# ----------------------------------------------------------------- merge
REVIEWED = ("approved", "legacy_hand_authored")


def merge_drafts(existing: pd.DataFrame | None, drafts: list[dict]) -> tuple[pd.DataFrame, dict]:
    """Merge freshly extracted drafts into the state_rules table, by extraction_key.

      * a key already reviewed (approved / legacy) is left untouched - never downgraded
      * a key previously rejected stays rejected - never resurrected by re-extraction
      * a key already in draft is replaced by the new draft (fresh quote, confidence, run)
      * a new key is added as a draft
    Returns (merged_df, counts)."""
    cols = STATE_RULE_COLUMNS
    if existing is None or len(existing) == 0:
        existing = pd.DataFrame(columns=cols)
    for c in cols:
        if c not in existing.columns:
            existing[c] = None
    existing = existing[cols].copy()
    by_key = {k: i for i, k in existing["extraction_key"].items() if isinstance(k, str) and k}
    by_id = {k: i for i, k in existing["rule_id"].items() if isinstance(k, str) and k}
    counts = {"added": 0, "replaced_draft": 0, "kept_reviewed": 0, "kept_rejected": 0}
    keep_rows = existing.to_dict("records")
    idx_to_drop = set()
    new_rows = []
    for d in drafts:
        key = d.get("extraction_key")
        i = by_key.get(key, by_id.get(d.get("rule_id")))
        if i is None:
            new_rows.append(d); counts["added"] += 1; continue
        status = existing.at[i, "review_status"]
        if status in REVIEWED:
            counts["kept_reviewed"] += 1
        elif status == "rejected":
            counts["kept_rejected"] += 1
        else:
            idx_to_drop.add(i); new_rows.append(d); counts["replaced_draft"] += 1
    merged = [r for j, r in enumerate(keep_rows) if j not in idx_to_drop] + new_rows
    out = pd.DataFrame(merged, columns=cols)
    for b in ("machine_checkable", "ambiguity_flag", "deactivated"):
        out[b] = out[b].fillna(False).astype(bool)
    return out, counts


# --------------------------------------------------------- review workbook
def _xlsx_write(df: pd.DataFrame, path: str) -> str:
    """openpyxl seeks while writing, which a UC Volume (FUSE) rejects with
    OSError 95. So write to local disk, then sequential-copy to the destination
    (sequential writes to a Volume are fine). Returns where the file actually is."""
    import os, shutil, tempfile
    if os.path.abspath(os.path.dirname(path) or ".") == os.path.abspath(tempfile.gettempdir()):
        df.to_excel(path, index=False, engine="openpyxl")
        return path
    local = os.path.join(tempfile.gettempdir(), os.path.basename(path))
    df.to_excel(local, index=False, engine="openpyxl")
    try:
        shutil.copyfile(local, path)
        return path
    except OSError as e:
        print(f"  could not copy workbook to {path} ({e}); it is at {local}")
        return local


def _xlsx_read(path: str) -> pd.DataFrame:
    """Read an .xlsx that may live on a UC Volume; fall back to a local copy."""
    import os, shutil, tempfile
    try:
        return pd.read_excel(path, engine="openpyxl")
    except OSError:
        local = os.path.join(tempfile.gettempdir(), os.path.basename(path))
        shutil.copyfile(path, local)
        return pd.read_excel(local, engine="openpyxl")


def export_review_workbook(spark, table: str, path: str,
                           statuses=("draft", "needs_review")) -> str:
    df = spark.table(table).toPandas()
    need = df[df["review_status"].isin(statuses)].copy()
    out = pd.DataFrame({
        "rule_id": need["rule_id"],
        "extraction_key": need["extraction_key"] if "extraction_key" in need else None,
        "chapter": need["chapter"],
        "policy_text_and_notes": need["notes"],
        "proposed_rule": need["statement"],
        "predicate": need["predicate"],
        "citation": need["source_doc"].astype(str) + " | " + need["source_locator"].astype(str),
        "source_url": need["source_url"],
        "doc_hash": need["doc_hash"] if "doc_hash" in need else None,   # version the decision binds to
        "verbatim_quote": need["verbatim_quote"] if "verbatim_quote" in need else None,
        "confidence": need["extraction_confidence"] if "extraction_confidence" in need else None,
        "open_question": need["ambiguity_note"],
        "decision": "",                 # Approve | Reject | Fix
        "corrected_rule": "",           # Fix: corrected plain-English statement
        "corrected_predicate": "",      # Fix: corrected predicate as JSON (REQUIRED to change the executable rule)
        "reviewer": "",
        "review_note": "",
    })
    if "confidence" in out.columns:
        out = out.sort_values("confidence", na_position="last")
    written = _xlsx_write(out, path)
    print(f"  exported {len(out)} rules for review -> {written}")
    return written


def ingest_review_decisions(spark, table: str, path: str, log_table: str):
    filled = _xlsx_read(path)
    filled["decision"] = filled["decision"].astype(str).str.strip().str.lower()
    filled = filled[filled["decision"].isin(["approve", "reject", "fix"])].copy()
    if filled.empty:
        print("  no decisions found in workbook")
        return
    for c in ("corrected_rule", "corrected_predicate", "extraction_key", "reviewer", "review_note", "doc_hash"):
        if c not in filled.columns:
            filled[c] = None
    filled["ingested_at"] = datetime.now().isoformat(timespec="seconds")
    # log carries extraction_key so an approval joins to the exact executable rule,
    # and corrected_predicate so the audit trail shows what the reviewer changed.
    # `applied` is filled in below: the log distinguishes a decision that was
    # ATTEMPTED from one that took effect (F07).
    log_cols = ["rule_id", "extraction_key", "decision", "corrected_rule",
                "corrected_predicate", "reviewer", "review_note", "ingested_at"]

    from extract_rules import predicate_valid, _FIELDS_BY_TYPE
    from validate_gate import _codes_from_predicate
    import schema
    df = spark.table(table).toPandas()
    applied = {"approve": 0, "reject": 0, "fix": 0, "fix_deferred": 0, "unmatched": 0,
               "no_reviewer": 0, "stale_version": 0}
    outcome: list[str] = []
    for rec in filled.to_dict("records"):
        m = df["rule_id"] == rec["rule_id"]
        if not m.any():
            applied["unmatched"] += 1
            outcome.append("unmatched")
            continue
        decision = rec["decision"]
        note = str(rec.get("review_note") or "").strip()

        # F07: a decision needs a named human. An unsigned approve/reject/fix is not
        # applied; it is parked so the workbook can be completed and re-imported.
        reviewer = str(rec.get("reviewer") or "").strip()
        if not reviewer or reviewer.lower() in ("nan", "none"):
            df.loc[m, "review_status"] = "needs_review"
            df.loc[m, "review_note"] = (note + " [decision not applied: reviewer name required]").strip()
            applied["no_reviewer"] += 1
            outcome.append("no_reviewer")
            continue
        # F07: the decision binds to the document version the reviewer saw. If the
        # rule has been re-extracted from a newer PDF since the export, park it.
        wb_hash = str(rec.get("doc_hash") or "").strip()
        cur_hash = str(df.loc[m, "doc_hash"].iloc[0] or "").strip() if "doc_hash" in df.columns else ""
        if wb_hash and cur_hash and wb_hash != cur_hash and wb_hash.lower() != "nan":
            df.loc[m, "review_status"] = "needs_review"
            df.loc[m, "review_note"] = (note + f" [decision not applied: reviewed doc {wb_hash[:8]} "
                                        f"but current rule is from doc {cur_hash[:8]}; re-review]").strip()
            applied["stale_version"] += 1
            outcome.append("stale_version")
            continue
        df.loc[m, "reviewer"] = reviewer
        df.loc[m, "reviewed_at"] = rec["ingested_at"]

        if decision == "approve":
            df.loc[m, "review_status"] = "approved"
            df.loc[m, "review_note"] = note
            applied["approve"] += 1
            outcome.append("approved")
            continue
        if decision == "reject":
            df.loc[m, "review_status"] = "rejected"
            df.loc[m, "review_note"] = note
            applied["reject"] += 1
            outcome.append("rejected")
            continue

        # decision == "fix": the PREDICATE is what compiles, so a fix must correct the
        # predicate — not just the prose. A prose-only fix must NOT silently approve the
        # old (wrong) predicate; it is parked as needs_review until a predicate is given.
        corr_stmt = rec.get("corrected_rule")
        if isinstance(corr_stmt, str) and corr_stmt.strip():
            df.loc[m, "statement"] = corr_stmt.strip()
        new_pred, cp = None, rec.get("corrected_predicate")
        if isinstance(cp, str) and cp.strip():
            try:
                new_pred = json.loads(cp)
            except Exception:
                new_pred = None
        if new_pred is None:
            df.loc[m, "review_status"] = "needs_review"
            df.loc[m, "review_note"] = (note + " [fix needs a valid corrected_predicate JSON; "
                                        "prose alone cannot change the executable rule]").strip()
            applied["fix_deferred"] += 1
            outcome.append("fix_deferred")
            continue
        ok, why = predicate_valid(new_pred)
        if not ok:
            df.loc[m, "review_status"] = "needs_review"
            df.loc[m, "review_note"] = (note + f" [fix rejected: invalid corrected_predicate: {why}]").strip()
            applied["fix_deferred"] += 1
            outcome.append("fix_deferred")
            continue
        compiled = new_pred.get("type") in schema.PREDICATE_TYPES_COMPILED
        # F07: everything DERIVED from the predicate is re-derived, so `codes`,
        # required fields and the grounding record never describe the old predicate.
        new_codes = _codes_from_predicate(new_pred)
        old_pred_json = df.loc[m, "predicate"].iloc[0]
        df.loc[m, "predicate"] = json.dumps(new_pred, default=str)
        df.loc[m, "codes"] = json.dumps(new_codes)
        df.loc[m, "required_claim_fields"] = json.dumps(_FIELDS_BY_TYPE.get(new_pred.get("type"), []))
        df.loc[m, "machine_checkable"] = compiled
        df.loc[m, "not_checkable_reason"] = None if compiled else new_pred.get("reason")
        if "grounding" in df.columns:
            df.loc[m, "grounding"] = json.dumps({"superseded_by_reviewer": True, "reviewer": reviewer,
                                                 "original_predicate": old_pred_json}, default=str)
        df.loc[m, "review_status"] = "approved"
        df.loc[m, "review_note"] = (note + " [predicate corrected by reviewer; codes/fields re-derived]").strip()
        applied["fix"] += 1
        outcome.append("fixed")

    filled["applied_outcome"] = outcome
    from kb_io import write_table
    write_table(spark, filled[log_cols + ["applied_outcome"]], log_table, mode="append")
    write_state_rules(spark, table, df.to_dict("records"))
    print(f"  applied decisions {applied}; log -> {log_table}")

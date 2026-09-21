"""
ingest_ncci_tables.py  -  Medicaid NCCI quarterly edit tables -> Rule objects.

Discovers every  medicaid_ncci_edit_<service>_<ptp|mue>_q<N>_<YYYY>  table in
the source schema and turns it into dated, cited, machine-checkable rules.

Design (MVP):
  * Scope filter is applied IN SPARK before anything reaches pandas, so a
    claims-scoped run over the full federal library stays cheap.
  * PTP: a pair is kept only when BOTH codes are in scope. A PTP violation
    needs both codes co-billed, so a pair with one never-paid code can never
    fire; keeping it only inflates the rule count.
  * PTP quarterly files are cumulative snapshots and each pair carries its own
    effective/deletion date, so only the LATEST PTP quarter per service is
    loaded. Loading several duplicates every active pair.
  * MUE files carry no dates, so validity is reconstructed from the quarter
    sequence: a value carries forward until it changes or the code disappears.
    A code that vanishes from a later quarter gets its run closed (previously
    a removed MUE stayed live forever).
  * Medicaid NCCI MUEs are per-claim-line edits. CMS's Medicaid NCCI Technical
    Guidance states the MUE is compared to the units of service on ONE claim line
    at a time and is NOT summed across lines for a date of service. So every MUE
    compiles per-line. MAI (MUE Adjudication Indicator) is not present in the
    Medicaid files; if a future file carries it, it is recorded for audit only and
    never switches adjudication to a date-of-service sum (that is the Medicare model).
  * No 'enforcement' plane. "A hard-denial pair was paid anyway" is a finding
    the detector labels at run time (finding_severity), not a second rule.
  * Every NCCI rule carries source_url and a content hash (doc_hash) so a
    reviewer can tell exactly which edit and which snapshot it came from.
"""
from __future__ import annotations

import hashlib
import re
from datetime import date, timedelta

import pandas as pd

import config
from schema import Rule

# --------------------------------------------------------------- constants
MEDICAID_NCCI_URL = ("https://www.cms.gov/medicare/coding-billing/"
                     "ncci-medicaid/medicaid-ncci-edit-files")

QUARTER_START = {1: "01-01", 2: "04-01", 3: "07-01", 4: "10-01"}

_TABLE_RE = re.compile(
    r"^medicaid_ncci_edit_(?P<service>[a-z]+)_(?P<edit>ptp|mue)_q(?P<q>[1-4])_(?P<y>\d{4})$",
    re.I)


# ----------------------------------------------------------------- helpers
def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def _find_col(columns, *cands):
    """Return the real column name whose normalised form matches any candidate
    (exact first, then prefix), or None."""
    norm = {_norm(c): c for c in columns}
    for cand in cands:
        k = _norm(cand)
        if k in norm:
            return norm[k]
    for cand in cands:
        k = _norm(cand)
        for nk, real in norm.items():
            if nk.startswith(k):
                return real
    for cand in cands:
        k = _norm(cand)
        for nk, real in norm.items():
            if k in nk:
                return real
    return None


def _iso(v) -> str | None:
    """CMS date cells: '20250101', '1/1/2025', '*', '', NaN. '*' and blank mean
    'no date' (for deletion date: still active)."""
    if v is None:
        return None
    s = str(v).strip()
    if s.endswith(".0"):                # CMS dates read as floats: 20241231.0
        s = s[:-2]
    if s in ("", "*", "nan", "NaT", "None", "99999999"):
        return None
    try:
        return pd.to_datetime(s).date().isoformat()
    except Exception:
        return None


def _hash(*parts) -> str:
    return hashlib.sha256(
        "|".join("" if p is None else str(p) for p in parts).encode()
    ).hexdigest()[:16]


def quarter_sort_key(meta: dict):
    return (int(meta["year"]), int(meta["quarter"]))


def discover_tables(spark, catalog: str, schema: str) -> list[dict]:
    rows = spark.sql(f"SHOW TABLES IN {catalog}.{schema}").collect()
    metas = []
    for r in rows:
        name = r["tableName"]
        m = _TABLE_RE.match(name)
        if not m:
            continue
        q, y = int(m["q"]), int(m["y"])
        metas.append({
            "table": name,
            "fqn": f"{catalog}.{schema}.{name}",
            "service": m["service"].lower(),
            "edit_type": m["edit"].lower(),
            "quarter": q, "year": y,
            "label": f"{y}Q{q}",
            "starts": f"{y}-{QUARTER_START[q]}",
        })
    metas.sort(key=lambda m: (m["service"], m["edit_type"], quarter_sort_key(m)))
    return metas


# ---------------------------------------------------- Spark-side scoping
def _scope_in_spark(sdf, edit_type: str, code_filter):
    """Filter the Spark frame to in-scope codes BEFORE toPandas(). Falls back
    to no filter (pandas re-filters) if the code columns cannot be resolved."""
    if not code_filter:
        return sdf
    from pyspark.sql import functions as F
    codes = [c.upper() for c in code_filter]
    if edit_type == "ptp":
        c1 = _find_col(sdf.columns, "column1", "col1", "comprehensive code")
        c2 = _find_col(sdf.columns, "column2", "col2", "component code")
        if not (c1 and c2):
            return sdf
        return sdf.filter(F.upper(F.trim(sdf[c1])).isin(codes)
                          & F.upper(F.trim(sdf[c2])).isin(codes))
    c = _find_col(sdf.columns, "hcpcs/cpt code", "hcpcs cpt code", "procedure code", "code")
    if not c:
        return sdf
    return sdf.filter(F.upper(F.trim(sdf[c])).isin(codes))


# --------------------------------------------------------------------- PTP
def ptp_rules(df: pd.DataFrame, meta: dict, code_filter=None) -> list[Rule]:
    cols = df.columns
    c1 = _find_col(cols, "column1", "col1", "comprehensive code")
    c2 = _find_col(cols, "column2", "col2", "component code")
    c_mod = _find_col(cols, "modifier indicator", "modifier")
    c_eff = _find_col(cols, "effective date", "effdt")
    c_del = _find_col(cols, "deletion date", "deldt")
    c_rat = _find_col(cols, "ptp edit rationale", "edit rationale", "rationale")
    if not (c1 and c2):
        raise ValueError(f"{meta['table']}: cannot find Column 1/2. Saw {list(cols)}")
    if not c_mod:
        # without this every row would be treated as indicator 9 and silently skipped
        raise ValueError(f"{meta['table']}: cannot find the modifier-indicator column. Saw {list(cols)}")
    if not (c_eff and c_del):
        print(f"    WARNING [{meta['table']}]: effective/deletion date column not found "
              f"(eff={c_eff}, del={c_del}); deleted edits would stay live. Saw {list(cols)}")

    service = meta["service"]
    rules, skipped9 = [], 0
    for row in df.to_dict("records"):
        a = str(row[c1]).strip().upper()
        b = str(row[c2]).strip().upper()
        if code_filter and not (a in code_filter and b in code_filter):
            continue
        mi = str(row[c_mod]).strip()[:1]
        if mi not in ("0", "1"):
            skipped9 += 1          # 9 = edit deleted; nothing to enforce
            continue
        eff = (_iso(row[c_eff]) if c_eff else None) or meta["starts"]
        end = _iso(row[c_del]) if c_del else None
        rat = str(row[c_rat]).strip() if c_rat and str(row[c_rat]).strip() not in ("nan", "") else ""
        bypass = ("an NCCI-associated modifier may bypass this edit when documented"
                  if mi == "1" else "no modifier can bypass this edit")
        rules.append(Rule(
            origin="ncci", plane="coding_edit", binding_status="mandatory",
            statement=(f"NCCI PTP ({service}): {b} should not be reported with {a} "
                       f"for the same member, provider and date of service "
                       f"(modifier indicator {mi})."),
            predicate={"type": "code_pair_prohibited", "code_a": a, "code_b": b,
                       "modifier_indicator": mi, "service_category": service},
            codes=[a, b], population="all",
            machine_checkable=True,
            required_claim_fields=["proc_cd", "member_id", "provider_id",
                                   "srvc_bgn_dt", "mod_1", "mod_2", "mod_3", "mod_4"],
            source_doc=f"CMS Medicaid NCCI PTP edits ({service})",
            source_locator=f"pair {a}/{b}, {meta['label']} file",
            source_url=MEDICAID_NCCI_URL,
            doc_version=meta["table"],
            doc_hash=_hash(meta["table"], "ptp", a, b, mi, eff, end),
            effective_date=eff, end_date=end,
            notes=(f"Modifier indicator {mi}: {bypass}."
                   + (f" Edit deleted {end}; binding for dates of service on or before that."
                      if end else "")
                   + (f" CMS rationale: {rat}." if rat else "")),
        ).assign_id())

    print(f"    [{meta['label']}] {service:12s} ptp  {len(rules):,} rules"
          f"  (skipped {skipped9:,} deleted-indicator rows)")
    return rules


# --------------------------------------------------------------------- MUE
def quarter_gaps(quarters: list[dict]) -> list[str]:
    """ISO start dates of quarters missing between consecutive loaded quarters."""
    missing = []
    for prev, nxt in zip(quarters, quarters[1:]):
        d = date.fromisoformat(prev["starts"])
        end = date.fromisoformat(nxt["starts"])
        while True:
            m = d.month + 3
            d = date(d.year + (m - 1) // 12, (m - 1) % 12 + 1, 1)
            if d >= end:
                break
            missing.append(d.isoformat())
    return missing


def mue_history(frames: list[tuple[dict, pd.DataFrame]], code_filter=None,
                allow_gaps: bool = False) -> list[Rule]:
    """Quarterly MUE snapshots -> dated rules (run-length over the sequence).

    A run opens when a code first appears or its value/MAI changes, and closes
    the day before the next quarter in which the value changes OR the code is
    absent. The last run stays open (end_date=None) only if the code is present
    in the most recent quarter loaded.
    """
    by_service: dict[str, list[tuple[dict, pd.DataFrame]]] = {}
    for meta, df in frames:
        by_service.setdefault(meta["service"], []).append((meta, df))

    rules: list[Rule] = []
    mai_seen = False
    for service, seq in by_service.items():
        seq.sort(key=lambda t: quarter_sort_key(t[0]))
        quarters = [m for m, _ in seq]
        # F15: the run-length reconstruction assumes CONSECUTIVE quarters. A missing
        # quarter would silently extend the previous value across it. Refuse to infer
        # across a gap unless the caller opts in, and then flag every run that spans one.
        gaps = quarter_gaps(quarters)
        if gaps and not allow_gaps:
            raise ValueError(f"{service}: MUE quarters are not contiguous; missing {gaps}. "
                             f"Load the missing tables or call mue_history(..., allow_gaps=True).")
        gap_starts = [g for g in gaps]

        # code -> {quarter_label: (value, rationale, mai)}
        obs: dict[str, dict[str, tuple]] = {}
        for meta, df in seq:
            cols = df.columns
            c_code = _find_col(cols, "hcpcs/cpt code", "hcpcs cpt code", "procedure code", "code")
            c_val = _find_col(cols, "mue value", "mue values")
            c_rat = _find_col(cols, "mue rationale", "rationale")
            c_mai = _find_col(cols, "mue adjudication indicator", "adjudication indicator", "mai")
            if not (c_code and c_val):
                raise ValueError(f"{meta['table']}: cannot find code/value. Saw {list(cols)}")
            if c_mai:
                mai_seen = True
            for row in df.to_dict("records"):
                code = str(row[c_code]).strip().upper()
                if code_filter and code not in code_filter:
                    continue
                try:
                    val = int(float(row[c_val]))
                except (TypeError, ValueError):
                    continue
                mai = None
                if c_mai:
                    try:
                        mai = int(str(row[c_mai]).strip()[:1])
                    except (TypeError, ValueError):
                        mai = None
                rat = str(row[c_rat]).strip() if c_rat else ""
                if rat in ("nan", "None"):
                    rat = ""
                obs.setdefault(code, {})[meta["label"]] = (val, rat, mai)

        for code, per_q in obs.items():
            runs = []  # [start_meta, end_iso|None, val, rat, mai]
            for i, meta in enumerate(quarters):
                cur = per_q.get(meta["label"])
                if cur is None:                        # code absent this quarter
                    if runs and runs[-1][1] is None:
                        runs[-1][1] = (date.fromisoformat(meta["starts"])
                                       - timedelta(days=1)).isoformat()
                    continue
                val, rat, mai = cur
                if runs and runs[-1][1] is None and runs[-1][2] == val and runs[-1][4] == mai:
                    continue                           # unchanged -> extend
                if runs and runs[-1][1] is None:       # value changed -> close prior
                    runs[-1][1] = (date.fromisoformat(meta["starts"])
                                   - timedelta(days=1)).isoformat()
                runs.append([meta, None, val, rat, mai])

            for start_meta, end, val, rat, mai in runs:
                # CMS Medicaid NCCI Technical Guidance: an MUE is applied to the units
                # of service on ONE claim line at a time and is NOT summed across lines
                # for a DOS. Every Medicaid MUE is therefore a per-line edit. MAI, when
                # present, is kept for audit only; it never routes to a per-DOS sum (the
                # per-DOS/same-provider compiler exists only for state daily caps).
                pred = {"type": "max_units_per_line", "code": code, "threshold": val,
                        "service_category": service}
                if mai is not None:
                    pred["mai"] = mai
                mai_note = (f" Source MAI={mai} recorded for reference; Medicaid NCCI "
                            f"adjudicates per claim line regardless." if mai in (2, 3) else "")
                spans_gap = [g for g in gap_starts
                             if g > start_meta["starts"] and (end is None or g <= end)]
                rules.append(Rule(
                    origin="ncci", plane="coding_edit", binding_status="mandatory",
                    statement=(f"NCCI MUE ({service}): {code} maximum {val} unit(s) "
                               f"per claim line."),
                    predicate=pred,
                    codes=[code], population="all",
                    machine_checkable=True,
                    required_claim_fields=["proc_cd", "units"],
                    source_doc=f"CMS Medicaid NCCI MUE values ({service})",
                    source_locator=f"code {code}, {start_meta['label']} file",
                    source_url=MEDICAID_NCCI_URL,
                    doc_version=start_meta["table"],
                    doc_hash=_hash(start_meta["table"], "mue", code, val, mai, end),
                    effective_date=start_meta["starts"], end_date=end,
                    ambiguity_flag=bool(spans_gap),
                    ambiguity_note=(f"Run spans unloaded quarter(s) {spans_gap}; value in "
                                    f"those quarters is inferred, not observed." if spans_gap else None),
                    notes=(("Validity reconstructed from the quarterly sequence; "
                            "MUE files carry no effective-date column.")
                           + mai_note
                           + (f" Value retired {end}." if end else "")
                           + (f" CMS rationale: {rat}." if rat else "")),
                ).assign_id())

    if not mai_seen:
        print("    MUE: source tables carry no MUE Adjudication Indicator (MAI); "
              "all MUEs modelled as per-claim-line caps.")
    return rules


# ------------------------------------------------------------------ driver
def ingest_tables(spark, catalog: str | None = None, schema: str | None = None,
                  code_filter=None, allow_mue_gaps: bool | None = None) -> list[Rule]:
    """Build NCCI rules. code_filter: iterable of procedure codes (None = all).
    allow_mue_gaps: infer MUE validity across an unloaded quarter (flagged ambiguous);
    default config.NCCI_ALLOW_MUE_GAPS (False = refuse and name the missing quarters)."""
    if allow_mue_gaps is None:
        allow_mue_gaps = getattr(config, "NCCI_ALLOW_MUE_GAPS", False)
    catalog = catalog or config.CATALOG
    schema = schema or config.SCHEMA
    cf = set(c.strip().upper() for c in code_filter) if code_filter else None

    metas = discover_tables(spark, catalog, schema)
    if not metas:
        print(f"  no NCCI tables found in {catalog}.{schema}")
        return []
    print(f"  found {len(metas)} NCCI tables"
          + (f"; scope = {len(cf):,} codes" if cf else "; scope = all codes"))

    # latest PTP quarter per service (files are cumulative snapshots)
    latest_ptp: dict[str, dict] = {}
    for m in metas:
        if m["edit_type"] == "ptp":
            cur = latest_ptp.get(m["service"])
            if cur is None or quarter_sort_key(m) > quarter_sort_key(cur):
                latest_ptp[m["service"]] = m

    rules: list[Rule] = []
    mue_frames: list[tuple[dict, pd.DataFrame]] = []
    for m in metas:
        if m["edit_type"] == "ptp" and m is not latest_ptp[m["service"]]:
            continue
        sdf = _scope_in_spark(spark.table(m["fqn"]), m["edit_type"], cf)
        df = sdf.toPandas()
        if m["edit_type"] == "ptp":
            rules += ptp_rules(df, m, cf)
        else:
            mue_frames.append((m, df))

    if mue_frames:
        mue = mue_history(mue_frames, cf, allow_gaps=allow_mue_gaps)
        by_svc: dict[str, int] = {}
        for r in mue:
            by_svc[r.predicate["service_category"]] = by_svc.get(r.predicate["service_category"], 0) + 1
        print(f"    MUE: {len(mue):,} dated rules " + str(by_svc))
        rules += mue

    ptp_labels = {s: m["label"] for s, m in latest_ptp.items()}
    print(f"  NCCI rules built: {len(rules):,}   (PTP snapshot used: {ptp_labels})")
    return rules


def route_note() -> str:
    return ("NCCI edits are edition-specific: CMS applies the practitioner (PRA), "
            "outpatient-hospital (OPH) and DME MUE/PTP files by the claim's provider "
            "context, not by procedure code. This claims table cannot distinguish those "
            "contexts — all three editions land on FORM_TYP 'A' — so the detector does "
            "NOT route NCCI by form (config.NCCI_ROUTE_BY_FORM=False); the procedure "
            "code selects the claims and a code shared by >1 edition is collapsed per "
            "exact date window (see rule_normalize.collapse_ncci_editions). This is a "
            "documented workaround for the data, not correct edition routing: flip "
            "NCCI_ROUTE_BY_FORM to True only once a real provider-type field exists.")

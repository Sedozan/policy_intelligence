"""Offline checks against the REAL package modules (no Spark). Run: cd pie_mvp && python -m pytest tests -q"""
import os, sys, json
import pandas as pd

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.dirname(HERE))

import ingest_ncci_tables as ing
import rule_normalize as rn
import detector_gen as dg
import state_rules as sr
import validate_gate as vg
from schema import Rule


def meta(service, edit, q, y):
    return {"table": f"medicaid_ncci_edit_{service}_{edit}_q{q}_{y}", "fqn": "x",
            "service": service, "edit_type": edit, "quarter": q, "year": y,
            "label": f"{y}Q{q}", "starts": f"{y}-{ing.QUARTER_START[q]}"}


# ------------------------------------------------------------------ PTP
def test_ptp_both_codes_and_indicator9_and_dates():
    df = pd.DataFrame({
        "Column 1": ["90791", "90791", "99213", "12345"],
        "Column 2": ["90832", "90833", "36415", "67890"],
        "Effective Date": ["20210101", "20210101", "20200101", "20200101"],
        "Deletion Date": ["*", "20241231", "*", "*"],
        "Modifier Indicator": ["0", "1", "9", "1"],
        "PTP Edit Rationale": ["Misuse", "Standards", "x", "y"],
    })
    scope = {"90791", "90832", "90833", "99213"}   # 12345/67890 out of scope
    rules = ing.ptp_rules(df, meta("practitioner", "ptp", 4, 2026), scope)
    assert [r.predicate["code_b"] for r in rules] == ["90832", "90833"]   # 9 skipped, pair out of scope skipped
    r0, r1 = rules
    assert r0.effective_date == "2021-01-01" and r0.end_date is None
    assert r1.end_date == "2024-12-31"
    assert r0.source_url and r0.doc_hash
    assert all(r.plane != "enforcement" for r in rules)


# ------------------------------------------------------------------ MUE
def test_mue_runs_close_on_change_and_absence():
    q2 = pd.DataFrame({"HCPCS/CPT Code": ["A", "B", "C"], "Practitioner Services MUE Values": [4, 2, 1], "MUE Rationale": ["", "", ""]})
    q3 = pd.DataFrame({"HCPCS/CPT Code": ["A", "B"], "Practitioner Services MUE Values": [4, 3], "MUE Rationale": ["", "", ][:2]})
    q4 = pd.DataFrame({"HCPCS/CPT Code": ["A", "B", "C"], "Practitioner Services MUE Values": [4, 3, 5], "MUE Rationale": ["", "", ""]})
    frames = [(meta("practitioner", "mue", 2, 2026), q2), (meta("practitioner", "mue", 3, 2026), q3), (meta("practitioner", "mue", 4, 2026), q4)]
    rules = ing.mue_history(frames)
    by = {}
    for r in rules:
        by.setdefault(r.predicate["code"], []).append((r.effective_date, r.end_date, r.predicate["threshold"]))
    assert by["A"] == [("2026-04-01", None, 4)]                              # unchanged across all quarters
    assert by["B"] == [("2026-04-01", "2026-06-30", 2), ("2026-07-01", None, 3)]  # value change
    assert by["C"] == [("2026-04-01", "2026-06-30", 1), ("2026-10-01", None, 5)]  # absent in Q3 -> closed, reopened Q4
    assert all(r.predicate["type"] == "max_units_per_line" for r in rules)     # no MAI column


def test_mue_mai_honoured_when_present():
    q = pd.DataFrame({"HCPCS/CPT Code": ["X", "Y"], "MUE Values": [2, 3], "MUE Adjudication Indicator": [1, 2]})
    rules = ing.mue_history([(meta("dme", "mue", 4, 2026), q)])
    t = {r.predicate["code"]: r.predicate["type"] for r in rules}
    assert t == {"X": "max_units_per_line", "Y": "max_units_per_dos"}


# ------------------------------------------------------- normalize + ids
def test_normalize_fixes_id_collisions_and_planes():
    from schema import Rule
    a = Rule(origin="manual", plane="coverage_scope", binding_status="state_policy", statement="rule one",
             predicate={"type": "not_checkable", "reason": "no_claim_field"}, codes=[], source_doc="Ch10",
             machine_checkable=False, not_checkable_reason="no_claim_field").assign_id()
    b = Rule(origin="manual", plane="coverage_scope", binding_status="state_policy", statement="rule two",
             predicate={"type": "not_checkable", "reason": "requires_medical_record"}, codes=[], source_doc="Ch10",
             machine_checkable=False, not_checkable_reason="requires_medical_record").assign_id()
    assert a.rule_id != b.rule_id                       # content-derived ids: no collision
    enf = {"origin": "ncci", "plane": "enforcement", "statement": "x", "predicate": {"type": "edit_failure"}}
    rows, rep = rn.normalize_rules([a.to_dict(), b.to_dict(), enf, a.to_dict()])
    assert len(rows) == 2 and len({r["rule_id"] for r in rows}) == 2
    assert rep["enforcement_rows_dropped"] == 1 and rep["duplicate_content_collapsed"] == 1
    assert rep["rule_ids_rederived"] == 0                # schema and normalize agree
    planes = {r["statement"]: r["plane"] for r in rows}
    assert planes == {"rule one": "coverage_scope", "rule two": "medical_necessity"}


# ------------------------------------------------------------ compilers
def _rule(pred, eff=None, end=None, mc=True, origin="ncci"):
    return {"rule_id": "T", "origin": origin, "statement": "s", "predicate": pred,
            "effective_date": eff, "end_date": end, "machine_checkable": mc}


def test_ptp_indicator0_has_no_bypass_clause_and_windows():
    d = dg.build_detectors([_rule({"type": "code_pair_prohibited", "code_a": "90791", "code_b": "90832",
                                   "modifier_indicator": "0", "service_category": "practitioner"},
                                  eff="2021-01-01", end="2024-12-31")])[0]
    assert d["compiled"] and d["finding_severity"] == "hard_denial"
    assert "PROC_MOD" not in d["sql"]
    assert "<= DATE '2024-12-31'" in d["sql"] and ">= DATE '2021-01-01'" in d["sql"]
    assert "FORM_TYP IN ('A')" in d["sql"] and d["routing"] == "form:A"


def test_ptp_indicator1_uses_full_bypass_set():
    d = dg.build_detectors([_rule({"type": "code_pair_prohibited", "code_a": "1", "code_b": "2",
                                   "modifier_indicator": "1", "service_category": "outpatient"})])[0]
    assert d["finding_severity"] == "unbypassed_pair"
    for m in ("'LT'", "'RT'", "'F1'", "'59'", "'XU'", "'25'"):
        assert m in d["sql"]
    assert "AND NOT (" in d["sql"]


def test_state_pair_is_unrouted_by_design_and_dme_is_flagged():
    st = dg.build_detectors([_rule({"type": "code_pair_prohibited", "code_a": "H2025", "code_b": "H2026",
                                    "modifier_indicator": "0", "service_category": "state_bh"}, origin="manual")])[0]
    assert st["routing"] == "all_forms" and "FORM_TYP" not in st["sql"]
    dme = dg.build_detectors([_rule({"type": "max_units_per_line", "code": "E0601", "threshold": 1,
                                     "service_category": "dme"})])[0]
    assert dme["compiled"] and dme["routing"] == "UNROUTED:dme"


def test_mue_line_vs_dos_vs_state_day():
    line = dg.build_detectors([_rule({"type": "max_units_per_line", "code": "A", "threshold": 4, "service_category": "practitioner"})])[0]
    dos = dg.build_detectors([_rule({"type": "max_units_per_dos", "code": "A", "threshold": 4, "mai": 2, "service_category": "practitioner"})])[0]
    day = dg.build_detectors([_rule({"type": "max_units_per_day", "code": "S5150", "threshold": 48}, origin="manual")])[0]
    assert "GROUP BY" not in line["sql"] and "QUANTITY_PAID > 4" in line["sql"]
    assert "GROUP BY a.MEMBER_KEY, a.SProv_ID, a.Svc_Begin_Dt" in dos["sql"]
    assert "GROUP BY a.MEMBER_KEY, a.Svc_Begin_Dt" in day["sql"] and "SProv_ID" not in day["sql"]


def test_period_and_not_covered_and_leads():
    per = dg.build_detectors([_rule({"type": "max_units_per_period", "code_set": ["98960", "98961", "98962"],
                                     "threshold": 24, "period": "month"}, origin="manual")])[0]
    assert "DATE_TRUNC('MONTH'" in per["sql"] and "IN ('98960','98961','98962')" in per["sql"]
    nc = dg.build_detectors([_rule({"type": "code_not_covered", "code": "11975"}, origin="manual")])[0]
    assert nc["finding_severity"] == "not_covered_paid"
    lead = dg.build_detectors([_rule({"type": "not_checkable", "reason": "requires_medical_record"}, mc=False, origin="manual")])[0]
    assert not lead["compiled"] and lead["reason"] == "requires_medical_record"
    unk = dg.build_detectors([_rule({"type": "frequency_limit"}, origin="manual")])[0]
    assert not unk["compiled"] and "not in grammar" in unk["reason"]
    bad = dg.build_detectors([_rule({"type": "code_pair_prohibited", "code_a": "1; DROP", "code_b": "2"})])[0]
    assert not bad["compiled"] and "compile error" in bad["reason"]


# ---------------------------------------------------------- state rows
def test_state_row_roundtrip():
    from schema import Rule
    r = Rule(origin="manual", plane="coverage_scope", binding_status="state_policy", statement="s",
             predicate={"type": "max_units_per_day", "code": "S5150", "threshold": 48}, codes=["S5150"],
             source_doc="Ch19", source_locator="p30", alternates=[{"reading": "x"}]).assign_id()
    row = sr.rule_to_row(r, chapter="19", review_status="legacy_hand_authored")
    assert set(sr.STATE_RULE_COLUMNS) <= set(row)
    back = sr.row_to_rule(row)
    assert back.predicate == r.predicate and back.codes == ["S5150"] and back.alternates == [{"reading": "x"}]


# ------------------------------------------------------------------ gate
def _mk(**kw):
    base = dict(origin="manual", plane="coverage_scope", binding_status="state_policy", statement="s",
                predicate={"type": "max_units_per_day", "code": "s5150", "threshold": 48}, codes=["s5150"],
                source_doc="Ch19", required_claim_fields=["proc_cd", "quantity_paid", "member_id", "srvc_bgn_dt"])
    base.update(kw)
    return Rule(**base).assign_id()


def test_gate_statuses():
    clean      = _mk(codes=["S5150"])
    patched    = _mk(codes=[])                                             # derived from predicate
    downgraded = _mk(required_claim_fields=["proc_cd", "member_dob"])      # not in extract
    unknown    = _mk(predicate={"type": "made_up"})
    ambiguous  = _mk(ambiguity_flag=True, ambiguity_note="two readings")
    authored   = _mk(predicate={"type": "not_checkable", "reason": "requires_medical_record"}, machine_checkable=False)
    selfpair   = _mk(predicate={"type": "code_pair_prohibited", "code_a": "A", "code_b": "A"}, codes=["A"])
    rejected   = _mk(statement="")
    acc, rej = vg.gate([clean, patched, downgraded, unknown, ambiguous, authored, selfpair, rejected])
    assert len(rej) == 1 and "empty statement" in rej[0]["reject_reason"]
    st = [d["validation_status"] for d in acc]
    assert st == ["clean", "patched", "downgraded", "downgraded", "needs_review", "needs_review", "downgraded"]
    assert acc[1]["codes"] == ["S5150"]
    assert acc[2]["not_checkable_reason"] == "missing_claim_field" and acc[2]["machine_checkable"] is False
    assert acc[3]["not_checkable_reason"] == "predicate_not_in_grammar"
    assert acc[5]["not_checkable_reason"] == "requires_medical_record"
    # gate output feeds the compiler: downgraded rules become leads, clean ones compile
    dets = dg.build_detectors(acc)
    assert [d["compiled"] for d in dets] == [True, True, False, False, True, False, False]


def test_authorable_but_uncompiled_predicate_becomes_lead_not_error():
    r = _mk(predicate={"type": "max_dollars_per_period", "threshold": 1000, "period": "benefit_year"}, codes=[])
    acc, _ = vg.gate([r])
    assert acc[0]["validation_status"] == "downgraded"
    d = dg.build_detectors(acc)[0]
    assert not d["compiled"] and d["reason"] == "predicate_not_in_grammar"


# ------------------------------------------------------------------ kb_io
def test_kb_io_explicit_schema_handles_all_none_and_nullable_ints():
    import types as _t
    # stand-in for pyspark.sql.types: only the names kb_io uses
    T = _t.SimpleNamespace(
        BooleanType=lambda: "boolean", LongType=lambda: "long", DoubleType=lambda: "double",
        StringType=lambda: "string",
        StructField=lambda n, t, nullable: (n, t),
        StructType=lambda fields: list(fields))
    sys.modules["pyspark"] = _t.ModuleType("pyspark")
    sys.modules["pyspark.sql"] = _t.ModuleType("pyspark.sql")
    sys.modules["pyspark.sql.types"] = T
    import kb_io; import importlib; importlib.reload(kb_io)

    captured = {}
    class FakeSpark:
        def createDataFrame(self, rows, schema=None):
            captured["rows"], captured["schema"] = rows, schema
            return "df"
    pdf = pd.DataFrame({
        "flag": [True, False],
        "hits": pd.array([3, None], dtype="Int64"),
        "secs": [1.5, None],
        "err": [None, None],                      # all-None column: the inference killer
        "pred": [{"type": "x"}, ["a"]],           # nested -> JSON string
        "name": ["a", None],
    })
    kb_io.to_spark(FakeSpark(), pdf)
    assert captured["schema"] == [("flag", "boolean"), ("hits", "long"), ("secs", "double"),
                                  ("err", "string"), ("pred", "string"), ("name", "string")]
    assert captured["rows"][0] == [True, 3, 1.5, None, '{"type": "x"}', "a"]
    assert captured["rows"][1] == [False, None, None, None, '["a"]', None]


# ------------------------------------------------ real column names (SB's tables, 2026-09-18)
def test_real_ptp_and_mue_column_names_and_types():
    import numpy as np
    # exactly as main.sedo.medicaid_ncci_edit_outpatient_ptp_q2_2026 presents them:
    # ints for dates/indicator; NULL deletion dates surface as NaN (float) in pandas
    ptp = pd.DataFrame({
        "Col1": ["0001A", "0001A", "0001A"],
        "Col2": ["0591T", "90473", "99202"],
        "EffDt": [20220101, 20220101, 20220101.0],
        "DelDt": [20231231, 20220101, np.nan],
        "ModifierIndicator_0_not_allowed_1_allowed_9_not_applicable": [1, 9, 0.0],
        "PTP_Edit_Rationale": ["CPT Manual or CMS manual coding instruction"] * 3,
    })
    rules = ing.ptp_rules(ptp, meta("outpatient", "ptp", 2, 2026))
    assert [(r.predicate["code_b"], r.predicate["modifier_indicator"], r.effective_date, r.end_date) for r in rules] == [
        ("0591T", "1", "2022-01-01", "2023-12-31"),     # deleted edit kept, windowed
        ("99202", "0", "2022-01-01", None),             # NULL DelDt -> still active
    ]                                                   # indicator 9 dropped
    assert "CMS rationale: CPT Manual" in rules[0].notes
    for svc, col in (("outpatient", "Outpatient Hospital Services MUE Values"),
                     ("practitioner", "Practitioner Services MUE Values"),
                     ("dme", "DME Supplier Services MUE Values")):
        mue = pd.DataFrame({"HCPCS/CPT Code": ["A4218"], col: [20], "MUE Rationale": ["CMS Policy"]})
        r = ing.mue_history([(meta(svc, "mue", 3, 2026), mue)])[0]
        assert r.predicate == {"type": "max_units_per_line", "code": "A4218", "threshold": 20, "service_category": svc}
    # Spark-side scoping resolves the same names
    assert ing._find_col(ptp.columns, "column1", "col1") == "Col1"
    assert ing._find_col(mue.columns, "hcpcs/cpt code") == "HCPCS/CPT Code"

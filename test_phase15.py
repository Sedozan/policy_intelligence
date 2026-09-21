"""Phase 1.5 checks: the ChatGPT review-#2 defects, each pinned by a test that
EXECUTES the compiled SQL (SQLite) wherever a compiler is involved. String
inspection is how the LT+LT bypass slipped through; execution is the bar now.
Run: cd pie_mvp && python -m pytest tests -q
"""
import os, sys, json, sqlite3, copy
import pandas as pd
import pytest

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.dirname(HERE))

import config
import detector_gen as dg
import extract_rules as ex
import ingest_ncci_tables as ing
import state_rules as sr
import validate_gate as vg
import state_seed
from schema import Rule


# ------------------------------------------------------------------ helpers
def _rule(pred, origin="manual", **kw):
    return Rule(origin=origin, plane="coverage_scope", binding_status="state_policy",
                statement="t", source_doc="t", predicate=pred, **kw)


_COLS = ("ClaimID TEXT, LN_NO INT, MEMBER_KEY TEXT, SProv_ID TEXT, Svc_Begin_Dt TEXT, PROC_CD TEXT, "
         "QUANTITY_PAID INT, FORM_TYP TEXT, PROC_MOD_1 TEXT, PROC_MOD_2 TEXT, PROC_MOD_3 TEXT, "
         "PROC_MOD_4 TEXT, EVENT_PA_NO TEXT")


@pytest.fixture
def claims(monkeypatch):
    db = sqlite3.connect(":memory:")
    db.execute(f"CREATE TABLE claims ({_COLS})")
    monkeypatch.setattr(dg, "CLAIMS", "claims")

    def load(rows):
        db.execute("DELETE FROM claims")
        db.executemany("INSERT INTO claims VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", rows)

    def run(rule_dict):
        d = dg.build_detectors([rule_dict])[0]
        assert d["compiled"], d["reason"]
        cur = db.execute(d["sql"])
        cols = [c[0] for c in cur.description]
        return [dict(zip(cols, r)) for r in cur.fetchall()]

    return load, run


def _line(cid, code, prov="p", m1="", m2="", pa=None, units=1):
    return (cid, 1, "m", prov, "2026-01-01", code, units, "A", m1, m2, "", "", pa)


def _pair(mi="1", origin="ncci", **extra):
    p = {"type": "code_pair_prohibited", "code_a": "11111", "code_b": "22222",
         "modifier_indicator": mi, "service_category": "practitioner", **extra}
    return _rule(p, origin=origin).to_dict()


# ===================================================== F04 / F05 / F10: PTP
def test_ptp_same_anatomic_modifier_is_not_a_bypass(claims):
    """CMS: same anatomic modifier on both lines and no 58/59/78/79/XE/XP/XS/XU -> edit applies."""
    load, run = claims
    load([_line("c1", "11111", m1="LT"), _line("c2", "22222", m1="LT")])
    assert len(run(_pair())) == 1                                   # LT+LT still flags
    load([_line("c1", "11111", m1="LT"), _line("c2", "22222", m1="LT", m2="59")])
    assert run(_pair()) == []                                       # 59 overrides the exception
    load([_line("c1", "11111", m1="LT", m2="XS"), _line("c2", "22222", m1="LT")])
    assert run(_pair()) == []                                       # override on either line
    load([_line("c1", "11111", m1="LT"), _line("c2", "22222", m1="RT")])
    assert run(_pair()) == []                                       # different sites -> bypass
    load([_line("c1", "11111", m1="LT"), _line("c2", "22222")])
    assert run(_pair()) == []                                       # modifier on one line -> bypass
    load([_line("c1", "11111"), _line("c2", "22222")])
    assert len(run(_pair())) == 1                                   # no modifier -> flag


def test_ptp_finding_targets_column_two_and_keeps_column_one_evidence(claims):
    load, run = claims
    load([_line("col1", "11111"), _line("col2", "22222")])
    rows = run(_pair(mi="0"))
    assert len(rows) == 1
    r = rows[0]
    assert r["claim_id"] == "col2" and r["col1_claim_id"] == "col1"   # CMS denies Column Two
    assert r["finding_severity"] == "hard_denial" and r["pair_scope"] == "member_provider_day"


def test_pair_scope_state_member_day_vs_ncci_provider(claims):
    load, run = claims
    load([_line("c1", "11111", prov="doctor_A"), _line("c2", "22222", prov="doctor_B")])
    assert run(_pair(mi="0", origin="ncci")) == []                  # NCCI: same provider only
    assert len(run(_pair(mi="0", origin="manual"))) == 1            # state: same member is enough
    assert len(run(_pair(mi="0", origin="ncci", scope="member_day"))) == 1   # explicit scope wins
    assert run(_pair(mi="0", origin="manual", scope="member_provider_day")) == []
    # the seed's Ch10 pair uses service_category='practitioner' (a routing label) and
    # must still be member-scoped: service_category never decides scope
    assert dg.pair_scope(_pair(mi="0", origin="manual")) == "member_day"
    assert "SProv_ID = b.SProv_ID" not in dg.build_detectors([_pair(mi="0", origin="manual")])[0]["sql"]


# =========================================================== F02 + PA: seed
def test_seed_chw_has_all_three_pairs_and_pa_note():
    rules = state_seed.seed_rules()
    chw_pairs = {tuple(sorted((r.predicate["code_a"], r.predicate["code_b"])))
                 for r in rules if r.predicate.get("type") == "code_pair_prohibited"
                 and r.predicate["code_a"].startswith("9896")}
    assert chw_pairs == {("98960", "98961"), ("98960", "98962"), ("98961", "98962")}
    assert all(r.predicate.get("scope") == "member_day" for r in rules
               if r.predicate.get("type") == "code_pair_prohibited")
    caps = [r for r in rules if r.predicate.get("type") in ("max_units_per_day", "max_units_per_period")
            and "9896" in "".join(r.codes)]
    assert caps and all("prior authorization" in r.notes for r in caps)


def test_pa_present_downgrades_state_findings_to_verify_pa_never_ncci(claims):
    load, run = claims
    load([_line("with_pa", "11975", pa="PA777"), _line("no_pa", "11975", pa=None),
          _line("blank_pa", "11975", pa="   ")])
    rows = {r["claim_id"]: (r["pa_present"], r["finding_severity"])
            for r in run(_rule({"type": "code_not_covered", "code": "11975"}).to_dict())}
    assert rows == {"with_pa": (1, "verify_pa"), "no_pa": (0, "not_covered_paid"),
                    "blank_pa": (0, "not_covered_paid")}
    # NCCI edits: CMS says PA is not a bypass -> PA is ignored entirely
    ncci = _rule({"type": "max_units_per_line", "code": "11975", "threshold": 0,
                  "service_category": "practitioner"}, origin="ncci").to_dict()
    assert {r["finding_severity"] for r in run(ncci)} == {"units_over_limit"}
    assert all(r["pa_present"] == 0 for r in run(ncci))


def test_aggregate_detectors_carry_contributing_line_ids_and_pa():
    day = dg.build_detectors([_rule({"type": "max_units_per_day", "code": "98960", "threshold": 4}).to_dict()])[0]["sql"]
    per = dg.build_detectors([_rule({"type": "max_units_per_period", "code_set": ["98960"],
                                     "threshold": 24, "period": "month"}).to_dict()])[0]["sql"]
    for sql in (day, per):
        assert "COLLECT_LIST(CONCAT(a.ClaimID, ':', a.LN_NO)) " in sql or "evidence_lines" in sql
        assert "MAX(CASE WHEN COALESCE(TRIM(a.EVENT_PA_NO), '') <> '' THEN 1 ELSE 0 END)" in sql
        assert "'verify_pa'" in sql


# ================================================ F06 / F12 / F14 / F11: extraction
def test_threshold_must_be_integral():
    ok, _ = ex.predicate_valid({"type": "max_units_per_day", "code": "98960", "threshold": 4.9})
    assert not ok
    assert ex.predicate_valid({"type": "max_units_per_day", "code": "98960", "threshold": 4.0})[0]
    assert ex.predicate_valid({"type": "max_units_per_day", "code": "98960", "threshold": "4"})[0]
    assert not ex.predicate_valid({"type": "max_units_per_day", "code": "98960", "threshold": True})[0]


class _Fixed(ex.Backend):
    name, model = "FAKE", "fake"

    def __init__(self, rules, raw=None):
        self.rules, self.raw, self.calls = rules, raw, 0

    def complete(self, s, u):
        self.calls += 1
        if self.raw is not None:
            return self.raw
        return json.dumps({"rules": copy.deepcopy(self.rules)})


def test_hybrid_key_distinguishes_period_and_bumps_only_across_sources():
    a = {"statement": "monthly", "predicate": {"type": "max_units_per_period", "code_set": ["98960"],
         "threshold": 4, "period": "month"}, "codes": ["98960"], "verbatim_quote": "four monthly", "confidence": 0.6}
    b = copy.deepcopy(a); b["predicate"]["period"] = "year"
    out = json.loads(ex.HybridBackend(_Fixed([b]), _Fixed([a])).complete("", ""))["rules"]
    assert len(out) == 2 and {r["source"] for r in out} == {"pattern", "model"}    # not corroborated
    # a genuine agreement still corroborates
    out = json.loads(ex.HybridBackend(_Fixed([a]), _Fixed([a])).complete("", ""))["rules"]
    assert len(out) == 1 and out[0]["source"] == "pattern+model" and out[0]["confidence"] == 0.7
    # two copies from the SAME backend are not evidence: no bump, source unchanged
    out = json.loads(ex.HybridBackend(_Fixed([a, a]), _Fixed([])).complete("", ""))["rules"]
    assert len(out) == 1 and out[0]["source"] == "model" and out[0]["confidence"] == 0.6


def _section():
    return {"text": "Code 98960 is covered.", "text_hash": "x", "section_id": "s", "heading": "H",
            "page_start": 1, "page_end": 1, "has_rule_signal": True}


def test_failed_or_degraded_extraction_is_not_cached():
    inv = {"chapter": "10", "title": "T", "url": "u", "doc_hash": "h"}
    # malformed model output: not cached, retried next run
    b = _Fixed([], raw="not json"); c = {}
    ex.extract_chapter([_section()], inv, b, "r1", cache=c, log_fn=lambda x: None)
    _, _, st = ex.extract_chapter([_section()], inv, b, "r2", cache=c, log_fn=lambda x: None)
    assert b.calls == 2 and c == {} and st["unparseable"] == 1 and st["sections_from_cache"] == 0
    assert st["retryable_sections"][0]["why"].startswith("unparseable")

    # backend exception: a retryable error, not a crash and not a cache entry
    class Boom(_Fixed):
        def complete(self, s, u):
            self.calls += 1
            raise TimeoutError("endpoint")
    b = Boom([]); c = {}
    drafts, _, st = ex.extract_chapter([_section()], inv, b, "r1", cache=c, log_fn=lambda x: None)
    assert drafts == [] and c == {} and st["retryable_sections"][0]["why"].startswith("backend error")

    # HYBRID with a dead model half: patterns-only result must not be cached as hybrid
    hy = ex.HybridBackend(Boom([]), ex.StubBackend()); c = {}
    ex.extract_chapter([_section()], inv, hy, "r1", cache=c, log_fn=lambda x: None)
    assert c == {}


def test_fuzzy_grounding_rejects_negation_and_number_drift():
    sec = {"text": "Code 98960 is covered for eligible members."}
    g = ex.ground({"predicate": {"type": "code_not_covered", "code": "98960"}, "codes": ["98960"],
                   "verbatim_quote": "Code 98960 is not covered for eligible members."}, sec)
    assert not g["keep"] and g["quote_found"] == "drift" and "negation" in g["drop_reason"]
    sec = {"text": "A maximum of four units per day applies to 98960."}
    g = ex.ground({"predicate": {"type": "max_units_per_day", "code": "98960", "threshold": 4}, "codes": ["98960"],
                   "verbatim_quote": "A maximum of 4 units per week applies to 98960."}, sec)
    assert not g["keep"] and g["quote_found"] == "drift"
    # harmless drift (punctuation / ligature / whitespace) is still accepted as fuzzy
    g = ex.ground({"predicate": {"type": "max_units_per_day", "code": "98960", "threshold": 4}, "codes": ["98960"],
                   "verbatim_quote": "A maximum of four units per day applies to 98960"}, sec)
    assert g["keep"] and g["quote_found"] in ("exact", "fuzzy")


def test_pattern_conditional_noncoverage_becomes_lead_not_rule():
    out = ex.StubBackend()._patterns("Do not bill 98960 unless prior authorization is obtained.")
    assert len(out) == 1 and out[0]["predicate"]["type"] == "not_checkable" and out[0]["ambiguous"]
    assert out[0]["codes"] == ["98960"]
    out = ex.StubBackend()._patterns("Do not bill 98960 on a FFS claim.")
    assert out[0]["predicate"] == {"type": "code_not_covered", "code": "98960"}


def test_semantic_key_separates_population_date_and_uncoded_obligations():
    p = {"type": "max_units_per_day", "code": "98960", "threshold": 4}
    base = ex.semantic_key("10", p, ["98960"])
    assert ex.semantic_key("10", p, ["98960"], population="all") == base
    assert ex.semantic_key("10", p, ["98960"], population="members under 21") != base
    assert ex.semantic_key("10", p, ["98960"], effective_date="2024-04-01") != base
    lead = {"type": "not_checkable", "reason": "requires_medical_record"}
    assert ex.semantic_key("10", lead, [], quote="Documentation A is required.") != \
           ex.semantic_key("10", lead, [], quote="Documentation B is required.")
    # end to end: two different uncoded obligations survive the within-run dedup
    inv = {"chapter": "10", "title": "T", "url": "u", "doc_hash": "h", "revision_date": None}
    sec = {"heading": "H", "page_start": 1, "page_end": 1, "section_id": "s",
           "text": "Documentation A is required. Documentation B is required."}
    rows = []
    for q in ("Documentation A is required.", "Documentation B is required."):
        pr = {"statement": q, "predicate": lead, "codes": [], "verbatim_quote": q, "confidence": 0.6}
        rows.append(ex.to_draft_row(pr, ex.ground(pr, sec), sec, inv, _Fixed([]), "r"))
    assert rows[0]["extraction_key"] != rows[1]["extraction_key"]


# ================================================ F01 / F07 / F15: gate, review, MUE
def test_gate_downgrades_population_scoped_rule_to_lead():
    r = _rule({"type": "max_units_per_day", "code": "98960", "threshold": 4}, codes=["98960"],
              population="members under 21")
    acc, _ = vg.gate([r])
    assert acc[0]["machine_checkable"] is False
    assert acc[0]["not_checkable_reason"] == "requires_member_attribute"
    assert acc[0]["population"] == "members under 21"          # preserved for the reviewer
    assert not dg.build_detectors(acc)[0]["compiled"]
    ok, _ = vg.gate([_rule({"type": "max_units_per_day", "code": "98960", "threshold": 4}, codes=["98960"])])
    assert ok[0]["machine_checkable"] is True


class _Frame:
    def __init__(self, rows): self.rows = rows
    def toPandas(self): return pd.DataFrame(self.rows).copy()


class _Spark:
    def __init__(self, rows): self.rows = rows
    def table(self, t): return _Frame(self.rows)


def _review(monkeypatch, workbook_rows, table_rows):
    captured, logged = {}, {}
    monkeypatch.setattr(sr, "_xlsx_read", lambda p: pd.DataFrame(workbook_rows))
    monkeypatch.setattr(sr, "write_state_rules", lambda s, t, rows: captured.update(rows={r["rule_id"]: r for r in rows}))
    import kb_io
    monkeypatch.setattr(kb_io, "write_table", lambda spark, df, fqn, mode="overwrite": logged.update(df=df) or 0)
    sr.ingest_review_decisions(_Spark(table_rows), "t", "x", "l")
    return captured["rows"], logged["df"]


def test_review_fix_requires_reviewer_and_rederives_codes(monkeypatch):
    orig = sr.rule_to_row(_rule({"type": "max_units_per_day", "code": "98960", "threshold": 4},
                                codes=["98960"]).assign_id(), "10")
    orig["doc_hash"] = "abc"
    fix = {"rule_id": orig["rule_id"], "decision": "fix", "doc_hash": "abc",
           "corrected_predicate": json.dumps({"type": "max_units_per_day", "code": "98961", "threshold": 4})}
    # unsigned -> not applied, parked
    rows, log = _review(monkeypatch, [dict(fix, reviewer=None)], [orig])
    r = rows[orig["rule_id"]]
    assert r["review_status"] == "needs_review" and "reviewer name required" in r["review_note"]
    assert json.loads(r["codes"]) == ["98960"] and list(log["applied_outcome"]) == ["no_reviewer"]
    # signed -> applied, and everything derived from the predicate follows it
    rows, log = _review(monkeypatch, [dict(fix, reviewer="sb")], [orig])
    r = rows[orig["rule_id"]]
    assert r["review_status"] == "approved" and r["reviewer"] == "sb"
    assert json.loads(r["predicate"])["code"] == "98961" and json.loads(r["codes"]) == ["98961"]
    assert list(log["applied_outcome"]) == ["fixed"]


def test_review_decision_on_stale_document_version_is_parked(monkeypatch):
    orig = sr.rule_to_row(_rule({"type": "max_units_per_day", "code": "98960", "threshold": 4},
                                codes=["98960"]).assign_id(), "10")
    orig["doc_hash"] = "NEWDOC"
    rows, log = _review(monkeypatch, [{"rule_id": orig["rule_id"], "decision": "approve",
                                       "reviewer": "sb", "doc_hash": "OLDDOC"}], [orig])
    r = rows[orig["rule_id"]]
    assert r["review_status"] == "needs_review" and "re-review" in r["review_note"]
    assert list(log["applied_outcome"]) == ["stale_version"]


def _meta(q, y):
    return {"table": f"medicaid_ncci_edit_practitioner_mue_q{q}_{y}", "fqn": "x", "service": "practitioner",
            "edit_type": "mue", "quarter": q, "year": y, "label": f"{y}Q{q}", "starts": f"{y}-{ing.QUARTER_START[q]}"}


def test_mue_quarter_gap_is_refused_or_flagged():
    q = lambda v: pd.DataFrame({"HCPCS/CPT Code": ["A"], "Practitioner Services MUE Values": [v], "MUE Rationale": [""]})
    frames = [(_meta(1, 2026), q(4)), (_meta(3, 2026), q(4))]          # Q2 missing
    with pytest.raises(ValueError, match="2026-04-01"):
        ing.mue_history(frames)
    rules = ing.mue_history(frames, allow_gaps=True)
    assert len(rules) == 1 and rules[0].ambiguity_flag and "2026-04-01" in rules[0].ambiguity_note
    assert ing.mue_history([(_meta(1, 2026), q(4)), (_meta(2, 2026), q(4))])[0].ambiguity_flag is False


# ============================================================ chapter profiler
def test_chapter_profile_classifies_fixture_and_synthetic_profiles():
    import chapter_profile as cp, policy_docs as pdx
    sys.path.insert(0, HERE)
    import fixture_pdf
    path = fixture_pdf.build(os.path.join(os.path.dirname(HERE), "_profile_fixture.pdf"))
    try:
        pages = pdx.parse_pdf(path)
        secs, meta = pdx.segment(pages, "10", "h")
        inv = {"chapter": "10", "title": "T", "url": "u", "doc_hash": "h", "revision_date": meta["revision_date"]}
        drafts, dropped, stats = ex.extract_chapter(secs, inv, ex.StubBackend(), "p", cache={}, log_fn=lambda x: None)
        r = cp.profile_chapter("10", "T", pages, secs, meta, drafts, dropped, stats)
    finally:
        os.remove(path)
    assert r["profile"] == "code_rich" and r["stub_compiled_types"] >= 5 and r["codes_pipeline"] >= 8

    def fake(text, pages=10, blank=0):
        pg = [{"page": i + 1, "lines": [{"text": text if i >= blank else "", "size": 10, "bold": False}], "tables": []}
              for i in range(pages)]
        secs = [{"text": text, "has_rule_signal": True}] if text else []
        return cp.profile_chapter("x", "x", pg, secs, {"headers_stripped": []}, [], [], {})["profile"]
    assert fake("DME cannot be billed without an appropriate modifier. Modifier RR, NU, UE, RA, RB, LL, KX. "
                "Modifiers are required. The modifier must match. " * 2) == "modifier_driven"
    assert fake("Reimbursed per diem. The per diem includes supplies. Included in the per diem, not billed separately. "
                "Not covered separately. Maximum one unit per day. Prior authorization required. " * 2) == "category_driven"
    assert fake("Revenue code 0450 emergency. Revenue code 0250 pharmacy. Revenue code 0300 lab. APR-DRG payment. "
                "DRG weight. NDC 12345-6789-01 required. NDC must be reported. " * 2) == "foreign_codes"
    assert fake("Submit claims within 6 months. Disputes in writing. The remittance advice lists claims. " * 3) == "process"
    assert fake("", blank=10) == "scanned"


# ================================================ model backend: never fail silently again
class _FakeChoice:
    def __init__(self, content, finish="stop"):
        self.message = type("M", (), {"content": content})()
        self.finish_reason = finish


class _FakeOpenAI:
    """Looks like openai.OpenAI().chat.completions; records calls."""
    def __init__(self, content='{"rules": [], "no_rule_reason": "none"}', finish="stop", reject_json_mode=False):
        self.calls, self.content, self.finish, self.reject = [], content, finish, reject_json_mode
        outer = self

        class _Completions:
            def create(self, **kw):
                outer.calls.append(kw)
                if outer.reject and "response_format" in kw:
                    raise ValueError("response_format not supported")
                return type("R", (), {"choices": [_FakeChoice(outer.content, outer.finish)]})()
        self.chat = type("C", (), {"completions": _Completions()})()


def _fake_sdk(client):
    class _SE:
        def get_open_ai_client(self): return client
    class _WC:
        def __init__(self): self.serving_endpoints = _SE()
    return types.SimpleNamespace(WorkspaceClient=_WC)


import types


def test_dbfm_reports_every_failed_route_and_never_keyerror(monkeypatch):
    monkeypatch.delenv("DATABRICKS_HOST", raising=False)
    monkeypatch.delenv("DATABRICKS_TOKEN", raising=False)
    for mod in ("databricks.sdk", "databricks", "mlflow.deployments", "mlflow", "openai", "IPython"):
        monkeypatch.setitem(sys.modules, mod, None)          # import -> ImportError
    b = ex.DBFMBackend("databricks-gpt-oss-120b")
    with pytest.raises(ex.ModelUnavailable) as ei:
        b.complete("s", "u")
    msg = str(ei.value)
    assert "databricks-sdk:" in msg and "mlflow.deployments:" in msg and "openai+workspace:" in msg
    assert "KeyError" not in msg and len(b.client_errors) == 3
    ok, detail = b.healthcheck()
    assert not ok and "no client route works" in detail


def test_dbfm_uses_sdk_route_and_surfaces_truncation(monkeypatch):
    fake = _FakeOpenAI()
    monkeypatch.setitem(sys.modules, "databricks.sdk", _fake_sdk(fake))
    monkeypatch.setitem(sys.modules, "databricks", types.SimpleNamespace(sdk=sys.modules["databricks.sdk"]))
    b = ex.DBFMBackend("ep")
    out = b.complete("sys", "usr")
    assert out.startswith('{"rules"') and b.client_route == "databricks-sdk openai client"
    assert fake.calls[0]["model"] == "ep" and fake.calls[0]["response_format"] == {"type": "json_object"}
    ok, detail = b.healthcheck()
    assert ok and "reply:" in detail
    # endpoint rejects json_object -> plain retry, error kept for the record, still succeeds
    fake2 = _FakeOpenAI(reject_json_mode=True)
    monkeypatch.setitem(sys.modules, "databricks.sdk", _fake_sdk(fake2))
    b2 = ex.DBFMBackend("ep")
    assert b2.complete("s", "u").startswith('{"rules"') and "json_object mode rejected" in b2.last_error
    assert len(fake2.calls) == 2 and "response_format" not in fake2.calls[1]
    # truncated output is a loud failure, not "no rules"
    fake3 = _FakeOpenAI(content='{"rules": [{"stat', finish="length")
    monkeypatch.setitem(sys.modules, "databricks.sdk", _fake_sdk(fake3))
    with pytest.raises(ValueError, match="truncated"):
        ex.DBFMBackend("ep").complete("s", "u")


def test_healthcheck_and_run_stats_expose_a_dead_model():
    class Boom(ex.Backend):
        name, model = "BOOM", "x"
        def complete(self, s, u): raise RuntimeError("endpoint down")
    ok, detail = ex.healthcheck(ex.StubBackend())
    assert ok and "patterns only" in detail
    hy = ex.HybridBackend(Boom(), ex.StubBackend())
    ok, detail = ex.healthcheck(hy)
    assert not ok and "endpoint down" in detail
    inv = {"chapter": "10", "title": "T", "url": "u", "doc_hash": "h"}
    sec = {"text": "A maximum of four units per day applies to 98960.", "text_hash": "x", "section_id": "s",
           "heading": "H", "page_start": 1, "page_end": 1, "has_rule_signal": True}
    drafts, _, st = ex.extract_chapter([sec], inv, hy, "r", cache={}, log_fn=lambda x: None)
    # a model failure is a MODEL failure: nothing extracted, nothing cached, counted as such
    assert drafts == [] and st["model_failed"] == 1 and st["unparseable"] == 0
    assert st["model_calls"] == 1 and st["model_failures"] == 1 and "endpoint down" in st["last_model_error"]
    assert st["patterns_only_allowed"] is False
    # opting in to patterns-only is visible in the stats too
    hy2 = ex.HybridBackend(Boom(), ex.StubBackend(), require_model=False)
    drafts, _, st = ex.extract_chapter([sec], inv, hy2, "r", cache={}, log_fn=lambda x: None)
    assert len(drafts) == 1 and st["degraded_patterns_only"] == 1 and st["patterns_only_allowed"] is True


# ============================================ HFLOCAL: mirror of generate_with_harmony
class _Tensor:
    """Minimal stand-in for a torch tensor of token ids: shape, slicing, .to(), .tolist()."""
    def __init__(self, ids): self.ids = list(ids)
    @property
    def shape(self): return (1, len(self.ids))
    def to(self, device): return self
    def __getitem__(self, k):
        if isinstance(k, int):
            return self
        return _Tensor(self.ids[k])
    def tolist(self): return list(self.ids)


class _Msg:
    def __init__(self, channel, text): self.channel, self.content = channel, [types.SimpleNamespace(text=text)]


class _FakeRuntime:
    """tokenizer + model + harmony_enc that behave like SB's notebook objects.
    `script` is a list of (analysis_text, final_text | None, n_tokens) per generate call."""
    def __init__(self, script):
        self.script, self.calls = list(script), []
        outer = self

        class _Tok:
            def apply_chat_template(self, messages, **kw):
                outer.calls.append({"messages": messages, **kw})
                return {"input_ids": _Tensor(range(10))}
        class _Model:
            device = "cpu"
            def generate(self, input_ids, max_new_tokens, do_sample):
                outer.calls[-1].update(max_new_tokens=max_new_tokens, do_sample=do_sample)
                _, _, n = outer.script[len(outer.calls) - 1]
                return _Tensor(list(range(10)) + list(range(min(n, max_new_tokens))))
        class _Enc:
            def parse_messages_from_completion_tokens(self, ids, role):
                analysis, final, _ = outer.script[len(outer.calls) - 1]
                out = [_Msg("analysis", analysis)]
                if final is not None:
                    out.append(_Msg("final", final))
                return out
        self.tokenizer, self.model, self.enc = _Tok(), _Model(), _Enc()

    def backend(self, **kw):
        return ex.HFLocalBackend("openai/gpt-oss-120b", model=self.model, tokenizer=self.tokenizer,
                                 harmony_enc=self.enc, **kw)


@pytest.fixture
def fake_torch(monkeypatch):
    class _NoGrad:
        def __enter__(self): return self
        def __exit__(self, *a): return False
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(no_grad=_NoGrad))
    monkeypatch.setitem(sys.modules, "openai_harmony", types.SimpleNamespace(Role=types.SimpleNamespace(ASSISTANT="assistant")))
    yield
    ex._HF_RUNTIME.update(model=None, tokenizer=None, harmony_enc=None, model_id=None)


def test_hflocal_mirrors_notebook_loop_and_keeps_only_final_channel(fake_torch):
    rt = _FakeRuntime([("thinking about codes...", '{"rules": [], "no_rule_reason": "none"}', 400),
                       ("ping", '{"ok": true}', 20)])                  # second entry serves the healthcheck
    b = rt.backend(reasoning_level="low")
    out = b.complete("SYS", "USR")
    assert out == '{"rules": [], "no_rule_reason": "none"}'            # analysis text never returned
    call = rt.calls[0]
    assert call["messages"] == [{"role": "system", "content": "SYS"}, {"role": "user", "content": "USR"}]
    assert call["add_generation_prompt"] is True and call["return_dict"] is True and call["return_tensors"] == "pt"
    assert call["reasoning_effort"] == "low" and call["do_sample"] is False
    assert call["max_new_tokens"] == config.REASONING_TOKEN_BUDGET["low"]
    assert b.last_meta["final_channel_present"] and not b.last_meta["hit_budget"] and b.last_meta["retried"] is False
    assert b.calls == 1 and b.failures == 0 and b.retries == 0
    ok, detail = b.healthcheck()
    assert ok and "HFLOCAL" in detail


def test_hflocal_retries_once_at_1_8x_when_truncated_without_final(fake_torch):
    budget = config.REASONING_TOKEN_BUDGET["low"]
    rt = _FakeRuntime([("long reasoning", None, budget),                       # hit budget, no final
                       ("short", '{"rules": []}', 200)])                        # retry succeeds
    b = rt.backend(reasoning_level="low")
    assert b.complete("s", "u") == '{"rules": []}'
    assert len(rt.calls) == 2 and rt.calls[1]["max_new_tokens"] == int(budget * config.RETRY_BUDGET_FACTOR)
    assert b.retries == 1 and b.last_meta["retried"] is True


def test_hflocal_raises_loudly_when_no_final_channel(fake_torch):
    budget = config.REASONING_TOKEN_BUDGET["low"]
    rt = _FakeRuntime([("r", None, budget), ("r", None, int(budget * 1.8))])   # truncated twice
    b = rt.backend(reasoning_level="low")
    with pytest.raises(ValueError, match="no final channel"):
        b.complete("s", "u")
    assert b.failures == 1 and "no final channel" in b.last_error
    # analysis only, budget NOT hit (model just never answered) -> no retry, loud failure
    rt = _FakeRuntime([("r", None, 50)])
    b = rt.backend()
    with pytest.raises(ValueError, match="no final channel"):
        b.complete("s", "u")
    assert len(rt.calls) == 1


def test_hflocal_through_hybrid_extract_chapter_with_checkpoint(fake_torch):
    def good(code):
        return json.dumps({"rules": [{"statement": f"{code}: maximum 4 units per day.",
                                      "predicate": {"type": "max_units_per_day", "code": code, "threshold": 4},
                                      "codes": [code],
                                      "verbatim_quote": f"A maximum of four units per day applies to {code}.",
                                      "confidence": 0.9}]})
    # call 1 = healthcheck, calls 2-3 = the two sections (distinct codes, so distinct rules)
    rt = _FakeRuntime([("a", '{"ok": true}', 20), ("a", good("98960"), 300), ("a", good("98961"), 300)])
    hy = ex.HybridBackend(rt.backend(), ex.StubBackend())
    ok, detail = ex.healthcheck(hy)
    assert ok
    inv = {"chapter": "10", "title": "T", "url": "u", "doc_hash": "h", "revision_date": None}
    secs = [{"text": f"A maximum of four units per day applies to {c}.", "text_hash": f"x{c}", "section_id": f"s{c}",
             "heading": "H", "page_start": 1, "page_end": 1, "has_rule_signal": True} for c in ("98960", "98961")]
    flushed = []
    drafts, dropped, st = ex.extract_chapter(secs, inv, hy, "r", cache={}, log_fn=lambda x: None,
                                             checkpoint_fn=lambda c: flushed.append(len(c)), checkpoint_every=1)
    assert st["model_calls"] == 2 and st["model_failures"] == 0 and st["checkpoints"] == 2 and flushed == [1, 2]
    assert len(drafts) == 2 and all("pattern+model" in d["notes"] for d in drafts)   # both halves agree -> corroborated
    assert st["model_route"].startswith("transformers")


# ================================================ GPU/CPU split: reads never lie, files round-trip
import kb_io
import extract_io


class _SparkRaises:
    def __init__(self, exc): self.exc = exc
    def table(self, fqn): raise self.exc


def test_table_or_none_distinguishes_missing_from_denied():
    class AnalysisException(Exception): pass
    assert kb_io.table_or_none(_SparkRaises(AnalysisException("[TABLE_OR_VIEW_NOT_FOUND] The table `main.x.y` cannot be found")), "main.x.y") is None
    assert kb_io.table_or_none(_SparkRaises(Exception("Table or view not found: main.x.y")), "main.x.y") is None
    denied = Exception("[CLOUD_ACCESS_DENIED] Access denied to n/a cloud path ... 403, GET ... AuthorizationFailure")
    with pytest.raises(kb_io.TableAccessError, match="NOT an empty table"):
        kb_io.table_or_none(_SparkRaises(denied), "main.sedo.state_rules")
    with pytest.raises(kb_io.TableAccessError):
        kb_io.table_or_none(_SparkRaises(TimeoutError("metastore timeout")), "main.sedo.state_rules")


def test_guard_refuses_to_shrink_reviewed_rules():
    existing = pd.DataFrame([{"rule_id": "a", "review_status": "approved"}, {"rule_id": "b", "review_status": "draft"}])
    ok = pd.DataFrame([{"rule_id": "a", "review_status": "approved"}, {"rule_id": "c", "review_status": "draft"}])
    extract_io.guard_state_rules_merge(existing, ok, "t")                       # reviewed count kept -> fine
    extract_io.guard_state_rules_merge(None, pd.DataFrame(columns=["review_status"]), "t")   # nothing existed
    with pytest.raises(RuntimeError, match="refusing to overwrite"):
        extract_io.guard_state_rules_merge(existing, pd.DataFrame(columns=["review_status"]), "t")   # the near-miss
    with pytest.raises(RuntimeError):
        extract_io.guard_state_rules_merge(existing, pd.DataFrame([{"rule_id": "b", "review_status": "draft"}]), "t")


def test_run_dir_round_trip_and_snapshots(tmp_path):
    run_dir = extract_io.new_run_dir(str(tmp_path / "runs"), "20260921T120000")
    drafts = [{"rule_id": "EXT-1", "predicate": json.dumps({"type": "max_units_per_day"}), "codes": json.dumps(["98960"]),
               "extraction_confidence": 0.8, "notes": "Found by: model.", "effective_date": None}]
    cache = {"k1|v1|HYBRID|m": [{"statement": "s"}]}
    counts = extract_io.write_run(run_dir, inventory_rows=[{"chapter": "10", "doc_hash": "h"}], section_rows=[],
                                  log_rows=[{"chapter": "10", "model_calls": 3}], dropped_rows=[], change_rows=[],
                                  drafts=drafts, cache=cache, manifest={"run_id": "20260921T120000", "finished": True})
    assert counts["drafts"] == 1 and counts["cache_entries"] == 1
    back = extract_io.read_run(run_dir)
    assert back["drafts"] == drafts and back["cache"] == cache and back["manifest"]["finished"] is True
    assert back["inventory"][0]["chapter"] == "10" and back["sections"] == []
    assert extract_io.latest_run_dir(str(tmp_path / "runs")) == run_dir
    # NaN in a frame becomes null on disk, not the string 'nan'
    df = pd.DataFrame([{"a": 1.0, "b": float("nan")}])
    extract_io.write_jsonl(str(tmp_path / "x.jsonl"), df)
    assert extract_io.read_jsonl(str(tmp_path / "x.jsonl")) == [{"a": 1.0, "b": None}]
    # snapshots: a missing stage dir loads as empty, not as an error
    c, rules, inv, meta = extract_io.load_snapshots(str(tmp_path / "nowhere"))
    assert c == {} and rules is None and inv is None and meta == {}


def test_export_snapshots_stops_on_denied_read(tmp_path):
    class Spark:
        def table(self, fqn): raise Exception("[CLOUD_ACCESS_DENIED] 403 AuthorizationFailure")
    with pytest.raises(kb_io.TableAccessError):
        extract_io.export_snapshots(Spark(), str(tmp_path), "c", "s", "i", log_fn=lambda x: None)
    assert not os.path.exists(tmp_path / "snapshots" / "state_rules.jsonl")   # nothing half-written

"""
rule_normalize.py  -  post-gate, pre-compile normalisation of rule dicts.

Runs on Rule.to_dict() output after validate_gate.gate(). Pure and idempotent.

  stable ids     rule_id is re-derived from CONTENT (origin, source, statement,
                 predicate, dates). The legacy assign_id() collided badly - one
                 id was shared by nine different Chapter 10 rules - which makes
                 policy_rules unusable as a keyed table.
  dedupe         identical content collapses to one row.
  plane          derived deterministically from predicate type; never hand-set,
                 never 'enforcement' (a claims outcome is not a policy plane).
  provenance     source_url / doc_hash filled for NCCI rows if missing.
"""
from __future__ import annotations

from typing import Iterable

MEDICAID_NCCI_URL = ("https://www.cms.gov/medicare/coding-billing/"
                     "ncci-medicaid/medicaid-ncci-edit-files")

# ----------------------------------------------------------------- plane
_PLANE_BY_PREDICATE = {
    "code_pair_prohibited":   "coding_edit",
    "max_units_per_line":     "coding_edit",
    "max_units_per_dos":      "coding_edit",
    "max_units_per_day":      "coding_edit",
    "max_units_per_period":   "coverage_scope",
    "max_dollars_per_period": "coverage_scope",
    "frequency_limit":        "coverage_scope",
    "code_not_covered":       "coverage_scope",
}
_PLANE_BY_REASON = {
    "requires_medical_record":   "medical_necessity",
    "requires_member_attribute": "coverage_scope",
    "cross_source_conflict":     "coverage_scope",
    "ambiguous_source":          "coverage_scope",
}


def derive_plane(rule: dict) -> str:
    pred = rule.get("predicate") or {}
    t = pred.get("type")
    if t == "not_checkable":
        reason = pred.get("reason") or rule.get("not_checkable_reason")
        return _PLANE_BY_REASON.get(reason, "coverage_scope")
    return _PLANE_BY_PREDICATE.get(t, "coverage_scope")


# ------------------------------------------------------------- stable id
from schema import content_signature, stable_rule_id, PLANES  # noqa: E402


# ------------------------------------------------------------ provenance
def enrich_provenance(rule: dict) -> dict:
    if rule.get("origin") != "ncci":
        return rule
    if not rule.get("source_url"):
        rule["source_url"] = MEDICAID_NCCI_URL
    if not rule.get("doc_hash"):
        rule["doc_hash"] = content_signature(rule)[:16]
    return rule


# ---------------------------------------------------------- orchestrator
def normalize_rules(rules: Iterable[dict]) -> tuple[list[dict], dict]:
    out: dict[str, dict] = {}
    dropped_enforcement = 0
    collapsed = 0
    id_changed = 0
    for r in rules:
        r = dict(r)
        pred = r.get("predicate") or {}
        if r.get("plane") == "enforcement" or pred.get("type") == "edit_failure":
            dropped_enforcement += 1
            continue
        new_id = stable_rule_id(r)
        if r.get("rule_id") != new_id:
            id_changed += 1
        r["rule_id"] = new_id
        r["plane"] = derive_plane(r)
        r = enrich_provenance(r)
        if new_id in out:
            collapsed += 1
            continue
        out[new_id] = r

    rows = list(out.values())
    planes: dict[str, int] = {}
    for r in rows:
        planes[r["plane"]] = planes.get(r["plane"], 0) + 1
    report = {
        "rules_out": len(rows),
        "enforcement_rows_dropped": dropped_enforcement,
        "duplicate_content_collapsed": collapsed,
        "rule_ids_rederived": id_changed,
        "plane_counts": planes,
    }
    return rows, report

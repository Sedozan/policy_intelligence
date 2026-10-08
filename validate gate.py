"""
validate_gate.py  -  nothing enters the KB unvetted.

gate(rules) -> (accepted: list[dict], rejected: list[dict])

  rejected     structurally unusable (no statement, no source, no predicate type)
  accepted     everything else, as dicts, with:
      validation_status  clean        nothing changed
                         patched      the gate normalised something (codes upper-cased,
                                      codes derived from the predicate, ...)
                         downgraded   machine_checkable was turned off because the
                                      predicate is not in the grammar or a required
                                      claim field is not in the extract
                         needs_review ambiguity_flag set, or authored not-checkable
      gate_notes         what was changed / why, one line per action

The gate never resolves ambiguity and never deletes a rule: a rule that cannot
be checked by machine is kept as a lead so the coverage finding survives.
"""
from __future__ import annotations

import config
import schema
from schema import Rule, PREDICATE_TYPES, PREDICATE_TYPES_COMPILED

_CODE_KEYS = ("code_a", "code_b", "code")


def _codes_from_predicate(p: dict) -> list[str]:
    out = []
    for k in _CODE_KEYS:
        if p.get(k):
            out.append(str(p[k]).strip().upper())
    for c in p.get("code_set") or []:
        out.append(str(c).strip().upper())
    return list(dict.fromkeys(out))


def _check(rule: Rule) -> tuple[Rule, list[str], str]:
    notes: list[str] = []
    status = "clean"
    p = rule.predicate or {}

    # ---- codes: normalise, derive from predicate if the author left them out
    codes = [str(c).strip().upper() for c in (rule.codes or []) if str(c).strip()]
    if codes != list(rule.codes or []):
        notes.append("codes normalised")
        status = "patched"
    derived = _codes_from_predicate(p)
    if not codes and derived:
        codes = derived
        notes.append("codes derived from predicate")
        status = "patched"
    rule.codes = codes

    # ---- predicate type must be in the grammar
    t = p.get("type")
    if t not in PREDICATE_TYPES:
        if rule.machine_checkable:
            rule.machine_checkable = False
            rule.not_checkable_reason = "predicate_not_in_grammar"
            notes.append(f"downgraded: predicate '{t}' not in grammar")
            status = "downgraded"
    elif t not in PREDICATE_TYPES_COMPILED and t != "not_checkable" and rule.machine_checkable:
        rule.machine_checkable = False
        rule.not_checkable_reason = rule.not_checkable_reason or "predicate_not_in_grammar"
        notes.append(f"downgraded: predicate '{t}' authorable but not yet compiled")
        status = "downgraded"

    # ---- authored not_checkable must say why
    if t == "not_checkable":
        rule.machine_checkable = False
        rule.not_checkable_reason = rule.not_checkable_reason or p.get("reason") or "no_claim_field"

    # ---- required claim fields must exist in the extract
    if rule.machine_checkable:
        missing = sorted(set(rule.required_claim_fields or []) - config.AVAILABLE_CLAIM_FIELDS)
        if missing:
            rule.machine_checkable = False
            rule.not_checkable_reason = "missing_claim_field"
            notes.append("downgraded: claim fields not in extract: " + ", ".join(missing))
            status = "downgraded"

    # ---- population (F01): a rule scoped to "members under 21" would compile as if it applied to
    # everyone and false-positive every other member. Since 2026-10-02 the grammar has ONE member
    # condition: age on the date of service (predicate['age'], from date_of_birth on the claim).
    # A scoped rule compiles only when it carries a valid age condition - the reviewer signs that the
    # condition IS the population (e.g. "members 21 and older" = {"min": 21}). Any other scope
    # (ALTCS, EPSDT status, setting) stays a lead with the population preserved for the reviewer.
    pop = (rule.population or "all").strip().lower()
    if rule.machine_checkable and pop not in ("all", "", "any", "all members"):
        age = p.get("age")
        if age and schema.validate_age(age)[0] and rule.origin != "ncci":
            notes.append(f"population '{rule.population}' expressed by age condition "
                         f"({schema.age_label(age)}) - signed by the reviewer")
        else:
            rule.machine_checkable = False
            rule.not_checkable_reason = "requires_member_attribute"
            notes.append(f"downgraded: population '{rule.population}' cannot be applied "
                         f"(no age condition, or not an age scope)")
            status = "downgraded"

    # ---- code-pair sanity: a pair needs two distinct codes (compared normalised - C1-13: '99213'
    # vs '99213 ' passed as two codes and compiled a self-pair that flagged ordinary visits)
    if t == "code_pair_prohibited" and rule.machine_checkable:
        a, b = schema.norm_code(p.get("code_a", "")), schema.norm_code(p.get("code_b", ""))
        if not a or not b or a == b:
            rule.machine_checkable = False
            rule.not_checkable_reason = "ambiguous_source"
            notes.append("downgraded: code pair incomplete or self-pair")
            status = "downgraded"
        elif rule.origin != "ncci":
            # C1-06: a blank indicator on a state pair means no modifier bypasses it ('0'); an
            # indicator that is neither 0 nor 1 is not a rule anyone signed
            mi = schema.normalize_indicator(p.get("modifier_indicator"), rule.origin)
            if mi is None:
                rule.machine_checkable = False
                rule.not_checkable_reason = "ambiguous_source"
                notes.append(f"downgraded: modifier_indicator {p.get('modifier_indicator')!r} is not 0 or 1")
                status = "downgraded"
            elif str(p.get("modifier_indicator")) != mi:
                p["modifier_indicator"] = mi
                notes.append(f"modifier_indicator normalised to '{mi}'")
                status = "patched" if status == "clean" else status

    # ---- the grammar contract (schema.validate_predicate): a compiled rule whose predicate is
    # malformed (bad period, bad age, non-integer threshold...) becomes a lead with the reason
    if rule.machine_checkable and t in PREDICATE_TYPES_COMPILED:
        ok, why = schema.validate_predicate(p, origin=rule.origin)
        if not ok:
            rule.machine_checkable = False
            rule.not_checkable_reason = "ambiguous_source"
            notes.append(f"downgraded: invalid predicate ({why})")
            status = "downgraded"

    if rule.ambiguity_flag or not rule.machine_checkable:
        status = "needs_review" if status in ("clean", "patched") else status
    return rule, notes, status


def gate(rules: list[Rule]) -> tuple[list[dict], list[dict]]:
    accepted, rejected = [], []
    for r in rules:
        probs = r.problems()
        if probs:
            d = r.to_dict()
            d["reject_reason"] = "; ".join(probs)
            rejected.append(d)
            continue
        r, notes, status = _check(r)
        d = r.to_dict()
        d["validation_status"] = status
        d["gate_notes"] = "; ".join(notes)
        accepted.append(d)
    return accepted, rejected


def summarize(accepted: list[dict], rejected: list[dict]) -> dict:
    by = {}
    for d in accepted:
        by[d["validation_status"]] = by.get(d["validation_status"], 0) + 1
    return {"accepted": len(accepted), "rejected": len(rejected), "by_status": by}

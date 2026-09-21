"""
chapter_profile.py  -  what does each manual chapter LOOK like to the pipeline?

Answers "is the framework overfit to Ch10/13/19/22?" with numbers instead of opinion.
No LLM. For each chapter: what the parser sees (pages, sections, scanned pages), what
the pre-filter keeps, which CODE FAMILIES the text uses (the pre-filter only recognises
CPT/HCPCS today), which rule cues dominate (modifier / per-diem / PA / rate), and what
the deterministic extractor yields. Then a heuristic PROFILE:

  code_rich        CPT/HCPCS-dense, rules in prose        -> ready for the current grammar
  category_driven  few codes, many rule cues              -> needs the category->code resolver
  modifier_driven  modifier language dominates            -> needs the modifier_required compiler
  foreign_codes    revenue / DRG / NDC codes dominate     -> needs new CODE_RE + predicates
  process          few codes, few rule cues               -> KB text only, no detectors

The numbers are the product; the profile label is a suggestion for a human to confirm.
"""
from __future__ import annotations

import re

from policy_docs import CODE_RE, TRIGGER_PHRASES

# ---------------------------------------------------------------- code families
# CODE_RE (pipeline) = CPT 5-digit / 4+letter, HCPCS letter+4. CDT dental (D####) matches
# it too. The families below are what the pipeline CANNOT see today.
_CPT_RE = re.compile(r"\b\d{4}[0-9A-Z]\b")
_HCPCS_RE = re.compile(r"\b[A-CE-Z]\d{4}\b")            # excludes D#### (dental)
_CDT_RE = re.compile(r"\bD\d{4}\b")
_REV_RE = re.compile(r"\brevenue codes?\b[^.\n]{0,60}?\b0\d{3}\b|\b0\d{3}\b(?=[^.\n]{0,40}\brevenue\b)", re.I)
_DRG_RE = re.compile(r"\b(?:APR[- ]?)?DRG\b", re.I)
_NDC_RE = re.compile(r"\b\d{4,5}-\d{3,4}-\d{1,2}\b|\bNDC\b")

# rule cues that the grammar does / does not express yet
_CUES = {
    "modifier":   re.compile(r"\bmodifiers?\b", re.I),
    "per_diem":   re.compile(r"\bper[- ]diem\b|\bincluded in the (?:per diem|daily rate)\b", re.I),
    "prior_auth": re.compile(r"\bprior authorization\b|\bPA\b(?= is| required| must)", re.I),
    "rate":       re.compile(r"\bfee schedule\b|\breimbursement rate\b|\bcapped fee\b|\brate\b", re.I),
    "age":        re.compile(r"\b(?:under|over|age[sd]?|years of age|EPSDT|under 21|21 and older)\b", re.I),
    "pos":        re.compile(r"\bplace of service\b|\bPOS\b", re.I),
    "units":      re.compile(r"\bunits?\b", re.I),
    "frequency":  re.compile(r"\bonce per\b|\bper (?:day|month|year|benefit year|contract year|lifetime)\b", re.I),
}


def profile_text(text: str) -> dict:
    cpt = set(_CPT_RE.findall(text))
    hcpcs = set(_HCPCS_RE.findall(text))
    cdt = set(_CDT_RE.findall(text))
    pipeline_codes = set(CODE_RE.findall(text))
    low = text.lower()
    return {
        "chars": len(text),
        "codes_pipeline": len(pipeline_codes),         # what CODE_RE / the pre-filter can see
        "codes_cpt": len(cpt), "codes_hcpcs": len(hcpcs), "codes_cdt": len(cdt),
        "mentions_revenue": len(_REV_RE.findall(text)),
        "mentions_drg": len(_DRG_RE.findall(text)),
        "mentions_ndc": len(_NDC_RE.findall(text)),
        "trigger_phrases": sum(low.count(p) for p in TRIGGER_PHRASES),
        **{f"cue_{k}": len(rx.findall(text)) for k, rx in _CUES.items()},
    }


def profile_chapter(chapter: str, title: str, pages: list[dict], sections: list[dict],
                    meta: dict, drafts: list[dict] | None = None, dropped: list[dict] | None = None,
                    stats: dict | None = None) -> dict:
    full = "\n".join(s["text"] for s in sections)
    p = profile_text(full)
    n_pages = len(pages)
    blank_pages = sum(1 for pg in pages if not any(ln["text"].strip() for ln in pg["lines"]))
    cand = [s for s in sections if s.get("has_rule_signal")]
    by_type: dict[str, int] = {}
    for d in drafts or []:
        try:
            import json
            t = json.loads(d["predicate"]).get("type") if isinstance(d.get("predicate"), str) else (d.get("predicate") or {}).get("type")
        except Exception:
            t = "?"
        by_type[t] = by_type.get(t, 0) + 1
    row = {
        "chapter": chapter, "title": title,
        "pages": n_pages, "blank_pages": blank_pages,
        "sections": len(sections), "candidate_sections": len(cand),
        "candidate_ratio": round(len(cand) / len(sections), 2) if sections else 0.0,
        "headers_stripped": len(meta.get("headers_stripped") or []),
        "revision_date": meta.get("revision_date"),
        **p,
        "stub_drafts": len(drafts or []), "stub_dropped": len(dropped or []),
        "stub_by_type": by_type,
        "stub_compiled_types": sum(v for k, v in by_type.items() if k not in ("not_checkable", None, "?")),
    }
    row["profile"], row["profile_reason"] = classify(row)
    return row


def classify(r: dict) -> tuple[str, str]:
    """Heuristic. Thresholds are deliberately simple and visible; adjust after one real run."""
    codes = r["codes_pipeline"]
    foreign = r["mentions_revenue"] + r["mentions_drg"] + r["mentions_ndc"]
    cues = r["trigger_phrases"]
    per_page = lambda k: r[k] / max(r["pages"], 1)
    if r["blank_pages"] and r["blank_pages"] >= 0.5 * max(r["pages"], 1):
        return "scanned", f"{r['blank_pages']}/{r['pages']} pages have no extractable text - needs OCR"
    if foreign >= 10 and foreign > codes:
        return "foreign_codes", (f"revenue/DRG/NDC mentions ({foreign}) exceed CPT/HCPCS codes ({codes}); "
                                 f"CODE_RE and the pre-filter cannot see these")
    # code_rich by evidence (the stub already compiled rules) or by density (codes per page),
    # not by an absolute count that penalises short chapters
    if r.get("stub_compiled_types", 0) >= 5:
        return "code_rich", (f"deterministic extractor already yields {r['stub_compiled_types']} compilable rules "
                             f"({codes} codes, {cues} cues) - current grammar applies")
    if (codes >= 20 or (codes >= 8 and per_page("codes_pipeline") >= 2)) and cues >= 10:
        return "code_rich", f"{codes} CPT/HCPCS codes ({per_page('codes_pipeline'):.1f}/page), {cues} rule cues - current grammar applies"
    if r["cue_modifier"] >= 8 and r["cue_modifier"] >= codes:
        return "modifier_driven", (f"{r['cue_modifier']} modifier mentions vs {codes} codes - "
                                   f"needs modifier_required compiler")
    if codes < 10 and (cues >= 10 or r["cue_per_diem"] >= 3 or per_page("cue_units") >= 1):
        return "category_driven", (f"{codes} codes but {cues} rule cues / {r['cue_per_diem']} per-diem "
                                   f"mentions - needs category->code resolver")
    if codes < 10 and cues < 10:
        return "process", f"{codes} codes, {cues} rule cues, {r['candidate_ratio']:.0%} candidate sections - KB text only"
    return "mixed", f"{codes} codes, {cues} cues, {r['cue_modifier']} modifier mentions - inspect"


PROFILE_NEEDS = {
    "code_rich":       "ready now (Phase 1.5 grammar)",
    "category_driven": "Phase 2: category->code resolver (B2 Matrix / fee schedule)",
    "modifier_driven": "Phase 2: modifier_required compiler + conditions",
    "foreign_codes":   "new CODE_RE families (revenue / DRG / NDC) + new predicate types",
    "process":         "no detectors; index as KB text for RAG / QA",
    "scanned":         "OCR before anything else",
    "mixed":           "human look",
}


def summarize(rows: list[dict]) -> dict:
    out: dict[str, list[str]] = {}
    for r in rows:
        out.setdefault(r["profile"], []).append(r["chapter"])
    return out

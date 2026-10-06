#!/usr/bin/env python3
"""
CaseScope — 'Why Investigate' case-lead renderer   (v2)
=======================================================
Renders investigator-facing case leads FROM the Spark engine's outputs. It no longer
re-implements any rule: the queue and hits come from casescope_triage_spark, so the lead
list, the CSV and the Power BI queue are the same entities by construction.

Databricks:
    from casescope_triage_spark import *
    import case_narratives as cn
    queue = score_providers(hits, paid)
    q_pdf, h_pdf, n_pdf = cn.from_spark(queue, hits, paid)        # corroborated subset only
    leads = cn.build_case_leads(q_pdf, h_pdf, n_pdf)
    cn.check_leads(leads, q_pdf)                                   # raises on any violation
    table = cn.queue_with_reasons(q_pdf, h_pdf, n_pdf)

Language rules (claims/billing data only):
  * "flagged for review", never an assertion of fraud or intent;
  * no medical-necessity language;
  * every statistic described exactly as computed (percentiles are over member-days or
    members, not over providers).
"""
from __future__ import annotations
import textwrap
import pandas as pd

FAMILY_NAME = {
    "intensity": "Billing intensity (units / claim volume per member-day)",
    "facility_pricing": "Outpatient facility pricing",
    "member_brokering": "Members shared across unusually many billers",
    "provider_identity": "Rendering-provider identity pattern",
    "duplicate_payment": "Same procedure, member and date paid more than once",
    "transport": "Transport without an associated service",
    "role": "Provider in a role it should not hold",
}

# Each template is filled from per-rule stats: occasions (hit rows), members (distinct),
# metric_max (largest underlying measure). Wording states exactly what was computed.
RULE_TEMPLATES = {
    "R1_IMPOSSIBLE_HOURS": (
        "Billed time above 24 hours in a day",
        "{r1_text}"),
    "R2_EXCESS_UNITS": (
        "Units per member-day above peers",
        "On {occasions} member-day(s), units billed for one member were above the 99.5th "
        "percentile of all provider/member/day groups in the same category of service "
        "(peak {metric_max:.0f} units)."),
    "R3_PHANTOM_TRIP": (
        "Transport with no service nearby",
        "{occasions} transport claim group(s) with no non-transport, non-pharmacy service for "
        "the member within one day."),
    "R4_MEMBER_FANOUT": (
        "Members seen by unusually many billers",
        "Billed for {members} member(s) who each received services from an unusually large "
        "number of different billing providers (up to {metric_max:.0f}), above the 99.5th "
        "percentile of all members. This pattern is consistent with patient brokering but also "
        "occurs with members who have complex care needs."),
    "R5_FACILITY_FEE_OUTLIER": (
        "Unusual outpatient facility encounters",
        "{r5_text}"),
    "R6_DUPLICATE_BILLING": (
        "Possible duplicate claims (unverified)",
        "{occasions} instance(s) of the same service appearing on more than one claim for the "
        "same member, date and amount. Not yet verified: these may be adjustments, voids, "
        "replacements or encounter copies. Not counted toward this lead's tier or dollars."),
    "R7_MEMBER_DAY_VOLUME": (
        "Claim lines per member-day above peers",
        "On {occasions} member-day(s), the number of claim lines for one member was above the "
        "99.5th percentile of all provider/member/day/category-of-service groups "
        "(peak {metric_max:.0f} lines)."),
    "R8A_MILL_RENDERERS": (
        "Many rendering providers for one member",
        "For {members} member(s), claims were submitted under an unusually large number of "
        "different rendering providers (up to {metric_max:.0f})."),
    "R8B_RENDERER_MANY_BILLERS": (
        "Rendering under many billing providers",
        "Appears as the rendering provider on claims submitted by {metric_max:.0f} different "
        "billing providers, above the 99.5th percentile of rendering providers."),
    "R9_CROSS_BILLER_DUP": (
        "Same procedure, member and date also paid to another biller",
        "On {occasions} occasion(s), a claim with the same member, procedure code, date and units "
        "was paid to this provider and to at least one other billing provider. This is a "
        "candidate for double billing, not proof: it can be legitimate (for example, separate "
        "transport legs by different vendors). Only this provider's own payments are counted; "
        "claims under the MSP line of business are excluded."),
    "R10_ROLE_CONTRADICTION": (
        "Role contradicts SIU reference",
        "On claims for {members} member(s), appears in a provider role that the SIU reference "
        "list says it should not hold."),
}

BANNED_PHRASES = ["committed fraud", "is fraudulent", "fraudulent provider", "is guilty",
                  "medically unnecessary", "not medically necessary", "lack of medical necessity"]


def provider_names(paid):
    """Most frequent name/type per entity and role, plus any other names seen (Spark)."""
    from pyspark.sql import functions as F, Window
    def one(key, name, ptype, etype):
        c = (paid.filter(F.col(key) != -1)
                 .groupBy(F.col(key).cast("long").alias("entity_id"),
                          F.col(name).alias("name"), F.col(ptype).alias("ptype"))
                 .count())
        w = Window.partitionBy("entity_id").orderBy(F.desc("count"), "name", "ptype")
        c = c.withColumn("rk", F.row_number().over(w))
        primary = c.filter("rk = 1").select("entity_id", F.col("name").alias("_primary"))
        alts = (c.filter("rk > 1").join(primary, "entity_id")
                 .filter(F.col("name") != F.col("_primary"))          # same name, other type: not an alias
                 .groupBy("entity_id")
                 .agg(F.concat_ws("; ", F.array_sort(F.collect_set("name"))).alias("other_names")))
        return (c.filter("rk = 1").drop("rk", "count").join(alts, "entity_id", "left")
                 .withColumn("entity_type", F.lit(etype)))
    return (one("BILLING_PROVIDER_KEY", "BProv_Name", "BProv_Type_Desc", "billing")
            .unionByName(one("SERVICING_PROVIDER_KEY", "SProv_Name", "SProv_Type_Desc", "servicing")))


def from_spark(queue, hits, paid, only_corroborated: bool = True):
    """Pull just what the renderer needs into pandas (corroborated entities only by default)."""
    from pyspark.sql import functions as F
    E = ["entity_id", "entity_type"]
    q = queue.filter("corroborated") if only_corroborated else queue
    h = hits.join(F.broadcast(q.select(*E)), E).drop("line_keys")
    n = provider_names(paid).join(F.broadcast(q.select(*E)), E)
    return q.toPandas(), h.toPandas(), n.toPandas()


def _r5_text(rh: pd.DataFrame) -> str:
    """State exactly which R5 condition fired, so 'stacked lines' is never called 'high price'."""
    d = rh.detail.fillna("")
    n_price = int(d.isin(["price", "price_and_stacked"]).sum())
    n_stack = int(d.isin(["stacked", "price_and_stacked"]).sum())
    parts = []
    if n_price:
        parts.append(f"{n_price} encounter(s) with facility payments far above the typical "
                     f"encounter (more than 8 robust deviations above the median; largest "
                     f"${rh.metric.max():,.0f})")
    if n_stack:
        parts.append(f"{n_stack} encounter(s) with more facility-fee lines than 99.5% of "
                     f"outpatient encounters")
    return "Outpatient facility claims show " + " and ".join(parts) + "."


def _r1_text(rh: pd.DataFrame) -> str:
    """Separate a single code over 24h (impossible) from several services added together."""
    single = rh[rh.tier == 1]
    combined = rh[rh.tier != 1]
    peak = rh.loc[rh.metric.idxmax()]
    breakdown = str(peak.detail).split(";")[0]
    s = (f"On {len(rh)} day(s), the time this provider billed for a single member added up to "
         f"more than 24 hours (peak {peak.metric:.1f} hours: {breakdown}). Time is computed from "
         f"units and each procedure's stated unit.")
    if len(single):
        s += (f" On {len(single)} of these day(s), one service code alone exceeds 24 hours, which "
              f"one person cannot deliver.")
    if len(combined):
        s += (f" On {len(combined)} day(s), 24 hours is exceeded only when different services are "
              f"added together or 'up to' units are counted at their maximum; this can happen if "
              f"several staff served the member at the same time, so these days are not treated "
              f"as an impossibility.")
    return s


def _rule_stats(rh: pd.DataFrame) -> dict:
    top = rh.sort_values("hit_dollars", ascending=False)
    keys = []
    for ek in top.evidence_keys.fillna(""):
        for k in ek.split("|"):
            if k and k not in keys:
                keys.append(k)
            if len(keys) >= 5:
                break
        if len(keys) >= 5:
            break
    details = sorted({x for d in rh.detail.dropna() for x in str(d).split(", ") if x})
    st = dict(occasions=len(rh), members=int(rh.member_key.dropna().nunique()),
              metric_max=float(rh.metric.max()), example_keys=keys, details=", ".join(details))
    if (rh.rule_id == "R5_FACILITY_FEE_OUTLIER").all():
        st["r5_text"] = _r5_text(rh)
    if (rh.rule_id == "R1_IMPOSSIBLE_HOURS").all():
        st["r1_text"] = _r1_text(rh)
    return st


def render_case_lead(row: pd.Series, ent_hits: pd.DataFrame, name: dict | None = None,
                     data_label: str | None = None) -> str:
    name = name or {}
    role = "Billing" if row.entity_type == "billing" else "Servicing (rendering)"
    activity = "claim lines billed" if row.entity_type == "billing" else "claim lines rendered"
    indep = [f for f in (row.families or "").split(",") if f]
    L = []
    title = f"CASE LEAD [{row.tier_label}] — {role} Provider {row.entity_id}"
    if isinstance(name.get("name"), str):
        title += f"  ({name['name']})"
    L += [title, "=" * 76]
    if isinstance(name.get("other_names"), str) and name["other_names"]:
        L.append(f"Also appears as   : {name['other_names']}")
    L.append(f"Provider type     : {name['ptype'] if isinstance(name.get('ptype'), str) else '(unknown)'}")
    L.append(f"Activity          : {int(row.n_claims):,} paid {activity} | {int(row.n_members):,} members"
             f" | last service {str(row.last_svc_dt)[:10]}")
    L.append(f"Total paid (role) : ${row.total_paid:,.0f}")
    L.append(f"Independent grounds: {int(row.independent_families)} — "
             + "; ".join(FAMILY_NAME.get(f, f) for f in indep))
    L.append(f"Breadth           : {int(row.members_flagged)} members, "
             f"{int(row.member_days_flagged)} member-days in flagged claims")
    L.append(f"$ in flagged claims: ${row.dollars_at_risk:,.0f} "
             f"({row.pct_dollars_at_risk * 100:.0f}% of this provider's paid $ in this role; "
             f"{int(row.lines_flagged)} distinct claim lines, each counted once)")
    extra = row.dollars_incl_untrusted - row.dollars_at_risk
    if extra > 0.5:
        L.append(f"                    + ${extra:,.0f} flagged only by an unverified rule (not counted)")
    L += ["", "WHY THIS PROVIDER IS FLAGGED FOR REVIEW", "-" * 76]

    fams = list(dict.fromkeys(indep + sorted(set(ent_hits.family) - set(indep))))
    n = 0
    for fam in fams:
        fh = ent_hits[ent_hits.family == fam]
        if fh.empty:
            continue
        tag = "" if fam in indep else "  [overlaps the claims above; not counted as a separate ground]"
        L.append(f"{FAMILY_NAME.get(fam, fam)}{tag}")
        for rule_id in sorted(fh.rule_id.unique()):
            n += 1
            st = _rule_stats(fh[fh.rule_id == rule_id])
            head, tmpl = RULE_TEMPLATES[rule_id]
            L.append(f"  {n}. {head}")
            L += ["     " + s for s in textwrap.wrap(tmpl.format(**st), 70)]
            L.append(f"     Example claim line IDs (UNIQUE_KEY): {', '.join(st['example_keys'])}")
        L.append("")

    L += ["INVESTIGATOR NEXT STEPS", "-" * 76,
          "  - Look up the example lines by UNIQUE_KEY and confirm each pattern on the raw claims."]
    rules = set(ent_hits.rule_id)
    if "R1_IMPOSSIBLE_HOURS" in rules:
        L.append("  - R1: check the service times and staff; >24h for one member in a day needs an")
        L.append("    explanation such as several staff serving the member at the same time.")
    if "R4_MEMBER_FANOUT" in rules:
        L.append("  - R4: check whether the shared members have complex care needs (a legitimate cause).")
    if "R9_CROSS_BILLER_DUP" in rules:
        L.append("  - R9: confirm both payments went to different tax IDs and neither was later recouped.")
    if "R6_DUPLICATE_BILLING" in rules:
        L.append("  - R6: rule out adjustments/voids/replacements/encounter copies before citing.")
    L.append("  - Check the provider against the suspension list and prior SIU dispositions.")
    L += ["", "  Basis: post-payment claims and billing data only. This is an analytic lead for",
          "  review, NOT a determination of fraud, intent, medical necessity or wrongdoing."]
    if data_label:
        L.append(f"  Source data: {data_label}")
    text = "\n".join(L)
    low = text.lower()
    hit = [p for p in BANNED_PHRASES if p in low]
    if hit:
        raise ValueError(f"Lead for {row.entity_id} contains prohibited phrasing: {hit}")
    return text


def build_case_leads(queue_pdf: pd.DataFrame, hits_pdf: pd.DataFrame,
                     names_pdf: pd.DataFrame | None = None, data_label: str | None = None) -> pd.DataFrame:
    """One rendered lead per corroborated entity, in queue order."""
    q = queue_pdf[queue_pdf.corroborated].reset_index(drop=True)
    names = {}
    if names_pdf is not None and len(names_pdf):
        names = {(int(r.entity_id), r.entity_type):
                 {"name": r["name"], "ptype": r.ptype, "other_names": r.get("other_names")}
                 for _, r in names_pdf.iterrows()}
    grouped = {k: g for k, g in hits_pdf.groupby(["entity_id", "entity_type"])}
    out = []
    for _, row in q.iterrows():
        k = (int(row.entity_id), row.entity_type)
        out.append(dict(entity_id=k[0], entity_type=k[1], tier_label=row.tier_label,
                        lead=render_case_lead(row, grouped.get(k, hits_pdf.iloc[0:0]),
                                              names.get(k), data_label)))
    return pd.DataFrame(out)


def queue_with_reasons(queue_pdf, hits_pdf, names_pdf=None, data_label=None) -> pd.DataFrame:
    """Shareable table: corroborated entities with plain-English reasons, same rows as the leads."""
    q = queue_pdf[queue_pdf.corroborated].copy()
    titles = (hits_pdf.groupby(["entity_id", "entity_type"]).rule_id
                      .apply(lambda s: " | ".join(RULE_TEMPLATES[r][0] for r in sorted(set(s)))))
    q = q.join(titles.rename("findings"), on=["entity_id", "entity_type"])
    q["grounds"] = q.families.apply(lambda s: " | ".join(FAMILY_NAME.get(f, f) for f in s.split(",") if f))
    q["rule_codes"] = q.rules.apply(lambda s: ", ".join(r.split("_")[0] for r in s.split(",") if r))
    if names_pdf is not None and len(names_pdf):
        keep = [c for c in ["entity_id", "entity_type", "name", "other_names", "ptype"] if c in names_pdf]
        q = q.merge(names_pdf[keep], on=["entity_id", "entity_type"], how="left")
    cols = ["entity_id", "entity_type", "name", "other_names", "ptype", "tier_label",
            "independent_families", "grounds", "members_flagged", "member_days_flagged",
            "dollars_at_risk", "pct_dollars_at_risk", "total_paid", "last_svc_dt",
            "days_before_data_end", "data_end_date", "rule_codes", "findings"]
    out = q[[c for c in cols if c in q.columns]].copy()
    out["pct_dollars_at_risk"] = out.pct_dollars_at_risk.round(3)
    out["dollars_at_risk"] = out.dollars_at_risk.round(2)
    if data_label:
        out.insert(0, "data_source", data_label)
    return out


def check_leads(leads: pd.DataFrame, queue_pdf: pd.DataFrame, table: pd.DataFrame | None = None):
    """Invariants between the queue, the leads and the shareable table. Raises on failure."""
    E = ["entity_id", "entity_type"]
    corr = queue_pdf[queue_pdf.corroborated]
    errors = []
    if set(map(tuple, leads[E].values)) != set(map(tuple, corr[E].values)):
        errors.append("leads do not match the corroborated entities in the queue")
    if table is not None and set(map(tuple, table[E].values)) != set(map(tuple, corr[E].values)):
        errors.append("shareable table does not match the corroborated entities in the queue")
    if (corr.pct_dollars_at_risk > 1 + 1e-9).any():
        errors.append("a corroborated entity has flagged $ above its paid $")
    for _, r in leads.iterrows():
        low = r.lead.lower()
        if any(p in low for p in BANNED_PHRASES):
            errors.append(f"prohibited phrasing in lead {r.entity_id}")
        if "top 0.5% of comparable providers" in low:
            errors.append(f"misstated statistic in lead {r.entity_id}")
    print("Case-lead checks:", "PASS" if not errors else "FAIL")
    for e in errors:
        print("  -", e)
    if errors:
        raise AssertionError(errors)
    return True

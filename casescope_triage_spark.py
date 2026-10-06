#!/usr/bin/env python3
"""
CASESCOPE (PySpark) — Claims-Based FWA Triage & Evidence Engine   (v3)
=====================================================================
Reads paid claims from a Unity Catalog Delta table and produces a ranked, explainable
entity-risk queue, the rule hits behind it, and the exact claim lines each flag rests on.

SOURCE  : main.prod_input.all_data_C_A   (CFG["table"])
RUNTIME : Databricks notebook or job. Every hot path is a groupBy / join; the only
          Python UDF runs once per flagged entity (thousands of rows, not millions).

NOTEBOOK USE (import, don't %run — this is a .py module):
    from casescope_triage_spark import *
    paid  = prep(spark.table(CFG["table"])).cache()
    hits  = run_rules(paid).cache()
    queue = score_providers(hits, paid).cache()
    run_checks(queue, hits, paid)            # raises if any invariant fails
    lines = flagged_lines(hits, paid)        # claim-level evidence

v3 CHANGES (each fixes a defect found in review):
  * Corroboration counts independent SIGNAL FAMILIES, not rules. R1/R2/R7 flag the same
    claims (measured containment 1.0), so they are one family ("intensity"). A family only
    counts if its claims are not mostly inside an already-counted family, and a corroborated
    entity must involve >= 2 distinct members. One busy member-day can no longer look like
    three independent findings.
  * Dollars are deduplicated at claim-line level: each line is priced once per entity.
    pct_dollars_at_risk is no longer capped at 100% — so if it ever exceeds 100%, that is
    a bug and run_checks() fails.
  * Entities are keyed by (entity_id, entity_type). Billing and servicing roles are separate
    queues; servicing-role evidence (R8B, R10) is never merged into a billing record.
  * R9 credits each biller only with its OWN payment, not the pair's combined dollars.
  * Tier labels come from evidence (families, tier-1 findings), not score quantiles — the
    old quantile labels guaranteed ~15% "Strong" regardless of evidence.
  * Untrusted rules (R6 until the adjustment/void exclusion lands) are still reported, but
    excluded from corroboration, score, tiering and dollars_at_risk.
  * run_checks(): runtime assertions you can run on the real table.
  * exact_percentiles: exact (pandas-matching) percentiles for parity tests.
  After independent audit (all computations matched a from-scratch re-derivation):
  * R1 reads time units from PROC_CD_DESC ("PER 15 MINUTES") instead of a COS placeholder.
  * R4 counts only for billers with >= 2 high-fan-out members (fan-out is a member property).
  * R5 "stacked lines" must be rare (above p99.5 of lines per encounter, floor 3).
  * R9 is tier 2: a code/member/date match is a candidate, not proof of the same service.

SCHEMA CONTRACT — confirm/remap before the first prod run:
    ADJU_STA, BILLING_PROVIDER_KEY, SERVICING_PROVIDER_KEY, MEMBER_KEY, COS_DESC, PROC_CD,
    PROC_CD_DESC, QUANTITY_PAID, PMT_AMT, UNIQUE_KEY, Claim_Hdr_ID, Svc_Begin_Dt,
    PAY_TO_TAX_ID, line_of_business
"""

from __future__ import annotations
import argparse, datetime
from pyspark.sql import DataFrame, SparkSession, functions as F
from pyspark.sql import types as T

# --------------------------------------------------------------------------- #
CFG = dict(
    table="main.prod_input.all_data_C_A",
    out_schema="main.siu_fwa",             # where Delta outputs are written; set to a schema you own
    write_outputs=False,                   # flip True in prod once out_schema is confirmed
    paid_status="A",
    units_pctile=0.995, units_floor=2,
    max_minutes_per_day=1440,              # R1: billed time per member-day per biller cannot exceed 24h
    z_facility=8.0,
    facility_lines_pctile=0.995, facility_min_lines=3,   # R5 stacked lines: above p99.5 of lines/encounter (floor 3)
    anchor_window_days=1, phantom_max_base_rate=0.25,
    fanout_pctile=0.995, volume_pctile=0.995,
    fanout_min_hot_members=2,              # R4: a biller needs >= 2 high-fan-out members (1 is incidental)
    pmi_min=2.0, pmi_min_shared=3, panel_min_members=20,
    tier_weight={1: 3.0, 2: 2.0, 3: 1.0},
    exact_percentiles=True,                # exact = reproducible and matches the audited definitions. The
                                           # approximate method can land on a different integer for count
                                           # cutoffs (R7: 8 vs 9 lines). Set False only if a run is too slow.
    approx_err=0.001, pctile_accuracy=10000,
    top_providers_cards=15, evidence_keys_cap=20,
    # --- window ---
    date_col="Svc_Begin_Dt",
    window_anchor="data_max",              # "data_max" (latest claim, never later than today) or "today"
    window_months=24,
    context_months=None,                   # None = one flat window. Set (e.g. 24) + window_months=6 to split
    # --- R8 ---
    svc_per_member_biller_pctile=0.995, svc_per_member_biller_floor=5,
    billers_per_svc_pctile=0.995, billers_per_svc_floor=3,
    # --- R9 ---
    cob_lob_values=["MSP"],                # coordination-of-benefits LOBs: two payers on one service is legit
    dup_match_units=True,
    # --- R10 ---
    role_reference_path=None,              # table/parquet: provider_key, expected_role ("billing"/"servicing")
    # --- corroboration & trust ---
    family_overlap_max=0.5,                # a family counts as independent only if < 50% of its claims
                                           # are already inside a counted family (for that entity)
    corroboration_min_families=2,
    corroboration_min_members=2,
    strong_min_families=3,                 # Strong = corroborated AND (>=3 families OR a tier-1 family)
    untrusted_rules=["R6_DUPLICATE_BILLING"],   # reported, but excluded from score/corroboration/$ until fixed
)

# Rules that flag the same underlying claims belong to one family (measured, not assumed:
# R1/R2/R7 containment = 1.0 on the synthetic file). Corroboration counts families.
FAMILY = {
    "R1_IMPOSSIBLE_HOURS": "intensity",
    "R2_EXCESS_UNITS": "intensity",
    "R7_MEMBER_DAY_VOLUME": "intensity",
    "R5_FACILITY_FEE_OUTLIER": "facility_pricing",
    "R3_PHANTOM_TRIP": "transport",
    "R4_MEMBER_FANOUT": "member_brokering",
    "R8A_MILL_RENDERERS": "provider_identity",
    "R8B_RENDERER_MANY_BILLERS": "provider_identity",
    "R6_DUPLICATE_BILLING": "duplicate_payment",
    "R9_CROSS_BILLER_DUP": "duplicate_payment",
    "R10_ROLE_CONTRADICTION": "role",
}

# R1 reads the unit of time from the procedure description ("PER 15 MINUTES", "UP TO 15
# MINUTES", "PER HOUR"), so it applies to exactly the time-billed codes, in any category of
# service. A quantity on a code without a time unit (DME items, per-diem, visits) is never
# converted to time.
def minutes_per_unit(desc_col: str = "PROC_CD_DESC"):
    d = F.upper(F.coalesce(F.col(desc_col), F.lit("")))
    mins = r"(?:PER|UP TO|EACH)\s+(\d+)\s*MINUTE"          # MINUTE, not MIN: "MINI-BUS" must not match
    hrs_n = r"(?:PER|UP TO|EACH)\s+(\d+)\s*HOUR"
    return (F.when(d.rlike(mins), F.regexp_extract(d, mins, 1).cast("double"))
             .when(d.rlike(hrs_n), F.regexp_extract(d, hrs_n, 1).cast("double") * 60)
             .when(d.rlike(r"(?:PER|UP TO|EACH)\s+HOUR"), F.lit(60.0))
             .otherwise(F.lit(None).cast("double")))

HIT_COLS = ["entity_id", "entity_type", "rule_id", "family", "tier", "title", "severity",
            "metric", "detail", "hit_dollars", "member_key", "cos", "evidence_keys", "line_keys"]
REG = []                                   # (rule_id, tier, title, fn)


def rule(rule_id, tier, title):
    def deco(fn):
        REG[:] = [r for r in REG if r[0] != rule_id]   # idempotent on notebook re-runs
        REG.append((rule_id, tier, title, fn)); return fn
    return deco


def _lines():
    """Every claim line behind a hit — the basis for deduplicated dollars and evidence."""
    return F.array_sort(F.collect_set("UNIQUE_KEY")).alias("line_keys")


def _q(df: DataFrame, col: str, p: float):
    """Scalar percentile. Exact mode uses linear interpolation (matches pandas)."""
    if df.limit(1).count() == 0:
        return None
    if CFG["exact_percentiles"]:
        return df.agg(F.expr(f"percentile({col}, {p})")).first()[0]
    return df.approxQuantile(col, [p], CFG["approx_err"])[0]


def _q_expr(col: str, p: float):
    if CFG["exact_percentiles"]:
        return F.expr(f"percentile({col}, {p})")
    return F.expr(f"percentile_approx({col}, {p}, {CFG['pctile_accuracy']})")


def _std(dfagg: DataFrame, rule_id, tier, title, severity_col, metric_col, dollars_col,
         member_col="MEMBER_KEY", cos_col=None, entity_col="BILLING_PROVIDER_KEY",
         entity_type="billing", detail_col=None) -> DataFrame:
    """Coerce any rule aggregation into the shared hit schema."""
    cos = (F.lit("(cross-COS)") if cos_col is None else
           F.col(cos_col) if isinstance(cos_col, str) else cos_col)
    etype = F.lit(entity_type) if isinstance(entity_type, str) else entity_type
    member = F.lit(None).cast("long") if member_col is None else F.col(member_col).cast("long")
    detail = F.lit(None).cast("string") if detail_col is None else detail_col
    tier_col = F.lit(tier) if isinstance(tier, int) else tier        # a rule may tier per hit
    return dfagg.select(
        F.col(entity_col).cast("long").alias("entity_id"),
        etype.alias("entity_type"),
        F.lit(rule_id).alias("rule_id"), F.lit(FAMILY[rule_id]).alias("family"),
        tier_col.cast("int").alias("tier"), F.lit(title).alias("title"),
        F.round(severity_col.cast("double"), 3).alias("severity"),
        F.col(metric_col).cast("double").alias("metric"),
        detail.alias("detail"),
        F.round(F.col(dollars_col).cast("double"), 2).alias("hit_dollars"),
        member.alias("member_key"), cos.alias("cos"),
        F.concat_ws("|", F.slice("line_keys", 1, CFG["evidence_keys_cap"])).alias("evidence_keys"),
        F.col("line_keys"))


# --------------------------------------------------------------------------- #
def prep(df: DataFrame) -> DataFrame:
    d = (df.filter(F.col("ADJU_STA") == CFG["paid_status"])
           .withColumn("svc_dt", F.to_date(CFG["date_col"]))
           .withColumn("member_day", F.concat_ws("|", "MEMBER_KEY", "Svc_Begin_Dt"))
           .withColumn("PMT_AMT", F.coalesce(F.col("PMT_AMT").cast("double"), F.lit(0.0)))
           .withColumn("QUANTITY_PAID", F.coalesce(F.col("QUANTITY_PAID").cast("double"), F.lit(0.0)))
           .filter(F.col("svc_dt").isNotNull()))
    # A service date in the future is a data error; left in, it would drag the window anchor.
    today = datetime.date.today()
    n_future = d.filter(F.col("svc_dt") > F.lit(today)).count()
    if n_future:
        print(f"  [prep] excluded {n_future:,} paid lines with a future service date")
        d = d.filter(F.col("svc_dt") <= F.lit(today))
    if CFG["window_anchor"] == "today":
        anchor = today
    else:
        anchor = d.agg(F.max("svc_dt")).first()[0]
    d = d.withColumn("_det_start", F.add_months(F.lit(anchor), -CFG["window_months"]))
    ctx = CFG.get("context_months")
    if ctx:
        d = d.filter(F.col("svc_dt") >= F.add_months(F.lit(anchor), -ctx))
    else:
        d = d.filter(F.col("svc_dt") >= F.col("_det_start"))
    d = d.withColumn("in_detection_window", F.col("svc_dt") >= F.col("_det_start"))
    return d.drop("_det_start")


def restrict_to_detection(hits: DataFrame, paid: DataFrame) -> DataFrame:
    """Keep entities active in the detection window, in the role they were flagged in."""
    w = paid.filter("in_detection_window")
    recent = (w.select(F.col("BILLING_PROVIDER_KEY").cast("long").alias("entity_id"),
                       F.lit("billing").alias("entity_type"))
               .unionByName(w.select(F.col("SERVICING_PROVIDER_KEY").cast("long").alias("entity_id"),
                                     F.lit("servicing").alias("entity_type")))
               .distinct())
    return hits.join(recent, ["entity_id", "entity_type"])


# ---- RULES ---------------------------------------------------------------- #
@rule("R1_IMPOSSIBLE_HOURS", 1, "Billed time for one member exceeds 24 hours in a day")
def r1(paid):
    if "PROC_CD_DESC" not in paid.columns:
        print("    [R1 SKIPPED] PROC_CD_DESC missing — cannot tell which codes are billed in time units")
        return None
    cap = CFG["max_minutes_per_day"]
    t = (paid.withColumn("mpu", minutes_per_unit("PROC_CD_DESC"))
             .filter(F.col("mpu").isNotNull())
             .withColumn("upto", F.upper(F.col("PROC_CD_DESC")).rlike(r"UP TO\s+\d+").cast("int"))
             .withColumn("minutes", F.col("QUANTITY_PAID") * F.col("mpu")))
    per_code = (t.groupBy("BILLING_PROVIDER_KEY", "member_day", "MEMBER_KEY", "PROC_CD")
                 .agg(F.sum("minutes").alias("code_min"), F.max("upto").alias("code_upto"),
                      F.sum("PMT_AMT").alias("code_dollars"),
                      F.collect_set("COS_DESC").alias("cos_set"), F.collect_set("UNIQUE_KEY").alias("keys")))
    g = (per_code.groupBy("BILLING_PROVIDER_KEY", "member_day", "MEMBER_KEY")
          .agg(F.sum("code_min").alias("minutes"),
               F.max(F.when(F.col("code_upto") == 0, F.col("code_min"))).alias("max_exact_code_min"),
               F.max("code_upto").alias("any_upto"), F.count(F.lit(1)).alias("n_codes"),
               F.sum("code_dollars").alias("dollars"),
               F.concat_ws(" + ", F.array_sort(F.collect_list(
                   F.format_string("%s %.1fh", F.col("PROC_CD"), F.col("code_min") / 60)))).alias("breakdown"),
               F.concat_ws(", ", F.array_sort(F.array_distinct(F.flatten(F.collect_list("cos_set"))))).alias("cos"),
               F.array_sort(F.array_distinct(F.flatten(F.collect_list("keys")))).alias("line_keys"))
          .filter(F.col("minutes") > cap)
          .withColumn("hours", F.col("minutes") / 60))
    # Tier 1 (impossible) only when ONE code with an exact unit exceeds 24h by itself. Summing
    # different services (S5125 + T1019) or 'UP TO n MINUTES' units can be legitimate (several
    # staff at once) or overstated, so those days are tier 2 and say so in the detail.
    tier = F.when(F.col("max_exact_code_min") > cap, 1).otherwise(2)
    detail = F.concat_ws("; ", F.col("breakdown"),
                         F.when(F.col("n_codes") > 1, F.lit("combined across codes")),
                         F.when(F.col("any_upto") == 1, F.lit("includes 'up to' codes: hours are a maximum")),
                         F.when(F.col("max_exact_code_min") > cap, F.lit("single code exceeds 24h")))
    return _std(g, "R1_IMPOSSIBLE_HOURS", tier, "Billed time for one member exceeds 24 hours in a day",
                F.col("minutes") / F.lit(cap), "hours", "dollars", cos_col="cos", detail_col=detail)


@rule("R2_EXCESS_UNITS", 2, "Units per member-day far above peer norm")
def r2(paid):
    g = (paid.groupBy("COS_DESC", "member_day", "BILLING_PROVIDER_KEY", "MEMBER_KEY")
             .agg(F.sum("QUANTITY_PAID").alias("units"), F.sum("PMT_AMT").alias("dollars"), _lines()))
    peer = g.groupBy("COS_DESC").agg(_q_expr("units", CFG["units_pctile"]).alias("peer"),
                                     _q_expr("units", 0.5).alias("med"))
    g = (g.join(F.broadcast(peer), "COS_DESC")
          .filter((F.col("units") > F.col("peer")) & (F.col("units") >= CFG["units_floor"])))
    sev = F.col("units") / F.when(F.col("med") > 0, F.col("med")).otherwise(F.lit(1.0))
    return _std(g, "R2_EXCESS_UNITS", 2, "Units per member-day far above peer norm",
                sev, "units", "dollars", cos_col="COS_DESC")


@rule("R3_PHANTOM_TRIP", 2, "Transport billed with no anchoring service that day")
def r3(paid):
    transport = ["NON-EMERGENCY TRANSPORTATION", "EMERGENCY TRANSPORTATION"]
    trips = paid.filter(F.col("COS_DESC").isin(transport))
    clinical = (paid.filter(~F.col("COS_DESC").isin(transport + ["PHARMACY"]))
                    .select(F.col("MEMBER_KEY").alias("m"), F.col("svc_dt").alias("cdt")).distinct())
    joined = (trips.join(clinical, (trips.MEMBER_KEY == clinical.m) &
                         (F.abs(F.datediff(trips.svc_dt, clinical.cdt)) <= CFG["anchor_window_days"]), "left")
                   .withColumn("anchored", F.col("cdt").isNotNull()))
    per_trip = joined.groupBy("UNIQUE_KEY").agg(F.max(F.col("anchored").cast("int")).alias("anch"))
    total = per_trip.count()
    if total == 0:
        return None
    phantom_rate = 1 - (per_trip.agg(F.sum("anch")).first()[0] or 0) / total
    if phantom_rate > CFG["phantom_max_base_rate"]:
        print(f"    [R3 SKIPPED] phantom base rate {phantom_rate:.1%} > "
              f"{CFG['phantom_max_base_rate']:.0%} — likely data structure, not a signal")
        return None
    ph = trips.join(per_trip.filter(F.col("anch") == 0).select("UNIQUE_KEY"), "UNIQUE_KEY")
    g = (ph.groupBy("BILLING_PROVIDER_KEY", "member_day", "MEMBER_KEY", "COS_DESC")
           .agg(F.count(F.lit(1)).alias("n"), F.sum("PMT_AMT").alias("dollars"), _lines()))
    return _std(g, "R3_PHANTOM_TRIP", 2, "Transport billed with no anchoring service that day",
                F.col("n"), "n", "dollars", cos_col="COS_DESC")


@rule("R4_MEMBER_FANOUT", 3, "Member sees abnormally many billing providers")
def r4(paid):
    fan = paid.groupBy("MEMBER_KEY").agg(F.countDistinct("BILLING_PROVIDER_KEY").alias("fanout"))
    cutoff = _q(fan, "fanout", CFG["fanout_pctile"])
    if cutoff is None:
        return None
    hot = fan.filter(F.col("fanout") > cutoff)
    g = (paid.join(F.broadcast(hot), "MEMBER_KEY")
             .groupBy("BILLING_PROVIDER_KEY", "MEMBER_KEY", "fanout")
             .agg(F.sum("PMT_AMT").alias("dollars"), _lines()))
    # Fan-out is a property of the MEMBER: every biller who saw a hot member would be "hit".
    # Only billers serving several such members carry a provider-level brokering signal.
    hubs = (g.groupBy("BILLING_PROVIDER_KEY").agg(F.count(F.lit(1)).alias("n_hot"))
             .filter(F.col("n_hot") >= CFG["fanout_min_hot_members"]))
    g = g.join(hubs, "BILLING_PROVIDER_KEY")
    return _std(g, "R4_MEMBER_FANOUT", 3, "Member sees abnormally many billing providers",
                F.col("fanout"), "fanout", "dollars")


@rule("R5_FACILITY_FEE_OUTLIER", 2, "Outpatient facility $/encounter above peer norm")
def r5(paid):
    g = (paid.filter(F.col("COS_DESC") == "OUT-PATIENT FACILITY FEES")
             .groupBy("BILLING_PROVIDER_KEY", "Claim_Hdr_ID", "MEMBER_KEY")
             .agg(F.sum("PMT_AMT").alias("dollars"), F.count(F.lit(1)).alias("lines"), _lines()))
    med = _q(g, "dollars", 0.5)
    if med is None:
        return None
    mad = _q(g.withColumn("ad", F.abs(F.col("dollars") - F.lit(med))), "ad", 0.5) or 1e-9
    # Stacked lines must be rare to count: 3+ lines alone covered 15.5% of encounters.
    lines_cut = max(_q(g, "lines", CFG["facility_lines_pctile"]) or 0, CFG["facility_min_lines"])
    price = F.col("z") > CFG["z_facility"]
    stacked = F.col("lines") > lines_cut
    g = (g.withColumn("z", F.lit(0.6745) * (F.col("dollars") - F.lit(med)) / F.lit(mad))
          .filter(price | stacked))
    detail = (F.when(price & stacked, "price_and_stacked").when(price, "price").otherwise("stacked"))
    return _std(g, "R5_FACILITY_FEE_OUTLIER", 2, "Outpatient facility $/encounter above peer norm",
                F.greatest("z", "lines"), "dollars", "dollars",
                cos_col=F.lit("OUT-PATIENT FACILITY FEES"), detail_col=detail)


@rule("R6_DUPLICATE_BILLING", 1, "Same service billed on multiple claims")
def r6(paid):
    # UNTRUSTED until adjustments/voids/replacements/encounter twins are excluded (CFG).
    key = ["MEMBER_KEY", "BILLING_PROVIDER_KEY", "PROC_CD", "Svc_Begin_Dt", "PMT_AMT", "COS_DESC"]
    g = (paid.filter(F.col("PMT_AMT") > 0).groupBy(*key)
             .agg(F.countDistinct("Claim_Hdr_ID").alias("hdrs"), F.count(F.lit(1)).alias("n"),
                  F.sum("PMT_AMT").alias("dollars"), _lines())
             .filter(F.col("hdrs") > 1))
    return _std(g, "R6_DUPLICATE_BILLING", 1, "Same service billed on multiple claims",
                F.col("n"), "n", "dollars", cos_col="COS_DESC")


@rule("R7_MEMBER_DAY_VOLUME", 3, "Claim volume per member-day above peer norm")
def r7(paid):
    g = (paid.groupBy("member_day", "BILLING_PROVIDER_KEY", "MEMBER_KEY", "COS_DESC")
             .agg(F.count(F.lit(1)).alias("n"), F.sum("PMT_AMT").alias("dollars"), _lines()))
    cutoff = _q(g, "n", CFG["volume_pctile"])
    if cutoff is None:
        return None
    g = g.filter(F.col("n") > cutoff)
    return _std(g, "R7_MEMBER_DAY_VOLUME", 3, "Claim volume per member-day above peer norm",
                F.col("n"), "n", "dollars", cos_col="COS_DESC")


@rule("R8A_MILL_RENDERERS", 2, "Many rendering providers for one member under one biller")
def r8a(paid):
    g = (paid.filter(F.col("SERVICING_PROVIDER_KEY") != -1)
             .groupBy("BILLING_PROVIDER_KEY", "MEMBER_KEY")
             .agg(F.countDistinct("SERVICING_PROVIDER_KEY").alias("n_svc"),
                  F.sum("PMT_AMT").alias("dollars"), _lines()))
    q = _q(g, "n_svc", CFG["svc_per_member_biller_pctile"])
    if q is None:
        return None
    g = g.filter(F.col("n_svc") > max(q, CFG["svc_per_member_biller_floor"]))
    return _std(g, "R8A_MILL_RENDERERS", 2, "Many rendering providers for one member under one biller",
                F.col("n_svc"), "n_svc", "dollars")


@rule("R8B_RENDERER_MANY_BILLERS", 2, "Rendering provider bills under many distinct billers/tax IDs")
def r8b(paid):
    # Keyed to the SERVICING provider: a separate entity type, never merged into a billing record.
    g = (paid.filter(F.col("SERVICING_PROVIDER_KEY") != -1)
             .groupBy("SERVICING_PROVIDER_KEY")
             .agg(F.countDistinct("BILLING_PROVIDER_KEY").alias("n_billers"),
                  F.countDistinct("PAY_TO_TAX_ID").alias("n_taxids"),
                  F.sum("PMT_AMT").alias("dollars"), _lines()))
    q = _q(g, "n_billers", CFG["billers_per_svc_pctile"])
    if q is None:
        return None
    g = g.filter(F.col("n_billers") > max(q, CFG["billers_per_svc_floor"]))
    return _std(g, "R8B_RENDERER_MANY_BILLERS", 2,
                "Rendering provider bills under many distinct billers/tax IDs",
                F.greatest("n_billers", "n_taxids"), "n_billers", "dollars", member_col=None,
                entity_col="SERVICING_PROVIDER_KEY", entity_type="servicing")


@rule("R9_CROSS_BILLER_DUP", 2, "Same procedure, member and date paid to two billers")
def r9(paid):
    # Same member + procedure + date (+ units) PAID to >= 2 distinct billers, MSP excluded.
    # A match is a candidate, not proof of the same service (e.g. two transport legs by two
    # vendors), so this is tier 2: it cannot make a lead "Strong" on its own.
    # Each biller is credited only with its OWN payment.
    elig = paid.filter(F.col("PMT_AMT") > 0)
    if CFG["cob_lob_values"] and "line_of_business" in paid.columns:
        elig = elig.filter(~F.col("line_of_business").isin(CFG["cob_lob_values"]))
    key = ["MEMBER_KEY", "PROC_CD", "Svc_Begin_Dt"] + (["QUANTITY_PAID"] if CFG["dup_match_units"] else [])
    dup = (elig.groupBy(*key).agg(F.countDistinct("BILLING_PROVIDER_KEY").alias("n_billers"))
               .filter(F.col("n_billers") >= 2))
    own = (elig.join(dup, key)
               .groupBy(*key, "BILLING_PROVIDER_KEY", "n_billers")
               .agg(F.sum("PMT_AMT").alias("dollars"), F.first("COS_DESC").alias("cos"), _lines()))
    return _std(own, "R9_CROSS_BILLER_DUP", 2,
                "Same procedure, member and date paid to two billers",
                F.col("n_billers"), "n_billers", "dollars", cos_col="cos")


@rule("R10_ROLE_CONTRADICTION", 1, "Provider appears in a role it should never hold")
def r10(paid):
    # Needs an EXTERNAL expected-role reference from SIU; dormant until configured.
    path = CFG.get("role_reference_path")
    if not path:
        return None
    spark = paid.sparkSession
    ref = spark.read.parquet(path) if str(path).endswith(".parquet") else spark.table(path)
    ref = ref.select(F.col("provider_key").cast("long").alias("provider_key"),
                     F.lower(F.trim("expected_role")).alias("expected_role"))
    base = ["PMT_AMT", "UNIQUE_KEY", "MEMBER_KEY", "COS_DESC"]
    obs = (paid.select(F.col("BILLING_PROVIDER_KEY").cast("long").alias("provider_key"),
                       F.lit("billing").alias("observed_role"), *base)
               .unionByName(paid.select(F.col("SERVICING_PROVIDER_KEY").cast("long").alias("provider_key"),
                                        F.lit("servicing").alias("observed_role"), *base)))
    g = (obs.filter(F.col("provider_key") != -1)
            .join(F.broadcast(ref), "provider_key")
            .filter(F.col("observed_role") != F.col("expected_role"))
            .groupBy("provider_key", "observed_role", "MEMBER_KEY", "COS_DESC")
            .agg(F.count(F.lit(1)).alias("n"), F.sum("PMT_AMT").alias("dollars"), _lines()))
    return _std(g, "R10_ROLE_CONTRADICTION", 1, "Provider appears in a role it should never hold",
                F.col("n"), "n", "dollars", cos_col="COS_DESC",
                entity_col="provider_key", entity_type=F.col("observed_role"))


def run_rules(paid) -> DataFrame:
    hits = None
    for rule_id, tier, title, fn in REG:
        out = fn(paid)
        if out is None:
            print(f"  {rule_id:<26} tier {tier}  ->       0 hits"); continue
        out = out.persist(); n = out.count()
        flag = "  (untrusted: reported, not scored)" if rule_id in CFG["untrusted_rules"] else ""
        print(f"  {rule_id:<26} tier {tier}  ->  {n:>7,} hits{flag}")
        hits = out if hits is None else hits.unionByName(out)
    return hits


# ---- CLAIM-LEVEL EVIDENCE ------------------------------------------------- #
def flagged_lines(hits: DataFrame, paid: DataFrame) -> DataFrame:
    """One row per (entity, rule, claim line) behind any hit, with the line's $/member/date."""
    info = paid.select("UNIQUE_KEY", "PMT_AMT", "MEMBER_KEY", "svc_dt",
                       F.col("BILLING_PROVIDER_KEY").cast("long").alias("billing_key"),
                       F.col("SERVICING_PROVIDER_KEY").cast("long").alias("servicing_key"))
    return (hits.select("entity_id", "entity_type", "rule_id", "family", "tier",
                        F.explode("line_keys").alias("UNIQUE_KEY"))
                .distinct()
                .join(info, "UNIQUE_KEY"))


_FAM_T = T.StructType([T.StructField("n", T.IntegerType()),
                       T.StructField("families", T.ArrayType(T.StringType())),
                       T.StructField("tier1", T.BooleanType())])


def _independent_families_udf(overlap_max):
    @F.udf(returnType=_FAM_T)
    def _f(fams):
        # Largest family first; a later family counts only if most of its claims are new.
        fams = sorted(fams or [], key=lambda f: (-len(f["keys"]), f["family"]))
        counted, seen, tier1 = [], set(), False
        for f in fams:
            k = set(f["keys"])
            if k and (not counted or len(k & seen) / len(k) < overlap_max):
                counted.append(f["family"]); tier1 = tier1 or bool(f["fam_tier1"])
            seen |= k
        return (len(counted), counted, tier1)
    return _f


# ---- SCORING -------------------------------------------------------------- #
def score_providers(hits: DataFrame, paid: DataFrame) -> DataFrame:
    """Entity-level queue keyed by (entity_id, entity_type). Name kept for notebook compatibility."""
    E = ["entity_id", "entity_type"]
    untrusted = F.col("rule_id").isin(CFG["untrusted_rules"])
    lines = flagged_lines(hits, paid).persist()
    trusted = lines.filter(~untrusted)

    # Dollars: each claim line priced ONCE per entity, never summed across rules.
    per_line = trusted.select(*E, "UNIQUE_KEY", "PMT_AMT", "MEMBER_KEY", "svc_dt").distinct()
    dollars = per_line.groupBy(*E).agg(
        F.sum("PMT_AMT").alias("dollars_at_risk"),
        F.countDistinct("UNIQUE_KEY").alias("lines_flagged"),
        F.countDistinct("MEMBER_KEY").alias("members_flagged"),
        F.countDistinct("MEMBER_KEY", "svc_dt").alias("member_days_flagged"))
    dollars_all = (lines.select(*E, "UNIQUE_KEY", "PMT_AMT").distinct()
                        .groupBy(*E).agg(F.sum("PMT_AMT").alias("dollars_incl_untrusted")))

    # Independent families.
    fam = (trusted.groupBy(*E, "family")
                  .agg(F.collect_set("UNIQUE_KEY").alias("keys"),
                       F.max((F.col("tier") == 1).cast("int")).alias("fam_tier1")))
    indep = (fam.groupBy(*E).agg(F.collect_list(F.struct("family", "keys", "fam_tier1")).alias("fams"))
                .withColumn("r", _independent_families_udf(CFG["family_overlap_max"])("fams"))
                .select(*E, F.col("r.n").alias("independent_families"),
                        F.concat_ws(",", "r.families").alias("families"),
                        F.col("r.tier1").alias("has_tier1_family")))

    # Score: within a family take the strongest rule (not the sum), then sum across families.
    tw = F.create_map([x for k, v in CFG["tier_weight"].items() for x in (F.lit(k), F.lit(v))])
    rule_w = (hits.filter(~untrusted)
                  .withColumn("w", tw[F.col("tier")] * (1 + F.log1p(F.greatest("severity", F.lit(0.0)))))
                  .groupBy(*E, "family", "rule_id").agg(F.sum("w").alias("rw")))
    score = (rule_w.groupBy(*E, "family").agg(F.max("rw").alias("fw"))
                   .groupBy(*E).agg(F.sum("fw").alias("risk_score")))

    summary = hits.groupBy(*E).agg(
        F.count(F.lit(1)).alias("n_hits"),
        F.countDistinct("rule_id").alias("distinct_rules"),
        F.concat_ws(",", F.array_sort(F.collect_set("rule_id"))).alias("rules"))

    # Context per ROLE: a servicing entity is compared with what it rendered, not what it billed.
    def _ctx(col, etype):
        return (paid.filter(F.col(col) != -1)
                    .groupBy(F.col(col).cast("long").alias("entity_id"))
                    .agg(F.sum("PMT_AMT").alias("total_paid"), F.count(F.lit(1)).alias("n_claims"),
                         F.countDistinct("MEMBER_KEY").alias("n_members"),
                         F.max("svc_dt").alias("last_svc_dt"))
                    .withColumn("entity_type", F.lit(etype)))
    ctx = _ctx("BILLING_PROVIDER_KEY", "billing").unionByName(_ctx("SERVICING_PROVIDER_KEY", "servicing"))
    ref_dt = paid.agg(F.max("svc_dt")).first()[0]

    q = (summary.join(score, E, "left").join(dollars, E, "left").join(dollars_all, E, "left")
                .join(indep, E, "left").join(ctx, E, "left")
                .fillna({"risk_score": 0.0, "dollars_at_risk": 0.0, "lines_flagged": 0,
                         "members_flagged": 0, "member_days_flagged": 0,
                         "dollars_incl_untrusted": 0.0, "independent_families": 0,
                         "families": "", "has_tier1_family": False})
                .withColumn("pct_dollars_at_risk", F.col("dollars_at_risk") / F.col("total_paid"))
                # Measured from the last service date in the DATA, not from today (claims lag).
                .withColumn("data_end_date", F.lit(ref_dt))
                .withColumn("days_before_data_end", F.datediff(F.lit(ref_dt), F.col("last_svc_dt"))))
    corr = ((F.col("independent_families") >= CFG["corroboration_min_families"]) &
            (F.col("members_flagged") >= CFG["corroboration_min_members"]))
    q = (q.withColumn("corroborated", corr)
          .withColumn("risk_score", F.when(corr, F.col("risk_score") * 1.3).otherwise(F.col("risk_score")))
          .withColumn("tier_label",
                      F.when(corr & ((F.col("independent_families") >= CFG["strong_min_families"]) |
                                     F.col("has_tier1_family")), "Strong")
                       .when(corr, "Moderate").otherwise("Contributing"))
          .withColumn("untrusted_only", F.col("independent_families") == 0)
          .withColumn("_rank", F.when(F.col("tier_label") == "Strong", 0)
                                .when(F.col("tier_label") == "Moderate", 1).otherwise(2)))
    return (q.orderBy("_rank", F.desc("independent_families"), F.desc("risk_score"))
             .drop("_rank"))


# ---- RUNTIME CHECKS ------------------------------------------------------- #
def run_checks(queue: DataFrame, hits: DataFrame, paid: DataFrame, strict: bool = True):
    """Invariants that must hold on ANY data. Run after every scoring pass, especially on prod."""
    E = ["entity_id", "entity_type"]
    lines = flagged_lines(hits, paid)
    untrusted = F.col("rule_id").isin(CFG["untrusted_rules"])
    results = []

    def check(name, bad, detail=""):
        results.append((name, bad == 0, bad, detail))

    check("claim keys are unique in paid",
          paid.groupBy("UNIQUE_KEY").count().filter("count > 1").count(),
          "dollar dedup assumes one row per claim line")
    owner = F.when(F.col("entity_type") == "billing", F.col("billing_key")).otherwise(F.col("servicing_key"))
    check("every flagged line belongs to its entity in the flagged role",
          lines.filter(owner != F.col("entity_id")).count())
    check("no entity's flagged $ exceeds its own paid $ (uncapped)",
          queue.filter(F.col("pct_dollars_at_risk") > 1.0 + 1e-9).count())
    check("queue has exactly one row per flagged entity",
          abs(queue.count() - hits.select(*E).distinct().count()) +
          queue.groupBy(*E).count().filter("count > 1").count())
    check("corroborated => >= min independent families and >= min members",
          queue.filter(F.col("corroborated") &
                       ((F.col("independent_families") < CFG["corroboration_min_families"]) |
                        (F.col("members_flagged") < CFG["corroboration_min_members"]))).count())
    check("Strong/Moderate only when corroborated",
          queue.filter((F.col("tier_label") != "Contributing") & ~F.col("corroborated")).count())
    check("untrusted rules do not contribute to dollars_at_risk",
          queue.filter(F.col("dollars_at_risk") > F.col("dollars_incl_untrusted") + 0.01).count())
    tot_flag = (lines.filter(~untrusted).select("UNIQUE_KEY", "PMT_AMT").distinct()
                     .agg(F.sum("PMT_AMT")).first()[0] or 0.0)
    tot_paid = paid.agg(F.sum("PMT_AMT")).first()[0] or 0.0
    check("distinct flagged $ <= total paid $", int(tot_flag > tot_paid + 0.01),
          f"flagged ${tot_flag:,.0f} of ${tot_paid:,.0f} paid")

    print("CaseScope checks:")
    for name, ok, bad, detail in results:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({bad} violations)" if not ok else "")
              + (f"  — {detail}" if detail else ""))
    failed = [r for r in results if not r[1]]
    if failed and strict:
        raise AssertionError(f"{len(failed)} CaseScope check(s) failed: {[r[0] for r in failed]}")
    return results


# ---- GRAPH ---------------------------------------------------------------- #
def graph_signals(paid: DataFrame):
    pm = paid.select("BILLING_PROVIDER_KEY", "MEMBER_KEY").distinct()
    panel = pm.groupBy("BILLING_PROVIDER_KEY").agg(F.count(F.lit(1)).alias("sz"))
    big = panel.filter(F.col("sz") >= CFG["panel_min_members"])
    pm = pm.join(F.broadcast(big.select("BILLING_PROVIDER_KEY")), "BILLING_PROVIDER_KEY")
    N = paid.select("MEMBER_KEY").distinct().count()
    a = pm.withColumnRenamed("BILLING_PROVIDER_KEY", "pa")
    b = pm.withColumnRenamed("BILLING_PROVIDER_KEY", "pb")
    pairs = (a.join(b, "MEMBER_KEY").filter(F.col("pa") < F.col("pb"))
              .groupBy("pa", "pb").agg(F.count(F.lit(1)).alias("shared"))
              .filter(F.col("shared") >= CFG["pmi_min_shared"]))
    sz = big.select(F.col("BILLING_PROVIDER_KEY").alias("p"), "sz")
    pairs = (pairs.join(sz.withColumnRenamed("p", "pa").withColumnRenamed("sz", "sza"), "pa")
                  .join(sz.withColumnRenamed("p", "pb").withColumnRenamed("sz", "szb"), "pb")
                  .withColumn("pmi", F.log(F.col("shared") * F.lit(N) / (F.col("sza") * F.col("szb"))))
                  .withColumn("jaccard", F.col("shared") / (F.col("sza") + F.col("szb") - F.col("shared"))))
    suspicious = pairs.filter(F.col("pmi") >= CFG["pmi_min"]).orderBy(F.desc("pmi"))
    fanout = (paid.groupBy("MEMBER_KEY").agg(F.countDistinct("BILLING_PROVIDER_KEY").alias("fanout"))
                  .orderBy(F.desc("fanout")))
    return suspicious, fanout


def communities(suspicious: DataFrame):
    rows = suspicious.select("pa", "pb", "pmi").limit(50000).collect()
    if not rows:
        return []
    try:
        import networkx as nx
        G = nx.Graph()
        for r in rows:
            G.add_edge(r["pa"], r["pb"], weight=r["pmi"])
        comms = nx.community.greedy_modularity_communities(G, weight="weight")
        return [{"community": i, "size": len(c), "providers": sorted(map(int, c))}
                for i, c in enumerate(comms)]
    except Exception:
        return [{"note": "networkx unavailable or graph trivial"}]


# ---- EVIDENCE CARDS ------------------------------------------------------- #
def evidence_cards(queue: DataFrame, hits: DataFrame):
    E = ["entity_id", "entity_type"]
    top = queue.filter("corroborated").limit(CFG["top_providers_cards"])
    top_rows = top.collect()
    hrows = (hits.join(F.broadcast(top.select(*E)), E).drop("line_keys")
                 .orderBy(F.desc("hit_dollars")).collect())
    by_ent = {}
    for h in hrows:
        by_ent.setdefault((h["entity_id"], h["entity_type"]), []).append(h)
    cards = []
    for r in top_rows:
        hs = by_ent.get((r["entity_id"], r["entity_type"]), [])[:10]
        cards.append(dict(
            entity_id=int(r["entity_id"]), entity_type=r["entity_type"], tier_label=r["tier_label"],
            independent_families=int(r["independent_families"]), families=r["families"],
            rules_triggered=r["rules"], members_flagged=int(r["members_flagged"]),
            total_paid=round(float(r["total_paid"] or 0), 2),
            dollars_at_risk=round(float(r["dollars_at_risk"] or 0), 2),
            pct_dollars_at_risk=round(float(r["pct_dollars_at_risk"] or 0), 3),
            findings=[dict(rule=h["rule_id"], family=h["family"], finding=h["title"], tier=h["tier"],
                           member_key=h["member_key"], cos=h["cos"], hit_dollars=h["hit_dollars"],
                           example_claim_keys=(h["evidence_keys"] or "").split("|")[:5]) for h in hs]))
    return cards


# --------------------------------------------------------------------------- #
def get_source(spark, local_csv):
    if local_csv:
        return spark.read.csv(local_csv, header=True, inferSchema=True)
    return spark.table(CFG["table"])


def main(local_csv=None):
    builder = SparkSession.builder.appName("CaseScope")
    if local_csv:
        builder = builder.master("local[*]").config("spark.sql.shuffle.partitions", "16")
    spark = builder.getOrCreate()

    print(f"Source: {'LOCAL ' + local_csv if local_csv else CFG['table']}")
    paid = prep(get_source(spark, local_csv)).persist()
    print(f"  paid lines: {paid.count():,} | billers: "
          f"{paid.select('BILLING_PROVIDER_KEY').distinct().count():,} | "
          f"members: {paid.select('MEMBER_KEY').distinct().count():,}\n")

    print("Running rules engine:")
    hits = run_rules(paid)
    if hits is None:
        print("No rule produced hits."); return None, None, []
    hits = restrict_to_detection(hits, paid).persist()
    queue = score_providers(hits, paid).persist()
    print()
    run_checks(queue, hits, paid)

    susp, fanout = graph_signals(paid)
    comms = communities(susp)
    print(f"\nGraph: surprising provider pairs (PMI>={CFG['pmi_min']}): {susp.count():,} | "
          f"communities: {len([c for c in comms if 'size' in c])}")
    cards = evidence_cards(queue, hits)

    if CFG["write_outputs"]:
        s = CFG["out_schema"]
        hits.drop("line_keys").write.mode("overwrite").saveAsTable(f"{s}.casescope_rule_hits")
        flagged_lines(hits, paid).write.mode("overwrite").saveAsTable(f"{s}.casescope_flagged_lines")
        queue.write.mode("overwrite").saveAsTable(f"{s}.casescope_entity_queue")
        susp.write.mode("overwrite").saveAsTable(f"{s}.casescope_graph_pairs")
        print(f"  wrote Delta tables to {s}.casescope_*")

    print("\nTier mix:")
    queue.groupBy("entity_type", "tier_label").count().orderBy("entity_type", "tier_label").show()
    print("=== TOP 10 ===")
    (queue.select("entity_id", "entity_type", "tier_label", "independent_families", "families",
                  "members_flagged", F.round("dollars_at_risk", 0).alias("$_at_risk"),
                  F.round("pct_dollars_at_risk", 3).alias("pct"), "rules")
          .show(10, truncate=48))
    return queue, hits, cards


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--local", help="path to a local CSV for validation instead of the table")
    ap.add_argument("--exact", action="store_true", help="exact percentiles (parity tests)")
    ap.add_argument("--role-ref", help="R10 expected-role reference (parquet path or table)")
    a = ap.parse_args()
    if a.exact:
        CFG["exact_percentiles"] = True
    if a.role_ref:
        CFG["role_reference_path"] = a.role_ref
    main(local_csv=a.local)

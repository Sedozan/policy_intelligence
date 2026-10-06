# =============================================================================
# LOOKALIKE PROVIDER SCREEN — MULTI-WINDOW, GROUPED
# Find BH Professional providers whose billing BEHAVIOR resembles the two
# reference cases (servicing 004526, billing entity 246427), across three time
# windows, and combine the results into ONE row per provider with a trend label.
# =============================================================================
# How it runs:
#   - ONE pass scores EVERY provider in the cohort. It does not loop over
#     provider IDs. The only per-provider inputs are the two references.
#   - Clinician-day hours and line-level exposure are computed ONCE on the full
#     data (they are date-local), then date-filtered per window.
#   - Pattern features (repeats, cookie-cutter) are recomputed per window,
#     because "repeated" depends on which dates are in scope.
#
# Design principles (read before handing results to SIU):
#   1. Similarity is on BEHAVIOR (rates, percentages, intensity), never SIZE.
#   2. Each provider lists the exact red flags that put it on the list.
#   3. Exposure is computed per clinician-day on TIMED codes only:
#        Tier 1 = paid on minutes beyond 24h in one clinician-day (impossible)
#        Tier 2 = paid on minutes beyond CAP_HOURS (implausible; includes Tier 1)
#   4. Windows are NESTED (6M ⊂ 1Y ⊂ Full). Dollars are therefore reported from
#      ONE window only — the recovery lookback — never summed across windows.
#   5. Servicing and billing exposure are the same dollars seen two ways.
#      Never add them together.
# =============================================================================

from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.window import Window
import pandas as pd
import numpy as np
from datetime import date
from dateutil.relativedelta import relativedelta

spark = SparkSession.builder.getOrCreate()

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
REF_SERVICING = "004526"
REF_BILLING   = "246427"
CATEGORY      = "BH PROFESSIONAL"
SERVICE_TABLE = "main.prod_input.all_paid_services"

SPROV_COL, BPROV_COL = "Servicing_Provider_ID", "Billing_Provider_ID"
MEMBER_COL, PROC_COL = "MEMBER_KEY", "PROC_CD"
QTY_COL, PMT_COL     = "QUANTITY_PAID", "PMT_AMT"
SRV_BEG_COL, SRV_END_COL = "SRV_BEG_CALENDAR_KEY", "SRV_END_CALENDAR_KEY"
MODIFIER_COLS = ["MODIFIER_1", "MODIFIER_2", "MODIFIER_3", "MODIFIER_4"]
CATEGORY_COL  = "Category"

# (label, suffix, months_back). The trend logic expects suffixes Full/1Y/6M.
TIME_WINDOWS = [
    ("Full History",   "Full", None),
    ("Last 12 Months", "1Y",   12),
    ("Last 6 Months",  "6M",   6),
]
# Activity floor scales with window length so small active providers in the
# 6-month window are not dropped by a bar meant for full history.
MIN_ACTIVE_DAYS = {"Full": 20, "1Y": 12, "6M": 8}

# The ONE window dollars are reported from. Set this to the AHCCCS recovery
# lookback (confirm with SIU). None = full history, labeled as such.
RECOVERY_LOOKBACK_MONTHS = None

# Minutes per paid unit for TIMED codes. Only these feed hour/exposure math.
# VERIFY against the AHCCCS fee schedule before sharing results.
UNIT_MINUTES = {
    "H0004": 15,  # BH counseling/therapy, per 15 min
    "H2014": 15,  # Skills training & development, per 15 min
    "H2017": 15,  # Psychosocial rehab, per 15 min
    "H2019": 15,  # Therapeutic behavioral services, per 15 min
    "H0038": 15,  # Self-help/peer services, per 15 min
    "H2027": 15,  # Psychoeducational service, per 15 min
    "T1016": 15,  # Case management, per 15 min
}
GROUP_MODIFIERS = ["HQ"]          # group setting — excluded from hour math

CAP_HOURS           = 12          # Tier 2 threshold (hours per clinician-day)
FLAG_PCTILE         = 0.95        # red flag = at or above cohort p95
SIGNATURE_PCTILE    = 0.90        # reference signature = features ≥ p90
MIN_FLAGS           = 3           # lookalike needs ≥ this many red flags …
MIN_SIGNATURE_MATCH = 0.60        # … AND ≥ 60% of the reference's signature

OUTPUT_PATH = (
    "/Workspace/Users/{ROOT}/Projects/all_projects_predictions/LLM/"
    f"lookalikes_{REF_SERVICING}_{REF_BILLING}_multiwindow.xlsx"
)

TODAY = date.today()
T1    = "exposure_tier1_over_24h"
T2    = f"exposure_tier2_over_{CAP_HOURS}h"
T2_PM = f"{T2}_per_month"


# ─────────────────────────────────────────────────────────────────────────────
# LOAD & PREPARE (once)
# ─────────────────────────────────────────────────────────────────────────────
bhp_ids = (
    df_scoring_master[df_scoring_master[CATEGORY_COL] == CATEGORY][SPROV_COL]
    .drop_duplicates().tolist()
)
bhp_spark = spark.createDataFrame([(p,) for p in bhp_ids], schema=[SPROV_COL])

df_base = (
    spark.table(SERVICE_TABLE)
    .join(bhp_spark, on=SPROV_COL, how="inner")
    .withColumn("srv_date", F.to_date(F.col(SRV_BEG_COL).cast("string"), "yyyyMMdd"))
    .withColumn("srv_end",  F.to_date(F.col(SRV_END_COL).cast("string"), "yyyyMMdd"))
    .withColumn("month",    F.trunc("srv_date", "month"))
    .withColumn("qty",  F.col(QTY_COL).cast("double"))
    .withColumn("paid", F.col(PMT_COL).cast("double"))      # avoids Decimal issues
    .withColumn(PROC_COL, F.upper(F.trim(F.col(PROC_COL))))
)

mods = [c for c in MODIFIER_COLS if c in df_base.columns]
df_base = df_base.withColumn(
    "billing_signature",
    F.concat_ws("|", F.col(PROC_COL),
                *[F.coalesce(F.col(c), F.lit("")) for c in mods],
                F.col(QTY_COL).cast("string")),
)

is_group = F.lit(False)
for c in mods:
    is_group = is_group | F.coalesce(F.col(c), F.lit("")).isin(GROUP_MODIFIERS)

unit_map   = F.create_map(*[F.lit(x) for kv in UNIT_MINUTES.items() for x in kv])
single_day = F.coalesce(F.col("srv_end"), F.col("srv_date")) == F.col("srv_date")

df_base = (
    df_base
    .withColumn("unit_minutes", unit_map[F.col(PROC_COL)])
    .withColumn("is_timed", F.col("unit_minutes").isNotNull() & ~is_group & single_day)
    .withColumn("timed_minutes",
                F.when(F.col("is_timed"), F.col("qty") * F.col("unit_minutes")).otherwise(0.0))
    .select(SPROV_COL, BPROV_COL, MEMBER_COL, PROC_COL, "srv_date", "month",
            "paid", "is_timed", "timed_minutes", "billing_signature")
    .cache()
)

# Clinician-day table — the grain where "impossible" is meaningful (date-local)
cday = (
    df_base.groupBy(SPROV_COL, "srv_date")
    .agg(F.sum("timed_minutes").alias("mins"),
         F.countDistinct(MEMBER_COL).alias("members"))
    .withColumn("hours", F.col("mins") / 60)
    .withColumn("r_24",  F.when(F.col("mins") > 1440,
                                (F.col("mins") - 1440) / F.col("mins")).otherwise(0.0))
    .withColumn("r_cap", F.when(F.col("mins") > CAP_HOURS * 60,
                                (F.col("mins") - CAP_HOURS * 60) / F.col("mins")).otherwise(0.0))
    .cache()
)

# Exposure allocated pro-rata to claim lines (date-local) — rolls up to either
# the clinician or the entity that was paid.
lines_x = (
    df_base.filter("is_timed")
    .join(cday.select(SPROV_COL, "srv_date", "r_24", "r_cap"), [SPROV_COL, "srv_date"])
    .select(SPROV_COL, BPROV_COL, "srv_date", "paid",
            (F.col("paid") * F.col("r_24")).alias("x24_paid"),
            (F.col("paid") * F.col("r_cap")).alias("xcap_paid"),
            F.when(F.col("r_cap") > 0, F.col("paid")).otherwise(0.0).alias("paid_on_cap_day"))
    .cache()
)

print(f"Cohort (full history): {df_base.select(SPROV_COL).distinct().count()} servicing / "
      f"{df_base.select(BPROV_COL).distinct().count()} billing providers")


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def cutoff_for(months):
    return None if months is None else (TODAY - relativedelta(months=months)).isoformat()

def by_date(sdf, cutoff):
    return sdf if cutoff is None else sdf.filter(F.col("srv_date") >= F.lit(cutoff))

def exposure_by(lx, key):
    return lx.groupBy(key).agg(
        F.sum("paid").alias("timed_paid"),
        F.sum("paid_on_cap_day").alias(f"paid_on_days_over_{CAP_HOURS}h"),
        F.sum("xcap_paid").alias(T2),
        F.sum("x24_paid").alias(T1),
    )

def pattern_features(d, KEY, unit_keys):
    """Cookie-cutter, repeated packages, session length, code concentration.
    unit_keys defines a 'day': for entities use (entity, clinician) so a large
    entity isn't flagged just for having many clinicians on standard sessions."""
    day_keys = unit_keys + ["srv_date"]
    dup = (d.groupBy(*day_keys, "billing_signature")
             .agg(F.countDistinct(MEMBER_COL).alias("n")).filter("n >= 2"))
    days     = d.select(*day_keys).distinct().groupBy(KEY).count().withColumnRenamed("count", "dd_tot")
    dup_days = dup.select(*day_keys).distinct().groupBy(KEY).count().withColumnRenamed("count", "dd")
    mem      = d.groupBy(KEY).agg(F.countDistinct(MEMBER_COL).alias("m"))
    mem_dup  = (d.join(dup.select(*day_keys, "billing_signature"), day_keys + ["billing_signature"])
                 .groupBy(KEY).agg(F.countDistinct(MEMBER_COL).alias("md")))
    pkg = (d.groupBy(*unit_keys, MEMBER_COL, "srv_date")
            .agg(F.concat_ws("||", F.sort_array(F.collect_list("billing_signature"))).alias("p")))
    rpt = (pkg.groupBy(*unit_keys, MEMBER_COL, "p").count().filter("count > 1")
              .groupBy(KEY).agg(F.countDistinct(MEMBER_COL).alias("mr")))
    sess = (d.groupBy(KEY, MEMBER_COL, "srv_date").agg(F.sum("timed_minutes").alias("mins"))
             .filter("mins > 0").groupBy(KEY).agg(F.avg(F.col("mins") / 60).alias("avg_session_hours")))
    w = Window.partitionBy(KEY)
    top = (d.groupBy(KEY, PROC_COL).agg(F.sum("paid").alias("cp"))
            .withColumn("tot", F.sum("cp").over(w))
            .withColumn("rk", F.row_number().over(w.orderBy(F.desc("cp"))))
            .filter("rk = 1")
            .select(KEY, F.col(PROC_COL).alias("top_proc_code"),
                    (F.col("cp") / F.col("tot") * 100).alias("top_code_paid_share")))
    return (
        days.join(dup_days, KEY, "left").join(mem, KEY, "left").join(mem_dup, KEY, "left")
            .join(rpt, KEY, "left").join(sess, KEY, "left").join(top, KEY, "left")
            .fillna(0, subset=["dd", "md", "mr"])
            .withColumn("pct_days_dup_pattern",   F.col("dd") / F.col("dd_tot") * 100)
            .withColumn("pct_members_in_dup",     F.col("md") / F.col("m") * 100)
            .withColumn("pct_members_repeat_pkg", F.col("mr") / F.col("m") * 100)
            .select(KEY, "pct_days_dup_pattern", "pct_members_in_dup",
                    "pct_members_repeat_pkg", "avg_session_hours",
                    "top_proc_code", "top_code_paid_share")
    )


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE TABLES (per window)
# ─────────────────────────────────────────────────────────────────────────────
SERV_FEATURES = [
    "avg_timed_hours_day", "max_timed_hours_day",
    "pct_days_over_8h", f"pct_days_over_{CAP_HOURS}h", "pct_days_over_24h",
    "avg_members_day", "avg_session_hours", "paid_per_member",
    "pct_days_dup_pattern", "pct_members_in_dup", "pct_members_repeat_pkg",
    "top_code_paid_share",
]
BILL_FEATURES = [
    "avg_clinician_hours_day",
    f"pct_clin_days_over_{CAP_HOURS}h", "pct_clin_days_over_24h",
    f"pct_timed_paid_on_days_over_{CAP_HOURS}h",
    "pct_clinicians_outlier", "pct_paid_via_outlier_clinicians",
    "avg_members_per_clin_day", "avg_session_hours", "paid_per_member",
    "pct_days_dup_pattern", "pct_members_in_dup", "pct_members_repeat_pkg",
    "top_code_paid_share",
]

def finish(pdf):
    pdf["paid_per_member"] = pdf["total_paid"] / pdf["total_members"]
    pdf[T2_PM] = pdf[T2].fillna(0) / pdf["active_months"]
    return pdf

def build_servicing(dw, cw, lw):
    s_days = cw.groupBy(SPROV_COL).agg(
        F.count("*").alias("active_days"),
        F.avg("hours").alias("avg_timed_hours_day"),
        F.max("hours").alias("max_timed_hours_day"),
        (F.avg((F.col("hours") > 8).cast("int")) * 100).alias("pct_days_over_8h"),
        (F.avg((F.col("hours") > CAP_HOURS).cast("int")) * 100).alias(f"pct_days_over_{CAP_HOURS}h"),
        (F.avg((F.col("hours") > 24).cast("int")) * 100).alias("pct_days_over_24h"),
        F.avg("members").alias("avg_members_day"),
    )
    s_tot = dw.groupBy(SPROV_COL).agg(
        F.sum("paid").alias("total_paid"),
        F.countDistinct(MEMBER_COL).alias("total_members"),
        F.countDistinct(BPROV_COL).alias("n_billing_entities"),
        F.countDistinct("month").alias("active_months"),
    )
    return finish(
        s_days.join(s_tot, SPROV_COL, "left")
              .join(pattern_features(dw, SPROV_COL, [SPROV_COL]), SPROV_COL, "left")
              .join(exposure_by(lw, SPROV_COL), SPROV_COL, "left")
              .toPandas()
    )

def build_billing(dw, cw, lw, serv_pd, min_days):
    bday = dw.select(BPROV_COL, SPROV_COL, "srv_date").distinct().join(cw, [SPROV_COL, "srv_date"])
    b_days = bday.groupBy(BPROV_COL).agg(
        F.countDistinct("srv_date").alias("active_days"),
        F.count("*").alias("clinician_days"),
        F.avg("hours").alias("avg_clinician_hours_day"),
        (F.avg((F.col("hours") > CAP_HOURS).cast("int")) * 100).alias(f"pct_clin_days_over_{CAP_HOURS}h"),
        (F.avg((F.col("hours") > 24).cast("int")) * 100).alias("pct_clin_days_over_24h"),
        F.avg("members").alias("avg_members_per_clin_day"),
    )
    b_tot = dw.groupBy(BPROV_COL).agg(
        F.sum("paid").alias("total_paid"),
        F.countDistinct(MEMBER_COL).alias("total_members"),
        F.countDistinct(SPROV_COL).alias("n_servicing_providers"),
        F.countDistinct("month").alias("active_months"),
    )
    bill = finish(
        b_days.join(b_tot, BPROV_COL, "left")
              .join(pattern_features(dw, BPROV_COL, [BPROV_COL, SPROV_COL]), BPROV_COL, "left")
              .join(exposure_by(lw, BPROV_COL), BPROV_COL, "left")
              .toPandas()
    )
    bill[f"pct_timed_paid_on_days_over_{CAP_HOURS}h"] = np.where(
        bill["timed_paid"] > 0,
        bill[f"paid_on_days_over_{CAP_HOURS}h"] / bill["timed_paid"] * 100, 0.0)

    # Share of each entity's clinicians (and dollars) that are high-hour outliers
    active = serv_pd[serv_pd.active_days >= min_days]
    p95 = active["avg_timed_hours_day"].quantile(0.95)
    outliers = set(active.loc[active.avg_timed_hours_day >= p95, SPROV_COL])
    pairs = dw.groupBy(BPROV_COL, SPROV_COL).agg(F.sum("paid").alias("pair_paid")).toPandas()
    pairs["is_out"] = pairs[SPROV_COL].isin(outliers)
    pairs["out_paid"] = np.where(pairs["is_out"], pairs["pair_paid"], 0.0)
    g = pairs.groupby(BPROV_COL).agg(pct_clinicians_outlier=("is_out", "mean"),
                                     out_paid=("out_paid", "sum"),
                                     pair_paid=("pair_paid", "sum")).reset_index()
    g["pct_clinicians_outlier"] *= 100
    g["pct_paid_via_outlier_clinicians"] = np.where(
        g["pair_paid"] > 0, g["out_paid"] / g["pair_paid"] * 100, 0.0)
    return bill.merge(g[[BPROV_COL, "pct_clinicians_outlier", "pct_paid_via_outlier_clinicians"]],
                      on=BPROV_COL, how="left")


# ─────────────────────────────────────────────────────────────────────────────
# SCORING (pandas — one row per provider, so this is small)
# ─────────────────────────────────────────────────────────────────────────────
def score_lookalikes(tbl, key, ref_id, features, min_days):
    t = tbl[(tbl.active_days >= min_days) | (tbl[key] == ref_id)].copy().reset_index(drop=True)
    if (t[key] == ref_id).sum() == 0:
        raise ValueError(f"reference {ref_id} has no claims in this window")

    P   = t[features].rank(pct=True)
    ref = P[t[key] == ref_id].iloc[0]
    signature = [f for f in features if pd.notna(ref[f]) and ref[f] >= SIGNATURE_PCTILE]

    flags = P[features] >= FLAG_PCTILE
    t["n_red_flags"]       = flags.sum(axis=1)
    t["red_flags"]         = flags.apply(lambda r: ", ".join(r.index[r]), axis=1)
    t["signature_match"]   = (P[signature] >= SIGNATURE_PCTILE).mean(axis=1) if signature else np.nan
    t["similarity_to_ref"] = 1 - (P[features] - ref[features]).abs().mean(axis=1)
    for f in features:
        t[f"pctile_{f}"] = (P[f] * 100).round(1)
    t["is_reference"] = t[key] == ref_id
    t["is_hit"] = ((t.n_red_flags >= MIN_FLAGS) & (t.signature_match >= MIN_SIGNATURE_MATCH)) \
                  | t.is_reference
    return t.sort_values(["is_hit", "signature_match", "similarity_to_ref"],
                         ascending=False).reset_index(drop=True), signature


# =============================================================================
#  RUN ALL WINDOWS
# =============================================================================
LEVELS = {
    "serv": dict(key=SPROV_COL, ref=REF_SERVICING, feats=SERV_FEATURES),
    "bill": dict(key=BPROV_COL, ref=REF_BILLING,   feats=BILL_FEATURES),
}
scored     = {"serv": {}, "bill": {}}   # level -> suffix -> scored DataFrame
signatures = []

for label, sfx, months in TIME_WINDOWS:
    cut = cutoff_for(months)
    print(f"\n{'=' * 70}\n  {label}" + (f"  (≥ {cut})" if cut else "") + f"\n{'=' * 70}")
    dw, cw, lw = by_date(df_base, cut), by_date(cday, cut), by_date(lines_x, cut)

    serv_pd = build_servicing(dw, cw, lw)
    bill_pd = build_billing(dw, cw, lw, serv_pd, MIN_ACTIVE_DAYS[sfx])

    for lvl, tbl in (("serv", serv_pd), ("bill", bill_pd)):
        cfg = LEVELS[lvl]
        try:
            s, sig = score_lookalikes(tbl, cfg["key"], cfg["ref"], cfg["feats"], MIN_ACTIVE_DAYS[sfx])
        except ValueError as e:
            print(f"  ⚠ {lvl}: {e} — window skipped for this level")
            continue
        scored[lvl][sfx] = s
        signatures.append([label, lvl, cfg["ref"], len(sig), ", ".join(sig)])
        print(f"  {lvl}: {len(s)} active, {int(s.is_hit.sum()) - 1} lookalikes "
              f"| signature: {sig}")


# =============================================================================
#  RECOVERY-WINDOW DOLLARS (the only dollars reported)
# =============================================================================
rec_cut = cutoff_for(RECOVERY_LOOKBACK_MONTHS)
rec_label = "full history" if rec_cut is None else f"service dates ≥ {rec_cut}"
d_rec, l_rec = by_date(df_base, rec_cut), by_date(lines_x, rec_cut)

def recovery_table(key):
    tot = d_rec.groupBy(key).agg(F.sum("paid").alias("total_paid"))
    return (tot.join(exposure_by(l_rec, key), key, "left").toPandas()
               .rename(columns=lambda c: c if c == key else f"recovery_{c}"))

rec = {"serv": recovery_table(SPROV_COL), "bill": recovery_table(BPROV_COL)}

# Relationships (any time) for network flags
pairs_all = df_base.groupBy(BPROV_COL, SPROV_COL).agg(F.sum("paid").alias("pair_paid_all")).toPandas()
pairs_rec = d_rec.groupBy(BPROV_COL, SPROV_COL).agg(F.sum("paid").alias("pair_paid_recovery")).toPandas()


# =============================================================================
#  COMBINE: one row per provider flagged in ANY window
# =============================================================================
SFX = [sfx for _, sfx, _ in TIME_WINDOWS]
TREND_RANK = {
    "Persistent": 1,                        # flagged in all three windows
    "Emerging": 2,                          # flagged recently, not over full history
    "Active - intermittent": 2,             # flagged 6M + Full, not 1Y
    "Recently normalized": 3,               # flagged 1Y, clear in 6M
    "Historical - pattern stopped": 4,      # flagged Full only, still billing, now clear
    "Historical - no recent billing": 4,    # flagged earlier, inactive in 6M
    "Unclassified (window skipped)": 5,
}

def trend_label(r):
    st = {s: r[f"status_{s}"] for s in ("Full", "1Y", "6M")}
    if "not run" in st.values():
        return "Unclassified (window skipped)"
    f, y, s = (st[x] == "FLAG" for x in ("Full", "1Y", "6M"))
    if f and y and s:
        return "Persistent"
    if s and not f:
        return "Emerging"
    if s:
        return "Active - intermittent"
    if st["6M"] == "inactive":
        return "Historical - no recent billing"
    if y:
        return "Recently normalized"
    return "Historical - pattern stopped"

def combine(lvl):
    key, ref = LEVELS[lvl]["key"], LEVELS[lvl]["ref"]
    wins = scored[lvl]
    ids = sorted(set().union(*[set(s.loc[s.is_hit, key]) for s in wins.values()])) if wins else []
    out = pd.DataFrame({key: ids})

    for sfx in SFX:
        s = wins.get(sfx)
        if s is None:
            out[f"status_{sfx}"] = "not run"
            continue
        hit_ids, active_ids = set(s.loc[s.is_hit, key]), set(s[key])
        out[f"status_{sfx}"] = np.where(out[key].isin(hit_ids), "FLAG",
                                np.where(out[key].isin(active_ids), "clear", "inactive"))
        cols = {"signature_match": f"sig_match_{sfx}", "similarity_to_ref": f"similarity_{sfx}",
                "n_red_flags": f"n_flags_{sfx}", "red_flags": f"red_flags_{sfx}",
                T2_PM: f"{T2_PM}_{sfx}"}
        out = out.merge(s[[key] + list(cols)].rename(columns=cols), on=key, how="left")

    out["windows_flagged"] = (out[[f"status_{x}" for x in SFX]] == "FLAG").sum(axis=1)
    out["trend"] = out.apply(trend_label, axis=1)
    out["trend_rank"] = out["trend"].map(TREND_RANK)
    out["is_reference"] = out[key] == ref

    # Top code from the widest window available
    for sfx in SFX:
        if sfx in wins:
            out = out.merge(wins[sfx][[key, "top_proc_code"]], on=key, how="left")
            break

    out = out.merge(rec[lvl], on=key, how="left")

    if lvl == "serv":
        via_ref = set(pairs_all.loc[pairs_all[BPROV_COL] == REF_BILLING, SPROV_COL])
        out["bills_through_ref_entity"] = out[key].isin(via_ref)
    else:
        ref_ents = set(pairs_all.loc[pairs_all[SPROV_COL] == REF_SERVICING, BPROV_COL])
        out["ref_clinician_bills_here"] = out[key].isin(ref_ents)

    lead = [key, "is_reference", "trend", "windows_flagged"] + [f"status_{x}" for x in SFX] + \
           [f"recovery_{c}" for c in ("total_paid", "timed_paid", T2, T1)] + ["top_proc_code"]
    rest = [c for c in out.columns if c not in lead and c != "trend_rank"]
    return (out.sort_values(["is_reference", "trend_rank", f"recovery_{T1}", f"recovery_{T2}"],
                            ascending=[False, True, False, False])
               [lead + rest].reset_index(drop=True))

serv_comb = combine("serv")
bill_comb = combine("bill")

network_links = (
    pairs_rec[pairs_rec[SPROV_COL].isin(serv_comb[SPROV_COL]) &
              pairs_rec[BPROV_COL].isin(bill_comb[BPROV_COL])]
    .merge(serv_comb[[SPROV_COL, "trend"]].rename(columns={"trend": "servicing_trend"}), on=SPROV_COL)
    .merge(bill_comb[[BPROV_COL, "trend"]].rename(columns={"trend": "billing_trend"}), on=BPROV_COL)
    .sort_values("pair_paid_recovery", ascending=False)
)


# =============================================================================
#  EXPOSURE SUMMARY (recovery window, by trend; reference shown separately)
# =============================================================================
def summarize(comb, level_name):
    rows = []
    ref_rows = comb[comb.is_reference]
    for _, r in ref_rows.iterrows():
        rows.append([level_name, "REFERENCE", 1, r[f"recovery_total_paid"],
                     r[f"recovery_{T2}"], r[f"recovery_{T1}"]])
    g = (comb[~comb.is_reference].groupby("trend")
         .agg(n=("trend", "size"), paid=("recovery_total_paid", "sum"),
              t2=(f"recovery_{T2}", "sum"), t1=(f"recovery_{T1}", "sum"))
         .reset_index())
    g["rank"] = g["trend"].map(TREND_RANK)
    for _, r in g.sort_values("rank").iterrows():
        rows.append([level_name, r.trend, r.n, r.paid, r.t2, r.t1])
    return rows

summary = pd.DataFrame(
    summarize(serv_comb, "Servicing") + summarize(bill_comb, "Billing"),
    columns=["Level", "Trend", "Providers", "Total paid",
             f"Tier 2 (> {CAP_HOURS}h)", "Tier 1 (> 24h)"],
)
print(f"\nExposure — {rec_label} (servicing and billing rows are the SAME dollars; do not add)")
print(summary.to_string(index=False))


# =============================================================================
#  EXCEL EXPORT
# =============================================================================
criteria = pd.DataFrame([
    ["References", f"Servicing {REF_SERVICING}; Billing {REF_BILLING}"],
    ["Cohort", f"{CATEGORY} servicing providers and every billing entity that bills for them"],
    ["Windows", "; ".join(f"{l} (min {MIN_ACTIVE_DAYS[s]} active days)" for l, s, _ in TIME_WINDOWS)],
    ["Dollar window", f"Recovery lookback: {rec_label}. Dollars are NOT summed across windows (windows are nested)."],
    ["Red flag", f"Feature ≥ cohort {int(FLAG_PCTILE*100)}th percentile within that window"],
    ["Reference signature", f"Features where the reference is ≥ {int(SIGNATURE_PCTILE*100)}th percentile (see Signatures sheet)"],
    ["Inclusion rule", f"≥ {MIN_FLAGS} red flags AND ≥ {int(MIN_SIGNATURE_MATCH*100)}% of reference signature, in at least one window"],
    ["Status values", "FLAG = lookalike in that window; clear = active but not a lookalike; inactive = below activity floor or no claims"],
    ["Trend", "Persistent (all 3) > Emerging (6M, not Full) > Recently normalized (1Y, clear 6M) > Historical"],
    ["Similarity", "1 − mean absolute percentile gap to the reference (behavior only, not size)"],
    ["Timed codes", ", ".join(f"{k}={v}min" for k, v in UNIT_MINUTES.items())],
    ["Hour math exclusions", f"Group modifiers {GROUP_MODIFIERS}; multi-day service spans"],
    ["Tier 1 exposure", "Paid on timed minutes beyond 24h in one clinician-day (pro-rated to lines)"],
    ["Tier 2 exposure", f"Paid on timed minutes beyond {CAP_HOURS}h in one clinician-day (includes Tier 1)"],
    ["Per-month exposure", f"{T2} ÷ active months in that window — comparable across windows"],
    ["Caveat", "Exposure is implausible/impossible time on the face of the claims, not an adjudicated overpayment"],
    ["Caveat", "Servicing and billing exposure are the same dollars viewed two ways — do not add them"],
    ["Caveat", "A servicing ID that is actually a group/organization NPI will show false 'impossible' days — verify"],
], columns=["Item", "Definition"])

sig_df = pd.DataFrame(signatures, columns=["Window", "Level", "Reference", "N features", "Signature features"])

with pd.ExcelWriter(OUTPUT_PATH, engine="openpyxl") as w:
    criteria.to_excel(w,      sheet_name="Criteria",            index=False)
    summary.to_excel(w,       sheet_name="Exposure_Summary",    index=False)
    serv_comb.to_excel(w,     sheet_name="Servicing_Combined",  index=False)
    bill_comb.to_excel(w,     sheet_name="Billing_Combined",    index=False)
    network_links.to_excel(w, sheet_name="Network_Links",       index=False)
    sig_df.to_excel(w,        sheet_name="Signatures",          index=False)
    for lvl, name in (("serv", "Serv"), ("bill", "Bill")):
        for sfx in SFX:
            if sfx in scored[lvl]:
                scored[lvl][sfx].to_excel(w, sheet_name=f"{name}_Scored_{sfx}", index=False)

print(f"\n✓ Saved to {OUTPUT_PATH}")
print(f"  Servicing lookalikes (any window): {int((~serv_comb.is_reference).sum())}")
print(f"  Billing lookalikes   (any window): {int((~bill_comb.is_reference).sum())}")

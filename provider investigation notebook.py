# =============================================================================
# COMPREHENSIVE PROVIDER INVESTIGATION NOTEBOOK
# Target: Servicing Provider vs BH Professional Peer Cohort
# =============================================================================
# Produces a full peer-comparison analysis for an individual servicing
# provider across THREE time windows: Full history, Last 12 months, Last 6
# months.  Each window gets its own set of Excel sheets so the investigator
# can compare trends over time.
#
#   PART A — Overall Provider Profile (all codes combined)
#     1. Daily volume & hour-threshold exceedance
#     2. Unique members per day
#     3. Billing entity breakdown
#     4. Identical billing-pattern detection (cookie-cutter)
#     5. Repeated package frequency
#     6. Percentile rank in cohort
#     7. Consolidated comparison table
#
#   PART B — Per-Procedure-Code Deep Dive
#     8.  Per-code volume, hours, thresholds, per-session hours
#     9.  Per-code pattern detection & repeated packages
#     10. Billing entity breakdown per code
#
#   PART C — Excel Export (multi-sheet workbook, one set per time window)
#
# Parameterised: change TARGET_PROVIDER and rerun for any provider.
#
# [MERGE] Changes (every section and measure is kept):
#   - Hours come only from TIMED codes (UNIT_MINUTES), excluding group sessions
#     (GROUP_MODIFIERS) and multi-day lines — the same definition as the lookalike
#     screen. Before, every unit of every code counted as 15 minutes, which overstated
#     hours and "% days over N hours". Unit counts are unchanged (all codes).
#   - Per-code hours are blank for codes that aren't billed in time units.
#   - Billing-entity sheets add implied (timed) hours.
#   - New sheet per window: Cross_Biller_Dups — the same member + procedure + date
#     paid to the target's line AND to another billing entity.
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
# CONFIGURATION  — EDIT THESE
# ─────────────────────────────────────────────────────────────────────────────
TARGET_PROVIDER = "004526"
CATEGORY        = "BH PROFESSIONAL"
SERVICE_TABLE   = "main.prod_input.all_paid_services"

SPROV_COL    = "Servicing_Provider_ID"
BPROV_COL    = "Billing_Provider_ID"
MEMBER_COL   = "MEMBER_KEY"
PROC_COL     = "PROC_CD"
QTY_COL      = "QUANTITY_PAID"
PMT_COL      = "PMT_AMT"
SRV_BEG_COL  = "SRV_BEG_CALENDAR_KEY"
SRV_END_COL  = "SRV_END_CALENDAR_KEY"
MODIFIER_COLS = ["MODIFIER_1", "MODIFIER_2", "MODIFIER_3", "MODIFIER_4"]
CATEGORY_COL  = "Category"

HOUR_THRESHOLDS = [4, 8, 12, 16, 20, 24]

# [MERGE] Timed codes: minutes per paid unit. Only these feed hours. Same list as the lookalike
# screen — VERIFY against the AHCCCS fee schedule before sharing results.
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
COB_LOB_VALUES  = ["MSP"]         # duplicates under these lines of business are coordination of benefits

OUTPUT_PATH = (
    f"/Workspace/Users/{{ROOT}}/Projects/all_projects_predictions/LLM/"
    f"provider_{TARGET_PROVIDER}_comprehensive_analysis.xlsx"
)

# Time windows: (label, suffix for sheet names, months_back or None for full)
# Both target AND peers are filtered to the same window for apples-to-apples.
TODAY = date.today()
TIME_WINDOWS = [
    ("Full History",    "Full", None),
    ("Last 12 Months",  "1Y",  12),
    ("Last 6 Months",   "6M",  6),
]


# ─────────────────────────────────────────────────────────────────────────────
# LOAD & PREPARE THE BH PROFESSIONAL COHORT (FULL — windowed later)
# ─────────────────────────────────────────────────────────────────────────────
df_all = spark.table(SERVICE_TABLE)

# df_scoring_master is a pandas DF — convert provider list to Spark
bhp_provider_ids = (
    df_scoring_master[df_scoring_master[CATEGORY_COL] == CATEGORY][SPROV_COL]
    .drop_duplicates()
    .tolist()
)
bhp_providers_spark = spark.createDataFrame(
    [(pid,) for pid in bhp_provider_ids],
    schema=[SPROV_COL],
)

df_bhp_full = df_all.join(bhp_providers_spark, on=SPROV_COL, how="inner")

# Parse service date & cast quantity
df_bhp_full = (
    df_bhp_full
    .withColumn("srv_date", F.to_date(F.col(SRV_BEG_COL).cast("string"), "yyyyMMdd"))
    .withColumn("srv_end",  F.to_date(F.col(SRV_END_COL).cast("string"), "yyyyMMdd"))   # [MERGE]
    .withColumn("qty", F.col(QTY_COL).cast("double"))
    .withColumn(PROC_COL, F.upper(F.trim(F.col(PROC_COL))))                             # [MERGE]
)

# Build billing signature
mod_cols_available = [c for c in MODIFIER_COLS if c in df_bhp_full.columns]
if mod_cols_available:
    sig_expr = F.concat_ws(
        "|", F.col(PROC_COL),
        *[F.coalesce(F.col(c), F.lit("")) for c in mod_cols_available],
        F.col(QTY_COL).cast("string"),
    )
else:
    sig_expr = F.concat_ws("|", F.col(PROC_COL), F.col(QTY_COL).cast("string"))

df_bhp_full = df_bhp_full.withColumn("billing_signature", sig_expr)

# [MERGE] Timed minutes: timed code, not a group session, single-day line
_is_group = F.lit(False)
for c in mod_cols_available:
    _is_group = _is_group | F.coalesce(F.col(c), F.lit("")).isin(GROUP_MODIFIERS)
_unit_map = F.create_map(*[F.lit(x) for kv in UNIT_MINUTES.items() for x in kv])
df_bhp_full = (
    df_bhp_full
    .withColumn("unit_minutes", _unit_map[F.col(PROC_COL)])
    .withColumn("is_timed", F.col("unit_minutes").isNotNull() & ~_is_group &
                (F.coalesce(F.col("srv_end"), F.col("srv_date")) == F.col("srv_date")))
    .withColumn("timed_minutes", F.when(F.col("is_timed"), F.col("qty") * F.col("unit_minutes")).otherwise(0.0))
)

# [MERGE] All paid lines (not cohort-filtered), for the cross-biller duplicate check
df_all_n = (
    df_all
    .withColumn("srv_date", F.to_date(F.col(SRV_BEG_COL).cast("string"), "yyyyMMdd"))
    .withColumn("qty", F.col(QTY_COL).cast("double"))
    .withColumn("paid", F.col(PMT_COL).cast("double"))
    .withColumn(PROC_COL, F.upper(F.trim(F.col(PROC_COL))))
)

# Cohort overview
cohort_size = df_bhp_full.select(SPROV_COL).distinct().count()
target_claims = df_bhp_full.filter(F.col(SPROV_COL) == TARGET_PROVIDER).count()
print(f"BH Professional cohort: {cohort_size} providers, {target_claims:,} target claim lines")

date_range = (
    df_bhp_full.filter(F.col(SPROV_COL) == TARGET_PROVIDER)
    .agg(F.min("srv_date").alias("earliest"), F.max("srv_date").alias("latest"))
    .toPandas()
)
print(f"Service date range: {date_range['earliest'].values[0]} – {date_range['latest'].values[0]}")


# =============================================================================
#  HELPERS
# =============================================================================
def safe_val(df, col, default="N/A"):
    try:
        v = df[col].values[0]
        if pd.isna(v):
            return default
        return v
    except Exception:
        return default

def _rnd(v, n):
    """[MERGE] round() that tolerates blanks (hours are blank for codes not billed in time units)."""
    try:
        return None if v is None or pd.isna(v) else round(float(v), n)
    except (TypeError, ValueError):
        return None

def ratio(target_val, peer_val):
    try:
        t, p = float(target_val), float(peer_val)
        return round(t / p, 1) if p > 0 else None
    except Exception:
        return None


# [MERGE] ── Cross-biller duplicates for the target ──────────────────────────────
def cross_biller_dups(cutoff_str, target_col, target):
    """Services on the target's lines (same member + procedure + date) that were ALSO paid
    to another billing entity. Searches all paid lines, not just the cohort, so an agency
    listing itself as the servicing provider is still found."""
    scope = df_all_n if cutoff_str is None else df_all_n.filter(F.col("srv_date") >= F.lit(cutoff_str))
    k = [MEMBER_COL, PROC_COL, "srv_date"]
    keys = scope.filter(F.col(target_col) == target).select(*k).distinct()
    elig = scope.join(keys, k).filter(F.col("paid") > 0)
    if COB_LOB_VALUES and "line_of_business" in scope.columns:
        elig = elig.filter(F.col("line_of_business").isNull() | ~F.col("line_of_business").isin(COB_LOB_VALUES))
    per_b = elig.groupBy(*k, BPROV_COL).agg(
        F.sum("paid").alias("b_paid"), F.sum("qty").alias("b_units"),
        F.array_sort(F.collect_set(SPROV_COL)).alias("b_servicers"))
    return (per_b.groupBy(*k).agg(
                F.count(F.lit(1)).alias("n_billers"),
                F.concat_ws(", ", F.array_sort(F.collect_list(BPROV_COL))).alias("billers"),
                F.concat_ws(", ", F.array_sort(F.array_distinct(F.flatten(F.collect_list("b_servicers"))))).alias("servicers"),
                F.concat_ws(", ", F.collect_list(F.col("b_units").cast("string"))).alias("units_by_biller"),
                (F.countDistinct("b_units") == 1).alias("same_units"),
                F.round(F.sum("b_paid"), 2).alias("paid_total"),
                F.round(F.sum("b_paid") - F.max("b_paid"), 2).alias("possible_duplicate_paid"))
            .filter(F.col("n_billers") >= 2)
            .orderBy("srv_date")
            .toPandas())


# =============================================================================
#  CORE ANALYSIS FUNCTION — runs for one time window
# =============================================================================
def run_analysis(df_bhp, window_label, window_suffix):
    """
    Run the full Part A + Part B analysis on a (possibly filtered) df_bhp.
    Returns a dict of pandas DataFrames keyed by sheet name (with suffix).
    """
    KEY = SPROV_COL
    TGT = TARGET_PROVIDER
    sheets = {}

    target_count = df_bhp.filter(F.col(KEY) == TGT).count()
    if target_count == 0:
        print(f"  ⚠ Target {TGT} has no claims in window '{window_label}' — skipping")
        return sheets

    cohort_n = df_bhp.select(KEY).distinct().count()
    print(f"\n  Window '{window_label}': {cohort_n} providers, {target_count:,} target lines")

    # ─── 1. Daily volume & thresholds ────────────────────────────────────
    daily_agg = (
        df_bhp.groupBy(KEY, "srv_date")
        .agg(
            F.sum("qty").alias("daily_units"),
            F.countDistinct(MEMBER_COL).alias("daily_members"),
            F.sum("timed_minutes").alias("daily_timed_minutes"),              # [MERGE]
        )
        .withColumn("daily_hours", F.col("daily_timed_minutes") / 60)         # [MERGE] timed codes only
    )

    provider_daily = (
        daily_agg.groupBy(KEY)
        .agg(
            F.count("srv_date").alias("active_days"),
            F.mean("daily_units").alias("avg_units_per_day"),
            F.expr("percentile_approx(daily_units, 0.5)").alias("median_units_per_day"),
            F.max("daily_units").alias("max_units_per_day"),
            F.mean("daily_hours").alias("avg_hours_per_day"),
            F.max("daily_hours").alias("max_hours_per_day"),
            F.mean("daily_members").alias("avg_members_per_day"),
            F.max("daily_members").alias("max_members_per_day"),
        )
    )

    for hrs in HOUR_THRESHOLDS:
        col_name = f"pct_days_over_{hrs}h"
        thresh_df = (
            daily_agg
            .withColumn("over", (F.col("daily_hours") > hrs).cast("int"))   # [MERGE] timed hours
            .groupBy(KEY)
            .agg((F.sum("over") / F.count("srv_date") * 100).alias(col_name))
        )
        provider_daily = provider_daily.join(thresh_df, on=KEY, how="left")

    target_daily_pd = provider_daily.filter(F.col(KEY) == TGT).toPandas()
    peer_daily = provider_daily.filter(F.col(KEY) != TGT)

    peer_daily_med = peer_daily.agg(
        F.expr("percentile_approx(avg_units_per_day,  0.5)").alias("peer_med_avg_units_day"),
        F.expr("percentile_approx(avg_hours_per_day,  0.5)").alias("peer_med_avg_hours_day"),
        F.expr("percentile_approx(max_hours_per_day,  0.5)").alias("peer_med_max_hours_day"),
        F.expr("percentile_approx(avg_members_per_day,0.5)").alias("peer_med_avg_members_day"),
        *[F.expr(f"percentile_approx(pct_days_over_{h}h, 0.5)").alias(f"peer_med_pct_over_{h}h")
          for h in HOUR_THRESHOLDS],
    ).toPandas()

    # ─── 2. Unique members per day ───────────────────────────────────────
    members_daily = (
        df_bhp.groupBy(KEY, "srv_date")
        .agg(F.countDistinct(MEMBER_COL).alias("unique_members"))
    )
    members_summary = (
        members_daily.groupBy(KEY)
        .agg(
            F.mean("unique_members").alias("avg_members_day"),
            F.expr("percentile_approx(unique_members, 0.5)").alias("median_members_day"),
            F.max("unique_members").alias("max_members_day"),
        )
    )
    target_members_pd = members_summary.filter(F.col(KEY) == TGT).toPandas()
    peer_members_med = (
        members_summary.filter(F.col(KEY) != TGT)
        .agg(
            F.expr("percentile_approx(avg_members_day, 0.5)").alias("peer_med_avg_members"),
            F.expr("percentile_approx(max_members_day, 0.5)").alias("peer_med_max_members"),
        ).toPandas()
    )

    # ─── 3. Billing entity breakdown ─────────────────────────────────────
    billing_breakdown_pd = (
        df_bhp.filter(F.col(KEY) == TGT)
        .groupBy(BPROV_COL)
        .agg(
            F.count("*").alias("line_count"),
            F.sum(PMT_COL).alias("total_paid"),
            F.sum("qty").alias("total_units"),
            F.countDistinct(MEMBER_COL).alias("unique_members"),
            F.countDistinct("srv_date").alias("active_days"),
            F.round(F.sum("timed_minutes") / 60, 2).alias("implied_timed_hours"),   # [MERGE]
        )
        .orderBy(F.desc("total_paid"))
        .toPandas()
    )
    sheets[f"Billing_Entities_{window_suffix}"] = billing_breakdown_pd

    entity_daily_rows = []
    for _, erow in billing_breakdown_pd.iterrows():
        bp_id = erow[BPROV_COL]
        bp_daily = (
            df_bhp
            .filter((F.col(KEY) == TGT) & (F.col(BPROV_COL) == bp_id))
            .groupBy("srv_date")
            .agg(
                F.sum("qty").alias("daily_units"),
                F.countDistinct(MEMBER_COL).alias("daily_members"),
                F.sum("timed_minutes").alias("daily_timed_minutes"),      # [MERGE]
            )
            .withColumn("daily_hours", F.col("daily_timed_minutes") / 60) # [MERGE] timed codes only
        )
        bp_stats = bp_daily.agg(
            F.mean("daily_units").alias("avg_units_day"),
            F.mean("daily_hours").alias("avg_hours_day"),
            F.max("daily_hours").alias("max_hours_day"),
            F.mean("daily_members").alias("avg_members_day"),
        ).toPandas().iloc[0]
        entity_daily_rows.append({
            "billing_provider": bp_id,
            "avg_units_day":   round(bp_stats["avg_units_day"],   1),
            "avg_hours_day":   round(bp_stats["avg_hours_day"],   1),
            "max_hours_day":   round(bp_stats["max_hours_day"],   1),
            "avg_members_day": round(bp_stats["avg_members_day"], 1),
        })
    entity_daily_df = pd.DataFrame(entity_daily_rows)
    sheets[f"Entity_Daily_{window_suffix}"] = entity_daily_df

    # ─── 4. Cookie-cutter detection ──────────────────────────────────────
    dup_patterns = (
        df_bhp.groupBy(KEY, "srv_date", "billing_signature")
        .agg(
            F.countDistinct(MEMBER_COL).alias("members_with_pattern"),
            F.count("*").alias("lines_with_pattern"),
        )
    )
    dup_flagged = dup_patterns.filter(F.col("members_with_pattern") >= 2)

    total_days = df_bhp.groupBy(KEY).agg(F.countDistinct("srv_date").alias("total_active_days"))
    days_with_dups = dup_flagged.groupBy(KEY).agg(F.countDistinct("srv_date").alias("days_with_dup_pattern"))
    dup_summary = (
        total_days.join(days_with_dups, on=KEY, how="left")
        .fillna(0, subset=["days_with_dup_pattern"])
        .withColumn("pct_days_with_dup_pattern",
                    F.col("days_with_dup_pattern") / F.col("total_active_days") * 100)
    )
    avg_dup_size = (
        dup_flagged.groupBy(KEY)
        .agg(
            F.mean("members_with_pattern").alias("avg_members_per_dup_pattern"),
            F.max("members_with_pattern").alias("max_members_per_dup_pattern"),
        )
    )
    dup_summary = dup_summary.join(avg_dup_size, on=KEY, how="left")

    members_in_dup = (
        df_bhp.join(
            dup_flagged.select(KEY, "srv_date", "billing_signature"),
            on=[KEY, "srv_date", "billing_signature"], how="inner",
        )
        .groupBy(KEY)
        .agg(F.countDistinct(MEMBER_COL).alias("members_in_dup_patterns"))
    )
    total_members_prov = df_bhp.groupBy(KEY).agg(F.countDistinct(MEMBER_COL).alias("total_members"))
    dup_summary = (
        dup_summary
        .join(total_members_prov, on=KEY, how="left")
        .join(members_in_dup, on=KEY, how="left")
        .fillna(0, subset=["members_in_dup_patterns"])
        .withColumn("pct_members_in_dup_patterns",
                    F.col("members_in_dup_patterns") / F.col("total_members") * 100)
    )

    target_dup_pd = dup_summary.filter(F.col(KEY) == TGT).toPandas()
    peer_dup_med = (
        dup_summary.filter(F.col(KEY) != TGT)
        .agg(
            F.expr("percentile_approx(pct_days_with_dup_pattern, 0.5)").alias("peer_med_pct_dup_days"),
            F.expr("percentile_approx(avg_members_per_dup_pattern, 0.5)").alias("peer_med_avg_dup_size"),
            F.expr("percentile_approx(pct_members_in_dup_patterns, 0.5)").alias("peer_med_pct_members_dup"),
        ).toPandas()
    )

    # ─── 5. Repeated packages ────────────────────────────────────────────
    daily_packages = (
        df_bhp.groupBy(KEY, MEMBER_COL, "srv_date")
        .agg(F.sort_array(F.collect_list("billing_signature")).alias("daily_package"))
        .withColumn("package_str", F.concat_ws("||", "daily_package"))
    )
    package_repeats = (
        daily_packages.groupBy(KEY, MEMBER_COL, "package_str")
        .agg(F.count("srv_date").alias("times_repeated"))
        .filter(F.col("times_repeated") > 1)
    )
    total_members_all = df_bhp.groupBy(KEY).agg(F.countDistinct(MEMBER_COL).alias("total_members"))
    members_with_rpt = (
        package_repeats.groupBy(KEY)
        .agg(
            F.countDistinct(MEMBER_COL).alias("members_with_repeated_pkg"),
            F.mean("times_repeated").alias("avg_repeat_count"),
            F.max("times_repeated").alias("max_repeat_count"),
        )
    )
    repeat_summary = (
        total_members_all.join(members_with_rpt, on=KEY, how="left")
        .fillna(0, subset=["members_with_repeated_pkg"])
        .withColumn("pct_members_with_repeats",
                    F.col("members_with_repeated_pkg") / F.col("total_members") * 100)
    )

    target_rpt_pd = repeat_summary.filter(F.col(KEY) == TGT).toPandas()
    peer_rpt_med = (
        repeat_summary.filter(F.col(KEY) != TGT)
        .agg(
            F.expr("percentile_approx(pct_members_with_repeats, 0.5)").alias("peer_med_pct_rpt"),
            F.expr("percentile_approx(avg_repeat_count, 0.5)").alias("peer_med_avg_rpt_count"),
        ).toPandas()
    )

    # ─── 6. Percentile rank ──────────────────────────────────────────────
    provider_ranks = provider_daily
    for metric in ["avg_units_per_day", "avg_hours_per_day", "max_hours_per_day"]:
        w = Window.orderBy(F.col(metric))
        provider_ranks = provider_ranks.withColumn(
            f"pctile_{metric}", F.percent_rank().over(w) * 100
        )
    target_ranks_pd = (
        provider_ranks.filter(F.col(KEY) == TGT)
        .select(KEY,
                "avg_units_per_day", "pctile_avg_units_per_day",
                "avg_hours_per_day", "pctile_avg_hours_per_day",
                "max_hours_per_day", "pctile_max_hours_per_day")
        .toPandas()
    )

    # ─── 7. Consolidated comparison table ────────────────────────────────
    metrics = [
        ("Avg units/day (all codes)",                                    # [MERGE] label: all codes, not only timed
         safe_val(target_daily_pd, "avg_units_per_day"),
         safe_val(peer_daily_med,  "peer_med_avg_units_day")),
        ("Avg timed hours/day",                                          # [MERGE] label
         safe_val(target_daily_pd, "avg_hours_per_day"),
         safe_val(peer_daily_med,  "peer_med_avg_hours_day")),
        ("Max timed hours in a single day",                              # [MERGE] label
         safe_val(target_daily_pd, "max_hours_per_day"),
         safe_val(peer_daily_med,  "peer_med_max_hours_day")),
        ("Avg unique members/day",
         safe_val(target_daily_pd, "avg_members_per_day"),
         safe_val(peer_daily_med,  "peer_med_avg_members_day")),
    ]
    for hrs in HOUR_THRESHOLDS:
        metrics.append((
            f"% days > {hrs} timed hours",                               # [MERGE] label
            safe_val(target_daily_pd, f"pct_days_over_{hrs}h"),
            safe_val(peer_daily_med,  f"peer_med_pct_over_{hrs}h"),
        ))
    metrics += [
        ("% days with identical patterns",
         safe_val(target_dup_pd, "pct_days_with_dup_pattern"),
         safe_val(peer_dup_med,  "peer_med_pct_dup_days")),
        ("Avg members per dup pattern",
         safe_val(target_dup_pd, "avg_members_per_dup_pattern"),
         safe_val(peer_dup_med,  "peer_med_avg_dup_size")),
        ("% members in dup patterns",
         safe_val(target_dup_pd, "pct_members_in_dup_patterns"),
         safe_val(peer_dup_med,  "peer_med_pct_members_dup")),
        ("% members with repeated packages",
         safe_val(target_rpt_pd, "pct_members_with_repeats"),
         safe_val(peer_rpt_med,  "peer_med_pct_rpt")),
    ]
    for m_col, label in [
        ("pctile_avg_units_per_day", "Percentile: avg units/day"),
        ("pctile_avg_hours_per_day", "Percentile: avg hours/day"),
        ("pctile_max_hours_per_day", "Percentile: max hours/day"),
    ]:
        metrics.append((label, safe_val(target_ranks_pd, m_col), "—"))

    consolidated_df = pd.DataFrame(
        metrics, columns=["Metric", f"Provider {TGT}", "Peer Median"]
    )
    consolidated_df["Ratio"] = consolidated_df.apply(
        lambda r: (
            f"{r[f'Provider {TGT}'] / r['Peer Median']:.1f}x"
            if isinstance(r[f"Provider {TGT}"], (int, float))
            and isinstance(r["Peer Median"], (int, float))
            and r["Peer Median"] > 0
            else "—"
        ), axis=1,
    )
    consolidated_df.insert(0, "Window", window_label)
    sheets[f"Overall_{window_suffix}"] = consolidated_df

    print(f"    Part A complete")
    print(consolidated_df.to_string(index=False))

    # ─── 8. Per-proc-code analysis ───────────────────────────────────────
    target_proc_codes = (
        df_bhp.filter(F.col(KEY) == TGT)
        .select(PROC_COL).distinct()
        .rdd.flatMap(lambda x: x).collect()
    )
    print(f"    {len(target_proc_codes)} procedure codes to analyze")

    def analyze_proc_code(proc_code):
        df_proc = df_bhp.filter(F.col(PROC_COL) == proc_code)

        prov_agg = (
            df_proc.groupBy(KEY)
            .agg(
                F.sum("qty").alias("total_units"),
                F.countDistinct(MEMBER_COL).alias("unique_members"),
                F.countDistinct("srv_date").alias("active_days"),
                F.count("*").alias("line_count"),
                F.sum(PMT_COL).alias("total_paid"),
                F.sum("timed_minutes").alias("timed_minutes"),                         # [MERGE]
            )
            .withColumn("units_per_day",    F.col("total_units") / F.col("active_days"))
            # [MERGE] hours only for codes billed in time units (blank otherwise)
            .withColumn("hours",            F.when(F.col("timed_minutes") > 0, F.col("timed_minutes") / 60))
            .withColumn("hours_per_day",    F.col("hours") / F.col("active_days"))
            .withColumn("units_per_member", F.col("total_units") / F.col("unique_members"))
            .withColumn("paid_per_unit",    F.col("total_paid") / F.col("total_units"))
        )

        proc_daily = (
            df_proc.groupBy(KEY, "srv_date")
            .agg(
                F.sum("qty").alias("daily_units"),
                F.countDistinct(MEMBER_COL).alias("daily_members"),
                F.sum("timed_minutes").alias("daily_timed_minutes"),                   # [MERGE]
            )
            # [MERGE] timed hours; blank for non-timed codes so their thresholds stay blank, not 0%
            .withColumn("daily_hours", F.when(F.col("daily_timed_minutes") > 0, F.col("daily_timed_minutes") / 60))
        )
        for hrs in HOUR_THRESHOLDS:
            t_df = (
                proc_daily
                .withColumn("over", (F.col("daily_hours") > hrs).cast("int"))         # [MERGE] timed hours
                .groupBy(KEY)
                .agg((F.sum("over") / F.count("srv_date") * 100).alias(f"pct_days_over_{hrs}h"))
            )
            prov_agg = prov_agg.join(t_df, on=KEY, how="left")

        # Per-session hours (provider × member × date)
        per_session = (
            df_proc.groupBy(KEY, MEMBER_COL, "srv_date")
            .agg(F.sum("timed_minutes").alias("_m"))                                   # [MERGE]
            .withColumn("session_hours", F.when(F.col("_m") > 0, F.col("_m") / 60))    # [MERGE] timed only
        )
        session_stats = (
            per_session.groupBy(KEY)
            .agg(
                F.mean("session_hours").alias("avg_session_hours"),
                F.expr("percentile_approx(session_hours, 0.5)").alias("median_session_hours"),
                F.max("session_hours").alias("max_session_hours"),
            )
        )
        prov_agg = prov_agg.join(session_stats, on=KEY, how="left")

        # Members per day
        mpd = (
            proc_daily.groupBy(KEY)
            .agg(
                F.mean("daily_members").alias("avg_members_day"),
                F.max("daily_members").alias("max_members_day"),
            )
        )
        prov_agg = prov_agg.join(mpd, on=KEY, how="left")

        target_row = prov_agg.filter(F.col(KEY) == TGT).toPandas()
        if target_row.empty:
            return None
        peer = prov_agg.filter(F.col(KEY) != TGT)
        peer_count = peer.count()
        if peer_count == 0:
            return None

        peer_cols = [
            "total_units", "unique_members", "active_days",
            "units_per_day", "hours_per_day", "units_per_member",
            "total_paid", "paid_per_unit",
            "avg_session_hours", "median_session_hours", "max_session_hours",
            "avg_members_day", "max_members_day",
        ] + [f"pct_days_over_{h}h" for h in HOUR_THRESHOLDS]
        peer_med = peer.agg(
            *[F.expr(f"percentile_approx({c}, 0.5)").alias(f"peer_med_{c}") for c in peer_cols]
        ).toPandas()

        rank_df = prov_agg.withColumn(
            "pct_rank_units_day", F.percent_rank().over(Window.orderBy("units_per_day")) * 100
        )
        target_rank = rank_df.filter(F.col(KEY) == TGT).select("pct_rank_units_day").toPandas()

        t = target_row.iloc[0]
        p = peer_med.iloc[0]

        result = {
            "proc_code": proc_code, "peer_providers": peer_count,
            "target_total_units": t.get("total_units"),
            "peer_med_total_units": p.get("peer_med_total_units"),
            "ratio_total_units": ratio(t.get("total_units"), p.get("peer_med_total_units")),
            "target_unique_members": t.get("unique_members"),
            "peer_med_unique_members": p.get("peer_med_unique_members"),
            "target_active_days": t.get("active_days"),
            "peer_med_active_days": p.get("peer_med_active_days"),
            "target_units_per_day": _rnd(t.get("units_per_day", 0), 2),
            "peer_med_units_per_day": _rnd(p.get("peer_med_units_per_day", 0), 2),
            "ratio_units_per_day": ratio(t.get("units_per_day"), p.get("peer_med_units_per_day")),
            "target_hours_per_day": _rnd(t.get("hours_per_day", 0), 2),
            "peer_med_hours_per_day": _rnd(p.get("peer_med_hours_per_day", 0), 2),
            "target_units_per_member": _rnd(t.get("units_per_member", 0), 2),
            "peer_med_units_per_member": _rnd(p.get("peer_med_units_per_member", 0), 2),
            "ratio_units_per_member": ratio(t.get("units_per_member"), p.get("peer_med_units_per_member")),
            "target_total_paid": _rnd(t.get("total_paid", 0), 2),
            "peer_med_total_paid": _rnd(p.get("peer_med_total_paid", 0), 2),
            "ratio_total_paid": ratio(t.get("total_paid"), p.get("peer_med_total_paid")),
            "target_paid_per_unit": _rnd(t.get("paid_per_unit", 0), 2),
            "peer_med_paid_per_unit": _rnd(p.get("peer_med_paid_per_unit", 0), 2),
            "target_avg_session_hours": _rnd(t.get("avg_session_hours", 0), 2),
            "peer_med_avg_session_hours": _rnd(p.get("peer_med_avg_session_hours", 0), 2),
            "target_median_session_hours": _rnd(t.get("median_session_hours", 0), 2),
            "target_max_session_hours": _rnd(t.get("max_session_hours", 0), 2),
            "target_avg_members_day": _rnd(t.get("avg_members_day", 0), 2),
            "peer_med_avg_members_day": _rnd(p.get("peer_med_avg_members_day", 0), 2),
            "target_max_members_day": t.get("max_members_day"),
            **{f"target_pct_over_{h}h": _rnd(t.get(f"pct_days_over_{h}h", 0), 1) for h in HOUR_THRESHOLDS},
            **{f"peer_med_pct_over_{h}h": _rnd(p.get(f"peer_med_pct_days_over_{h}h", 0), 1) for h in HOUR_THRESHOLDS},
            "pctile_units_per_day": round(target_rank["pct_rank_units_day"].values[0], 1) if not target_rank.empty else None,
            "target_line_count": t.get("line_count"),
        }
        return result

    code_results = []
    for i, pc in enumerate(sorted(target_proc_codes)):
        print(f"      [{i+1}/{len(target_proc_codes)}] {pc}")
        row = analyze_proc_code(pc)
        if row:
            code_results.append(row)

    code_summary_df = (
        pd.DataFrame(code_results)
        .sort_values("ratio_total_units", ascending=False)
        .reset_index(drop=True)
    )

    # ─── 9. Per-code patterns ────────────────────────────────────────────
    def analyze_patterns_per_code(proc_code):
        df_proc = df_bhp.filter(F.col(PROC_COL) == proc_code)

        dup_pat = (
            df_proc.groupBy(KEY, "srv_date", "billing_signature")
            .agg(F.countDistinct(MEMBER_COL).alias("members_with_pattern"))
        ).filter(F.col("members_with_pattern") >= 2)

        total_d = df_proc.groupBy(KEY).agg(F.countDistinct("srv_date").alias("total_days"))
        days_d  = dup_pat.groupBy(KEY).agg(F.countDistinct("srv_date").alias("days_dup"))
        dup_code = (
            total_d.join(days_d, on=KEY, how="left")
            .fillna(0, subset=["days_dup"])
            .withColumn("pct_days_dup", F.col("days_dup") / F.col("total_days") * 100)
        )

        pkg = (
            df_proc.groupBy(KEY, MEMBER_COL, "srv_date")
            .agg(F.sort_array(F.collect_list("billing_signature")).alias("pkg"))
            .withColumn("pkg_str", F.concat_ws("||", "pkg"))
        )
        rpt = (
            pkg.groupBy(KEY, MEMBER_COL, "pkg_str")
            .agg(F.count("srv_date").alias("times_rpt"))
            .filter(F.col("times_rpt") > 1)
        )
        total_m = df_proc.groupBy(KEY).agg(F.countDistinct(MEMBER_COL).alias("total_members"))
        m_rpt   = rpt.groupBy(KEY).agg(F.countDistinct(MEMBER_COL).alias("members_rpt"))
        rpt_code = (
            total_m.join(m_rpt, on=KEY, how="left")
            .fillna(0, subset=["members_rpt"])
            .withColumn("pct_members_rpt", F.col("members_rpt") / F.col("total_members") * 100)
        )

        t_dup = dup_code.filter(F.col(KEY) == TGT).toPandas()
        t_rpt = rpt_code.filter(F.col(KEY) == TGT).toPandas()
        if t_dup.empty and t_rpt.empty:
            return None

        p_dup = dup_code.filter(F.col(KEY) != TGT).agg(
            F.expr("percentile_approx(pct_days_dup, 0.5)").alias("peer_med")
        ).toPandas()
        p_rpt = rpt_code.filter(F.col(KEY) != TGT).agg(
            F.expr("percentile_approx(pct_members_rpt, 0.5)").alias("peer_med")
        ).toPandas()

        return {
            "proc_code": proc_code,
            "target_pct_days_dup":      round(safe_val(t_dup, "pct_days_dup", 0), 1),
            "peer_med_pct_days_dup":    round(safe_val(p_dup, "peer_med", 0), 1),
            "target_pct_members_rpt":   round(safe_val(t_rpt, "pct_members_rpt", 0), 1),
            "peer_med_pct_members_rpt": round(safe_val(p_rpt, "peer_med", 0), 1),
        }

    pattern_results = []
    for i, pc in enumerate(sorted(target_proc_codes)):
        row = analyze_patterns_per_code(pc)
        if row:
            pattern_results.append(row)
    pattern_df = pd.DataFrame(pattern_results)

    # Merge patterns into code summary
    if not pattern_df.empty and not code_summary_df.empty:
        code_full_df = code_summary_df.merge(pattern_df, on="proc_code", how="left")
    else:
        code_full_df = code_summary_df.copy()

    sheets[f"All_Proc_Codes_{window_suffix}"] = code_full_df

    # Anomalous codes
    anomalous_df = code_full_df[
        (code_full_df["ratio_total_units"].fillna(0) > 2)
        | (code_full_df.get("ratio_units_per_day", pd.Series(dtype=float)).fillna(0) > 2)
        | (code_full_df.get("ratio_total_paid", pd.Series(dtype=float)).fillna(0) > 2)
    ].copy()
    sheets[f"Anomalous_GT2x_{window_suffix}"] = anomalous_df

    # ─── 10. Billing entity breakdown per code ───────────────────────────
    billing_by_code_pd = (
        df_bhp.filter(F.col(KEY) == TGT)
        .groupBy(PROC_COL, BPROV_COL)
        .agg(
            F.sum("qty").alias("total_units"),
            F.sum(PMT_COL).alias("total_paid"),
            F.countDistinct(MEMBER_COL).alias("unique_members"),
            F.countDistinct("srv_date").alias("active_days"),
            F.count("*").alias("line_count"),
        )
        .orderBy(PROC_COL, F.desc("total_units"))
        .toPandas()
    )
    sheets[f"Entity_By_Code_{window_suffix}"] = billing_by_code_pd

    if not pattern_df.empty:
        sheets[f"Patterns_{window_suffix}"] = pattern_df

    print(f"    Part B complete — {len(code_results)} codes, {len(anomalous_df)} anomalous")

    return sheets


# =============================================================================
#  RUN ALL TIME WINDOWS
# =============================================================================
all_sheets = {}

for window_label, window_suffix, months_back in TIME_WINDOWS:
    print(f"\n{'=' * 80}")
    print(f"  TIME WINDOW: {window_label}")
    print(f"{'=' * 80}")

    if months_back is not None:
        cutoff = TODAY - relativedelta(months=months_back)
        cutoff_str = cutoff.isoformat()
        df_bhp_window = df_bhp_full.filter(F.col("srv_date") >= F.lit(cutoff_str))
        print(f"  Cutoff date: {cutoff_str}")
    else:
        df_bhp_window = df_bhp_full

    window_sheets = run_analysis(df_bhp_window, window_label, window_suffix)
    all_sheets.update(window_sheets)

    # [MERGE] cross-biller duplicates for the target in this window
    dups = cross_biller_dups(cutoff_str if months_back is not None else None, SPROV_COL, TARGET_PROVIDER)
    print(f"  Cross-biller duplicate services: {len(dups):,} "
          f"(same units: {int(dups['same_units'].sum()) if len(dups) else 0}) | possible duplicate $: "
          f"{dups['possible_duplicate_paid'].sum() if len(dups) else 0:,.2f}")
    # an empty sheet would be skipped below; write a note so "none found" is visible
    all_sheets[f"Cross_Biller_Dups_{window_suffix}"] = dups if len(dups) else pd.DataFrame(
        {"note": [f"No cross-biller duplicate services for {TARGET_PROVIDER} in {window_label}"]})


# =============================================================================
#  PART C — EXCEL EXPORT
# =============================================================================
print(f"\n{'=' * 80}")
print("SAVING TO EXCEL")
print(f"{'=' * 80}")

# Order sheets logically: Overall first, then proc codes, then detail sheets
# grouped by window
sheet_order = []
for _, ws, _ in TIME_WINDOWS:
    sheet_order.append(f"Overall_{ws}")
for _, ws, _ in TIME_WINDOWS:
    sheet_order.append(f"All_Proc_Codes_{ws}")
for _, ws, _ in TIME_WINDOWS:
    sheet_order.append(f"Anomalous_GT2x_{ws}")
for _, ws, _ in TIME_WINDOWS:
    for prefix in ["Billing_Entities", "Entity_Daily", "Entity_By_Code", "Patterns",
                   "Cross_Biller_Dups"]:                                              # [MERGE]
        sheet_order.append(f"{prefix}_{ws}")

with pd.ExcelWriter(OUTPUT_PATH, engine="openpyxl") as writer:
    for sheet_name in sheet_order:
        if sheet_name in all_sheets and not all_sheets[sheet_name].empty:
            # Excel sheet names max 31 chars
            safe_name = sheet_name[:31]
            all_sheets[sheet_name].to_excel(writer, sheet_name=safe_name, index=False)

# List what was written
written = [s[:31] for s in sheet_order if s in all_sheets and not all_sheets[s].empty]
print(f"\n✓ Saved to: {OUTPUT_PATH}")
print(f"  {len(written)} sheets:")
for s in written:
    print(f"    - {s}")
print("\nDone.")

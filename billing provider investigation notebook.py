# =============================================================================
# COMPREHENSIVE BILLING PROVIDER INVESTIGATION NOTEBOOK
# Target: Billing Provider vs BH Professional Billing-Provider Peers
# =============================================================================
# Mirror of the servicing-provider notebook, but keyed on BILLING provider.
#
# The peer cohort = all billing providers that submit claims for ≥1 servicing
# provider in the target category. This is the right lens when the entity
# under investigation is the billing organisation (e.g. Alliance / 246427),
# not the individual clinician.
#
# Produces a full peer-comparison analysis across THREE time windows:
#   Full History, Last 12 Months, Last 6 Months
# Each window gets its own set of Excel sheets.
#
#   PART A — Overall Billing-Provider Profile (all codes combined)
#     1. Daily volume & hour-threshold exceedance
#     2. Unique members per day
#     3. Servicing provider breakdown (who bills through this entity)
#     4. Identical billing-pattern detection (cookie-cutter)
#     5. Repeated package frequency
#     6. Percentile rank in cohort
#     7. Consolidated comparison table
#
#   PART B — Per-Procedure-Code Deep Dive
#     8.  Per-code volume, hours, thresholds, per-session hours
#     9.  Per-code pattern detection & repeated packages
#     10. Servicing provider breakdown per code
#
#   PART C — Excel Export (multi-sheet workbook, one set per time window)
#
# Parameterised: change TARGET_BILLING_PROVIDER and rerun for any entity.
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
TARGET_BILLING_PROVIDER = "246427"
CATEGORY               = "BH PROFESSIONAL"
SERVICE_TABLE           = "main.prod_input.all_paid_services"

# Column mappings
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

# Hour thresholds (units are 15-min; 4 units = 1 hour)
HOUR_THRESHOLDS = [4, 8, 12, 16, 20, 24]

# Output path
OUTPUT_PATH = (
    f"/Workspace/Users/{{ROOT}}/Projects/all_projects_predictions/LLM/"
    f"billing_provider_{TARGET_BILLING_PROVIDER}_comprehensive_analysis.xlsx"
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

# Step 1: Get servicing providers in the BH Professional category
# (df_scoring_master is a pandas DF — convert to Spark to avoid _jdf error)
bhp_sprov_ids = (
    df_scoring_master[df_scoring_master[CATEGORY_COL] == CATEGORY][SPROV_COL]
    .drop_duplicates()
    .tolist()
)
bhp_sprovs_spark = spark.createDataFrame(
    [(pid,) for pid in bhp_sprov_ids],
    schema=[SPROV_COL],
)

# Step 2: Filter claims to those involving a BH Professional servicing provider
# Then the billing-provider cohort = all billing providers on those claims
df_bhp_full = df_all.join(bhp_sprovs_spark, on=SPROV_COL, how="inner")

# Parse service date & cast quantity
df_bhp_full = (
    df_bhp_full
    .withColumn("srv_date", F.to_date(F.col(SRV_BEG_COL).cast("string"), "yyyyMMdd"))
    .withColumn("qty", F.col(QTY_COL).cast("double"))
)

# Build billing signature expression
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

# Cohort overview
bp_cohort_size = df_bhp_full.select(BPROV_COL).distinct().count()
target_claims  = df_bhp_full.filter(F.col(BPROV_COL) == TARGET_BILLING_PROVIDER).count()
print(f"BH Professional billing-provider cohort: {bp_cohort_size} billing providers")
print(f"Target billing provider {TARGET_BILLING_PROVIDER}: {target_claims:,} claim lines")

date_range = (
    df_bhp_full.filter(F.col(BPROV_COL) == TARGET_BILLING_PROVIDER)
    .agg(F.min("srv_date").alias("earliest"), F.max("srv_date").alias("latest"))
    .toPandas()
)
print(f"Service date range: {date_range['earliest'].values[0]} – {date_range['latest'].values[0]}")

sprov_count = (
    df_bhp_full.filter(F.col(BPROV_COL) == TARGET_BILLING_PROVIDER)
    .select(SPROV_COL).distinct().count()
)
print(f"Servicing providers billing through {TARGET_BILLING_PROVIDER}: {sprov_count}")


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

def ratio(target_val, peer_val):
    try:
        t, p = float(target_val), float(peer_val)
        return round(t / p, 1) if p > 0 else None
    except Exception:
        return None


# =============================================================================
#  CORE ANALYSIS FUNCTION — runs for one time window
# =============================================================================
def run_analysis(df_bhp, window_label, window_suffix):
    """
    Run the full Part A + Part B analysis on a (possibly filtered) df_bhp.
    Returns a dict of pandas DataFrames keyed by sheet name (with suffix).
    Keyed on BILLING provider (BPROV_COL).
    """
    KEY = BPROV_COL
    TGT = TARGET_BILLING_PROVIDER
    sheets = {}

    target_count = df_bhp.filter(F.col(KEY) == TGT).count()
    if target_count == 0:
        print(f"  ⚠ Target {TGT} has no claims in window '{window_label}' — skipping")
        return sheets

    cohort_n = df_bhp.select(KEY).distinct().count()
    print(f"\n  Window '{window_label}': {cohort_n} billing providers, {target_count:,} target lines")

    # ─── 1. Daily volume & thresholds ────────────────────────────────────
    daily_agg = (
        df_bhp
        .groupBy(KEY, "srv_date")
        .agg(
            F.sum("qty").alias("daily_units"),
            F.countDistinct(MEMBER_COL).alias("daily_members"),
            F.countDistinct(SPROV_COL).alias("daily_servicing_provs"),
        )
        .withColumn("daily_hours", F.col("daily_units") * 15 / 60)
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
            F.mean("daily_servicing_provs").alias("avg_servicing_provs_per_day"),
            F.max("daily_servicing_provs").alias("max_servicing_provs_per_day"),
        )
    )

    # Overall aggregates (total units, members, paid, servicing providers)
    provider_totals = (
        df_bhp.groupBy(KEY)
        .agg(
            F.sum("qty").alias("total_units"),
            F.sum(PMT_COL).alias("total_paid"),
            F.countDistinct(MEMBER_COL).alias("total_unique_members"),
            F.countDistinct(SPROV_COL).alias("total_servicing_providers"),
            F.countDistinct(PROC_COL).alias("total_unique_proc_codes"),
            F.count("*").alias("total_line_count"),
        )
    )
    provider_daily = provider_daily.join(provider_totals, on=KEY, how="left")

    # Threshold columns
    for hrs in HOUR_THRESHOLDS:
        col_name = f"pct_days_over_{hrs}h"
        thresh_df = (
            daily_agg
            .withColumn("over", (F.col("daily_units") > hrs * 4).cast("int"))
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
        F.expr("percentile_approx(total_units,         0.5)").alias("peer_med_total_units"),
        F.expr("percentile_approx(total_paid,          0.5)").alias("peer_med_total_paid"),
        F.expr("percentile_approx(total_unique_members,0.5)").alias("peer_med_total_members"),
        F.expr("percentile_approx(total_servicing_providers, 0.5)").alias("peer_med_total_sprovs"),
        F.expr("percentile_approx(avg_servicing_provs_per_day, 0.5)").alias("peer_med_avg_sprovs_day"),
        *[
            F.expr(f"percentile_approx(pct_days_over_{h}h, 0.5)").alias(f"peer_med_pct_over_{h}h")
            for h in HOUR_THRESHOLDS
        ],
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
            F.expr("percentile_approx(avg_members_day,    0.5)").alias("peer_med_avg_members"),
            F.expr("percentile_approx(median_members_day, 0.5)").alias("peer_med_median_members"),
            F.expr("percentile_approx(max_members_day,    0.5)").alias("peer_med_max_members"),
        ).toPandas()
    )

    # ─── 3. Servicing provider breakdown ─────────────────────────────────
    sprov_breakdown = (
        df_bhp.filter(F.col(KEY) == TGT)
        .groupBy(SPROV_COL)
        .agg(
            F.count("*").alias("line_count"),
            F.sum(PMT_COL).alias("total_paid"),
            F.sum("qty").alias("total_units"),
            F.countDistinct(MEMBER_COL).alias("unique_members"),
            F.countDistinct("srv_date").alias("active_days"),
            F.countDistinct(PROC_COL).alias("unique_proc_codes"),
        )
        .withColumn("units_per_day", F.col("total_units") / F.col("active_days"))
        .withColumn("hours_per_day", F.col("total_units") * 15 / 60 / F.col("active_days"))
        .orderBy(F.desc("total_paid"))
    )
    sprov_breakdown_pd = sprov_breakdown.toPandas()
    sheets[f"Servicing_Provs_{window_suffix}"] = sprov_breakdown_pd

    # Per-servicing-provider daily stats (top 20)
    sprov_daily_rows = []
    for row in sprov_breakdown.limit(20).collect():
        sp_id = row[SPROV_COL]
        sp_daily = (
            df_bhp
            .filter((F.col(KEY) == TGT) & (F.col(SPROV_COL) == sp_id))
            .groupBy("srv_date")
            .agg(
                F.sum("qty").alias("daily_units"),
                F.countDistinct(MEMBER_COL).alias("daily_members"),
            )
            .withColumn("daily_hours", F.col("daily_units") * 15 / 60)
        )
        sp_stats = sp_daily.agg(
            F.mean("daily_units").alias("avg_units_day"),
            F.mean("daily_hours").alias("avg_hours_day"),
            F.max("daily_hours").alias("max_hours_day"),
            F.mean("daily_members").alias("avg_members_day"),
        ).toPandas().iloc[0]
        sprov_daily_rows.append({
            "servicing_provider": sp_id,
            "avg_units_day":   round(sp_stats["avg_units_day"],   1),
            "avg_hours_day":   round(sp_stats["avg_hours_day"],   1),
            "max_hours_day":   round(sp_stats["max_hours_day"],   1),
            "avg_members_day": round(sp_stats["avg_members_day"], 1),
        })

    sprov_daily_df = pd.DataFrame(sprov_daily_rows)
    sheets[f"SProv_Daily_{window_suffix}"] = sprov_daily_df

    # Flag outlier servicing providers (>2× cohort median units/day)
    sprov_peer_median = (
        df_bhp
        .groupBy(SPROV_COL, "srv_date")
        .agg(F.sum("qty").alias("daily_units"))
        .groupBy(SPROV_COL)
        .agg(F.mean("daily_units").alias("avg_units_day"))
        .agg(F.expr("percentile_approx(avg_units_day, 0.5)").alias("sprov_median"))
        .toPandas()["sprov_median"].values[0]
    )

    outlier_sprovs = sprov_breakdown_pd[
        sprov_breakdown_pd["units_per_day"] > 2 * sprov_peer_median
    ]
    if not outlier_sprovs.empty:
        sheets[f"Outlier_SProvs_{window_suffix}"] = outlier_sprovs[
            [SPROV_COL, "total_paid", "total_units", "unique_members",
             "units_per_day", "hours_per_day"]
        ].copy()

    # ─── 4. Cookie-cutter detection ──────────────────────────────────────
    dup_patterns = (
        df_bhp
        .groupBy(KEY, "srv_date", "billing_signature")
        .agg(
            F.countDistinct(MEMBER_COL).alias("members_with_pattern"),
            F.count("*").alias("lines_with_pattern"),
        )
    )
    dup_flagged = dup_patterns.filter(F.col("members_with_pattern") >= 2)

    total_days = df_bhp.groupBy(KEY).agg(
        F.countDistinct("srv_date").alias("total_active_days")
    )
    days_with_dups = dup_flagged.groupBy(KEY).agg(
        F.countDistinct("srv_date").alias("days_with_dup_pattern")
    )
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
            F.count("*").alias("total_dup_instances"),
        )
    )
    dup_summary = dup_summary.join(avg_dup_size, on=KEY, how="left")

    members_in_dup = (
        df_bhp.join(
            dup_flagged.select(KEY, "srv_date", "billing_signature"),
            on=[KEY, "srv_date", "billing_signature"],
            how="inner",
        )
        .groupBy(KEY)
        .agg(F.countDistinct(MEMBER_COL).alias("members_in_dup_patterns"))
    )
    total_members_prov = df_bhp.groupBy(KEY).agg(
        F.countDistinct(MEMBER_COL).alias("total_members")
    )
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
            F.expr("percentile_approx(max_members_per_dup_pattern, 0.5)").alias("peer_med_max_dup_size"),
            F.expr("percentile_approx(pct_members_in_dup_patterns, 0.5)").alias("peer_med_pct_members_dup"),
        ).toPandas()
    )

    # ─── 5. Repeated packages ────────────────────────────────────────────
    daily_packages = (
        df_bhp
        .groupBy(KEY, MEMBER_COL, "srv_date")
        .agg(F.sort_array(F.collect_list("billing_signature")).alias("daily_package"))
        .withColumn("package_str", F.concat_ws("||", "daily_package"))
    )
    package_repeats = (
        daily_packages
        .groupBy(KEY, MEMBER_COL, "package_str")
        .agg(F.count("srv_date").alias("times_repeated"))
        .filter(F.col("times_repeated") > 1)
    )

    total_members_all = df_bhp.groupBy(KEY).agg(
        F.countDistinct(MEMBER_COL).alias("total_members")
    )
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
    for metric in ["avg_units_per_day", "avg_hours_per_day", "max_hours_per_day",
                   "total_paid", "total_unique_members", "total_servicing_providers"]:
        w = Window.orderBy(F.col(metric))
        provider_ranks = provider_ranks.withColumn(
            f"pctile_{metric}", F.percent_rank().over(w) * 100
        )

    target_ranks_pd = (
        provider_ranks.filter(F.col(KEY) == TGT)
        .select(
            KEY,
            "avg_units_per_day",  "pctile_avg_units_per_day",
            "avg_hours_per_day",  "pctile_avg_hours_per_day",
            "max_hours_per_day",  "pctile_max_hours_per_day",
            "total_paid",         "pctile_total_paid",
            "total_unique_members", "pctile_total_unique_members",
            "total_servicing_providers", "pctile_total_servicing_providers",
        )
        .toPandas()
    )

    # ─── 7. Consolidated comparison table ────────────────────────────────
    metrics = [
        ("Total units",
         safe_val(target_daily_pd, "total_units"),
         safe_val(peer_daily_med,  "peer_med_total_units")),
        ("Total paid ($)",
         safe_val(target_daily_pd, "total_paid"),
         safe_val(peer_daily_med,  "peer_med_total_paid")),
        ("Total unique members",
         safe_val(target_daily_pd, "total_unique_members"),
         safe_val(peer_daily_med,  "peer_med_total_members")),
        ("Total servicing providers",
         safe_val(target_daily_pd, "total_servicing_providers"),
         safe_val(peer_daily_med,  "peer_med_total_sprovs")),
        ("Avg timed units/day",
         safe_val(target_daily_pd, "avg_units_per_day"),
         safe_val(peer_daily_med,  "peer_med_avg_units_day")),
        ("Avg hours/day",
         safe_val(target_daily_pd, "avg_hours_per_day"),
         safe_val(peer_daily_med,  "peer_med_avg_hours_day")),
        ("Max hours in a single day",
         safe_val(target_daily_pd, "max_hours_per_day"),
         safe_val(peer_daily_med,  "peer_med_max_hours_day")),
        ("Avg unique members/day",
         safe_val(target_daily_pd, "avg_members_per_day"),
         safe_val(peer_daily_med,  "peer_med_avg_members_day")),
        ("Avg servicing providers/day",
         safe_val(target_daily_pd, "avg_servicing_provs_per_day"),
         safe_val(peer_daily_med,  "peer_med_avg_sprovs_day")),
    ]
    for hrs in HOUR_THRESHOLDS:
        metrics.append((
            f"% days > {hrs} hours",
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
        ("Max members per dup pattern",
         safe_val(target_dup_pd, "max_members_per_dup_pattern"),
         safe_val(peer_dup_med,  "peer_med_max_dup_size")),
        ("% members in dup patterns",
         safe_val(target_dup_pd, "pct_members_in_dup_patterns"),
         safe_val(peer_dup_med,  "peer_med_pct_members_dup")),
        ("% members with repeated packages",
         safe_val(target_rpt_pd, "pct_members_with_repeats"),
         safe_val(peer_rpt_med,  "peer_med_pct_rpt")),
    ]
    for m_col, label in [
        ("pctile_avg_units_per_day",        "Percentile: avg units/day"),
        ("pctile_avg_hours_per_day",        "Percentile: avg hours/day"),
        ("pctile_max_hours_per_day",        "Percentile: max hours/day"),
        ("pctile_total_paid",               "Percentile: total paid"),
        ("pctile_total_unique_members",     "Percentile: total members"),
        ("pctile_total_servicing_providers","Percentile: total servicing provs"),
    ]:
        metrics.append((label, safe_val(target_ranks_pd, m_col), "—"))

    consolidated_df = pd.DataFrame(
        metrics, columns=["Metric", f"Billing Provider {TGT}", "Peer Median"]
    )
    consolidated_df["Ratio"] = consolidated_df.apply(
        lambda r: (
            f"{r[f'Billing Provider {TGT}'] / r['Peer Median']:.1f}x"
            if isinstance(r[f"Billing Provider {TGT}"], (int, float))
            and isinstance(r["Peer Median"], (int, float))
            and r["Peer Median"] > 0
            else "—"
        ),
        axis=1,
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
        """Full target-vs-peer analysis for one procedure code, keyed on billing provider."""
        df_proc = df_bhp.filter(F.col(PROC_COL) == proc_code)

        prov_agg = (
            df_proc.groupBy(KEY)
            .agg(
                F.sum("qty").alias("total_units"),
                F.countDistinct(MEMBER_COL).alias("unique_members"),
                F.countDistinct("srv_date").alias("active_days"),
                F.countDistinct(SPROV_COL).alias("servicing_providers"),
                F.count("*").alias("line_count"),
                F.sum(PMT_COL).alias("total_paid"),
            )
            .withColumn("units_per_day",    F.col("total_units") / F.col("active_days"))
            .withColumn("hours",            F.col("total_units") * 15 / 60)
            .withColumn("hours_per_day",    F.col("hours") / F.col("active_days"))
            .withColumn("members_per_day",  F.col("unique_members") / F.col("active_days"))
            .withColumn("units_per_member", F.col("total_units") / F.col("unique_members"))
            .withColumn("paid_per_unit",    F.col("total_paid") / F.col("total_units"))
        )

        # Daily for thresholds
        proc_daily = (
            df_proc.groupBy(KEY, "srv_date")
            .agg(
                F.sum("qty").alias("daily_units"),
                F.countDistinct(MEMBER_COL).alias("daily_members"),
            )
            .withColumn("daily_hours", F.col("daily_units") * 15 / 60)
        )
        for hrs in HOUR_THRESHOLDS:
            t_df = (
                proc_daily
                .withColumn("over", (F.col("daily_units") > hrs * 4).cast("int"))
                .groupBy(KEY)
                .agg((F.sum("over") / F.count("srv_date") * 100).alias(f"pct_days_over_{hrs}h"))
            )
            prov_agg = prov_agg.join(t_df, on=KEY, how="left")

        # Per-session hours (billing_provider × member × date)
        per_session = (
            df_proc.groupBy(KEY, MEMBER_COL, "srv_date")
            .agg((F.sum("qty") * 15 / 60).alias("session_hours"))
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

        # Members per day distribution
        mpd_stats = (
            proc_daily.groupBy(KEY)
            .agg(
                F.mean("daily_members").alias("avg_members_day"),
                F.expr("percentile_approx(daily_members, 0.5)").alias("median_members_day"),
                F.max("daily_members").alias("max_members_day"),
            )
        )
        prov_agg = prov_agg.join(mpd_stats, on=KEY, how="left")

        # Target row
        target_row = prov_agg.filter(F.col(KEY) == TGT).toPandas()
        if target_row.empty:
            return None

        peer = prov_agg.filter(F.col(KEY) != TGT)
        peer_count = peer.count()
        if peer_count == 0:
            return None

        # Peer medians
        peer_agg_exprs = [
            F.expr(f"percentile_approx({c}, 0.5)").alias(f"peer_med_{c}")
            for c in [
                "total_units", "unique_members", "active_days", "servicing_providers",
                "units_per_day", "hours_per_day", "members_per_day",
                "units_per_member", "total_paid", "paid_per_unit",
                "avg_session_hours", "median_session_hours", "max_session_hours",
                "avg_members_day", "median_members_day", "max_members_day",
            ] + [f"pct_days_over_{h}h" for h in HOUR_THRESHOLDS]
        ]
        peer_med = peer.agg(*peer_agg_exprs).toPandas()

        # Percentile rank
        rank_df = prov_agg.withColumn(
            "pct_rank_units_day",
            F.percent_rank().over(Window.orderBy("units_per_day")) * 100,
        ).withColumn(
            "pct_rank_hours_day",
            F.percent_rank().over(Window.orderBy("hours_per_day")) * 100,
        )
        target_rank = (
            rank_df.filter(F.col(KEY) == TGT)
            .select("pct_rank_units_day", "pct_rank_hours_day")
            .toPandas()
        )

        t = target_row.iloc[0]
        p = peer_med.iloc[0]

        result = {
            "proc_code":                     proc_code,
            "peer_billing_provs_with_code":  peer_count,
            # Volume
            "target_total_units":            t.get("total_units"),
            "peer_med_total_units":          p.get("peer_med_total_units"),
            "ratio_total_units":             ratio(t.get("total_units"), p.get("peer_med_total_units")),
            "target_unique_members":         t.get("unique_members"),
            "peer_med_unique_members":       p.get("peer_med_unique_members"),
            "target_active_days":            t.get("active_days"),
            "peer_med_active_days":          p.get("peer_med_active_days"),
            "target_servicing_providers":    t.get("servicing_providers"),
            "peer_med_servicing_providers":  p.get("peer_med_servicing_providers"),
            # Daily rates
            "target_units_per_day":          round(t.get("units_per_day", 0), 2),
            "peer_med_units_per_day":        round(p.get("peer_med_units_per_day", 0), 2),
            "ratio_units_per_day":           ratio(t.get("units_per_day"), p.get("peer_med_units_per_day")),
            "target_hours_per_day":          round(t.get("hours_per_day", 0), 2),
            "peer_med_hours_per_day":        round(p.get("peer_med_hours_per_day", 0), 2),
            # Per-member
            "target_units_per_member":       round(t.get("units_per_member", 0), 2),
            "peer_med_units_per_member":     round(p.get("peer_med_units_per_member", 0), 2),
            "ratio_units_per_member":        ratio(t.get("units_per_member"), p.get("peer_med_units_per_member")),
            # Payment
            "target_total_paid":             round(t.get("total_paid", 0), 2),
            "peer_med_total_paid":           round(p.get("peer_med_total_paid", 0), 2),
            "ratio_total_paid":              ratio(t.get("total_paid"), p.get("peer_med_total_paid")),
            "target_paid_per_unit":          round(t.get("paid_per_unit", 0), 2),
            "peer_med_paid_per_unit":        round(p.get("peer_med_paid_per_unit", 0), 2),
            # Per-session hours (billing_provider × member × date grain)
            "target_avg_session_hours":      round(t.get("avg_session_hours", 0), 2),
            "peer_med_avg_session_hours":    round(p.get("peer_med_avg_session_hours", 0), 2),
            "target_median_session_hours":   round(t.get("median_session_hours", 0), 2),
            "peer_med_median_session_hours": round(p.get("peer_med_median_session_hours", 0), 2),
            "target_max_session_hours":      round(t.get("max_session_hours", 0), 2),
            "peer_med_max_session_hours":    round(p.get("peer_med_max_session_hours", 0), 2),
            # Members per day
            "target_avg_members_day":        round(t.get("avg_members_day", 0), 2),
            "peer_med_avg_members_day":      round(p.get("peer_med_avg_members_day", 0), 2),
            "target_max_members_day":        t.get("max_members_day"),
            "peer_med_max_members_day":      p.get("peer_med_max_members_day"),
            # Thresholds
            **{f"target_pct_over_{h}h": round(t.get(f"pct_days_over_{h}h", 0), 1) for h in HOUR_THRESHOLDS},
            **{f"peer_med_pct_over_{h}h": round(p.get(f"peer_med_pct_days_over_{h}h", 0), 1) for h in HOUR_THRESHOLDS},
            # Percentile rank
            "pctile_units_per_day": round(target_rank["pct_rank_units_day"].values[0], 1) if not target_rank.empty else None,
            "pctile_hours_per_day": round(target_rank["pct_rank_hours_day"].values[0], 1) if not target_rank.empty else None,
            "target_line_count":    t.get("line_count"),
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
        """Cookie-cutter and repeated-package stats for one proc code, keyed on billing provider."""
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
            "proc_code":                proc_code,
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

    # Anomalous codes (any key ratio > 2×)
    anomalous_df = code_full_df[
        (code_full_df["ratio_total_units"].fillna(0) > 2)
        | (code_full_df.get("ratio_units_per_day", pd.Series(dtype=float)).fillna(0) > 2)
        | (code_full_df.get("ratio_total_paid", pd.Series(dtype=float)).fillna(0) > 2)
    ].copy()
    sheets[f"Anomalous_GT2x_{window_suffix}"] = anomalous_df

    # ─── 10. Servicing provider breakdown per code ───────────────────────
    sprov_by_code_pd = (
        df_bhp.filter(F.col(KEY) == TGT)
        .groupBy(PROC_COL, SPROV_COL)
        .agg(
            F.sum("qty").alias("total_units"),
            F.sum(PMT_COL).alias("total_paid"),
            F.countDistinct(MEMBER_COL).alias("unique_members"),
            F.countDistinct("srv_date").alias("active_days"),
            F.count("*").alias("line_count"),
        )
        .withColumn("units_per_day", F.col("total_units") / F.col("active_days"))
        .orderBy(PROC_COL, F.desc("total_units"))
        .toPandas()
    )
    sheets[f"SProv_By_Code_{window_suffix}"] = sprov_by_code_pd

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
    for prefix in ["Servicing_Provs", "SProv_Daily", "Outlier_SProvs",
                    "SProv_By_Code", "Patterns"]:
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

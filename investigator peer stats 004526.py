# =============================================================================
# Investigator Peer Statistics — Provider 004526 (Noble, Hortense)
# Category: BH PROFESSIONAL
# =============================================================================
# Run in Databricks with access to main.prod_input.all_paid_services
# Produces detailed peer comparison statistics as requested by the investigator.
# =============================================================================

from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.window import Window
import pandas as pd
import numpy as np

spark = SparkSession.builder.getOrCreate()

# ─── Configuration ───────────────────────────────────────────────────────────
TARGET_PROVIDER = "004526"
CATEGORY = "BH PROFESSIONAL"  # adjust if your Category column uses a different label
SERVICE_TABLE = "main.prod_input.all_paid_services"

# Column name mappings — adjust if your schema differs
SPROV_COL = "Servicing_Provider_ID"   # or "SProv_ID"
BPROV_COL = "Billing_Provider_ID"     # or "BProv_ID"
MEMBER_COL = "MEMBER_KEY"
PROC_COL = "PROC_CD"
QTY_COL = "QUANTITY_PAID"
PMT_COL = "PMT_AMT"
SRV_BEG_COL = "SRV_BEG_CALENDAR_KEY"  # YYYYMMDD int or date
SRV_END_COL = "SRV_END_CALENDAR_KEY"
MODIFIER_COLS = ["MODIFIER_1", "MODIFIER_2", "MODIFIER_3", "MODIFIER_4"]
CATEGORY_COL = "Category"

# ─── Load BH Professional cohort ────────────────────────────────────────────
df_all = spark.table(SERVICE_TABLE)

# Filter to BH Professional category
# If your Category column is pre-joined, use it directly:
# df_bhp = df_all.filter(F.col(CATEGORY_COL) == CATEGORY)
#
# If Category is only in df_scoring_master, join on provider ID:
# df_bhp = df_all.join(
#     df_scoring_master.select(SPROV_COL, CATEGORY_COL).distinct(),
#     on=SPROV_COL, how="inner"
# ).filter(F.col(CATEGORY_COL) == CATEGORY)
#
# Simplest approach — filter by providers in the BH Professional scoring cohort:
bhp_providers = (
    df_scoring_master
    .filter(F.col(CATEGORY_COL) == CATEGORY)
    .select(SPROV_COL)
    .distinct()
)
df_bhp = df_all.join(bhp_providers, on=SPROV_COL, how="inner")

# Parse service date to proper date type if stored as int (YYYYMMDD)
df_bhp = (
    df_bhp
    .withColumn("srv_date", F.to_date(F.col(SRV_BEG_COL).cast("string"), "yyyyMMdd"))
    .withColumn("qty", F.col(QTY_COL).cast("double"))
)

print(f"BH Professional cohort: {df_bhp.select(SPROV_COL).distinct().count()} providers")
print(f"Target provider {TARGET_PROVIDER} claims: {df_bhp.filter(F.col(SPROV_COL) == TARGET_PROVIDER).count()}")


# =============================================================================
# 1. TIMED UNITS PER SERVICE DAY
# =============================================================================
# For BH Professional, QUANTITY_PAID typically represents 15-min timed units.
# 1 unit = 15 min → hours = qty * 15 / 60

units_per_day = (
    df_bhp
    .groupBy(SPROV_COL, "srv_date")
    .agg(
        F.sum("qty").alias("daily_units"),
        F.countDistinct(MEMBER_COL).alias("daily_unique_members"),
    )
    .withColumn("daily_hours", F.col("daily_units") * 15 / 60)
)

# Per-provider summary
provider_daily_stats = (
    units_per_day
    .groupBy(SPROV_COL)
    .agg(
        F.count("srv_date").alias("active_days"),
        F.mean("daily_units").alias("avg_units_per_day"),
        F.expr("percentile_approx(daily_units, 0.5)").alias("median_units_per_day"),
        F.max("daily_units").alias("max_units_per_day"),
        F.mean("daily_hours").alias("avg_hours_per_day"),
        F.max("daily_hours").alias("max_hours_per_day"),
        F.mean("daily_unique_members").alias("avg_members_per_day"),
        F.max("daily_unique_members").alias("max_members_per_day"),
    )
)

# ─── % days exceeding hour thresholds ───
for threshold_hrs in [8, 12, 16, 20, 24]:
    threshold_units = threshold_hrs * 4  # 4 units per hour
    col_name = f"pct_days_over_{threshold_hrs}h"
    provider_daily_stats_with_thresh = (
        units_per_day
        .withColumn(f"over_{threshold_hrs}h", (F.col("daily_units") > threshold_units).cast("int"))
        .groupBy(SPROV_COL)
        .agg(
            (F.sum(f"over_{threshold_hrs}h") / F.count("srv_date") * 100).alias(col_name)
        )
    )
    provider_daily_stats = provider_daily_stats.join(
        provider_daily_stats_with_thresh, on=SPROV_COL, how="left"
    )

# Target vs peers
target_stats = provider_daily_stats.filter(F.col(SPROV_COL) == TARGET_PROVIDER).toPandas()
peer_stats = provider_daily_stats.filter(F.col(SPROV_COL) != TARGET_PROVIDER)

peer_summary = peer_stats.agg(
    F.expr("percentile_approx(avg_units_per_day, 0.5)").alias("peer_median_avg_units_day"),
    F.expr("percentile_approx(avg_hours_per_day, 0.5)").alias("peer_median_avg_hours_day"),
    F.expr("percentile_approx(max_hours_per_day, 0.5)").alias("peer_median_max_hours_day"),
    F.expr("percentile_approx(pct_days_over_8h, 0.5)").alias("peer_median_pct_over_8h"),
    F.expr("percentile_approx(pct_days_over_12h, 0.5)").alias("peer_median_pct_over_12h"),
    F.expr("percentile_approx(pct_days_over_16h, 0.5)").alias("peer_median_pct_over_16h"),
    F.expr("percentile_approx(pct_days_over_20h, 0.5)").alias("peer_median_pct_over_20h"),
    F.expr("percentile_approx(pct_days_over_24h, 0.5)").alias("peer_median_pct_over_24h"),
    F.expr("percentile_approx(avg_members_per_day, 0.5)").alias("peer_median_members_day"),
).toPandas()

print("\n" + "="*80)
print("1. TIMED UNITS PER SERVICE DAY")
print("="*80)
print(f"\nProvider {TARGET_PROVIDER}:")
print(target_stats.to_string(index=False))
print(f"\nPeer Medians (BH Professional cohort):")
print(peer_summary.to_string(index=False))


# =============================================================================
# 2. H0004 (BEHAVIORAL HEALTH COUNSELING) DEEP DIVE
# =============================================================================
H0004_CODE = "H0004"

df_h0004 = df_bhp.filter(F.col(PROC_COL) == H0004_CODE)

h0004_provider = (
    df_h0004
    .groupBy(SPROV_COL)
    .agg(
        F.sum("qty").alias("h0004_total_units"),
        F.countDistinct(MEMBER_COL).alias("h0004_unique_members"),
        F.countDistinct("srv_date").alias("h0004_active_days"),
        F.count("*").alias("h0004_line_count"),
    )
    .withColumn("h0004_hours", F.col("h0004_total_units") * 15 / 60)
    .withColumn("h0004_units_per_day", F.col("h0004_total_units") / F.col("h0004_active_days"))
    .withColumn("h0004_hours_per_member_day",
                F.col("h0004_hours") / (F.col("h0004_unique_members") * F.col("h0004_active_days")))
)

# H0004 per-day detail for threshold analysis
h0004_daily = (
    df_h0004
    .groupBy(SPROV_COL, "srv_date")
    .agg(F.sum("qty").alias("h0004_daily_units"))
    .withColumn("h0004_daily_hours", F.col("h0004_daily_units") * 15 / 60)
)

for threshold_hrs in [8, 12, 16]:
    threshold_units = threshold_hrs * 4
    col_name = f"h0004_pct_days_over_{threshold_hrs}h"
    thresh_agg = (
        h0004_daily
        .withColumn(f"over", (F.col("h0004_daily_units") > threshold_units).cast("int"))
        .groupBy(SPROV_COL)
        .agg((F.sum("over") / F.count("srv_date") * 100).alias(col_name))
    )
    h0004_provider = h0004_provider.join(thresh_agg, on=SPROV_COL, how="left")

target_h0004 = h0004_provider.filter(F.col(SPROV_COL) == TARGET_PROVIDER).toPandas()
peer_h0004 = h0004_provider.filter(F.col(SPROV_COL) != TARGET_PROVIDER)

peer_h0004_summary = peer_h0004.agg(
    F.expr("percentile_approx(h0004_total_units, 0.5)").alias("peer_med_h0004_units"),
    F.expr("percentile_approx(h0004_unique_members, 0.5)").alias("peer_med_h0004_members"),
    F.expr("percentile_approx(h0004_units_per_day, 0.5)").alias("peer_med_h0004_units_day"),
    F.expr("percentile_approx(h0004_hours_per_member_day, 0.5)").alias("peer_med_h0004_hrs_mbr_day"),
    F.expr("percentile_approx(h0004_pct_days_over_8h, 0.5)").alias("peer_med_h0004_pct_over_8h"),
).toPandas()

print("\n" + "="*80)
print("2. H0004 ANALYSIS")
print("="*80)
print(f"\nProvider {TARGET_PROVIDER}:")
print(target_h0004.to_string(index=False))
print(f"\nPeer Medians:")
print(peer_h0004_summary.to_string(index=False))


# =============================================================================
# 3. UNIQUE MEMBERS PER DAY
# =============================================================================
members_per_day = (
    df_bhp
    .groupBy(SPROV_COL, "srv_date")
    .agg(F.countDistinct(MEMBER_COL).alias("unique_members"))
)

members_summary = (
    members_per_day
    .groupBy(SPROV_COL)
    .agg(
        F.mean("unique_members").alias("avg_unique_members_per_day"),
        F.expr("percentile_approx(unique_members, 0.5)").alias("median_unique_members_per_day"),
        F.max("unique_members").alias("max_unique_members_per_day"),
    )
)

target_members = members_summary.filter(F.col(SPROV_COL) == TARGET_PROVIDER).toPandas()
peer_members_med = members_summary.filter(F.col(SPROV_COL) != TARGET_PROVIDER).agg(
    F.expr("percentile_approx(avg_unique_members_per_day, 0.5)").alias("peer_med"),
).toPandas()

print("\n" + "="*80)
print("3. UNIQUE MEMBERS PER DAY")
print("="*80)
print(f"\nProvider {TARGET_PROVIDER}:")
print(target_members.to_string(index=False))
print(f"\nPeer Median avg unique members/day: {peer_members_med['peer_med'].values[0]:.1f}")


# =============================================================================
# 4. BILLING ENTITY COMPARISON
#    Alliance bills / Noble services  vs  Noble bills / Noble services
# =============================================================================
# Identify billing provider(s) for this servicing provider
billing_breakdown = (
    df_bhp
    .filter(F.col(SPROV_COL) == TARGET_PROVIDER)
    .groupBy(BPROV_COL)
    .agg(
        F.count("*").alias("line_count"),
        F.sum(PMT_COL).alias("total_paid"),
        F.sum("qty").alias("total_units"),
        F.countDistinct(MEMBER_COL).alias("unique_members"),
        F.countDistinct("srv_date").alias("active_days"),
    )
    .orderBy(F.desc("line_count"))
)

print("\n" + "="*80)
print("4. BILLING ENTITY COMPARISON (Servicing Provider = 004526)")
print("="*80)
billing_breakdown.show(truncate=False)

# Per billing entity, compute daily stats
for bprov_row in billing_breakdown.collect():
    bp_id = bprov_row[BPROV_COL]
    bp_data = (
        df_bhp
        .filter((F.col(SPROV_COL) == TARGET_PROVIDER) & (F.col(BPROV_COL) == bp_id))
    )
    bp_daily = (
        bp_data
        .groupBy("srv_date")
        .agg(
            F.sum("qty").alias("daily_units"),
            F.countDistinct(MEMBER_COL).alias("daily_members"),
        )
        .withColumn("daily_hours", F.col("daily_units") * 15 / 60)
    )
    bp_summary = bp_daily.agg(
        F.mean("daily_units").alias("avg_units_day"),
        F.mean("daily_hours").alias("avg_hours_day"),
        F.max("daily_hours").alias("max_hours_day"),
        F.mean("daily_members").alias("avg_members_day"),
    ).toPandas()
    print(f"\n  Billing Provider {bp_id}:")
    print(f"    Avg units/day: {bp_summary['avg_units_day'].values[0]:.1f}")
    print(f"    Avg hours/day: {bp_summary['avg_hours_day'].values[0]:.1f}")
    print(f"    Max hours/day: {bp_summary['max_hours_day'].values[0]:.1f}")
    print(f"    Avg members/day: {bp_summary['avg_members_day'].values[0]:.1f}")


# =============================================================================
# 5. IDENTICAL CODE/UNIT PATTERN DETECTION
#    How often does this provider bill the exact same (PROC_CD, modifiers, qty)
#    combo across members on the same day?
# =============================================================================
# Build a "billing signature" per line
mod_cols_available = [c for c in MODIFIER_COLS if c in df_bhp.columns]
if mod_cols_available:
    sig_expr = F.concat_ws("|", F.col(PROC_COL), *[F.coalesce(F.col(c), F.lit("")) for c in mod_cols_available], F.col(QTY_COL).cast("string"))
else:
    sig_expr = F.concat_ws("|", F.col(PROC_COL), F.col(QTY_COL).cast("string"))

df_signatures = df_bhp.withColumn("billing_signature", sig_expr)

# Per provider-day: count how many members got the exact same signature
dup_patterns = (
    df_signatures
    .groupBy(SPROV_COL, "srv_date", "billing_signature")
    .agg(
        F.countDistinct(MEMBER_COL).alias("members_with_pattern"),
        F.count("*").alias("lines_with_pattern"),
    )
)

# Flag patterns where ≥2 members got identical billing on the same day
dup_flagged = dup_patterns.filter(F.col("members_with_pattern") >= 2)

# Per provider: what % of their service days have ≥1 duplicated pattern?
total_days_per_provider = (
    df_signatures
    .groupBy(SPROV_COL)
    .agg(F.countDistinct("srv_date").alias("total_active_days"))
)

days_with_dups = (
    dup_flagged
    .groupBy(SPROV_COL)
    .agg(F.countDistinct("srv_date").alias("days_with_dup_pattern"))
)

dup_summary = (
    total_days_per_provider
    .join(days_with_dups, on=SPROV_COL, how="left")
    .fillna(0, subset=["days_with_dup_pattern"])
    .withColumn("pct_days_with_dup_pattern",
                F.col("days_with_dup_pattern") / F.col("total_active_days") * 100)
)

# Also compute: avg # of members sharing a pattern when duplication occurs
avg_dup_size = (
    dup_flagged
    .groupBy(SPROV_COL)
    .agg(
        F.mean("members_with_pattern").alias("avg_members_per_dup_pattern"),
        F.max("members_with_pattern").alias("max_members_per_dup_pattern"),
        F.count("*").alias("total_dup_pattern_instances"),
    )
)

dup_summary = dup_summary.join(avg_dup_size, on=SPROV_COL, how="left")

target_dup = dup_summary.filter(F.col(SPROV_COL) == TARGET_PROVIDER).toPandas()
peer_dup = dup_summary.filter(F.col(SPROV_COL) != TARGET_PROVIDER).agg(
    F.expr("percentile_approx(pct_days_with_dup_pattern, 0.5)").alias("peer_med_pct_dup_days"),
    F.expr("percentile_approx(avg_members_per_dup_pattern, 0.5)").alias("peer_med_avg_dup_size"),
).toPandas()

print("\n" + "="*80)
print("5. IDENTICAL BILLING PATTERN DETECTION")
print("="*80)
print(f"\nProvider {TARGET_PROVIDER}:")
print(target_dup.to_string(index=False))
print(f"\nPeer Medians:")
print(peer_dup.to_string(index=False))


# =============================================================================
# 6. REPEATED PACKAGE FREQUENCY
#    How often does the same member get the exact same set of codes on
#    different dates?
# =============================================================================
# Per (provider, member, date): collect sorted set of proc codes
daily_packages = (
    df_signatures
    .groupBy(SPROV_COL, MEMBER_COL, "srv_date")
    .agg(
        F.sort_array(F.collect_list("billing_signature")).alias("daily_package"),
    )
    .withColumn("package_str", F.concat_ws("||", "daily_package"))
)

# Per (provider, member): count how many distinct dates the same package repeats
package_repeats = (
    daily_packages
    .groupBy(SPROV_COL, MEMBER_COL, "package_str")
    .agg(F.count("srv_date").alias("times_repeated"))
    .filter(F.col("times_repeated") > 1)  # same package on >1 date
)

# Per provider: what % of members have repeated packages?
total_members = (
    df_bhp
    .groupBy(SPROV_COL)
    .agg(F.countDistinct(MEMBER_COL).alias("total_members"))
)

members_with_repeats = (
    package_repeats
    .groupBy(SPROV_COL)
    .agg(
        F.countDistinct(MEMBER_COL).alias("members_with_repeated_package"),
        F.mean("times_repeated").alias("avg_repeat_count"),
        F.max("times_repeated").alias("max_repeat_count"),
    )
)

repeat_summary = (
    total_members
    .join(members_with_repeats, on=SPROV_COL, how="left")
    .fillna(0, subset=["members_with_repeated_package"])
    .withColumn("pct_members_with_repeats",
                F.col("members_with_repeated_package") / F.col("total_members") * 100)
)

target_repeat = repeat_summary.filter(F.col(SPROV_COL) == TARGET_PROVIDER).toPandas()
peer_repeat = repeat_summary.filter(F.col(SPROV_COL) != TARGET_PROVIDER).agg(
    F.expr("percentile_approx(pct_members_with_repeats, 0.5)").alias("peer_med_pct_repeats"),
    F.expr("percentile_approx(avg_repeat_count, 0.5)").alias("peer_med_avg_repeat_count"),
).toPandas()

print("\n" + "="*80)
print("6. REPEATED PACKAGE FREQUENCY")
print("="*80)
print(f"\nProvider {TARGET_PROVIDER}:")
print(target_repeat.to_string(index=False))
print(f"\nPeer Medians:")
print(peer_repeat.to_string(index=False))


# =============================================================================
# 7. PERCENTILE RANK OF PROVIDER WITHIN COHORT
# =============================================================================
# Rank on key metrics
rank_window = Window.orderBy(F.col("avg_units_per_day"))
provider_ranks = (
    provider_daily_stats
    .withColumn("rank_units_day", F.percent_rank().over(rank_window) * 100)
)

rank_window_h = Window.orderBy(F.col("avg_hours_per_day"))
provider_ranks = (
    provider_ranks
    .withColumn("rank_hours_day", F.percent_rank().over(rank_window_h) * 100)
)

target_ranks = provider_ranks.filter(F.col(SPROV_COL) == TARGET_PROVIDER).select(
    SPROV_COL, "avg_units_per_day", "rank_units_day", "avg_hours_per_day", "rank_hours_day"
).toPandas()

print("\n" + "="*80)
print("7. PERCENTILE RANK IN BH PROFESSIONAL COHORT")
print("="*80)
print(target_ranks.to_string(index=False))


# =============================================================================
# 8. CONSOLIDATED COMPARISON TABLE
# =============================================================================
print("\n" + "="*80)
print("CONSOLIDATED: Provider 004526 vs BH Professional Peers")
print("="*80)

def safe_val(df, col, default="N/A"):
    try:
        return df[col].values[0]
    except:
        return default

metrics = [
    ("Avg timed units/day",
     safe_val(target_stats, "avg_units_per_day"),
     safe_val(peer_summary, "peer_median_avg_units_day")),
    ("Avg hours/day",
     safe_val(target_stats, "avg_hours_per_day"),
     safe_val(peer_summary, "peer_median_avg_hours_day")),
    ("Max hours in a single day",
     safe_val(target_stats, "max_hours_per_day"),
     safe_val(peer_summary, "peer_median_max_hours_day")),
    ("% days > 8 hours",
     safe_val(target_stats, "pct_days_over_8h"),
     safe_val(peer_summary, "peer_median_pct_over_8h")),
    ("% days > 12 hours",
     safe_val(target_stats, "pct_days_over_12h"),
     safe_val(peer_summary, "peer_median_pct_over_12h")),
    ("% days > 16 hours",
     safe_val(target_stats, "pct_days_over_16h"),
     safe_val(peer_summary, "peer_median_pct_over_16h")),
    ("% days > 20 hours",
     safe_val(target_stats, "pct_days_over_20h"),
     safe_val(peer_summary, "peer_median_pct_over_20h")),
    ("% days > 24 hours",
     safe_val(target_stats, "pct_days_over_24h"),
     safe_val(peer_summary, "peer_median_pct_over_24h")),
    ("Avg unique members/day",
     safe_val(target_stats, "avg_members_per_day"),
     safe_val(peer_summary, "peer_median_members_day")),
    ("H0004 total units",
     safe_val(target_h0004, "h0004_total_units"),
     safe_val(peer_h0004_summary, "peer_med_h0004_units")),
    ("H0004 unique members",
     safe_val(target_h0004, "h0004_unique_members"),
     safe_val(peer_h0004_summary, "peer_med_h0004_members")),
    ("H0004 units/day",
     safe_val(target_h0004, "h0004_units_per_day"),
     safe_val(peer_h0004_summary, "peer_med_h0004_units_day")),
    ("% days with identical patterns (≥2 members)",
     safe_val(target_dup, "pct_days_with_dup_pattern"),
     safe_val(peer_dup, "peer_med_pct_dup_days")),
    ("% members with repeated packages",
     safe_val(target_repeat, "pct_members_with_repeats"),
     safe_val(peer_repeat, "peer_med_pct_repeats")),
]

comparison_df = pd.DataFrame(metrics, columns=["Metric", "Provider 004526", "Peer Median"])
comparison_df["Ratio"] = comparison_df.apply(
    lambda r: f"{r['Provider 004526'] / r['Peer Median']:.1f}x"
    if isinstance(r['Provider 004526'], (int, float)) and isinstance(r['Peer Median'], (int, float)) and r['Peer Median'] > 0
    else "N/A", axis=1
)

print(comparison_df.to_string(index=False))

# ─── Save to CSV for the investigator ───
output_path = "/Workspace/Users/{ROOT}/Projects/investigator_peer_stats_004526.csv"
# comparison_df.to_csv(output_path, index=False)
# print(f"\nSaved to {output_path}")

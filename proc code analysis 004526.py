# =============================================================================
# All Procedure Codes Analysis — Provider 004526 vs BH Professional Peers
# Saves to Excel with summary + per-code sheets
# =============================================================================

import pyspark.sql.functions as F
from pyspark.sql.window import Window
import pandas as pd

TARGET_PROVIDER = "004526"

# ─── Get all proc codes billed by this provider ─────────────────────────────
target_proc_codes = (
    df_bhp
    .filter(F.col(SPROV_COL) == TARGET_PROVIDER)
    .select(PROC_COL)
    .distinct()
    .rdd.flatMap(lambda x: x)
    .collect()
)
print(f"Provider {TARGET_PROVIDER} bills {len(target_proc_codes)} distinct procedure codes: {sorted(target_proc_codes)}")

# ─── Per-proc-code analysis function ─────────────────────────────────────────
def analyze_proc_code(proc_code, df_bhp, target_provider, sprov_col, member_col):
    """Compute target vs peer stats for a single procedure code."""

    df_proc = df_bhp.filter(F.col(PROC_COL) == proc_code)

    # Per-provider aggregation
    provider_agg = (
        df_proc
        .groupBy(sprov_col)
        .agg(
            F.sum("qty").alias("total_units"),
            F.countDistinct(member_col).alias("unique_members"),
            F.countDistinct("srv_date").alias("active_days"),
            F.count("*").alias("line_count"),
            F.sum(PMT_COL).alias("total_paid"),
        )
        .withColumn("units_per_day", F.col("total_units") / F.col("active_days"))
        .withColumn("hours", F.col("total_units") * 15 / 60)
        .withColumn("hours_per_day", F.col("hours") / F.col("active_days"))
        .withColumn("members_per_day", F.col("unique_members") / F.col("active_days"))
        .withColumn("units_per_member", F.col("total_units") / F.col("unique_members"))
        .withColumn("paid_per_unit", F.col("total_paid") / F.col("total_units"))
    )

    # Per-provider daily detail for threshold analysis
    daily = (
        df_proc
        .groupBy(sprov_col, "srv_date")
        .agg(
            F.sum("qty").alias("daily_units"),
            F.countDistinct(member_col).alias("daily_members"),
        )
        .withColumn("daily_hours", F.col("daily_units") * 15 / 60)
    )

    # Threshold exceedance
    thresh_cols = {}
    for hrs in [4, 8, 12, 16]:
        col_name = f"pct_days_over_{hrs}h"
        thresh_agg = (
            daily
            .withColumn("over", (F.col("daily_units") > hrs * 4).cast("int"))
            .groupBy(sprov_col)
            .agg((F.sum("over") / F.count("srv_date") * 100).alias(col_name))
        )
        provider_agg = provider_agg.join(thresh_agg, on=sprov_col, how="left")

    # Target row
    target_row = provider_agg.filter(F.col(sprov_col) == target_provider).toPandas()

    if target_row.empty:
        return None

    # Peer stats
    peer = provider_agg.filter(F.col(sprov_col) != target_provider)
    peer_count = peer.count()

    if peer_count == 0:
        return None

    peer_medians = peer.agg(
        F.expr("percentile_approx(total_units, 0.5)").alias("peer_med_total_units"),
        F.expr("percentile_approx(unique_members, 0.5)").alias("peer_med_unique_members"),
        F.expr("percentile_approx(active_days, 0.5)").alias("peer_med_active_days"),
        F.expr("percentile_approx(units_per_day, 0.5)").alias("peer_med_units_per_day"),
        F.expr("percentile_approx(hours_per_day, 0.5)").alias("peer_med_hours_per_day"),
        F.expr("percentile_approx(units_per_member, 0.5)").alias("peer_med_units_per_member"),
        F.expr("percentile_approx(total_paid, 0.5)").alias("peer_med_total_paid"),
        F.expr("percentile_approx(paid_per_unit, 0.5)").alias("peer_med_paid_per_unit"),
        F.expr("percentile_approx(pct_days_over_4h, 0.5)").alias("peer_med_pct_over_4h"),
        F.expr("percentile_approx(pct_days_over_8h, 0.5)").alias("peer_med_pct_over_8h"),
        F.expr("percentile_approx(pct_days_over_12h, 0.5)").alias("peer_med_pct_over_12h"),
        F.expr("percentile_approx(pct_days_over_16h, 0.5)").alias("peer_med_pct_over_16h"),
    ).toPandas()

    # Percentile rank of target within cohort
    rank_df = provider_agg.withColumn(
        "pct_rank_units_day",
        F.percent_rank().over(Window.orderBy("units_per_day")) * 100
    )
    target_rank = rank_df.filter(F.col(sprov_col) == target_provider).select("pct_rank_units_day").toPandas()

    # Build comparison row
    def ratio(t, p):
        try:
            t_val = float(t)
            p_val = float(p)
            if p_val > 0:
                return round(t_val / p_val, 1)
        except:
            pass
        return None

    t = target_row.iloc[0]
    p = peer_medians.iloc[0]

    result = {
        'proc_code': proc_code,
        'peer_providers_billing_code': peer_count,
        'target_total_units': t.get('total_units'),
        'peer_med_total_units': p.get('peer_med_total_units'),
        'ratio_total_units': ratio(t.get('total_units'), p.get('peer_med_total_units')),
        'target_unique_members': t.get('unique_members'),
        'peer_med_unique_members': p.get('peer_med_unique_members'),
        'target_active_days': t.get('active_days'),
        'peer_med_active_days': p.get('peer_med_active_days'),
        'target_units_per_day': round(t.get('units_per_day', 0), 2),
        'peer_med_units_per_day': round(p.get('peer_med_units_per_day', 0), 2),
        'ratio_units_per_day': ratio(t.get('units_per_day'), p.get('peer_med_units_per_day')),
        'target_hours_per_day': round(t.get('hours_per_day', 0), 2),
        'peer_med_hours_per_day': round(p.get('peer_med_hours_per_day', 0), 2),
        'target_units_per_member': round(t.get('units_per_member', 0), 2),
        'peer_med_units_per_member': round(p.get('peer_med_units_per_member', 0), 2),
        'ratio_units_per_member': ratio(t.get('units_per_member'), p.get('peer_med_units_per_member')),
        'target_total_paid': round(t.get('total_paid', 0), 2),
        'peer_med_total_paid': round(p.get('peer_med_total_paid', 0), 2),
        'ratio_total_paid': ratio(t.get('total_paid'), p.get('peer_med_total_paid')),
        'target_paid_per_unit': round(t.get('paid_per_unit', 0), 2),
        'peer_med_paid_per_unit': round(p.get('peer_med_paid_per_unit', 0), 2),
        'target_pct_days_over_4h': round(t.get('pct_days_over_4h', 0), 1),
        'peer_med_pct_over_4h': round(p.get('peer_med_pct_over_4h', 0), 1),
        'target_pct_days_over_8h': round(t.get('pct_days_over_8h', 0), 1),
        'peer_med_pct_over_8h': round(p.get('peer_med_pct_over_8h', 0), 1),
        'target_pct_days_over_12h': round(t.get('pct_days_over_12h', 0), 1),
        'target_pct_days_over_16h': round(t.get('pct_days_over_16h', 0), 1),
        'percentile_rank_units_day': round(target_rank['pct_rank_units_day'].values[0], 1) if not target_rank.empty else None,
        'target_line_count': t.get('line_count'),
    }

    return result


# ─── Run for all proc codes ──────────────────────────────────────────────────
results = []
for i, pc in enumerate(sorted(target_proc_codes)):
    print(f"  [{i+1}/{len(target_proc_codes)}] Analyzing {pc}...")
    row = analyze_proc_code(pc, df_bhp, TARGET_PROVIDER, SPROV_COL, MEMBER_COL)
    if row:
        results.append(row)
    else:
        print(f"    Skipped {pc} (no peer data or not found)")

summary_df = pd.DataFrame(results)

# Sort by ratio_total_units descending to surface the most anomalous codes first
summary_df = summary_df.sort_values("ratio_total_units", ascending=False).reset_index(drop=True)

print(f"\nAnalyzed {len(results)} procedure codes with peer comparisons")
display(summary_df)


# ─── Save to Excel ───────────────────────────────────────────────────────────
output_path = f"/Workspace/Users/{{ROOT}}/Projects/all_projects_predictions/LLM/provider_004526_proc_code_analysis.xlsx"

with pd.ExcelWriter(output_path, engine="openpyxl") as writer:

    # Sheet 1: Summary — all proc codes ranked by anomaly
    summary_df.to_excel(writer, sheet_name="Summary_All_Codes", index=False)

    # Sheet 2: Top anomalous codes (ratio > 2x on any key metric)
    top_anomalous = summary_df[
        (summary_df['ratio_total_units'].fillna(0) > 2) |
        (summary_df['ratio_units_per_day'].fillna(0) > 2) |
        (summary_df['ratio_total_paid'].fillna(0) > 2)
    ].copy()
    top_anomalous.to_excel(writer, sheet_name="Anomalous_Codes_GT_2x", index=False)

    # Sheet 3: Billing entity breakdown per proc code
    billing_by_proc = (
        df_bhp
        .filter(F.col(SPROV_COL) == TARGET_PROVIDER)
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
    billing_by_proc.to_excel(writer, sheet_name="Billing_Entity_By_Code", index=False)

print(f"\nSaved to {output_path}")

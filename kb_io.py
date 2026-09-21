"""
kb_io.py  -  pandas -> Delta without schema-inference surprises.

spark.createDataFrame(pdf) infers types from values. A column that is all None
(common: end_date, reviewer, error, sql for leads on a small run) cannot be
inferred and the write fails with "Can not merge type" / "can not infer schema".
This helper builds an explicit schema from pandas dtypes instead: bool -> boolean,
int -> long, float -> double, everything else -> string (None preserved).
"""
from __future__ import annotations

import json

import pandas as pd


def _spark_types():
    from pyspark.sql import types as T
    return T


def to_spark(spark, pdf: pd.DataFrame):
    T = _spark_types()
    pdf = pdf.copy()
    fields = []
    for c in pdf.columns:
        s = pdf[c]
        if pd.api.types.is_bool_dtype(s):
            t = T.BooleanType()
            pdf[c] = s.astype(bool)
        elif pd.api.types.is_integer_dtype(s):
            t = T.LongType()
        elif pd.api.types.is_float_dtype(s):
            t = T.DoubleType()
            pdf[c] = s.astype(float)
        else:
            t = T.StringType()
            pdf[c] = s.map(lambda v: None if v is None or (isinstance(v, float) and pd.isna(v))
                           else (json.dumps(v, default=str) if isinstance(v, (list, dict)) else str(v)))
        fields.append(T.StructField(str(c), t, True))
    pdf = pdf.astype(object).where(pdf.notna(), None)
    return spark.createDataFrame(pdf.values.tolist(), schema=T.StructType(fields))


_NOT_FOUND_MARKERS = ("TABLE_OR_VIEW_NOT_FOUND", "SCHEMA_NOT_FOUND", "does not exist", "cannot be found",
                      "not found", "NoSuchTableException", "AnalysisException: Table")


class TableAccessError(RuntimeError):
    """A table exists (or may exist) but could not be read. NEVER treat this as 'no rows'."""


def table_or_none(spark, fqn: str) -> pd.DataFrame | None:
    """Read a table to pandas. Returns None ONLY when the table does not exist.

    Any other failure - CLOUD_ACCESS_DENIED / 403 on the metastore storage, a cluster
    without Unity Catalog access, a transient error - is raised as TableAccessError.
    On 2026-09-21 a bare `except: return None` turned a 403 on state_rules into an
    empty frame, and the merge step then tried to OVERWRITE the reviewed rules table
    with nothing; only the skip-on-empty guard in write_table stopped it.
    """
    try:
        return spark.table(fqn).toPandas()
    except Exception as e:
        msg = f"{type(e).__name__}: {e}"
        if any(m.lower() in msg.lower() for m in _NOT_FOUND_MARKERS) and "ACCESS_DENIED" not in msg.upper():
            return None
        raise TableAccessError(
            f"could not read {fqn} - this is NOT an empty table. If the message mentions CLOUD_ACCESS_DENIED / "
            f"403 / AuthorizationFailure, this cluster's identity cannot reach the Unity Catalog storage; use a "
            f"UC-enabled cluster (the one the summarization pipeline's CPU notebook uses). Original: {msg[:600]}"
        ) from e


def write_table(spark, rows_or_df, fqn: str, mode: str = "overwrite") -> int:
    pdf = rows_or_df if isinstance(rows_or_df, pd.DataFrame) else pd.DataFrame(rows_or_df)
    if pdf.empty:
        print(f"  (skipped {fqn}: no rows)")
        return 0
    w = to_spark(spark, pdf).write.format("delta").mode(mode)
    w = w.option("overwriteSchema", "true") if mode == "overwrite" else w.option("mergeSchema", "true")
    w.saveAsTable(fqn)
    print(f"  wrote {fqn}  ({len(pdf):,} rows, {mode})")
    return len(pdf)

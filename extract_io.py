"""
extract_io.py  -  files that let the GPU notebook and the CPU notebook hand off.

Serverless GPU compute runs the model but cannot be trusted with Delta writes (SB's
llm_summarization pipeline hit this: the GPU notebook writes files, a CPU notebook loads
them into tables). PIE follows the same split:

  stage_inputs        (CPU, UC)   PDFs + table snapshots  ->  STAGE_DIR   (Workspace files)
  extract_chapters_gpu(GPU)       STAGE_DIR -> model -> RUN_DIR/<run_id>/  (JSONL + cache.json)
  persist_extraction  (CPU, UC)   RUN_DIR/<run_id>/  ->  Delta tables + review workbook

Everything here is plain JSON/JSONL on ordinary files. No Spark. Rows with pandas NaN
are written as null; lists/dicts are kept as JSON. Workspace files (/Workspace/Users/...)
are readable and writable from Serverless GPU, which is where hf_token.txt already lives.
"""
from __future__ import annotations

import json
import os
from datetime import datetime

import pandas as pd

RUN_FILES = {
    "inventory": "inventory.jsonl",
    "sections":  "sections.jsonl",
    "log":       "log.jsonl",
    "dropped":   "dropped.jsonl",
    "changes":   "changes.jsonl",
    "drafts":    "drafts.jsonl",
}
CACHE_FILE = "cache.json"
MANIFEST_FILE = "manifest.json"
SNAPSHOT_FILES = {"cache": "cache.json", "state_rules": "state_rules.jsonl", "inventory": "inventory_prev.jsonl"}


# ------------------------------------------------------------------ primitives
def _clean(v):
    if isinstance(v, float) and pd.isna(v):
        return None
    if hasattr(v, "isoformat"):
        return v.isoformat()
    return v


def write_jsonl(path: str, rows) -> int:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if isinstance(rows, pd.DataFrame):
        rows = rows.astype(object).where(rows.notna(), None).to_dict("records")
    tmp = path + ".tmp"
    n = 0
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps({k: _clean(v) for k, v in r.items()}, default=str) + "\n")
            n += 1
    os.replace(tmp, path)                      # atomic: a crash never leaves a half file
    return n


def read_jsonl(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_json(path: str, obj) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, default=str)
    os.replace(tmp, path)


def read_json(path: str, default=None):
    if not os.path.exists(path):
        return default
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _frame_or_none(rows: list[dict]) -> pd.DataFrame | None:
    return pd.DataFrame(rows) if rows else None


# --------------------------------------------------------------- snapshots (CPU)
def export_snapshots(spark, stage_dir: str, cache_table: str, state_rules_table: str,
                     inventory_table: str, log_fn=print) -> dict:
    """Table state the GPU notebook needs, as files. Uses kb_io.table_or_none, so a
    cluster that cannot READ a table stops here instead of exporting an empty snapshot."""
    from kb_io import table_or_none
    snap = os.path.join(stage_dir, "snapshots")
    os.makedirs(snap, exist_ok=True)
    out = {}
    cache_df = table_or_none(spark, cache_table)
    cache = {} if cache_df is None else {r["cache_key"]: json.loads(r["proposals"]) for r in cache_df.to_dict("records")}
    write_json(os.path.join(snap, SNAPSHOT_FILES["cache"]), cache)
    out["cache_entries"] = len(cache)
    for key, table in (("state_rules", state_rules_table), ("inventory", inventory_table)):
        df = table_or_none(spark, table)
        n = write_jsonl(os.path.join(snap, SNAPSHOT_FILES[key]), df if df is not None else [])
        out[f"{key}_rows"] = n
        out[f"{key}_table_exists"] = df is not None
    write_json(os.path.join(snap, "exported_at.json"), {"at": datetime.now().isoformat(timespec="seconds"), **out})
    log_fn(f"snapshots -> {snap}: {out}")
    return out


def load_snapshots(stage_dir: str) -> tuple[dict, pd.DataFrame | None, pd.DataFrame | None, dict]:
    """(cache, existing_rules, prev_inventory, meta) from the files stage_inputs wrote."""
    snap = os.path.join(stage_dir, "snapshots")
    meta = read_json(os.path.join(snap, "exported_at.json"), {})
    cache = read_json(os.path.join(snap, SNAPSHOT_FILES["cache"]), {})
    rules = _frame_or_none(read_jsonl(os.path.join(snap, SNAPSHOT_FILES["state_rules"])))
    inv = _frame_or_none(read_jsonl(os.path.join(snap, SNAPSHOT_FILES["inventory"])))
    return cache, rules, inv, meta


# ------------------------------------------------------------------ run dir (GPU)
def new_run_dir(run_root: str, run_id: str) -> str:
    d = os.path.join(run_root, run_id)
    os.makedirs(d, exist_ok=True)
    return d


def write_cache(run_dir: str, cache: dict) -> None:
    write_json(os.path.join(run_dir, CACHE_FILE), cache)


def write_run(run_dir: str, *, inventory_rows, section_rows, log_rows, dropped_rows, change_rows,
              drafts, cache: dict, manifest: dict) -> dict:
    """Write (or rewrite) every output of a run. Called after every chapter, so a crash
    leaves the completed chapters on disk; `manifest['finished']` says whether the run
    reached its end."""
    counts = {}
    for key, rows in (("inventory", inventory_rows), ("sections", section_rows), ("log", log_rows),
                      ("dropped", dropped_rows), ("changes", change_rows), ("drafts", drafts)):
        counts[key] = write_jsonl(os.path.join(run_dir, RUN_FILES[key]), rows)
    write_cache(run_dir, cache)
    counts["cache_entries"] = len(cache)
    write_json(os.path.join(run_dir, MANIFEST_FILE), {**manifest, "counts": counts,
                                                       "written_at": datetime.now().isoformat(timespec="seconds")})
    return counts


def read_run(run_dir: str) -> dict:
    out = {key: read_jsonl(os.path.join(run_dir, fn)) for key, fn in RUN_FILES.items()}
    out["cache"] = read_json(os.path.join(run_dir, CACHE_FILE), {})
    out["manifest"] = read_json(os.path.join(run_dir, MANIFEST_FILE), {})
    return out


def latest_run_dir(run_root: str) -> str | None:
    if not os.path.isdir(run_root):
        return None
    runs = sorted(d for d in os.listdir(run_root) if os.path.isdir(os.path.join(run_root, d)))
    return os.path.join(run_root, runs[-1]) if runs else None


# ------------------------------------------------------------- persist guard (CPU)
def guard_state_rules_merge(existing: pd.DataFrame | None, merged: pd.DataFrame, table: str) -> None:
    """Refuse to overwrite a populated state_rules table with fewer reviewed rows than it
    has. merge_drafts never drops reviewed rows, so a shrink means the 'existing' frame
    was wrong (a failed read, a stale snapshot) - exactly the 2026-09-21 near-miss."""
    if existing is None or len(existing) == 0:
        return
    from state_rules import REVIEWED
    before = int(existing["review_status"].isin(REVIEWED).sum()) if "review_status" in existing else 0
    after = int(merged["review_status"].isin(REVIEWED).sum()) if "review_status" in merged else 0
    if len(merged) == 0 or after < before:
        raise RuntimeError(
            f"refusing to overwrite {table}: merged result has {len(merged)} rows / {after} reviewed, "
            f"existing has {len(existing)} rows / {before} reviewed. The existing frame is probably not the "
            f"real table (failed read or stale snapshot). Nothing was written.")

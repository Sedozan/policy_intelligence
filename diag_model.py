# Databricks notebook source
# MAGIC %md
# MAGIC # Model backend diagnostic (DBFM serving-endpoint route only)
# MAGIC
# MAGIC **The production path is `HFLOCAL`** (gpt-oss-120b loaded on the cluster, as in `llm_summarization_notebook`);
# MAGIC its health check is the first cell of `extract_chapters` and prints the model's own reply. This notebook is
# MAGIC only for the optional `DBFM` endpoint route.
# MAGIC
# MAGIC Run this when `extract_chapters` reports a dead model. It tries each client route on its own and
# MAGIC prints the REAL error (the 2026-09-20 run showed only `KeyError`, which was the fallback's
# MAGIC `os.environ["DATABRICKS_HOST"]`, not the original failure). Then it lists the serving endpoints
# MAGIC this workspace can see, so a wrong `config.DBFM_ENDPOINT` name is obvious. Read-only.

# COMMAND ----------

import os, sys, time, traceback
# ---- find the package: this notebook's folder or its parent must contain config.py.
# No hard-coded workspace path; set PKB_PACKAGE_DIR if you keep the notebooks elsewhere.
_here = os.getcwd()
for _cand in (os.environ.get("PKB_PACKAGE_DIR"), _here, os.path.dirname(_here)):
    if _cand and os.path.exists(os.path.join(_cand, "config.py")):
        sys.path.insert(0, _cand); break
else:
    raise ImportError(f"config.py not found in {_here} or its parent; run this notebook from inside the pie_mvp folder "
                      f"or set PKB_PACKAGE_DIR")
import importlib, config, extract_rules
for m in (config, extract_rules):
    importlib.reload(m)

ENDPOINT = config.DBFM_ENDPOINT
print(f"config.DBFM_ENDPOINT = {ENDPOINT!r}   config.LLM_MAX_TOKENS = {config.LLM_MAX_TOKENS}")
print(f"env DATABRICKS_HOST set: {bool(os.environ.get('DATABRICKS_HOST'))}   DATABRICKS_TOKEN set: {bool(os.environ.get('DATABRICKS_TOKEN'))}")
for pkg in ("databricks.sdk", "mlflow", "openai"):
    try:
        mod = importlib.import_module(pkg)
        print(f"  {pkg:15s} {getattr(mod, '__version__', '?')}")
    except Exception as e:
        print(f"  {pkg:15s} NOT IMPORTABLE: {type(e).__name__}: {e}")

# COMMAND ----------

# MAGIC %md ## Route 1 — databricks-sdk OpenAI client (needs `openai` installed)

# COMMAND ----------

try:
    from databricks.sdk import WorkspaceClient
    w = WorkspaceClient()
    print("WorkspaceClient() OK; host =", w.config.host)
    try:
        c = w.serving_endpoints.get_open_ai_client()
        r = c.chat.completions.create(model=ENDPOINT, max_tokens=20, temperature=0,
                                      messages=[{"role": "user", "content": 'Return exactly {"ok": true}'}])
        print("ROUTE 1 OK  ->", repr(r.choices[0].message.content))
    except Exception:
        print("ROUTE 1 FAILED at get_open_ai_client()/create():")
        traceback.print_exc(limit=3)
except Exception:
    print("ROUTE 1 FAILED at WorkspaceClient():")
    traceback.print_exc(limit=3)

# COMMAND ----------

# MAGIC %md ## Route 2 — mlflow.deployments (always available on a Databricks cluster)

# COMMAND ----------

try:
    from mlflow.deployments import get_deploy_client
    c = get_deploy_client("databricks")
    r = c.predict(endpoint=ENDPOINT, inputs={"messages": [{"role": "user", "content": 'Return exactly {"ok": true}'}],
                                             "max_tokens": 20, "temperature": 0})
    print("ROUTE 2 OK  ->", repr(r["choices"][0]["message"]["content"]))
except Exception:
    print("ROUTE 2 FAILED:")
    traceback.print_exc(limit=3)

# COMMAND ----------

# MAGIC %md ## Route 3 — openai client against `<workspace>/serving-endpoints` with the notebook's own token

# COMMAND ----------

try:
    host, token = extract_rules._workspace_host_token()
    print("host =", host, "| token present:", bool(token))
    from openai import OpenAI
    c = OpenAI(api_key=token, base_url=f"{host}/serving-endpoints")
    r = c.chat.completions.create(model=ENDPOINT, max_tokens=20, temperature=0,
                                  messages=[{"role": "user", "content": 'Return exactly {"ok": true}'}])
    print("ROUTE 3 OK  ->", repr(r.choices[0].message.content))
except Exception:
    print("ROUTE 3 FAILED:")
    traceback.print_exc(limit=3)

# COMMAND ----------

# MAGIC %md ## Endpoints this workspace can see (is `config.DBFM_ENDPOINT` spelled right?)

# COMMAND ----------

try:
    from databricks.sdk import WorkspaceClient
    names = sorted(e.name for e in WorkspaceClient().serving_endpoints.list())
    print(f"{len(names)} serving endpoints visible")
    hits = [n for n in names if "gpt" in n.lower() or "llama" in n.lower() or "claude" in n.lower() or "dbrx" in n.lower()]
    print("LLM-looking endpoints:", hits)
    print("configured endpoint present:", ENDPOINT in names)
except Exception:
    print("could not list endpoints:")
    traceback.print_exc(limit=2)

# COMMAND ----------

# MAGIC %md ## What the package will do with this

# COMMAND ----------

b = extract_rules.DBFMBackend()
t0 = time.time()
ok, detail = b.healthcheck()
print(f"{'OK ' if ok else 'FAIL'} ({time.time() - t0:.1f}s) {detail}")
if ok:
    print("extract_chapters will run with the model. Expect minutes, not seconds: one model call per candidate section.")
else:
    print("extract_chapters will STOP at its health check until this is fixed (or ALLOW_PATTERNS_ONLY=True for a regex-only smoke run).")

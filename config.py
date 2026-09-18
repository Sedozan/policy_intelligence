"""
config.py  -  environment and claims-schema mapping for the Policy KB.

Everything that names a table or a column lives here. Nothing else in the
package hard-codes a physical column name.
"""
import os

MODE = os.environ.get("PKB_MODE", "DATABRICKS")

# ---------------------------------------------------------------- sources
CATALOG = "main"
SCHEMA = "sedo"                                   # NCCI tables + state_rules live here
STATE_RULES_TABLE = f"{CATALOG}.{SCHEMA}.state_rules"

# ---------------------------------------------------------------- outputs
OUTPUT_CATALOG = "main"
OUTPUT_SCHEMA = "policy_kb"

# ---------------------------------------------------------------- claims
CLAIMS_TABLE = os.environ.get("PKB_CLAIMS", "main.prod_input.all_data_C_A")

# logical name -> physical column (verified against DESCRIBE all_data_C_A)
CLAIM_COLS = {
    "claim_id":            "ClaimID",
    "line_no":             "LN_NO",
    "member":              "MEMBER_KEY",
    "servicing_provider":  "SProv_ID",
    "billing_provider":    "BProv_ID",
    "dos":                 "Svc_Begin_Dt",
    "code":                "PROC_CD",
    "units":               "QUANTITY_PAID",      # confirmed: units of service, decimal(15,4)
    "paid_amt":            "PMT_AMT",
    "pos":                 "POS_CODE",
    "form":                "FORM_TYP",
    "mods":                ["PROC_MOD_1", "PROC_MOD_2", "PROC_MOD_3", "PROC_MOD_4"],
}

# Logical claim fields a rule may require. The gate downgrades a rule that
# needs anything not listed here to a review lead (reason=missing_claim_field)
# instead of emitting SQL that silently returns nothing.
AVAILABLE_CLAIM_FIELDS = {
    "proc_cd", "member_id", "provider_id", "billing_provider_id", "srvc_bgn_dt",
    "units", "quantity_paid", "mod_1", "mod_2", "mod_3", "mod_4",
    "claim_id", "line_no", "pos_cd", "paid_amt", "paid_ind", "form_type",
}

# NCCI editions are claim-form specific. FORM_TYP value -> service category.
FORM_TYPE_MAP = {"A": "practitioner", "O": "outpatient"}

# Extra routing for service categories not covered by FORM_TYPE_MAP, e.g.
# {"dme": ["A"]} once the DME form value is confirmed in the claims table.
SERVICE_FORM_MAP: dict = {}

# ---------------------------------------------------------------- legacy
# Kept so the PoC authoring modules (ingest_ahcccs, demo_*, ch10_full_extract)
# still import cleanly. Not used by the MVP build.
CLAIM_COLS["provider"] = CLAIM_COLS["servicing_provider"]
CLAIM_COLS["claim"] = CLAIM_COLS["claim_id"]
CLAIM_COLS["member_id"] = CLAIM_COLS["member"]
POC_CODE_SET = None                                # None = scope from claims
AHCCCS_PDF_DIR = os.environ.get("PKB_AHCCCS_PDFS", "data/ahcccs")
LLM_BACKEND = os.environ.get("PKB_LLM", "STUB")
DBFM_ENDPOINT = os.environ.get("PKB_DBFM_ENDPOINT", "databricks-gpt-oss-20b")
HF_MODEL = os.environ.get("PKB_HF_MODEL", "openai/gpt-oss-20b")
LLM_MAX_TOKENS = 2048
LLM_TEMPERATURE = 0.0

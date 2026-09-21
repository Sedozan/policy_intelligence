"""
extract_rules.py  -  Stages 3-5: pre-filter, propose (LLM), ground, draft.

The model PROPOSES rules into the closed predicate grammar; it never decides.
Every proposal passes a deterministic grounding check (the quoted sentence must
exist in the section, every code must exist in the section, thresholds must be
present or be marked derived) and then lands in main.sedo.state_rules as a
DRAFT for a named reviewer. Nothing reaches a detector without that approval.

Backends (config.LLM_BACKEND):
  STUB     deterministic pattern extractor. Runs the whole pipeline with no
           model; also a floor for the simplest rule shapes.
  HFLOCAL  openai/gpt-oss-120b via transformers + openai-harmony ON THE CLUSTER -
           a mirror of SB's llm_summarization_v3 loop (final channel only, retry on
           truncation). The production path (2026-09-21).
  DBFM     Databricks Foundation Model serving endpoint, OpenAI-compatible. Optional.
  HYBRID   patterns + one of the above; refuses to run without the model unless
           told otherwise. A dead model is a loud failure, never a silent regex run.
"""
from __future__ import annotations

import difflib
import hashlib
import json
import re
from datetime import datetime

import config
import schema
from policy_docs import CODE_RE, WORD_NUMBERS, rule_signal

PROMPT_VERSION = config.PROMPT_VERSION

# --------------------------------------------------------------- prompt
SYSTEM_PROMPT = """You extract billing rules from Arizona Medicaid (AHCCCS) policy text into a fixed JSON grammar.
You are an extractor, not a decider: propose only what the text explicitly states, quote it verbatim, and
flag anything ambiguous instead of resolving it.

OUTPUT: a single JSON object: {"rules": [...], "no_rule_reason": string|null}
Each rule object:
  "statement"        one plain-English sentence stating the rule as written
  "predicate"        one of the types below, with its fields
  "codes"            list of CPT/HCPCS codes the rule applies to (may be empty)
  "verbatim_quote"   the EXACT sentence(s) from the text that state the rule. Copy it character for character.
  "derived"          true if any number in the predicate is computed rather than written (say how in derivation_note)
  "derivation_note"  string|null
  "ambiguous"        true if the text is contradictory, conditional on facts not in a claim, or unclear
  "ambiguity_note"   string|null
  "not_checkable_reason" string|null (required when predicate.type is "not_checkable")
  "population"       "all" unless the text restricts (e.g. "members under 21", "ALTCS members")
  "confidence"       0.0-1.0, your honest confidence that the rule is exactly what the text says

PREDICATE TYPES (closed grammar - use nothing else):
  {"type":"max_units_per_day","code":"98960","threshold":4}
  {"type":"max_units_per_period","code_set":["98960","98961","98962"],"threshold":24,"period":"month"}   period: month|year|benefit_year
  {"type":"code_pair_prohibited","code_a":"H2025","code_b":"H2026","modifier_indicator":"0","service_category":"state"}
  {"type":"code_not_covered","code":"11975"}
  {"type":"max_dollars_per_period","code_set":[...],"threshold":1000,"period":"benefit_year"}
  {"type":"frequency_limit","code_set":[...],"threshold":8,"period":"benefit_year","unit":"visits"}
  {"type":"modifier_required","code_set":[...],"modifier":"GT"}
  {"type":"provider_type_allowed","code_set":[...],"provider_types":["07"]}
  {"type":"not_checkable","reason":"<one of: requires_medical_record | requires_member_attribute | ambiguous_source |
        cross_source_conflict | unit_incommensurability | policy_reference | no_claim_field>"}

RULES OF EXTRACTION
1. One rule per distinct obligation. "Codes A, B and C cannot be billed together on the same day" -> one
   code_pair_prohibited per pair (A/B, A/C, B/C). "maximum of four units per day (codes X, Y)" -> one
   max_units_per_day per code.
2. Numbers written as words ("four units") are fine; put the integer in the predicate and quote the words.
3. If a limit depends on information not on a claim (diagnosis justification, medical necessity, member age
   or program, prior visits) emit not_checkable with the right reason. Do not guess.
4. If the text contradicts itself or is unclear, emit not_checkable with reason ambiguous_source and explain.
5. A code RANGE (99201-99499) is not an enumerated set: emit not_checkable (cross_source_conflict or
   no_claim_field) and name the range in the statement.
6. Policy adoption statements ("AHCCCS follows CCI") are not_checkable with reason policy_reference.
7. Never invent a code, a number, or a quote. If you cannot quote it, do not emit it.
8. If the section states no billing rule, return {"rules": [], "no_rule_reason": "..."}.
"""

USER_TEMPLATE = """CHAPTER {chapter}: {title}
SECTION: {heading}   (pages {page_start}-{page_end}, revision {revision})

TEXT:
\"\"\"
{text}
\"\"\"

Return the JSON object now."""

_REQUIRED_KEYS = {
    "code_pair_prohibited": ("code_a", "code_b"),
    "max_units_per_line": ("code", "threshold"),
    "max_units_per_dos": ("code", "threshold"),
    "max_units_per_day": ("code", "threshold"),
    "max_units_per_period": ("threshold", "period"),
    "code_not_covered": (),
    "max_dollars_per_period": ("threshold", "period"),
    "frequency_limit": ("threshold",),
    "modifier_required": ("modifier",),
    "provider_type_allowed": ("provider_types",),
    "not_checkable": ("reason",),
}
_FIELDS_BY_TYPE = {
    "code_pair_prohibited": ["proc_cd", "member_id", "provider_id", "srvc_bgn_dt", "mod_1", "mod_2", "mod_3", "mod_4"],
    "max_units_per_day": ["proc_cd", "units", "member_id", "srvc_bgn_dt"],
    "max_units_per_period": ["proc_cd", "units", "member_id", "srvc_bgn_dt"],
    "max_units_per_line": ["proc_cd", "units"],
    "max_units_per_dos": ["proc_cd", "units", "member_id", "provider_id", "srvc_bgn_dt"],
    "code_not_covered": ["proc_cd"],
}


def _period_token(word: str) -> str:
    """Map the period phrase in the source to a compiler period token.
    AHCCCS 'contract year' / 'benefit year' both mean Oct 1 - Sep 30 (verified in
    the manual), which the compiler models as 'benefit_year'."""
    w = (word or "").strip().lower()
    if w in ("contract year", "benefit year"):
        return "benefit_year"
    if w == "year":
        return "year"
    return "month"


# ------------------------------------------------------------- backends
class Backend:
    name = "base"
    model = ""

    def complete(self, system: str, user: str) -> str:
        raise NotImplementedError


class ModelUnavailable(RuntimeError):
    """The model backend cannot be reached. Carries every attempt's real error."""


def _workspace_host_token() -> tuple[str, str]:
    """Workspace URL + token for the OpenAI-compatible serving route.
    Env vars first (jobs, local dev), then the notebook context (interactive
    notebooks do NOT export DATABRICKS_HOST/TOKEN - reading os.environ[...] there
    raises KeyError, which is exactly the 2026-09-20 failure)."""
    import os
    host, token = os.environ.get("DATABRICKS_HOST"), os.environ.get("DATABRICKS_TOKEN")
    if host and token:
        return host.rstrip("/"), token
    try:
        import IPython
        dbutils = IPython.get_ipython().user_ns["dbutils"]
        ctx = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
        return ctx.apiUrl().get().rstrip("/"), ctx.apiToken().get()
    except Exception as e:
        raise RuntimeError(f"DATABRICKS_HOST/DATABRICKS_TOKEN not set and notebook context unavailable "
                           f"({type(e).__name__}: {e})")


class DBFMBackend(Backend):
    """Databricks Foundation Model serving endpoint.

    Three ways to reach it, tried in order; EVERY failure is recorded with its full
    message and, if all fail, raised together as ModelUnavailable. Nothing is
    swallowed: a bare `except: fall back` hid the real error for a whole run.
      1. databricks-sdk  WorkspaceClient().serving_endpoints.get_open_ai_client()  (needs `openai`)
      2. mlflow.deployments  get_deploy_client("databricks").predict(...)          (always on a cluster)
      3. openai.OpenAI(base_url=<workspace>/serving-endpoints, api_key=<token>)   (env or notebook ctx)
    """
    name = "DBFM"

    def __init__(self, endpoint: str | None = None):
        self.model = endpoint or config.DBFM_ENDPOINT
        self._chat = None            # callable(msgs, json_mode) -> (content, finish_reason)
        self.client_route = None
        self.client_errors: list[str] = []
        self.calls = 0
        self.failures = 0
        self.last_error: str | None = None

    # ---- client construction -------------------------------------------------
    def _build(self):
        if self._chat is not None:
            return self._chat
        errors = []
        kw = dict(temperature=config.LLM_TEMPERATURE, max_tokens=config.LLM_MAX_TOKENS)

        def _via_openai_client(client):
            def chat(msgs, json_mode):
                extra = {"response_format": {"type": "json_object"}} if json_mode else {}
                r = client.chat.completions.create(model=self.model, messages=msgs, **kw, **extra)
                return (r.choices[0].message.content or ""), getattr(r.choices[0], "finish_reason", None)
            return chat

        try:
            from databricks.sdk import WorkspaceClient
            client = WorkspaceClient().serving_endpoints.get_open_ai_client()
            self._chat, self.client_route = _via_openai_client(client), "databricks-sdk openai client"
            return self._chat
        except Exception as e:
            errors.append(f"databricks-sdk: {type(e).__name__}: {e}")
        try:
            from mlflow.deployments import get_deploy_client
            client = get_deploy_client("databricks")

            def chat(msgs, json_mode):
                inputs = {"messages": msgs, **kw}
                if json_mode:
                    inputs["response_format"] = {"type": "json_object"}
                r = client.predict(endpoint=self.model, inputs=inputs)
                ch = r["choices"][0]
                return (ch.get("message", {}).get("content") or ""), ch.get("finish_reason")
            self._chat, self.client_route = chat, "mlflow.deployments"
            return self._chat
        except Exception as e:
            errors.append(f"mlflow.deployments: {type(e).__name__}: {e}")
        try:
            from openai import OpenAI
            host, token = _workspace_host_token()
            client = OpenAI(api_key=token, base_url=f"{host}/serving-endpoints")
            self._chat, self.client_route = _via_openai_client(client), f"openai client @ {host}"
            return self._chat
        except Exception as e:
            errors.append(f"openai+workspace: {type(e).__name__}: {e}")
        self.client_errors = errors
        raise ModelUnavailable(f"DBFM endpoint {self.model!r}: no client route works:\n  - " + "\n  - ".join(errors))

    # ---- completion ----------------------------------------------------------
    def complete(self, system: str, user: str) -> str:
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        chat = self._build()
        self.calls += 1
        try:
            try:
                content, finish = chat(msgs, True)
            except Exception as e_json:                 # endpoint may not support json_object
                content, finish = chat(msgs, False)     # plain retry; the json-mode error is kept for the record
                self.last_error = f"json_object mode rejected ({type(e_json).__name__}: {str(e_json)[:160]}); plain mode used"
        except Exception as e:
            self.failures += 1
            self.last_error = f"{type(e).__name__}: {e}"
            raise
        if finish == "length":
            # a truncated JSON is not a "no rules" answer; make it a visible, retryable failure
            raise ValueError(f"model output truncated at max_tokens={config.LLM_MAX_TOKENS} (finish_reason=length); "
                             f"raise config.LLM_MAX_TOKENS or split the section")
        return content

    def healthcheck(self) -> tuple[bool, str]:
        """One tiny call. Returns (ok, human-readable detail including the model's reply)."""
        try:
            reply = self.complete("Reply with a JSON object.", 'Return exactly {"ok": true}.')
            return True, (f"DBFM {self.model} OK via {self.client_route}; reply: {reply.strip()[:120]!r}")
        except Exception as e:
            return False, f"DBFM {self.model} FAILED: {type(e).__name__}: {e}"


# ------------------------------------------------ HFLOCAL: gpt-oss on the cluster
# One model per Python process. The notebook loads it once (exactly as the
# summarization notebook does) and hands the objects in; re-instantiating the
# backend never reloads 120B parameters.
_HF_RUNTIME: dict = {"model": None, "tokenizer": None, "harmony_enc": None, "model_id": None}


def load_hf_runtime(model_id: str | None = None, hf_token_path: str | None = None, log_fn=print) -> dict:
    """Load tokenizer + model + harmony encoding ONCE, the same way llm_summarization_v3 does:
    login(hf_token) ; AutoTokenizer/AutoModelForCausalLM.from_pretrained(torch_dtype="auto",
    device_map="auto", trust_remote_code=True) ; load_harmony_encoding(HARMONY_GPT_OSS)."""
    model_id = model_id or config.HF_MODEL
    if _HF_RUNTIME["model"] is not None and _HF_RUNTIME["model_id"] == model_id:
        return _HF_RUNTIME
    import time as _t
    t0 = _t.time()
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from openai_harmony import load_harmony_encoding, HarmonyEncodingName
    token_path = config.HF_TOKEN_PATH if hf_token_path is None else hf_token_path
    if token_path:
        from huggingface_hub import login
        with open(token_path) as f:
            login(f.read().strip())
    log_fn(f"Loading model: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto", device_map="auto",
                                                 trust_remote_code=True)
    harmony_enc = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
    _HF_RUNTIME.update(model=model, tokenizer=tokenizer, harmony_enc=harmony_enc, model_id=model_id)
    dev = getattr(model, "hf_device_map", None) or getattr(model, "device", "auto")
    log_fn(f"Model loaded: {model_id} in {_t.time() - t0:.0f}s | device map: {dev}")
    return _HF_RUNTIME


def _extract_channel_text(msg) -> tuple[str | None, str]:
    """(channel, text) from an openai-harmony Message: channel is 'analysis' | 'final' |
    'commentary' | None; content is a list of items carrying .text."""
    channel = getattr(msg, "channel", None)
    content = getattr(msg, "content", msg)
    if isinstance(content, str):
        return channel, content
    parts = []
    for item in (content or []):
        parts.append(getattr(item, "text", None) or (item.get("text", "") if isinstance(item, dict) else str(item)))
    return channel, "".join(parts)


class HFLocalBackend(Backend):
    """gpt-oss via transformers + openai-harmony, mirroring SB's generate_with_harmony /
    generate_final_with_retry:

      tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_dict=True,
                                    reasoning_effort=<level>)
      model.generate(input_ids, max_new_tokens=<budget>, do_sample=False)
      harmony_enc.parse_messages_from_completion_tokens(new_ids, role=Role.ASSISTANT)
      keep channel == "final"; "analysis" is reasoning and is never returned
      no final + hit budget + no parse error  -> retry once at RETRY_BUDGET_FACTOR x budget

    Differences from the summarizer, on purpose: the "leak" check is replaced by the
    caller's JSON grammar check (a final that is not our JSON is a failure, never a
    rule); and there is NO deterministic fallback - a failed call raises, so the
    section is recorded as a model failure and retried next run (never cached).
    """
    name = "HFLOCAL"

    def __init__(self, model_id: str | None = None, reasoning_level: str | None = None,
                 model=None, tokenizer=None, harmony_enc=None):
        self.model = model_id or config.HF_MODEL          # Backend.model = the model's name
        self.reasoning_level = reasoning_level or config.LLM_REASONING
        if model is not None and tokenizer is not None and harmony_enc is not None:
            _HF_RUNTIME.update(model=model, tokenizer=tokenizer, harmony_enc=harmony_enc, model_id=self.model)
        self.calls = 0
        self.failures = 0
        self.retries = 0
        self.last_error: str | None = None
        self.last_meta: dict | None = None
        self.client_route = "transformers+openai-harmony (local)"

    def _rt(self) -> dict:
        rt = _HF_RUNTIME
        if rt["model"] is None or rt["tokenizer"] is None or rt["harmony_enc"] is None:
            rt = load_hf_runtime(self.model)
        return rt

    # ---- one generate, SB's generate_with_harmony -----------------------------
    def _generate(self, messages: list[dict], max_new_tokens: int | None = None) -> tuple[str, str, dict]:
        import torch
        from openai_harmony import Role
        rt = self._rt()
        model, tokenizer, enc = rt["model"], rt["tokenizer"], rt["harmony_enc"]
        budget = max_new_tokens or config.REASONING_TOKEN_BUDGET.get(self.reasoning_level, 3000)
        inputs = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True,
                                               return_dict=True, reasoning_effort=self.reasoning_level)
        input_ids = inputs["input_ids"].to(model.device)
        with torch.no_grad():
            output_ids = model.generate(input_ids, max_new_tokens=budget, do_sample=False)
        new_token_ids = output_ids[0][input_ids.shape[1]:].tolist()

        final_text, analysis_text, parse_error = "", "", None
        try:
            for msg in enc.parse_messages_from_completion_tokens(new_token_ids, role=Role.ASSISTANT):
                channel, content = _extract_channel_text(msg)
                if channel == "final":
                    final_text = content
                elif channel == "analysis":
                    analysis_text += content + " "
        except Exception as e:
            parse_error = f"{type(e).__name__}: {e}"
        meta = {"final_channel_present": bool(final_text.strip()), "parse_error": parse_error,
                "tokens_generated": len(new_token_ids), "budget": budget,
                "hit_budget": len(new_token_ids) >= budget - 2, "reasoning_level": self.reasoning_level}
        return final_text.strip(), analysis_text.strip(), meta

    # ---- SB's generate_final_with_retry, fail-loud ------------------------------
    def complete(self, system: str, user: str) -> str:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        self.calls += 1
        try:
            final, _, meta = self._generate(messages)
            meta["retried"] = False
            if not meta["final_channel_present"] and meta["hit_budget"] and not meta["parse_error"]:
                retry_budget = int(meta["budget"] * config.RETRY_BUDGET_FACTOR)
                self.retries += 1
                final, _, meta = self._generate(messages, max_new_tokens=retry_budget)
                meta["retried"] = True
            self.last_meta = meta
            if meta["parse_error"] and not meta["final_channel_present"]:
                raise ValueError(f"harmony parse failed: {meta['parse_error']}")
            if not meta["final_channel_present"]:
                raise ValueError(f"no final channel after {meta['tokens_generated']}/{meta['budget']} tokens "
                                 f"(retried={meta['retried']}); raise REASONING_TOKEN_BUDGET or split the section")
            return final
        except Exception as e:
            self.failures += 1
            self.last_error = f"{type(e).__name__}: {e}"
            raise

    def healthcheck(self) -> tuple[bool, str]:
        try:
            import time as _t
            t0 = _t.time()
            reply = self.complete("Reply with a JSON object.", 'Return exactly {"ok": true}.')
            m = self.last_meta or {}
            return True, (f"HFLOCAL {self.model} OK ({_t.time() - t0:.1f}s, {m.get('tokens_generated')} tokens, "
                          f"reasoning={self.reasoning_level}); reply: {reply[:120]!r}")
        except Exception as e:
            return False, f"HFLOCAL {self.model} FAILED: {type(e).__name__}: {e}"


class StubBackend(Backend):
    """Deterministic pattern extractor. No model. Emits the same JSON contract."""
    name = "STUB"
    model = "pattern-v1"

    _SENT = re.compile(r"(?<=[.;])\s+(?=[A-Z(])")

    def complete(self, system: str, user: str) -> str:
        m = re.search(r'TEXT:\n"""\n(.*?)\n"""', user, re.S)
        text = m.group(1) if m else user
        rules, prev_codes = [], []
        for sent in self._SENT.split(text.replace("\n", " ")):
            s = sent.strip()
            if not s:
                continue
            codes = sorted(set(CODE_RE.findall(s)))
            # "These codes cannot be billed together..." refers back to the previous sentence
            carry = prev_codes if (not codes and re.match(r"^(these|those|both|the above) codes?\b", s.lower())) else []
            rules += self._patterns(s, carry)
            if codes:
                prev_codes = codes
        rules += self._section_level(text)
        return json.dumps({"rules": rules, "no_rule_reason": None if rules else "no rule pattern matched"})

    @staticmethod
    def _num(tok: str) -> int | None:
        tok = tok.lower()
        if tok.isdigit():
            return int(tok)
        return WORD_NUMBERS.get(tok)

    def _patterns(self, s: str, carry: list[str] | None = None) -> list[dict]:
        out = []
        low = s.lower()
        codes = sorted(set(CODE_RE.findall(s))) or list(carry or [])
        base = {"verbatim_quote": s, "derived": False, "derivation_note": None, "ambiguous": False,
                "ambiguity_note": None, "not_checkable_reason": None, "population": "all", "confidence": 0.6}

        m = re.search(r"maximum of (\w+(?:-\w+)?) units? per day", low)
        if m and codes and self._num(m.group(1)):
            n = self._num(m.group(1))
            for c in codes:
                out.append({**base, "statement": f"{c}: maximum {n} unit(s) per day.",
                            "predicate": {"type": "max_units_per_day", "code": c, "threshold": n}, "codes": [c]})
        m = re.search(r"(?:up to|maximum of|no more than|not to exceed|limited to) "
                      r"(\w+(?:-\w+)?) units? per (contract year|benefit year|month|year)", low)
        if m and codes and self._num(m.group(1)):
            n = self._num(m.group(1))
            period = _period_token(m.group(2))
            out.append({**base, "statement": f"{'/'.join(codes)}: maximum {n} units per {m.group(2)}.",
                        "predicate": {"type": "max_units_per_period", "code_set": codes, "threshold": n,
                                      "period": period}, "codes": codes})
        if re.search(r"cannot be billed (together )?on the same day", low) and len(codes) >= 2:
            for i in range(len(codes)):
                for j in range(i + 1, len(codes)):
                    a, b = codes[i], codes[j]
                    out.append({**base, "statement": f"{a} and {b} may not be billed on the same day for the same member.",
                                "predicate": {"type": "code_pair_prohibited", "code_a": a, "code_b": b,
                                              "modifier_indicator": "0", "service_category": "state"}, "codes": [a, b]})
        if re.search(r"do not bill|is not a covered service|not an ahcccs[- ]covered service|not reimburs", low) and codes:
            cond = re.search(r"\bunless\b|\bexcept\b|\bprior authorization\b|\bwith (?:a )?pa\b|\bif\b|\bwhen\b|\bonly\b", low)
            for c in codes:
                if cond:
                    # "Do not bill X unless prior authorization is obtained" is CONDITIONAL
                    # non-coverage. The grammar has no condition yet, so this is an honest
                    # lead with the quote - never an unconditional code_not_covered (F11).
                    out.append({**base, "statement": f"{c}: conditional non-coverage ('{cond.group(0)}' clause).",
                                "predicate": {"type": "not_checkable", "reason": "predicate_not_in_grammar"},
                                "codes": [c], "ambiguous": True,
                                "ambiguity_note": f"sentence carries a condition ('{cond.group(0)}'); "
                                                  "conditional coverage is not in the grammar yet",
                                "not_checkable_reason": "predicate_not_in_grammar", "confidence": 0.5})
                else:
                    out.append({**base, "statement": f"{c} is not a covered/reimbursable service.",
                                "predicate": {"type": "code_not_covered", "code": c}, "codes": [c]})
        if "modifier" in low and re.search(r"\d{5}\s*-\s*\d{5}", s):
            out.append({**base, "statement": s, "predicate": {"type": "not_checkable", "reason": "cross_source_conflict"},
                        "codes": [], "ambiguous": True,
                        "ambiguity_note": "modifier restriction over a code range; conflicts with NCCI bypass rules",
                        "not_checkable_reason": "cross_source_conflict", "confidence": 0.5})
        if re.search(r"follows medicare'?s correct coding initiative", low):
            out.append({**base, "statement": s, "predicate": {"type": "not_checkable", "reason": "policy_reference"},
                        "codes": [], "not_checkable_reason": "policy_reference", "confidence": 0.7})
        return out


def _stub_section_level(self, text: str) -> list[dict]:
    """Contradictions usually span sentences: 'not covered.' ... 'will be re-instated'."""
    low = text.lower()
    out = []
    if "not covered" in low and re.search(r"re-?instated", low):
        sents = [x for x in self._SENT.split(text.replace("\n", " ")) if re.search(r"re-?instated", x, re.I)]
        if sents:
            out.append({"statement": "Coverage statement is contradictory: the section says both 'not covered' and "
                                     "'re-instated' for the same service/population.",
                        "predicate": {"type": "not_checkable", "reason": "ambiguous_source"}, "codes": [],
                        "verbatim_quote": sents[0].strip(), "derived": False, "derivation_note": None,
                        "ambiguous": True, "ambiguity_note": "text states both 'not covered' and 're-instated'",
                        "not_checkable_reason": "ambiguous_source", "population": "all", "confidence": 0.4})
    return out


StubBackend._section_level = _stub_section_level


class HybridBackend(Backend):
    """Run the deterministic pattern extractor AND a model on the same section,
    then union their proposals. A rule BOTH find is marked corroborated and gets
    a confidence bump; a rule only one finds is kept and flows to review like any
    other. Every proposal still passes the same grounding gate.

    Why combine them rather than pick one:
      * patterns are 100% precise on the shapes they know (a written 'maximum of
        four units per day') and cost nothing - a high-precision floor.
      * the model catches the prose-heavy rules patterns can't express.
      * agreement between two independent methods is the strongest signal a rule
        is real; disagreement surfaces exactly what a reviewer should look at.
      * if the model is down or slow, the patterns still produce a KB - the
        pipeline degrades instead of failing.
    """
    name = "HYBRID"

    def __init__(self, model_backend: Backend, patterns: Backend | None = None,
                 require_model: bool = True):
        """require_model=True (default): a failed model call RAISES, so the section is
        recorded as a model failure and retried next run. It never silently becomes a
        patterns-only result - on 2026-09-20 every model call failed for a whole run and
        the output looked like a successful extraction. require_model=False opts into
        degraded patterns-only output, flagged `degraded` and never cached."""
        self.model_backend = model_backend
        self.patterns = patterns or StubBackend()
        self.require_model = require_model
        self.model = f"{self.patterns.model}+{model_backend.name}:{model_backend.model}"
        self.model_calls = 0
        self.model_failures = 0
        self.last_model_error: str | None = None

    @staticmethod
    def _key(r: dict):
        """Full canonical proposition (F12): two proposals are the SAME rule only if
        every semantic field agrees - a monthly and an annual cap on the same code
        are different rules, not corroboration."""
        p = r.get("predicate") or {}
        codes = tuple(sorted(str(c).upper() for c in (r.get("codes") or [])))
        return (p.get("type"), codes, p.get("threshold"), p.get("period"), p.get("scope"),
                str(p.get("code_a") or "").upper(), str(p.get("code_b") or "").upper(),
                str(p.get("code") or "").upper(), str(p.get("modifier") or "").upper(),
                str(p.get("reason") or ""), (r.get("population") or "all").strip().lower())

    @staticmethod
    def _rules(js: str) -> list[dict]:
        try:
            return (json.loads(js) or {}).get("rules") or []
        except Exception:
            return []

    def complete(self, system: str, user: str) -> str:
        pat = self._rules(self.patterns.complete(system, user))
        for r in pat:
            r["source"] = "pattern"
        degraded = False
        self.model_calls += 1
        try:
            mdl = self._rules(self.model_backend.complete(system, user))
        except Exception as e:
            self.model_failures += 1
            self.last_model_error = f"{type(e).__name__}: {e}"
            if self.require_model:
                raise                                   # propose() records it as a model failure
            mdl, degraded = [], True
            if self.model_failures in (1, 10, 100) or self.model_failures % 500 == 0:
                print(f"    HYBRID: model backend failed #{self.model_failures} - {self.last_model_error[:300]}\n"
                      f"    HYBRID: continuing with patterns only because require_model=False; "
                      f"this output is NOT a complete extraction")
        for r in mdl:
            r["source"] = "model"

        merged: dict = {}
        for r in pat + mdl:                       # patterns first; model can enrich
            k = self._key(r)
            if k in merged:
                prev = merged[k]
                if r["source"] != prev["source"] and prev["source"] != "pattern+model":
                    # corroboration means two INDEPENDENT methods agree; a duplicate
                    # from the same backend is not evidence and earns no bump.
                    prev["source"] = "pattern+model"
                    prev["confidence"] = round(min(0.97, max(float(prev.get("confidence", 0.5)),
                                                              float(r.get("confidence", 0.5))) + 0.1), 2)
                if len(r.get("verbatim_quote", "")) > len(prev.get("verbatim_quote", "")):
                    prev["verbatim_quote"] = r["verbatim_quote"]        # keep the fuller quote
            else:
                merged[k] = dict(r)
        return json.dumps({"rules": list(merged.values()), "degraded": degraded,
                           "no_rule_reason": None if merged else "no rule found by patterns or model"})


def make_backend(name: str | None = None, allow_patterns_only: bool = False) -> Backend:
    name = (name or config.LLM_BACKEND).upper()
    if name == "STUB":
        return StubBackend()
    if name == "DBFM":
        return DBFMBackend()
    if name == "HFLOCAL":
        return HFLocalBackend()
    if name == "HYBRID":
        inner = getattr(config, "HYBRID_MODEL_BACKEND", "DBFM").upper()
        if inner in ("HYBRID", "STUB"):
            raise ValueError("config.HYBRID_MODEL_BACKEND must be a model backend (DBFM or HFLOCAL)")
        return HybridBackend(make_backend(inner), require_model=not allow_patterns_only)
    raise ValueError(f"unknown LLM_BACKEND {name!r}")


def healthcheck(backend: Backend) -> tuple[bool, str]:
    """Is the MODEL half alive? One real call, before any chapter is touched.
    STUB is honest about being patterns-only. Returns (ok, detail)."""
    target = backend.model_backend if isinstance(backend, HybridBackend) else backend
    if isinstance(target, StubBackend):
        return True, "STUB: deterministic patterns only - no model. Output is a floor, not an extraction."
    if hasattr(target, "healthcheck"):
        return target.healthcheck()
    try:
        reply = target.complete("Reply with a JSON object.", 'Return exactly {"ok": true}.')
        return True, f"{target.name}:{target.model} OK; reply: {reply.strip()[:120]!r}"
    except Exception as e:
        return False, f"{target.name}:{target.model} FAILED: {type(e).__name__}: {e}"


def model_health(backend: Backend) -> dict:
    """Call/failure counters for the run log (zeros for STUB)."""
    hy = backend if isinstance(backend, HybridBackend) else None
    mb = hy.model_backend if hy else backend
    return {
        "model_backend": f"{mb.name}:{mb.model}",
        "model_route": getattr(mb, "client_route", None),
        "model_calls": hy.model_calls if hy else getattr(mb, "calls", 0),
        "model_failures": hy.model_failures if hy else getattr(mb, "failures", 0),
        "model_retries": getattr(mb, "retries", 0),
        "last_model_error": (hy.last_model_error if hy else getattr(mb, "last_error", None)),
        "patterns_only_allowed": (not hy.require_model) if hy else isinstance(mb, StubBackend),
    }


# ------------------------------------------------------------- proposing
def _extract_json(text: str) -> dict:
    """Find the last well-formed JSON object in a model response (tolerates fences, harmony channels)."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.S)
    try:
        return json.loads(text)
    except Exception:
        pass
    starts = [m.start() for m in re.finditer(r"\{", text)]
    for st in reversed(starts):
        depth = 0
        for i in range(st, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(text[st:i + 1])
                        if isinstance(obj, dict) and "rules" in obj:
                            return obj
                    except Exception:
                        break
    raise ValueError("no JSON object with a 'rules' key in model output")


def propose(section: dict, inv: dict, backend: Backend) -> tuple[list[dict], str | None]:
    user = USER_TEMPLATE.format(chapter=inv["chapter"], title=inv["title"], heading=section["heading"],
                                page_start=section["page_start"], page_end=section["page_end"],
                                revision=inv.get("revision_date") or "unknown", text=section["text"])
    try:
        raw = backend.complete(SYSTEM_PROMPT, user)
    except Exception as e:                       # endpoint timeout etc.: a retryable error, never a crash
        return [], f"backend error: {type(e).__name__}: {e}"
    try:
        obj = _extract_json(raw)
    except Exception as e:
        return [], f"unparseable model output: {e}"
    if obj.get("degraded"):                      # HYBRID ran patterns only; the model half failed
        return _clean_rules(obj), "degraded: model backend failed; patterns only"
    return _clean_rules(obj), obj.get("no_rule_reason")


def _clean_rules(obj: dict) -> list[dict]:
    clean = []
    for r in obj.get("rules") or []:
        if not isinstance(r, dict) or not isinstance(r.get("predicate"), dict):
            continue
        r.setdefault("codes", [])
        r.setdefault("confidence", 0.5)
        r.setdefault("population", "all")
        clean.append(r)
    return clean


# outcomes that must NOT be cached: a rerun has to retry them (F14)
def _retryable(why: str | None) -> bool:
    return bool(why) and (why.startswith("unparseable") or why.startswith("backend error")
                          or why.startswith("degraded"))


def _failure_kind(why: str) -> str:
    if why.startswith("backend error"):
        return "model_failed"
    if why.startswith("degraded"):
        return "degraded_patterns_only"
    return "unparseable"


# ------------------------------------------------------------- grounding
def _norm(s: str) -> str:
    s = (s or "").lower().replace("–", "-").replace("—", "-").replace("’", "'")
    s = re.sub(r"[\"“”]", "", s)
    return re.sub(r"\s+", " ", s).strip().rstrip(".;:")


def _sentences(text: str) -> list[str]:
    return [x.strip() for x in re.split(r"(?<=[.;])\s+", text.replace("\n", " ")) if x.strip()]


def _threshold_present(n, text_norm: str) -> bool:
    if n is None:
        return True
    try:
        n = int(n)
    except (TypeError, ValueError):
        return False
    if re.search(rf"\b{n}\b", text_norm):
        return True
    words = [w for w, v in WORD_NUMBERS.items() if v == n]
    return any(re.search(rf"\b{re.escape(w)}\b", text_norm) for w in words)


# Words that flip or condition the meaning of a billing sentence. A fuzzy quote match
# (ratio >= 0.85) is only trusted when the quote and the source sentence agree on
# these EXACTLY, plus every number: "is covered" vs "is not covered" scores 0.9 on
# characters and means the opposite (F11).
_MEANING_TOKENS = {
    "not", "no", "never", "cannot", "non", "without", "unless", "except", "excluding",
    "only", "prior", "authorization", "authorized", "if", "when", "must", "may", "shall",
    "per", "day", "month", "year", "week", "lifetime", "each", "maximum", "minimum",
}


def _meaning_signature(s_norm: str) -> tuple:
    toks = re.findall(r"[a-z]+|\d+", s_norm)
    return (tuple(t for t in toks if t in _MEANING_TOKENS), tuple(t for t in toks if t.isdigit()))


def _predicate_codes(p: dict) -> list[str]:
    out = []
    for k in ("code_a", "code_b", "code"):
        if p.get(k):
            out.append(str(p[k]).upper())
    out += [str(c).upper() for c in (p.get("code_set") or [])]
    return list(dict.fromkeys(out))


def predicate_valid(p: dict) -> tuple[bool, str]:
    t = p.get("type")
    if t not in schema.PREDICATE_TYPES:
        return False, f"unknown predicate type {t!r}"
    for k in _REQUIRED_KEYS.get(t, ()):
        if p.get(k) in (None, "", []):
            return False, f"{t} missing {k}"
    if t == "code_pair_prohibited" and str(p["code_a"]).upper() == str(p["code_b"]).upper():
        return False, "self pair"
    if t in ("max_units_per_period", "code_not_covered", "max_dollars_per_period") and not _predicate_codes(p):
        return False, f"{t} needs code or code_set"
    if t == "max_units_per_period" and p.get("period") not in ("month", "year", "benefit_year"):
        return False, f"bad period {p.get('period')!r}"
    if "threshold" in p:
        n = _integral(p["threshold"])
        if n is None:
            return False, f"threshold not an integer ({p['threshold']!r})"   # 4.9 must not become 4
        if n <= 0:
            return False, "threshold must be positive"
    return True, ""


def _integral(v) -> int | None:
    """int, integral float (4.0) or digit string ('4') -> int; anything else -> None."""
    if isinstance(v, bool):
        return None
    if isinstance(v, int):
        return v
    if isinstance(v, float):
        return int(v) if v.is_integer() else None
    if isinstance(v, str) and v.strip().isdigit():
        return int(v.strip())
    return None


def ground(proposal: dict, section: dict) -> dict:
    """Deterministic verification of one proposal against its source section.

    This is an evidence gate, not a semantic proof: it confirms the quote exists,
    every code the predicate names is in the section, and the threshold appears in
    the QUOTED SENTENCE (not merely somewhere in the section). Two tightenings over
    the PoC, both from external review:
      * codes: a proposal is dropped unless ALL its codes are in the source. Passing
        on 'some' let a real code + an invented code through together.
      * threshold: matched against the quoted sentence, so a section that says
        "4 units" in one rule and "24 units" in another cannot ground 24 onto the
        first. A threshold not found in the quote is kept but flagged (derived).
    Semantic correspondence (does the quote actually assert THIS relation?) is still
    the reviewer's job; see the review workbook's editable predicate.
    """
    text_norm = _norm(section["text"])
    quote = proposal.get("verbatim_quote") or ""
    qn = _norm(quote)
    matched = ""                       # the source sentence the quote resolves to
    if qn and qn in text_norm:
        quote_found, matched = "exact", qn
    elif qn:
        best_ratio, best_sent = 0.0, ""
        for s in _sentences(section["text"]):
            r = difflib.SequenceMatcher(None, qn, _norm(s)).ratio()
            if r > best_ratio:
                best_ratio, best_sent = r, _norm(s)
        quote_found = "fuzzy" if best_ratio >= 0.85 else "none"
        matched = best_sent if quote_found == "fuzzy" else ""
        if quote_found == "fuzzy" and _meaning_signature(qn) != _meaning_signature(matched):
            # characters agree, meaning may not: negation / condition / number drift
            quote_found, matched = "drift", best_sent
    else:
        quote_found = "none"

    p = proposal.get("predicate") or {}
    codes = _predicate_codes(p) + [str(c).upper() for c in proposal.get("codes") or []]
    codes = list(dict.fromkeys(codes))
    if codes:
        missing = [c for c in codes if c.lower() not in text_norm]
        codes_found = "all" if not missing else ("some" if len(missing) < len(codes) else "none")
    else:
        codes_found, missing = "n/a", []
    # threshold must appear in the QUOTED sentence, not anywhere in the section
    thr_scope = matched or (qn if quote_found == "exact" else "")
    thr_ok = _threshold_present(p.get("threshold"), thr_scope) if "threshold" in p else None
    valid, why = predicate_valid(p)

    score = {"exact": 0.5, "fuzzy": 0.35, "drift": 0.0, "none": 0.0}[quote_found]
    score += {"all": 0.25, "n/a": 0.25, "some": 0.10, "none": 0.0}[codes_found]
    score += 0.15 if thr_ok in (True, None) else 0.05
    score += 0.10 if valid else 0.0

    keep, reason = True, None
    if quote_found == "none":
        keep, reason = False, "quote not found in source"
    elif quote_found == "drift":
        keep, reason = False, ("quote differs from source sentence in negation/condition/number "
                               f"tokens (nearest: {matched[:120]!r})")
    elif not valid:
        keep, reason = False, f"invalid predicate: {why}"
    elif codes_found == "none":
        keep, reason = False, "codes not in source"
    elif codes_found == "some":
        keep, reason = False, f"not all codes in source (missing {', '.join(missing)})"
    return {"quote_found": quote_found, "codes_found": codes_found, "threshold_found": thr_ok,
            "predicate_valid": valid, "score": round(score, 2), "keep": keep, "drop_reason": reason,
            "codes": codes}


# -------------------------------------------------------------- drafts
def semantic_key(chapter: str, p: dict, codes: list[str], population: str = "all",
                 effective_date: str | None = None, quote: str | None = None) -> str:
    """Logical identity of a proposed rule (F06). Includes population and effective
    date so a scope or date change re-enters review instead of inheriting the old
    decision. For UNCODED proposals (leads) the normalised quote is part of the key:
    two different obligations that both say "requires documentation" must not
    collapse into one. Statement wording is deliberately NOT in the key (the model
    rephrases; the rule does not change)."""
    sig = {"c": str(chapter), "p": p, "codes": sorted(codes),
           "pop": (population or "all").strip().lower(), "eff": effective_date}
    if not codes:
        sig["q"] = _norm(quote or "")
    return hashlib.sha256(json.dumps(sig, sort_keys=True, default=str).encode()).hexdigest()[:16]


_MONTHS = {m: i for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july",
     "august", "september", "october", "november", "december"], 1)}
# an effective-date cue that AHCCCS actually uses, in any of its phrasings
_EFF_TRIGGER = re.compile(
    r"(effective|beginning|dates? of service|on and after|on or after)", re.I)
# a date, numeric (4/1/2018) or written (April 1st, 2018 / April 1 2018)
_DATE_TOKEN = re.compile(
    r"(\d{1,2}/\d{1,2}/\d{4})"
    r"|([A-Za-z]{3,9}\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4})", re.I)


def _parse_date_token(tok: str) -> str | None:
    tok = tok.strip()
    m = re.match(r"(\d{1,2})/(\d{1,2})/(\d{4})", tok)
    if m:
        mo, d, y = m.groups()
        return f"{y}-{int(mo):02d}-{int(d):02d}"
    m = re.match(r"([A-Za-z]{3,9})\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})", tok)
    if m and m.group(1).lower() in _MONTHS:
        return f"{m.group(3)}-{_MONTHS[m.group(1).lower()]:02d}-{int(m.group(2)):02d}"
    return None


def _effective_from_quote(quote: str) -> str | None:
    """First date in the quote that is introduced by an effective-date cue.

    Handles the phrasings AHCCCS actually uses, e.g. 'effective 4/1/2018',
    'Beginning with dates of service on and after April 1st, 2018', and
    'effective for dates of service on or after January 1, 2020'. A date with no
    such cue is ignored (a random date in the prose is not an effective date)."""
    q = quote or ""
    for m in _DATE_TOKEN.finditer(q):
        if _EFF_TRIGGER.search(q[max(0, m.start() - 60):m.start()]):
            d = _parse_date_token(m.group(0))
            if d:
                return d
    return None


def to_draft_row(proposal: dict, g: dict, section: dict, inv: dict, backend: Backend, run_id: str) -> dict:
    from state_rules import rule_to_row
    p = dict(proposal["predicate"])
    t = p["type"]
    codes = g["codes"]
    ambiguous = bool(proposal.get("ambiguous")) or (g["threshold_found"] is False)
    anote = proposal.get("ambiguity_note")
    if g["threshold_found"] is False:
        anote = ((anote + " ") if anote else "") + "threshold not found verbatim in source; treat as derived."
    not_reason = proposal.get("not_checkable_reason") or (p.get("reason") if t == "not_checkable" else None)
    machine = t in schema.PREDICATE_TYPES_COMPILED and not not_reason
    pages = (f"page {section['page_start']}" if section["page_start"] == section["page_end"]
             else f"pages {section['page_start']}-{section['page_end']}")
    notes = f'Verbatim: "{proposal.get("verbatim_quote", "").strip()}"'
    if proposal.get("derived") and proposal.get("derivation_note"):
        notes += f" Derived: {proposal['derivation_note']}"
    if proposal.get("population") and proposal["population"] != "all":
        notes += f" Population: {proposal['population']}."
    if proposal.get("source"):
        notes += f" Found by: {proposal['source']}."

    rule = schema.Rule(
        origin="extracted", plane="coverage_scope", binding_status="state_policy",
        statement=(proposal.get("statement") or "").strip(),
        predicate=p, codes=codes, population=proposal.get("population") or "all",
        machine_checkable=machine, not_checkable_reason=not_reason,
        required_claim_fields=_FIELDS_BY_TYPE.get(t, []),
        ambiguity_flag=ambiguous, ambiguity_note=anote,
        source_doc=f"AHCCCS FFS Provider Billing Manual, Chapter {inv['chapter']} {inv['title']}",
        source_locator=f"{pages} — {section['heading']}", source_url=inv["url"],
        doc_version=f"rev {inv.get('revision_date') or 'unknown'}", doc_hash=inv["doc_hash"],
        effective_date=_effective_from_quote(proposal.get("verbatim_quote", "")),
        notes=notes,
    )
    row = rule_to_row(rule, chapter=str(inv["chapter"]), review_status="draft",
                      review_note=f"extracted by {backend.name}:{backend.model} prompt {PROMPT_VERSION}")
    key = semantic_key(inv["chapter"], p, codes, population=rule.population,
                       effective_date=rule.effective_date, quote=proposal.get("verbatim_quote"))
    llm_conf = float(proposal.get("confidence") or 0.5)
    row.update({
        "rule_id": f"EXT-{key}",
        "extraction_key": key,
        "verbatim_quote": proposal.get("verbatim_quote"),
        "section_id": section["section_id"],
        "extraction_confidence": round(0.5 * llm_conf + 0.5 * g["score"], 2),
        "extraction_run_id": run_id,
        "extraction_backend": f"{backend.name}:{backend.model}",
        "grounding": json.dumps(g, default=str),
    })
    return row


# ---------------------------------------------------------- orchestrate
def extract_chapter(sections: list[dict], inv: dict, backend: Backend, run_id: str,
                    only_candidates: bool = True, cache: dict | None = None,
                    log_fn=print, checkpoint_fn=None, checkpoint_every: int | None = None
                    ) -> tuple[list[dict], list[dict], dict]:
    """Returns (draft_rows, dropped, stats). `cache` maps cache_key -> proposals list.

    checkpoint_fn(cache) is called after every `checkpoint_every` NEW model calls (cache
    hits do not count) so a crash mid-chapter loses at most that many calls - the cache
    is the expensive artifact; drafts are re-derived from it for free on the next run."""
    cache = cache if cache is not None else {}
    checkpoint_every = checkpoint_every or getattr(config, "CACHE_FLUSH_EVERY", 0)
    new_calls_since_ckpt = 0
    drafts, dropped = [], []
    stats = {"chapter": inv["chapter"], "doc_hash": inv["doc_hash"], "run_id": run_id,
             "backend": f"{backend.name}:{backend.model}", "prompt_version": PROMPT_VERSION,
             "sections": len(sections), "sections_candidate": 0, "sections_from_cache": 0,
             "sections_skipped_noncandidate": 0, "skipped_headings": [],
             "proposals": 0, "grounded": 0, "dropped": 0,
             "unparseable": 0, "model_failed": 0, "degraded_patterns_only": 0,
             "started_at": datetime.now().isoformat(timespec="seconds")}
    for s in sections:
        if only_candidates and not s.get("has_rule_signal"):
            # the pre-filter is a hard recall gate: a skipped section never reaches the
            # model. Record what it dropped so recall can be measured, not assumed.
            stats["sections_skipped_noncandidate"] += 1
            if len(stats["skipped_headings"]) < 100:
                stats["skipped_headings"].append(
                    {"section_id": s.get("section_id"), "heading": s.get("heading"),
                     "n_chars": s.get("n_chars"), "rule_signal": s.get("rule_signal")})
            continue
        stats["sections_candidate"] += 1
        ck = f"{s['text_hash']}|{PROMPT_VERSION}|{backend.name}|{backend.model}"
        if ck in cache:
            proposals, why = cache[ck], None
            stats["sections_from_cache"] += 1
        else:
            proposals, why = propose(s, inv, backend)
            if _retryable(why):
                # F14: a failed or degraded call is NOT a "no rules" result. Leave it out
                # of the cache so the next run retries instead of reporting a cache hit.
                kind = _failure_kind(why)
                stats[kind] += 1
                stats.setdefault("retryable_sections", []).append(
                    {"section_id": s.get("section_id"), "heading": s.get("heading"), "kind": kind, "why": why})
                if kind != "model_failed" or stats["model_failed"] in (1, 5, 25, 100):
                    log_fn(f"    [{s['heading'][:40]}] {why[:300]}")
            else:
                cache[ck] = proposals
                new_calls_since_ckpt += 1
                if checkpoint_fn and checkpoint_every and new_calls_since_ckpt >= checkpoint_every:
                    try:
                        checkpoint_fn(cache)
                        stats["checkpoints"] = stats.get("checkpoints", 0) + 1
                    except Exception as e:                  # never let a checkpoint kill the run
                        log_fn(f"    checkpoint failed: {type(e).__name__}: {e}")
                    new_calls_since_ckpt = 0
        stats["proposals"] += len(proposals)
        for pr in proposals:
            g = ground(pr, s)
            if g["keep"]:
                drafts.append(to_draft_row(pr, g, s, inv, backend, run_id))
                stats["grounded"] += 1
            else:
                dropped.append({"chapter": inv["chapter"], "section_id": s["section_id"], "heading": s["heading"],
                                "statement": pr.get("statement"), "predicate": json.dumps(pr.get("predicate")),
                                "verbatim_quote": pr.get("verbatim_quote"), "drop_reason": g["drop_reason"],
                                "run_id": run_id})
                stats["dropped"] += 1
    # de-duplicate within the run by semantic key (keep the highest-confidence copy)
    best: dict[str, dict] = {}
    for d in drafts:
        k = d["extraction_key"]
        if k not in best or d["extraction_confidence"] > best[k]["extraction_confidence"]:
            best[k] = d
    drafts = list(best.values())
    stats["drafts_unique"] = len(drafts)
    stats.update(model_health(backend))
    stats["finished_at"] = datetime.now().isoformat(timespec="seconds")
    return drafts, dropped, stats

"""
Optimizer runner for **P:NeedPlansVerbalizer** (Gemini prompt tuning).

Input bundles ``simulate_financial_strategy`` outcome and ``users.goal_plan`` (recommended + alternative scenarios).

Run from ``finance-ai-penny`` repo root (``finance-ai-penny/.venv`` or ``finance-ai-llm-server/llm``):

  python3 active_experiments/need_plans_verbalizer_optimizer.py --test 0
  python3 active_experiments/need_plans_verbalizer_optimizer.py --test all
  python3 active_experiments/need_plans_verbalizer_optimizer.py --simulate-agent-outcome-id 1148 --print-input-only
  python3 active_experiments/need_plans_verbalizer_optimizer.py --simulate-agent-outcome-id 1148

DB-backed runs read ``SLAVE_DB`` from ``finance-ai-llm-server/config.ini``. Requires ``psycopg2-binary``.
"""

from __future__ import annotations

import argparse
import configparser
import json
import os
import re
import sys
import warnings
from datetime import date, datetime
from pathlib import Path
from typing import Any

try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
except Exception:  # pragma: no cover
    psycopg2 = None  # type: ignore[assignment]
    RealDictCursor = None  # type: ignore[assignment,misc]

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover
    load_dotenv = None

try:
    from google import genai
    from google.genai import types
    from google.genai.errors import ClientError
except Exception:  # pragma: no cover
    genai = None  # type: ignore[assignment]
    types = None  # type: ignore[assignment]
    ClientError = Exception  # type: ignore[misc, assignment]

if load_dotenv is not None:
    load_dotenv()

GEMINI_FLASH_LITE = "gemini-flash-lite-latest"
NEED_PLANS_VERBALIZER_THINKING_BUDGET = 512
NEED_PLANS_VERBALIZER_MAX_OUTPUT_TOKENS = 2048

_FINANCIAL_NEEDS_H1 = "# Financial Needs"
_SIMULATE_OUTCOME_TYPE = "simulate_financial_strategy"
_REPO_ROOT = Path(__file__).resolve().parent.parent


def _resolve_llm_server_root() -> Path:
    """Prod target is penny2; fall back to penny1 sibling for local config.ini."""
    docs_root = _REPO_ROOT.parent.parent
    candidates = [
        docs_root / "penny2" / "finance-ai-llm-server",
        _REPO_ROOT.parent / "finance-ai-llm-server",
        docs_root / "penny1" / "finance-ai-llm-server",
    ]
    for path in candidates:
        if (path / "config.py").is_file():
            return path
    return candidates[0]


_LLM_SERVER_ROOT = _resolve_llm_server_root()

SYSTEM_PROMPT = """You are Penny — a sharp, witty money coach who turns the diagnosed financial need into a hook users actually want to read.

`financial_need`: restate the identified need from ``# Financial Needs`` (one sentence, max **18 words**; punchy hook, no jargon).

`solution_options`: one entry per ``GOAL_PLAN`` scenario — recommended first per ``# Financial Strategy``. Copy each ``scenario_id`` and ``scenario_title`` verbatim. Each ``summary``: one sentence (max **20 words**); lead with the core move or payoff.

Max **70 words** total across ``financial_need`` and all ``summary`` values. Ground every **$** and timeline in the input. Fun and confident, never cheesy, patronizing, or naggy. No exclamation marks, superlatives, or emoji.
"""


def _build_output_schema() -> "types.Schema":
    if types is None:  # pragma: no cover
        raise RuntimeError("Install `google-genai` for this optimizer.")
    solution_option = types.Schema(
        type=types.Type.OBJECT,
        required=["scenario_id", "scenario_title", "summary"],
        properties={
            "scenario_id": types.Schema(
                type=types.Type.STRING,
                description="Verbatim scenario_id from the input GOAL_PLAN JSON.",
            ),
            "scenario_title": types.Schema(
                type=types.Type.STRING,
                description="Verbatim scenario_title from the input GOAL_PLAN JSON.",
            ),
            "summary": types.Schema(
                type=types.Type.STRING,
                description="One sentence per plan (max 20 words); lead with the core move or payoff.",
            ),
        },
    )
    return types.Schema(
        type=types.Type.OBJECT,
        required=["financial_need", "solution_options"],
        properties={
            "financial_need": types.Schema(
                type=types.Type.STRING,
                description="Punchy hook restating the identified need (one sentence, max 18 words).",
            ),
            "solution_options": types.Schema(
                type=types.Type.ARRAY,
                items=solution_option,
                description="One entry per GOAL_PLAN scenario; recommended first.",
            ),
        },
    )


def _validate_need_plans_response(
    parsed: Any,
    *,
    expected_plan_count: int | None = None,
) -> dict[str, Any]:
    if not isinstance(parsed, dict):
        raise ValueError("Response must be a JSON object")
    financial_need = parsed.get("financial_need")
    if not isinstance(financial_need, str) or not financial_need.strip():
        raise ValueError("financial_need must be a non-empty string")
    options = parsed.get("solution_options")
    if not isinstance(options, list) or not options:
        raise ValueError("solution_options must be a non-empty array")
    for i, entry in enumerate(options):
        if not isinstance(entry, dict):
            raise ValueError(f"solution_options[{i}] must be an object")
        for key in ("scenario_id", "scenario_title", "summary"):
            val = entry.get(key)
            if not isinstance(val, str) or not val.strip():
                raise ValueError(f"solution_options[{i}].{key} must be a non-empty string")
    if expected_plan_count is not None and expected_plan_count >= 1:
        if len(options) < expected_plan_count:
            raise ValueError(
                f"solution_options expected {expected_plan_count} entries, got {len(options)}"
            )
    return parsed


def _resolve_ideal_response(tc: dict[str, Any]) -> dict[str, Any] | None:
    ideal = tc.get("ideal_response")
    return ideal if isinstance(ideal, dict) else None


def _load_slave_db_connect_kwargs() -> dict[str, Any]:
    """Read ``finance-ai-llm-server`` ``config.ini`` SLAVE_DB (prod-aligned; no Hermes import)."""
    params = {
        "dbname": "fasandbox",
        "user": "fasandbox",
        "password": "f1n@nc3@1sb",
        "host": "127.0.0.1",
        "port": 25432,
    }
    config_path = _LLM_SERVER_ROOT / "config.ini"
    if config_path.is_file():
        cp = configparser.ConfigParser()
        cp.read(config_path)
        if "SLAVE_DB" in cp:
            section = cp["SLAVE_DB"]
            params["dbname"] = section.get("name", params["dbname"])
            params["user"] = section.get("user", params["user"])
            params["password"] = section.get("password", params["password"])
            params["host"] = section.get("host", params["host"])
            params["port"] = section.getint("port", params["port"])

    if os.environ.get("SLAVE_DB_HOST"):
        params["host"] = os.environ["SLAVE_DB_HOST"]
    if os.environ.get("SLAVE_DB_PORT"):
        params["port"] = int(os.environ["SLAVE_DB_PORT"])
    if os.environ.get("SLAVE_DB_NAME"):
        params["dbname"] = os.environ["SLAVE_DB_NAME"]
    if os.environ.get("SLAVE_DB_USER"):
        params["user"] = os.environ["SLAVE_DB_USER"]
    if os.environ.get("SLAVE_DB_PASSWORD"):
        params["password"] = os.environ["SLAVE_DB_PASSWORD"]
    return params


def _coerce_json_maybe(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return None
    return None


def _json_default(value: Any) -> Any:
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _scenario_title_from_id(scenario_id: str) -> str:
    return str(scenario_id or "").replace("_", " ").strip().title()


_RE_SIMULATE_RECOMMENDED_PLAN = re.compile(
    r"## Recommended plan:\s*([a-z][a-z0-9_]*)",
    re.IGNORECASE,
)
_RE_SIMULATE_ALTERNATIVE_PLAN = re.compile(
    r"## Alternative plan:\s*([a-z][a-z0-9_]*)",
    re.IGNORECASE,
)


def _scenario_ids_from_simulate_strategy(simulate_md: str) -> tuple[str | None, str | None]:
    rec = _RE_SIMULATE_RECOMMENDED_PLAN.search(simulate_md or "")
    alt = _RE_SIMULATE_ALTERNATIVE_PLAN.search(simulate_md or "")
    rec_id = rec.group(1).strip() if rec else None
    alt_id = alt.group(1).strip() if alt else None
    return rec_id, alt_id


def _plan_description_from_simulate(simulate_md: str, scenario_id: str, *, is_active: bool) -> str:
    header = "Recommended plan" if is_active else "Alternative plan"
    pattern = (
        rf"## {header}:\s*{re.escape(scenario_id)}\s*\n"
        rf"(?:\*+\s*)?(.+?)(?=\n## |\Z)"
    )
    match = re.search(pattern, simulate_md or "", re.IGNORECASE | re.DOTALL)
    if not match:
        return ""
    line = (match.group(1) or "").strip().split("\n")[0]
    return re.sub(r"^\*+\s*", "", line).strip()


def _align_goal_plan_to_simulate_outcome(goal_plan: Any, simulate_md: str) -> list[dict[str, Any]]:
    """Ensure ``GOAL_PLAN`` covers recommended + alternative ids from the simulate outcome."""
    rec_id, alt_id = _scenario_ids_from_simulate_strategy(simulate_md)
    if not rec_id and not alt_id:
        return _normalize_goal_plan_list(goal_plan)

    by_id: dict[str, dict[str, Any]] = {}
    for entry in _normalize_goal_plan_list(goal_plan):
        sid = str(entry.get("scenario_id") or "").strip()
        if sid:
            by_id[sid] = entry

    aligned: list[dict[str, Any]] = []
    for sid, is_active in ((rec_id, True), (alt_id, False)):
        if not sid:
            continue
        if sid in by_id:
            entry = dict(by_id[sid])
            entry["is_active"] = is_active
            if not str(entry.get("scenario_title") or "").strip():
                entry["scenario_title"] = _scenario_title_from_id(sid)
        else:
            entry = {
                "scenario_id": sid,
                "scenario_title": _scenario_title_from_id(sid),
                "scenario_description": _plan_description_from_simulate(
                    simulate_md,
                    sid,
                    is_active=is_active,
                ),
                "is_active": is_active,
            }
        aligned.append(entry)

    return aligned or _normalize_goal_plan_list(goal_plan)


def _coerce_tool_call_arguments(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return {}
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _simulate_tool_args_by_scenario_id(calls: Any) -> dict[str, dict[str, Any]]:
    """Map ``scenario_id`` → simulate tool args from persisted outcome ``calls``."""
    by_sid: dict[str, dict[str, Any]] = {}
    payload = _coerce_json_maybe(calls)
    if not isinstance(payload, dict):
        return by_sid

    rounds = payload.get("llm_calls")
    if not isinstance(rounds, list):
        return by_sid

    for round_row in rounds:
        if not isinstance(round_row, dict):
            continue
        tool_calls = round_row.get("tool_calls")
        if not isinstance(tool_calls, list):
            continue
        for tc in tool_calls:
            if not isinstance(tc, dict):
                continue
            name = str(tc.get("tool_name") or tc.get("name") or "").strip()
            if name != "simulate_financial_strategy":
                continue
            args = _coerce_tool_call_arguments(tc.get("arguments") or tc.get("args"))
            batch = args.get("scenarios")
            if isinstance(batch, list):
                for entry in batch:
                    if not isinstance(entry, dict):
                        continue
                    sid = str(entry.get("scenario_id") or "").strip()
                    if sid:
                        by_sid[sid] = dict(entry)
                continue
            sid = str(args.get("scenario_id") or "").strip()
            if sid:
                by_sid[sid] = dict(args)
    return by_sid


def _attach_spending_schedules_to_goal_plan(
    goal_plan: list[dict[str, Any]],
    tool_args_by_id: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Ensure each scenario includes ``spending_schedule`` when available from simulate tool calls."""
    if not goal_plan:
        return goal_plan

    enriched: list[dict[str, Any]] = []
    for entry in goal_plan:
        row = dict(entry)
        sid = str(row.get("scenario_id") or "").strip()
        if sid and not row.get("spending_schedule"):
            args = tool_args_by_id.get(sid) or {}
            schedule = args.get("spending_schedule")
            if isinstance(schedule, list) and schedule:
                row["spending_schedule"] = schedule
        enriched.append(row)
    return enriched


def _finalize_goal_plan_for_bundle(
    goal_plan: Any,
    simulate_md: str,
    *,
    simulate_calls: Any = None,
) -> list[dict[str, Any]]:
    aligned = _align_goal_plan_to_simulate_outcome(goal_plan, simulate_md)
    tool_args_by_id = _simulate_tool_args_by_scenario_id(simulate_calls)
    return _attach_spending_schedules_to_goal_plan(aligned, tool_args_by_id)


def _expected_plan_count_from_bundle(bundle_md: str) -> int:
    marker = "<GOAL_PLAN>"
    if marker not in bundle_md:
        return 0
    block = bundle_md.split(marker, 1)[1]
    fence = re.search(r"```json\s*(\[.*?\])\s*```", block, re.DOTALL)
    if not fence:
        return 0
    try:
        parsed = json.loads(fence.group(1))
    except json.JSONDecodeError:
        return 0
    return len(parsed) if isinstance(parsed, list) else 0


def _normalize_goal_plan_list(goal_plan: Any) -> list[dict[str, Any]]:
    parsed = _coerce_json_maybe(goal_plan)
    if parsed is not None:
        goal_plan = parsed
    if isinstance(goal_plan, list):
        return [entry for entry in goal_plan if isinstance(entry, dict)]
    if isinstance(goal_plan, dict) and goal_plan:
        return [goal_plan]
    return []


def _goal_plan_active_scenario(goal_plan: Any) -> dict[str, Any] | None:
    """Return the recommended (``is_active: true``) scenario from ``users.goal_plan``."""
    for entry in _normalize_goal_plan_list(goal_plan):
        if entry.get("is_active"):
            return entry
    entries = _normalize_goal_plan_list(goal_plan)
    return entries[0] if entries else None


def _goal_plan_entries_for_bundle(goal_plan: Any) -> list[dict[str, Any]]:
    """Strip fields not sent to the model in ``<GOAL_PLAN>``."""
    rows: list[dict[str, Any]] = []
    for entry in _normalize_goal_plan_list(goal_plan):
        row = dict(entry)
        row.pop("scenario_description", None)
        rows.append(row)
    return rows


def _format_goal_plan(goal_plan: Any) -> str:
    entries = _goal_plan_entries_for_bundle(goal_plan)
    if not entries:
        return ""
    payload = json.dumps(entries, indent=2, default=_json_default)
    return f"<GOAL_PLAN>\n\n```json\n{payload}\n```\n\n</GOAL_PLAN>"


def trim_simulate_outcome_for_bundle(simulate_outcome_md: str) -> str:
    """Drop simulate outcome content before ``# Financial Needs`` for the LLM bundle."""
    body = (simulate_outcome_md or "").strip()
    idx = body.find(_FINANCIAL_NEEDS_H1)
    if idx < 0:
        raise ValueError(f"simulate outcome must include {_FINANCIAL_NEEDS_H1}")
    return body[idx:].strip() + "\n"


def build_need_plans_verbalizer_input_bundle(
    *,
    simulate_outcome_md: str,
    goal_plan: Any = None,
) -> str:
    """Hermes-aligned bundle for **P:NeedPlansVerbalizer** prompt tuning."""
    simulate = trim_simulate_outcome_for_bundle(simulate_outcome_md)

    parts = [
        f"<SIMULATE_FINANCIAL_STRATEGY_OUTCOME>\n\n{simulate}\n\n</SIMULATE_FINANCIAL_STRATEGY_OUTCOME>",
    ]

    plan_block = _format_goal_plan(goal_plan)
    if plan_block:
        parts.append(plan_block)

    return "\n\n".join(parts) + "\n"


def _fetch_ai_agent_outcome_row(agent_outcome_id: int) -> dict[str, Any] | None:
    if psycopg2 is None or RealDictCursor is None:
        raise RuntimeError(
            "Missing dependency `psycopg2`. For DB-backed runs use "
            "`finance-ai-llm-server/llm` venv (has psycopg2-binary) or "
            "`pip install psycopg2-binary` in finance-ai-penny/.venv."
        )

    conn = psycopg2.connect(**_load_slave_db_connect_kwargs())
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT agent_outcome_id, user_id, type, agent_outcome, type_metadata, calls
                FROM ai_agent_outcomes
                WHERE agent_outcome_id = %s
                LIMIT 1
                """,
                (int(agent_outcome_id),),
            )
            row = cur.fetchone()
    finally:
        conn.close()
    return dict(row) if row else None


def _fetch_user_goal_plan(user_id: int) -> Any:
    if psycopg2 is None:
        raise RuntimeError("Missing dependency `psycopg2`.")

    conn = psycopg2.connect(**_load_slave_db_connect_kwargs())
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT goal_plan
                FROM users
                WHERE user_id = %s
                LIMIT 1
                """,
                (int(user_id),),
            )
            row = cur.fetchone()
    finally:
        conn.close()

    if not row:
        return None
    raw = row.get("goal_plan")
    parsed = _coerce_json_maybe(raw)
    if parsed in (None, {}, []):
        return None
    return parsed


def load_simulate_agent_outcome_markdown(simulate_agent_outcome_id: int) -> tuple[int, str]:
    """Load persisted ``simulate_financial_strategy`` outcome markdown."""
    sim_id = int(simulate_agent_outcome_id)
    row = _fetch_ai_agent_outcome_row(sim_id)

    if not row:
        raise ValueError(f"simulate_agent_outcome_id not found: {sim_id}")
    row_type = row.get("type")
    if row_type != _SIMULATE_OUTCOME_TYPE:
        raise ValueError(
            f"simulate_agent_outcome_id={sim_id} has type {row_type!r}; expected {_SIMULATE_OUTCOME_TYPE!r}"
        )
    md = (row.get("agent_outcome") or "").strip()
    if not md:
        raise ValueError(f"simulate_agent_outcome_id={sim_id} has empty agent_outcome")
    try:
        uid = int(row.get("user_id"))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"simulate_agent_outcome_id={sim_id} has invalid user_id") from exc
    return uid, md + "\n"


def build_need_plans_verbalizer_input(*, simulate_agent_outcome_id: int) -> str:
    """Build bundled input from DB outcomes + ``users.goal_plan``."""
    sim_uid, simulate_md = load_simulate_agent_outcome_markdown(simulate_agent_outcome_id)
    sim_row = _fetch_ai_agent_outcome_row(int(simulate_agent_outcome_id))
    if not sim_row:
        raise ValueError(f"simulate_agent_outcome_id not found: {simulate_agent_outcome_id}")

    goal_plan = _fetch_user_goal_plan(sim_uid)
    if goal_plan is None:
        raise ValueError(
            f"users.goal_plan is empty for user_id={sim_uid}; "
            "run simulate_financial_strategy with persistence first"
        )
    goal_plan = _finalize_goal_plan_for_bundle(
        goal_plan,
        simulate_md,
        simulate_calls=sim_row.get("calls"),
    )

    return build_need_plans_verbalizer_input_bundle(
        simulate_outcome_md=simulate_md,
        goal_plan=goal_plan,
    )


def _bundled_input_from_test_case(test_case: dict[str, Any]) -> str | None:
    """Return pre-built bundled LLM input when ``input`` / ``bundled_input`` is already tagged."""
    for key in ("input", "bundled_input"):
        raw = test_case.get(key)
        if isinstance(raw, str) and raw.strip() and "<SIMULATE_FINANCIAL_STRATEGY_OUTCOME>" in raw:
            text = raw.strip()
            return text + ("\n" if not text.endswith("\n") else "")
    return None


def resolve_test_case_input(test_case: dict[str, Any]) -> str:
    """Return bundled optimizer input from a test-case dict."""
    bundled = _bundled_input_from_test_case(test_case)
    if bundled:
        return bundled
    raise ValueError("test case must include bundled input")


def format_need_plans_verbalizer_user_message(profile_input: str) -> str:
    body = (profile_input or "").strip()
    if not body:
        raise ValueError("profile_input must be non-empty markdown.")
    return body + "\n"


class NeedPlansVerbalizerOptimizer:
    """Gemini runner for the need-plans verbalizer system prompt."""

    def __init__(
        self,
        model_name: str = GEMINI_FLASH_LITE,
        *,
        thinking_budget: int = NEED_PLANS_VERBALIZER_THINKING_BUDGET,
        max_output_tokens: int = NEED_PLANS_VERBALIZER_MAX_OUTPUT_TOKENS,
    ):
        if genai is None or types is None:  # pragma: no cover
            raise RuntimeError("Install `google-genai` (and optionally `python-dotenv`) for this optimizer.")
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY is not set.")
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name
        self.thinking_budget = thinking_budget
        self.max_output_tokens = max_output_tokens
        self.temperature = 0.35
        self.top_p = 0.95
        self.top_k = 40
        self.safety_settings = [
            types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"),
            types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"),
            types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"),
            types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="OFF"),
        ]
        self.system_prompt = SYSTEM_PROMPT
        self.output_schema = _build_output_schema()

    def generate_response(self, profile_input: str) -> dict[str, Any]:
        user_text = format_need_plans_verbalizer_user_message(profile_input)
        request_text = types.Part.from_text(text=user_text)
        contents = [types.Content(role="user", parts=[request_text])]
        cfg = types.GenerateContentConfig(
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            max_output_tokens=self.max_output_tokens,
            safety_settings=self.safety_settings,
            system_instruction=[types.Part.from_text(text=self.system_prompt)],
            thinking_config=types.ThinkingConfig(
                thinking_budget=self.thinking_budget,
                include_thoughts=True,
            ),
            response_mime_type="application/json",
            response_schema=self.output_schema,
        )

        output_text = ""
        thought_summary = ""
        finish_reason = None
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=contents,
                config=cfg,
            )
            for cand in getattr(response, "candidates", None) or []:
                reason = getattr(cand, "finish_reason", None)
                if reason is not None:
                    finish_reason = reason
                content = getattr(cand, "content", None)
                if not content:
                    continue
                for part in getattr(content, "parts", None) or []:
                    t = getattr(part, "text", None)
                    if not isinstance(t, str) or not t:
                        continue
                    if getattr(part, "thought", False):
                        thought_summary = (thought_summary + t) if thought_summary else t
                    else:
                        output_text += t
            if not output_text:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message=r".*non-text parts in the response.*")
                    agg = getattr(response, "text", None)
                    if isinstance(agg, str) and agg:
                        output_text = agg
        except ClientError as e:
            if self.thinking_budget == 0 and "only works in thinking mode" in (str(e) or ""):
                print(
                    "\n[NOTE] This model requires thinking mode; use default (no --no-thinking) or a different model.",
                    flush=True,
                )
                sys.exit(1)
            raise

        if thought_summary:
            print("\n" + "-" * 80)
            print("THOUGHT SUMMARY:")
            print("-" * 80)
            print(thought_summary.strip())
            print("-" * 80 + "\n")

        if not (output_text or "").strip():
            raise ValueError("Empty response from model. Check API key and model availability.")
        try:
            parsed = json.loads(output_text)
        except json.JSONDecodeError as exc:
            reason = str(finish_reason or "unknown")
            raise ValueError(f"Invalid JSON response. finish_reason={reason!r}") from exc

        expected_plans = _expected_plan_count_from_bundle(user_text)
        return _validate_need_plans_response(parsed, expected_plan_count=expected_plans or None)


TEST_CASES: list[dict[str, Any]] = [
    {
        "name": "debt_paydown_interest_drag",
        "batch": 1,
        "input": """
<SIMULATE_FINANCIAL_STRATEGY_OUTCOME>

# Financial Needs

## Primary needs
1. **Reduce interest drag**: Venture balance **$8,400** with **$312** interest paid over 90 days while spending tracks near income.

## Evidence
* **Reduce interest drag**
  - Interest tool: **$312** on Venture in 90 days.
  - Next due **2026-04-18** per payment schedule.

## Credit Interest Rates
* **Venture (11)**: **~24.9%** (from recent interest charges vs average daily balance).

# Financial Strategy

## Recommended plan: gradual_paydown_savings
* Phased dining and leisure trims keep month-1 cuts modest, then deepen after month 3 while routing **$200**/mo to savings once the card hits **$0**.

## Alternative plan: steady_cut
* Flat **$700** meals and **$350** leisure from month 1 hits **$0** debt about two months sooner but leaves thinner checking buffers in the first quarter.


</SIMULATE_FINANCIAL_STRATEGY_OUTCOME>

<GOAL_PLAN>

```json
[
  {
    "scenario_id": "gradual_paydown_savings",
    "scenario_title": "Gradual Paydown Savings",
    "is_active": true,
    "current_spending": {
      "meals": 974,
      "leisure": 520
    },
    "spending_schedule": [
      {
        "start_end_month": "1-3",
        "categories": {
          "meals": 850,
          "leisure": 450
        }
      },
      {
        "start_end_month": "4+",
        "categories": {
          "meals": 700,
          "leisure": 350
        }
      }
    ],
    "spending_timeline": [
      {
        "categories": {
          "meals": 850,
          "leisure": 450
        },
        "start_month": "04/26",
        "end_month": "06/26"
      },
      {
        "categories": {
          "meals": 700,
          "leisure": 350
        },
        "start_month": "07/26",
        "end_month": "03/28"
      }
    ],
    "credit_balance_target": 0,
    "savings_per_month": 200,
    "savings_targets": [
      6500
    ]
  },
  {
    "scenario_id": "steady_cut",
    "scenario_title": "Steady Cut",
    "is_active": false,
    "current_spending": {
      "meals": 974,
      "leisure": 520
    },
    "spending_schedule": [
      {
        "start_end_month": "1+",
        "categories": {
          "meals": 700,
          "leisure": 350
        }
      }
    ],
    "spending_timeline": [
      {
        "categories": {
          "meals": 700,
          "leisure": 350
        },
        "start_month": "04/26",
        "end_month": "03/28"
      }
    ],
    "credit_balance_target": 0
  }
]
```

</GOAL_PLAN>
""",
        "ideal_response": {
            "financial_need": "Venture interest is costing you $312 every 90 days on an $8,400 balance.",
            "solution_options": [
                {
                    "scenario_id": "gradual_paydown_savings",
                    "scenario_title": "Gradual Paydown Savings",
                    "summary": "Phase meal and leisure cuts, then steer $200/mo to savings once debt hits $0."
                },
                {
                    "scenario_id": "steady_cut",
                    "scenario_title": "Steady Cut",
                    "summary": "Hold meals at $700 and leisure at $350 from month 1 to clear debt faster."
                }
            ]
        },
    },
    {
        "name": "cash_flow_crunch_before_mortgage",
        "batch": 1,
        "input": """
<SIMULATE_FINANCIAL_STRATEGY_OUTCOME>

# Financial Needs

## Primary needs
1. **Stabilize cash flow**: Checking **$800** with **$2,100** mortgage due **2026-04-01** — liquidity risk before flexible spend cuts matter.

## Evidence
* **Stabilize cash flow**
  - Checking **$800** vs mortgage **$2,100** on the 1st.
  - Forecast committed outflows **$3,600**/mo vs income **$4,000**/mo.

## Credit Interest Rates
* **Rewards (21)**: **~22.4%** (derived from last-cycle interest charge).

# Financial Strategy

## Recommended plan: protect_fixed_cut_flex
* Hold checking above **$2,200** before the mortgage, trim **$200**/mo from meals and shopping months 1–3, then reassess.

## Alternative plan: aggressive_flex_cut
* Cut meals to **$450** and shopping to **$150** immediately — debt-free by **Aug 2026** but checking may dip below **$500** in April.


</SIMULATE_FINANCIAL_STRATEGY_OUTCOME>

<GOAL_PLAN>

```json
[
  {
    "scenario_id": "protect_fixed_cut_flex",
    "scenario_title": "Protect Fixed Cut Flex",
    "is_active": true,
    "current_spending": {
      "meals": 620,
      "shopping": 280
    },
    "spending_schedule": [
      {
        "start_end_month": "1-3",
        "categories": {
          "meals": 520,
          "shopping": 180
        }
      },
      {
        "start_end_month": "4+",
        "categories": {
          "meals": 520,
          "shopping": 180
        }
      }
    ],
    "spending_timeline": [
      {
        "categories": {
          "meals": 520,
          "shopping": 180
        },
        "start_month": "04/26",
        "end_month": "06/26"
      },
      {
        "categories": {
          "meals": 520,
          "shopping": 180
        },
        "start_month": "07/26",
        "end_month": "03/28"
      }
    ],
    "account_balance_targets": [
      {
        "account_id": 20,
        "balance_target": 2200
      }
    ]
  },
  {
    "scenario_id": "aggressive_flex_cut",
    "scenario_title": "Aggressive Flex Cut",
    "is_active": false,
    "current_spending": {
      "meals": 620,
      "shopping": 280
    },
    "spending_schedule": [
      {
        "start_end_month": "1+",
        "categories": {
          "meals": 450,
          "shopping": 150
        }
      }
    ],
    "spending_timeline": [
      {
        "categories": {
          "meals": 450,
          "shopping": 150
        },
        "start_month": "04/26",
        "end_month": "03/28"
      }
    ],
    "credit_balance_target": 0
  }
]
```

</GOAL_PLAN>
""",
        "ideal_response": {
            "financial_need": "Checking at $800 cannot cover the $2,100 mortgage due April 1.",
            "solution_options": [
                {
                    "scenario_id": "protect_fixed_cut_flex",
                    "scenario_title": "Protect Fixed Cut Flex",
                    "summary": "Keep checking above $2,200 and trim meals and shopping $200/mo for three months."
                },
                {
                    "scenario_id": "aggressive_flex_cut",
                    "scenario_title": "Aggressive Flex Cut",
                    "summary": "Cut meals to $450 and shopping to $150 now to be debt-free by Aug 2026."
                }
            ]
        },
    },
    {
        "name": "slow_debt_creep",
        "batch": 2,
        "input": """
<SIMULATE_FINANCIAL_STRATEGY_OUTCOME>

# Financial Needs

## Primary needs
1. **Settle debt**: Platinum **$4,800** with slow paydown at minimum-style payments.

## Evidence
* **Settle debt**
  - Balance up **$300** over three months despite **$115**/mo payments.
  - APR tool: **~21.8%** on Platinum.

## Credit Interest Rates
* **Platinum (31)**: **~21.8%**

# Financial Strategy

## Recommended plan: balanced_trim
* Trim meals to **$520** and leisure to **$300** from month 1; **$0** debt by **Dec 2026**, saves about **$420** interest vs status quo.

## Alternative plan: leisure_first
* Protect leisure at **$380** but cut meals harder to **$450** — similar debt-free date with more dining sacrifice and less social spend risk.


</SIMULATE_FINANCIAL_STRATEGY_OUTCOME>

<GOAL_PLAN>

```json
[
  {
    "scenario_id": "balanced_trim",
    "scenario_title": "Balanced Trim",
    "is_active": true,
    "current_spending": {
      "meals": 640,
      "leisure": 410
    },
    "spending_schedule": [
      {
        "start_end_month": "1+",
        "categories": {
          "meals": 520,
          "leisure": 300
        }
      }
    ],
    "spending_timeline": [
      {
        "categories": {
          "meals": 520,
          "leisure": 300
        },
        "start_month": "04/26",
        "end_month": "03/28"
      }
    ],
    "credit_balance_target": 0
  },
  {
    "scenario_id": "leisure_first",
    "scenario_title": "Leisure First",
    "is_active": false,
    "current_spending": {
      "meals": 640,
      "leisure": 410
    },
    "spending_schedule": [
      {
        "start_end_month": "1+",
        "categories": {
          "meals": 450,
          "leisure": 380
        }
      }
    ],
    "spending_timeline": [
      {
        "categories": {
          "meals": 450,
          "leisure": 380
        },
        "start_month": "04/26",
        "end_month": "03/28"
      }
    ],
    "credit_balance_target": 0
  }
]
```

</GOAL_PLAN>
""",
        "ideal_response": {
            "financial_need": "Platinum debt at $4,800 is creeping up despite $115/mo payments.",
            "solution_options": [
                {
                    "scenario_id": "balanced_trim",
                    "scenario_title": "Balanced Trim",
                    "summary": "Trim meals to $520 and leisure to $300 from month 1; debt-free by Dec 2026."
                },
                {
                    "scenario_id": "leisure_first",
                    "scenario_title": "Leisure First",
                    "summary": "Protect leisure at $380 while cutting meals to $450 for a similar payoff date."
                }
            ]
        },
    },
]




def _run_test(
    profile_input: str,
    optimizer: NeedPlansVerbalizerOptimizer | None = None,
    *,
    ideal: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if optimizer is None:
        optimizer = NeedPlansVerbalizerOptimizer()
    wrapped = format_need_plans_verbalizer_user_message(profile_input)
    print("=" * 80)
    print("LLM INPUT:")
    print("=" * 80)
    print(wrapped)
    result = optimizer.generate_response(profile_input)
    print("=" * 80)
    print("LLM OUTPUT:")
    print("=" * 80)
    print(json.dumps(result, indent=2))
    if ideal is not None:
        print("=" * 80)
        print("IDEAL RESPONSE:")
        print("=" * 80)
        print(json.dumps(ideal, indent=2))
    print("=" * 80 + "\n")
    return result


def get_test_case(test_name_or_index: str | int) -> dict[str, Any] | None:
    if isinstance(test_name_or_index, int):
        if 0 <= test_name_or_index < len(TEST_CASES):
            return TEST_CASES[test_name_or_index]
        return None
    for tc in TEST_CASES:
        if tc["name"] == test_name_or_index:
            return tc
    return None


def run_test(
    test_name_or_index_or_dict: str | int | dict[str, Any],
    optimizer: NeedPlansVerbalizerOptimizer | None = None,
) -> dict[str, Any] | None:
    if optimizer is None:
        optimizer = NeedPlansVerbalizerOptimizer()

    if isinstance(test_name_or_index_or_dict, dict):
        tc = test_name_or_index_or_dict
        name = tc.get("name", "custom_test")
        try:
            payload = resolve_test_case_input(tc)
        except ValueError as exc:
            print(f"Invalid test dict: {exc}")
            return None
        print(f"\n{'=' * 80}\nRunning test: {name}\n{'=' * 80}\n")
        ideal = _resolve_ideal_response(tc)
        return _run_test(payload, optimizer, ideal=ideal)

    tc = get_test_case(test_name_or_index_or_dict)
    if tc is None:
        print(f"Test case '{test_name_or_index_or_dict}' not found.")
        return None
    print(f"\n{'=' * 80}\nRunning test: {tc['name']}\n{'=' * 80}\n")
    ideal = _resolve_ideal_response(tc)
    return _run_test(resolve_test_case_input(tc), optimizer, ideal=ideal)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run P:NeedPlansVerbalizer optimizer tests")
    parser.add_argument("--test", type=str, help='Test name or index (e.g. "0" or "debt_paydown_interest_drag")')
    parser.add_argument("--batch", type=int, help="Run all tests in batch N")
    parser.add_argument(
        "--simulate-agent-outcome-id",
        type=int,
        help="simulate_financial_strategy ai_agent_outcomes.agent_outcome_id",
    )
    parser.add_argument(
        "--print-input-only",
        action="store_true",
        help="Only print built markdown input (no model call)",
    )
    parser.add_argument("--model", type=str, default=GEMINI_FLASH_LITE)
    parser.add_argument("--no-thinking", action="store_true", help="Set thinking_budget=0")
    args = parser.parse_args()

    if args.simulate_agent_outcome_id is not None:
        built = build_need_plans_verbalizer_input(simulate_agent_outcome_id=args.simulate_agent_outcome_id)
        print("BUILT NEED PLANS VERBALIZER INPUT")
        print("-" * 80)
        print(built)
        if args.print_input_only:
            return
        thinking_budget = 0 if args.no_thinking else NEED_PLANS_VERBALIZER_THINKING_BUDGET
        optimizer = NeedPlansVerbalizerOptimizer(model_name=args.model, thinking_budget=thinking_budget)
        print("\nNEED PLANS VERBALIZER LLM OUTPUT")
        print("-" * 80)
        print(json.dumps(optimizer.generate_response(built), indent=2))
        return

    if args.print_input_only:
        print("Error: --print-input-only requires --simulate-agent-outcome-id", file=sys.stderr)
        raise SystemExit(1)

    if args.batch is None and args.test is None:
        _print_usage()
        return

    thinking_budget = 0 if args.no_thinking else NEED_PLANS_VERBALIZER_THINKING_BUDGET
    optimizer = NeedPlansVerbalizerOptimizer(model_name=args.model, thinking_budget=thinking_budget)

    if args.batch is not None:
        cases = [tc for tc in TEST_CASES if int(tc.get("batch") or 0) == int(args.batch)]
        if not cases:
            raise SystemExit(f"No tests found for batch={args.batch}")
        for i, tc in enumerate(cases):
            if i:
                print("\n" + "-" * 80 + "\n")
            run_test(tc, optimizer)
        return

    if args.test is not None:
        if args.test.strip().lower() == "all":
            for i in range(len(TEST_CASES)):
                run_test(i, optimizer)
                if i < len(TEST_CASES) - 1:
                    print("\n" + "-" * 80 + "\n")
            return
        test_val: str | int = int(args.test) if args.test.isdigit() else args.test
        run_test(test_val, optimizer)
        return


def _print_usage() -> None:
    print("Usage:")
    print("  Run a single test: --test <name_or_index>")
    print("  Run all tests: --test all")
    print("  Run batch: --batch <N>")
    print("  Build input from DB: --simulate-agent-outcome-id <id>")
    print("  Print built input only: --simulate-agent-outcome-id <id> --print-input-only")
    print("\nAvailable test cases:")
    for i, tc in enumerate(TEST_CASES):
        batch = tc.get("batch", "?")
        print(f"  {i}: {tc['name']} (batch {batch})")


if __name__ == "__main__":
    main()

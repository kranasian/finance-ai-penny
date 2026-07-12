"""
Optimizer runner for **P:DetermineUI** (Gemini prompt tuning).

Input is the raw (non-verbalized) Penny assistant response from ``/real_penny_chat``
(before verbalization). Output is structured JSON picking complementary in-app UI:
cashflow, account, transaction, chart, or none.

  python3 active_experiments/determine_ui_optimizer.py --test 0
  python3 active_experiments/determine_ui_optimizer.py --test all
  python3 active_experiments/determine_ui_optimizer.py --raw-response "Groceries in category 4 are up."
  python3 active_experiments/determine_ui_optimizer.py --response-file scratch/raw_reply.txt
  python3 active_experiments/determine_ui_optimizer.py --raw-response "..." --stream
  python3 active_experiments/determine_ui_optimizer.py --raw-response "..." --print-input-only

Requires ``GEMINI_API_KEY``. Keep prompts in sync with ``finance-ai-hermes/agent_logic/determine_ui.py``; promote to ``penny_templates`` as **H:DetermineUI** (system) and **H:DetermineUITask** (task).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import warnings
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

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

from active_experiments.need_verbalizer_optimizer import _parse_model_json_object
from active_experiments.verbalizer_optimizer_db import _resolve_ideal_response

if load_dotenv is not None:
    load_dotenv()

GEMINI_FLASH_LITE = "gemini-flash-lite-latest"
DETERMINE_UI_THINKING_BUDGET = 256
DETERMINE_UI_MAX_OUTPUT_TOKENS = 1024

_UI_TYPES = frozenset({"cashflow", "account", "transaction", "chart", "none"})
_PERIOD_GRANULARITIES = frozenset({"year", "month", "week"})
_YEAR_RE = re.compile(r"^\d{4}$")
_MONTH_RE = re.compile(r"^\d{4}-\d{2}$")
_WEEK_RE = re.compile(r"^\d{4}-W\d{2}$")
_NULL_STRINGS = frozenset({"null", "none", ""})

SYSTEM_PROMPT = """You are a UI routing assistant for **Penny**, a personal finance app.

## Tools (read-only)

- ``describe_categories`` — official category taxonomy (canonical slugs).
- ``resolve_category_id`` — map a category label or slug to ``category_id``.
- ``retrieve_user_accounts_by_sql`` — bounded SELECT on ``user_accounts``.
- ``retrieve_user_spending_transactions_by_sql`` — bounded SELECT on ``user_spending_transactions``.
"""

TASK_PROMPT = """Task: Read the **raw assistant response** below (non-verbalized Penny assistant output) and pick the single best complementary in-app UI screen (`ui_type` with `params`) that lets the user verify or explore what Penny just explained.

## Available UI types

1. **cashflow** — category spending over time. Optionally scoped to a year, month, or week.
2. **account** — balances and activity for one linked bank, card, or loan account.
3. **transaction** — one specific charge, deposit, or merchant transaction.
4. **chart** — a custom time-series view (e.g. merchant spend by month).
5. **none** — no complementary drill-down is helpful.

## params by ui_type

- **cashflow** — `category_id` (required); optional `period_granularity` + `period`
- **account** — `account_id`
- **transaction** — `transaction_id`
- **chart** — `x_axis` + `y_axis` (required); optional `series_label`
  - Example monthly Netflix trend: `{"x_axis": "month", "y_axis": "netflix_spending", "series_label": "Netflix"}`
- **none** — `params` must be `null`

Routing rules:
- Do not invent numeric ids; prefer ids already present in the response text.
- If ``account_id``, ``transaction_id``, or ``category_id`` already appears in the raw response, use it directly — **do not call tools**.
- When the response focuses on one account balance or activity (even if rent or a category is mentioned), choose **account** using that ``account_id``.
- When unsure, choose ``none``.
- Only when the response clearly targets one category, account, or transaction but **omits** the numeric id, call **at most one** read-only lookup tool, then answer immediately.
- Do not call tools for general advice, multi-topic summaries, or when the response explicitly says no drill-down id is available.
- Match UI type to focus: category spend → cashflow; one account → account; one charge → transaction; merchant/tag trend → chart; otherwise → none.

Output: **JSON only** — object with `rationale`, `ui_type`, and `params`. No markdown or prose."""


def _build_params_schema() -> "types.Schema":
    return types.Schema(
        type=types.Type.OBJECT,
        description=(
            "Fields for the chosen ui_type only; null when ui_type is none. "
            "cashflow: category_id (required); optionally period_granularity (year, month, week) "
            "and period (YYYY, YYYY-MM, or YYYY-Www matching granularity). "
            "account: account_id. transaction: transaction_id. "
            "chart: x_axis and y_axis (required); series_label (optional)."
        ),
        properties={
            "category_id": types.Schema(type=types.Type.INTEGER, description="Penny category id."),
            "period_granularity": types.Schema(
                type=types.Type.STRING,
                description="Cashflow time bucket: year, month, or week.",
            ),
            "period": types.Schema(
                type=types.Type.STRING,
                description="Period value matching period_granularity: YYYY, YYYY-MM, or YYYY-Www.",
            ),
            "account_id": types.Schema(type=types.Type.INTEGER, description="Penny account id."),
            "transaction_id": types.Schema(type=types.Type.INTEGER, description="Penny transaction id."),
            "x_axis": types.Schema(
                type=types.Type.STRING,
                description="Chart time bucket: month, week, or year.",
            ),
            "y_axis": types.Schema(
                type=types.Type.STRING,
                description="Chart spending metric slug, e.g. netflix_spending.",
            ),
            "series_label": types.Schema(
                type=types.Type.STRING,
                description="Optional chart series label, e.g. Netflix.",
            ),
        },
    )


def _build_output_schema() -> "types.Schema":
    if types is None:  # pragma: no cover
        raise RuntimeError("Install `google-genai` for this optimizer.")
    return types.Schema(
        type=types.Type.OBJECT,
        required=["rationale", "ui_type", "params"],
        properties={
            "rationale": types.Schema(
                type=types.Type.STRING,
                description=(
                    "Write first. One sentence on what complementary UI would help the user "
                    "verify or explore the response, or why none."
                ),
            ),
            "ui_type": types.Schema(
                type=types.Type.STRING,
                description="Exactly one of: cashflow, account, transaction, chart, none.",
            ),
            "params": _build_params_schema(),
        },
    )


def _require_non_empty_str(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


def _require_meaningful_str(value: Any, field: str) -> str:
    stripped = _require_non_empty_str(value, field)
    if stripped.lower() in _NULL_STRINGS:
        raise ValueError(f"{field} must be a non-empty string")
    return stripped


def _optional_positive_int(value: Any, field: str) -> int | None:
    if value is None:
        return None
    try:
        n = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be an integer") from exc
    if n <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return n


def _validate_period_for_granularity(period: str, granularity: str) -> str:
    if granularity == "year":
        if not _YEAR_RE.match(period):
            raise ValueError("params.period must be YYYY when period_granularity is year")
    elif granularity == "month":
        if not _MONTH_RE.match(period):
            raise ValueError("params.period must be YYYY-MM when period_granularity is month")
    elif not _WEEK_RE.match(period):
        raise ValueError("params.period must be YYYY-Www when period_granularity is week")
    return period


def _resolve_params_raw(parsed: dict[str, Any], ui_type: str) -> Any:
    if "params" in parsed:
        return parsed.get("params")
    if ui_type == "none":
        return None
    return parsed.get(ui_type)


def _validate_cashflow_params(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("params must be a JSON object when ui_type is cashflow")
    category_id = _optional_positive_int(value.get("category_id"), "params.category_id")
    if category_id is None:
        raise ValueError("params.category_id is required when ui_type is cashflow")
    params: dict[str, Any] = {"category_id": category_id}

    period_granularity_raw = value.get("period_granularity")
    period_raw = value.get("period")
    month_legacy = value.get("month")
    if month_legacy is not None and period_granularity_raw is None:
        period_granularity_raw = "month"
        period_raw = month_legacy

    if period_granularity_raw is not None or period_raw is not None:
        if period_granularity_raw is None or period_raw is None:
            raise ValueError("params.period_granularity and params.period must both be set")
        granularity = _require_non_empty_str(period_granularity_raw, "params.period_granularity").lower()
        if granularity not in _PERIOD_GRANULARITIES:
            raise ValueError("params.period_granularity must be one of: year, month, week")
        period = _require_non_empty_str(period_raw, "params.period")
        params["period_granularity"] = granularity
        params["period"] = _validate_period_for_granularity(period, granularity)

    return params


def _validate_account_params(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("params must be a JSON object when ui_type is account")
    account_id = _optional_positive_int(value.get("account_id"), "params.account_id")
    if account_id is None:
        raise ValueError("params.account_id is required when ui_type is account")
    return {"account_id": account_id}


def _validate_transaction_params(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("params must be a JSON object when ui_type is transaction")
    transaction_id = _optional_positive_int(value.get("transaction_id"), "params.transaction_id")
    if transaction_id is None:
        raise ValueError("params.transaction_id is required when ui_type is transaction")
    return {"transaction_id": transaction_id}


def _validate_chart_params(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("params must be a JSON object when ui_type is chart")
    return {
        "x_axis": _require_meaningful_str(value.get("x_axis"), "params.x_axis"),
        "y_axis": _require_meaningful_str(value.get("y_axis"), "params.y_axis"),
        "series_label": (
            value.get("series_label").strip()
            if isinstance(value.get("series_label"), str) and value.get("series_label").strip()
            and value.get("series_label").strip().lower() not in _NULL_STRINGS
            else None
        ),
    }


def _validate_determine_ui_response(parsed: Any) -> dict[str, Any]:
    if not isinstance(parsed, dict):
        raise ValueError("Response must be a JSON object")

    rationale = _require_non_empty_str(parsed.get("rationale"), "rationale")

    ui_type = _require_non_empty_str(parsed.get("ui_type"), "ui_type").lower()
    if ui_type not in _UI_TYPES:
        raise ValueError(f"ui_type must be one of: {', '.join(sorted(_UI_TYPES))}")

    params_raw = _resolve_params_raw(parsed, ui_type)
    if ui_type == "none":
        params = None
    elif params_raw is None:
        raise ValueError(f"params is required when ui_type is {ui_type}")
    elif ui_type == "cashflow":
        params = _validate_cashflow_params(params_raw)
    elif ui_type == "account":
        params = _validate_account_params(params_raw)
    elif ui_type == "transaction":
        params = _validate_transaction_params(params_raw)
    else:
        params = _validate_chart_params(params_raw)

    return {
        "rationale": rationale,
        "ui_type": ui_type,
        "params": params,
    }


def format_determine_ui_response_body(raw_response: str) -> str:
    body = (raw_response or "").strip()
    if not body:
        raise ValueError("raw_response must be non-empty")
    return f"# Raw Assistant Response\n\n{body}\n"


def build_determine_ui_user_task(task_prompt: str, raw_response: str) -> str:
    instructions = (task_prompt or "").strip()
    body = format_determine_ui_response_body(raw_response)
    if instructions:
        return f"{instructions}\n\n{body}"
    return body


def format_determine_ui_user_message(raw_response: str) -> str:
    return build_determine_ui_user_task(TASK_PROMPT, raw_response)


def resolve_test_case_input(test_case: dict[str, Any]) -> str:
    for key in ("raw_response", "input", "response"):
        raw = test_case.get(key)
        if isinstance(raw, str) and raw.strip():
            return raw.strip()
    raise ValueError("test case must include raw_response, input, or response")


class DetermineUiOptimizer:
    """Gemini runner for the determine-UI system prompt."""

    def __init__(
        self,
        model_name: str = GEMINI_FLASH_LITE,
        *,
        thinking_budget: int = DETERMINE_UI_THINKING_BUDGET,
        max_output_tokens: int = DETERMINE_UI_MAX_OUTPUT_TOKENS,
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
        self.temperature = 0.2
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

    def _build_config(self) -> "types.GenerateContentConfig":
        return types.GenerateContentConfig(
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

    def _contents_for_raw_response(self, raw_response: str) -> list["types.Content"]:
        user_text = format_determine_ui_user_message(raw_response)
        return [types.Content(role="user", parts=[types.Part.from_text(text=user_text)])]

    def _parse_response_text(self, output_text: str, thought_summary: str, finish_reason: Any) -> dict[str, Any]:
        parsed: dict[str, Any] | None = None
        parse_error: Exception | None = None
        for source in (output_text, thought_summary):
            if not (source or "").strip():
                continue
            try:
                parsed = _parse_model_json_object(source)
                break
            except (json.JSONDecodeError, ValueError) as exc:
                parse_error = exc

        if parsed is None:
            reason = str(finish_reason or "unknown")
            detail = str(parse_error or "unknown parse error")
            raise ValueError(f"Invalid JSON response. finish_reason={reason!r}; {detail}") from parse_error

        return _validate_determine_ui_response(parsed)

    def generate_response(self, raw_response: str) -> dict[str, Any]:
        contents = self._contents_for_raw_response(raw_response)
        cfg = self._build_config()

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

        if not (output_text or "").strip() and not (thought_summary or "").strip():
            raise ValueError("Empty response from model. Check API key and model availability.")

        return self._parse_response_text(output_text, thought_summary, finish_reason)

    def generate_response_stream(self, raw_response: str) -> dict[str, Any]:
        """Stream JSON tokens to stdout (matches /real_penny_chat chunk UX)."""
        contents = self._contents_for_raw_response(raw_response)
        cfg = self._build_config()
        accumulated = ""
        finish_reason = None

        print("--- stream ---", file=sys.stderr)
        try:
            stream = self.client.models.generate_content_stream(
                model=self.model_name,
                contents=contents,
                config=cfg,
            )
            for chunk in stream:
                for cand in getattr(chunk, "candidates", None) or []:
                    reason = getattr(cand, "finish_reason", None)
                    if reason is not None:
                        finish_reason = reason
                    content = getattr(cand, "content", None)
                    if not content:
                        continue
                    for part in getattr(content, "parts", None) or []:
                        t = getattr(part, "text", None)
                        if not isinstance(t, str) or not t or getattr(part, "thought", False):
                            continue
                        accumulated += t
                        sys.stdout.write(t)
                        sys.stdout.flush()
        except ClientError as e:
            if self.thinking_budget == 0 and "only works in thinking mode" in (str(e) or ""):
                print(
                    "\n[NOTE] This model requires thinking mode; use default (no --no-thinking) or a different model.",
                    flush=True,
                )
                sys.exit(1)
            raise

        print(file=sys.stdout)
        print("--- end stream ---", file=sys.stderr)

        if not accumulated.strip():
            raise ValueError("Empty streamed response from model.")

        return self._parse_response_text(accumulated, "", finish_reason)


TEST_CASES: list[dict[str, Any]] = [
    {
        "name": "cashflow_groceries_category",
        "batch": 1,
        "input": """Groceries spending in **category 4** is up **20%** in **January 2026** compared to your three-month average. Your top merchants were Whole Foods and Trader Joe's.""",
        "ideal_response": {
            "rationale": "Response focuses on groceries category 4 spending for January 2026.",
            "ui_type": "cashflow",
            "params": {
                "category_id": 4,
                "period_granularity": "month",
                "period": "2026-01",
            },
        },
    },
    {
        "name": "account_checking_balance",
        "batch": 1,
        "input": """Your Chase checking account (**account_id 12858**) dropped to **$842** this week after rent and utilities posted.""",
        "ideal_response": {
            "rationale": "Response is about a specific checking account balance.",
            "ui_type": "account",
            "params": {"account_id": 12858},
        },
    },
    {
        "name": "transaction_netflix_charge",
        "batch": 1,
        "input": """That **$15.99 Netflix** charge on **transaction_id 99123** looks like your usual monthly subscription.""",
        "ideal_response": {
            "rationale": "Response highlights one specific Netflix transaction.",
            "ui_type": "transaction",
            "params": {"transaction_id": 99123},
        },
    },
    {
        "name": "chart_netflix_trend",
        "batch": 2,
        "input": """Here is how much you have spent on **Netflix** by month over the last six months — it has been steady around **$16** except for a **$32** double-charge in March.""",
        "ideal_response": {
            "rationale": "User needs a month-by-month Netflix spending trend chart.",
            "ui_type": "chart",
            "params": {
                "x_axis": "month",
                "y_axis": "netflix_spending",
                "series_label": "Netflix",
            },
        },
    },
    {
        "name": "none_general_budget_advice",
        "batch": 2,
        "input": """A simple budget rule is 50/30/20: needs, wants, and savings. Start by tracking discretionary spend for two weeks before cutting categories.""",
        "ideal_response": {
            "rationale": "General budgeting advice with no specific account, category id, or transaction.",
            "ui_type": "none",
            "params": None,
        },
    },
    {
        "name": "none_category_name_without_id",
        "batch": 2,
        "input": """Your **Groceries** spending is elevated this month, but I do not have a category drill-down id in this summary.""",
        "ideal_response": {
            "rationale": "Mentions Groceries by name but no explicit category_id to deep-link.",
            "ui_type": "none",
            "params": None,
        },
    },
    {
        "name": "none_goal_created_eating_out",
        "batch": 3,
        "input": """I am Penny, your AI financial bestie. I am here to help you with your financial questions and concerns. Successfully created 1 goal.

Created:
- Weekly Eating Out Budget.""",
        "ideal_response": {
            "rationale": "Goal creation confirmation with no category id or other drill-down target.",
            "ui_type": "none",
            "params": None,
        },
    },
    {
        "name": "none_emergency_fund_savings_advice",
        "batch": 3,
        "input": """To reach a secure foundation of 6 months of living expenses ($24,000), you would need to save your $2,000 surplus for 12 months.
Do you have any existing savings toward this goal?
If you link your accounts, Penny can track your progress toward this emergency fund in real-time and provide personalized insights on how to reach your goal faster.""",
        "ideal_response": {
            "rationale": "General savings-planning advice without a specific account, category, or transaction to open.",
            "ui_type": "none",
            "params": None,
        },
    },
    {
        "name": "none_streaming_subscriptions_last_month",
        "batch": 3,
        "input": """I am Penny, your AI financial bestie. I am here to help you with your financial questions and concerns. Streaming transactions/subscriptions from last month:
Transactions:
- Disney+: $8 on 2026-05-27 (account_id: 12725, transaction_id: 3660248)
- Netflix: $15 on 2026-05-13 (account_id: 12725, transaction_id: 3660212)
- Spotify: $11 on 2026-05-08 (account_id: 12724, transaction_id: 3659438)
Subscriptions:
- Disney+: 7.99 on 2026-05-27
- Spotify: 10.99 on 2026-05-08""",
        "ideal_response": {
            "rationale": "Multi-subscription summary listing several transactions without focusing on one drill-down target.",
            "ui_type": "none",
            "params": None,
        },
    },
    {
        "name": "none_multiple_account_balances",
        "batch": 3,
        "input": """I am Penny, your AI financial bestie. I am here to help you with your financial questions and concerns. Depository Accounts:
- 360 Checking (ID: 12854): $1645
Credit/Loan Accounts:
- Venture Rewards (ID: 12855): $3418""",
        "ideal_response": {
            "rationale": "Summary spans multiple accounts rather than one specific account to open.",
            "ui_type": "none",
            "params": None,
        },
    },
    {
        "name": "chart_groceries_three_month_breakdown",
        "batch": 3,
        "input": """I am Penny, your AI financial bestie. I am here to help you with your financial questions and concerns. Total spent on groceries over the past 3 months: $8771
Monthly breakdown:
- 2026-04: $4035
- 2026-05: $3892
- 2026-06: $844""",
        "ideal_response": {
            "rationale": "Response is a month-by-month groceries spending breakdown best shown as a chart.",
            "ui_type": "chart",
            "params": {
                "x_axis": "month",
                "y_axis": "groceries_spending",
                "series_label": "Groceries",
            },
        },
    },
]


def _run_test(
    raw_response: str,
    optimizer: DetermineUiOptimizer | None = None,
    *,
    ideal: dict[str, Any] | None = None,
    stream: bool = False,
) -> dict[str, Any]:
    if optimizer is None:
        optimizer = DetermineUiOptimizer()
    wrapped = format_determine_ui_user_message(raw_response)
    print("=" * 80)
    print("LLM INPUT:")
    print("=" * 80)
    print(wrapped)
    if stream:
        result = optimizer.generate_response_stream(raw_response)
    else:
        result = optimizer.generate_response(raw_response)
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
    optimizer: DetermineUiOptimizer | None = None,
    *,
    stream: bool = False,
) -> dict[str, Any] | None:
    if optimizer is None:
        optimizer = DetermineUiOptimizer()

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
        return _run_test(payload, optimizer, ideal=ideal, stream=stream)

    tc = get_test_case(test_name_or_index_or_dict)
    if tc is None:
        print(f"Test case '{test_name_or_index_or_dict}' not found.")
        return None
    print(f"\n{'=' * 80}\nRunning test: {tc['name']}\n{'=' * 80}\n")
    ideal = _resolve_ideal_response(tc)
    return _run_test(resolve_test_case_input(tc), optimizer, ideal=ideal, stream=stream)


def _load_raw_response_arg(args: argparse.Namespace) -> str:
    if args.response_file:
        text = Path(args.response_file).read_text(encoding="utf-8")
        if args.raw_response:
            raise ValueError("use either --raw-response or --response-file, not both")
        return text.strip()
    if not args.raw_response:
        raise ValueError("--raw-response or --response-file is required for ad-hoc runs")
    return args.raw_response.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run P:DetermineUI optimizer tests")
    parser.add_argument("--test", type=str, help='Test name or index (e.g. "0" or "cashflow_groceries_category")')
    parser.add_argument("--batch", type=int, help="Run all tests in batch N")
    parser.add_argument("--raw-response", type=str, help="Raw non-verbalized assistant text")
    parser.add_argument("--response-file", type=str, help="File containing raw assistant text")
    parser.add_argument(
        "--print-input-only",
        action="store_true",
        help="Only print formatted LLM user message (no model call)",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Stream JSON tokens while generating (matches /real_penny_chat)",
    )
    parser.add_argument("--model", type=str, default=GEMINI_FLASH_LITE)
    parser.add_argument("--no-thinking", action="store_true", help="Set thinking_budget=0")
    args = parser.parse_args()

    if args.raw_response or args.response_file:
        try:
            raw_response = _load_raw_response_arg(args)
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            raise SystemExit(1) from exc
        wrapped = format_determine_ui_user_message(raw_response)
        print("FORMATTED DETERMINE UI INPUT")
        print("-" * 80)
        print(wrapped)
        if args.print_input_only:
            return
        thinking_budget = 0 if args.no_thinking else DETERMINE_UI_THINKING_BUDGET
        optimizer = DetermineUiOptimizer(model_name=args.model, thinking_budget=thinking_budget)
        print("\nDETERMINE UI LLM OUTPUT")
        print("-" * 80)
        if args.stream:
            result = optimizer.generate_response_stream(raw_response)
            print(json.dumps(result, indent=2))
        else:
            print(json.dumps(optimizer.generate_response(raw_response), indent=2))
        return

    if args.print_input_only:
        print("Error: --print-input-only requires --raw-response or --response-file", file=sys.stderr)
        raise SystemExit(1)

    if args.batch is None and args.test is None:
        _print_usage()
        return

    thinking_budget = 0 if args.no_thinking else DETERMINE_UI_THINKING_BUDGET
    optimizer = DetermineUiOptimizer(model_name=args.model, thinking_budget=thinking_budget)

    if args.batch is not None:
        cases = [tc for tc in TEST_CASES if int(tc.get("batch") or 0) == int(args.batch)]
        if not cases:
            raise SystemExit(f"No tests found for batch={args.batch}")
        for i, tc in enumerate(cases):
            if i:
                print("\n" + "-" * 80 + "\n")
            run_test(tc, optimizer, stream=args.stream)
        return

    if args.test is not None:
        if args.test.strip().lower() == "all":
            for i in range(len(TEST_CASES)):
                run_test(i, optimizer, stream=args.stream)
                if i < len(TEST_CASES) - 1:
                    print("\n" + "-" * 80 + "\n")
            return
        test_val: str | int = int(args.test) if args.test.isdigit() else args.test
        run_test(test_val, optimizer, stream=args.stream)
        return


def _print_usage() -> None:
    print("Usage:")
    print("  Run a single test: --test <name_or_index>")
    print("  Run all tests: --test all")
    print("  Run batch: --batch <N>")
    print("  Ad-hoc raw response: --raw-response <text> | --response-file <path>")
    print("  Stream tokens: add --stream")
    print("  Print formatted input only: --raw-response <text> --print-input-only")
    print("\nAvailable test cases:")
    for i, tc in enumerate(TEST_CASES):
        batch = tc.get("batch", "?")
        print(f"  {i}: {tc['name']} (batch {batch})")


if __name__ == "__main__":
    main()

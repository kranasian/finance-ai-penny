"""
Optimizer runner for **P:NeedVerbalizer** (Gemini prompt tuning).

Input is trimmed ``simulate_financial_strategy`` markdown (``# Financial Needs`` and evidence only).

Run from ``finance-ai-penny`` repo root (``finance-ai-penny/.venv`` or ``finance-ai-llm-server/llm``):

  python3 active_experiments/need_verbalizer_optimizer.py --test 0
  python3 active_experiments/need_verbalizer_optimizer.py --test all
  python3 active_experiments/need_verbalizer_optimizer.py --simulate-agent-outcome-id 1148 --print-input-only
  python3 active_experiments/need_verbalizer_optimizer.py --simulate-agent-outcome-id 1148
  python3 active_experiments/need_verbalizer_optimizer.py --user-id 3

DB-backed runs read ``SLAVE_DB`` from ``finance-ai-llm-server/config.ini``. Requires ``psycopg2-binary``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import warnings
from calendar import monthrange
from datetime import date
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

from active_experiments.verbalizer_optimizer_db import (
    _bundled_input_from_test_case,
    _normalize_goal_plan_list,
    _resolve_ideal_response,
    load_simulate_agent_outcome_markdown,
    resolve_simulate_agent_outcome_id,
)

if load_dotenv is not None:
    load_dotenv()

GEMINI_FLASH_LITE = "gemini-flash-lite-latest"
NEED_VERBALIZER_THINKING_BUDGET = 256
NEED_VERBALIZER_MAX_OUTPUT_TOKENS = 2048

_FINANCIAL_NEEDS_H1 = "# Financial Needs"
_FINANCIAL_STRATEGY_H1 = "# Financial Strategy"
_SIMULATE_OUTCOME_TYPE = "simulate_financial_strategy"

_EXCLUDED_SIMULATE_SECTIONS = (
    "## Credit Interest Rates",
    "## Immediate Things to Do",
    "## Next Set of Milestones to Aspire",
    "## Next Set of Milestones",
)

_MONTH_RANGE_RE = re.compile(r"^\s*(\d+)\s*(?:-\s*(\d+)|\+)?\s*$")
_RE_PLAN_SCENARIO_HEADING = re.compile(
    r"(?m)^(## (?:Recommended|Alternative) plan:\s*[^\n]+)\n(?!\n)",
)
_SPENDING_SCHEDULE_H3 = "### Spending Schedule"
_OPEN_ENDED_PLAN_HORIZON_MONTHS = 24

_CATEGORY_ORDER = ("meals", "leisure", "shopping", "health", "education", "uncategorized")
_CATEGORY_DISPLAY_LABELS: dict[str, str] = {
    "meals": "food",
}


def _category_display_label(slug: str) -> str:
    return _CATEGORY_DISPLAY_LABELS.get(slug, slug)

SYSTEM_PROMPT = """You are Penny — a sharp, witty money coach who turns the diagnosed financial need into copy users actually want to read.

`need_title`: short headline for the primary need from ``# Financial Needs`` (max **8 words**; punchy, no jargon).

`need_summary`: one sentence expanding the need (max **25 words**); ground every **$** and date in the input. Fun and confident, never cheesy, patronizing, or naggy.

Max **35 words** total across ``need_title`` and ``need_summary``. No exclamation marks, superlatives, or emoji.
"""


def _build_output_schema() -> "types.Schema":
    if types is None:  # pragma: no cover
        raise RuntimeError("Install `google-genai` for this optimizer.")
    return types.Schema(
        type=types.Type.OBJECT,
        required=["need_title", "need_summary"],
        properties={
            "need_title": types.Schema(
                type=types.Type.STRING,
                description="Short headline for the primary financial need (max 8 words).",
            ),
            "need_summary": types.Schema(
                type=types.Type.STRING,
                description="One-sentence expansion of the need (max 25 words).",
            ),
        },
    )


def _validate_need_response(parsed: Any) -> dict[str, Any]:
    if not isinstance(parsed, dict):
        raise ValueError("Response must be a JSON object")
    need_title = parsed.get("need_title")
    if not isinstance(need_title, str) or not need_title.strip():
        raise ValueError("need_title must be a non-empty string")
    need_summary = parsed.get("need_summary")
    if not isinstance(need_summary, str) or not need_summary.strip():
        raise ValueError("need_summary must be a non-empty string")
    return {
        "need_title": need_title.strip(),
        "need_summary": need_summary.strip(),
    }


def _parse_model_json_object(text: str) -> dict[str, Any]:
    raw = (text or "").strip()
    if not raw:
        raise ValueError("empty JSON text")

    candidates: list[str] = [raw]
    for pattern in (
        r"```(?:json)?\s*(\{.*?\})\s*```",
        r"(\{[^{}]*\})",
        r"(\{.*\})",
    ):
        match = re.search(pattern, raw, re.DOTALL)
        if match:
            candidates.append(match.group(1).strip())

    seen: set[str] = set()
    last_exc: json.JSONDecodeError | None = None
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError as exc:
            last_exc = exc
            continue
        if isinstance(parsed, dict):
            return parsed
        raise ValueError("Response must be a JSON object")
    if last_exc is not None:
        raise last_exc
    raise ValueError("Invalid JSON response")


def _strip_excluded_simulate_sections(body: str) -> str:
    text = (body or "").strip()
    for header in _EXCLUDED_SIMULATE_SECTIONS:
        pattern = (
            rf"(?ms)^[ \t]*{re.escape(header)}[ \t]*\n"
            rf"(.*?)(?=^[ \t]*# |\Z)"
        )
        text = re.sub(pattern, "", text)
    text = re.sub(r"\n{3,}", "\n\n", text.strip())
    return text + "\n"


def _trim_simulate_outcome_from_financial_needs(simulate_outcome_md: str) -> str:
    body = (simulate_outcome_md or "").strip()
    idx = body.find(_FINANCIAL_NEEDS_H1)
    if idx < 0:
        raise ValueError(f"simulate outcome must include {_FINANCIAL_NEEDS_H1}")
    return _strip_excluded_simulate_sections(body[idx:])


def trim_simulate_outcome_for_need_bundle(simulate_outcome_md: str) -> str:
    """Drop content before ``# Financial Needs``, excluded needs subsections, and strategy prose."""
    text = _trim_simulate_outcome_from_financial_needs(simulate_outcome_md)
    strategy_idx = text.find(_FINANCIAL_STRATEGY_H1)
    if strategy_idx >= 0:
        text = text[:strategy_idx].rstrip() + "\n"
    return text


def trim_simulate_outcome_for_plan_bundle(simulate_outcome_md: str) -> str:
    """Drop content before ``# Financial Needs`` and excluded subsections; keep ``# Financial Strategy``."""
    text = _trim_simulate_outcome_from_financial_needs(simulate_outcome_md)
    strategy_idx = text.find(_FINANCIAL_STRATEGY_H1)
    if strategy_idx < 0:
        return text
    after_strategy = text[strategy_idx + len(_FINANCIAL_STRATEGY_H1):]
    next_h1 = re.search(r"(?m)^# [^#]", after_strategy)
    if next_h1:
        end = strategy_idx + len(_FINANCIAL_STRATEGY_H1) + next_h1.start()
        text = text[:end].rstrip() + "\n"
    return text


def _parse_start_end_month(spec: str) -> tuple[int, int | None]:
    text = str(spec or "").strip()
    match = _MONTH_RANGE_RE.match(text)
    if not match:
        raise ValueError(f"invalid start_end_month: {spec!r}")
    start = int(match.group(1))
    if start < 1:
        raise ValueError(f"start_end_month must be >= 1: {spec!r}")
    if text.endswith("+"):
        return start, None
    if match.group(2) is not None:
        end = int(match.group(2))
        if end < start:
            raise ValueError(f"invalid month range: {spec!r}")
        return start, end
    return start, start


def _add_months(d: date, n: int) -> date:
    m0 = d.month - 1 + n
    year = d.year + m0 // 12
    month = m0 % 12 + 1
    return date(year, month, min(d.day, monthrange(year, month)[1]))


def _calendar_month_label(today: date, month_index: int) -> str:
    return _add_months(today, max(0, month_index)).strftime("%m/%y")


def _timeline_end_index(start_idx: int, end_idx: int | None) -> int:
    if end_idx is not None:
        return end_idx
    return max(start_idx, _OPEN_ENDED_PLAN_HORIZON_MONTHS)


def _positive_category_amounts(categories: dict[str, Any]) -> dict[str, int]:
    out: dict[str, int] = {}
    for key, value in categories.items():
        slug = str(key).strip()
        if not slug:
            continue
        try:
            amt = max(0, int(round(float(value or 0))))
        except (TypeError, ValueError):
            continue
        if amt > 0:
            out[slug] = amt
    return out


def _spending_timeline_from_schedule(
    schedule: list[Any],
    *,
    today: date | None = None,
) -> list[dict[str, Any]]:
    """Mirror Hermes ``spending_timeline_from_schedule`` for persisted ``spending_schedule`` rows."""
    anchor = today or date.today()
    timeline: list[dict[str, Any]] = []
    for entry in schedule:
        if not isinstance(entry, dict):
            continue
        spec = str(entry.get("start_end_month") or "").strip()
        try:
            start_idx, parsed_end = _parse_start_end_month(spec)
        except ValueError:
            continue
        open_ended = spec.endswith("+")
        end_idx = _timeline_end_index(start_idx, parsed_end)
        categories = entry.get("categories")
        if not isinstance(categories, dict):
            continue
        pos_cats = _positive_category_amounts(categories)
        if not pos_cats:
            continue
        timeline.append({
            "categories": pos_cats,
            "start_month": _calendar_month_label(anchor, start_idx),
            "end_month": _calendar_month_label(anchor, end_idx),
            "open_ended": open_ended,
        })
    return timeline


def _infer_open_ended_periods(
    timeline: list[dict[str, Any]],
    *,
    today: date | None = None,
) -> list[dict[str, Any]]:
    """Mark persisted timeline rows that end at the open-range horizon (``N+`` schedules)."""
    if not timeline:
        return []
    anchor = today or date.today()
    horizon_end = _calendar_month_label(anchor, _OPEN_ENDED_PLAN_HORIZON_MONTHS)
    out: list[dict[str, Any]] = []
    for i, period in enumerate(timeline):
        row = dict(period)
        if not row.get("open_ended") and i == len(timeline) - 1:
            start = str(row.get("start_month") or "").strip()
            end = str(row.get("end_month") or "").strip()
            if start and end == horizon_end and start != end:
                row["open_ended"] = True
        out.append(row)
    return out


def _resolve_spending_timeline(entry: dict[str, Any]) -> list[dict[str, Any]]:
    schedule = entry.get("spending_schedule")
    if isinstance(schedule, list) and schedule:
        return _spending_timeline_from_schedule(schedule)
    timeline = entry.get("spending_timeline")
    if isinstance(timeline, list) and timeline:
        periods = [p for p in timeline if isinstance(p, dict)]
        return _infer_open_ended_periods(periods)
    return []


def _merge_adjacent_timeline_periods(timeline: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    for period in timeline:
        if not isinstance(period, dict):
            continue
        row = dict(period)
        categories = _positive_category_amounts(row.get("categories") or {})
        if merged:
            prev = merged[-1]
            if prev.get("open_ended"):
                merged.append(row)
                continue
            prev_categories = _positive_category_amounts(prev.get("categories") or {})
            if prev_categories == categories:
                end_month = str(row.get("end_month") or "").strip()
                if end_month:
                    prev["end_month"] = end_month
                continue
        merged.append(row)
    return merged


def _period_window_compact(start: str, end: str, *, open_ended: bool = False) -> str:
    start_label = (start or "").strip()
    end_label = (end or "").strip()
    if open_ended:
        return f"{start_label}-" if start_label else ""
    if start_label and end_label and start_label != end_label:
        return f"{start_label}-{end_label}"
    return start_label or end_label


def _format_period_categories(categories: dict[str, int]) -> str:
    parts = [
        f"{_category_display_label(slug)} ${amount}"
        for slug, amount in _ordered_category_amounts(categories)
    ]
    if not parts:
        return ""
    return f"Cap {', '.join(parts)} monthly"


def _format_period_bullet(period: dict[str, Any]) -> str:
    categories = _positive_category_amounts(period.get("categories") or {})
    if not categories:
        return ""
    window = _period_window_compact(
        str(period.get("start_month") or "").strip(),
        str(period.get("end_month") or "").strip(),
        open_ended=bool(period.get("open_ended")),
    )
    body = _format_period_categories(categories)
    if not body:
        return ""
    if window:
        return f"- {window}: {body}"
    return f"- {body}"


def _format_spending_timeline(timeline: list[dict[str, Any]]) -> str:
    periods = _merge_adjacent_timeline_periods([
        period for period in timeline if isinstance(period, dict)
    ])
    bullets = [
        bullet
        for period in periods
        for bullet in [_format_period_bullet(period)]
        if bullet
    ]
    if not bullets:
        return ""
    return "\n".join(bullets)


def _ordered_category_amounts(categories: dict[str, int]) -> list[tuple[str, int]]:
    ordered: list[tuple[str, int]] = []
    seen: set[str] = set()
    for slug in _CATEGORY_ORDER:
        if slug in categories:
            ordered.append((slug, categories[slug]))
            seen.add(slug)
    for slug, amount in sorted(categories.items()):
        if slug not in seen:
            ordered.append((slug, amount))
    return ordered


def _goal_plan_rows_for_need_bundle(goal_plan: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for entry in _normalize_goal_plan_list(goal_plan):
        sid = str(entry.get("scenario_id") or "").strip()
        if not sid:
            continue
        rows.append({
            "scenario_id": sid,
            "spending_timeline": _resolve_spending_timeline(entry),
        })
    return rows


def ensure_blank_line_after_plan_headings(markdown: str) -> str:
    """Insert a blank line after ``## Recommended/Alternative plan:`` headings."""
    return _RE_PLAN_SCENARIO_HEADING.sub(r"\1\n\n", markdown or "")


def _format_goal_plan_narrative(goal_plan: Any) -> str:
    rows = _goal_plan_rows_for_need_bundle(goal_plan)
    if not rows:
        return ""

    bullets: list[str] = []
    for row in rows:
        timeline_block = _format_spending_timeline(row.get("spending_timeline") or [])
        if timeline_block:
            bullets.extend(timeline_block.splitlines())
    if not bullets:
        return ""
    return _SPENDING_SCHEDULE_H3 + "\n\n" + "\n".join(bullets) + "\n"


def build_need_verbalizer_input_bundle(
    *,
    simulate_outcome_md: str,
) -> str:
    """Bundle for **P:NeedVerbalizer** prompt tuning."""
    return trim_simulate_outcome_for_need_bundle(simulate_outcome_md)


def build_need_verbalizer_input(
    *,
    simulate_agent_outcome_id: int | None = None,
    user_id: int | None = None,
) -> str:
    """Build input from a simulate_financial_strategy outcome."""
    sim_id = resolve_simulate_agent_outcome_id(
        user_id=user_id,
        simulate_agent_outcome_id=simulate_agent_outcome_id,
    )
    _, simulate_md = load_simulate_agent_outcome_markdown(sim_id)
    return build_need_verbalizer_input_bundle(simulate_outcome_md=simulate_md)


def resolve_test_case_input(test_case: dict[str, Any]) -> str:
    for key in ("input", "bundled_input"):
        raw = test_case.get(key)
        if isinstance(raw, str) and raw.strip():
            return raw.strip() + "\n"
    bundled = _bundled_input_from_test_case(test_case)
    if bundled:
        return bundled
    raise ValueError("test case must include bundled input")


def format_need_verbalizer_user_message(profile_input: str) -> str:
    body = (profile_input or "").strip()
    if not body:
        raise ValueError("profile_input must be non-empty markdown.")
    return body + "\n"


class NeedVerbalizerOptimizer:
    """Gemini runner for the need verbalizer system prompt."""

    def __init__(
        self,
        model_name: str = GEMINI_FLASH_LITE,
        *,
        thinking_budget: int = NEED_VERBALIZER_THINKING_BUDGET,
        max_output_tokens: int = NEED_VERBALIZER_MAX_OUTPUT_TOKENS,
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
        user_text = format_need_verbalizer_user_message(profile_input)
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

        if not (output_text or "").strip() and not (thought_summary or "").strip():
            raise ValueError("Empty response from model. Check API key and model availability.")

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

        return _validate_need_response(parsed)


TEST_CASES: list[dict[str, Any]] = [
    {
        "name": "debt_paydown_interest_drag",
        "batch": 1,
        "input": """
# Financial Needs

## Primary needs
1. **Reduce interest drag**: Venture balance **$8,400** with **$312** interest paid over 90 days while spending tracks near income.

## Evidence
* **Reduce interest drag**
  - Interest tool: **$312** on Venture in 90 days.
  - Next due **2026-04-18** per payment schedule.
""",
        "ideal_response": {
            "need_title": "Venture interest drag",
            "need_summary": "$312 in interest every 90 days on your $8,400 balance while spending tracks income.",
        },
    },
    {
        "name": "cash_flow_crunch_before_mortgage",
        "batch": 1,
        "input": """
# Financial Needs

## Primary needs
1. **Stabilize cash flow**: Checking **$800** with **$2,100** mortgage due **2026-04-01** — liquidity risk before flexible spend cuts matter.

## Evidence
* **Stabilize cash flow**
  - Checking **$800** vs mortgage **$2,100** on the 1st.
  - Forecast committed outflows **$3,600**/mo vs income **$4,000**/mo.
""",
        "ideal_response": {
            "need_title": "April mortgage crunch",
            "need_summary": "Checking at $800 cannot cover the $2,100 mortgage due April 1.",
        },
    },
    {
        "name": "slow_debt_creep",
        "batch": 2,
        "input": """
# Financial Needs

## Primary needs
1. **Settle debt**: Platinum **$4,800** with slow paydown at minimum-style payments.

## Evidence
* **Settle debt**
  - Balance up **$300** over three months despite **$115**/mo payments.
  - APR tool: **~21.8%** on Platinum.
""",
        "ideal_response": {
            "need_title": "Platinum debt creep",
            "need_summary": "Balance rose $300 in three months despite $115/mo payments on $4,800 owed.",
        },
    },
]


def _run_test(
    profile_input: str,
    optimizer: NeedVerbalizerOptimizer | None = None,
    *,
    ideal: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if optimizer is None:
        optimizer = NeedVerbalizerOptimizer()
    wrapped = format_need_verbalizer_user_message(profile_input)
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
    optimizer: NeedVerbalizerOptimizer | None = None,
) -> dict[str, Any] | None:
    if optimizer is None:
        optimizer = NeedVerbalizerOptimizer()

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
    parser = argparse.ArgumentParser(description="Run P:NeedVerbalizer optimizer tests")
    parser.add_argument("--test", type=str, help='Test name or index (e.g. "0" or "debt_paydown_interest_drag")')
    parser.add_argument("--batch", type=int, help="Run all tests in batch N")
    parser.add_argument(
        "--user-id",
        type=int,
        help="User id; when simulate-agent-outcome-id is omitted, use the latest simulate_financial_strategy outcome.",
    )
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

    if args.user_id is not None or args.simulate_agent_outcome_id is not None:
        sim_id = resolve_simulate_agent_outcome_id(
            user_id=args.user_id,
            simulate_agent_outcome_id=args.simulate_agent_outcome_id,
        )
        built = build_need_verbalizer_input(
            simulate_agent_outcome_id=sim_id,
        )
        print(f"Using simulate_agent_outcome_id={sim_id}")
        print("BUILT NEED VERBALIZER INPUT")
        print("-" * 80)
        print(built)
        if args.print_input_only:
            return
        thinking_budget = 0 if args.no_thinking else NEED_VERBALIZER_THINKING_BUDGET
        optimizer = NeedVerbalizerOptimizer(model_name=args.model, thinking_budget=thinking_budget)
        print("\nNEED VERBALIZER LLM OUTPUT")
        print("-" * 80)
        print(json.dumps(optimizer.generate_response(built), indent=2))
        return

    if args.print_input_only:
        print("Error: --print-input-only requires --user-id or --simulate-agent-outcome-id", file=sys.stderr)
        raise SystemExit(1)

    if args.batch is None and args.test is None:
        _print_usage()
        return

    thinking_budget = 0 if args.no_thinking else NEED_VERBALIZER_THINKING_BUDGET
    optimizer = NeedVerbalizerOptimizer(model_name=args.model, thinking_budget=thinking_budget)

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
    print("  Build input from DB: --user-id <id> | --simulate-agent-outcome-id <id>")
    print("  Print built input only: --user-id <id> --print-input-only")
    print("\nAvailable test cases:")
    for i, tc in enumerate(TEST_CASES):
        batch = tc.get("batch", "?")
        print(f"  {i}: {tc['name']} (batch {batch})")


if __name__ == "__main__":
    main()

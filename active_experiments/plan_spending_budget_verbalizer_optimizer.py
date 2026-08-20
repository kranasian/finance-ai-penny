"""
Optimizer runner for **P:PlanSpendingBudgetVerbalizer** (Gemini prompt tuning).

Input is `### Current Spending` (table with variance) and `### Spending Schedule` (table with phased caps) for one goal-plan scenario.

Objective: markdown spending comparison table only.

Run from `finance-ai-penny` repo root:

  python3 active_experiments/plan_spending_budget_verbalizer_optimizer.py --test 0
  python3 active_experiments/plan_spending_budget_verbalizer_optimizer.py --test all
  python3 active_experiments/plan_spending_budget_verbalizer_optimizer.py --simulate-agent-outcome-id 1148 --print-input-only
  python3 active_experiments/plan_spending_budget_verbalizer_optimizer.py --user-id 3
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None

try:
    from google import genai
    from google.genai import types
    from google.genai.errors import ClientError
except Exception:
    genai = None
    types = None
    ClientError = Exception

from active_experiments.plan_verbalizer_optimizer import (
    CURRENT_SPENDING_H3,
    GEMINI_FLASH_LITE,
    SPENDING_SCHEDULE_H3,
    _collect_model_response,
    build_plan_spending_budget_verbalizer_input,
)

from active_experiments.verbalizer_optimizer_db import (
    _LLM_SERVER_ROOT,
    resolve_simulate_agent_outcome_id,
)

if str(_LLM_SERVER_ROOT) not in sys.path:
    sys.path.insert(0, str(_LLM_SERVER_ROOT))

from propose_next_steps.persist_plan_spending_budget_verbalizer import (
    format_plan_spending_budget_verbalizer_markdown,
    post_process_plan_spending_budget_verbalizer_response,
)

if load_dotenv is not None:
    load_dotenv()

PLAN_SPENDING_BUDGET_VERBALIZER_THINKING_BUDGET = 128
PLAN_SPENDING_BUDGET_VERBALIZER_MAX_OUTPUT_TOKENS = 2048

SYSTEM_PROMPT = """You are Penny — a sharp, witty money coach who builds a spending comparison table for one financial plan.

Use `### Current Spending` (table rows with current amount and variance range) and caps under `### Spending Schedule` (table rows with phased caps and percent-change labels).

Return one markdown table only with columns `Spending`, `Current`, `Budget` (separate rows using standard markdown newlines `\\n`, do NOT use `<br>` to separate rows).
  - One row per category in `### Spending Schedule` (use display names from the schedule, capitalized for a premium look).
  - `Current` from `### Current Spending` for that category (ground every **$**).
  - `Budget` may use multiple lines in a cell (separate with `<br>`) when the schedule has multiple phases:
    - first phase: `$amount (n% cut)` vs Current, or `$amount (n% up)` when higher; omit the percent label when unchanged (no `0% cut` / `0% up`)
    - later phases: `$amount N months later` (months from plan start to that phase)
  - Final row: `Total` with summed Current and Budget totals (Budget totals also multi-line when phased).

Do not invent Current amounts — only use `### Current Spending`. Output markdown table only — no title heading, no JSON, no code fences, no extra prose.
"""


def _validate_spending_budget_response(parsed: Any, *, profile_input: str = "") -> dict[str, Any]:
    if isinstance(parsed, dict):
        cleaned = post_process_plan_spending_budget_verbalizer_response(parsed, profile_input)
    elif isinstance(parsed, str):
        cleaned = post_process_plan_spending_budget_verbalizer_response(parsed, profile_input)
    else:
        raise ValueError("Response must be markdown text or a JSON object")
    return cleaned


TEST_CASES: list[dict[str, Any]] = [
    {
        "name": "debt_paydown_recommended",
        "batch": 1,
        "input": """### Current Spending
| food | $1,000 |  $900 ~ $1,100  |
| leisure | $500 |  $450 ~ $550  |

### Spending Schedule
| 04/2026 to 06/2026 |  Cap food to $850 (15% less), leisure $450 (10% less) monthly |
| 07/2026 to 03/2028 |  Cap food to $700 (30% less), leisure $350 (30% less) monthly |
""",
        "ideal_response": """| Spending | Current | Budget |
| --- | --- | --- |
| food | $1,000 | $850 (15% cut)<br>$700 3 months later |
| leisure | $500 | $450 (10% cut)<br>$350 3 months later |
| Total | $1,500 | $1,300<br>$1,050 |""",
    },
    {
        "name": "debt_paydown_alternative",
        "batch": 1,
        "scenario_id": "steady_cut",
        "input": """### Current Spending
| food | $1,000 |  $900 ~ $1,100  |
| leisure | $500 |  $450 ~ $550  |

### Spending Schedule
| 04/2026 to 03/2028 |  Cap food to $700 (30% less), leisure $350 (30% less) monthly |
""",
        "ideal_response": """| Spending | Current | Budget |
| --- | --- | --- |
| food | $1,000 | $700 (30% cut) |
| leisure | $500 | $350 (30% cut) |
| Total | $1,500 | $1,050 |""",
    },
    {
        "name": "cash_flow_recommended",
        "batch": 1,
        "input": """### Current Spending
| food | $650 |  $600 ~ $700  |
| shopping | $250 |  $220 ~ $280  |

### Spending Schedule
| 04/2026 to 03/2028 |  Cap food to $520 (20% less), shopping $180 (28% less) monthly |
""",
        "ideal_response": """| Spending | Current | Budget |
| --- | --- | --- |
| food | $650 | $520 (20% cut) |
| shopping | $250 | $180 (28% cut) |
| Total | $900 | $700 |""",
    },
    {
        "name": "cash_flow_alternative",
        "batch": 1,
        "scenario_id": "aggressive_flex_cut",
        "input": """### Current Spending
| food | $650 |  $600 ~ $700  |
| shopping | $250 |  $220 ~ $280  |

### Spending Schedule
| 04/2026 to 03/2028 |  Cap food to $450 (31% less), shopping $150 (40% less) monthly |
""",
        "ideal_response": """| Spending | Current | Budget |
| --- | --- | --- |
| food | $650 | $450 (31% cut) |
| shopping | $250 | $150 (40% cut) |
| Total | $900 | $600 |""",
    },
    {
        "name": "slow_debt_recommended",
        "batch": 2,
        "input": """### Current Spending
| food | $650 |  $600 ~ $700  |
| leisure | $400 |  $350 ~ $430  |

### Spending Schedule
| 04/2026 to 03/2028 |  Cap food to $520 (20% less), leisure $300 (25% less) monthly |
""",
        "ideal_response": """| Spending | Current | Budget |
| --- | --- | --- |
| food | $650 | $520 (20% cut) |
| leisure | $400 | $300 (25% cut) |
| Total | $1,050 | $820 |""",
    },
    {
        "name": "slow_debt_alternative",
        "batch": 2,
        "scenario_id": "leisure_first",
        "input": """### Current Spending
| food | $650 |  $600 ~ $700  |
| leisure | $400 |  $350 ~ $430  |

### Spending Schedule
| 04/2026 to 03/2028 |  Cap food to $450 (31% less), leisure $380 (5% less) monthly |
""",
        "ideal_response": """| Spending | Current | Budget |
| --- | --- | --- |
| food | $650 | $450 (31% cut) |
| leisure | $400 | $380 (5% cut) |
| Total | $1,050 | $830 |""",
    },
    {
        "name": "spending_drift_recommended",
        "batch": 3,
        "simulate_agent_outcome_id": 1252,
        "input": """### Current Spending
| food | $1,400 |  $1,200 ~ $1,500  |
| leisure | $400 |  $350 ~ $450  |
| shopping | $80 |  $60 ~ $100  |
| health | $80 |  $70 ~ $90  |
| education | $450 |  $400 ~ $500  |
| uncategorized | $350 |  $300 ~ $400  |

### Spending Schedule
| 08/2026 to 10/2026 |  Cap food to $1,200 (14% less), leisure $300 (25% less), shopping $50 (38% less), health $80, education $450, uncategorized $300 (14% less) monthly |
| 11/2026 to 01/2027 |  Cap food to $750 (46% less), leisure $200 (50% less), shopping $50 (38% less), health $80, education $450, uncategorized $300 (14% less) monthly |
| 02/2027 to future |  Cap food to $500 (64% less), leisure $100 (75% less), shopping $50 (38% less), health $80, education $450, uncategorized $300 (14% less) monthly |
""",
        "ideal_response": """| Spending | Current | Budget |
| --- | --- | --- |
| food | $1,400 | $1,200 (14% cut)<br>$750 3 months later<br>$500 6 months later |
| leisure | $400 | $300 (25% cut)<br>$200 3 months later<br>$100 6 months later |
| shopping | $80 | $50 (38% cut)<br>$50 3 months later<br>$50 6 months later |
| health | $80 | $80<br>$80 3 months later<br>$80 6 months later |
| education | $450 | $450<br>$450 3 months later<br>$450 6 months later |
| uncategorized | $350 | $300 (14% cut)<br>$300 3 months later<br>$300 6 months later |
| Total | $2,760 | $2,380<br>$1,830<br>$1,480 |""",
    },
    {
        "name": "emergency_savings_target_recommended",
        "batch": 4,
        "input": """### Current Spending
| food | $650 |  $600 ~ $700  |
| leisure | $400 |  $350 ~ $430  |
| shopping | $80 |  $60 ~ $100  |

### Spending Schedule
| 04/2026 to 03/2028 |  Cap food to $520 (20% less), leisure $300 (25% less), shopping $50 (38% less) monthly |
""",
        "ideal_response": """| Spending | Current | Budget |
| --- | --- | --- |
| food | $650 | $520 (20% cut) |
| leisure | $400 | $300 (25% cut) |
| shopping | $80 | $50 (38% cut) |
| Total | $1,130 | $870 |""",
    },
]


def format_plan_spending_budget_verbalizer_user_message(profile_input: str) -> str:
    body = (profile_input or "").strip()
    if not body:
        raise ValueError("profile_input must be non-empty markdown.")
    if CURRENT_SPENDING_H3 not in body:
        raise ValueError(f"profile_input must include {CURRENT_SPENDING_H3}.")
    if SPENDING_SCHEDULE_H3 not in body:
        raise ValueError(f"profile_input must include {SPENDING_SCHEDULE_H3}.")
    return body + "\n"


def _parse_spending_budget_markdown_response(text: str) -> str:
    raw = (text or "").strip()
    if raw.startswith("```"):
        blocks = raw.split("```")
        for block in blocks:
            candidate = block.strip()
            if candidate.startswith("markdown\n"):
                candidate = candidate.split("\n", 1)[1].strip()
            if candidate.startswith("|"):
                return candidate + ("\n" if not candidate.endswith("\n") else "")
    if not raw:
        raise ValueError("Empty markdown response from model")
    if raw.startswith("###"):
        lines = raw.splitlines()
        index = 1
        while index < len(lines) and not lines[index].strip():
            index += 1
        raw = "\n".join(lines[index:]).strip()
    if not raw.startswith("|"):
        raise ValueError("response must be a markdown table")
    return raw + ("\n" if not raw.endswith("\n") else "")


def resolve_spending_budget_test_case_input(test_case: dict[str, Any]) -> str:
    raw = test_case.get("input")
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError("test case must include input")
    return raw.strip() + "\n"


class PlanSpendingBudgetVerbalizerOptimizer:
    def __init__(
        self,
        model_name: str = GEMINI_FLASH_LITE,
        *,
        thinking_budget: int = PLAN_SPENDING_BUDGET_VERBALIZER_THINKING_BUDGET,
        max_output_tokens: int = PLAN_SPENDING_BUDGET_VERBALIZER_MAX_OUTPUT_TOKENS,
    ):
        if genai is None or types is None:
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

    def _build_generate_config(self, *, max_output_tokens: int) -> "types.GenerateContentConfig":
        return types.GenerateContentConfig(
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            max_output_tokens=max_output_tokens,
            safety_settings=self.safety_settings,
            system_instruction=[types.Part.from_text(text=self.system_prompt)],
            thinking_config=types.ThinkingConfig(
                thinking_budget=self.thinking_budget,
                include_thoughts=False,
            ),
        )

    def generate_response(self, profile_input: str) -> dict[str, Any]:
        user_text = format_plan_spending_budget_verbalizer_user_message(profile_input)
        request_text = types.Part.from_text(text=user_text)
        contents = [types.Content(role="user", parts=[request_text])]

        token_limits = [self.max_output_tokens]
        retry_limit = self.max_output_tokens * 2
        if retry_limit > self.max_output_tokens:
            token_limits.append(retry_limit)

        last_error: Exception | None = None
        for attempt_idx, max_tokens in enumerate(token_limits):
            cfg = self._build_generate_config(max_output_tokens=max_tokens)
            output_text = ""
            finish_reason = None
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=contents,
                    config=cfg,
                )
                output_text, _, finish_reason = _collect_model_response(response)
            except ClientError as e:
                if self.thinking_budget == 0 and "only works in thinking mode" in (str(e) or ""):
                    print(
                        "\n[NOTE] This model requires thinking mode; use default (no --no-thinking) or a different model.",
                        flush=True,
                    )
                    sys.exit(1)
                raise

            if not (output_text or "").strip():
                last_error = ValueError(
                    f"Empty markdown response from model. finish_reason={finish_reason!r}"
                )
                if attempt_idx < len(token_limits) - 1:
                    print(
                        f"\n[RETRY] Empty response at max_output_tokens={max_tokens}; "
                        f"retrying with {token_limits[attempt_idx + 1]}.\n",
                        flush=True,
                    )
                    continue
                raise last_error

            try:
                markdown = _parse_spending_budget_markdown_response(output_text)
            except ValueError as exc:
                reason = str(finish_reason or "unknown")
                preview = output_text.strip()[:240].replace("\n", " ")
                last_error = ValueError(
                    f"Invalid markdown response. finish_reason={reason!r}; "
                    f"max_output_tokens={max_tokens}; preview={preview!r}"
                )
                last_error.__cause__ = exc
                if "MAX_TOKENS" in reason and attempt_idx < len(token_limits) - 1:
                    print(
                        f"\n[RETRY] MAX_TOKENS at max_output_tokens={max_tokens}; "
                        f"retrying with {token_limits[attempt_idx + 1]}.\n",
                        flush=True,
                    )
                    continue
                raise last_error from exc

            try:
                return _validate_spending_budget_response(markdown, profile_input=profile_input)
            except ValueError as exc:
                raise ValueError(f"Response failed validation: {exc}") from exc

        if last_error is not None:
            raise last_error
        raise ValueError("Invalid markdown response from model.")


def _run_test(
    profile_input: str,
    optimizer: PlanSpendingBudgetVerbalizerOptimizer | None = None,
    *,
    ideal: str | None = None,
) -> dict[str, Any]:
    if optimizer is None:
        optimizer = PlanSpendingBudgetVerbalizerOptimizer()
    wrapped = format_plan_spending_budget_verbalizer_user_message(profile_input)
    print("=" * 80)
    print("LLM INPUT:")
    print("=" * 80)
    print(wrapped)
    result = optimizer.generate_response(profile_input)
    print("=" * 80)
    print("LLM OUTPUT:")
    print("=" * 80)
    print(format_plan_spending_budget_verbalizer_markdown(result))
    if ideal is not None:
        print("=" * 80)
        print("IDEAL RESPONSE:")
        print("=" * 80)
        print(ideal.strip() + "\n")
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
    optimizer: PlanSpendingBudgetVerbalizerOptimizer | None = None,
    *,
    scenario_id: str | None = None,
) -> dict[str, Any] | None:
    if optimizer is None:
        optimizer = PlanSpendingBudgetVerbalizerOptimizer()

    if isinstance(test_name_or_index_or_dict, dict):
        tc = test_name_or_index_or_dict
        name = tc.get("name", "custom_test")
        try:
            payload = resolve_spending_budget_test_case_input(tc)
        except ValueError as exc:
            print(f"Invalid test dict: {exc}")
            return None
        print(f"\n{'=' * 80}\nRunning test: {name}\n{'=' * 80}\n")
        ideal = tc.get("ideal_response")
        ideal_text = ideal.strip() + "\n" if isinstance(ideal, str) and ideal.strip() else None
        return _run_test(payload, optimizer, ideal=ideal_text)

    tc = get_test_case(test_name_or_index_or_dict)
    if tc is None:
        print(f"Test case '{test_name_or_index_or_dict}' not found.")
        return None
    name = tc["name"]
    print(f"\n{'=' * 80}\nRunning test: {name}\n{'=' * 80}\n")
    ideal = tc.get("ideal_response")
    ideal_text = ideal.strip() + "\n" if isinstance(ideal, str) and ideal.strip() else None
    return _run_test(resolve_spending_budget_test_case_input(tc), optimizer, ideal=ideal_text)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run P:PlanSpendingBudgetVerbalizer optimizer tests")
    parser.add_argument("--test", type=str, help='Test name or index (e.g. "0" or "debt_paydown_recommended")')
    parser.add_argument("--batch", type=int, help="Run all tests in batch N")
    parser.add_argument("--user-id", type=int)
    parser.add_argument("--simulate-agent-outcome-id", type=int)
    parser.add_argument("--scenario-id", type=str)
    parser.add_argument("--print-input-only", action="store_true")
    parser.add_argument("--model", type=str, default=GEMINI_FLASH_LITE)
    parser.add_argument("--no-thinking", action="store_true")
    args = parser.parse_args()

    if args.user_id is not None or args.simulate_agent_outcome_id is not None:
        sim_id = resolve_simulate_agent_outcome_id(
            user_id=args.user_id,
            simulate_agent_outcome_id=args.simulate_agent_outcome_id,
        )
        built = build_plan_spending_budget_verbalizer_input(
            simulate_agent_outcome_id=sim_id,
            scenario_id=args.scenario_id,
        )
        print(f"Using simulate_agent_outcome_id={sim_id}")
        print("BUILT PLAN SPENDING BUDGET VERBALIZER INPUT")
        print("-" * 80)
        print(built)
        if args.print_input_only:
            return
        thinking_budget = 0 if args.no_thinking else PLAN_SPENDING_BUDGET_VERBALIZER_THINKING_BUDGET
        optimizer = PlanSpendingBudgetVerbalizerOptimizer(
            model_name=args.model,
            thinking_budget=thinking_budget,
        )
        print("\nPLAN SPENDING BUDGET VERBALIZER LLM OUTPUT")
        print("-" * 80)
        print(format_plan_spending_budget_verbalizer_markdown(optimizer.generate_response(built)))
        return

    if args.print_input_only:
        print("Error: --print-input-only requires --user-id or --simulate-agent-outcome-id", file=sys.stderr)
        raise SystemExit(1)

    if args.batch is None and args.test is None:
        _print_usage()
        return

    thinking_budget = 0 if args.no_thinking else PLAN_SPENDING_BUDGET_VERBALIZER_THINKING_BUDGET
    optimizer = PlanSpendingBudgetVerbalizerOptimizer(
        model_name=args.model,
        thinking_budget=thinking_budget,
    )

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
        run_test(test_val, optimizer, scenario_id=args.scenario_id)
        return


def _print_usage() -> None:
    print("Usage:")
    print("  Run a single test: --test <name_or_index>")
    print("  Run all tests: --test all")
    print("  Run batch: --batch <N>")
    print("  Build input from DB: --user-id <id> | --simulate-agent-outcome-id <id> [--scenario-id <id>]")
    print("  Print built input only: --user-id <id> --print-input-only")
    print("\nAvailable test cases:")
    for i, tc in enumerate(TEST_CASES):
        print(f"  [{i}] {tc['name']} (batch {tc.get('batch', '?')})")


if __name__ == "__main__":
    main()

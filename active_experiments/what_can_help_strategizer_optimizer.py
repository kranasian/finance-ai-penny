"""
WhatCanHelp strategizer (Gemini): emits ``execute_plan`` using lookup tools + optional ``refine_strategy`` follow-up.

Engine is ``WhatCanHelpEngine``.
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from typing import Any, Callable

from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai.errors import ClientError

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
  sys.path.insert(0, parent_dir)

from penny.strategizer.what_can_help_engine import (
  WhatCanHelpEngine,
  format_lookup_user_turn,
  wch_latest_outcome_is_empty,
  wch_lookup_is_empty,
)

load_dotenv()

LOOKUP_TX_MAX_VISIBLE = 10

# Optimizer ``system_prompt`` — fixed for every WhatCanHelp turn; only user ``prompt_override`` changes.
SYSTEM_PROMPT = f"""You are WhatCanHelpStrategizer. Respond with one ```python``` block that defines only `execute_plan() -> tuple[bool, str]`.

The chat user message matches the host’s structured turn plus optional **HOST** reminders on refine passes.

The host injects `WCH_USER_TURN`, `wch_latest_outcome_is_empty`, and `wch_lookup_is_empty` (deprecated alias) into `execute_plan`’s globals—do not import them. Prefer `wch_latest_outcome_is_empty(WCH_USER_TURN)`.

`WCH_USER_TURN` has no **HOST** lines: `# Latest Outcome` (prefilled and/or tool-aggregated data, or the token `None` when nothing is loaded yet), then optionally `# Previous Outcomes` with numbered prior strategizer outcomes (`1. **Outcome #1**: …`). `wch_latest_outcome_is_empty(WCH_USER_TURN)` is true only when the **# Latest Outcome** body is exactly `None`. Honor `# Previous Outcomes` when present; do not contradict them without evidence from new lookups. Never invent amounts or accounts.

Branch:
- `wch_latest_outcome_is_empty(WCH_USER_TURN)`: call the minimum tools needed, concatenate into one string, `return refine_strategy(aggregated)` with that string (it becomes the next turn’s `# Latest Outcome` body). Do not `return True, ...` with the final strategy here.
- else: `return True, strategy_string` with plain-text sections Urgency:, User vs system:, Goals:, Tactics (code / LLM): — urgency high/medium/low in one line each; 2–5 goals tied to data; per goal name what to re-check (tools/thresholds) vs how a coach should guide the user. No markdown inside the string, no meta narration. No tools, no `refine_strategy`.

`refine_strategy(aggregated)` takes one string: concatenated tool output for the next turn’s `# Latest Outcome`.

Tools already in scope:
- `lookup_accounts()`
- `lookup_spending_forecasts(horizon_months=3)`, `lookup_income_forecasts(horizon_months=3)`
- `lookup_transactions(start, end, name_contains="", amount_larger_than=None, amount_less_than=None, in_category=None)` — official slugs in `in_category`; host may accept `max_visible` (default {LOOKUP_TX_MAX_VISIBLE})
- `lookup_spending_transactions(start, end, name_contains="", in_category=None, max_visible={LOOKUP_TX_MAX_VISIBLE})`, `lookup_income_transactions(start, end, name_contains="", max_visible={LOOKUP_TX_MAX_VISIBLE})`
- `lookup_monthly_spending_by_category(months_back=6)`, `lookup_future_spending_by_category(months_ahead=3)`, `lookup_avg_monthly_spending(category)`

You may use `from datetime import date` for bounds. Keep `execute_plan` ~≤25 lines, no filler comments.
"""


class StrategizerOptimizer:
  """Gemini wrapper for WhatCanHelpStrategizer (`execute_plan`; optional tools + refine_strategy)."""

  def __init__(
    self,
    model_name: str = "gemini-flash-lite-latest",
    thinking_budget: int = 512,
    max_output_tokens: int = 1024,
  ):
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
      raise ValueError(
        "GEMINI_API_KEY environment variable is not set. Please set it in your .env file or environment."
      )
    self.client = genai.Client(api_key=api_key)
    self.model_name = model_name
    self.thinking_budget = thinking_budget
    self.temperature = 0.25
    self.top_p = 0.95
    self.max_output_tokens = max_output_tokens
    self.safety_settings = [
      types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"),
      types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"),
      types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"),
      types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="OFF"),
    ]
    self.system_prompt = SYSTEM_PROMPT
    self._last_lookup_user_turn = ""

  def generate_response(
    self,
    task_description: str,
    insight: str,
    top_transactions_recent_period: str,
    top_transactions_previous_period: str,
    recent_insight_date_range: str = "—",
    previous_insight_date_range: str = "—",
    previous_outcomes: dict[int | str, str] | list[str] | str | None = None,
    prompt_override: str | None = None,
    system_prompt_override: str | None = None,
    print_thought_summary: bool = True,
  ) -> str:
    _ = (
      task_description,
      insight,
      top_transactions_recent_period,
      top_transactions_previous_period,
      recent_insight_date_range,
      previous_insight_date_range,
      previous_outcomes,
    )
    body = prompt_override if prompt_override is not None else ""
    self._last_lookup_user_turn = body
    request_text = types.Part.from_text(text=body)
    contents = [types.Content(role="user", parts=[request_text])]
    system_instruction_text = system_prompt_override if system_prompt_override is not None else self.system_prompt
    generate_content_config = types.GenerateContentConfig(
      temperature=self.temperature,
      top_p=self.top_p,
      max_output_tokens=self.max_output_tokens,
      safety_settings=self.safety_settings,
      system_instruction=[types.Part.from_text(text=system_instruction_text)],
      thinking_config=types.ThinkingConfig(
        thinking_budget=self.thinking_budget,
        include_thoughts=True,
      ),
    )
    output_text = ""
    thought_summary = ""
    try:
      for chunk in self.client.models.generate_content_stream(
        model=self.model_name,
        contents=contents,
        config=generate_content_config,
      ):
        if chunk.text is not None:
          output_text += chunk.text
        if hasattr(chunk, "candidates") and chunk.candidates:
          for candidate in chunk.candidates:
            if hasattr(candidate, "content") and candidate.content:
              if hasattr(candidate.content, "parts") and candidate.content.parts:
                for part in candidate.content.parts:
                  if getattr(part, "thought", False) and getattr(part, "text", None):
                    thought_summary = (thought_summary + part.text) if thought_summary else part.text
    except ClientError as e:
      if self.thinking_budget == 0 and "only works in thinking mode" in (str(e) or ""):
        print("\n[NOTE] This model requires thinking mode; API rejected thinking_budget=0.", flush=True)
        sys.exit(1)
      raise
    if thought_summary and print_thought_summary:
      print("\n## Thought Summary\n")
      print(thought_summary.strip() + "\n")
    return output_text


def extract_python_code(text: str) -> str:
  code_start = text.find("```python")
  if code_start != -1:
    code_start += len("```python")
    code_end = text.find("```", code_start)
    if code_end != -1:
      return text[code_start:code_end].strip()
    return text[code_start:].strip()
  return text.strip()


# --- Dummy tool responses (used by tests; document expected shape for hosts) ---

TOOL_DUMMY_RESPONSES: dict[str, str] = {
  "lookup_accounts": """Accounts (as of 2026-04-07):
- Checking ****4401 | Chase | balance $2,840
- Credit ****9912 | Amex | balance $1,120 (limit $8,000)
- Savings ****2200 | Ally | balance $14,200
""",
  "lookup_spending_forecasts": """Spending forecasts (next 3 months):
- 2026-04: $4,200
- 2026-05: $4,050 
- 2026-06: $4,100
""",
  "lookup_income_forecasts": """Income forecasts (next 3 months):
- 2026-04: $5,100
- 2026-05: $5,100
- 2026-06: $5,050
""",
  "lookup_transactions": """From 2026-03-01 through 2026-03-31 (top by amount, max 10):

- $1,481 at Acme Corp Payroll as income_salary.
- $420 at Whole Foods as meals_groceries.
- $310 at City Property Mgmt as shelter_home.
- $210 at Shell as transportation_car.
- $185 at DoorDash as meals_delivered_food.
- $160 at Netflix as leisure_entertainment.
- $95 at CVS as health_medical_pharmacy.
- $88 at Uber as transportation_car.
- $72 at Thai Garden as meals_dining_out.
- $61 at Spotify as leisure_entertainment.
+22 transactions
""",
  "lookup_spending_transactions": """Spending only, 2026-03-01..2026-03-31 (max 10):

- $420 at Whole Foods as meals_groceries.
- $310 at City Property Mgmt as shelter_home.
- $210 at Shell as transportation_car.
- $185 at DoorDash as meals_delivered_food.
- $160 at Netflix+Hulu as leisure_entertainment.
- $95 at CVS as health_medical_pharmacy.
- $88 at Uber as transportation_car.
- $72 at Thai Garden as meals_dining_out.
- $61 at Spotify as leisure_entertainment.
- $54 at AMC as leisure_entertainment.
+20 spending rows
""",
  "lookup_income_transactions": """Income only, 2026-03-01..2026-03-31 (max 10):

- $1,481 at Acme Corp Payroll as income_salary.
- $300 at Design Gig LLC as income_sidegig.
+0 other income rows
""",
  "lookup_monthly_spending_by_category": """Actual spending by month (category totals, USD):
- 2026-03: meals_groceries $420, shelter_home $310, transportation_car $298, meals_delivered_food $185, leisure_entertainment $221
- 2026-02: meals_groceries $390, shelter_home $310, transportation_car $260, leisure_entertainment $340
- 2026-01: meals_groceries $410, shelter_home $310, transportation_car $305, leisure_entertainment $180
""",
  "lookup_future_spending_by_category": """Forecasted monthly spending by category (next 3 months):
- shelter_home: $310
- meals_groceries: $450
- meals_delivered_food: $120
- transportation_car: $280
""",
  "lookup_avg_monthly_spending": """Rolling 6-mo average for meals_delivered_food: $162/mo (current month $185).
""",
}


def _wrap_tools_for_test(
  engine: WhatCanHelpEngine,
  tool_mocks: dict[str, str],
) -> dict[str, Callable[..., str]]:
  """Return namespace tools that print and return dummy strings."""

  def _make(name: str) -> Callable[..., str]:
    def _fn(*args: Any, **kwargs: Any) -> str:
      out = tool_mocks.get(name, TOOL_DUMMY_RESPONSES.get(name, ""))
      print(f"\n## {name} returned\n")
      print(out)
      print()
      return out

    return _fn

  out: dict[str, Callable[..., str]] = {}
  bindings = engine.tool_bindings
  for key in (
    "lookup_accounts",
    "lookup_spending_forecasts",
    "lookup_income_forecasts",
    "lookup_transactions",
    "lookup_spending_transactions",
    "lookup_income_transactions",
    "lookup_monthly_spending_by_category",
    "lookup_future_spending_by_category",
    "lookup_avg_monthly_spending",
  ):
    _ = bindings[key]
    out[key] = _make(key)
  return out


# Each entry must contain exactly: ``batch``, ``name``, ``tool_mocks``, ``input``, ``output``.
# ``input``: latest-outcome payload ``str | None``; ``None`` or ``""`` → ``# Latest Outcome`` is token ``None``.
TEST_CASES: list[dict[str, Any]] = [
  {
    "batch": 1,
    "name": "wch_first_call_empty_then_tools_and_refine",
    "tool_mocks": dict(TOOL_DUMMY_RESPONSES),
    "input": None,
    "output": (
      "Expected: wch_latest_outcome_is_empty(WCH_USER_TURN); execute_plan calls lookups, builds aggregated string, "
      "`return refine_strategy(aggregated)` (not final True on first pass). Second turn (simulated by host) "
      "would return True with strategy."
    ),
  },
  {
    "batch": 1,
    "name": "wch_first_call_empty_income_gap_mocks",
    "tool_mocks": {
      **TOOL_DUMMY_RESPONSES,
      "lookup_income_forecasts": """Income forecasts (next 3 months):
- 2026-04: $4,200
- 2026-05: $5,100
- 2026-06: $5,100
""",
    },
    "input": None,
    "output": (
      "Expected: first pass aggregates forecasts + accounts (or similar), `refine_strategy(aggregated)`; "
      "follow-up turn returns True with cut-order / savings guidance grounded in lookup text."
    ),
  },
  {
    "batch": 1,
    "name": "wch_prefilled_lookup_final_only",
    "tool_mocks": dict(TOOL_DUMMY_RESPONSES),
    "input": """Accounts (as of 2026-04-07):
- Checking ****4401 | Chase | balance $2,840
- Credit ****9912 | Amex | balance $1,120 (limit $8,000)
- Savings ****2200 | Ally | balance $14,200

Spending forecasts (next 3 months):
- 2026-04: $4,200
- 2026-05: $4,050 
- 2026-06: $4,100

Income forecasts (next 3 months):
- 2026-04: $4,200
- 2026-05: $5,100
- 2026-06: $5,050

Forecasted monthly spending by category (next 3 months):
- shelter_home: $310
- meals_groceries: $450
- meals_delivered_food: $120
- transportation_car: $280
""",
    "output": (
      "Expected: not wch_latest_outcome_is_empty(WCH_USER_TURN); execute_plan returns (True, summary) only—no tools, "
      "no refine_strategy; tight cash flow and savings guidance from prefilled tool-shaped latest-outcome text."
    ),
  },
]


_TEST_CASE_KEYS = frozenset({"batch", "name", "tool_mocks", "input", "output"})


def run_test(test_dict: dict[str, Any], optimizer: StrategizerOptimizer | None = None):
  if set(test_dict.keys()) != _TEST_CASE_KEYS:
    raise ValueError(
      f"test dict must contain exactly keys {sorted(_TEST_CASE_KEYS)}; got {sorted(test_dict.keys())}"
    )
  raw_input = test_dict["input"]
  if raw_input is not None and not isinstance(raw_input, str):
    raise TypeError("test dict 'input' must be str or None.")
  test_name = test_dict["name"]
  print(f"\n# Test: **{test_name}**\n")
  if optimizer is None:
    optimizer = StrategizerOptimizer()
  tool_mocks = test_dict["tool_mocks"]
  engine = WhatCanHelpEngine(optimizer, print_thought_summary=False)
  wrapped = _wrap_tools_for_test(engine, tool_mocks)
  engine.tool_bindings.update(wrapped)

  lookup_payload = raw_input
  prompt_body = format_lookup_user_turn(lookup_payload)
  setattr(optimizer, "_last_lookup_user_turn", prompt_body)
  print("## LLM Input\n")
  print(prompt_body)
  print()
  llm_out = optimizer.generate_response(
    "",
    "",
    "",
    "",
    prompt_override=prompt_body,
  )
  print("## LLM Output:\n")
  print(llm_out)
  print()
  code = extract_python_code(llm_out)
  execution_result = None
  if code:
    try:
      namespace: dict[str, Any] = dict(wrapped)
      namespace["WCH_USER_TURN"] = prompt_body
      namespace["wch_latest_outcome_is_empty"] = wch_latest_outcome_is_empty
      namespace["wch_lookup_is_empty"] = wch_lookup_is_empty
      namespace["refine_strategy"] = engine.sandbox_refinement_callback()
      exec(code, namespace)
      if "execute_plan" in namespace:
        execution_result = namespace["execute_plan"]()
        print("\n## Execution Final Result:\n")
        print("```")
        print(f"  success: {execution_result[0]}")
        print(f"  output: {execution_result[1]}")
        print("```")
    except Exception as e:
      print(f"Error executing generated code: {e!s}")
      print(traceback.format_exc())
  print(f"\n## Expected behavior:\n\n{test_dict['output']}\n")
  return execution_result


def run_all_tests_batch(optimizer: StrategizerOptimizer | None = None, batch_num: int = 1):
  if optimizer is None:
    optimizer = StrategizerOptimizer()
  cases = [tc for tc in TEST_CASES if tc["batch"] == batch_num]
  batch_results = []
  for tc in cases:
    result = run_test(tc, optimizer)
    batch_results.append((tc["name"], result))
  for name, result in batch_results:
    success = result[0] if isinstance(result, tuple) and len(result) > 0 else None
    print(f"- {name}: success={success}")
  return batch_results


def main(
  test: str | None = None,
  run_batch: bool = False,
  batch_num: int = 1,
  no_thinking: bool = False,
  thinking_budget: int | None = None,
  max_output_tokens: int | None = None,
  model: str | None = None,
):
  tb = 0 if no_thinking else (thinking_budget if thinking_budget is not None else 512)
  kw: dict[str, Any] = {"thinking_budget": tb}
  if max_output_tokens is not None:
    kw["max_output_tokens"] = max_output_tokens
  if model is not None:
    kw["model_name"] = model
  optimizer = StrategizerOptimizer(**kw)
  if run_batch:
    run_all_tests_batch(optimizer, batch_num=batch_num)
    return
  if test is not None:
    if test.strip().lower() == "all":
      run_all_tests_batch(optimizer, batch_num=1)
      return
    test_val = int(test) if test.isdigit() else test
    if isinstance(test_val, int):
      tc = TEST_CASES[test_val] if 0 <= test_val < len(TEST_CASES) else None
    else:
      tc = next((t for t in TEST_CASES if t["name"] == test_val), None)
    if tc:
      run_test(tc, optimizer)
    else:
      print("Unknown test:", test)
    return
  print("Available test cases:")
  for i, tc in enumerate(TEST_CASES):
    print(f"  {i}: {tc['name']} (batch {tc['batch']})")
  print("\nDummy responses per tool (keys of TOOL_DUMMY_RESPONSES):")
  for k in TOOL_DUMMY_RESPONSES:
    print(f"  - {k}")


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--test", type=str)
  parser.add_argument("--no-thinking", action="store_true")
  parser.add_argument("--thinking-budget", type=int, default=None)
  parser.add_argument("--max-output-tokens", type=int, default=None)
  parser.add_argument("--model", type=str, default=None)
  parser.add_argument(
    "--batch",
    type=int,
    nargs="?",
    const=1,
    default=None,
    metavar="N",
    help="Run all test cases in batch N (default 1).",
  )
  args = parser.parse_args()
  batch_num = 1 if args.batch is None else args.batch
  run_batch = args.batch is not None
  main(
    test=args.test,
    run_batch=run_batch,
    batch_num=batch_num,
    no_thinking=args.no_thinking,
    thinking_budget=args.thinking_budget,
    max_output_tokens=args.max_output_tokens,
    model=args.model,
  )

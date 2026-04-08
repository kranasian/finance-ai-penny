"""
WhatCanHelp strategizer (Gemini): emits ``execute_plan`` using lookup tools + optional ``refine_strategy`` follow-up.

Engine is ``WhatCanHelpEngine``.
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from datetime import date
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
  latest_outcome_payload,
)

load_dotenv()

LOOKUP_TX_MAX_VISIBLE = 10

# Optimizer ``system_prompt`` — fixed for every WhatCanHelp turn; only user ``prompt_override`` changes.
SYSTEM_PROMPT = f"""You are WhatCanHelpStrategizer. Respond with one ```python``` block that defines only `execute_plan() -> tuple[bool, str]`.

**Input** is the financial snapshot you reason over. Inside `execute_plan` that text is bound as the global `WCH_USER_TURN` (do not import it). The chat user message for a turn is the same merged payload (snapshot plus any appended lookup excerpts). The host always supplies a non-empty snapshot. Never invent amounts or accounts.

**Input** includes:
- Accounts
- Last month spending for the five top-level categories (Food, Others, Bills, Shopping, Income)
- This month current spending for those five
- This month expected spending for those five
- Expected account balances next month

Use matching section headers in the snapshot text, e.g. ``# Accounts``, ``# Last month spending``, ``# This month current spending``, ``# This month expected spending``, ``# Expected account balances next month``.

**Input** may also include **extra labeled excerpts** from **tool results on earlier turns** (e.g. forecasts or transactions). Treat those as factual context alongside the five sections.

**Your job:** From this input, infer what the user most needs help with and **strategize** how to support them (priorities, tradeoffs, what to watch). If the input is **insufficient** to ground that plan—or you need **verification** against ledger/forecast reality—call tool(s), build one factual string `aggregated` (no coaching prose), and **`return refine_strategy(aggregated)`** as the **only** return from `execute_plan` on that pass (it is already a `(bool, str)` from the orchestrator). **Never** return lookup text as the second element of a plain tuple—e.g. **do not** `return False, lookup_income_forecasts(...) + lookup_transactions(...)`; **always** pass combined tool output **into** `refine_strategy(...)`.

If the input is already enough, `return True, strategy_string` with plain-text sections **Urgency:**, **User vs system:**, **Goals:**, **Tactics:** — urgency high/medium/low in one line each; 2–5 goals tied to data; **Tactics:** as numbered items. **Do not** promise ledger or forecast re-checks in **Tactics** unless those facts are **already** in `WCH_USER_TURN` or you **just** retrieved them via tools on this same turn. If a tactic would need income/spending **forecasts** or **transaction rows** you do not yet have, call the right tool(s) and **`return refine_strategy(aggregated)`** instead of `return True, …`. **Coach-only** tactics (employment, intent, habits) are fine without tools; anything that depends on **amounts, categories, or payees in the ledger** needs tools first when missing. No markdown inside the string, no meta narration. When returning `True`, use **no** tools and **no** `refine_strategy`.

**Efficiency:** Reach a grounded plan in the **fewest** turns. When you must refine, batch **all** still-missing facts into **one** `aggregated` string for that turn (e.g. combine forecast + transaction pulls with clear labels)—avoid a chain of narrow refines. **`return True, strategy_string`** as soon as snapshot plus merged excerpts are enough for concrete **Tactics**; **do not** repeat lookups whose rows are already in `WCH_USER_TURN`. **Do not** call `lookup_transactions` again with the **same** date window and **`in_category`** (or a subset that adds nothing new) if that block is **already** present in merged text—reuse it and only fetch genuinely **missing** slices (e.g. forecasts if not yet appended).

**Tool coverage:** `lookup_income_forecasts` / `lookup_spending_forecasts` give **monthly aggregates**; `lookup_transactions` gives **line-level** history (date range, optional category filter). Together they are enough for numeric verification implied by the snapshot (e.g. top-level **Income** vs payroll lines in `lookup_transactions`). They do **not** replace asking the user subjective questions—put those under coach-style tactics after data is clear.

**Where users can improve (dig deeper):** Do not stop at “Food is high” or “Shopping is high.” Use the snapshot to spot **outliers** (vs income, vs other categories, or vs expected). When a category warrants it and you lack line detail, **`return refine_strategy`** with targeted pulls—e.g. **Food:** compare dining out vs delivery vs groceries via separate or combined `in_category` lists of official slugs (e.g. ``["meals_dining_out"]``, ``["meals_delivered_food"]``, ``["meals_groceries"]``) over the same months as the snapshot, or one broad `lookup_transactions` window and read merchant lines. **Shopping:** pull transactions filtered by shopping-related slugs or scan top merchants if unfiltered. **Bills / Others:** look for large recurring payees. After you have rows, **Goals** and **Tactics** should name **specific levers** (merchants, sub-categories, frequency) the user can change—not only generic “spend less.” If one refine already added transaction excerpts, mine them in `execute_plan` before asking for more.

`refine_strategy(aggregated)` supplies new lookup text; the **next** user turn repeats the **original** snapshot and appends that text (cumulative on further refines). The orchestrator may cap further `refine_strategy` rounds—treat each refine as expensive and exit with **`return True, …`** once information is sufficient.

<OFFICIAL_CATEGORIES>
income: income_salary, income_sidegig, income_business, income_interest
meals: meals_groceries, meals_dining_out, meals_delivered_food
leisure: leisure_entertainment, leisure_travel
bills: bills_connectivity, bills_insurance, bills_tax, bills_service_fees
shelter: shelter_home, shelter_utilities, shelter_upkeep
education: education_kids_activities, education_tuition
shopping: shopping_clothing, shopping_gadgets, shopping_kids, shopping_pets
transportation: transportation_public, transportation_car
health: health_medical_pharmacy, health_gym_wellness, health_personal_care
donations_gifts, uncategorized, transfers, miscellaneous
</OFFICIAL_CATEGORIES>

**Tools available (only these):**
- `lookup_spending_forecasts(horizon_months=3)`, `lookup_income_forecasts(horizon_months=3)`
- `lookup_transactions(start, end, name_contains="", amount_larger_than=None, amount_less_than=None, in_category=None)` — `in_category` is a **list** of official slugs from `<OFFICIAL_CATEGORIES>` (or `None`); optional `max_visible` (default {LOOKUP_TX_MAX_VISIBLE})

You may use `from datetime import date` for bounds. Keep `execute_plan` ~≤25 lines, no filler comments.
"""


class StrategizerOptimizer:
  """Gemini wrapper for WhatCanHelpStrategizer (`execute_plan`; optional tools + refine_strategy)."""

  def __init__(
    self,
    model_name: str = "gemini-flash-lite-latest",
    thinking_budget: int = 512,
    max_output_tokens: int = 2048,
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
# Filtered ``lookup_transactions`` uses ``_MOCK_TX_ROWS`` so ``in_category`` / name / amount filters match row slugs.

_MOCK_TX_ROWS: list[tuple[int, str, str]] = [
  (1481, "Acme Corp Payroll", "income_salary"),
  (420, "Whole Foods", "meals_groceries"),
  (400, "BrightLink Fiber", "bills_connectivity"),
  (310, "City Property Mgmt", "shelter_home"),
  (300, "Design Gig LLC", "income_sidegig"),
  (220, "State Estimated Tax", "bills_tax"),
  (210, "Shell", "transportation_car"),
  (185, "DoorDash", "meals_delivered_food"),
  (175, "HOA Management", "bills_service_fees"),
  (160, "Netflix", "leisure_entertainment"),
  (160, "State Farm", "bills_insurance"),
  (120, "Comcast", "bills_connectivity"),
  (99, "App Store Fees", "bills_service_fees"),
  (95, "CVS", "health_medical_pharmacy"),
  (88, "Uber", "transportation_car"),
  (72, "Thai Garden", "meals_dining_out"),
  (61, "Spotify", "leisure_entertainment"),
]


def _coerce_date_wch(val: Any) -> date | None:
  if val is None:
    return None
  if isinstance(val, date):
    return val
  if isinstance(val, str):
    try:
      return date.fromisoformat(val.strip()[:10])
    except ValueError:
      return None
  return None


def _lookup_tx_kwargs_from_call(*args: Any, **kwargs: Any) -> dict[str, Any]:
  out = dict(kwargs)
  if len(args) >= 1 and "start" not in out:
    out["start"] = args[0]
  if len(args) >= 2 and "end" not in out:
    out["end"] = args[1]
  return out


def _mock_lookup_transactions_filtered(kw: dict[str, Any]) -> str:
  s = _coerce_date_wch(kw.get("start")) or date(2026, 3, 1)
  e = _coerce_date_wch(kw.get("end")) or date(2026, 3, 31)
  in_category = kw.get("in_category")
  cats = frozenset(in_category) if in_category else None
  name_contains = (kw.get("name_contains") or "").strip().lower()
  lo = kw.get("amount_larger_than")
  hi = kw.get("amount_less_than")
  max_vis = int(kw.get("max_visible") or LOOKUP_TX_MAX_VISIBLE)

  rows = list(_MOCK_TX_ROWS)
  if cats:
    rows = [r for r in rows if r[2] in cats]
  if name_contains:
    rows = [r for r in rows if name_contains in r[1].lower()]
  if lo is not None:
    rows = [r for r in rows if r[0] >= int(lo)]
  if hi is not None:
    rows = [r for r in rows if r[0] <= int(hi)]
  rows.sort(key=lambda r: r[0], reverse=True)
  visible = rows[:max_vis]
  remainder = max(0, len(rows) - len(visible))

  lines = [
    f"From {s.isoformat()} through {e.isoformat()} (top by amount, max {max_vis}):",
    "",
  ]
  if not visible:
    lines.append("(no rows in this window for the given filters)")
  else:
    for amt, merch, slug in visible:
      lines.append(f"- ${amt:,} at {merch} as {slug}.")
  lines.append(f"+{remainder} transactions")
  return "\n".join(lines) + "\n"


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
- 2026-04: $1,481
- 2026-05: $1,481
- 2026-06: $1,481
""",
  "lookup_transactions": """From 2026-03-01 through 2026-03-31 (top by amount, max 10):

- $1,481 at Acme Corp Payroll as income_salary.
- $420 at Whole Foods as meals_groceries.
- $400 at BrightLink Fiber as bills_connectivity.
- $310 at City Property Mgmt as shelter_home.
- $300 at Design Gig LLC as income_sidegig.
- $220 at State Estimated Tax as bills_tax.
- $210 at Shell as transportation_car.
- $185 at DoorDash as meals_delivered_food.
- $175 at HOA Management as bills_service_fees.
- $160 at Netflix as leisure_entertainment.
+7 transactions
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
  """Return namespace tools that return dummy strings (no console logging)."""

  def _make(name: str) -> Callable[..., str]:
    def _fn(*args: Any, **kwargs: Any) -> str:
      return tool_mocks.get(name, TOOL_DUMMY_RESPONSES.get(name, ""))

    return _fn

  static_tx = tool_mocks.get("lookup_transactions", TOOL_DUMMY_RESPONSES["lookup_transactions"])

  def _lookup_transactions_mock(*args: Any, **kwargs: Any) -> str:
    kw = _lookup_tx_kwargs_from_call(*args, **kwargs)
    ic = kw.get("in_category")
    use_filtered = bool(
      ic
      or (kw.get("name_contains") or "").strip()
      or kw.get("amount_larger_than") is not None
      or kw.get("amount_less_than") is not None
    )
    if use_filtered:
      kw.setdefault("max_visible", LOOKUP_TX_MAX_VISIBLE)
      return _mock_lookup_transactions_filtered(kw)
    return static_tx

  out: dict[str, Callable[..., str]] = {}
  bindings = engine.tool_bindings
  for key in (
    "lookup_spending_forecasts",
    "lookup_income_forecasts",
  ):
    _ = bindings[key]
    out[key] = _make(key)
  _ = bindings["lookup_transactions"]
  out["lookup_transactions"] = _lookup_transactions_mock
  return out


# Each entry must contain exactly: ``batch``, ``name``, ``tool_mocks``, ``input``, ``output``.
# ``input``: top-body payload ``str | None``; prefer prefilled five-block input (literal str per case).
TEST_CASES: list[dict[str, Any]] = [
  {
    "batch": 1,
    "name": "wch_first_call_empty_then_tools_and_refine",
    "tool_mocks": dict(TOOL_DUMMY_RESPONSES),
    "input": """# Accounts (as of 2026-04-07):
- Chase Checking (checking): $2,840
- Amex Credit (credit card): $1,120
- Ally Savings (savings): $14,200

# Last month spending
- 2026-03: Food $1,240  Others $560  Bills $1,980  Shopping $720  Income $1,481

# This month current spending
- 2026-04: Food $810  Others $180  Bills $1,240  Shopping $390  Income $1,481

# This month expected spending
- 2026-04: Food $1,260  Others $520  Bills $1,980  Shopping $700  Income $1,481

# Expected account balances next month
- Remaining expected spending this month: $2,030
- Chase Checking (checking): $810
- Amex Credit (credit card): -$910
- Ally Savings (savings): $12,170
""",
    "output": (
      "multiple turns: lookups then `(False, refine_strategy(aggregated))`. last turn: `(True, strategy)`."
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
    "input": """# Accounts (as of 2026-04-07):
- Chase Checking (checking): $2,840
- Amex Credit (credit card): $1,120
- Ally Savings (savings): $14,200

# Last month spending
- 2026-03: Food $1,240  Others $560  Bills $1,980  Shopping $720  Income $1,481

# This month current spending
- 2026-04: Food $810  Others $180  Bills $1,240  Shopping $390  Income $1,481

# This month expected spending
- 2026-04: Food $1,260  Others $520  Bills $1,980  Shopping $700  Income $1,481

# Expected account balances next month
- Remaining expected spending this month: $2,030
- Chase Checking (checking): $810
- Amex Credit (credit card): -$910
- Ally Savings (savings): $12,170
""",
    "output": (
       "may call lookups in multiple turns calling (False, refine_strategy(aggregated)), then last turn: `(True, strategy)`."
    ),
  },
  {
    "batch": 1,
    "name": "wch_prefilled_lookup_final_only",
    "tool_mocks": dict(TOOL_DUMMY_RESPONSES),
    "input": """# Accounts (as of 2026-04-07):
- Chase Checking (checking): $2,840
- Amex Credit (credit card): $1,120
- Ally Savings (savings): $14,200

# Last month spending
- 2026-03: Food $1,240  Others $560  Bills $1,980  Shopping $720  Income $1,481

# This month current spending
- 2026-04: Food $810  Others $180  Bills $1,240  Shopping $390  Income $1,481

# This month expected spending
- 2026-04: Food $1,260  Others $520  Bills $1,980  Shopping $700  Income $1,481

# Expected account balances next month
- Remaining expected spending this month: $2,030
- Chase Checking (checking): $810
- Amex Credit (credit card): -$910
- Ally Savings (savings): $12,170
""",
    "output": (
      "may call lookups in multiple turns calling (False, refine_strategy(aggregated)), then last turn: `(True, strategy)`."
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
  engine._wch_snapshot_for_merge = latest_outcome_payload(lookup_payload).strip()
  setattr(optimizer, "_last_lookup_user_turn", prompt_body)

  _orig_generate_response = optimizer.generate_response
  _llm_call = 0

  def _trace_generate_response(*args: Any, **kwargs: Any) -> str:
    nonlocal _llm_call
    _llm_call += 1
    prompt = kwargs.get("prompt_override")
    if prompt is None:
      prompt = ""
    print(f"## LLM Input (call {_llm_call})\n")
    print(prompt)
    print()
    out = _orig_generate_response(*args, **kwargs)
    print(f"## LLM Output (call {_llm_call}):\n")
    print(out)
    print()
    return out

  optimizer.generate_response = _trace_generate_response
  execution_result = None
  try:
    llm_out = optimizer.generate_response(
      "",
      "",
      "",
      "",
      prompt_override=prompt_body,
      print_thought_summary=False,
    )
    code = extract_python_code(llm_out)
    if code:
      try:
        namespace: dict[str, Any] = dict(wrapped)
        namespace["WCH_USER_TURN"] = prompt_body
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
  finally:
    optimizer.generate_response = _orig_generate_response

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

from __future__ import annotations

from google import genai
from google.genai import types
from google.genai.errors import ClientError
import os
import sys
from typing import Any, Optional, Tuple
from dotenv import load_dotenv

# Add the parent directory to the path so we can import the tools
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
  sys.path.insert(0, parent_dir)

import categories
from penny.strategizer.rationalize_change_engine import RationalizeChangeEngine
from penny.tool_funcs.lookup_transactions import lookup_transactions

# Load environment variables
load_dotenv()

# Defaults: `gemini-flash-lite-latest` + thinking. Cheapest configs that passed batch-1 in this file: 256/450; 384/512 works with the final prompt.

RATIONALIZE_CHANGE_MAX_VISIBLE_TRANSACTIONS = 10

SYSTEM_PROMPT = f"""You are **RationalizeChange**. Reply with **one** ```python``` block that defines only `execute_plan() -> tuple[bool, str]`.

**0. Read the user message first.** If it already contains **`# Supplemental lookup`**, your **entire** `execute_plan` is only `return True, "…"` (≤3 sentences). Use **# Insight**, **# Expected …**, both Top Transactions sections, **and** the supplemental block (merchants/amounts there are authoritative). **Forbidden** on that turn: `lookup_transactions`, `rationalize`, or any other call.

**Code:** No `#` comments, no docstrings. At most `from datetime import date`. Keep `execute_plan` ≤12 lines. `lookup_transactions` and `rationalize` are in scope—never import them.

**First turn only** (no `# Supplemental lookup` in the message): Either `return True, "…"` if excerpts justify the Insight amounts, **or** assign `lookup_info = lookup_transactions(…)` then **`return rationalize(USER_MESSAGE, lookup_info)`**. **Never** `return lookup_transactions(...)` alone—`execute_plan` always returns `(bool, str)`.
- **`rationalize` first argument:** the identifier **`USER_MESSAGE`** only. **Never** a string literal (drops sections and breaks the pipeline).

**When to answer directly:** The **# Expected …** block (header matches the task period), **Insight**, **Insight Period** lines, and **Previous Period** lines already let you explain the Insight vs target—e.g. Insight-period shows merchants/amounts and **`- Total: $…`** that align with the Insight. **Do not** call `lookup_transactions` just to “confirm” when those excerpts are present and consistent. Map loose labels to official slugs. Optional: **# Previous 2 Period Spending…** or **# Previous 2 Period Earning…** (after **# Previous Period Top Transactions**; income tree uses *Earning*), **## Previous Outcomes**.

**When to lookup:** Insight states an amount for a category but **Insight Period** is empty (**`No matching transactions`**) or obviously cannot explain the stated total (e.g. `+K more` without enough detail). Use `start`/`end` as `date` from the **Insight Period** header; `in_category` = relevant official slugs only.

**Benchmark:** Follow **# Task Description** (spend vs spent, forecast vs goal). **# Previous Period Top Transactions** is context, not the benchmark.

**Return string:** ≤3 short sentences, plain English, no markdown inside the string. Tie amounts to the right **# Expected …** line (parent rollup vs child category). Name merchants from excerpts or supplemental lookup. Do not invent forecast/goal figures. Do not contradict Insight totals; if a list ends with `+K more totaling $…`, consider lookup on the first turn only.

**lookup_transactions:** Required `start`, `end` (inclusive `date`). Optional `name_contains`, `amount_larger_than`, `amount_less_than`, `in_category`, `max_visible` (default {RATIONALIZE_CHANGE_MAX_VISIBLE_TRANSACTIONS}). Output lines: `- $N at Merchant as slug.`; partial list → `+K more totaling $…`; full list → `- Total: $…`.

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
"""


class StrategizerOptimizer:
  """Gemini API wrapper for RationalizeChange (`execute_plan`; optional lookup + rationalize).

  **Defaults** (`gemini-flash-lite-latest`, thinking 512, max_output 600)—best batch-1 quality in this experiment; use CLI flags to try lower budgets.
  """

  def __init__(self, model_name="gemini-flash-lite-latest", thinking_budget=512, max_output_tokens=600):
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
      raise ValueError("GEMINI_API_KEY environment variable is not set. Please set it in your .env file or environment.")
    self.client = genai.Client(api_key=api_key)
    self.model_name = model_name
    self.thinking_budget = thinking_budget
    self.temperature = 0.2
    self.top_p = 0.95
    self.max_output_tokens = max_output_tokens

    self.safety_settings = [
      types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"),
      types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"),
      types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"),
      types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="OFF")
    ]

    self.system_prompt = SYSTEM_PROMPT
    self._last_formatted_user_message = ""

  def generate_response(
    self,
    task_description: str,
    insight: str,
    top_transactions_recent_period: str,
    top_transactions_previous_period: str,
    recent_insight_date_range: str = "—",
    previous_insight_date_range: str = "—",
    previous_outcomes: dict[int | str, str] | list[str] | str | None = None,
    previous_two_periods_category_spending: str | None = None,
    previous_two_periods_date_ranges: str | None = None,
    expected_category_amounts: str | None = None,
    category_id: int | None = None,
    prompt_override: str | None = None,
    system_prompt_override: str | None = None,
    print_thought_summary: bool = True,
  ) -> str:
    if prompt_override is not None:
      body = prompt_override
    else:
      body = _format_rationalize_change_prompt(
        task_description,
        insight,
        top_transactions_recent_period,
        top_transactions_previous_period,
        recent_insight_date_range=recent_insight_date_range,
        previous_insight_date_range=previous_insight_date_range,
        previous_outcomes=previous_outcomes,
        previous_two_periods_category_spending=previous_two_periods_category_spending,
        previous_two_periods_date_ranges=previous_two_periods_date_ranges,
        expected_category_amounts=expected_category_amounts,
        category_id=category_id,
      )
    self._last_formatted_user_message = body
    request_text = types.Part.from_text(text=body)

    contents = [types.Content(role="user", parts=[request_text])]

    system_instruction_text = (
      system_prompt_override if system_prompt_override is not None else self.system_prompt
    )

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

    output_text: str = ""
    thought_summary: str = ""
    last_chunk = None
    try:
      for chunk in self.client.models.generate_content_stream(
        model=self.model_name,
        contents=contents,
        config=generate_content_config,
      ):
        last_chunk = chunk
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


def generate_rationalization_text(
  input_info: str,
  lookup_info: str,
  *,
  model_name: str | None = None,
  thinking_budget: int | None = None,
  max_output_tokens: int | None = None,
) -> tuple[bool, str]:
  """Same strategizer template as the first turn (``SYSTEM_PROMPT``), with supplemental lookup in the user message."""
  opt_kw: dict = {}
  if model_name is not None:
    opt_kw["model_name"] = model_name
  if thinking_budget is not None:
    opt_kw["thinking_budget"] = thinking_budget
  if max_output_tokens is not None:
    opt_kw["max_output_tokens"] = max_output_tokens
  optimizer = StrategizerOptimizer(**opt_kw)
  engine = RationalizeChangeEngine(optimizer, print_thought_summary=False)
  return engine.run_followup_turn(input_info, lookup_info)


# Task Description: period + benchmark + as_of + imperative (see SYSTEM_PROMPT **Inputs**). Long instructions live in system.
RATIONALIZE_SPEND_FAMILY_INSIGHT_TYPES: Tuple[str, ...] = (
  "month_spend_vs_forecast",
  "week_spend_vs_forecast",
  "month_spend_vs_goal",
  "week_spend_vs_goal",
  "month_spent_vs_forecast",
  "week_spent_vs_forecast",
  "month_spent_vs_goal",
  "week_spent_vs_goal",
)


def rationalize_task_period_label_from_insight_type(insight_type: str) -> str:
  """**Month-to-date** vs **Week-to-date** from ``month_*`` / ``week_*`` on ``insight_type``."""
  t = (insight_type or "").strip().lower()
  if t.startswith("week_"):
    return "Week-to-date"
  if t.startswith("month_"):
    return "Month-to-date"
  raise ValueError(f"Cannot infer period label from insight_type={insight_type!r}")


def rationalize_task_benchmark_phrases(insight_type: Optional[str]) -> Tuple[str, str]:
  """``(amount_phrase, target_noun)`` for the task line (spend vs spent; forecast vs goal). Defaults: spend + forecast."""
  if not insight_type:
    return "actual spend", "forecast"
  tl = insight_type.strip().lower()
  amount = "amount spent" if "spent" in tl else "actual spend"
  target = "goal" if "goal" in tl else "forecast"
  return amount, target


RATIONALIZE_TASK_DESCRIPTION_TEMPLATE = (
  "**{period_label}** {amount_phrase} vs {target_noun}, **as of {as_of}**. "
  "Explain why the Insight holds versus that {target_noun} using the matching **# Expected …** section, the Insight, "
  "and the Insight-period and previous-period transaction excerpts; call **lookup_transactions** only if those sources "
  "cannot justify the Insight’s stated amounts."
)

RATIONALIZE_TASK_DESCRIPTION_BY_INSIGHT_TYPE: dict[str, str] = {}
for _it in RATIONALIZE_SPEND_FAMILY_INSIGHT_TYPES:
  _amt, _tgt = rationalize_task_benchmark_phrases(_it)
  RATIONALIZE_TASK_DESCRIPTION_BY_INSIGHT_TYPE[_it] = RATIONALIZE_TASK_DESCRIPTION_TEMPLATE.format(
    period_label=rationalize_task_period_label_from_insight_type(_it),
    amount_phrase=_amt,
    target_noun=_tgt,
    as_of="{as_of}",
  )


def rationalize_task_description_for_insight_type(
  insight_type: str,
  *,
  as_of: str = "2026/03/31",
  period_label: str | None = None,
) -> str:
  """Short task for ``insight_type``. ``as_of`` is ``YYYY/MM/DD`` (recent window end in headers).

  Pass ``period_label`` to override **Week-to-date** / **Month-to-date** (e.g. from ``combined_from`` / metadata).
  Full strategizer rules are in ``SYSTEM_PROMPT``, not here.
  """
  if period_label is None:
    try:
      tpl = RATIONALIZE_TASK_DESCRIPTION_BY_INSIGHT_TYPE[insight_type]
    except KeyError as e:
      raise KeyError(
        f"No task description for insight_type={insight_type!r}; extend RATIONALIZE_SPEND_FAMILY_INSIGHT_TYPES."
      ) from e
    return tpl.format(as_of=as_of)
  amount_phrase, target_noun = rationalize_task_benchmark_phrases(insight_type)
  return RATIONALIZE_TASK_DESCRIPTION_TEMPLATE.format(
    period_label=period_label,
    as_of=as_of,
    amount_phrase=amount_phrase,
    target_noun=target_noun,
  )


def rationalize_task_description_for_period(
  *,
  period_label: str,
  as_of: str,
  insight_type: Optional[str] = None,
) -> str:
  """Same task line as production try-script: dominant period + benchmark from row ``insight_type`` when set."""
  amount_phrase, target_noun = rationalize_task_benchmark_phrases(insight_type)
  return RATIONALIZE_TASK_DESCRIPTION_TEMPLATE.format(
    period_label=period_label,
    as_of=as_of,
    amount_phrase=amount_phrase,
    target_noun=target_noun,
  )


def _format_previous_outcomes(previous_outcomes: dict[int | str, str] | list[str] | str | None) -> str:
  if previous_outcomes is None:
    return ""
  if isinstance(previous_outcomes, str):
    return f"1. **Outcome #1**: {previous_outcomes}"
  if isinstance(previous_outcomes, dict):
    numbered_rows = []
    for key in sorted(previous_outcomes, key=lambda x: int(x) if str(x).isdigit() else str(x)):
      value = previous_outcomes[key]
      if isinstance(value, str) and value.strip():
        label_num = int(key) if str(key).isdigit() else key
        numbered_rows.append(f"{label_num}. **Outcome #{label_num}**: {value.strip()}")
    if numbered_rows:
      return "\n".join(numbered_rows)
    return ""
  clean_outcomes = [item.strip() for item in previous_outcomes if isinstance(item, str) and item.strip()]
  if not clean_outcomes:
    return ""
  return "\n".join(
    f"{idx}. **Outcome #{idx}**: {outcome}" for idx, outcome in enumerate(clean_outcomes, start=1)
  )


def _expected_section_heading_from_task(task_description: str) -> str:
  t = task_description or ""
  if "**Week-to-date**" in t:
    return "# Expected Week-to-date"
  return "# Expected Month-to-date"


def _previous_two_periods_section_title(*, category_id: int | None) -> str:
  if category_id is not None and int(category_id) in categories.INCOME_CATEGORY_IDS:
    return "# Previous 2 Period Earning per category and period"
  return "# Previous 2 Period Spending per category and period"


def _format_rationalize_change_prompt(
  task_description: str,
  insight: str,
  top_transactions_recent_period: str,
  top_transactions_previous_period: str,
  *,
  recent_insight_date_range: str = "—",
  previous_insight_date_range: str = "—",
  previous_outcomes: dict[int | str, str] | list[str] | str | None = None,
  previous_two_periods_category_spending: str | None = None,
  previous_two_periods_date_ranges: str | None = None,
  expected_category_amounts: str | None = None,
  category_id: int | None = None,
) -> str:
  previous_outcomes_block = _format_previous_outcomes(previous_outcomes)
  expected_body = (expected_category_amounts or "").strip() or "—"
  expected_title = _expected_section_heading_from_task(task_description)
  expected_block = f"\n\n{expected_title}\n\n{expected_body}\n"
  extra_periods = (previous_two_periods_category_spending or "").strip()
  ranges = (previous_two_periods_date_ranges or "").strip()
  if extra_periods:
    title = _previous_two_periods_section_title(category_id=category_id)
    if ranges:
      title = f"{title} ({ranges})"
    prev_two_block = f"\n\n{title}\n\n{extra_periods}\n"
  else:
    prev_two_block = ""
  body = f"""# Task Description

{task_description}

# Insight

{insight}{expected_block}
# Insight Period Top Transactions (from {recent_insight_date_range})

{top_transactions_recent_period}

# Previous Period Top Transactions (from {previous_insight_date_range})

{top_transactions_previous_period}{prev_two_block}"""
  if previous_outcomes_block.strip():
    body += f"""

## Previous Outcomes

{previous_outcomes_block}"""
  return body


TEST_CASES = [
  {
    "batch": 1,
    "name": "rationalize_sufficient_context_shopping_month_to_date",
    "mock_lookup_transactions": """From 2026-04-01 through 2026-04-30, with categories shopping_clothing, shopping_gadgets, and uncategorized.

Transactions:

No matching transactions.
""",
    "input": """# Task Description

**Month-to-date** actual spend vs forecast, **as of 2026/04/30**. Explain why the Insight holds versus that forecast using the matching **# Expected …** section, the Insight, and the Insight-period and previous-period transaction excerpts; call **lookup_transactions** only if those sources cannot justify the Insight’s stated amounts.

# Insight

Clothing is significantly down this month at $27.  Shopping is thus significantly down this month to $35.

# Expected Month-to-date

- uncategorized: $1307
- shopping_clothing: $686

# Insight Period Top Transactions (from 2026-04-01 to 2026-04-30)

- $27 at H&M as shopping_clothing.
- $8 at Best Buy as shopping_gadgets.
- Total: $35

# Previous Period Top Transactions (from 2026-03-01 to 2026-03-31)

- $142 at Nike as shopping_clothing.
- $141 at Nike as shopping_clothing.
- $76 at Dick's Sporting Goods as shopping_clothing.
- $71 at Dick's Sporting Goods as shopping_clothing.
- $37 at Kohl's as shopping_clothing.
- $35 at Kohl's as shopping_clothing.
- $33 at H&M as shopping_clothing.
- $27 at H&M as shopping_clothing.
- Total: $562

""",
    "output": (
      "Expected: `execute_plan` → `(True, str)` with no `lookup_transactions` / no `rationalize` (excerpts enough). "
      "Return text (≤3 sentences, plain English): treat **forecast** as `# Expected Month-to-date`—clothing actual **$27** vs **$686** "
      "forecast; acknowledge shopping total **$35** matches the excerpt (**$27** H&M clothing + **$8** Best Buy gadgets, "
      "**Total: $35**). Explain **why** April MTD is light vs forecast using **specific** bullets (sparse apparel vs March "
      "Nike / Dick's / Kohl's lines). Do not contradict the Insight dollar amounts; previous-period section is context, "
      "not the benchmark."
    ),
  },
  {
    "batch": 1,
    "name": "rationalize_groceries_missing_from_excerpt_needs_lookup",
    "mock_lookup_transactions": """From 2026-03-01 through 2026-03-31, with categories meals_groceries.

Transactions:

- $412 at Costco Wholesale as meals_groceries.
- $398 at Costco Wholesale as meals_groceries.
- $52 at Whole Foods as meals_groceries.
- $48 at Trader Joe's as meals_groceries.
- Total: $910

""",
    "input": """# Task Description

**Month-to-date** actual spend vs forecast, **as of 2026/03/31**. Explain why the Insight holds versus that forecast using the matching **# Expected …** section, the Insight, and the Insight-period and previous-period transaction excerpts; call **lookup_transactions** only if those sources cannot justify the Insight’s stated amounts.

# Insight

Groceries is significantly up this month at $910.  Dining Out is slightly up this month at $0.  Delivered Food is slightly up this month at $0.  Food is thus significantly up this month to $910.

# Expected Month-to-date

- meals_groceries: $320

# Insight Period Top Transactions (from 2026-03-01 to 2026-03-31)

No matching transactions.

# Previous Period Top Transactions (from 2026-02-01 to 2026-02-28)

- $55 at Trader Joe's as meals_groceries.
- $52 at Whole Foods as meals_groceries.
- $48 at Safeway as meals_groceries.
- Total: $155

""",
    "output": (
      "Expected: `execute_plan` calls `lookup_transactions` (March MTD from headers; `in_category` includes "
      "`meals_groceries`) because **Insight Period** has **no grocery lines** (`No matching transactions`) while the "
      "Insight still states **$910** on groceries—there is nothing in the excerpt to anchor **$910** vs **$320** "
      "forecast to specific spend. Then **`return rationalize(USER_MESSAGE, lookup_info)`**. Follow-up turn may run; "
      "final text **≤3 sentences**, use mock lookup (Costco-heavy lines totaling **$910**) for concrete drivers. Do "
      "**not** `return True, \"…\"` on the first hop without lookup here."
    ),
  },
  {
    "batch": 1,
    "name": "rationalize_leisure_down_sufficient_context",
    "mock_lookup_transactions": """From 2026-03-01 through 2026-03-31, with categories leisure_entertainment and leisure_travel.

Transactions:

- $150 at Concert Hall as leisure_entertainment.
- $100 at AMC Theaters as leisure_entertainment.
- $59 at Annual Pass as leisure_entertainment.
- $47 at Regional Air as leisure_travel.
- Total: $356
""",
    "input": """# Task Description

**Month-to-date** actual spend vs forecast, **as of 2026/03/31**. Explain why the Insight holds versus that forecast using the matching **# Expected …** section, the Insight, and the Insight-period and previous-period transaction excerpts; call **lookup_transactions** only if those sources cannot justify the Insight’s stated amounts.

# Insight

Entertainment is significantly below forecast this month at $309. Travel & Vacations is significantly below forecast this month at $47. Leisure is thus significantly below forecast this month to $356.

# Expected Month-to-date

- leisure: $660
- leisure_entertainment: $520
- leisure_travel: $140

# Insight Period Top Transactions (from 2026-03-01 to 2026-03-31)

- $150 at Concert Hall as leisure_entertainment.
- $100 at AMC Theaters as leisure_entertainment.
- $59 at Annual Pass as leisure_entertainment.
- $47 at Regional Air as leisure_travel.
- Total: $356

# Previous Period Top Transactions (from 2026-02-01 to 2026-02-28)

- $890 at Marriott as leisure_travel.
- $412 at Delta Air as leisure_travel.
- $220 at StubHub as leisure_entertainment.
- $180 at Concert Hall as leisure_entertainment.
- $64 at AMC Theaters as leisure_entertainment.
- $16 at Netflix as leisure_entertainment.
- $11 at Spotify as leisure_entertainment.
+24 more totaling $1205

""",
    "output": (
      "Expected: `execute_plan` → `(True, str)` without lookup—both transaction sections are **leisure_entertainment** / "
      "**leisure_travel** only (same scope as lookup). Reconcile **$309** / **$47** / **$356** vs **# Expected Month-to-date**; "
      "prior period shows heavier leisure_travel + leisure_entertainment vs March’s lighter mix. If lookup runs anyway, "
      "supplemental repeats the same four lines and **Total: $356**."
    ),
  },
  {
    "batch": 1,
    "name": "rationalize_shopping_up_requires_lookup",
    "mock_lookup_transactions": """From 2026-03-01 through 2026-03-31, with categories shopping_clothing, shopping_gadgets, shopping_kids, and shopping_pets.

Transactions:

- $46 at Amazon as shopping_clothing.
- $46 at Target as shopping_gadgets.
- $46 at Best Buy as shopping_kids.
- $46 at Costco as shopping_pets.
- $46 at Walmart as shopping_clothing.
- $46 at Etsy as shopping_gadgets.
- $46 at Apple Store as shopping_kids.
- $46 at Nordstrom as shopping_pets.
- $46 at Home Depot as shopping_clothing.
- $46 at Kohl's as shopping_gadgets.
+10 transactions
Total: $920
""",
    "input": """# Task Description

**Month-to-date** actual spend vs forecast, **as of 2026/03/31**. Explain why the Insight holds versus that forecast using the matching **# Expected …** section, the Insight, and the Insight-period and previous-period transaction excerpts; call **lookup_transactions** only if those sources cannot justify the Insight’s stated amounts.

# Insight

Shopping is significantly above forecast this month at $920 vs ~$200 expected for shopping.

# Expected Month-to-date

- shopping_clothing: $50
- shopping_gadgets: $50
- shopping_kids: $50
- shopping_pets: $50

# Insight Period Top Transactions (from 2026-03-01 to 2026-03-31)

- $46 at Amazon as shopping_clothing.
- $46 at Target as shopping_gadgets.
- $46 at Best Buy as shopping_kids.
- $46 at Costco as shopping_pets.
- $46 at Walmart as shopping_clothing.
- $46 at Etsy as shopping_gadgets.
- $46 at Apple Store as shopping_kids.
- $46 at Nordstrom as shopping_pets.
- $46 at Home Depot as shopping_clothing.
- $46 at Kohl's as shopping_gadgets.
+90 more

# Previous Period Top Transactions (from 2026-02-01 to 2026-02-28)

- $88 at Target as shopping_clothing.
- $72 at Old Navy as shopping_clothing.
- $65 at Petco as shopping_pets.
- $54 at Best Buy as shopping_gadgets.
- $48 at Gap as shopping_clothing.
- $41 at Apple Store as shopping_gadgets.
- $38 at Carter's as shopping_kids.
- $32 at Kohl's as shopping_clothing.
- $28 at Etsy as shopping_gadgets.
- $24 at Walmart as shopping_clothing.
+36 more

""",
    "output": (
      "Expected: `lookup_transactions` then `return rationalize(USER_MESSAGE, lookup_info)`—excerpt is **shopping_* only** "
      "but ends with **+90 more** with **no hidden total**, so **$920** in the Insight is not fully anchored; mock lookup "
      "returns ten **$46** lines + tail **Total: $920**."
    ),
  },
  {
    "batch": 1,
    "name": "rationalize_week_spend_followup_turn",
    "mock_lookup_transactions": """From 2026-03-25 through 2026-03-31, with category meals_dining_out.

Transactions:

No matching transactions.
""",
    "input": """# Task Description

**Week-to-date** actual spend vs forecast, **as of 2026/03/31**. Explain why the Insight holds versus that forecast using the matching **# Expected …** section, the Insight, and the Insight-period and previous-period transaction excerpts; call **lookup_transactions** only if those sources cannot justify the Insight’s stated amounts.

# Insight

Dining out is significantly above your weekly forecast at about $180 vs ~$45 planned. Prior turn: lookup_transactions returned no extra rows for the same 7-day window; answer from the top transactions below only—do not repeat that lookup.

# Expected Week-to-date

- meals_dining_out: $45

# Insight Period Top Transactions (from 2026-03-25 to 2026-03-31)

- $62 at Olive Garden as meals_dining_out.
- $44 at DoorDash as meals_dining_out.
- $43 at Northside Bistro as meals_dining_out.
- $19 at Chipotle as meals_dining_out.
- $12 at Starbucks as meals_dining_out.
+6 transactions

# Previous Period Top Transactions (from 2026-03-18 to 2026-03-24)

- $35 at Corner Cafe as meals_dining_out.
- $28 at Sushi Den as meals_dining_out.
- $14 at Chipotle as meals_dining_out.
- $8 at Starbucks as meals_dining_out.
+8 transactions

""",
    "output": (
      "Expected: `execute_plan` → `(True, str)` without redundant lookup—dining lines in the Insight period support ~**$180** "
      "vs **# Expected Week-to-date** **$45**; mock lookup is empty for dining_out window."
    ),
  },
  {
    "batch": 1,
    "name": "rationalize_week_spend_vs_forecast_calendar_week",
    "mock_lookup_transactions": """From 2026-04-05 through 2026-04-07, with categories shelter_home and shelter.

Transactions:

No matching transactions.
""",
    "input": """# Task Description

**Week-to-date** actual spend vs forecast, **as of 2026/04/07**. Explain why the Insight holds versus that forecast using the matching **# Expected …** section, the Insight, and the Insight-period and previous-period transaction excerpts; call **lookup_transactions** only if those sources cannot justify the Insight’s stated amounts.

# Insight

Home received refunds this week, totaling $679.

# Expected Week-to-date

- shelter: $405
- shelter_home: $254

# Insight Period Top Transactions (from 2026-04-05 to 2026-04-07)

- $357 at Rent Share Transfer as shelter_home.
- $321 at Rent Share Transfer as shelter_home.
- Total: $679

# Previous Period Top Transactions (from 2026-03-29 to 2026-04-04)

No matching transactions.

# Previous 2 Period Spending per category and period (2026-03-15 to 2026-03-28)

2026-03-15 to 2026-03-21: shelter $600  shelter_home $560
2026-03-22 to 2026-03-28: shelter $487  shelter_home $384

""",
    "output": (
      "Expected: `execute_plan` → `(True, str)` with no `lookup_transactions` / no `rationalize`—**week_spend_vs_forecast** shape: "
      "Insight-period lines (**$357** + **$321** = **$679**) match **`- Total: $679`** and the Insight’s **$679** refunds figure; "
      "reconcile vs **# Expected Week-to-date** (**shelter** / **shelter_home** forecasts). **# Previous Period Top Transactions** "
      "is **No matching transactions.**; **# Previous 2 Period Spending…** (after previous-period block, header span **2026-03-15 to 2026-03-28**) "
      "gives trailing shelter rollups for context only. Do not contradict excerpt totals."
    ),
  },
  {
    "batch": 2,
    "name": "rationalize_salary_income_down_sufficient_context",
    "mock_lookup_transactions": """From 2026-03-01 through 2026-03-31, with categories income_salary, income_sidegig, income_business, income_interest.

Transactions:

No matching transactions.
""",
    "input": """# Task Description

**Month-to-date** actual spend vs forecast, **as of 2026/03/31**. Explain why the Insight holds versus that forecast using the matching **# Expected …** section, the Insight, and the Insight-period and previous-period transaction excerpts; call **lookup_transactions** only if those sources cannot justify the Insight’s stated amounts.

# Insight

Salary is significantly down this month at $1481.  Income is thus down this month to $1481. February’s excerpt shows two full payroll deposits versus a single March deposit in the top lines.

# Expected Month-to-date

- income_salary: $3300

# Insight Period Top Transactions (from 2026-03-01 to 2026-03-31)

- $1481 at Acme Corp Payroll as income_salary.
- Total: $1481

# Previous Period Top Transactions (from 2026-02-01 to 2026-02-28)

- $1650 at Acme Corp Payroll as income_salary.
- $1650 at Acme Corp Payroll as income_salary.
- Total: $3300


""",
    "output": (
      "Expected: `execute_plan` → `(True, str)` without lookup—March shows one **$1481** salary line matching income **$1481** "
      "vs **# Expected Month-to-date** salary; February shows two **$1650** payroll lines, explaining the Insight."
    ),
  },
  {
    "batch": 2,
    "name": "rationalize_shopping_down_discount_vs_prior_upscale",
    "mock_lookup_transactions": """From 2026-03-01 through 2026-03-25, with categories shopping_clothing, shopping_gadgets, shopping_kids, shopping_pets.

Transactions:

No matching transactions.
""",
    "input": """# Task Description

**Month-to-date** actual spend vs forecast, **as of 2026/03/25**. Explain why the Insight holds versus that forecast using the matching **# Expected …** section, the Insight, and the Insight-period and previous-period transaction excerpts; call **lookup_transactions** only if those sources cannot justify the Insight’s stated amounts.

# Insight

Clothing is significantly down this month at $124. Kids is significantly down this month at $0. Gadgets is significantly down this month at $0. Shopping is thus significantly down this month to $124.

# Expected Month-to-date

- shopping_clothing: $480
- shopping_gadgets: $350
- shopping_kids: $280

# Insight Period Top Transactions (from 2026-03-01 to 2026-03-25)

- $52 at Target as shopping_clothing.
- $45 at Kohl's as shopping_clothing.
- $27 at TJ Maxx as shopping_clothing.
- Total: $124

# Previous Period Top Transactions (from 2026-02-01 to 2026-02-25)

- $2199 at Apple Fifth Avenue as shopping_gadgets.
- $890 at Saks Fifth Avenue as shopping_clothing.
- $625 at Bonpoint Madison as shopping_kids.
- $48 at Gap Kids as shopping_kids.
- $42 at Zara as shopping_clothing.
- $38 at Best Buy as shopping_gadgets.
- $32 at Old Navy as shopping_clothing.
- $28 at PetSmart as shopping_pets.
- $24 at Target as shopping_clothing.
- $19 at Etsy as shopping_gadgets.
+28 more totaling $1842


""",
    "output": (
      "Expected: `execute_plan` → `(True, str)` without lookup—March 1–25 shows mid-tier clothing only (**$52+$45+$27=$124**) "
      "vs **# Expected** shopping-category lines; prior window shows flagship gadget/clothing/kids spend, supporting the Insight."
    ),
  },
  {
    "batch": 2,
    "name": "rationalize_health_down_sufficient_context",
    "mock_lookup_transactions": """From 2026-03-01 through 2026-03-31, with categories health_personal_care and health_medical_pharmacy.

Transactions:

No matching transactions.
""",
    "input": """# Task Description

**Month-to-date** actual spend vs forecast, **as of 2026/03/31**. Explain why the Insight holds versus that forecast using the matching **# Expected …** section, the Insight, and the Insight-period and previous-period transaction excerpts; call **lookup_transactions** only if those sources cannot justify the Insight’s stated amounts.

# Insight

Personal Care is significantly down this month at $169. Medical & Pharmacy is significantly down this month at $72. Health is thus significantly down this month to $241.

# Expected Month-to-date

- health_personal_care: $310
- health_medical_pharmacy: $195

# Insight Period Top Transactions (from 2026-03-01 to 2026-03-31)

- $98 at Sephora as health_personal_care.
- $71 at Drybar as health_personal_care.
- $45 at CVS as health_medical_pharmacy.
- $27 at Walgreens as health_medical_pharmacy.
- Total: $241

# Previous Period Top Transactions (from 2026-02-01 to 2026-02-28)

- $185 at Bliss Spa as health_personal_care.
- $145 at Ulta Beauty as health_personal_care.
- $110 at CVS Pharmacy as health_medical_pharmacy.
- $95 at Rite Aid as health_medical_pharmacy.
- $88 at CityMD as health_medical_pharmacy.
- $72 at Sephora as health_personal_care.
- $64 at Duane Reade as health_medical_pharmacy.
- $55 at Drybar as health_personal_care.
- $48 at Local Pharmacy as health_medical_pharmacy.
- $41 at Ulta as health_personal_care.
+18 more totaling $412


""",
    "output": (
      "Expected: `execute_plan` → `(True, str)` without lookup—recent lines foot personal care (**$98+$71=$169**) and "
      "medical/pharmacy (**$45+$27=$72**) for health **$241** vs **# Expected Month-to-date**; February shows larger health "
      "charges, explaining the Insight."
    ),
  },
]


def run_test(test_name_or_index_or_dict, optimizer: StrategizerOptimizer | None = None):
  if isinstance(test_name_or_index_or_dict, dict):
    if "input" not in test_name_or_index_or_dict:
      raise ValueError(
        "test dict must include 'input' (and typically 'name', 'batch', 'output', 'mock_lookup_transactions')."
      )
    test_name = test_name_or_index_or_dict.get("name", "custom_test")
    print(f"\n# Test: **{test_name}**\n")
    if optimizer is None:
      optimizer = StrategizerOptimizer()
    engine = RationalizeChangeEngine(optimizer, print_thought_summary=False)
    prompt_body = test_name_or_index_or_dict["input"]

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
          mock_lookup = test_name_or_index_or_dict.get("mock_lookup_transactions")

          def wrapped_lookup_transactions(*args, **kwargs):
            if mock_lookup is not None:
              out = mock_lookup
            else:
              out = lookup_transactions(*args, **kwargs)
            print("\n## lookup_transactions returned\n")
            print(out)
            print()
            return out

          namespace = {
            "USER_MESSAGE": prompt_body,
            "lookup_transactions": wrapped_lookup_transactions,
            "rationalize": engine.sandbox_rationalize_callback(),
          }
          exec(code, namespace)
          if "execute_plan" in namespace:
            execution_result = namespace["execute_plan"]()
            print("\n## Execution Final Result:\n")
            print("```")
            print(f"  success: {execution_result[0]}")
            print(f"  output: {execution_result[1]}")
            print("```")
        except Exception as e:
          print(f"Error executing generated code: {str(e)}")
          import traceback
          print(traceback.format_exc())
    finally:
      optimizer.generate_response = _orig_generate_response

    if test_name_or_index_or_dict.get("output"):
      print(f"\n## Expected behavior:\n\n{test_name_or_index_or_dict['output']}\n")
    return execution_result

  if isinstance(test_name_or_index_or_dict, int):
    tc = TEST_CASES[test_name_or_index_or_dict] if 0 <= test_name_or_index_or_dict < len(TEST_CASES) else None
  else:
    tc = next((t for t in TEST_CASES if t["name"] == test_name_or_index_or_dict), None)
  if not tc:
    return None
  return run_test(tc, optimizer)


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
  kw: dict = {"thinking_budget": tb}
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
    run_test(test_val, optimizer)
    return

  print("Available test cases:")
  for i, tc in enumerate(TEST_CASES):
    print(f"  {i}: {tc['name']} (batch {tc['batch']})")


# python active_experiments/rationalize_change_strategizer_optimizer_v3.py --batch 1
if __name__ == "__main__":
  import argparse
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

from google import genai
from google.genai import types
from google.genai.errors import ClientError
import os
import sys
from dotenv import load_dotenv

# Add the parent directory to the path so we can import the tools
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
  sys.path.insert(0, parent_dir)

from penny.strategizer.rationalize_change_engine import RationalizeChangeEngine
from penny.tool_funcs.lookup_transactions import lookup_transactions

# Load environment variables
load_dotenv()

# Defaults (StrategizerOptimizer): flash-lite + thinking—see class __init__. Tuned for concise code + low fluff.
# Keep in sync with penny2 finance-ai-llm-server ``RATIONALIZE_CHANGE_SYSTEM_PROMPT`` (rationalize_change_strategizer.py).

RATIONALIZE_CHANGE_MAX_VISIBLE_TRANSACTIONS = 10

SYSTEM_PROMPT = f"""You are **RationalizeChange**. Reply with **one** ```python``` block defining `execute_plan() -> tuple[bool, str]` only.

**Benchmark:** For `*_vs_forecast`, compare actuals to **forecast/plan**. The **previous** `# Top Transactions` section is context, not the benchmark. Prior-period phrases in the Insight do not redefine the task type.

**Inputs:** **Task Description** (caller-authored; obey). **Insight.** Two predetermined `# Top Transactions` blocks (recent + previous). Optional **Previous Outcomes**. Recent header’s end date = Task **as of** (insight cutoff).

**Flow:** (1) Map Insight labels loosely to official slugs; reconcile $ to bullets using **both** sections. (2) If excerpts already explain vs forecast → `return True, "…"` (no tools). (3) If still implausible → `lookup_transactions` (header date ranges; `in_category` official slugs only) then `return rationalize(USER_MESSAGE, lookup_info)`. Never `rationalize` without lookup. (4) Follow Insight constraints (e.g. skip redundant lookup). (5) **Host follow-up:** If the user message already includes **`# Supplemental lookup`**, you are **past** the merge step: **`execute_plan` must `return True, "…"` only** (≤3 sentences)—**never** call `rationalize` or `lookup_transactions` again. Copying the first-turn pattern `return rationalize(USER_MESSAGE, lookup_info)` here is **wrong** on this turn.

**Explain the change (required):** The return string must **account for why** the Insight’s situation holds—not only repeat current dollar totals vs forecast. Tie the narrative to **concrete drivers**: which subcategories or merchants moved (from bullets and/or lookup), **direction** (higher/lower vs forecast or vs previous period), and **one** clear causal link. Do **not** end on a vague line like “reflects a decrease compared to forecast” without saying **what** decreased or increased and **how** the listed transactions support it.

**Return string:** Plain English, **≤3 short sentences**, no “Based on…”, no markdown inside quotes, no meta narration. **Do not contradict** dollar totals given in the Insight (visible bullets may be partial when `+K more` applies).

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

**Tools:** `lookup_transactions` and `rationalize` are **already in scope**—do **not** import them from any module. Only add `from datetime import date` (or `import datetime`) for bounds. **`lookup_transactions`** signature: required **`start`**, **`end`** (`date`, inclusive); optional **`name_contains`** (substring on merchant), **`amount_larger_than`**, **`amount_less_than`** (ints, absolute dollars), **`in_category`** (list of official slugs), **`max_visible`** (default {RATIONALIZE_CHANGE_MAX_VISIBLE_TRANSACTIONS}). Host accepts common aliases (e.g. `start_date`/`end_date`, `merchant_contains`, `categories`) but prefer the names above. Host returns `- $N at Merchant as slug.` lines, optional `+K more`. After lookup, **`return rationalize(USER_MESSAGE, lookup_info)`** only—never paraphrase the prompt as the first argument. Do not invent data.

**Lists:** Global top-{RATIONALIZE_CHANGE_MAX_VISIBLE_TRANSACTIONS} by amount (unless `max_visible` is overridden); category totals may hide below the cutoff (`+K more`).

**Code:** Keep `execute_plan` small (~≤15 lines). For `lookup_transactions` bounds use `from datetime import date` and `date(y, m, d)` (or `import datetime` and `datetime.date`). On the **first** turn only: **`return rationalize(USER_MESSAGE, lookup_info)`** after lookup—never pass a hand-written summary as the first argument. On a turn that already has **`# Supplemental lookup`** in the user message: **`return True, "your summary"`** only. Skip filler comments.
"""


class StrategizerOptimizer:
  """Gemini API wrapper for RationalizeChange (`execute_plan`; optional lookup + rationalize).

  **Defaults** (`gemini-flash-lite-latest`, thinking 512, max_output 600, temperature 0.2) passed batch 1–2 in this experiment.
  Cutting thinking/output further (e.g. <512 / <600) can truncate Python or drop the shopping+lookup path.
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


# Task Description: static wording + `{insight_type}`, `{period_label}`, `{as_of}` (as_of = insight cutoff, end of recent window).
_RATIONALIZE_TASK_PERIOD_BY_INSIGHT_TYPE: dict[str, str] = {
  "month_spend_vs_forecast": "Month-to-date",
  "week_spend_vs_forecast": "Week-to-date",
}

RATIONALIZE_TASK_DESCRIPTION_TEMPLATE = (
  "**{insight_type}** — **{period_label}** actual spend vs forecast, **as of {as_of}**. "
  "Use the Insight and the two **# Top Transactions** sections below. "
  "Forecast/plan is the benchmark; the previous section is context only. "
  "Reconcile stated amounts to bullet `as` slugs across **recent and previous** when possible. "
  "If excerpts suffice, `return True` with a short dashboard string. "
  "Otherwise `lookup_transactions` (official `in_category` slugs, dates from section headers) then `return rationalize(USER_MESSAGE, lookup_info)`."
)

RATIONALIZE_TASK_DESCRIPTION_BY_INSIGHT_TYPE = {
  k: RATIONALIZE_TASK_DESCRIPTION_TEMPLATE.format(
    insight_type=k,
    period_label=_RATIONALIZE_TASK_PERIOD_BY_INSIGHT_TYPE[k],
    as_of="{as_of}",
  )
  for k in _RATIONALIZE_TASK_PERIOD_BY_INSIGHT_TYPE
}


def rationalize_task_description_for_insight_type(insight_type: str, *, as_of: str = "2026/03/31") -> str:
  """Return Task Description for ``insight_type``. ``as_of`` is ``YYYY/MM/DD`` (same as recent window end in headers)."""
  try:
    tpl = RATIONALIZE_TASK_DESCRIPTION_BY_INSIGHT_TYPE[insight_type]
  except KeyError as e:
    raise KeyError(
      f"No task description for insight_type={insight_type!r}; extend _RATIONALIZE_TASK_PERIOD_BY_INSIGHT_TYPE."
    ) from e
  return tpl.format(as_of=as_of)


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


def _format_rationalize_change_prompt(
  task_description: str,
  insight: str,
  top_transactions_recent_period: str,
  top_transactions_previous_period: str,
  *,
  recent_insight_date_range: str = "—",
  previous_insight_date_range: str = "—",
  previous_outcomes: dict[int | str, str] | list[str] | str | None = None,
) -> str:
  previous_outcomes_block = _format_previous_outcomes(previous_outcomes)
  body = f"""# Task Description

{task_description}

# Insight

{insight}

# Top Transactions — recent insight period ({recent_insight_date_range})

{top_transactions_recent_period}

# Top Transactions — previous period ({previous_insight_date_range})

{top_transactions_previous_period}"""
  if previous_outcomes_block.strip():
    body += f"""

## Previous Outcomes

{previous_outcomes_block}"""
  return body


TEST_CASES = [
  {
    "batch": 1,
    "name": "rationalize_leisure_down_sufficient_context",
    "mock_lookup_transactions": """From 2026-03-01 through 2026-03-31, with categories leisure_entertainment and leisure_travel.

Transactions:

No matching transactions.
""",
    "input": """# Task Description

**Month-to-date** actual spend vs forecast, **as of 2026/03/31**. Use the Insight and the two **# Top Transactions** sections below (predetermined). Forecast/plan is the benchmark; the previous section is context only. Reconcile stated amounts to bullet `as` slugs across **recent and previous** when possible. If excerpts suffice, `return True` with a short dashboard string. Otherwise `lookup_transactions` (official `in_category` slugs, dates from section headers) then `return rationalize(USER_MESSAGE, lookup_info)`.

# Insight

Entertainment is significantly below forecast this month at $309. Travel & Vacations is significantly below forecast at $47. Leisure is thus significantly below forecast this month at $356. The prior month’s excerpt shows much larger lodging, flights, and events than March’s top lines do for travel and entertainment.

# Top Transactions — recent insight period (2026-03-01 to 2026-03-31)

- $142 at Whole Foods as meals_groceries.
- $62 at Shell Gas as transportation_gas.
- $58 at Thai Garden as meals_dining_out.
- $48 at AMC Theaters as entertainment_movies.
- $35 at City Parking as transportation_parking.
- $24 at Uber as transportation_rideshare.
- $19 at CVS as health_pharmacy.
- $16 at Netflix as entertainment_streaming.
- $11 at Spotify as entertainment_streaming.
- $0 at Delta Air as travel_flights.
+38 transactions

# Top Transactions — previous period (2026-02-01 to 2026-02-28)

- $890 at Marriott as travel_lodging.
- $412 at Delta Air as travel_flights.
- $220 at StubHub as entertainment_events.
- $180 at Concert Hall as entertainment_events.
- $128 at Whole Foods as meals_groceries.
- $64 at AMC Theaters as entertainment_movies.
- $55 at Shell Gas as transportation_gas.
- $31 at Uber as transportation_rideshare.
- $16 at Netflix as entertainment_streaming.
- $11 at Spotify as entertainment_streaming.
+31 transactions

""",
    "output": "Expected: execute_plan returns (True, explanation) without lookup—large travel/event lines in the previous-period list explain leisure below forecast vs lighter entertainment/travel in the recent list.",
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

**Month-to-date** actual spend vs forecast, **as of 2026/03/31**. Use the Insight and the two **# Top Transactions** sections below (predetermined). Forecast/plan is the benchmark; the previous section is context only. Reconcile stated amounts to bullet `as` slugs across **recent and previous** when possible. If excerpts suffice, `return True` with a short dashboard string. Otherwise `lookup_transactions` (official `in_category` slugs, dates from section headers) then `return rationalize(USER_MESSAGE, lookup_info)`.

# Insight

Shopping is significantly above forecast this month at $920 vs ~$200 expected for shopping.

# Top Transactions — recent insight period (2026-03-01 to 2026-03-31)

- $118 at Whole Foods as meals_groceries.
- $67 at Trader Joe's as meals_groceries.
- $62 at Peak Fitness as health_gym.
- $58 at Smile Dental as health_dental.
- $55 at City Parking as transportation_parking.
- $54 at Urban Pet as pets_supplies.
- $53 at Downtown Dry Clean as services_laundry.
- $52 at Ace Hardware as home_improvement.
- $51 at Campus Books as education_books.
- $48 at Thai Garden as meals_dining_out.
+54 transactions

# Top Transactions — previous period (2026-02-01 to 2026-02-28)

- $105 at Whole Foods as meals_groceries.
- $72 at Trader Joe's as meals_groceries.
- $44 at Thai Garden as meals_dining_out.
- $40 at City Parking as transportation_parking.
- $39 at Shell Gas as transportation_gas.
- $28 at Uber as transportation_rideshare.
- $21 at CVS as health_pharmacy.
- $19 at Walgreens as health_pharmacy.
- $19 at Starbucks as meals_coffee.
- $16 at Netflix as entertainment_streaming.
+42 transactions

""",
    "output": (
      "Expected: execute_plan calls lookup once; output reflects many sub-floor $46 official shopping charges totaling $920 (consistent with $48 10th-place global top txn)."
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

**Week-to-date** actual spend vs forecast, **as of 2026/03/31**. Use the Insight and the two **# Top Transactions** sections below (predetermined). Forecast/plan is the benchmark; the previous section is context only. Reconcile stated amounts to bullet `as` slugs across **recent and previous** when possible. If excerpts suffice, `return True` with a short dashboard string. Otherwise `lookup_transactions` (official `in_category` slugs, dates from section headers) then `return rationalize(USER_MESSAGE, lookup_info)`.

# Insight

Dining out is significantly above your weekly forecast at about $180 vs ~$45 planned. Prior turn: lookup_transactions returned no extra rows for the same 7-day window; answer from the top transactions below only—do not repeat that lookup.

# Top Transactions — recent insight period (2026-03-25 to 2026-03-31)

- $95 at Whole Foods as meals_groceries.
- $62 at Olive Garden as meals_dining_out.
- $44 at DoorDash as meals_dining_out.
- $19 at Chipotle as meals_dining_out.
- $12 at Starbucks as meals_coffee.
+11 transactions

# Top Transactions — previous period (2026-03-18 to 2026-03-24)

- $72 at Trader Joe's as meals_groceries.
- $22 at CVS as health_pharmacy.
- $14 at Chipotle as meals_dining_out.
- $8 at Starbucks as meals_coffee.
- $0 at Home cooking transfer as internal.
+14 transactions

""",
    "output": "Expected: direct rationalization vs weekly dining forecast using top transactions (more dining lines and higher tickets), no redundant lookup.",
  },
  {
    "batch": 2,
    "name": "rationalize_salary_income_down_sufficient_context",
    "mock_lookup_transactions": """From 2026-03-01 through 2026-03-31, with categories income_salary, income_sidegig, income_business, income_interest.

Transactions:

No matching transactions.
""",
    "input": """# Task Description

**Month-to-date** actual spend vs forecast, **as of 2026/03/31**. Use the Insight and the two **# Top Transactions** sections below (predetermined). Forecast/plan is the benchmark; the previous section is context only. Reconcile stated amounts to bullet `as` slugs across **recent and previous** when possible. If excerpts suffice, `return True` with a short dashboard string. Otherwise `lookup_transactions` (official `in_category` slugs, dates from section headers) then `return rationalize(USER_MESSAGE, lookup_info)`.

# Insight

Salary is significantly down this month at $1481.  Income is thus down this month to $1481. February’s excerpt shows two full payroll deposits versus a single March deposit in the top lines.

# Top Transactions — recent insight period (2026-03-01 to 2026-03-31)

- $1481 at Acme Corp Payroll as income_salary.
- $118 at Whole Foods as meals_groceries.
- $67 at Trader Joe's as meals_groceries.
- $62 at Shell Gas as transportation_gas.
- $58 at Thai Garden as meals_dining_out.
- $35 at City Parking as transportation_parking.
- $24 at Uber as transportation_rideshare.
- $19 at CVS as health_pharmacy.
- $16 at Netflix as entertainment_streaming.
- $11 at Spotify as entertainment_streaming.
+33 transactions

# Top Transactions — previous period (2026-02-01 to 2026-02-28)

- $1650 at Acme Corp Payroll as income_salary.
- $1650 at Acme Corp Payroll as income_salary.
- $890 at Marriott as travel_lodging.
- $105 at Whole Foods as meals_groceries.
- $72 at Trader Joe's as meals_groceries.
- $44 at Thai Garden as meals_dining_out.
- $40 at City Parking as transportation_parking.
- $39 at Shell Gas as transportation_gas.
- $28 at Uber as transportation_rideshare.
- $21 at CVS as health_pharmacy.
+36 transactions

""",
    "output": (
      "Expected: execute_plan returns (True, explanation) without lookup—March shows one $1481 salary line matching total income $1481; February shows two larger payroll deposits, explaining income vs prior month and supporting the insight without supplemental income lookup."
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

**Month-to-date** actual spend vs forecast, **as of 2026/03/25**. Use the Insight and the two **# Top Transactions** sections below (predetermined). Forecast/plan is the benchmark; the previous section is context only. Reconcile stated amounts to bullet `as` slugs across **recent and previous** when possible. If excerpts suffice, `return True` with a short dashboard string. Otherwise `lookup_transactions` (official `in_category` slugs, dates from section headers) then `return rationalize(USER_MESSAGE, lookup_info)`.

# Insight

Clothing is significantly down this month at $124. Kids is significantly down this month at $0. Gadgets is significantly down this month at $0. Shopping is thus significantly down this month to $124.

# Top Transactions — recent insight period (2026-03-01 to 2026-03-25)

- $228 at Eataly as meals_groceries.
- $156 at Shell V-Power as transportation_gas.
- $132 at Carbone as meals_dining_out.
- $52 at Target as shopping_clothing.
- $45 at Kohl's as shopping_clothing.
- $27 at TJ Maxx as shopping_clothing.
- $88 at Garage Parking as transportation_parking.
- $54 at Uber as transportation_rideshare.
- $42 at CVS as health_pharmacy.
- $16 at Netflix as entertainment_streaming.
+41 transactions

# Top Transactions — previous period (2026-02-01 to 2026-02-25)

- $2199 at Apple Fifth Avenue as shopping_gadgets.
- $890 at Saks Fifth Avenue as shopping_clothing.
- $625 at Bonpoint Madison as shopping_kids.
- $385 at Whole Foods 365 as meals_groceries.
- $298 at Equinox as health_gym.
- $245 at Eleven Madison Park as meals_dining_out.
- $165 at Hotel Valet as transportation_parking.
- $112 at Shell V-Power as transportation_gas.
- $95 at Uber Black as transportation_rideshare.
- $48 at CVS as health_pharmacy.
+38 transactions

""",
    "output": (
      "Expected: execute_plan returns (True, explanation) without lookup—March 1–25 shows only mid-tier clothing (Target/Kohl's/TJ Maxx, $52+$45+$27=$124) and no kids/gadgets lines; prior window shows flagship Apple, Saks, and luxury kids spend plus costly dining and services, supporting shopping down vs a much pricier February mix."
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

**Month-to-date** actual spend vs forecast, **as of 2026/03/31**. Use the Insight and the two **# Top Transactions** sections below (predetermined). Forecast/plan is the benchmark; the previous section is context only. Reconcile stated amounts to bullet `as` slugs across **recent and previous** when possible. If excerpts suffice, `return True` with a short dashboard string. Otherwise `lookup_transactions` (official `in_category` slugs, dates from section headers) then `return rationalize(USER_MESSAGE, lookup_info)`.

# Insight

Personal Care is significantly down this month at $169. Medical & Pharmacy is significantly down this month at $72. Health is thus significantly down this month to $241.

# Top Transactions — recent insight period (2026-03-01 to 2026-03-31)

- $98 at Sephora as health_personal_care.
- $71 at Drybar as health_personal_care.
- $45 at CVS as health_medical_pharmacy.
- $95 at Whole Foods as meals_groceries.
- $67 at Trader Joe's as meals_groceries.
- $62 at Shell Gas as transportation_gas.
- $58 at Thai Garden as meals_dining_out.
- $35 at City Parking as transportation_parking.
- $27 at Walgreens as health_medical_pharmacy.
- $24 at Uber as transportation_rideshare.
+34 transactions

# Top Transactions — previous period (2026-02-01 to 2026-02-28)

- $185 at Bliss Spa as health_personal_care.
- $145 at Ulta Beauty as health_personal_care.
- $110 at CVS Pharmacy as health_medical_pharmacy.
- $95 at Rite Aid as health_medical_pharmacy.
- $105 at Whole Foods as meals_groceries.
- $72 at Trader Joe's as meals_groceries.
- $44 at Thai Garden as meals_dining_out.
- $40 at City Parking as transportation_parking.
- $39 at Shell Gas as transportation_gas.
- $28 at Uber as transportation_rideshare.
+32 transactions

""",
    "output": (
      "Expected: execute_plan returns (True, explanation) without lookup—recent lines foot personal care ($98+$71=$169) and medical/pharmacy ($45+$27=$72) for health $241; February shows larger health_personal_care and health_medical_pharmacy charges, explaining health down vs the prior month without lookup."
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
    if test_name_or_index_or_dict.get("output"):
      print(f"\n## Output:\n\n{test_name_or_index_or_dict['output']}\n")
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


# python experiments/rationalize_change_strategizer_optimizer.py --test 0
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

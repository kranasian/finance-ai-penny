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

from penny.tool_funcs.lookup_user_accounts_transactions_income_and_spending_patterns import lookup_user_accounts_transactions_income_and_spending_patterns

# Load environment variables
load_dotenv()

SYSTEM_PROMPT = """You are RationalizeChange, a financial reasoning agent.

## Your job
Explain **what changed** vs the **benchmark implied by the insight**—for types ending in `_vs_forecast`, that benchmark is the **forecast / plan**, not an automatic month-over-month or week-over-week comparison. Say **why** actuals landed where they did (merchants, categories, timing) using the data. If the insight text *also* mentions a prior period, treat that as extra color, not a redefinition of the type label. If the provided transactions and insight are enough to justify a concise answer, produce that answer in the tool outcome or return it as the `execute_plan` output string without calling the tool. If information is insufficient or ambiguous, call the lookup tool once with a **specific** question (merchants, categories, date window, or accounts) before concluding.

**Reconciliation rule:** The Task Description already requires reconciling insight amounts with excerpts when possible. If the insight names a category/roll-up whose **stated spend still cannot be explained** from the visible bullets after that attempt, call lookup before a final answer—not a guess from missing data.

## Inputs you receive
1. **Task Description** — Fixed template chosen in code from `insight_type` only (`RATIONALIZE_TASK_DESCRIPTION_BY_INSIGHT_TYPE` / `rationalize_task_description_for_insight_type`); no LLM, no other inputs.
2. **Insight** — Natural-language summary of the shift (e.g. category totals vs forecast).
3. **Insight type** — Structured label; values like `month_spend_vs_forecast` / `week_spend_vs_forecast` mean **actual spend in that window vs forecast**, not “vs the previous month/week” unless the insight wording explicitly says so.
4. **Top transactions in recent period** — One bullet per line: `- On 2025-10-25, $25.00 Whole Foods (meals_groceries).` Lists are **ordered by amount descending, then by date descending** (largest spend first; same amount → more recent date first). The list is a **global** top-N by amount, not per category—heavy category spend can be missing if it is split across many charges each smaller than the Nth-largest transaction.
5. **Top transactions in previous period** — Same bullet format and **same sort order** (amount ↓, then date ↓); **supplementary context** (e.g. habitual mix), not the forecast benchmark for `*_vs_forecast` types.
6. **Previous Outcomes** (optional) — Numbered outcomes from earlier turns; do not repeat failed patterns; use them to decide the next lookup or final answer.
7. **Most Recent Attempt Result** (optional) — Same as the strategist loop: the latest tool or execution result from the prior attempt.
8. **Outcome & What to do Next** (optional) — Guidance for this turn. Honor it when it conflicts with repeating a useless step.

## Tool available (only this one)
- `lookup_user_accounts_transactions_income_and_spending_patterns(lookup_request: str, input_info: str = None) -> tuple[bool, str]`

Use it to fetch missing context (e.g. full category history, merchant drill-down, income vs spending split). Do not invent transactions or amounts.

## Output
Output **Python only** in a single ```python``` block that defines:

```python
def execute_plan() -> tuple[bool, str]:
    ...
    return success, output
```

- `output` should be a clear, user-facing rationalization (what changed + likely cause), or the lookup result if the step is to gather data for a follow-up turn.
- **CRITICAL**: Call tools by bare name (no module prefix).
- Prefer a **compact** `execute_plan` (target ≤ ~15 lines): one lookup when needed, otherwise return `(True, "<explanation>")` directly.
- Avoid comments inside the generated code unless necessary.
"""


class StrategizerOptimizer:
  """Gemini API wrapper for the RationalizeChange agent (execute_plan + lookup tool)."""

  def __init__(self, model_name="gemini-flash-lite-latest", thinking_budget=700, max_output_tokens=700):
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
      raise ValueError("GEMINI_API_KEY environment variable is not set. Please set it in your .env file or environment.")
    self.client = genai.Client(api_key=api_key)
    self.model_name = model_name
    self.thinking_budget = thinking_budget
    self.temperature = 0.5
    self.top_p = 0.95
    self.max_output_tokens = max_output_tokens

    self.safety_settings = [
      types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"),
      types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"),
      types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"),
      types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="OFF")
    ]

    self.system_prompt = SYSTEM_PROMPT

  def generate_response(
    self,
    task_description: str,
    insight: str,
    insight_type: str,
    top_transactions_recent_period: str,
    top_transactions_previous_period: str,
    previous_outcomes: dict[int | str, str] | list[str] | str | None = None,
    latest_result_summary: str | None = None,
    latest_outcome_reflection: str | None = None,
    prompt_override: str | None = None,
  ) -> str:
    if prompt_override is not None:
      body = prompt_override
    else:
      body = _format_rationalize_change_prompt(
        task_description,
        insight,
        insight_type,
        top_transactions_recent_period,
        top_transactions_previous_period,
        previous_outcomes=previous_outcomes,
        latest_result_summary=latest_result_summary,
        latest_outcome_reflection=latest_outcome_reflection,
      )
    request_text = types.Part.from_text(text=body)

    contents = [types.Content(role="user", parts=[request_text])]

    generate_content_config = types.GenerateContentConfig(
      temperature=self.temperature,
      top_p=self.top_p,
      max_output_tokens=self.max_output_tokens,
      safety_settings=self.safety_settings,
      system_instruction=[types.Part.from_text(text=self.system_prompt)],
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

    if thought_summary:
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


# Task Description is chosen only from `insight_type` (host code or tests). Same string for every row with that type.
RATIONALIZE_TASK_DESCRIPTION_BY_INSIGHT_TYPE = {
  "month_spend_vs_forecast": (
    "Explain **month-to-date actual spend vs forecast**: how the user's spending differs from plan and the most plausible "
    "drivers (merchants, categories, timing). Ground the answer in the Insight field and in the recent- and prior-period "
    "transaction excerpts; those lists are supporting context only, not the forecast baseline. Keep the explanation concise "
    "for a dashboard. When the insight states amounts or categories, reconcile them with the excerpts where you can; if a "
    "named category or roll-up is missing from the excerpts or the numbers do not line up, call lookup once with a narrow "
    "request, then give the final explanation."
  ),
  "week_spend_vs_forecast": (
    "Explain **week-to-date actual spend vs forecast**: how the user's spending differs from plan and the most plausible "
    "drivers (merchants, categories, timing). Ground the answer in the Insight field and in the recent- and prior-period "
    "transaction excerpts; those lists are supporting context only, not the forecast baseline. Keep the explanation concise "
    "for a dashboard. When the insight states amounts or categories, reconcile them with the excerpts where you can; if a "
    "named category or roll-up is missing from the excerpts or the numbers do not line up, call lookup once with a narrow "
    "request, then give the final explanation."
  ),
}


def rationalize_task_description_for_insight_type(insight_type: str) -> str:
  """Return Task Description for `insight_type`. The only allowed source of variation between templates is the type key."""
  try:
    return RATIONALIZE_TASK_DESCRIPTION_BY_INSIGHT_TYPE[insight_type]
  except KeyError as e:
    raise KeyError(
      f"No task description for insight_type={insight_type!r}; add an entry to RATIONALIZE_TASK_DESCRIPTION_BY_INSIGHT_TYPE."
    ) from e


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
  insight_type: str,
  top_transactions_recent_period: str,
  top_transactions_previous_period: str,
  *,
  previous_outcomes: dict[int | str, str] | list[str] | str | None = None,
  latest_result_summary: str | None = None,
  latest_outcome_reflection: str | None = None,
) -> str:
  previous_outcomes_block = _format_previous_outcomes(previous_outcomes)
  body = f"""# Task Description

{task_description}

# Insight

{insight}

# Insight type

`{insight_type}`

# Top transactions — recent period

Sort: amount descending, then date descending.

{top_transactions_recent_period}

# Top transactions — previous period

Sort: amount descending, then date descending.

{top_transactions_previous_period}"""
  if previous_outcomes_block.strip():
    body += f"""

## Previous Outcomes

{previous_outcomes_block}"""
  if latest_result_summary is not None or latest_outcome_reflection is not None:
    lrs = latest_result_summary if latest_result_summary is not None else ""
    lor = latest_outcome_reflection if latest_outcome_reflection is not None else ""
    body += f"""

## Most Recent Attempt Result

{lrs}

## Outcome & What to do Next

{lor}"""
  return body


def _run_test_with_logging(
  task_description: str,
  insight: str,
  insight_type: str,
  top_transactions_recent_period: str,
  top_transactions_previous_period: str,
  optimizer: StrategizerOptimizer | None = None,
  mock_execution_result: str | None = None,
  output: str | None = None,
  previous_outcomes: dict[int | str, str] | list[str] | str | None = None,
  latest_result_summary: str | None = None,
  latest_outcome_reflection: str | None = None,
  prompt_override: str | None = None,
):
  if optimizer is None:
    optimizer = StrategizerOptimizer()
  mock_output = mock_execution_result
  execution_result = None

  if prompt_override is not None:
    llm_input_display = prompt_override
  else:
    llm_input_display = _format_rationalize_change_prompt(
      task_description,
      insight,
      insight_type,
      top_transactions_recent_period,
      top_transactions_previous_period,
      previous_outcomes=previous_outcomes,
      latest_result_summary=latest_result_summary,
      latest_outcome_reflection=latest_outcome_reflection,
    )

  print("## LLM Input\n")
  print(llm_input_display)
  print()

  llm_out = optimizer.generate_response(
    task_description,
    insight,
    insight_type,
    top_transactions_recent_period,
    top_transactions_previous_period,
    previous_outcomes=previous_outcomes,
    latest_result_summary=latest_result_summary,
    latest_outcome_reflection=latest_outcome_reflection,
    prompt_override=prompt_override,
  )

  print("## LLM Output:\n")
  print(llm_out)
  print()

  code = extract_python_code(llm_out)

  if code:
    try:
      def wrapped_lookup(*args, **kwargs):
        if mock_output is not None:
          return True, mock_output
        return lookup_user_accounts_transactions_income_and_spending_patterns(*args, **kwargs)

      namespace = {
        "lookup_user_accounts_transactions_income_and_spending_patterns": wrapped_lookup,
      }

      exec(code, namespace)

      if "execute_plan" in namespace:
        execution_result = namespace["execute_plan"]()
        print("\n## Execution Final Result:\n")
        print("```")
        print(f"  success: {execution_result[0]}")
        print(f"  output: {execution_result[1]}")
        print("```")
      else:
        print("Warning: execute_plan() function not found in generated code")
    except Exception as e:
      print(f"Error executing generated code: {str(e)}")
      import traceback
      print(traceback.format_exc())

  if output:
    print(f"\n## Output:\n\n{output}\n")

  return execution_result


MOCK_LOOKUP_SHOPPING_RECONCILE = """--- Shopping category — current month detail (lookup; forecast reconciliation) ---
Twenty shopping_* charges at $46.00 each (total $920); each is below the global top-10 floor ($48.00) so none appear in the excerpt. Sort: amount desc, date desc.
- On 2026-03-30, $46.00 Amazon (shopping_online).
- On 2026-03-29, $46.00 Target (shopping_general).
- On 2026-03-28, $46.00 Best Buy (shopping_electronics).
- On 2026-03-27, $46.00 Costco (shopping_warehouse).
- On 2026-03-26, $46.00 Walmart (shopping_general).
- On 2026-03-25, $46.00 Etsy (shopping_online).
- On 2026-03-24, $46.00 Apple Store (shopping_electronics).
- On 2026-03-23, $46.00 Nordstrom (shopping_clothing).
- On 2026-03-22, $46.00 Home Depot (shopping_home).
- On 2026-03-21, $46.00 Kohl's (shopping_general).
- On 2026-03-20, $46.00 Amazon (shopping_online).
- On 2026-03-19, $46.00 Target (shopping_general).
- On 2026-03-18, $46.00 Best Buy (shopping_electronics).
- On 2026-03-17, $46.00 Costco (shopping_warehouse).
- On 2026-03-16, $46.00 Walmart (shopping_general).
- On 2026-03-15, $46.00 Etsy (shopping_online).
- On 2026-03-14, $46.00 Apple Store (shopping_electronics).
- On 2026-03-13, $46.00 Nordstrom (shopping_clothing).
- On 2026-03-12, $46.00 Home Depot (shopping_home).
- On 2026-03-11, $46.00 Kohl's (shopping_general).
Category subtotal (current month shopping_*): $920. Monthly shopping forecast ~$200.
"""


def _sample_recent_txns_leisure_down():
  """Top-style list: amount descending, then date descending."""
  return """- On 2026-03-30, $142.10 Whole Foods (meals_groceries).
- On 2026-03-25, $62.00 Shell Gas (transportation_gas).
- On 2026-03-08, $58.00 Thai Garden (meals_dining_out).
- On 2026-03-28, $48.00 AMC Theaters (entertainment_movies).
- On 2026-03-10, $35.00 City Parking (transportation_parking).
- On 2026-03-18, $23.50 Uber (transportation_rideshare).
- On 2026-03-05, $19.20 CVS (health_pharmacy).
- On 2026-03-02, $15.99 Netflix (entertainment_streaming).
- On 2026-03-15, $10.99 Spotify (entertainment_streaming).
- On 2026-03-22, $0.00 Delta Air (travel_flights)."""


def _sample_prev_txns_leisure_down():
  """Top-style list: amount descending, then date descending."""
  return """- On 2026-02-12, $890.00 Marriott (travel_lodging).
- On 2026-02-22, $412.00 Delta Air (travel_flights).
- On 2026-02-28, $220.00 StubHub (entertainment_events).
- On 2026-02-03, $180.00 Concert Hall (entertainment_events).
- On 2026-02-15, $128.00 Whole Foods (meals_groceries).
- On 2026-02-20, $64.00 AMC Theaters (entertainment_movies).
- On 2026-02-05, $55.00 Shell Gas (transportation_gas).
- On 2026-02-10, $31.00 Uber (transportation_rideshare).
- On 2026-02-08, $15.99 Netflix (entertainment_streaming).
- On 2026-02-01, $10.99 Spotify (entertainment_streaming)."""


def _sample_recent_txns_shopping_mismatch():
  """Top-10 by amount (global); no shopping_* lines because shopping is many $46 txs below $48 floor. Amount desc, date desc."""
  return """- On 2026-03-28, $118.40 Whole Foods (meals_groceries).
- On 2026-03-20, $67.10 Trader Joe's (meals_groceries).
- On 2026-03-25, $62.00 Peak Fitness (health_gym).
- On 2026-03-23, $58.00 Smile Dental (health_dental).
- On 2026-03-15, $55.00 City Parking (transportation_parking).
- On 2026-03-19, $54.00 Urban Pet (pets_supplies).
- On 2026-03-17, $53.00 Downtown Dry Clean (services_laundry).
- On 2026-03-14, $52.00 Ace Hardware (home_improvement).
- On 2026-03-11, $51.00 Campus Books (education_books).
- On 2026-03-12, $48.00 Thai Garden (meals_dining_out)."""


def _sample_prev_txns_shopping_mismatch():
  """Amount desc, then date desc."""
  return """- On 2026-02-25, $105.00 Whole Foods (meals_groceries).
- On 2026-02-14, $72.00 Trader Joe's (meals_groceries).
- On 2026-02-05, $44.00 Thai Garden (meals_dining_out).
- On 2026-02-08, $40.00 City Parking (transportation_parking).
- On 2026-02-27, $38.90 Shell Gas (transportation_gas).
- On 2026-02-18, $28.00 Uber (transportation_rideshare).
- On 2026-02-20, $21.00 CVS (health_pharmacy).
- On 2026-02-02, $19.00 Walgreens (health_pharmacy).
- On 2026-02-22, $18.50 Starbucks (meals_coffee).
- On 2026-02-10, $15.99 Netflix (entertainment_streaming)."""


TEST_CASES = [
  {
    "batch": 1,
    "name": "rationalize_leisure_down_sufficient_context",
    "insight_type": "month_spend_vs_forecast",
    "task_description": (
      "Explain **month-to-date actual spend vs forecast**: how the user's spending differs from plan and the most plausible "
      "drivers (merchants, categories, timing). Ground the answer in the Insight field and in the recent- and prior-period "
      "transaction excerpts; those lists are supporting context only, not the forecast baseline. Keep the explanation concise "
      "for a dashboard. When the insight states amounts or categories, reconcile them with the excerpts where you can; if a "
      "named category or roll-up is missing from the excerpts or the numbers do not line up, call lookup once with a narrow "
      "request, then give the final explanation."
    ),
    "insight": (
      "Entertainment is significantly below forecast this month at $309. Travel & Vacations is significantly below forecast at $47. "
      "Leisure is thus significantly below forecast this month at $356."
    ),
    "top_transactions_recent_period": _sample_recent_txns_leisure_down(),
    "top_transactions_previous_period": _sample_prev_txns_leisure_down(),
    "output": "Expected: execute_plan returns (True, explanation) without lookup—large travel/event lines in the previous-period list explain leisure below forecast vs lighter entertainment/travel in the recent list.",
    "mock_execution_result": None,
  },
  {
    "batch": 1,
    "name": "rationalize_shopping_up_requires_lookup",
    "insight_type": "month_spend_vs_forecast",
    "task_description": (
      "Explain **month-to-date actual spend vs forecast**: how the user's spending differs from plan and the most plausible "
      "drivers (merchants, categories, timing). Ground the answer in the Insight field and in the recent- and prior-period "
      "transaction excerpts; those lists are supporting context only, not the forecast baseline. Keep the explanation concise "
      "for a dashboard. When the insight states amounts or categories, reconcile them with the excerpts where you can; if a "
      "named category or roll-up is missing from the excerpts or the numbers do not line up, call lookup once with a narrow "
      "request, then give the final explanation."
    ),
    "insight": "Shopping is significantly above forecast this month at $920 vs ~$200 expected for shopping.",
    "top_transactions_recent_period": _sample_recent_txns_shopping_mismatch(),
    "top_transactions_previous_period": _sample_prev_txns_shopping_mismatch(),
    "output": (
      "Expected: execute_plan calls lookup once; output reflects many sub-floor $46 shopping_* charges totaling $920 (consistent with $48.00 10th-place global top txn)."
    ),
    "mock_execution_result": MOCK_LOOKUP_SHOPPING_RECONCILE,
  },
  {
    "batch": 1,
    "name": "rationalize_week_spend_followup_turn",
    "insight_type": "week_spend_vs_forecast",
    "task_description": (
      "Explain **week-to-date actual spend vs forecast**: how the user's spending differs from plan and the most plausible "
      "drivers (merchants, categories, timing). Ground the answer in the Insight field and in the recent- and prior-period "
      "transaction excerpts; those lists are supporting context only, not the forecast baseline. Keep the explanation concise "
      "for a dashboard. When the insight states amounts or categories, reconcile them with the excerpts where you can; if a "
      "named category or roll-up is missing from the excerpts or the numbers do not line up, call lookup once with a narrow "
      "request, then give the final explanation."
    ),
    "insight": "Dining out is significantly above your weekly forecast at about $180 vs ~$45 planned.",
    "top_transactions_recent_period": """- On 2026-03-27, $95.00 Whole Foods (meals_groceries).
- On 2026-03-31, $62.00 Olive Garden (meals_dining_out).
- On 2026-03-29, $44.00 DoorDash (meals_dining_out).
- On 2026-03-30, $18.50 Chipotle (meals_dining_out).
- On 2026-03-28, $12.00 Starbucks (meals_coffee).""",
    "top_transactions_previous_period": """- On 2026-03-21, $72.00 Trader Joe's (meals_groceries).
- On 2026-03-20, $22.00 CVS (health_pharmacy).
- On 2026-03-24, $14.00 Chipotle (meals_dining_out).
- On 2026-03-22, $8.00 Starbucks (meals_coffee).
- On 2026-03-23, $0.00 Home cooking transfer (internal).""",
    "previous_outcomes": {
      1: "Lookup returned no extra rows; prior attempt asked for duplicate week window.",
    },
    "latest_result_summary": "lookup_user_accounts_transactions_income_and_spending_patterns returned empty supplemental list for the same 7-day window.",
    "latest_outcome_reflection": "Answer from provided top transactions only; do not repeat the same lookup request.",
    "output": "Expected: direct rationalization vs weekly dining forecast using top transactions (more dining lines and higher tickets), no redundant lookup.",
    "mock_execution_result": None,
  },
]

for _tc in TEST_CASES:
  _tid = _tc["insight_type"]
  if _tc["task_description"] != RATIONALIZE_TASK_DESCRIPTION_BY_INSIGHT_TYPE[_tid]:
    raise AssertionError(
      f"Test {_tc['name']!r}: task_description must equal RATIONALIZE_TASK_DESCRIPTION_BY_INSIGHT_TYPE[{_tid!r}] "
      "(only insight_type may choose the template)."
    )


def run_test(test_name_or_index_or_dict, optimizer: StrategizerOptimizer | None = None):
  if isinstance(test_name_or_index_or_dict, dict):
    if "input" in test_name_or_index_or_dict:
      test_name = test_name_or_index_or_dict.get("name", "custom_test")
      print(f"\n# Test: **{test_name}**\n")
      if optimizer is None:
        optimizer = StrategizerOptimizer()
      print("## LLM Input\n")
      print(test_name_or_index_or_dict["input"])
      print()
      llm_out = optimizer.generate_response(
        "",
        "",
        "",
        "",
        "",
        prompt_override=test_name_or_index_or_dict["input"],
      )
      print("## LLM Output:\n")
      print(llm_out)
      print()
      code = extract_python_code(llm_out)
      execution_result = None
      if code:
        try:
          def wrapped_lookup(*args, **kwargs):
            m = test_name_or_index_or_dict.get("mock_execution_result")
            if m is not None:
              return True, m
            return lookup_user_accounts_transactions_income_and_spending_patterns(*args, **kwargs)

          namespace = {"lookup_user_accounts_transactions_income_and_spending_patterns": wrapped_lookup}
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

    if "task_description" in test_name_or_index_or_dict:
      test_name = test_name_or_index_or_dict.get("name", "custom_test")
      print(f"\n# Test: **{test_name}**\n")
      return _run_test_with_logging(
        test_name_or_index_or_dict["task_description"],
        test_name_or_index_or_dict.get("insight", ""),
        test_name_or_index_or_dict.get("insight_type", ""),
        test_name_or_index_or_dict.get("top_transactions_recent_period", ""),
        test_name_or_index_or_dict.get("top_transactions_previous_period", ""),
        optimizer,
        mock_execution_result=test_name_or_index_or_dict.get("mock_execution_result"),
        output=test_name_or_index_or_dict.get("output"),
        previous_outcomes=test_name_or_index_or_dict.get("previous_outcomes"),
        latest_result_summary=test_name_or_index_or_dict.get("latest_result_summary"),
        latest_outcome_reflection=test_name_or_index_or_dict.get("latest_outcome_reflection"),
      )

  if isinstance(test_name_or_index_or_dict, int):
    tc = TEST_CASES[test_name_or_index_or_dict] if 0 <= test_name_or_index_or_dict < len(TEST_CASES) else None
  else:
    tc = next((t for t in TEST_CASES if t["name"] == test_name_or_index_or_dict), None)
  if not tc:
    return None
  print(f"\n# Test: **{tc['name']}**\n")
  return _run_test_with_logging(
    tc["task_description"],
    tc["insight"],
    tc["insight_type"],
    tc["top_transactions_recent_period"],
    tc["top_transactions_previous_period"],
    optimizer,
    mock_execution_result=tc.get("mock_execution_result"),
    output=tc.get("output"),
    previous_outcomes=tc.get("previous_outcomes"),
    latest_result_summary=tc.get("latest_result_summary"),
    latest_outcome_reflection=tc.get("latest_outcome_reflection"),
  )


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
  tb = 0 if no_thinking else (thinking_budget if thinking_budget is not None else 700)
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

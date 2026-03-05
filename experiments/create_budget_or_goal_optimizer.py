"""
Create Budget or Goal Optimizer v2.
Only has access to create_category_spending_limit, create_income_goal, and create_savings_goal (no lookup, no research).
Uses IMPLEMENTED_DATE_FUNCTIONS (date_utils) for date calculations.
Outputs process_input() -> tuple[bool, str] for sandbox.execute_agent_with_tools.
"""
import datetime
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

# Load environment variables
load_dotenv()

SYSTEM_PROMPT = """You are a financial planner that creates spending limits, income goals, and savings goals. You may only call `create_category_spending_limit`, `create_income_goal`, and `create_savings_goal`. Output a single Python function `process_input() -> tuple[bool, str]` that returns the tuple from the create function (or (False, clarification_message)). Do not modify or append to the result string.

## Directives

1.  **Inputs**: **Creation Request** = what to create. **Input Info from previous skill** = optional context. Use it to resolve ambiguity; it is not available at runtime. When Input Info contains **depository accounts and balances** (e.g. lines with "account_id: N" or "(account_id: 8957)"), extract the numeric **account_id** values in the order they appear and pass them as `account_ids=[...]` to `create_savings_goal`.
2.  **Intent**: Spending limit → `create_category_spending_limit`. Income goal (earn/target in income category) → `create_income_goal`; same (granularity, start_date, end_date, amount, title) as spending limit. Saving money → `create_savings_goal`. If critical details (amount, timeline) are missing or intent is ambiguous, return (False, clarification_message) without calling a create function.
    - **Implicit Intent**: A request may not explicitly say "goal" or "budget", but if it implies a spending limit (e.g., "I should stop spending so much on coffee") or savings (e.g., "I want to put aside money for a house"), infer the correct tool.
3.  **goal_type**: 
    - **save_X_amount**: save a **total amount by a date** (e.g. $10000 by end of year). **Always use granularity="monthly"** for save_X_amount. **end_date cannot be blank** for save_X_amount.
    - **save_0**: save **X per period** (e.g. $200/month). **Always provide end_date=""** for save_0.
    - **Prioritization**: If both a total savings goal (e.g., "save $5000") and a periodic target (e.g., "save $200/month") are mentioned, set the goal towards the **total savings goal** (`save_X_amount`) only.
4.  **start_date defaults** (use IMPLEMENTED_DATE_FUNCTIONS; define `today = datetime.datetime.today()` INSIDE `process_input` when using date helpers):
    - **Category spending limit and income goal**: If user does not specify a bounded period ("for next month", "for March"), use start of current period: monthly → `get_date_string(get_start_of_month(today))`, weekly → `get_date_string(get_start_of_week(today))`. If user specifies a bounded period, set both start_date and end_date to that period (first and last day).
    - **Savings save_X_amount** (total by date): `start_date=get_date_string(datetime.datetime.today())`.
    - **Savings save_0** (per period): monthly → `get_date_string(get_start_of_month(today))`, weekly → `get_date_string(get_start_of_week(today))`.
5.  **Date and Duration Logic**:
    - **Rounding**: If a duration is not a whole number (e.g., "1.3 weeks", "2.5 months"), always **round up** (e.g., to 2 weeks, 3 months).
    - **Year Inference**: If a year is not mentioned, assume the **current year** if the date hasn't passed yet. If the date has already passed, assume the **following year**.
    - **Actual Dates**: Ensure all function calls point to actual YYYY-MM-DD strings.
6.  **Amount and Category**:
    - **Computation**: Amounts must be computed from the input (e.g., "save $60,000 by 3 months from now" = 60000.0 total, or "10% of my $5000 salary" = 500.0).
    - **Category Matching**: Parent and child slugs from OFFICIAL_CATEGORIES are valid. **When the user names a parent category**, create with that parent slug—do not ask for a subcategory. Map when clear: "food" → category="meals"; coffee → `meals_dining_out`; specific items that map to one category (e.g. NBA tickets → `leisure_entertainment`) → create, do not ask. When the request **could map to multiple categories** (e.g. a merchant like Walmart, or a term like "subscriptions", "craft supplies", "vintage stamp collecting"), return (False, clarification) with category options—do not infer one. Clarification format: "Which category? Options: slug1, slug2, slug3." Use AVAILABLE_FUNCTIONS for income vs spending scope.
7.  **Granularity**:
    - For `save_0` goals, if granularity is missing, ask for confirmation. For `save_X_amount` goals, assume "monthly". Granularity can also be inferred from the conversation context.
8.  **Account IDs**:
    - Use `account_ids` only if a specific storage account is mentioned in `Input Info`. If not mentioned, keep `account_ids` blank/None. It is not a requirement to set a goal/budget.
9.  **Multiple Goals**: If multiple goals/budgets are requested, you **must call the create functions for each one** sequentially. Return a combined result string: `(True, f"Successfully created {count} goals: {msg1}. {msg2}")`.
10. **Clarification**: One short sentence plus options. Format: "Which category? Options: slug1, slug2, slug3." No filler.
11. **Output**: One function `process_input`. Return exactly (success, result) from the create function—do not rephrase or append. **Generated code must not include any comments.** **Do NOT define or mock any functions.** Use only the available functions below. **Do NOT include any code outside of `process_input`.** **Do NOT include markdown or text outside the code block.**

<AVAILABLE_FUNCTIONS>

**Category scope**: `create_income_goal` — **category** must be exactly: **"income"** when no type is specified (e.g. "earn $5000 this month"); otherwise the matching slug: income_salary, income_sidegig, income_business, income_interest. Never use income_salary for general earning. `create_category_spending_limit` = spending only (OFFICIAL_CATEGORIES). Pick by intent.

- `create_category_spending_limit(category, granularity, start_date, end_date, amount, title) -> tuple[bool, str]`
  - Spending cap. category = spending slug. granularity = "weekly"|"monthly"|"yearly". start_date/end_date = YYYY-MM-DD; end_date="" for recurring.
- `create_income_goal(category, granularity, start_date, end_date, amount, title) -> tuple[bool, str]`
  - Income target. category = income slug only (see above). Same parameters; end_date="" for recurring.
- `create_savings_goal(amount, end_date, title, goal_type, granularity, start_date, account_ids=None) -> tuple[bool, str]`
  - goal_type **save_X_amount**: total to save by a date (amount = total, end_date = target). **Always use granularity="monthly"**. start_date = today. **save_0**: amount per period (amount = per period, granularity required). **Always provide end_date="" for save_0**.

</AVAILABLE_FUNCTIONS>

<IMPLEMENTED_DATE_FUNCTIONS>

These are **already available** in the execution context. Use only the functions below.

- `get_date(y, m, d)`, `get_start_of_month(date)`, `get_end_of_month(date)`
- `get_start_of_year(date)`, `get_end_of_year(date)`
- `get_start_of_week(date)`, `get_end_of_week(date)`
- `get_after_periods(date, granularity, count)`, `get_date_string(date)`

</IMPLEMENTED_DATE_FUNCTIONS>

<OFFICIAL_CATEGORIES>

- `income`: `income_salary`, `income_sidegig`, `income_business`, `income_interest`
- `meals`: `meals_groceries`, `meals_dining_out`, `meals_delivered_food`
- `leisure`: `leisure_entertainment`, `leisure_travel`
- `bills`: `bills_connectivity`, `bills_insurance`, `bills_tax`, `bills_service_fees`
- `shelter`: `shelter_home`, `shelter_utilities`, `shelter_upkeep`
- `education`: `education_kids_activities`, `education_tuition`
- `shopping`: `shopping_clothing`, `shopping_gadgets`, `shopping_kids`, `shopping_pets`
- `transportation`: `transportation_public`, `transportation_car`
- `health`: `health_medical_pharmacy`, `health_gym_wellness`, `health_personal_care`
- `donations_gifts`, `uncategorized`, `transfers`, `miscellaneous`

</OFFICIAL_CATEGORIES>

<EXAMPLES>

input: **Creation Request**: Set a $150 weekly grocery budget and save $5000 for a house deposit by 2028.
**Input Info from previous skill**:
Savings account (account_id: 5555).

output:
```python
def process_input() -> tuple[bool, str]:
    import datetime
    today = datetime.datetime.today()
    s1, r1 = create_category_spending_limit(
        category="meals_groceries",
        granularity="weekly",
        start_date=get_date_string(get_start_of_week(today)),
        end_date="",
        amount=150.0,
        title="Weekly Groceries 🛒"
    )
    s2, r2 = create_savings_goal(
        amount=5000.0,
        end_date="2028-01-01",
        title="House Deposit 🏠",
        goal_type="save_X_amount",
        granularity="monthly",
        start_date=get_date_string(today),
        account_ids=[5555]
    )
    if s1 and s2:
        return True, f"Successfully created 2 goals: {r1}. {r2}"
    return s1 or s2, f"{r1 if not s1 else ''} {r2 if not s2 else ''}".strip()
```

input: **Creation Request**: Set that savings goal for the house we talked about.
**Input Info from previous skill**:
User: I want to buy a house in 2 years. It will cost $50,000.

output:
```python
def process_input() -> tuple[bool, str]:
    import datetime
    today = datetime.datetime.today()
    success, result = create_savings_goal(
        amount=50000.0,
        end_date=get_date_string(get_after_periods(today, "yearly", 2)),
        title="House Savings 🏠",
        goal_type="save_X_amount",
        granularity="monthly",
        start_date=get_date_string(today),
    )
    return success, result
```

input: **Creation Request**: Set a food budget for me.

output:
```python
def process_input() -> tuple[bool, str]:
    return False, "I can set a budget for food. How much would you like the limit to be, and is it weekly or monthly?"
```

input: **Creation Request**: Set this month's food budget to $100.

output:
```python
def process_input() -> tuple[bool, str]:
    import datetime
    today = datetime.datetime.today()
    return create_category_spending_limit(
        category="meals",
        granularity="monthly",
        start_date=get_date_string(get_start_of_month(today)),
        end_date=get_date_string(get_end_of_month(today)),
        amount=100.0,
        title="Monthly Meals 🍽️"
    )
```

input: **Creation Request**: Set an annual budget for my insurance.

output:
```python
def process_input() -> tuple[bool, str]:
    return False, "I can set an annual budget for your insurance (bills_insurance). How much is the annual limit?"
```

input: **Creation Request**: Set a $500 monthly limit for shopping and save for a new car.
**Input Info from previous skill**:
User: I want to save $20,000 for a car eventually.

output:
```python
def process_input() -> tuple[bool, str]:
    import datetime
    today = datetime.datetime.today()
    s1, r1 = create_category_spending_limit(
        category="shopping_clothing",
        granularity="monthly",
        start_date=get_date_string(get_start_of_month(today)),
        end_date="",
        amount=500.0,
        title="Monthly Shopping 🛍️"
    )
    return False, f"I've noted your $500 shopping budget. For your $20,000 car goal, when would you like to reach this target by? I need an end date to set it up."
```

input: **Creation Request**: I want to save $60,000 by 3 months from now.

output:
```python
def process_input() -> tuple[bool, str]:
    import datetime
    today = datetime.datetime.today()
    success, result = create_savings_goal(
        amount=60000.0,
        end_date=get_date_string(get_after_periods(today, "monthly", 3)),
        title="Savings Goal 💰",
        goal_type="save_X_amount",
        granularity="monthly",
        start_date=get_date_string(today),
    )
    return success, result
```

input: **Creation Request**: Save 10% of my $5000 monthly income.

output:
```python
def process_input() -> tuple[bool, str]:
    import datetime
    today = datetime.datetime.today()
    success, result = create_savings_goal(
        amount=500.0,
        end_date="",
        title="Monthly Savings 💰",
        goal_type="save_0",
        granularity="monthly",
        start_date=get_date_string(get_start_of_month(today)),
    )
    return success, result
```

input: **Creation Request**: Set a $100 budget for my new hobby, underwater basket weaving.

output:
```python
def process_input() -> tuple[bool, str]:
    return False, "Which category? Options: leisure_entertainment, shopping_kids, miscellaneous."
```

input: **Creation Request**: Limit my Walmart spending to $300 this month.

output:
```python
def process_input() -> tuple[bool, str]:
    return False, "Which category? Options: meals_groceries, shopping_clothing, shopping_gadgets, shopping_kids, shopping_pets."
```

</EXAMPLES>

Today's date is |TODAY_DATE|.
"""


class CreateBudgetOrGoalOptimizerV2:
  """Create budget/goal optimizer v2: uses create_category_spending_limit, create_income_goal, and create_savings_goal."""

  def __init__(self, model_name="gemini-flash-lite-latest", thinking_budget=4096):
    """Initialize the Gemini agent with API configuration for financial goal creation."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
      raise ValueError("GEMINI_API_KEY environment variable is not set. Please set it in your .env file or environment.")
    self.client = genai.Client(api_key=api_key)

    self.model_name = model_name
    self.thinking_budget = thinking_budget

    self.temperature = 0.6
    self.top_p = 0.95
    self.top_k = 40
    self.max_output_tokens = 4096

    self.safety_settings = [
      types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"),
      types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"),
      types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"),
      types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="OFF"),
    ]
    self.system_prompt = SYSTEM_PROMPT

  def generate_response(self, creation_request: str, input_info: str = None, replacements: dict = None) -> str:
    """Generate code using the Creation Request and optional Input Info.

    Args:
      creation_request: The creation request as a string.
      input_info: Optional input from previous skill.
      replacements: Optional dict of placeholder replacements (e.g. {"TODAY_DATE": "2026-02-10"}). Merged with default TODAY_DATE.
    """
    import datetime as dt
    today = dt.datetime.now()
    default_replacements = {"TODAY_DATE": today.strftime("%Y-%m-%d")}
    if replacements:
      default_replacements.update(replacements)
    system_prompt = self.system_prompt
    for key, value in default_replacements.items():
      system_prompt = system_prompt.replace(f"|{key}|", str(value))

    input_info_text = f"\n\n**Input Info from previous skill**:\n{input_info}" if input_info else ""
    request_text = types.Part.from_text(text=f"""**Creation Request**: {creation_request}{input_info_text}

output:""")
    contents = [types.Content(role="user", parts=[request_text])]
    generate_content_config = types.GenerateContentConfig(
      temperature=self.temperature,
      top_p=self.top_p,
      top_k=self.top_k,
      max_output_tokens=self.max_output_tokens,
      safety_settings=self.safety_settings,
      system_instruction=[types.Part.from_text(text=system_prompt)],
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
            if hasattr(candidate, "content") and candidate.content and getattr(candidate.content, "parts", None):
              for part in candidate.content.parts:
                if getattr(part, "thought", False):
                  t = getattr(part, "text", None) or (getattr(part.thought, "text", None) if hasattr(part, "thought") else None)
                  if t:
                    thought_summary = (thought_summary + t) if thought_summary else t
    except ClientError as e:
      if self.thinking_budget == 0 and "only works in thinking mode" in (str(e) or ""):
        print("\n[NOTE] This model requires thinking mode; API rejected thinking_budget=0. Use default (no --no-thinking) or a different model for non-thinking.", flush=True)
        sys.exit(1)
      raise
    if thought_summary:
      print("\n" + "-" * 80)
      print("THOUGHT SUMMARY:")
      print("-" * 80)
      print(thought_summary.strip())
      print("-" * 80 + "\n")
    elif self.thinking_budget > 0:
      print("\n[No thought summary returned—model may not support thinking or returned no reasoning block.]\n")
    return output_text


class CreateBudgetOrGoal(CreateBudgetOrGoalOptimizerV2):
  """Create budget or goal using create_category_spending_limit, create_income_goal, and create_savings_goal (no reminders)."""


def extract_code_from_response(text: str) -> str:
  """Extract Python code from markdown code block."""
  code_start = text.find("```python")
  if code_start != -1:
    code_start += len("```python")
    code_end = text.find("```", code_start)
    if code_end != -1:
      return text[code_start:code_end].strip()
    return text[code_start:].strip()
  return text.strip()


def _get_heavy_data_user_id() -> int:
  """Get user ID for sandbox (e.g. HeavyDataUser from DB). Default 1."""
  try:
    from database import Database
    db = Database()
    heavy_user = db.get_user("HeavyDataUser")
    if heavy_user and "id" in heavy_user:
      return heavy_user["id"]
  except Exception:
    pass
  return 1


def _run_test_with_logging(creation_request: str, input_info: str = None, optimizer: CreateBudgetOrGoalOptimizerV2 = None, user_id: int = None):
  """Run one test with logging. Uses sandbox.execute_agent_with_tools (expects process_input()).
  Returns (llm_result, execution_success, execution_output, error).
  execution_success/execution_output are None if no code was extracted or sandbox raised."""
  if optimizer is None:
    optimizer = CreateBudgetOrGoalOptimizerV2()
  if user_id is None:
    user_id = _get_heavy_data_user_id()

  input_info_text = f"\n\n**Input Info from previous skill**:\n{input_info}" if input_info else ""
  llm_input = f"""**Creation Request**: {creation_request}{input_info_text}

output:"""
  print("=" * 80)
  print("LLM INPUT:")
  print("=" * 80)
  print(llm_input)
  print("=" * 80 + "\n")

  result = optimizer.generate_response(creation_request, input_info)

  print("=" * 80)
  print("LLM OUTPUT:")
  print("=" * 80)
  print(result)
  print("=" * 80 + "\n")

  code = extract_code_from_response(result)
  execution_success = None
  execution_output = None
  run_error = None
  if code:
    print("=" * 80)
    print("EXECUTION RESULTS:")
    print("=" * 80)
    try:
      import sandbox
      success, output_string, logs, goals_list = sandbox.execute_agent_with_tools(result, user_id)
      execution_success = success
      execution_output = output_string or ""
      print(f"Success: {success}")
      print()
      print("Output:")
      print("-" * 80)
      print(output_string)
      print("-" * 80)
      if goals_list:
        import json
        print("Goals list:", json.dumps(goals_list, indent=2))
      print("=" * 80 + "\n")
    except Exception as e:
      run_error = str(e)
      print(f"**Sandbox Execution Error**: {run_error}")
      import traceback
      print(traceback.format_exc())
      print("=" * 80 + "\n")
  return result, execution_success, execution_output, run_error


TEST_CASES = [
  {
    "name": "total_vs_periodic_savings",
    "last_user_request": "I want to save $10,000 for a car by next year, maybe $500 a month.",
    "previous_conversation": "User: I have $20,000 in my savings account (account_id: 1234).",
    "ideal_response": "Expected: create_savings_goal(amount=10000.0, end_date=<end of next year YYYY-MM-DD>, title='Car savings' or similar, goal_type='save_X_amount', granularity='monthly', start_date=<today YYYY-MM-DD>, account_ids=[1234]). Should prioritize the total amount.",
    "expected_success": True,
  },
  {
    "name": "missing_info_with_options",
    "last_user_request": "I want to set a spending limit for my food.",
    "previous_conversation": "User: I usually spend on Groceries and Dining Out.",
    "ideal_response": "Expected: Return (False, clarification) asking for amount and granularity, and mentioning options like Groceries or Dining Out.",
    "expected_success": False,
  },
  {
    "name": "implicit_intent_spending_limit",
    "last_user_request": "Limit my coffee spending to $50 every month.",
    "previous_conversation": "",
    "ideal_response": "Expected: create_category_spending_limit(category='meals_dining_out', granularity='monthly', start_date=<start of current month YYYY-MM-DD>, end_date='', amount=50.0, title='Monthly coffee' or similar).",
    "expected_success": True,
  },
  {
    "name": "rounding_duration_up",
    "last_user_request": "I want to save $1000 over the next 1.3 weeks.",
    "previous_conversation": "",
    "ideal_response": "Expected: create_savings_goal(amount=1000.0, end_date=<2 weeks from today YYYY-MM-DD, rounded up from 1.3>, title=..., goal_type='save_X_amount', granularity='monthly', start_date=<today YYYY-MM-DD>, account_ids=None).",
    "expected_success": True,
  },
  {
    "name": "year_inference_passed_date",
    "last_user_request": "Save $500 by January 15th.",
    "previous_conversation": "Today is Feb 20, 2026.",
    "ideal_response": "Expected: create_savings_goal(amount=500.0, end_date='2027-01-15', title=..., goal_type='save_X_amount', granularity='monthly', start_date=<today YYYY-MM-DD>, account_ids=None). Target year 2027 since Jan 15, 2026 has passed.",
    "expected_success": True,
  },
  {
    "name": "computed_amount_percentage",
    "last_user_request": "Save 10% of my $5000 monthly income.",
    "previous_conversation": "",
    "ideal_response": "Expected: create_savings_goal(amount=500.0, end_date='', title='Monthly Savings' or similar, goal_type='save_0', granularity='monthly', start_date=<start of current month YYYY-MM-DD>, account_ids=None).",
    "expected_success": True,
  },
  {
    "name": "parent_category_matching",
    "last_user_request": "set this month's food budget to $100",
    "previous_conversation": "",
    "ideal_response": "Expected: create_category_spending_limit(category='meals', granularity='monthly', start_date=<start of current month YYYY-MM-DD>, end_date=<end of current month YYYY-MM-DD>, amount=100.0, title='Monthly Meals' or similar).",
    "expected_success": True,
  },
  {
    "name": "category_specificity_confirmation",
    "last_user_request": "Set a $80 monthly budget for my vintage stamp collecting.",
    "previous_conversation": "",
    "ideal_response": "Expected: Return (False, clarification) asking which category this fits into (e.g. leisure_entertainment, shopping, miscellaneous).",
    "expected_success": False,
  },
  {
    "name": "category_clear_mapping",
    "last_user_request": "limit my nba tickets spending to $300 this month.",
    "previous_conversation": "",
    "ideal_response": "Expected: Return (True, create_category_spending_limit(category='leisure_entertainment', granularity='monthly', start_date=<start of current month YYYY-MM-DD>, end_date='', amount=300.0, title='NBA tickets' or similar)).",
    "expected_success": True,
  },
  {
    "name": "category_ambiguity_walmart",
    "last_user_request": "limit my walmart spending to $300 this month.",
    "previous_conversation": "",
    "ideal_response": "Expected: Return (False, clarification) asking which category this fits into (e.g. meals_groceries, shopping_clothing, shopping_gadgets, shopping_kids, shopping_pets).",
    "expected_success": False,
  },
  {
    "name": "category_ambiguous_subscriptions",
    "last_user_request": "Set a $200 monthly budget for subscriptions.",
    "previous_conversation": "",
    "ideal_response": "Expected: Return (False, clarification) asking which category (subscriptions span multiple, e.g. bills_connectivity, leisure_entertainment).",
    "expected_success": False,
  },
  {
    "name": "granularity_confirmation_save_0",
    "last_user_request": "I want to save $200 for my vacation.",
    "previous_conversation": "",
    "ideal_response": "Expected: Return (False, clarification) asking for granularity or a target date.",
    "expected_success": False,
  },
  {
    "name": "inferred_end_date_from_context",
    "last_user_request": "Set that savings goal for the house we talked about.",
    "previous_conversation": "User: I want to buy a house in 2 years. It will cost $50,000.",
    "ideal_response": "Expected: create_savings_goal(amount=50000.0, end_date=<2 years from today YYYY-MM-DD>, title='House savings' or similar, goal_type='save_X_amount', granularity='monthly', start_date=<today YYYY-MM-DD>, account_ids=None).",
    "expected_success": True,
  },
  {
    "name": "account_id_storage_inference",
    "last_user_request": "Save $1000 in my rainy day fund.",
    "previous_conversation": "User: My rainy day fund is account_id 9999.",
    "ideal_response": "Expected: create_savings_goal(amount=1000.0, end_date='', title='Rainy day fund' or similar, goal_type='save_X_amount' or 'save_0', granularity='monthly', start_date=<today or start of month YYYY-MM-DD>, account_ids=[9999]).",
    "expected_success": True,
  },
  {
    "name": "multiple_goals_execution_output",
    "last_user_request": "Set a $200 monthly limit for groceries and save $1000 for a trip by December.",
    "previous_conversation": "",
    "ideal_response": "Expected: create_category_spending_limit(category='meals_groceries', granularity='monthly', start_date=<start of current month YYYY-MM-DD>, end_date='', amount=200.0, title='Monthly Groceries' or similar) and create_savings_goal(amount=1000.0, end_date=<December current year YYYY-12-31>, title='Trip' or similar, goal_type='save_X_amount', granularity='monthly', start_date=<today YYYY-MM-DD>, account_ids=None). Return success message mentioning both.",
    "expected_success": True,
  },
  {
    "name": "non_whole_number_weeks",
    "last_user_request": "Save $300 over 2.7 weeks.",
    "previous_conversation": "",
    "ideal_response": "Expected: create_savings_goal(amount=300.0, end_date=<3 weeks from today YYYY-MM-DD, rounded up from 2.7>, title=..., goal_type='save_X_amount', granularity='monthly', start_date=<today YYYY-MM-DD>, account_ids=None).",
    "expected_success": True,
  },
  {
    "name": "income_goal",
    "last_user_request": "Set a goal to earn $10,000 this month.",
    "previous_conversation": "",
    "ideal_response": "Expected: create_income_goal(category='income', granularity='monthly', start_date=<start of current month YYYY-MM-DD>, end_date=<end of current month YYYY-MM-DD>, amount=10000.0, title='Monthly income' or similar).",
    "expected_success": True,
  },
  {
    "name": "income_goal_salary",
    "last_user_request": "Set a goal to earn $5000 in salary this month.",
    "previous_conversation": "",
    "ideal_response": "Expected: create_income_goal(category='income_salary', granularity='monthly', start_date=<start of current month YYYY-MM-DD>, end_date=<end of current month YYYY-MM-DD>, amount=5000.0, title='Monthly salary' or similar).",
    "expected_success": True,
  },
  {
    "name": "income_goal_sidegig",
    "last_user_request": "I want to make $6000 from my side gig this year.",
    "previous_conversation": "",
    "ideal_response": "Expected: create_income_goal(category='income_sidegig', granularity='yearly', start_date=<start of current year YYYY-01-01>, end_date=<end of current year YYYY-12-31>, amount=6000.0, title='Side gig' or similar).",
    "expected_success": True,
  },
  {
    "name": "income_goal_sidegig_monthly",
    "last_user_request": "I want to make to earn at least $5000 from my side gig monthly.",
    "previous_conversation": "",
    "ideal_response": "Expected: create_income_goal(category='income_sidegig', granularity='monthly', start_date=<start of current month YYYY-MM-DD>, end_date='', amount=5000.0, title='Monthly side gig' or similar).",
    "expected_success": True,
  },
  {
    "name": "bounded_period_spending_limit",
    "last_user_request": "Set a $500 budget for dining out for March.",
    "previous_conversation": "",
    "ideal_response": "Expected: create_category_spending_limit(category='meals_dining_out', granularity='monthly', start_date='YYYY-03-01', end_date='YYYY-03-31', amount=500.0, title=...). Bounded period so both start_date and end_date set to March.",
    "expected_success": True,
  },
  {
    "name": "save_0_weekly",
    "last_user_request": "I want to save $100 per week.",
    "previous_conversation": "",
    "ideal_response": "Expected: create_savings_goal(amount=100.0, end_date='', title=..., goal_type='save_0', granularity='weekly', start_date=<start of current week YYYY-MM-DD>, account_ids=None).",
    "expected_success": True,
  },
  {
    "name": "parent_category_bills",
    "last_user_request": "Set a $400 monthly budget for bills.",
    "previous_conversation": "",
    "ideal_response": "Expected: create_category_spending_limit(category='bills', granularity='monthly', start_date=<start of current month YYYY-MM-DD>, end_date='', amount=400.0, title='Monthly Bills' or similar).",
    "expected_success": True,
  },
  {
    "name": "income_goal_interest",
    "last_user_request": "I want to earn $500 in interest this year.",
    "previous_conversation": "",
    "ideal_response": "Expected: create_income_goal(category='income_interest', granularity='yearly', start_date=<start of current year YYYY-01-01>, end_date=<end of current year YYYY-12-31>, amount=500.0, title='Interest' or similar).",
    "expected_success": True,
  },
  {
    "name": "intent_ambiguous_clarification",
    "last_user_request": "I want to set aside some money.",
    "previous_conversation": "",
    "ideal_response": "Expected: Return (False, clarification) asking for amount and whether they want a spending limit, savings goal, or income goal.",
    "expected_success": False,
  },
]


def get_test_case(test_name_or_index):
  if isinstance(test_name_or_index, int):
    if 0 <= test_name_or_index < len(TEST_CASES):
      return TEST_CASES[test_name_or_index]
    return None
  if isinstance(test_name_or_index, str):
    for tc in TEST_CASES:
      if tc["name"] == test_name_or_index:
        return tc
    return None
  return None


def run_test(test_name_or_index_or_dict, optimizer: CreateBudgetOrGoalOptimizerV2 = None):
  """Run a single test. Returns dict with keys: passed (bool), name (str), reason (str or None)."""
  if isinstance(test_name_or_index_or_dict, dict):
    if "last_user_request" not in test_name_or_index_or_dict:
      print("Invalid test dict: must contain 'last_user_request'.")
      return {"passed": False, "name": "custom_test", "reason": "invalid dict"}
    tc = test_name_or_index_or_dict
    name = tc.get("name", "custom_test")
    print(f"\n{'='*80}\nRunning test: {name}\n{'='*80}\n")
    result, execution_success, execution_output, run_error = _run_test_with_logging(
      tc["last_user_request"],
      tc.get("previous_conversation", ""),
      optimizer,
    )
    if tc.get("ideal_response"):
      print("\n" + "=" * 80 + "\nIDEAL RESPONSE:\n" + "=" * 80 + "\n" + tc["ideal_response"] + "\n" + "=" * 80 + "\n")
    expected = tc.get("expected_success")
    passed, reason = _check_pass_fail(name, expected, execution_success, run_error)
    if not passed:
      print(f"\n*** FAIL: {name} — {reason}\n")
    return {"passed": passed, "name": name, "reason": reason}
  tc = get_test_case(test_name_or_index_or_dict)
  if tc is None:
    print(f"Test case '{test_name_or_index_or_dict}' not found.")
    return {"passed": False, "name": str(test_name_or_index_or_dict), "reason": "test not found"}
  print(f"\n{'='*80}\nRunning test: {tc['name']}\n{'='*80}\n")
  result, execution_success, execution_output, run_error = _run_test_with_logging(
    tc["last_user_request"], tc.get("previous_conversation", ""), optimizer
  )
  if tc.get("ideal_response"):
    print("\n" + "=" * 80 + "\nIDEAL RESPONSE:\n" + "=" * 80 + "\n" + tc["ideal_response"] + "\n" + "=" * 80 + "\n")
  expected = tc.get("expected_success")
  passed, reason = _check_pass_fail(tc["name"], expected, execution_success, run_error)
  if not passed:
    print(f"\n*** FAIL: {tc['name']} — {reason}\n")
  return {"passed": passed, "name": tc["name"], "reason": reason}


def _check_pass_fail(name: str, expected_success: bool, execution_success, run_error: str):
  """Return (passed: bool, reason: str)."""
  if expected_success is None:
    return True, None
  if run_error:
    return False, f"Sandbox error: {run_error}"
  if execution_success is None:
    return False, "No code extracted or no execution (expected_success cannot be checked)"
  if execution_success != expected_success:
    return False, f"expected success={expected_success}, got success={execution_success}"
  return True, None


def main(test: str = None, no_thinking: bool = False):
  """Run single test (--test <index|name>) or all tests (--test all). --no-thinking sets thinking_budget=0.
  Exit code 1 if any test fails (when expected_success != execution success)."""
  optimizer = CreateBudgetOrGoalOptimizerV2(thinking_budget=0 if no_thinking else 4096)

  if test is not None:
    if test.strip().lower() == "all":
      print(f"\n{'='*80}\nRunning ALL test cases\n{'='*80}\n")
      results = []
      for i in range(len(TEST_CASES)):
        outcome = run_test(i, optimizer)
        results.append(outcome)
        if i < len(TEST_CASES) - 1:
          print("\n" + "-" * 80 + "\n")
      passed_count = sum(1 for r in results if r["passed"])
      failed_count = len(results) - passed_count
      print("\n" + "=" * 80)
      print("SUMMARY")
      print("=" * 80)
      print(f"Passed: {passed_count}  Failed: {failed_count}  Total: {len(results)}")
      if failed_count > 0:
        print("\nFailed tests:")
        for r in results:
          if not r["passed"]:
            print(f"  - {r['name']}: {r.get('reason', 'unknown')}")
      print("=" * 80 + "\n")
      return 0 if failed_count == 0 else 1
    test_val = int(test) if test.isdigit() else test
    outcome = run_test(test_val, optimizer)
    if outcome.get("reason") == "test not found":
      print("\nAvailable test cases:")
      for i, tc in enumerate(TEST_CASES):
        print(f"  {i}: {tc['name']}")
      print("  all: run all test cases")
      return 0
    return 0 if outcome["passed"] else 1

  print("Usage:")
  print("  Run a single test: --test <name_or_index>")
  print("  Run all tests: --test all")
  print("  Disable thinking: --no-thinking (thinking_budget=0)")
  print("\nAvailable test cases:")
  for i, tc in enumerate(TEST_CASES):
    print(f"  {i}: {tc['name']}")
  print("  all: run all test cases")
  return 0


if __name__ == "__main__":
  import argparse
  import sys
  parser = argparse.ArgumentParser(description="Create budget or goal optimizer v2 (create_category_spending_limit, create_income_goal, create_savings_goal)")
  parser.add_argument("--test", type=str, help='Test name or index (e.g. "0" or "food_budget_next_month" or "all")')
  parser.add_argument("--no-thinking", action="store_true", help="Set thinking_budget=0 (Thinking OFF) for comparison")
  args = parser.parse_args()
  sys.exit(main(test=args.test, no_thinking=args.no_thinking))

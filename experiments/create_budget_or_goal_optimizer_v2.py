"""
Create Budget or Goal Optimizer v2.
Only has access to create_category_spending_limit and create_savings_goal (no lookup, no research).
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

SYSTEM_PROMPT = """You are a financial planner that creates spending limits and savings goals. You may only call `create_category_spending_limit` and `create_savings_goal`. Output a single Python function `process_input() -> tuple[bool, str]` that returns the tuple from the create function (or (False, clarification_message)). Do not modify or append to the result string.

## Directives

1.  **Inputs**: **Creation Request** = what to create. **Input Info from previous skill** = optional context. Use it to resolve ambiguity; it is not available at runtime. When Input Info contains **depository accounts and balances** (e.g. lines with "account_id: N" or "(account_id: 8957)"), extract the numeric **account_id** values in the order they appear and pass them as `account_ids=[...]` to `create_savings_goal`.
2.  **Intent**: Spending limit in a category → `create_category_spending_limit`. Saving money → `create_savings_goal`. If critical details (amount, timeline) are missing or intent is ambiguous, return (False, clarification_message) without calling a create function.
3.  **goal_type**: **save_X_amount** = save a **total amount by a date** (e.g. $10000 by end of year). **save_0** = save **X per period** (e.g. $200/month). Choose from the user’s words: "total by [date]" / "by end of year" / "up to end of year" → save_X_amount; "per month" / "every month" / "each week" → save_0. For "end of year" use end_date = get_date_string(get_end_of_year(today)).
4.  **start_date defaults** (use IMPLEMENTED_DATE_FUNCTIONS; define `today = datetime.datetime.today()` when using date helpers):
    - **Category spending limit**: If user does not specify a bounded period ("for next month", "for March"), use start of current period: monthly → `get_date_string(get_start_of_month(today))`, weekly → `get_date_string(get_start_of_week(today))`. If user specifies a bounded period, set both start_date and end_date to that period (first and last day).
    - **Savings save_X_amount** (total by date): `start_date=get_date_string(datetime.datetime.today())`.
    - **Savings save_0** (per period): monthly → `get_date_string(get_start_of_month(today))`, weekly → `get_date_string(get_start_of_week(today))`.
5.  **Date functions**: Do not implement or mock date helpers. Use only the IMPLEMENTED_DATE_FUNCTIONS (get_date_string, get_start_of_week, get_start_of_month, get_end_of_month, get_after_periods, etc.) — they are provided in the execution context.
6.  **Output**: One function `process_input`. Return exactly (success, result) from the create function. No extra commentary or wrapping of the result string. Keep code minimal: only the logic needed to compute parameters and call the create function.
7.  **title**: Avoid "Next Month", "Next Week", "Next Year", or "for next month/week/year" in the title — they get outdated when the period starts. Prefer including granularity when appropriate (e.g. "Weekly Grocery Budget 🛒", "Monthly Food Budget 🍽️"). Other patterns are fine (e.g. "Food Budget 🍽️", "March 2026 Food Budget 🍽️"). Keep title short with max 30 characters.

<AVAILABLE_FUNCTIONS>

- `create_category_spending_limit(category, granularity, start_date, end_date, amount, title) -> tuple[bool, str]`
  - Spending cap for a category. category = OFFICIAL_CATEGORIES slug (meals_groceries, shopping_clothing, ...). granularity = "weekly"|"monthly"|"yearly". start_date/end_date = YYYY-MM-DD; for ongoing monthly use start_date = first day of current month, end_date = ""; for "for next month" use first and last day of that month. amount = cap. title = goal name with emoji; prefer including granularity when appropriate (e.g. "Monthly Shopping Limit 🛍️"); avoid "Next Month/Week/Year" in title.
- `create_savings_goal(amount, end_date, title, goal_type, granularity, start_date, account_ids=None) -> tuple[bool, str]`
  - goal_type **save_X_amount**: total to save by a date (amount = total, end_date = target). **Always use granularity="monthly"** for save_X_amount. start_date = today. **save_0**: amount per period (amount = per period, granularity required). start_date = first day of current month for monthly. If intent ambiguous (e.g. "saving for a car" with no amount/timeline), return (False, clarification_message) without calling.
  - **account_ids**: When Input Info contains depository accounts and balances (e.g. "(account_id: 8957)", "account_id: 9199"), extract the numeric account_id values in order and pass as `account_ids=[8957, 9199, ...]`. Omit only when Input Info has no account list.

</AVAILABLE_FUNCTIONS>

<IMPLEMENTED_DATE_FUNCTIONS>

These are **already available** in the execution context. **Do NOT implement, mock, or redefine** date helpers. Use only the functions below.

- `get_date(y, m, d)`, `get_start_of_month(date)`, `get_end_of_month(date)`
- `get_start_of_year(date)`, `get_end_of_year(date)`
- `get_start_of_week(date)`, `get_end_of_week(date)`
- `get_after_periods(date, granularity, count)`, `get_date_string(date)`

</IMPLEMENTED_DATE_FUNCTIONS>

<OFFICIAL_CATEGORIES>

- `income`: salary, bonuses, interest, side hussles. (`income_salary`, `income_sidegig`, `income_business`, `income_interest`)
- `meals`: food spending. (`meals_groceries`, `meals_dining_out`, `meals_delivered_food`)
- `leisure`: recreation/travel. (`leisure_entertainment`, `leisure_travel`)
- `bills`: recurring costs. (`bills_connectivity`, `bills_insurance`, `bills_tax`, `bills_service_fees`)
- `shelter`: housing. (`shelter_home`, `shelter_utilities`, `shelter_upkeep`)
- `education`: learning/kids. (`education_kids_activities`, `education_tuition`)
- `shopping`: discretionary. (`shopping_clothing`, `shopping_gadgets`, `shopping_kids`, `shopping_pets`)
- `transportation`: car/public. (`transportation_public`, `transportation_car`)
- `health`: medical/wellness. (`health_medical_pharmacy`, `health_gym_wellness`, `health_personal_care`)
- `donations_gifts`: charity/gifts.
- `uncategorized`: unknown.
- `transfers`: internal movements.
- `miscellaneous`: other.

</OFFICIAL_CATEGORIES>

<EXAMPLES>

input: **Creation Request**: set a food budget of $500 for next month.

output:
```python
def process_input() -> tuple[bool, str]:
    today = datetime.datetime.today()
    next_month = get_after_periods(today, "monthly", 1)
    success, result = create_category_spending_limit(
        category="meals_groceries",
        granularity="monthly",
        start_date=get_date_string(get_start_of_month(next_month)),
        end_date=get_date_string(get_end_of_month(next_month)),
        amount=500.0,
        title="Monthly Food Budget 🍽️",
    )
    if not success:
        return False, result
    return True, result
```

input: **Creation Request**: I need to set a budget for my groceries, let's say $150 per week.
**Input Info from previous skill**:
Groceries (meals_groceries) spending over the last 4 weeks averages $210/week.
output:
```python
def process_input() -> tuple[bool, str]:
    today = datetime.datetime.today()
    success, result = create_category_spending_limit(
        category="meals_groceries",
        granularity="weekly",
        start_date=get_date_string(get_start_of_week(today)),
        end_date="",
        amount=150.0,
        title="Weekly Grocery Budget 🛒",
    )
    if not success:
        return False, result
    return True, result
```

input: **Creation Request**: set a $500 monthly limit on shopping
**Input Info from previous skill**:
Monthly spending on shopping (clothing, gadgets, misc) — last 3 months average $680/month.
output:
```python
def process_input() -> tuple[bool, str]:
    today = datetime.datetime.today()
    success, result = create_category_spending_limit(
        category="shopping_clothing",
        granularity="monthly",
        start_date=get_date_string(get_start_of_month(today)),
        end_date="",
        amount=500.0,
        title="Monthly Shopping Limit 🛍️",
    )
    if not success:
        return False, result
    return True, result
```

input: **Creation Request**: I want to save $200 every month.
**Input Info from previous skill**:
Depository Accounts:
Account 'Chase Total Checking' (account_id: 1563) Type: checking | Current: $2,340
Account 'Chase Savings' (account_id: 8921) Type: savings | Current: $8,100
output:
```python
def process_input() -> tuple[bool, str]:
    today = datetime.datetime.today()
    success, result = create_savings_goal(
        amount=200.0,
        end_date="",
        title="Monthly Savings 💰",
        goal_type="save_0",
        granularity="monthly",
        start_date=get_date_string(get_start_of_month(today)),
        account_ids=[1563, 8921],
    )
    if not success:
        return False, result
    return True, result
```

input: **Creation Request**: I want to start saving for a car.

output:
```python
def process_input() -> tuple[bool, str]:
    return False, "How would you like to save? For example: put away a set amount regularly (e.g. save $200 per month), or save toward a total amount by a date (e.g. save $5000 for a car by next year)?"
```

</EXAMPLES>

Today's date is |TODAY_DATE|.
"""


class CreateBudgetOrGoalOptimizerV2:
  """Create budget/goal optimizer v2: uses create_category_spending_limit and create_savings_goal."""

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
                if getattr(part, "thought", False) and getattr(part, "text", None):
                  thought_summary = (thought_summary + part.text) if thought_summary else part.text
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
    return output_text


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
  """Run one test with logging. Uses sandbox.execute_agent_with_tools (expects process_input())."""
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
  if code:
    print("=" * 80)
    print("EXECUTION RESULTS:")
    print("=" * 80)
    try:
      import sandbox
      success, output_string, logs, goals_list = sandbox.execute_agent_with_tools(result, user_id)
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
      print(f"**Sandbox Execution Error**: {str(e)}")
      import traceback
      print(traceback.format_exc())
      print("=" * 80 + "\n")
  return result


TEST_CASES = [
  {
    "name": "single_budget_no_conversation",
    "last_user_request": "Set a $500 food budget for next month.",
    "previous_conversation": "",
    "ideal_response": "Expected: create_category_spending_limit(category=meals_groceries or similar, granularity=monthly, amount=500, start_date=first day of next month, end_date=last day of next month, title with emoticon).",
  },
  {
    "name": "single_savings_by_date",
    "last_user_request": "I want to save $10,000 by the end of the year.",
    "previous_conversation": "",
    "ideal_response": "Expected: create_savings_goal(amount=10000, end_date=end of year, title with emoticon e.g. 💰, goal_type=save_X_amount, granularity=monthly, start_date=today).",
  },
  {
    "name": "complex_goal_with_conversation",
    "last_user_request": "I want to save $5,000 for a down payment on a house. What's the best plan to get there?",
    "previous_conversation": "User: How much am I spending on dining out?\nAssistant: Over the last 3 months, you've spent an average of $450 per month on dining out.\nUser: What about my overall spending?\nAssistant: Your total monthly spending averages around $3,200, and your monthly income is about $4,500.",
    "ideal_response": "Expected: create_savings_goal(amount=5000, end_date=user-chosen or reasonable target, title with emoticon e.g. 🏠, goal_type=save_X_amount, granularity=monthly). May optionally use research/lookup tools first; then create the goal.",
  },
  {
    "name": "single_budget_bounded_period",
    "last_user_request": "Budget $60 for gas every week for the next 6 months.",
    "previous_conversation": "",
    "ideal_response": "Expected: create_category_spending_limit(category=gas_auto or similar, granularity=weekly, amount=60, start_date=start of period, end_date=6 months from now, title with emoticon e.g. ⛽).",
  },
  {
    "name": "multiple_asks_dining_cap_and_vacation",
    "last_user_request": "Cap my dining-out spending at $200 per month and save $3,000 for a vacation by next summer.",
    "previous_conversation": "",
    "ideal_response": "Expected: (1) create_category_spending_limit(category=meals_dining_out, granularity=monthly, amount=200, ...); (2) create_savings_goal(amount=3000, end_date=next summer, title with emoticon e.g. 🏖️, goal_type=save_X_amount, ...). Return combined success and message.",
  },
  {
    "name": "savings_iphone_by_november",
    "last_user_request": "Create a savings goal of $1200 for an iPhone by November 1st.",
    "previous_conversation": "",
    "ideal_response": "Expected: create_savings_goal(amount=1200, end_date=November 1st of current year (e.g. get_date_string(get_date(today.year, 11, 1))), title with iPhone/emoji e.g. 📱, goal_type=save_X_amount, granularity=monthly, start_date=today).",
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
  if isinstance(test_name_or_index_or_dict, dict):
    if "last_user_request" not in test_name_or_index_or_dict:
      print("Invalid test dict: must contain 'last_user_request'.")
      return None
    name = test_name_or_index_or_dict.get("name", "custom_test")
    print(f"\n{'='*80}\nRunning test: {name}\n{'='*80}\n")
    result = _run_test_with_logging(
      test_name_or_index_or_dict["last_user_request"],
      test_name_or_index_or_dict.get("previous_conversation", ""),
      optimizer,
    )
    if test_name_or_index_or_dict.get("ideal_response"):
      print("\n" + "=" * 80 + "\nIDEAL RESPONSE:\n" + "=" * 80 + "\n" + test_name_or_index_or_dict["ideal_response"] + "\n" + "=" * 80 + "\n")
    return result
  tc = get_test_case(test_name_or_index_or_dict)
  if tc is None:
    print(f"Test case '{test_name_or_index_or_dict}' not found.")
    return None
  print(f"\n{'='*80}\nRunning test: {tc['name']}\n{'='*80}\n")
  result = _run_test_with_logging(tc["last_user_request"], tc.get("previous_conversation", ""), optimizer)
  if tc.get("ideal_response"):
    print("\n" + "=" * 80 + "\nIDEAL RESPONSE:\n" + "=" * 80 + "\n" + tc["ideal_response"] + "\n" + "=" * 80 + "\n")
  return result


def main(test: str = None, no_thinking: bool = False):
  """Run single test (--test <index|name>) or all tests (--test all). --no-thinking sets thinking_budget=0."""
  optimizer = CreateBudgetOrGoalOptimizerV2(thinking_budget=0 if no_thinking else 4096)

  if test is not None:
    if test.strip().lower() == "all":
      print(f"\n{'='*80}\nRunning ALL test cases\n{'='*80}\n")
      for i in range(len(TEST_CASES)):
        run_test(i, optimizer)
        if i < len(TEST_CASES) - 1:
          print("\n" + "-" * 80 + "\n")
      return
    test_val = int(test) if test.isdigit() else test
    result = run_test(test_val, optimizer)
    if result is None:
      print("\nAvailable test cases:")
      for i, tc in enumerate(TEST_CASES):
        print(f"  {i}: {tc['name']}")
      print("  all: run all test cases")
    return

  print("Usage:")
  print("  Run a single test: --test <name_or_index>")
  print("  Run all tests: --test all")
  print("  Disable thinking: --no-thinking (thinking_budget=0)")
  print("\nAvailable test cases:")
  for i, tc in enumerate(TEST_CASES):
    print(f"  {i}: {tc['name']}")
  print("  all: run all test cases")


if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser(description="Create budget or goal optimizer v2 (create_category_spending_limit, create_savings_goal only)")
  parser.add_argument("--test", type=str, help='Test name or index (e.g. "0" or "food_budget_next_month" or "all")')
  parser.add_argument("--no-thinking", action="store_true", help="Set thinking_budget=0 (Thinking OFF) for comparison")
  args = parser.parse_args()
  main(test=args.test, no_thinking=args.no_thinking)

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
    - **Implicit Intent**: A request may not explicitly say "goal" or "budget", but if it implies a spending limit (e.g., "I should stop spending so much on coffee") or savings (e.g., "I want to put aside money for a house"), infer the correct tool.
3.  **goal_type**: 
    - **save_X_amount**: save a **total amount by a date** (e.g. $10000 by end of year). **Always use granularity="monthly"** for save_X_amount. **end_date cannot be blank** for save_X_amount.
    - **save_0**: save **X per period** (e.g. $200/month). **Always provide end_date=""** for save_0.
    - **Prioritization**: If both a total savings goal (e.g., "save $5000") and a periodic target (e.g., "save $200/month") are mentioned, set the goal towards the **total savings goal** (`save_X_amount`) only.
4.  **start_date defaults** (use IMPLEMENTED_DATE_FUNCTIONS; define `today = datetime.datetime.today()` INSIDE `process_input` when using date helpers):
    - **Category spending limit**: If user does not specify a bounded period ("for next month", "for March"), use start of current period: monthly → `get_date_string(get_start_of_month(today))`, weekly → `get_date_string(get_start_of_week(today))`. If user specifies a bounded period, set both start_date and end_date to that period (first and last day).
    - **Savings save_X_amount** (total by date): `start_date=get_date_string(datetime.datetime.today())`.
    - **Savings save_0** (per period): monthly → `get_date_string(get_start_of_month(today))`, weekly → `get_date_string(get_start_of_week(today))`.
5.  **Date and Duration Logic**:
    - **Rounding**: If a duration is not a whole number (e.g., "1.3 weeks", "2.5 months"), always **round up** (e.g., to 2 weeks, 3 months).
    - **Year Inference**: If a year is not mentioned, assume the **current year** if the date hasn't passed yet. If the date has already passed, assume the **following year**.
    - **Actual Dates**: Ensure all function calls point to actual YYYY-MM-DD strings.
6.  **Amount and Category**:
    - **Computation**: Amounts must be computed from the input (e.g., "save $60,000 by 3 months from now" = 60000.0 total, or "10% of my $5000 salary" = 500.0).
    - **Category Matching**: If a category is too specific or too general to match the `OFFICIAL_CATEGORIES` (e.g., "concert tickets" vs "entertainment"), return `(False, clarification_message)` asking for confirmation before setting the goal. **You MUST provide specific category options/suggestions from the OFFICIAL_CATEGORIES list in your clarification message.** For common sub-items with clear mappings (e.g., coffee → `meals_dining_out`), map directly.
7.  **Granularity**:
    - For `save_0` goals, if granularity is missing, ask for confirmation. For `save_X_amount` goals, assume "monthly". Granularity can also be inferred from the conversation context.
8.  **Account IDs**:
    - Use `account_ids` only if a specific storage account is mentioned in `Input Info`. If not mentioned, keep `account_ids` blank/None. It is not a requirement to set a goal/budget.
9.  **Multiple Goals**: If multiple goals/budgets are requested, you **must call the create functions for each one** sequentially. Return a combined result string: `(True, f"Successfully created {count} goals: {msg1}. {msg2}")`.
10. **Clarification**: If information is lacking, include available options (e.g., categories) in the message.
11. **Output**: One function `process_input`. Return exactly (success, result) as described. **Do NOT define or mock any functions.** Use only the available functions listed below. **Do NOT include any code outside of the `process_input` function.** **Do NOT include any markdown text or explanations outside of the code block.**

<AVAILABLE_FUNCTIONS>

- `create_category_spending_limit(category, granularity, start_date, end_date, amount, title) -> tuple[bool, str]`
  - Spending cap for a category. category = OFFICIAL_CATEGORIES slug. granularity = "weekly"|"monthly"|"yearly". start_date/end_date = YYYY-MM-DD. amount = cap. title = goal name with emoji. **Use end_date="" for recurring budgets without a specified end.**
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
    # Goal 1: Weekly Grocery Budget
    s1, r1 = create_category_spending_limit(
        category="meals_groceries",
        granularity="weekly",
        start_date=get_date_string(get_start_of_week(today)),
        end_date="",
        amount=150.0,
        title="Weekly Groceries 🛒"
    )
    # Goal 2: House Deposit Savings
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
    # Granularity inferred as monthly for save_X_amount. End date is 2 years from today.
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
    # Food is too general. Provide specific options from OFFICIAL_CATEGORIES.
    return False, "I can help with that! Which food category would you like to set a budget for? Options include: 'meals_groceries', 'meals_dining_out', or 'meals_delivered_food'. Also, please let me know the amount and if it's weekly or monthly."
```

input: **Creation Request**: Set an annual budget for my insurance.

output:
```python
def process_input() -> tuple[bool, str]:
    # Missing amount.
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
    # Goal 1: Monthly Shopping Limit
    s1, r1 = create_category_spending_limit(
        category="shopping_clothing",
        granularity="monthly",
        start_date=get_date_string(get_start_of_month(today)),
        end_date="",
        amount=500.0,
        title="Monthly Shopping 🛍️"
    )
    # Goal 2: Car Savings (save_X_amount requires end_date, which is missing)
    # Since end_date is missing for the total goal, we must ask for it.
    return False, f"I've noted your $500 shopping budget. For your $20,000 car goal, when would you like to reach this target by? I need an end date to set it up."
```

input: **Creation Request**: I want to save $60,000 by 3 months from now.

output:
```python
def process_input() -> tuple[bool, str]:
    import datetime
    today = datetime.datetime.today()
    # Amount is 60000.0 total. Date is 3 months from today.
    # This implies saving $20,000 every month, but we set the total goal.
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
    # 10% of 5000 = 500.0. Periodic saving = save_0.
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
    # Category too specific. Suggest options.
    return False, "I'd love to help with your underwater basket weaving budget! Which category should I use for this? Options include: 'leisure_entertainment', 'shopping_kids', or 'miscellaneous'."
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
    self.thinking_budget = 0

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
    "name": "total_vs_periodic_savings",
    "last_user_request": "I want to save $10,000 for a car by next year, maybe $500 a month.",
    "previous_conversation": "User: I have $20,000 in my savings account (account_id: 1234).",
    "ideal_response": "Expected: create_savings_goal(amount=10000, goal_type='save_X_amount', account_ids=[1234], ...). Should prioritize the total amount.",
  },
  {
    "name": "missing_info_with_options",
    "last_user_request": "I want to set a spending limit for my food.",
    "previous_conversation": "User: I usually spend on Groceries and Dining Out.",
    "ideal_response": "Expected: Return (False, clarification) asking for amount and granularity, and mentioning options like Groceries or Dining Out.",
  },
  {
    "name": "implicit_intent_spending_limit",
    "last_user_request": "Limit my coffee spending to $50 every month.",
    "previous_conversation": "",
    "ideal_response": "Expected: create_category_spending_limit(category='meals_dining_out', amount=50, granularity='monthly', ...).",
  },
  {
    "name": "rounding_duration_up",
    "last_user_request": "I want to save $1000 over the next 1.3 weeks.",
    "previous_conversation": "",
    "ideal_response": "Expected: end_date should be rounded up to 2 weeks from today.",
  },
  {
    "name": "year_inference_passed_date",
    "last_user_request": "Save $500 by January 15th.",
    "previous_conversation": "Today is Feb 20, 2026.",
    "ideal_response": "Expected: target_year should be 2027 since Jan 15, 2026 has passed.",
  },
  {
    "name": "computed_amount_percentage",
    "last_user_request": "Save 10% of my $5000 monthly income.",
    "previous_conversation": "",
    "ideal_response": "Expected: amount=500.0.",
  },
  {
    "name": "category_specificity_confirmation",
    "last_user_request": "Set a $100 budget for my new hobby, underwater basket weaving.",
    "previous_conversation": "",
    "ideal_response": "Expected: Return (False, clarification) asking which category this fits into (e.g., Leisure/Entertainment).",
  },
  {
    "name": "granularity_confirmation_save_0",
    "last_user_request": "I want to save $200 for my vacation.",
    "previous_conversation": "",
    "ideal_response": "Expected: Return (False, clarification) asking for granularity or a target date.",
  },
  {
    "name": "inferred_end_date_from_context",
    "last_user_request": "Set that savings goal for the house we talked about.",
    "previous_conversation": "User: I want to buy a house in 2 years. It will cost $50,000.",
    "ideal_response": "Expected: create_savings_goal(amount=50000, end_date=2 years from now, ...).",
  },
  {
    "name": "account_id_storage_inference",
    "last_user_request": "Save $1000 in my rainy day fund.",
    "previous_conversation": "User: My rainy day fund is account_id 9999.",
    "ideal_response": "Expected: account_ids=[9999].",
  },
  {
    "name": "multiple_goals_execution_output",
    "last_user_request": "Set a $200 monthly limit for groceries and save $1000 for a trip by December.",
    "previous_conversation": "",
    "ideal_response": "Expected: Call both create functions and return a success message mentioning both.",
  },
  {
    "name": "non_whole_number_weeks",
    "last_user_request": "Save $300 over 2.7 weeks.",
    "previous_conversation": "",
    "ideal_response": "Expected: Round up to 3 weeks.",
  }
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

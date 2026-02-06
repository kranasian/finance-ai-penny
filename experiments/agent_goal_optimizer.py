"""
Agent Goal Optimizer: only create_category_spending_limit and create_savings_goal.
No lookup, research, or categorization. Uses Last User Request + Previous Conversation.
"""
import datetime
from google import genai
from google.genai import types
from google.genai.errors import ClientError
import os
import sys
from dotenv import load_dotenv

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
  sys.path.insert(0, parent_dir)

from penny.tool_funcs.create_category_spending_limit import create_category_spending_limit
from penny.tool_funcs.create_savings_goal import create_savings_goal
from penny.tool_funcs.date_utils import (
    get_date_string,
    get_start_of_month,
    get_end_of_month,
    get_start_of_week,
    get_end_of_year,
    get_after_periods,
)

load_dotenv()

SYSTEM_PROMPT = """You are a financial planner agent that creates spending limits and savings goals. You may only call `create_category_spending_limit` and `create_savings_goal`. No lookup, research, or categorization—they are not available.

## Your Tasks

1. **Prioritize the Last User Request**: Fulfill the **Last User Request** directly.
2. **Use Previous Conversation for Context**: Resolve ambiguity from **Previous Conversation** (amounts, categories, dates). If the conversation already gives the needed info, use it; do not ask for data you don't have.
3. **Output Python Code**: Write a single function `execute_plan() -> tuple[bool, str]` that:
   - Calls exactly one create function with correct parameters, then returns `(success, result)` from it, or
   - Returns `(False, clarification_message)` when amount/timeline is missing or intent is ambiguous (e.g. "saving for a car" with no amount or plan).
   - **CRITICAL**: The final output MUST be a valid Python code block in ```python ... ```. Return the tuple unchanged; do not wrap or append to the result string.

## Critical Rules

**1. Execution result—concise and complete:**
- The second element of the return tuple must be either the create function's exact return string or a **brief** clarification (one short sentence; if asking how to save, give two options only). No filler, no preamble.

**2. goal_type (UserGoals):**
- **save_X_amount** = save a **total by a date** (e.g. $10000 by end of year). Use end_date = get_date_string(get_end_of_year(today)) for "end of year".
- **save_0** = save **X per period** (e.g. $200/month). "per month" / "every month" → save_0.

**3. start_date:** Define `today = datetime.datetime.today()`. Category ongoing: monthly → get_date_string(get_start_of_month(today)), weekly → get_date_string(get_start_of_week(today)), end_date = "". Bounded period ("for next month"): use get_after_periods(today, "monthly", 1) then get_start_of_month/get_end_of_month for that month. Savings save_X_amount: start_date = get_date_string(today). Savings save_0 monthly: start_date = get_date_string(get_start_of_month(today)).

**4. Code minimal:** Only the logic to compute parameters and one create call (or one return (False, message)). No comments unless necessary.

**5. Match the examples:** One `execute_plan`, one create call or one `return (False, message)`. Use the same parameter patterns (today, date helpers, goal_type) as in the examples for the scenario type.

<AVAILABLE_FUNCTIONS>

Only these two functions exist. Do not call any other skill.

- `create_category_spending_limit(category, granularity, start_date, end_date, amount, title) -> tuple[bool, str]`
  - category = OFFICIAL_CATEGORIES slug (meals_groceries, shopping_clothing, ...). granularity = "weekly"|"monthly"|"yearly". start_date/end_date = YYYY-MM-DD; ongoing → end_date = "".
- `create_savings_goal(amount, end_date, title, goal_type, granularity, start_date) -> tuple[bool, str]`
  - save_X_amount: total by date (amount=total, end_date=target). save_0: per period (amount=per period, granularity required). If ambiguous, return (False, brief_message) without calling.

</AVAILABLE_FUNCTIONS>

<IMPLEMENTED_DATE_FUNCTIONS>

- get_date(y, m, d), get_start_of_month(date), get_end_of_month(date)
- get_start_of_year(date), get_end_of_year(date)
- get_start_of_week(date), get_end_of_week(date)
- get_after_periods(date, granularity, count), get_date_string(date)

</IMPLEMENTED_DATE_FUNCTIONS>

<EXAMPLES>

Each example is one execute_plan: either one create call returning (success, result) or one return (False, brief_message).

input: **Last User Request**: set a food budget of $500 for next month.
**Previous Conversation**:

output:
```python
def execute_plan() -> tuple[bool, str]:
    today = datetime.datetime.today()
    next_month = get_after_periods(today, "monthly", 1)
    success, result = create_category_spending_limit(
        category="meals_groceries",
        granularity="monthly",
        start_date=get_date_string(get_start_of_month(next_month)),
        end_date=get_date_string(get_end_of_month(next_month)),
        amount=500.0,
        title="Food budget for next month",
    )
    if not success:
        return False, result
    return True, result
```

input: **Last User Request**: I need to set a budget for my groceries, let's say $150 per week.
**Previous Conversation**:
User: My food spending is out of control.
Assistant: Looking at your recent transactions, you are spending about $210 per week on groceries.
output:
```python
def execute_plan() -> tuple[bool, str]:
    today = datetime.datetime.today()
    success, result = create_category_spending_limit(
        category="meals_groceries",
        granularity="weekly",
        start_date=get_date_string(get_start_of_week(today)),
        end_date="",
        amount=150.0,
        title="Weekly grocery budget",
    )
    if not success:
        return False, result
    return True, result
```

input: **Last User Request**: set a $500 monthly limit on shopping
**Previous Conversation**:
User: How much am I spending on shopping each month?
Assistant: Over the last 3 months, you've spent an average of $680 per month on shopping.
output:
```python
def execute_plan() -> tuple[bool, str]:
    today = datetime.datetime.today()
    success, result = create_category_spending_limit(
        category="shopping_clothing",
        granularity="monthly",
        start_date=get_date_string(get_start_of_month(today)),
        end_date="",
        amount=500.0,
        title="Monthly shopping limit",
    )
    if not success:
        return False, result
    return True, result
```

input: **Last User Request**: save $10000 up to end of year.
**Previous Conversation**:

output:
```python
def execute_plan() -> tuple[bool, str]:
    today = datetime.datetime.today()
    success, result = create_savings_goal(
        amount=10000.0,
        end_date=get_date_string(get_end_of_year(today)),
        title="Save $10000 by end of year",
        goal_type="save_X_amount",
        granularity="monthly",
        start_date=get_date_string(today),
    )
    if not success:
        return False, result
    return True, result
```

input: **Last User Request**: I want to save $200 every month.
**Previous Conversation**:

output:
```python
def execute_plan() -> tuple[bool, str]:
    today = datetime.datetime.today()
    success, result = create_savings_goal(
        amount=200.0,
        end_date="2099-12-31",
        title="Monthly savings",
        goal_type="save_0",
        granularity="monthly",
        start_date=get_date_string(get_start_of_month(today)),
    )
    if not success:
        return False, result
    return True, result
```

input: **Last User Request**: I want to start saving for a car.
**Previous Conversation**:

output:
```python
def execute_plan() -> tuple[bool, str]:
    return False, "How would you like to save? (1) A set amount per period (e.g. $200/month), or (2) a total by a date (e.g. $5000 by next year)."
```

</EXAMPLES>
"""


class AgentGoalOptimizer:
  """Goal optimizer with only create_category_spending_limit and create_savings_goal."""

  # def __init__(self, model_name="gemini-3-pro-preview", thinking_budget=4096):
  def __init__(self, model_name="gemini-flash-lite-latest", thinking_budget=0):
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
      raise ValueError("GEMINI_API_KEY environment variable is not set.")
    self.client = genai.Client(api_key=api_key)
    self.model_name = model_name
    self.thinking_budget = thinking_budget
    self.temperature = 0.6
    self.top_p = 0.95
    self.top_k = 40
    self.max_output_tokens = 8192
    self.safety_settings = [
      types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"),
      types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"),
      types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"),
      types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="OFF"),
    ]
    self.system_prompt = SYSTEM_PROMPT

  def generate_response(self, last_user_request: str, previous_conversation: str) -> str:
    request_text = types.Part.from_text(text=f"""**Last User Request**: {last_user_request}

**Previous Conversation**:

{previous_conversation}

output:""")
    contents = [types.Content(role="user", parts=[request_text])]
    generate_content_config = types.GenerateContentConfig(
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

  def get_available_models(self):
    try:
      models = genai.list_models()
      return [m.name for m in models if "generateContent" in m.supported_generation_methods]
    except Exception as e:
      raise Exception(f"Failed to get models: {str(e)}")


def extract_python_code(text: str) -> str:
  code_start = text.find("```python")
  if code_start != -1:
    code_start += len("```python")
    code_end = text.find("```", code_start)
    if code_end != -1:
      return text[code_start:code_end].strip()
    return text[code_start:].strip()
  return text.strip()


def _run_test_with_logging(last_user_request: str, previous_conversation: str, optimizer: "AgentGoalOptimizer" = None):
  if optimizer is None:
    optimizer = AgentGoalOptimizer()
  llm_input = f"""**Last User Request**: {last_user_request}

**Previous Conversation**:

{previous_conversation}

output:"""
  print("=" * 80)
  print("LLM INPUT:")
  print("=" * 80)
  print(llm_input)
  print("=" * 80 + "\n")
  result = optimizer.generate_response(last_user_request, previous_conversation)
  print("=" * 80)
  print("LLM OUTPUT:")
  print("=" * 80)
  print(result)
  print("=" * 80 + "\n")
  code = extract_python_code(result)
  if code:
    print("=" * 80)
    print("EXECUTING GENERATED CODE:")
    print("=" * 80)
    try:
      def wrapped_create_category_spending_limit(category, granularity, start_date, end_date, amount, title):
        print(f"\n[FUNCTION CALL] create_category_spending_limit")
        print(f"  category: {category!r}, granularity: {granularity!r}, start_date: {start_date!r}, end_date: {end_date!r}, amount: {amount}, title: {title!r}")
        out = create_category_spending_limit(category=category, granularity=granularity, start_date=start_date, end_date=end_date, amount=amount, title=title)
        print(f"  [RETURN] success: {out[0]}")
        print(f"  [RETURN] output: {out[1]}")
        return out

      def wrapped_create_savings_goal(amount, end_date, title, goal_type="save_X_amount", granularity=None, start_date=""):
        print(f"\n[FUNCTION CALL] create_savings_goal")
        print(f"  amount: {amount}, end_date: {end_date!r}, title: {title!r}, goal_type: {goal_type!r}, granularity: {granularity!r}, start_date: {start_date!r}")
        out = create_savings_goal(amount=amount, end_date=end_date, title=title, goal_type=goal_type, granularity=granularity, start_date=start_date)
        print(f"  [RETURN] success: {out[0]}")
        print(f"  [RETURN] output: {out[1]}")
        return out

      namespace = {
        "datetime": datetime,
        "get_date_string": get_date_string,
        "get_start_of_month": get_start_of_month,
        "get_end_of_month": get_end_of_month,
        "get_start_of_week": get_start_of_week,
        "get_end_of_year": get_end_of_year,
        "get_after_periods": get_after_periods,
        "create_category_spending_limit": wrapped_create_category_spending_limit,
        "create_savings_goal": wrapped_create_savings_goal,
      }
      exec(code, namespace)
      if "execute_plan" in namespace:
        print("\n" + "=" * 80)
        print("Calling execute_plan()...")
        print("=" * 80)
        execution_result = namespace["execute_plan"]()
        print("\n" + "=" * 80)
        print("EXECUTION RESULT:")
        print("=" * 80)
        print(f"  success: {execution_result[0]}")
        print(f"  output: {execution_result[1]}")
        print("=" * 80 + "\n")
      else:
        print("Warning: execute_plan() function not found in generated code")
        print("=" * 80)
    except Exception as e:
      print(f"Error executing generated code: {str(e)}")
      import traceback
      print(traceback.format_exc())
      print("=" * 80)
  return result


TEST_CASES = [
  {
    "name": "category_budget_food_monthly",
    "last_user_request": "set a food budget of $500 for next month.",
    "previous_conversation": "",
    "ideal_response": """Expected: create_category_spending_limit(category=meals_groceries or similar, granularity=monthly, amount=500, start_date=first day of next month, end_date=last day of next month, ...).""",
  },
  {
    "name": "category_budget_shopping",
    "last_user_request": "set a $500 monthly limit on shopping",
    "previous_conversation": "User: How much am I spending on shopping each month?\nAssistant: Over the last 3 months, you've spent an average of $680 per month on shopping.",
    "ideal_response": """Expected: create_category_spending_limit(category=shopping_clothing, granularity=monthly, amount=500, ...).""",
  },
  {
    "name": "savings_total_by_date_save_X_amount",
    "last_user_request": "save $10000 up to end of year.",
    "previous_conversation": "",
    "ideal_response": """Expected: create_savings_goal(amount=10000, end_date=end of year, goal_type=save_X_amount, start_date=today, ...).""",
  },
  {
    "name": "savings_per_period_save_0",
    "last_user_request": "I want to save $200 every month.",
    "previous_conversation": "",
    "ideal_response": """Expected: create_savings_goal(amount=200, goal_type=save_0, granularity=monthly, start_date=first day of current month, ...).""",
  },
  {
    "name": "savings_ambiguous_missing_type",
    "last_user_request": "I want to start saving for a car.",
    "previous_conversation": "",
    "ideal_response": """Expected: return (False, message) asking how they want to save. Do not call create_savings_goal until intent is clear.""",
  },
  {
    "name": "savings_complex_save_X_amount",
    "last_user_request": "I want to save $5,000 for a down payment on a house. What's the best plan to get there?",
    "previous_conversation": "User: What about my overall spending?\nAssistant: Your total monthly spending averages around $3,200, and your monthly income is about $4,500.",
    "ideal_response": """Expected: create_savings_goal(amount=5000, goal_type=save_X_amount, ...). Use conversation context for dates/title.""",
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


def run_test(test_name_or_index_or_dict, optimizer: AgentGoalOptimizer = None):
  if isinstance(test_name_or_index_or_dict, dict):
    if "last_user_request" not in test_name_or_index_or_dict:
      print("Invalid test dict: must contain 'last_user_request'.")
      return None
    name = test_name_or_index_or_dict.get("name", "custom_test")
    print(f"\n{'='*80}\nRunning test: {name}\n{'-'*80}\n")
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
  print(f"\n{'='*80}\nRunning test: {tc['name']}\n{'-'*80}\n")
  result = _run_test_with_logging(tc["last_user_request"], tc.get("previous_conversation", ""), optimizer)
  if tc.get("ideal_response"):
    print("\n" + "=" * 80 + "\nIDEAL RESPONSE:\n" + "=" * 80 + "\n" + tc["ideal_response"] + "\n" + "=" * 80 + "\n")
  return result


def run_tests(test_names_or_indices_or_dicts, optimizer: AgentGoalOptimizer = None):
  if test_names_or_indices_or_dicts is None:
    test_names_or_indices_or_dicts = list(range(len(TEST_CASES)))
  return [run_test(x, optimizer) for x in test_names_or_indices_or_dicts]


def test_with_inputs(last_user_request: str, previous_conversation: str, optimizer: AgentGoalOptimizer = None):
  return _run_test_with_logging(last_user_request, previous_conversation, optimizer)


def main(batch: int = None, test: str = None, no_thinking: bool = False):
  """Run by batch (1 or 2) or single test. no_thinking=True sets thinking_budget=0 for comparison."""
  optimizer = AgentGoalOptimizer(thinking_budget=0 if no_thinking else 4096)
  BATCHES = {
    1: {"name": "Agent Goal Optimizer - Batch 1", "tests": [0, 1, 2]},
    2: {"name": "Agent Goal Optimizer - Batch 2", "tests": [3, 4, 5]},
  }
  if batch is not None:
    if batch not in BATCHES:
      print(f"Invalid batch: {batch}. Available: {list(BATCHES.keys())}")
      for b, info in BATCHES.items():
        print(f"  Batch {b}: {', '.join(TEST_CASES[i]['name'] for i in info['tests'])}")
      return
    print(f"\n{'='*80}\nBATCH {batch}: {BATCHES[batch]['name']}\n{'='*80}\n")
    for idx in BATCHES[batch]["tests"]:
      run_test(idx, optimizer)
      print("\n" + "-" * 80 + "\n")
  elif test is not None:
    test_int = int(test) if test.isdigit() else test
    r = run_test(test_int, optimizer)
    if r is None:
      print("\nAvailable test cases:")
      for i, tc in enumerate(TEST_CASES):
        print(f"  {i}: {tc['name']}")
  else:
    print("Usage: --batch <1|2> or --test <name|index>")
    print("\nBatches:")
    for b, info in BATCHES.items():
      print(f"  {b}: {info['name']}")
      for idx in info["tests"]:
        print(f"    - {idx}: {TEST_CASES[idx]['name']}")
    print("\nAll test cases:")
    for i, tc in enumerate(TEST_CASES):
      print(f"  {i}: {tc['name']}")


if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser(description="Run agent goal optimizer (create_category_spending_limit, create_savings_goal only)")
  parser.add_argument("--batch", type=int, choices=[1, 2], help="Batch number")
  parser.add_argument("--test", type=str, help="Test name or index")
  parser.add_argument("--no-thinking", action="store_true", help="Set thinking_budget=0 (Thinking OFF) for comparison")
  args = parser.parse_args()
  main(batch=args.batch, test=args.test, no_thinking=args.no_thinking)

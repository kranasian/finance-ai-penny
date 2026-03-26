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

# Import tool functions
from penny.tool_funcs.lookup_user_accounts_transactions_income_and_spending_patterns import lookup_user_accounts_transactions_income_and_spending_patterns
from penny.tool_funcs.update_transaction_category_or_create_category_rules import update_transaction_category_or_create_category_rules
from penny.tool_funcs.create_budget_or_goal_or_reminder import create_budget_or_goal_or_reminder
from penny.tool_funcs.research_and_strategize_financial_outcomes import research_and_strategize_financial_outcomes
from strategizer.prompts import STRATEGIZER_SYSTEM_PROMPT

# Load environment variables
load_dotenv()


class StrategizerOptimizer:
  """Handles all Gemini API interactions for the proactive Strategizer agent"""
  
  def __init__(self, model_name="gemini-flash-lite-latest", thinking_budget=4096, max_output_tokens=4096):
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
      raise ValueError("GEMINI_API_KEY environment variable is not set. Please set it in your .env file or environment.")
    self.client = genai.Client(api_key=api_key)
    self.model_name = model_name
    self.thinking_budget = thinking_budget
    self.temperature = 0.6
    self.top_p = 0.95
    self.max_output_tokens = max_output_tokens
    
    # Safety Settings
    self.safety_settings = [
      types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"),
      types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"),
      types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"),
      types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="OFF")
    ]
    
    self.system_prompt = STRATEGIZER_SYSTEM_PROMPT

  
  def generate_response(self, task_description: str, previous_outcomes: str) -> str:
    request_text = types.Part.from_text(text=f"""**Task Description**: {task_description}

**Previous Outcomes**:

{previous_outcomes}

output:""")
    
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

    if thought_summary:
      print("\n" + "-" * 80)
      print("THOUGHT SUMMARY:")
      print("-" * 80)
      print(thought_summary.strip())
      print("-" * 80 + "\n")

    return output_text


def extract_python_code(text: str) -> str:
    code_start = text.find("```python")
    if code_start != -1:
        code_start += len("```python")
        code_end = text.find("```", code_start)
        if code_end != -1:
            return text[code_start:code_end].strip()
        else:
            return text[code_start:].strip()
    else:
        return text.strip()


def _format_mock_lookup_output(income_txns):
  """Format a list of income transaction dicts into the lookup result string."""
  lines = []
  total = 0
  for t in income_txns:
    lines.append(f"- ${t['amount']} was received from {t['merchant']} on {t['date']} ({t['account']}) categorized as {t['category']}.")
    total += t["amount"]
  body = "\n".join(lines) + f"\nTotal recent income: earned ${total}."
  return f"""--- Recent Income (Last 30 Days) ---
Recent Income Transactions:
{body}
"""


def _run_test_with_logging(task_description: str, previous_outcomes: str, optimizer: StrategizerOptimizer = None, mock_income: list = None):
  if optimizer is None:
    optimizer = StrategizerOptimizer()
  mock_output = _format_mock_lookup_output(mock_income) if mock_income else None

  llm_input = f"""**Task Description**: {task_description}

**Previous Outcomes**:

{previous_outcomes}

output:"""
  
  print("=" * 80)
  print("LLM INPUT:")
  print("=" * 80)
  print(llm_input)
  print("=" * 80)
  print()
  
  result = optimizer.generate_response(task_description, previous_outcomes)
  
  print("=" * 80)
  print("LLM OUTPUT:")
  print("=" * 80)
  print(result)
  print("=" * 80)
  print()
  
  code = extract_python_code(result)
  
  if code:
    print("=" * 80)
    print("EXECUTING GENERATED CODE:")
    print("=" * 80)
    try:
      def wrapped_lookup(*args, **kwargs):
        print(f"\n[FUNCTION CALL] lookup_user_accounts_transactions_income_and_spending_patterns")
        print(f"  args: {args}")
        if mock_output is not None:
          res = (True, mock_output)
          print(f"  [RETURN] (mock) success: {res[0]}")
        else:
          res = lookup_user_accounts_transactions_income_and_spending_patterns(*args, **kwargs)
          print(f"  [RETURN] success: {res[0]}")
        out = res[1]
        print(f"  [RETURN] output: {out[:300]}{'...' if len(out) > 300 else ''}")
        return res
      
      def wrapped_update(*args, **kwargs):
        print(f"\n[FUNCTION CALL] update_transaction_category_or_create_category_rules")
        print(f"  args: {args}")
        result = update_transaction_category_or_create_category_rules(*args, **kwargs)
        print(f"  [RETURN] success: {result[0]}")
        print(f"  [RETURN] output: {result[1]}")
        return result
        
      def wrapped_research(*args, **kwargs):
        print(f"\n[FUNCTION CALL] research_and_strategize_financial_outcomes")
        print(f"  args: {args}")
        result = research_and_strategize_financial_outcomes(*args, **kwargs)
        print(f"  [RETURN] success: {result[0]}")
        print(f"  [RETURN] output: {result[1]}")
        return result
      
      def wrapped_create(*args, **kwargs):
        print(f"\n[FUNCTION CALL] create_budget_or_goal_or_reminder")
        print(f"  args: {args}")
        print(f"  kwargs: {kwargs}")
        result = create_budget_or_goal_or_reminder(*args, **kwargs)
        print(f"  [RETURN] success: {result[0]}")
        print(f"  [RETURN] output: {result[1]}")
        return (result[0], result[1])
      
      namespace = {
        'lookup_user_accounts_transactions_income_and_spending_patterns': wrapped_lookup,
        'update_transaction_category_or_create_category_rules': wrapped_update,
        'research_and_strategize_financial_outcomes': wrapped_research,
        'create_budget_or_goal_or_reminder': wrapped_create,
      }
      
      exec(code, namespace)
      
      if 'execute_plan' in namespace:
        print("\n" + "=" * 80)
        print("Calling execute_plan()...")
        print("=" * 80)
        result = namespace['execute_plan']()
        print("\n" + "=" * 80)
        print("execute_plan() FINAL RESULT:")
        print("=" * 80)
        print(f"  success: {result[0]}")
        print(f"  output: {result[1]}")
        print("=" * 80)
      else:
        print("Warning: execute_plan() function not found in generated code")
        print("=" * 80)
    except Exception as e:
      print(f"Error executing generated code: {str(e)}")
      import traceback
      print(traceback.format_exc())
      print("=" * 80)
  
  return result


# Mock income data per test case: list of dicts with merchant, amount, date, account, category.
MOCK_INCOME_ITERATION_1 = [
  {"merchant": "CA State Payroll", "amount": 1440, "date": "2025-11-18", "account": "Chase Total Checking **1563", "category": "income_salary"},
  {"merchant": "CA State Payroll", "amount": 1340, "date": "2025-10-31", "account": "Chase Total Checking **1563", "category": "income_salary"},
]
MOCK_INCOME_ITERATION_2 = [
  {"merchant": "ADP PAYROLL", "amount": 2500, "date": "2025-11-20", "account": "Chase Total Checking **1563", "category": "uncategorized"},
  {"merchant": "Gusto", "amount": 2500, "date": "2025-10-22", "account": "Chase Total Checking **1563", "category": "uncategorized"},
]
MOCK_INCOME_ITERATION_3 = [
  {"merchant": "ADP PAYROLL", "amount": 2500, "date": "2025-11-20", "account": "Chase Total Checking **1563", "category": "uncategorized"},
  {"merchant": "Gusto", "amount": 2500, "date": "2025-10-22", "account": "Chase Total Checking **1563", "category": "uncategorized"},
  {"merchant": "Savings Interest", "amount": 3, "date": "2025-11-01", "account": "Chase Savings **3052", "category": "income_interest"},
]

TEST_CASES = [
  {
    "name": "salary_check_iteration_1",
    "task_description": "Check if there's salary detected for this user and whether it looks as expected. If we can't find it, look at the list of amounts coming in and check if there are salary transactions mixed up over there. Fix the categorization of these if they are indeed Salary.",
    "previous_outcomes": "None. This is the first attempt.",
    "ideal_response": "Expected: lookup finding total income and whether there are salary categories already mapped.",
    "mock_income": MOCK_INCOME_ITERATION_1,
  },
  {
    "name": "salary_check_iteration_2",
    "task_description": "Check if there's salary detected for this user and whether it looks as expected. If we can't find it, look at the list of amounts coming in and check if there are salary transactions mixed up over there. Fix the categorization of these if they are indeed Salary.",
    "previous_outcomes": "Outcome 1: Tried to lookup the salary. Found total income is 5000, but no specific 'Salary' transactions found. Only uncategorized income.",
    "ideal_response": "Expected: lookup to get a detailed list of uncategorized income transactions to identify patterns like 'Gusto' or 'ADP'.",
    "mock_income": MOCK_INCOME_ITERATION_2,
  },
  {
    "name": "salary_check_iteration_3",
    "task_description": "Check if there's salary detected for this user and whether it looks as expected. If we can't find it, look at the list of amounts coming in and check if there are salary transactions mixed up over there. Fix the categorization of these if they are indeed Salary.",
    "previous_outcomes": "Outcome 1: Tried to lookup the salary. Found no specific 'Salary' transactions found. Only uncategorized income.\nOutcome 2: Looked up detailed list of uncategorized income. Found 2 recurring transactions of $2500 marked as 'ADP PAYROLL'.",
    "ideal_response": "Expected: update_transaction_category_or_create_category_rules to set 'ADP PAYROLL' as Salary and create a rule.",
    "mock_income": MOCK_INCOME_ITERATION_3,
  }
]


def run_test(test_name_or_index_or_dict, optimizer: StrategizerOptimizer = None):
  if isinstance(test_name_or_index_or_dict, dict):
    if "task_description" in test_name_or_index_or_dict:
      test_name = test_name_or_index_or_dict.get("name", "custom_test")
      print(f"\n{'='*80}\nRunning test: {test_name}\n{'='*80}\n")
      return _run_test_with_logging(
        test_name_or_index_or_dict["task_description"],
        test_name_or_index_or_dict.get("previous_outcomes", ""),
        optimizer,
        mock_income=test_name_or_index_or_dict.get("mock_income"),
      )
  if isinstance(test_name_or_index_or_dict, int):
    tc = TEST_CASES[test_name_or_index_or_dict] if 0 <= test_name_or_index_or_dict < len(TEST_CASES) else None
  else:
    tc = next((t for t in TEST_CASES if t["name"] == test_name_or_index_or_dict), None)
  if not tc:
    return None
  print(f"\n{'='*80}\nRunning test: {tc['name']}\n{'='*80}\n")
  return _run_test_with_logging(tc["task_description"], tc["previous_outcomes"], optimizer, mock_income=tc.get("mock_income"))


def main(test: str = None, no_thinking: bool = False, thinking_budget: int = None, max_output_tokens: int = None, model: str = None):
  tb = 0 if no_thinking else (thinking_budget if thinking_budget is not None else 4096)
  kw = {"thinking_budget": tb}
  if max_output_tokens is not None:
    kw["max_output_tokens"] = max_output_tokens
  if model is not None:
    kw["model_name"] = model
  optimizer = StrategizerOptimizer(**kw)

  if test is not None:
    if test.strip().lower() == "all":
      for i in range(len(TEST_CASES)): run_test(i, optimizer)
      return
    test_val = int(test) if test.isdigit() else test
    run_test(test_val, optimizer)
    return

  print("Available test cases:")
  for i, tc in enumerate(TEST_CASES):
    print(f"  {i}: {tc['name']}")

if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument("--test", type=str)
  parser.add_argument("--no-thinking", action="store_true")
  parser.add_argument("--thinking-budget", type=int, default=None)
  parser.add_argument("--max-output-tokens", type=int, default=None)
  parser.add_argument("--model", type=str, default=None)
  args = parser.parse_args()
  main(test=args.test, no_thinking=args.no_thinking, thinking_budget=args.thinking_budget, max_output_tokens=args.max_output_tokens, model=args.model)

from google import genai
from google.genai import types
from google.genai.errors import ClientError
import os
import sys
import time
from dotenv import load_dotenv

# Add the parent directory to the path so we can import the tools
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
  sys.path.insert(0, parent_dir)

# Import tool functions
from penny.tool_funcs.lookup_user_accounts_transactions_income_and_spending_patterns import lookup_user_accounts_transactions_income_and_spending_patterns
from penny.tool_funcs.update_transaction_category_or_create_category_rules import update_transaction_category_or_create_category_rules
from penny.tool_funcs.create_budget_or_goal_or_reminder import create_budget_or_goal_or_reminder as create_budget_or_goal
from penny.tool_funcs.research_and_strategize_financial_outcomes import research_and_strategize_financial_outcomes

# Load environment variables
load_dotenv()

STRATEGIZER_SYSTEM_PROMPT = """You are Strategizer, a proactive financial AI agent.
Your job is to make measurable progress on the user's task by selecting the smartest next action based on prior attempts.

You have access to `tool_funcs` and must write Python code that calls them.

## Inputs You Receive
1. **Task Description**
2. **Previous Outcomes** (what was already attempted and what happened)

## Strategy Rules (High Priority)
1. **Do not repeat failed patterns blindly.** If a similar action already failed, modify approach, wording, scope, or sequencing.
2. **Generate options before acting.** Briefly consider 2-3 plausible next actions, then choose one that maximizes expected progress and minimizes repeating past failure.
3. **Be creative but grounded.** Use specific hypotheses from prior outcomes (merchant names, amounts, categories, time windows) to craft a better request.
4. **Prefer decisive moves.** If evidence is strong, execute corrective action; if evidence is weak/contradictory, gather targeted data.
5. **One primary next step per turn.** Keep execution focused and auditable.
6. **Failure-aware retry policy.** If the last attempt failed, your retry must change at least one dimension: request specificity, scope, sequencing, or fallback path.
7. **Lean verification.** Verify only when needed to reduce uncertainty; avoid bulky multi-step flows unless verification changes decision quality.
8. **Token-efficient execution code.** Prefer the shortest valid `execute_plan` that achieves the chosen next action.
9. **Compact creative retries.** For retry scenarios, prefer concise but different tactics (e.g., split by merchant, tighten amount/category filters, or narrower scope) over repeating the same broad request.

## Tool Functions Available
- `lookup_user_accounts_transactions_income_and_spending_patterns(lookup_request: str, input_info: str = None) -> tuple[bool, str]`
- `update_transaction_category_or_create_category_rules(categorize_request: str, input_info: str = None) -> tuple[bool, str]`
- `create_budget_or_goal(creation_request: str, input_info: str = None) -> tuple[bool, str]`
- `research_and_strategize_financial_outcomes(strategize_request: str, input_info: str = None) -> tuple[bool, str]`

## Output Requirements
- Output Python code in a single ```python``` block defining:

```python
def execute_plan() -> tuple[bool, str]:
    ...
    return success, output
```

**CRITICAL RULE**: Do not prefix tool functions with module names. Call them directly by function name.
**CODE CONCISENESS RULES**:
- Keep `execute_plan` compact (target <= 10 lines inside function body).
- Avoid explanatory comments inside code.
- Avoid unnecessary branching/wrapping if a direct return from tool output is sufficient.
- Prefer a single tool call per turn unless a second call is essential.
"""

class StrategizerOptimizer:
  """Handles all Gemini API interactions for the proactive Strategizer agent"""
  
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
    
    # Safety Settings
    self.safety_settings = [
      types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"),
      types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"),
      types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"),
      types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="OFF")
    ]
    
    self.system_prompt = STRATEGIZER_SYSTEM_PROMPT

  
  def generate_response(
    self,
    task_description: str,
    previous_outcomes: dict[int | str, str] | list[str] | str | None,
    latest_result_summary: str | None = None,
    latest_outcome_reflection: str | None = None,
  ) -> str:
    previous_outcomes_block = _format_previous_outcomes(previous_outcomes)
    body = f"""# Task Description

{task_description}

## Previous Outcomes

{previous_outcomes_block}
"""
    if latest_result_summary is not None or latest_outcome_reflection is not None:
      lrs = latest_result_summary if latest_result_summary is not None else ""
      lor = latest_outcome_reflection if latest_outcome_reflection is not None else ""
      body += f"""
## Most Recent Attempt Result

{lrs}

## Outcome & What to do Next

{lor}
"""
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
    t0 = time.perf_counter()
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

    latency_s = time.perf_counter() - t0
    um = getattr(last_chunk, "usage_metadata", None) if last_chunk else None
    pt = ot = tt = th = 0
    if um is not None:
      pt = um.prompt_token_count or 0
      ot = um.candidates_token_count or 0
      th = um.thoughts_token_count or 0
      tt = um.total_token_count or 0
    print(
      f"[STRATEGIZER_METRICS] latency_s={latency_s:.3f} prompt_token_count={pt} "
      f"candidates_token_count={ot} thoughts_token_count={th} total_token_count={tt}",
      flush=True,
    )

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


def _format_mock_spending_output(
  spending_txns,
  *,
  header: str = "Recent Spending Patterns (Last 30 Days)",
  transactions_heading: str = "Recent Spending Transactions:",
):
  """Format a list of spending transaction dicts into the lookup result string.

  Args:
    spending_txns: Rows with merchant, amount, date, account, category.
    header: Inner title for the top banner; rendered as ``--- {header} ---``.
    transactions_heading: Subheading above the transaction lines.
  """
  lines = []
  total = 0
  for t in spending_txns:
    lines.append(f"- ${t['amount']} was paid to {t['merchant']} on {t['date']} ({t['account']}) categorized as {t['category']}.")
    total += t["amount"]
  body = "\n".join(lines) + f"\nTotal recent spending: spent ${total}."
  banner = f"--- {header} ---"
  return f"""{banner}
{transactions_heading}
{body}
"""


def _format_previous_outcomes(previous_outcomes: dict[int | str, str] | list[str] | str | None) -> str:
  if previous_outcomes is None:
    return "None. This is the first attempt."
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
    return "None. This is the first attempt."
  clean_outcomes = [item.strip() for item in previous_outcomes if isinstance(item, str) and item.strip()]
  if not clean_outcomes:
    return "None. This is the first attempt."
  return "\n".join(
    f"{idx}. **Outcome #{idx}**: {outcome}" for idx, outcome in enumerate(clean_outcomes, start=1)
  )


def _run_test_with_logging(
  task_description: str,
  previous_outcomes: dict[int | str, str] | list[str] | str | None,
  optimizer: StrategizerOptimizer | None = None,
  mock_execution_result: str | None = None,
  ideal_response: str | None = None,
  latest_result_summary: str | None = None,
  latest_outcome_reflection: str | None = None,
):
  if optimizer is None:
    optimizer = StrategizerOptimizer()
  mock_output = mock_execution_result
  previous_outcomes_block = _format_previous_outcomes(previous_outcomes)

  llm_input_display = f"""# Task Description

{task_description}

## Previous Outcomes

{previous_outcomes_block}
"""
  if latest_result_summary is not None or latest_outcome_reflection is not None:
    lrs = latest_result_summary if latest_result_summary is not None else ""
    lor = latest_outcome_reflection if latest_outcome_reflection is not None else ""
    llm_input_display += f"""
## Most Recent Attempt Result

{lrs}

## Outcome & What to do Next

{lor}"""
  
  print(f"## LLM Input\n")
  print(llm_input_display)
  print()
  
  result = optimizer.generate_response(
    task_description,
    previous_outcomes,
    latest_result_summary=latest_result_summary,
    latest_outcome_reflection=latest_outcome_reflection,
  )
  
  print(f"## LLM Output:\n")
  print(result)
  print()
  
  code = extract_python_code(result)
  
  if code:
    try:
      def wrapped_lookup(*args, **kwargs):
        # Suppressed intermediate function call logging to match Markdown style
        # print(f"\n[FUNCTION CALL] lookup_user_accounts_transactions_income_and_spending_patterns")
        # print(f"  args: {args}")
        if mock_output is not None:
          res = (True, mock_output)
          # print(f"  [RETURN] (mock) success: {res[0]}")
        else:
          res = lookup_user_accounts_transactions_income_and_spending_patterns(*args, **kwargs)
          # print(f"  [RETURN] success: {res[0]}")
        out = res[1]
        # print(f"  [RETURN] output: {out[:300]}{'...' if len(out) > 300 else ''}")
        return res
      
      def wrapped_update(*args, **kwargs):
        # print(f"\n[FUNCTION CALL] update_transaction_category_or_create_category_rules")
        # print(f"  args: {args}")
        result = update_transaction_category_or_create_category_rules(*args, **kwargs)
        # print(f"  [RETURN] success: {result[0]}")
        # print(f"  [RETURN] output: {result[1]}")
        return result
        
      def wrapped_research(*args, **kwargs):
        # print(f"\n[FUNCTION CALL] research_and_strategize_financial_outcomes")
        # print(f"  args: {args}")
        result = research_and_strategize_financial_outcomes(*args, **kwargs)
        # print(f"  [RETURN] success: {result[0]}")
        # print(f"  [RETURN] output: {result[1]}")
        return result
      
      def wrapped_create(*args, **kwargs):
        # print(f"\n[FUNCTION CALL] create_budget_or_goal_or_reminder")
        # print(f"  args: {args}")
        # print(f"  kwargs: {kwargs}")
        result = create_budget_or_goal(*args, **kwargs)
        # print(f"  [RETURN] success: {result[0]}")
        # print(f"  [RETURN] output: {result[1]}")
        return (result[0], result[1])
      
      namespace = {
        'lookup_user_accounts_transactions_income_and_spending_patterns': wrapped_lookup,
        'update_transaction_category_or_create_category_rules': wrapped_update,
        'research_and_strategize_financial_outcomes': wrapped_research,
        'create_budget_or_goal_or_reminder': wrapped_create,
      }
      
      exec(code, namespace)
      
      if 'execute_plan' in namespace:
        result = namespace['execute_plan']()
        print("\n## Execution Final Result:\n")
        print("```")
        print(f"  success: {result[0]}")
        print(f"  output: {result[1]}")
        print("```")
      else:
        print("Warning: execute_plan() function not found in generated code")
    except Exception as e:
      print(f"Error executing generated code: {str(e)}")
      import traceback
      print(traceback.format_exc())
  
  if ideal_response:
    print(f"\n## Ideal Response:\n\n{ideal_response}\n")
  
  return result


# Mock lookup outputs for tests that set mock_execution_result (formatted like API output strings).
MOCK_INCOME_ITERATION_1A = _format_mock_lookup_output([
  {"merchant": "CA State Payroll", "amount": 1440, "date": "2025-11-18", "account": "Chase Total Checking **1563", "category": "income_salary"},
  {"merchant": "CA State Payroll", "amount": 1340, "date": "2025-10-31", "account": "Chase Total Checking **1563", "category": "income_salary"},
])
MOCK_INCOME_ITERATION_1B = _format_mock_lookup_output([
  {"merchant": "ADP PAYROLL", "amount": 2500, "date": "2025-11-20", "account": "Chase Total Checking **1563", "category": "uncategorized"},
  {"merchant": "Gusto", "amount": 2500, "date": "2025-10-22", "account": "Chase Total Checking **1563", "category": "uncategorized"},
])
MOCK_SHELTER_ITERATION_1A = _format_mock_spending_output([
  {"merchant": "Apartments LLC", "amount": 2000, "date": "2025-11-18", "account": "Chase Total Checking **1563", "category": "shelter_home"},
  {"merchant": "Apartments LLC", "amount": 2000, "date": "2025-10-18", "account": "Chase Total Checking **1563", "category": "shelter_home"},
  {"merchant": "Apartments LLC", "amount": 2000, "date": "2025-09-18", "account": "Chase Total Checking **1563", "category": "shelter_home"},
  {"merchant": "Bank of America", "amount": 1650, "date": "2025-10-31", "account": "Chase Total Checking **1563", "category": "shelter_home"},
  {"merchant": "Bank of America", "amount": 1650, "date": "2025-09-30", "account": "Chase Total Checking **1563", "category": "shelter_home"},
  {"merchant": "Bank of America", "amount": 1650, "date": "2025-08-31", "account": "Chase Total Checking **1563", "category": "shelter_home"},
  {"merchant": "Community College", "amount": 1200, "date": "2025-11-15", "account": "Chase Total Checking **1563", "category": "education_tuition"},
  {"merchant": "Community College", "amount": 1200, "date": "2024-11-15", "account": "Chase Total Checking **1563", "category": "education_tuition"},
], header="Top Spending", transactions_heading="Recent Largest Spending Transactions:")

TEST_CASES = [
  {
    "name": "salary_check_iteration_1a",
    "batch": 1,
    "task_description": "Check if there's salary detected for this user and whether it looks as expected. If we can't find it, look at the list of amounts coming in and check if there are salary transactions mixed up over there. Fix the categorization of these if they are indeed Salary.",
    "previous_outcomes": None,
    "latest_result_summary": None,
    "latest_outcome_reflection": None,
    "ideal_response": "Expected: lookup finding total income and whether there are salary categories already mapped.",
    "mock_execution_result": MOCK_INCOME_ITERATION_1A,
  },
  {
    "name": "salary_check_iteration_1b",
    "batch": 1,
    "task_description": "Check if there's salary detected for this user and whether it looks as expected. If we can't find it, look at the list of amounts coming in and check if there are salary transactions mixed up over there. Fix the categorization of these if they are indeed Salary.",
    "previous_outcomes": None,
    "latest_result_summary": None,
    "latest_outcome_reflection": None,
    "ideal_response": "Expected: lookup finding total income and whether there are only uncategorized income transactions.",
    "mock_execution_result": MOCK_INCOME_ITERATION_1B,
  },
  {
    "name": "salary_check_iteration_2b",
    "batch": 1,
    "task_description": "Check if there's salary detected for this user and whether it looks as expected. If we can't find it, look at the list of amounts coming in and check if there are salary transactions mixed up over there. Fix the categorization of these if they are indeed Salary.",
    "previous_outcomes": None,
    "latest_result_summary": "Two income transactions for $2500 each were found from ADP PAYROLL and Gusto, both categorized as 'uncategorized'.",
    "latest_outcome_reflection": "The execution successfully listed recent income transactions, confirming the presence of two deposits ($2500 from ADP PAYROLL and $2500 from Gusto) that appear to be salary but are currently uncategorized. The task requires fixing the categorization if they are salary. Since the categorization fix has not yet occurred, the status must remain IN_PROGRESS to pursue the required correction.",
    "ideal_response": "Expected: recategorize ADP PAYROLL and Gusto uncategorized income transactions as Salary (and create/update rule if appropriate).",
    "mock_execution_result": None,
  },
  {
    "name": "salary_check_iteration_3",
    "batch": 1,
    "task_description": "Check if there's salary detected for this user and whether it looks as expected. If we can't find it, look at the list of amounts coming in and check if there are salary transactions mixed up over there. Fix the categorization of these if they are indeed Salary.",
    "previous_outcomes": [
      "The execution successfully listed recent income transactions, confirming the presence of two deposits ($2500 from ADP PAYROLL and $2500 from Gusto) that appear to be salary but are currently uncategorized. The task requires fixing the categorization if they are salary. Since the categorization fix has not yet occurred, the status must remain IN_PROGRESS to pursue the required correction.",
    ],
    "latest_result_summary": "The execution returned an explicit failure message: 'An unexpected error was encountered.'",
    "latest_outcome_reflection": "The previous step successfully identified two potential salary transactions ($2500 from ADP PAYROLL and $2500 from Gusto) that needed categorization. However, the current execution step failed with a generic error message, preventing any further progress towards fixing the categorization or confirming the salary expectation. Therefore, the status is FAILED.",
    "ideal_response": "Expected: recategorize ADP PAYROLL or Gusto to isolate the error (or both if safe approach is not chosen).",
    "mock_execution_result": None,
  },
  # shelter (batch 2)
  {
    "name": "shelter_check_iteration_1a",
    "batch": 2,
    "task_description": "Look into the user's rent or mortgage spending and make sure it's categorized correctly as shelter_home. Check if amount is expected, and if it isn't transactions might be in uncategorized. Look at the largest spending to see if it is indeed rent or mortgage and fix the categorization.",
    "previous_outcomes": None,
    "latest_result_summary": None,
    "latest_outcome_reflection": None,
    "ideal_response": "Expected: lookup rent or mortgage spending and whether they're categorized correctly as shelter_home.",
    "mock_execution_result": MOCK_SHELTER_ITERATION_1A,
  },
  {
    "name": "shelter_check_iteration_2b",
    "batch": 2,
    "task_description": "Look into the user's rent or mortgage spending and make sure it's categorized correctly as shelter_home. Check if amount is expected, and if it isn't transactions might be in uncategorized. Look at the largest spending to see if it is indeed rent or mortgage and fix the categorization.",
    "previous_outcomes": None,
    "latest_result_summary": "The analysis identified several large transactions. A $2000 payment to 'Apartments LLC' on 2025-11-18 is miscategorized as 'meals_dining_out' when it should likely be 'shelter_home'. Additionally, a $1650 payment to 'Bank of America' on 2025-10-31 is 'uncategorized'. The task requires fixing these categorizations.",
    "latest_outcome_reflection": "The task requires checking amounts and fixing categorizations for rent/mortgage spending. The execution result successfully listed the top spending and revealed miscategorizations: a $2000 payment to Apartments LLC on 2025-11-18 is incorrectly labeled 'meals_dining_out', and a $1650 payment to Bank of America on 2025-10-31 is 'uncategorized'. Since these required fixes have not yet been applied, the workflow must continue to the remediation step. Rubric 0 does not apply as there are no two differing large housing payments on the same date.",
    "ideal_response": "Expected: recategorize at least the confirmed miscategorization for Apartments LLC as shelter_home, or attempt all.",
    "mock_execution_result": None,
  },
]


def run_test(test_name_or_index_or_dict, optimizer: StrategizerOptimizer = None):
  if isinstance(test_name_or_index_or_dict, dict):
    if "task_description" in test_name_or_index_or_dict:
      test_name = test_name_or_index_or_dict.get("name", "custom_test")
      print(f"\n# Test: **{test_name}**\n")
      return _run_test_with_logging(
        test_name_or_index_or_dict["task_description"],
        test_name_or_index_or_dict.get("previous_outcomes"),
        optimizer,
        mock_execution_result=test_name_or_index_or_dict.get("mock_execution_result"),
        ideal_response=test_name_or_index_or_dict.get("ideal_response"),
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
    tc.get("previous_outcomes"),
    optimizer,
    mock_execution_result=tc.get("mock_execution_result"),
    ideal_response=tc.get("ideal_response"),
    latest_result_summary=tc.get("latest_result_summary"),
    latest_outcome_reflection=tc.get("latest_outcome_reflection"),
  )


def run_all_tests_batch(optimizer: StrategizerOptimizer = None, batch_num: int = 1):
  if optimizer is None:
    optimizer = StrategizerOptimizer()
  cases = [tc for tc in TEST_CASES if tc["batch"] == batch_num]
  batch_results = []
  label = "salary" if batch_num == 1 else "shelter" if batch_num == 2 else f"batch {batch_num}"
  for tc in cases:
    result = run_test(tc, optimizer)
    batch_results.append((tc["name"], result))
  for name, result in batch_results:
    success = result[0] if isinstance(result, tuple) and len(result) > 0 else None
    print(f"- {name}: success={success}")
  return batch_results


def main(
  test: str = None,
  run_batch: bool = False,
  batch_num: int = 1,
  no_thinking: bool = False,
  thinking_budget: int = None,
  max_output_tokens: int = None,
  model: str = None,
):
  tb = 0 if no_thinking else (thinking_budget if thinking_budget is not None else 700)
  kw = {"thinking_budget": tb}
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
      run_all_tests_batch(optimizer, batch_num=2)
      return
    test_val = int(test) if test.isdigit() else test
    run_test(test_val, optimizer)
    return

  print("Available test cases:")
  for i, tc in enumerate(TEST_CASES):
    print(f"  {i}: {tc['name']} (batch {tc['batch']})")

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
    help="Run test cases in group N: 1=salary, 2=shelter (omit N to use 1).",
  )
  args = parser.parse_args()
  if args.batch is not None and args.batch not in (1, 2):
    parser.error("batch N must be 1 (salary) or 2 (shelter)")
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

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
from create_budget_or_goal_optimizer import create_budget_or_goal
from penny.tool_funcs.research_and_strategize_financial_outcomes import research_and_strategize_financial_outcomes

# Load environment variables
load_dotenv()

SYSTEM_PROMPT = """You are a financial planner agent that is very good at understanding conversation and creating a plan to achieve a user's financial goals.

## Core Directives

1.  **Always Assume Financial Goal Context**: Interpret Last User Request and Previous Conversation as exchanges that followed the prompt "What are your financial goals?".
2.  **Analyze User Intent**: Analyze the **Last User Request** in the context of the other previous messages in **Previous Conversation**. Determine if the user is stating an explicit/implicit financial goal, or asking a question. Return clarification if **Last User Request** is unintelligible.
3.  **Goal-Oriented Flow**: If the user states a financial goal, follow this flow:
    *   **Step 1: Gather Data (If Necessary).** Call lookup only when you need a baseline (e.g. current spending), feasibility check, or data the user did not provide. If the user already gave amount, scope (category or goal type), and period/timeline, call `create_budget_or_goal` directly with that and optional `input_info` from Previous Conversation. Use `research_and_strategize_financial_outcomes` only for data outside the user's finances (e.g. market estimates, travel costs).
    *   **Step 2: Strategize.** If the goal is complex (e.g. retirement, college savings, debt paydown), use `research_and_strategize_financial_outcomes` once. Simple budgets or savings goals (e.g. "$X per week for groceries", "save $Y monthly for emergency fund") do not require this step.
    *   **Step 3: Create Goal.** Final step: one precise `create_budget_or_goal` call per goal. `creation_request` must be one sentence: amount + scope + period (e.g. "Create a weekly grocery budget of $150."). `create_budget_or_goal` will ask for any missing information; do not preempt with extra lookup.
4.  **Information-Seeking Flow**: If the user asks a question, the plan should consist of the necessary `lookup` or `research` skills to acquire the information. The plan's final output should be the information itself.
5.  **Extract Key Information**: Vigilantly identify and extract critical details from the user's request, such as **amounts and timelines**.
6.  **Use Conversation History for Clarity**: Refer to the `Previous Conversation` to resolve ambiguity.
7.  **Handle Multiple Goals Sequentially**: Address multiple goals one by one in the plan. When the plan has **multiple** `create_budget_or_goal` calls: collect (success, create_result) for each; **no early return** on failure. Use variables like success1, create_result1, success2, create_result2, outputs. If all fail, return `(False, chr(10).join(outputs))`. If at least one succeeds, return `(True, f"{n} of {y} goals successfully created.")` with n = number of successes, y = total. Do not return the joined outputs when any call succeeded.
8.  **Output Python Code**: The plan must be a Python function `execute_plan`.
9.  **Request and result conciseness**: Keep every request parameter (`lookup_request`, `creation_request`, `strategize_request`) to one clear sentence (or two only when necessary). Do not add filler or paragraphs. Return exactly the tuple from the last step: `(success, output)` — no extra commentary, prefixes, or wrapping. The execution result must be concise but complete: the skill's output string or the "n of y goals" summary only.

<AVAILABLE_SKILL_FUNCTIONS>

These are the **available skills** that can be stacked and sequenced using `input_info` for efficient information flow between steps.
- All **skill functions** return `tuple[bool, str]`: (success, output). Use output as `input_info` for the next step when relevant.
- Keep request parameters to one sentence; incorporate **Previous Conversation** and `input_info` where needed to address **Last User Request**. Return exactly `(success, output)` from the plan — no extra text.

### List of all Skill Functions

- `lookup_user_accounts_transactions_income_and_spending_patterns(lookup_request: str, input_info: str = None) -> tuple[bool, str]`
  - `lookup_request`: One clear sentence for what to lookup (accounts, transactions, income/spending, subscriptions, forecasts). When `input_info` is available, incorporate it concisely. Use only when baseline or feasibility is needed; if the user already gave amount, scope, and period, skip to `create_budget_or_goal`.
- `create_budget_or_goal (creation_request: str, input_info: str = None) -> tuple[bool, str]`
  - `creation_request`: One sentence with amount, scope (budget category or savings goal), and period (e.g. weekly, monthly, or by date). When `input_info` is available, use it for context but keep the sentence concise. Only for budgets or savings goals; NOT for categorization.
  - Returns (success, output). Output is the created detail or, if more info needed from user, the question to ask. **Multiple goals**: One call per goal; collect (success, create_result) for each; no early return. If none succeeded return (False, chr(10).join(outputs)); else return (True, f"{n} of {y} goals successfully created.").
- `research_and_strategize_financial_outcomes(strategize_request: str, input_info: str = None) -> tuple[bool, str]`
  - `strategize_request`: One sentence for research or strategy using external data (e.g. "Average dining out for a couple in Chicago.", "Strategy to pay off $5000 credit card debt with timeline."). Do not use for the user's own data — use lookup for that.
</AVAILABLE_SKILL_FUNCTIONS>

<EXAMPLES>

input: **Last User Request**: I need to set a budget for my groceries, let's say $150 per week.
**Previous Conversation**:
User: My food spending is out of control.
Assistant: I can help with that. Looking at your recent transactions, you are spending about $210 per week on groceries.
output:
```python
def execute_plan() -> tuple[bool, str]:
    success, create_result = create_budget_or_goal(
        creation_request="Create a weekly grocery budget of $150.",
        input_info="User is currently spending $210/week on groceries."
    )
    return success, create_result
```

input: **Last User Request**: set a $500 monthly limit on shopping
**Previous Conversation**:
User: How much am I spending on shopping each month?
Assistant: Over the last 3 months, you've spent an average of $680 per month on shopping, including clothing, gadgets, and miscellaneous purchases.
output:
```python
def execute_plan() -> tuple[bool, str]:
    success, lookup_result = lookup_user_accounts_transactions_income_and_spending_patterns(
        lookup_request="Analyze current monthly shopping spending to determine a baseline for setting a budget limit."
    )
    if not success:
        return False, lookup_result

    success, create_result = create_budget_or_goal(
        creation_request="Set a monthly shopping budget limit of $500, incorporating current spending patterns.",
        input_info=lookup_result
    )
    return success, create_result
```

input: **Last User Request**: I want to pay off my credit card debt of $5000.
**Previous Conversation**:
None
output:
```python
def execute_plan() -> tuple[bool, str]:
    success, lookup_result = lookup_user_accounts_transactions_income_and_spending_patterns(
        lookup_request="Analyze current income, spending, and credit card debt to assess the feasibility of paying off $5000."
    )
    if not success:
        return False, lookup_result

    success, strategy_result = research_and_strategize_financial_outcomes(
        strategize_request="Develop a strategy to pay off the $5000 credit card debt, including a timeline and monthly payment plan.",
        input_info=lookup_result
    )
    if not success:
        return False, strategy_result
    
    success, create_result = create_budget_or_goal(
        creation_request="Create a goal to pay off the $5000 credit card debt based on the developed strategy.",
        input_info=strategy_result
    )
    return success, create_result
```

input: **Last User Request**: I want a $400 monthly food budget and to save $200 every month for my emergency fund.
**Previous Conversation**: None
output:
```python
def execute_plan() -> tuple[bool, str]:
    outputs = []
    success1, create_result1 = create_budget_or_goal(
        creation_request="Create a monthly food budget of $400.",
        input_info=None
    )
    outputs.append(create_result1)
    success2, create_result2 = create_budget_or_goal(
        creation_request="Create a savings goal of $200 every month for emergency fund.",
        input_info=None
    )
    outputs.append(create_result2)
    if not (success1 or success2):
        return (False, chr(10).join(outputs))
    n = (1 if success1 else 0) + (1 if success2 else 0)
    return (True, f"{n} of 2 goals successfully created.")
```

</EXAMPLES>
"""

class GoalAgentOptimizer:
  """Handles all Gemini API interactions for financial goal creation and optimization"""
  
  def __init__(self, model_name="gemini-flash-lite-latest", thinking_budget=4096):
    """Initialize the Gemini agent with API configuration for financial goal creation."""
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
      raise ValueError("GEMINI_API_KEY environment variable is not set. Please set it in your .env file or environment.")
    self.client = genai.Client(api_key=api_key)
    self.model_name = model_name
    self.thinking_budget = thinking_budget
    
    # Generation Configuration Constants
    self.temperature = 0.6
    self.top_p = 0.95
    self.max_output_tokens = 4096
    
    # Safety Settings
    self.safety_settings = [
      types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"),
      types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"),
      types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"),
      types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="OFF")
    ]
    
    # System Prompt
    self.system_prompt = SYSTEM_PROMPT

  
  def generate_response(self, last_user_request: str, previous_conversation: str) -> str:
    """
    Generate a response using Gemini API for financial goal creation.
    
    Args:
      last_user_request: The last user request as a string
      previous_conversation: The previous conversation as a string
      
    Returns:
      Generated code as a string
    """
    # Create request text with Last User Request and Previous Conversation
    request_text = types.Part.from_text(text=f"""**Last User Request**: {last_user_request}

**Previous Conversation**:

{previous_conversation}

output:""")
    
    # Create content and configuration
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
    """
    Get list of available Gemini models.
    
    Returns:
      List of available model names
    """
    try:
      models = genai.list_models()
      return [m.name for m in models if 'generateContent' in m.supported_generation_methods]
    except Exception as e:
      raise Exception(f"Failed to get models: {str(e)}")


def extract_python_code(text: str) -> str:
    """Extract Python code from generated response (look for ```python blocks).
    
    Args:
        text: The generated response containing Python code
        
    Returns:
        str: Extracted Python code
    """
    code_start = text.find("```python")
    if code_start != -1:
        code_start += len("```python")
        code_end = text.find("```", code_start)
        if code_end != -1:
            return text[code_start:code_end].strip()
        else:
            # No closing ``` found, use the entire response as code
            return text[code_start:].strip()
    else:
        # No ```python found, try to use the entire response as code
        return text.strip()


def _run_test_with_logging(last_user_request: str, previous_conversation: str, optimizer: GoalAgentOptimizer = None):
  """
  Internal helper function that runs a test with consistent logging.
  
  Args:
    last_user_request: The last user request as a string
    previous_conversation: The previous conversation as a string
    optimizer: Optional GoalAgentOptimizer instance. If None, creates a new one.
    
  Returns:
    The generated response string
  """
  if optimizer is None:
    optimizer = GoalAgentOptimizer()
  
  # Construct LLM input
  llm_input = f"""**Last User Request**: {last_user_request}

**Previous Conversation**:

{previous_conversation}

output:"""
  
  # Print the input
  print("=" * 80)
  print("LLM INPUT:")
  print("=" * 80)
  print(llm_input)
  print("=" * 80)
  print()
  
  result = optimizer.generate_response(last_user_request, previous_conversation)
  
  # Print the output
  print("=" * 80)
  print("LLM OUTPUT:")
  print("=" * 80)
  print(result)
  print("=" * 80)
  print()
  
  # Extract and execute the generated code
  code = extract_python_code(result)
  
  if code:
    print("=" * 80)
    print("EXECUTING GENERATED CODE:")
    print("=" * 80)
    try:
      # Create wrapper functions that print their returns and handle return types
      def wrapped_lookup(*args, **kwargs):
        print(f"\n[FUNCTION CALL] lookup_user_accounts_transactions_income_and_spending_patterns")
        print(f"  args: {args}")
        result = lookup_user_accounts_transactions_income_and_spending_patterns(*args, **kwargs)
        print(f"  [RETURN] success: {result[0]}")
        print(f"  [RETURN] output: {result[1]}")
        return result  # Returns tuple[bool, str]
      
      def wrapped_research(*args, **kwargs):
        print(f"\n[FUNCTION CALL] research_and_strategize_financial_outcomes")
        print(f"  args: {args}")
        result = research_and_strategize_financial_outcomes(*args, **kwargs)
        print(f"  [RETURN] success: {result[0]}")
        print(f"  [RETURN] output: {result[1]}")
        return result  # Returns tuple[bool, str]
      
      def wrapped_create(*args, **kwargs):
        print(f"\n[FUNCTION CALL] create_budget_or_goal")
        print(f"  args: {args}")
        print(f"  kwargs: {kwargs}")
        
        result = create_budget_or_goal(*args, **kwargs)
        # create_budget_or_goal returns tuple[bool, str, list] but we return tuple[bool, str] to match system prompt
        print(f"  [RETURN] success: {result[0]}")
        print(f"  [RETURN] output: {result[1]}")
        if len(result) >= 3:
          print(f"  [RETURN] goals list: {result[2]}")
        # Return only (success, output) to match system prompt specification
        return (result[0], result[1])  # Returns tuple[bool, str]
      
      # Create a namespace with the wrapped tool functions
      namespace = {
        'lookup_user_accounts_transactions_income_and_spending_patterns': wrapped_lookup,
        'research_and_strategize_financial_outcomes': wrapped_research,
        'create_budget_or_goal': wrapped_create,
      }
      
      # Execute the code
      exec(code, namespace)
      
      # Call execute_plan if it exists
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


# Test cases: distinct scopes, not identical to prompt examples.
TEST_CASES = [
  {
    "name": "single_budget_no_conversation",
    "last_user_request": "Set a $500 food budget for next month.",
    "previous_conversation": "",
    "ideal_response": "Expected: direct create_budget_or_goal(creation_request with $500 food budget for next month, input_info=None). No lookup. Return (success, create_result)."
  },
  {
    "name": "single_savings_by_date",
    "last_user_request": "I want to save $10,000 by the end of the year.",
    "previous_conversation": "",
    "ideal_response": "Expected: lookup (savings rate / cash flow for $10k by year-end) → research (required monthly amount to reach $10k by EOY) → create_budget_or_goal(creation_request for $10k by end of year, input_info=strategy_result). Return (success, create_result). Goal end_date = end of current year."
  },
  {
    "name": "complex_goal_with_conversation",
    "last_user_request": "I want to save $5,000 for a down payment on a house. What's the best plan to get there?",
    "previous_conversation": """User: How much am I spending on dining out?
Assistant: Over the last 3 months, you've spent an average of $450 per month on dining out.
User: What about my overall spending?
Assistant: Your total monthly spending averages around $3,200, and your monthly income is about $4,500.""",
    "ideal_response": "Expected: lookup (confirm income ~$4,500 and spending ~$3,200 / surplus) → research (savings strategy for $5k down payment) → create_budget_or_goal(creation_request for $5k house down payment with plan/timeline, input_info=strategy_result). Return (success, create_result). Goal should have end_date and clear timeline."
  },
  {
    "name": "single_budget_bounded_period",
    "last_user_request": "Budget $60 for gas every week for the next 6 months.",
    "previous_conversation": "",
    "ideal_response": "Expected: direct create_budget_or_goal(creation_request for weekly $60 gas budget for the next 6 months, input_info=None). No lookup. Return (success, create_result)."
  },
  {
    "name": "multiple_asks_dining_cap_and_vacation",
    "last_user_request": "Cap my dining-out spending at $200 per month and save $3,000 for a vacation by next summer.",
    "previous_conversation": "",
    "ideal_response": "Expected: two create_budget_or_goal calls (1: monthly dining cap $200; 2: savings $3,000 for vacation by next summer). Collect success1, create_result1, success2, create_result2, outputs; no early return. Return (True, '2 of 2 goals successfully created.') when both succeed. 'Next summer' = summer of current/next calendar year (e.g. July 2026)."
  },
]


def get_test_case(test_name_or_index):
  """
  Get a test case by name or index.
  
  Args:
    test_name_or_index: Test case name (str) or index (int)
    
  Returns:
    Test case dict or None if not found
  """
  if isinstance(test_name_or_index, int):
    if 0 <= test_name_or_index < len(TEST_CASES):
      return TEST_CASES[test_name_or_index]
    return None
  elif isinstance(test_name_or_index, str):
    for test_case in TEST_CASES:
      if test_case["name"] == test_name_or_index:
        return test_case
    return None
  return None


def run_test(test_name_or_index_or_dict, optimizer: GoalAgentOptimizer = None):
  """
  Run a single test by name, index, or by passing test data directly.
  
  Args:
    test_name_or_index_or_dict: One of:
      - Test case name (str): e.g., "save_10000_to_end_of_year"
      - Test case index (int): e.g., 0
      - Test data dict: {"last_user_request": "...", "previous_conversation": "..."}
    optimizer: Optional GoalAgentOptimizer instance. If None, creates a new one.
    
  Returns:
    The generated response string, or None if test not found
  """
  # Check if it's a dict with test data
  if isinstance(test_name_or_index_or_dict, dict):
    if "last_user_request" in test_name_or_index_or_dict:
      test_name = test_name_or_index_or_dict.get("name", "custom_test")
      print(f"\n{'='*80}")
      print(f"Running test: {test_name}")
      print(f"{'='*80}\n")
      result = _run_test_with_logging(
        test_name_or_index_or_dict["last_user_request"],
        test_name_or_index_or_dict.get("previous_conversation", ""),
        optimizer
      )
      if test_name_or_index_or_dict.get("ideal_response"):
        print("\n" + "=" * 80 + "\nIDEAL RESPONSE:\n" + "=" * 80 + "\n" + test_name_or_index_or_dict["ideal_response"] + "\n" + "=" * 80 + "\n")
      return result
    else:
      print(f"Invalid test dict: must contain 'last_user_request' key.")
      return None
  
  # Otherwise, treat it as a test name or index
  test_case = get_test_case(test_name_or_index_or_dict)
  if test_case is None:
    print(f"Test case '{test_name_or_index_or_dict}' not found.")
    return None
  
  print(f"\n{'='*80}")
  print(f"Running test: {test_case['name']}")
  print(f"{'='*80}\n")
  result = _run_test_with_logging(
    test_case["last_user_request"],
    test_case["previous_conversation"],
    optimizer
  )
  if test_case.get("ideal_response"):
    print("\n" + "=" * 80 + "\nIDEAL RESPONSE:\n" + "=" * 80 + "\n" + test_case["ideal_response"] + "\n" + "=" * 80 + "\n")
  return result


def run_tests(test_names_or_indices_or_dicts, optimizer: GoalAgentOptimizer = None):
  """
  Run multiple tests by names, indices, or by passing test data directly.
  
  Args:
    test_names_or_indices_or_dicts: One of:
      - None: Run all tests from TEST_CASES
      - List of test case names (str), indices (int), or test data dicts
        Each dict should have: {"last_user_request": "...", "previous_conversation": "..."}
    optimizer: Optional GoalAgentOptimizer instance. If None, creates a new one.
    
  Returns:
    List of generated response strings
  """
  if test_names_or_indices_or_dicts is None:
    # Run all tests
    test_names_or_indices_or_dicts = list(range(len(TEST_CASES)))
  
  results = []
  for test_item in test_names_or_indices_or_dicts:
    result = run_test(test_item, optimizer)
    results.append(result)
  
  return results


def test_with_inputs(last_user_request: str, previous_conversation: str, optimizer: GoalAgentOptimizer = None):
  """
  Convenient method to test the goal agent optimizer with custom inputs.
  
  Args:
    last_user_request: The last user request as a string
    previous_conversation: The previous conversation as a string
    optimizer: Optional GoalAgentOptimizer instance. If None, creates a new one.
    
  Returns:
    The generated response string
  """
  return _run_test_with_logging(last_user_request, previous_conversation, optimizer)


def main(test: str = None, no_thinking: bool = False):
  """Main: run single test (name or index), all tests, or show usage. no_thinking=True sets thinking_budget=0."""
  optimizer = GoalAgentOptimizer(thinking_budget=0 if no_thinking else 4096)

  if test is not None:
    if test.strip().lower() == "all":
      print(f"\n{'='*80}")
      print("Running ALL test cases")
      print(f"{'='*80}\n")
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
  parser = argparse.ArgumentParser(description="Run goal agent optimizer tests")
  parser.add_argument("--test", type=str, help='Test name or index (e.g. "set_food_budget_500_next_month" or "0")')
  parser.add_argument("--no-thinking", action="store_true", help="Set thinking_budget=0 (Thinking OFF) for comparison")
  args = parser.parse_args()
  main(test=args.test, no_thinking=args.no_thinking)


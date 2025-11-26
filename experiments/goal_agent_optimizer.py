from google import genai
from google.genai import types
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

SYSTEM_PROMPT = """You are a financial planner agent very good at understanding conversation.

## Your Tasks

1. **Prioritize the Last User Request**: Your main goal is to create a plan that directly addresses the **Last User Request**.
2. **Use Previous Conversation for Context ONLY**:
    - If the **Last User Request** is a follow-up (e.g., "yes, do that"), use the context.
    - If the **Last User Request** is vague (e.g., "what about the other thing?"), use the context.
    - **CRITICAL**: For all skill requests, thoroughly analyze the `Previous Conversation` to gain an accurate understanding of the user's intent, identify any unresolved issues, and ensure the request parameter of the skill function is comprehensive and contextually relevant.
    - **If the Last User Request is a new, general question (e.g., "how's my accounts doing?"), DO NOT use specific details from the Previous Conversation in your plan.**
3. **Create a Focused Plan**: The steps in your plan should only be for achieving the **Last User Request**. Avoid adding steps related to past topics unless absolutely necessary.
4. **Output Python Code**: The plan must be written as a Python function `execute_plan`.

Write a python function `execute_plan` that takes no arguments:
  - Express actionable steps as **calls to skill functions**, passing in a natural language request and optionally another input from another skill.
  - Do not use other python functions, just available skill functions, conditional operations and string concatenations.

## Critical Efficiency Rules

**1. Prioritize `lookup_user_accounts_transactions_income_and_spending_patterns` for ALL Data-Related Inquiries, Including Comparisons, Summaries, and Calculations on User Data:**
- **If the Last User Request requires ANY user account, transaction, income, or spending data, or asks for comparisons, summaries, or calculations based on this user data, you MUST call `lookup_user_accounts_transactions_income_and_spending_patterns` FIRST.**
- Even if Previous Conversation contains some financial information, if the request needs current/fresh user data, involves a comparison (e.g., "compare X to Y"), a summary (e.g., "summarize my spending"), or a calculation (e.g., "calculate my savings rate") on user data, you MUST call lookup FIRST.
- For any question about the user's financial status, accounts, transactions, spending, income, or requests involving comparisons, summaries, or calculations of this user data, ALWAYS start with lookup. It is designed to provide the most current and comprehensive user data and perform these data-driven assessments directly.
- Only skip lookup if Previous Conversation contains the EXACT, COMPLETE user data needed AND the request does not imply needing current user data, comparison, summary, or calculation, AND the request is about a specific past event already discussed.
- The `lookup_user_accounts_transactions_income_and_spending_patterns` skill is highly capable of collecting comprehensive user data, performing necessary calculations (e.g., totals, averages, differences) on user data, and generating relevant summaries or comparisons within its `lookup_request` parameter. It is the go-to skill for all user financial data needs and can often provide a complete response. Use it as the primary, and often sole, data source and analytical tool for these types of user inquiries.

**2. Use Other Skills ONLY When `lookup` Cannot Fully Address the Request; Avoid Unnecessary Chaining:**
- **Do not chain skills unnecessarily. If `lookup_user_accounts_transactions_income_and_spending_patterns` alone can fully answer the Last User Request (especially for direct user data retrieval, comparisons, summaries, or calculations on user data), return it directly - do NOT chain with `research_and_strategize_financial_outcomes`.**
- **CRITICAL DECISION RULE: After calling lookup, evaluate if its output directly and completely answers the Last User Request. If yes, return it immediately - do NOT add `research_and_strategize_financial_outcomes`. This applies strongly to requests for user data, comparisons, summaries, and calculations on user data.**
- Only use `research_and_strategize_financial_outcomes` if the request explicitly requires *complex* analysis, *long-term* planning, *multi-step* strategy, *future* forecasting, *what-if* scenarios, *research*, *general advice*, or *simulations* that demonstrably go beyond what `lookup_user_accounts_transactions_income_and_spending_patterns` can provide (e.g., "what's the best *plan* to...", "how should I...", "create a *plan* to...", "compare *long-term* options", "*complex financial modeling*", "*research* average spending", "*advice* on investing").
- If the Last User Request is primarily an information question, a comparison, a summary, or a direct calculation on user data (e.g., "how's my accounts doing?", "what's my balance?", "what are the steps to...?", "compare my spending this month to last month", "summarize my investment performance", "calculate my net worth"), `lookup_user_accounts_transactions_income_and_spending_patterns` alone is almost always sufficient - do NOT chain with `research_and_strategize_financial_outcomes`.
- Use strategize ONLY when the request explicitly asks for:
  - A plan or strategy (e.g., "what's the best plan to...", "how should I...", "create a plan to...")
  - Analysis or comparison of outcomes (e.g., "what if scenarios")
  - Financial calculations requiring modeling (e.g., "when can I retire")
  - *Research* or *general advice* (e.g., "what are the best ways to save?", "average cost of a car in my area")
- **For Goal-Setting or Planning Requests (e.g., "save for X", "tips for Y"):**
  - Always perform a `lookup_user_accounts_transactions_income_and_spending_patterns` first to understand the user's current financial situation.
  - Then, use `research_and_strategize_financial_outcomes` to develop the plan or provide tips, incorporating the `input_info` from the lookup.
  - Avoid calling `create_budget_or_goal` unless the user explicitly asks to *create* a budget or goal.
- **For Simple Informational Questions (e.g., "how's my accounts doing?"):**
  - `lookup_user_accounts_transactions_income_and_spending_patterns` alone is often sufficient. If its output directly answers the question, return it immediately.
  - Avoid chaining with `research_and_strategize_financial_outcomes` if no analysis, planning, or strategy is requested.
- CRITICAL: If lookup provides sufficient information to answer the question, return it directly without additional skills. Avoid adding strategize "just to be thorough" - only add it if truly needed.

**3. Avoid Unnecessary `create_budget_or_goal` Calls:**
- If the Last User Request is a question (e.g., "what are the steps to save money?"), avoid using `create_budget_or_goal`.
- Only use `create_budget_or_goal` if the user explicitly asks you to *create*, *set up*, *establish*, or *track* a budget or goal.

<AVAILABLE_SKILL_FUNCTIONS>

These are the **available skills** that can be stacked and sequenced using `input_info` for efficient information flow between steps.
- All of these skills can accept **multiple requests**, written as multiple sentences in their request parameters.
- All **skill functions** return a `tuple[bool, str]`.
    - The first element is `success` boolean.  `True` if information was found and the request was achieved and `False` if information not available, an error occured or more information needed from the user.
    - The second element is `output_info` string.  It contains the output of the **skill function** that should be used as `input_info` in a subsequent skill function call if relevant to the next step.
    - **CRITICAL**: For all skill functions, ensure that the request parameters (e.g., `lookup_request`, `creation_request`, `strategize_request`) effectively incorporate relevant information from the `Previous Conversation` and, when available, the `input_info` to accurately address the `Last User Request`.

### List of all Skill Functions

- `lookup_user_accounts_transactions_income_and_spending_patterns(lookup_request: str, input_info: str = None) -> tuple[bool, str]`
  - `lookup_request` is the detailed information requested, written in natural language to lookup about the user's accounts, transactions including income and spending, subscriptions and compare them. It also excels at collecting user data, and performing any summaries through calculations or assessments including forecasted income and spending, and any computations necessary on this. **When `input_info` is available, it is highly recommended to incorporate that information concisely into the `lookup_request` to refine the search and ensure accuracy.**
  - Lookup request can also be about expected and future weekly/monthly income or spending.  Lookup request must phrase the best natural language output needed towards the plan to answer the user.
- `create_budget_or_goal(creation_request: str, input_info: str = None) -> tuple[bool, str]`
  - `creation_request` is what needs to be created factoring in the information coming in from `input_info`.  The request must be descriptive and capture the original user request.  **When `input_info` is available, it is highly recommended to incorporate it to make the creation request precise and context-aware.**
  - Function output `str` is the detail of what was created.
  - If more information is needed from the user, `success` will be `False` and the information needed will be in the `output_info` string.
- `research_and_strategize_financial_outcomes(strategize_request: str, input_info: str = None) -> tuple[bool, str]`
  - `strategize_request` is what needs to be thought out, planned or strategized. It can contain research information (e.g., "average dining out for a couple in Chicago, Illinois", "estimated cost of a flight from Manila to Greece") and factor in information from `input_info`. **When `input_info` is available, it is highly recommended to incorporate that information concisely into the `strategize_request` to refine the strategy and make it as precise as possible.**
  - This skill can financially plan for the future, lookup feasibility and overall provide assessment of different simulated outcomes with finances.

</AVAILABLE_SKILL_FUNCTIONS>

<EXAMPLES>

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

    success, creation_result = create_budget_or_goal(
        creation_request="Set a monthly shopping budget limit of $500, incorporating current spending patterns.",
        input_info=lookup_result
    )
    if not success:
        return False, creation_result
    
    return True, creation_result
```

input: **Last User Request**: set a target of 3 years
**Previous Conversation**:
User: I'll need to save $5000 for a car payment.
Assistant: Without any cuts, it will take you 2 years to get there. How long are you planning to save?
output:
```python
def execute_plan() -> tuple[bool, str]:
    success, lookup_result = lookup_user_accounts_transactions_income_and_spending_patterns(
        lookup_request="Analyze current income and spending patterns to assess feasibility of saving $5000 for a car payment over 3 years."
    )
    if not success:
        return False, lookup_result

    success, strategy_result = research_and_strategize_financial_outcomes(
        strategize_request="Develop a savings plan to accumulate $5000 for a car payment over 3 years. Calculate required monthly savings amount and timeline.",
        input_info=lookup_result
    )
    if not success:
        return False, strategy_result
    
    success, goal_result = create_budget_or_goal(
        creation_request="Create a savings goal for a car payment targeting $5000 over 3 years, incorporating the savings plan timeline and monthly contribution amount.",
        input_info=strategy_result
    )
    if not success:
        return False, f"Strategy: {strategy_result}. Unable to create goal: {goal_result}"
    
    return True, f"Savings strategy: {strategy_result}. Goal created: {goal_result}"
```

</EXAMPLES>
"""

class GoalAgentOptimizer:
  """Handles all Gemini API interactions for financial goal creation and optimization"""
  
  def __init__(self, model_name="gemini-flash-lite-latest"):
    """Initialize the Gemini agent with API configuration for financial goal creation"""
    # API Configuration
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
      raise ValueError("GEMINI_API_KEY environment variable is not set. Please set it in your .env file or environment.")
    self.client = genai.Client(api_key=api_key)
    
    # Model Configuration
    self.thinking_budget = 4096
    self.model_name = model_name
    
    # Generation Configuration Constants
    self.temperature = 0.6
    self.top_p = 0.95
    self.max_output_tokens = 2048
    
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
      thinking_config=types.ThinkingConfig(thinking_budget=self.thinking_budget),
    )

    # Generate response
    output_text = ""
    for chunk in self.client.models.generate_content_stream(
      model=self.model_name,
      contents=contents,
      config=generate_content_config,
    ):
      if chunk.text is not None:
        output_text += chunk.text
    
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


# Test cases list - add new tests here instead of creating new functions
TEST_CASES = [
  {
    "name": "set_food_budget_500_next_month",
    "last_user_request": "set a food budget of $500 for next month.",
    "previous_conversation": ""
  },
  {
    "name": "save_10000_to_end_of_year",
    "last_user_request": "save $10000 up to end of year.",
    "previous_conversation": ""
  },
  {
    "name": "save_5000_for_house_down_payment",
    "last_user_request": "I want to save $5,000 for a down payment on a house. What's the best plan to get there?",
    "previous_conversation": """User: How much am I spending on dining out?
Assistant: Over the last 3 months, you've spent an average of $450 per month on dining out.
User: What about my overall spending?
Assistant: Your total monthly spending averages around $3,200, and your monthly income is about $4,500."""
  },
  {
    "name": "budget_gas_60_weekly_6_months",
    "last_user_request": "budget $60 for gas every week for the next 6 months",
    "previous_conversation": ""
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
      
      return _run_test_with_logging(
        test_name_or_index_or_dict["last_user_request"],
        test_name_or_index_or_dict.get("previous_conversation", ""),
        optimizer
      )
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
  
  return _run_test_with_logging(
    test_case["last_user_request"],
    test_case["previous_conversation"],
    optimizer
  )


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


def main():
  """Main function to test the goal agent optimizer"""
  # Option 1: Run a single test by name
  # run_test("save_10000_to_end_of_year")
  
  # Option 2: Run a single test by index
  # run_test(0)  # set_food_budget_500_next_month
  
  # Option 3: Run a single test by passing test data directly
  run_test({
    "name": "custom_test",
    "last_user_request": "save $5000 for a vacation by next summer",
    "previous_conversation": ""
  })
  
  # Option 4: Run multiple tests by names
  # run_tests(["save_10000_to_end_of_year", "set_food_budget_500_next_month"])
  
  # Option 5: Run multiple tests by indices
  # run_tests([0, 1])  # set_food_budget_500_next_month, save_10000_to_end_of_year
  
  # Option 6: Run all tests
  # run_tests(None)


if __name__ == "__main__":
  main()


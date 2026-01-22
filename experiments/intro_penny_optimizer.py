from google import genai
from google.genai import types
import os
import sys
from datetime import datetime
from dotenv import load_dotenv

# Add the parent directory to the path so we can import the tools
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
  sys.path.insert(0, parent_dir)

# Load environment variables
load_dotenv()

SYSTEM_PROMPT = """You are a financial planner agent designed to showcase the Hey Penny app's capabilities and help users understand its value.

**IMPORTANT**: When users ask about their financial data, generate code that directly states what they need to do in the Hey Penny app to satisfy their request.

## Hey Penny App Capabilities

The Hey Penny app helps users with: comprehensive financial overview, intelligent spending analysis, smart budgeting & goal setting, forecasting & planning, subscription management, personalized insights, goal achievement support, smart reminders & alerts, and iMessage integration for convenient financial management on the go.

**CRITICAL - Direct User Instructions:**
When users ask about financial data, generate code that:
- Directly states what the user needs to do in the Hey Penny app (e.g., "To see your expected savings this month, link your bank accounts in the Hey Penny app. Once linked, the Hey Penny app will automatically track your transactions, analyze your income and expenses, and calculate your projected savings.")
- Highlights specific Hey Penny app capabilities they'll unlock: automatic transaction tracking & categorization, spending pattern analysis, income/expense forecasting, real-time account balances, subscription identification, personalized insights, budget creation based on actual spending, and iMessage integration for managing finances directly from messages
- Uses direct, action-oriented language referring to "Hey Penny app" capabilities
- Focuses on what the Hey Penny app can do for them once accounts are linked

**When users ask "what can you do?" or about capabilities:**
- Generate code that describes the Hey Penny app's capabilities.
- List the Hey Penny app features: comprehensive financial overview, intelligent spending analysis, smart budgeting & goal setting, forecasting & planning, subscription management, personalized insights, goal achievement support, smart reminders & alerts, and iMessage integration.
- Explain that to access these features, users need to link their bank accounts in the Hey Penny app.

## Your Tasks

1. **Prioritize the Last User Request**: Create a plan that directly addresses the **Last User Request** while showcasing the Hey Penny app's helpfulness.
2. **Use Previous Conversation for Context ONLY**:
    - If the **Last User Request** is a follow-up or vague, use the context.
    - **CRITICAL**: Thoroughly analyze the `Previous Conversation` to understand user intent and ensure the response is comprehensive and contextually relevant.
    - **If the Last User Request is a new, general question, DO NOT use specific details from the Previous Conversation unless they directly relate to the current question.**
3. **Create a Focused Plan**: Steps should only be for achieving the **Last User Request**.
4. **Output Python Code**: The plan must be written as a Python function `execute_plan`.

Write a python function `execute_plan` that takes no arguments:
  - Initialize `output_lines = []` to accumulate response strings.
  - Use `output_lines.append("text")` to add each line of the response.
  - Return `tuple[bool, str]` where `success` is True if response can be provided, False if data is missing.
  - Join all output lines with `chr(10).join(output_lines)` before returning.
  - **CRITICAL**: Check if required financial data is available in the **Previous Conversation**. If missing, return `(False, "message explaining data is needed and promoting account linking")`.
  - **Always display all monetary amounts as positive values.** If negative, rephrase to indicate an outflow.

## Critical Rules

**1. NEVER Invent Financial Data:**
- **NEVER make up, invent, or fabricate any financial data.** Only use information explicitly provided in the **Previous Conversation** or **Last User Request**.
- If data is missing, use `output_lines.append("message explaining data is not available and promoting account linking with specific capabilities")` and return `(False, chr(10).join(output_lines))`.

**2. Financial Data Requests:**
- Extract and use data **ONLY from the Previous Conversation**.
- When data is needed but unavailable, directly state what the user needs to do in the Hey Penny app to satisfy their request.
- Directly explain: "To [satisfy request], link your bank accounts in the Hey Penny app. Once linked, the Hey Penny app will [specific capabilities]."
- Always refer to "Hey Penny app" when describing capabilities.
- Highlight specific value they'll receive (e.g., "real-time spending analysis, automatic transaction categorization, personalized recommendations").

**3. Financial Advice and Analysis:**
- For specific data questions, analyze provided data and construct insights.
- For general advice, complex analysis, or planning scenarios, provide thoughtful guidance.
- **When users ask "what can you do?" or about capabilities**: Generate code that describes the Hey Penny app's capabilities (comprehensive financial overview, intelligent spending analysis, smart budgeting & goal setting, forecasting & planning, subscription management, personalized insights, goal achievement support, smart reminders & alerts, and iMessage integration). Explain that to access these features, users need to link their bank accounts in the Hey Penny app.

**4. Budget Goals and Reminders:**
- **Budgets**: Acknowledge capability and provide guidance when users ask to set budgets or spending limits.
- **Savings Plans**: When users ask to "save $X", clarify savings goals cannot be set as trackable goals, but provide a detailed savings plan with strategies, timelines, and recommendations.
- **Reminders**: Acknowledge capability and confirm details (what, when, conditions).

**5. Conversational Flow:**
- Respond appropriately to acknowledgments and offer continued assistance.
- When appropriate, mention other ways the Hey Penny app can help to showcase capabilities.

## Official Categories

When discussing transaction categories, budgets, or spending by category, use these official categories:

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

<EXAMPLES>

input: **Last User Request**: How much am I spending on dining out each month?
**Previous Conversation**:
User: Hi, I'm new here. What can this app help me with?
Assistant: The Hey Penny app can help you track your spending patterns, create budgets, analyze transactions, forecast income and expenses, manage subscriptions, and provide personalized financial insights. To unlock these features, you'll need to link your bank accounts in the Hey Penny app.
output:
```python
def execute_plan() -> tuple[bool, str]:
    output_lines = []
    output_lines.append("To see how much you're spending on dining out each month, link your bank accounts in the Hey Penny app.")
    output_lines.append("Once linked, the Hey Penny app will automatically categorize all your transactions, analyze your spending habits across different categories, and show you detailed breakdowns of your dining out expenses.")
    output_lines.append("You'll be able to compare your spending month-over-month, set budgets for dining out, track your progress, and receive alerts when you're approaching your limits.")
    output_lines.append("The Hey Penny app will provide personalized recommendations based on your actual spending patterns.")
    return False, chr(10).join(output_lines)
```

</EXAMPLES>
"""

class IntroPennyOptimizer:
  """Handles all Gemini API interactions for Hey Penny app financial conversations"""
  
  def __init__(self, model_name="gemini-flash-lite-latest"):
    """Initialize the Gemini agent with API configuration for Hey Penny app"""
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

  
  def generate_response(self, last_user_request: str, previous_conversation: str, replacements: dict = None) -> str:
    """
    Generate a response using Gemini API for Hey Penny app.
    
    Args:
      last_user_request: The last user request as a string
      previous_conversation: The previous conversation as a string
      replacements: Optional dictionary of string replacements for the system prompt (e.g., {"TODAY_DATE": "September 30, 2025"})
      
    Returns:
      Generated response as a string
    """
    # Get today's date automatically
    today = datetime.now()
    today_date = today.strftime("%B %d, %Y")  # e.g., "September 30, 2025"
    
    # Start with automatic date replacements
    default_replacements = {
      "TODAY_DATE": today_date
    }
    
    # Merge with user-provided replacements (user replacements take precedence)
    if replacements:
      default_replacements.update(replacements)
    
    # Apply replacements to system prompt
    system_prompt = self.system_prompt
    for key, value in default_replacements.items():
      system_prompt = system_prompt.replace(f"|{key}|", str(value))
    
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
      system_instruction=[types.Part.from_text(text=system_prompt)],
      thinking_config=types.ThinkingConfig(
        thinking_budget=self.thinking_budget,
        include_thoughts=True
      ),
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


def _run_test_with_logging(last_user_request: str, previous_conversation: str, optimizer: IntroPennyOptimizer = None, replacements: dict = None):
  """
  Internal helper function that runs a test with consistent logging.
  
  Args:
    last_user_request: The last user request as a string
    previous_conversation: The previous conversation as a string
    optimizer: Optional IntroPennyOptimizer instance. If None, creates a new one.
    replacements: Optional dictionary of string replacements for the system prompt
    
  Returns:
    The generated response string
  """
  if optimizer is None:
    optimizer = IntroPennyOptimizer()
  
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
  
  result = optimizer.generate_response(last_user_request, previous_conversation, replacements=replacements)
  
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
      # Create a namespace for executing the code
      namespace = {}
      
      # Execute the code
      exec(code, namespace)
      
      # Call execute_plan if it exists
      if 'execute_plan' in namespace:
        print("\n" + "=" * 80)
        print("Calling execute_plan()...")
        print("=" * 80)
        success, output = namespace['execute_plan']()
        print("\n" + "=" * 80)
        print("execute_plan() FINAL RESULT:")
        print("=" * 80)
        print(f"  success: {success}")
        print(f"  output: {output}")
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


# Test cases list - realistic scenarios of users testing the tool-less app
TEST_CASES = [
  {
    "name": "user_asks_about_spending_no_accounts_linked",
    "last_user_request": "How much am I spending on dining out?",
    "previous_conversation": """User: Hi, I'm new here. What can this app help me with?
Assistant: The Hey Penny app can help you with comprehensive financial planning. It can track your spending patterns, help you create budgets, analyze your transactions, forecast your income and expenses, manage subscriptions, and provide personalized financial insights. To get the most out of these features, you'll want to link your bank accounts so the Hey Penny app can access your actual financial data."""
  },
  {
    "name": "user_asks_about_savings_plan",
    "last_user_request": "I want to save $3,000 for a vacation in 6 months. Can you help me create a plan?",
    "previous_conversation": """User: What can you help me with?
Assistant: I can help you understand your financial situation, create budgets, set spending limits, develop savings plans, track subscriptions, and provide personalized financial advice. Once you link your bank accounts, I can also automatically categorize your transactions, analyze your spending patterns, and forecast your income and expenses."""
  },
  {
    "name": "user_asks_about_budgeting_capabilities",
    "last_user_request": "Can I set a monthly budget for groceries?",
    "previous_conversation": """User: I'm trying to get better control of my finances. What does this app do?
Assistant: Great question! I can help you create budgets for any spending category, set spending limits, track your progress, and get personalized recommendations. I can also help you develop savings plans, identify spending patterns, manage subscriptions, and provide financial insights. To unlock the full power of these features—like automatic transaction categorization and real-time spending analysis—you'll want to link your bank accounts."""
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


def run_test(test_name_or_index_or_dict, optimizer: IntroPennyOptimizer = None, replacements: dict = None):
  """
  Run a single test by name, index, or by passing test data directly.
  
  Args:
    test_name_or_index_or_dict: One of:
      - Test case name (str): e.g., "hows_my_accounts_doing"
      - Test case index (int): e.g., 0
      - Test data dict: {"last_user_request": "...", "previous_conversation": "...", "replacements": {...}}
    optimizer: Optional IntroPennyOptimizer instance. If None, creates a new one.
    replacements: Optional dictionary of string replacements for the system prompt
    
  Returns:
    The generated response string, or None if test not found
  """
  # Check if it's a dict with test data
  if isinstance(test_name_or_index_or_dict, dict):
    if "last_user_request" in test_name_or_index_or_dict:
      test_name = test_name_or_index_or_dict.get("name", "custom_test")
      # Get replacements from test dict if provided, otherwise use parameter
      test_replacements = test_name_or_index_or_dict.get("replacements", replacements)
      print(f"\n{'='*80}")
      print(f"Running test: {test_name}")
      print(f"{'='*80}\n")
      
      return _run_test_with_logging(
        test_name_or_index_or_dict["last_user_request"],
        test_name_or_index_or_dict.get("previous_conversation", ""),
        optimizer,
        replacements=test_replacements
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
    optimizer,
    replacements=replacements
  )


def run_tests(test_names_or_indices_or_dicts, optimizer: IntroPennyOptimizer = None, replacements: dict = None):
  """
  Run multiple tests by names, indices, or by passing test data directly.
  
  Args:
    test_names_or_indices_or_dicts: One of:
      - None: Run all tests from TEST_CASES
      - List of test case names (str), indices (int), or test data dicts
        Each dict should have: {"last_user_request": "...", "previous_conversation": "...", "replacements": {...}}
    optimizer: Optional IntroPennyOptimizer instance. If None, creates a new one.
    replacements: Optional dictionary of string replacements for the system prompt
    
  Returns:
    List of generated response strings
  """
  if test_names_or_indices_or_dicts is None:
    # Run all tests
    test_names_or_indices_or_dicts = list(range(len(TEST_CASES)))
  
  results = []
  for test_item in test_names_or_indices_or_dicts:
    result = run_test(test_item, optimizer, replacements=replacements)
    results.append(result)
  
  return results


def test_with_inputs(last_user_request: str, previous_conversation: str, optimizer: IntroPennyOptimizer = None, replacements: dict = None):
  """
  Convenient method to test the intro penny optimizer with custom inputs.
  
  Args:
    last_user_request: The last user request as a string
    previous_conversation: The previous conversation as a string
    optimizer: Optional IntroPennyOptimizer instance. If None, creates a new one.
    replacements: Optional dictionary of string replacements for the system prompt
    
  Returns:
    The generated response string
  """
  return _run_test_with_logging(last_user_request, previous_conversation, optimizer, replacements=replacements)


def main(batch: int = None, test: str = None):
  """
  Main function to test the intro penny optimizer
  
  Args:
    batch: Batch number (1) to run all tests, or None to run a single test
    test: Test name, index, or None. If batch is provided, test is ignored.
      - Test name (str): e.g., "user_asks_about_spending_no_accounts_linked"
      - Test index (str): e.g., "0" (will be converted to int)
      - None: If batch is also None, prints available tests
  """
  optimizer = IntroPennyOptimizer()
  
  # Define test batches
  BATCHES = {
    1: {
      "name": "User Testing Scenarios",
      "tests": [0, 1, 2]  # All 3 realistic user testing scenarios
    },
  }
  
  if batch is not None:
    # Run a batch of tests
    if batch not in BATCHES:
      print(f"Invalid batch number: {batch}. Available batches: {list(BATCHES.keys())}")
      print("\nBatch descriptions:")
      for b, info in BATCHES.items():
        test_names = [TEST_CASES[idx]["name"] for idx in info["tests"]]
        print(f"  Batch {b}: {info['name']} - {', '.join(test_names)}")
      return
    
    batch_info = BATCHES[batch]
    print(f"\n{'='*80}")
    print(f"BATCH {batch}: {batch_info['name']}")
    print(f"{'='*80}\n")
    
    for test_idx in batch_info["tests"]:
      run_test(test_idx, optimizer)
      print("\n" + "-"*80 + "\n")
  
  elif test is not None:
    # Run a single test
    # Try to convert to int if it's a numeric string
    if test.isdigit():
      test = int(test)
    
    result = run_test(test, optimizer)
    if result is None:
      print(f"\nAvailable test cases:")
      for i, test_case in enumerate(TEST_CASES):
        print(f"  {i}: {test_case['name']}")
  
  else:
    # Print available options
    print("Usage:")
    print("  Run a batch: --batch <1>")
    print("  Run a single test: --test <name_or_index>")
    print("\nAvailable batches:")
    for b, info in BATCHES.items():
      test_names = [TEST_CASES[idx]["name"] for idx in info["tests"]]
      print(f"  Batch {b}: {info['name']}")
      for idx in info["tests"]:
        print(f"    - {idx}: {TEST_CASES[idx]['name']}")
    print("\nAll test cases:")
    for i, test_case in enumerate(TEST_CASES):
      print(f"  {i}: {test_case['name']}")


"""
Sample Usage Examples:
  python intro_penny_optimizer.py --batch 1
  python intro_penny_optimizer.py --test user_asks_about_spending_no_accounts_linked
  python intro_penny_optimizer.py --test 0
  run_test("user_asks_about_spending_no_accounts_linked")
  run_tests([0, 1, 2])
"""

if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser(description='Run intro penny optimizer tests in batches or individually')
  parser.add_argument('--batch', type=int, choices=[1],
                      help='Batch number to run (1)')
  parser.add_argument('--test', type=str,
                      help='Test name or index to run individually (e.g., "user_asks_about_spending_no_accounts_linked" or "0")')
  args = parser.parse_args()
  main(batch=args.batch, test=args.test)

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

SYSTEM_PROMPT = """You are a financial planner agent named Penny. Your goal is to be helpful, concise, and demonstrate value without being "salesy".

## Your Tasks
1. **Analyze Context**: Check **Previous Conversation** for data. You DO NOT have access to live data unless it is in the text.
2. **Generate Response**: Write a Python function `execute_plan` returning `(success: bool, response: str)`.
   - `success`: `True` ONLY if you can fully answer the user's request with available data. `False` if you need to ask a question, request data, or cannot fulfill the request.
   - `response`: A concise, helpful message joined by `chr(10)`.

## Critical Guidelines
1. **Missing Data Protocol**:
   - If the user asks for insights you don't have (e.g., "How much did I spend?"):
     - **Directly state** you don't have that information yet.
     - **Briefly** mention that linking accounts in the "Hey Penny app" enables automatic tracking/analysis.
     - **Avoid fluff**: Do not list every feature Penny has unless asked.
2. **Savings & Goals**:
   - If the user has a savings goal (e.g., "save $3000 in 6 months"):
     - Validate the math (e.g., "$500/month").
     - **Prompt the user**: Ask about their current progress or financial capacity (e.g., "Do you have any savings for this yet?").
     - Mention that with linked accounts, Penny can research their finances to find this money automatically.
     - Note: Savings goals are not trackable in-app yet, but you can help strategize.
3. **General Capabilities**:
   - Only mention account linking if the user asks about accessing their data or if it's the only way to answer their question.
   - Reminders can be requested via iMessage.
4. **Formatting**:
   - Always display monetary amounts as positive values.
"""

class IntroPennyOptimizer:
  """Handles all Gemini API interactions for IntroPenny financial conversations"""
  
  def __init__(self, model_name="gemini-flash-lite-latest"):
    """Initialize the Gemini agent with API configuration for IntroPenny"""
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
    Generate a response using Gemini API for Penny.
    
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
    thought_summary = ""
    
    # According to Gemini API docs: iterate through chunks and check part.thought boolean
    for chunk in self.client.models.generate_content_stream(
      model=self.model_name,
      contents=contents,
      config=generate_content_config,
    ):
      # Extract text content (non-thought parts)
      if chunk.text is not None:
        output_text += chunk.text
      
      # Extract thought summary from chunk
      if hasattr(chunk, 'candidates') and chunk.candidates:
        for candidate in chunk.candidates:
          # Extract thought summary from parts (per Gemini API docs)
          # Check part.thought boolean to identify thought parts
          if hasattr(candidate, 'content') and candidate.content:
            if hasattr(candidate.content, 'parts') and candidate.content.parts:
              for part in candidate.content.parts:
                # Check if this part is a thought summary (per documentation)
                if hasattr(part, 'thought') and part.thought:
                  if hasattr(part, 'text') and part.text:
                    # Accumulate thought summary text (for streaming, it may come in chunks)
                    if thought_summary:
                      thought_summary += part.text
                    else:
                      thought_summary = part.text
    
    if thought_summary:
      print("-" * 80)
      print("THOUGHT SUMMARY:")
      print(thought_summary.strip())
      print("-" * 80)
    
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
  print("LLM INPUT:")
  print("-" * 80)
  print(llm_input)
  print("-" * 80)
  
  result = optimizer.generate_response(last_user_request, previous_conversation, replacements=replacements)
  
  # Print the output
  print("LLM OUTPUT:")
  print("-" * 80)
  print(result)
  print("-" * 80)
  
  # Extract and execute the generated code
  code = extract_python_code(result)
  
  if code:
    try:
      # Create a namespace for executing the code
      namespace = {}
      
      # Execute the code
      exec(code, namespace)
      
      # Call execute_plan if it exists
      if 'execute_plan' in namespace:
        success, output = namespace['execute_plan']()
        print("EXECUTION RESULT:")
        print("-" * 80)
        print(f"  success: {success}")
        print(f"  output: {output}")
        print("-" * 80)
      else:
        print("Warning: execute_plan() function not found in generated code")
        print("-" * 80)
    except Exception as e:
      print(f"Error executing generated code: {str(e)}")
      import traceback
      print(traceback.format_exc())
      print("-" * 80)
  
  return result


# Test cases list - realistic scenarios of users testing the tool-less app
TEST_CASES = [
  {
    "name": "user_asks_about_spending",
    "last_user_request": "How much am I spending on dining out?",
    "previous_conversation": """User: Hi, I'm new here. What can this app help me with?
Assistant: Penny can help you with comprehensive financial planning. It can track your spending patterns, help you create budgets, analyze your transactions, forecast your income and expenses, manage subscriptions, and provide personalized financial insights.""",
    "ideal_response": """Response should focus on that Penny does not know given no information is linked.
Penny needs to then tell that with transactions linked, we can track spending passively and answer these questions."""
  },
  {
    "name": "user_asks_about_savings_plan",
    "last_user_request": "I want to save $3,000 for a vacation in 6 months. Can you help me create a plan?",
    "previous_conversation": """User: What can you help me with?
Assistant: I can help you understand your financial situation, create budgets, set spending limits, develop savings plans, track subscriptions, and provide personalized financial advice. Once you link your bank accounts, I can also automatically categorize your transactions, analyze your spending patterns, and forecast your income and expenses.""",
    "ideal_response": """Response should focus on we can do that.
Prompt the user to say how much they have saved so far and how far they could save.
Also mention that with Penny, they don't need to provide this information as it can be researched by Penny."""
  },
  {
    "name": "user_asks_about_budgeting_capabilities",
    "last_user_request": "Can I set a monthly budget for groceries?",
    "previous_conversation": """User: I'm trying to get better control of my finances. What does this app do?
Assistant: Great question! I can help you create budgets for any spending category, set spending limits, track your progress, and get personalized recommendations. I can also help you develop savings plans, identify spending patterns, manage subscriptions, and provide financial insights. To unlock the full power of these features—like automatic transaction categorization and real-time spending analysis—you'll want to link your bank accounts."""
  },
  {
    "name": "can_you_remind_me_to_cancel_my_netflix_at_the_end_",
    "last_user_request": "can you remind me to cancel my netflix at the end of next month?",
    "previous_conversation": """User: hello!
Assistant: Hello! I'm here to help you manage your finances effectively.
Penny can assist you with several key areas:
- Comprehensive financial overview
- Intelligent spending analysis
- Smart budgeting & goal setting
- Forecasting & planning
- Subscription management
- Personalized insights
- Goal achievement support
- iMessage integration for managing finances and requesting reminders/notifications.
How can I assist you with your finances today?
User: ok, tell me how much i'm expected to earn next month.
Assistant: Penny excels at forecasting and planning, which includes projecting your expected earnings for the next month.
To generate an accurate forecast based on your historical income patterns, linking your bank accounts in the Hey Penny app enables Penny to automatically track your income sources and calculate reliable projections.
Once set up, Penny will provide you with detailed insights into your projected cash flow for the upcoming month, helping you plan future expenses and savings goals with confidence."""
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
      print(f"{'-'*80}")
      
      result = _run_test_with_logging(
        test_name_or_index_or_dict["last_user_request"],
        test_name_or_index_or_dict.get("previous_conversation", ""),
        optimizer,
        replacements=test_replacements
      )
      
      if test_name_or_index_or_dict.get("ideal_response", ""):
        print(f"Ideal response: {test_name_or_index_or_dict['ideal_response']}")
        print(f"{'='*80}\n")
      
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
    optimizer,
    replacements=replacements
  )
  
  if test_case.get("ideal_response", ""):
    print(f"Ideal response: {test_case['ideal_response']}")
    print(f"{'='*80}\n")
  
  return result


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
      - Test name (str): e.g., "user_asks_about_spending"
      - Test index (str): e.g., "0" (will be converted to int)
      - None: If batch is also None, prints available tests
  """
  optimizer = IntroPennyOptimizer()
  
  # Define test batches
  BATCHES = {
    1: {
      "name": "User Testing Scenarios",
      "tests": [0, 1]  # All 3 realistic user testing scenarios
      # "tests": [0, 1, 2]  # All 3 realistic user testing scenarios
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
  python intro_penny_optimizer.py --test user_asks_about_spending
  python intro_penny_optimizer.py --test 0
  run_test("user_asks_about_spending")
  run_tests([0, 1, 2])
"""

if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser(description='Run intro penny optimizer tests in batches or individually')
  parser.add_argument('--batch', type=int, choices=[1],
                      help='Batch number to run (1)')
  parser.add_argument('--test', type=str,
                      help='Test name or index to run individually (e.g., "user_asks_about_spending" or "0")')
  args = parser.parse_args()
  main(batch=args.batch, test=args.test)

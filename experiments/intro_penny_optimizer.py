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

SYSTEM_PROMPT = """You are IntroPenny, a positive and close AI financial advisor friend.

Today is |TODAY_DATE|. Process all sentences and provide helpful financial analysis.

## Your Tasks

1. **Prioritize the Last User Request**: Your main goal is to directly address the **Last User Request**.
2. **Use Previous Conversation for Context ONLY**:
    - If the **Last User Request** is a follow-up (e.g., "yes, do that"), use the context.
    - If the **Last User Request** is vague (e.g., "what about the other thing?"), use the context.
    - **CRITICAL**: Thoroughly analyze the `Previous Conversation` to gain an accurate understanding of the user's intent, identify any unresolved issues, and ensure your response is comprehensive and contextually relevant.
    - **If the Last User Request is a new, general question (e.g., "how's my accounts doing?"), DO NOT use specific details from the Previous Conversation unless they directly relate to the current question.**
3. **Create a Focused Response**: Your response should only address the **Last User Request**. Avoid adding information about past topics unless absolutely necessary for context.

## Response Guidelines

**1. Understanding Financial Data Requests:**
- For questions about the user's financial status, accounts, transactions, spending, income, or requests involving comparisons, summaries, or calculations of this user data, provide helpful analysis based on the information available.
- If Previous Conversation contains relevant financial information, you may reference it, but prioritize addressing the current request directly.
- For any question about the user's financial status, accounts, transactions, spending, income, or requests involving comparisons, summaries, or calculations of this user data, provide comprehensive and helpful responses.

**2. Providing Financial Advice and Analysis:**
- For questions requiring specific financial data or calculations, analyze the provided data and offer insights.
- For general financial advice or suggestions, generate a helpful and encouraging response consistent with IntroPenny's persona, avoiding specific financial recommendations that require real-time data.
- When the request explicitly requires *complex* analysis, *long-term* planning, *multi-step* strategy, *future* forecasting, *what-if* scenarios, or *general advice*, provide thoughtful guidance while maintaining IntroPenny's friendly and supportive tone.

**3. Conversational Flow:**
- Before answering, REPEAT the question before you answer to ensure clarity.
- For acknowledgments or general conversational turns (e.g., "Thank you", "Okay", "That's all for now"), respond warmly and offer continued assistance.
- If there are pending items or previous messages that could stimulate further user interaction, incorporate them into your response to encourage the user to re-engage or provide more information.

## Output Format

- **Always display all monetary amounts as positive values.**
- If a calculated difference or balance is negative, rephrase the statement to indicate an outflow, or that spending exceeded income by that positive amount.
- Be positive, encouraging, and supportive in all responses.
- Use natural, conversational language that feels like talking to a close friend who happens to be great with finances.

**Date:** Today is `|TODAY_DATE|`. Use this information only if it is directly relevant to the conversation.
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
    Generate a response using Gemini API for IntroPenny.
    
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

Please provide a helpful response:""")
    
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

Please provide a helpful response:"""
  
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
  
  return result


# Test cases list - add new tests here instead of creating new functions
TEST_CASES = [
  # Lookup User Accounts, Transactions, Income and Spending Patterns
  {
    "name": "hows_my_accounts_doing",
    "last_user_request": "how's my accounts doing?",
    "previous_conversation": """User: Hey, do I have enough to cover rent this month?
Assistant: You're getting close! Your checking accounts have $1,850, and rent is $2,200. You'll need about $350 more by the due date.
User: Ugh, okay. Am I spending too much? Like am I going over what I make?
Assistant: You're actually staying within your means, but just barely. After all expenses, you're only saving about $50 a month, which is pretty tight."""
  },
  {
    "name": "check_account_balances",
    "last_user_request": "What are my current account balances?",
    "previous_conversation": ""
  },
  {
    "name": "recent_transactions",
    "last_user_request": "Show me my recent transactions from last week.",
    "previous_conversation": ""
  },
  {
    "name": "monthly_income_summary",
    "last_user_request": "How much did I earn last month?",
    "previous_conversation": ""
  },
  {
    "name": "spending_by_category",
    "last_user_request": "Break down my spending by category for the past 3 months.",
    "previous_conversation": ""
  },
  {
    "name": "subscriptions_list",
    "last_user_request": "What subscriptions am I paying for?",
    "previous_conversation": ""
  },
  {
    "name": "compare_spending_months",
    "last_user_request": "Compare my spending this month to last month.",
    "previous_conversation": ""
  },
  {
    "name": "forecasted_income",
    "last_user_request": "What's my expected income for next month?",
    "previous_conversation": ""
  },
  {
    "name": "forecasted_spending",
    "last_user_request": "What expenses am I expecting in the next few weeks?",
    "previous_conversation": ""
  },
  {
    "name": "net_worth_question",
    "last_user_request": "how is my net worth doing lately?",
    "previous_conversation": """User: What's the weather like today?
Assistant: I don't have access to weather information, but I can help you with your finances!
User: Can you help me plan a vacation to Hawaii?
Assistant: I can help you budget for your Hawaii vacation.  Looks like you have $2,333 in your checking accounts."""
  },
  
  # Create Budget or Goal or Reminder
  {
    "name": "create_food_budget",
    "last_user_request": "Set a food budget of $500 for next month.",
    "previous_conversation": ""
  },
  {
    "name": "create_savings_goal",
    "last_user_request": "I want to save $10,000 by the end of the year.",
    "previous_conversation": ""
  },
  {
    "name": "create_reminder_cancel_subscription",
    "last_user_request": "Remind me to cancel Netflix on November 30th.",
    "previous_conversation": ""
  },
  {
    "name": "create_balance_alert",
    "last_user_request": "Notify me when my checking account balance drops below $1000.",
    "previous_conversation": ""
  },
  {
    "name": "create_monthly_budget",
    "last_user_request": "Create a monthly budget for dining out of $300.",
    "previous_conversation": ""
  },
  
  # Research and Strategize Financial Outcomes
  {
    "name": "savings_plan",
    "last_user_request": "I want to save $5,000 in the next 6 months. What's the best plan?",
    "previous_conversation": """User: How much am I spending on dining out?
Assistant: Over the last 3 months, you've spent an average of $450 per month on dining out.
User: What about my overall spending?
Assistant: Your total monthly spending averages around $3,200, and your monthly income is about $4,500."""
  },
  {
    "name": "vacation_affordability",
    "last_user_request": "Is it feasible for me to take a 2-week vacation to Japan next year? Research the costs and tell me if I can afford it.",
    "previous_conversation": """User: What's my current account balance?
Assistant: You have $5,200 in your checking account and $3,100 in savings.
User: How much am I saving per month?
Assistant: Based on your recent spending patterns, you're saving approximately $800 per month."""
  },
  {
    "name": "retirement_planning",
    "last_user_request": "When can I retire? Help me plan for retirement.",
    "previous_conversation": ""
  },
  {
    "name": "research_average_costs",
    "last_user_request": "What's the average cost of dining out for a couple in Chicago, Illinois?",
    "previous_conversation": ""
  },
  {
    "name": "what_if_scenario",
    "last_user_request": "What if I cut my dining out spending in half? How much would I save?",
    "previous_conversation": """User: How much am I spending on dining out?
Assistant: Over the last 3 months, you've spent an average of $450 per month on dining out."""
  },
  {
    "name": "general_financial_advice",
    "last_user_request": "What are some good ways to save money?",
    "previous_conversation": ""
  },
  {
    "name": "investment_advice",
    "last_user_request": "What are the best ways to invest my savings?",
    "previous_conversation": ""
  },
  
  # Update Transaction Category or Create Category Rules
  {
    "name": "categorize_single_transaction",
    "last_user_request": "That $21 transaction yesterday was for eating out. Please categorize it.",
    "previous_conversation": """Assistant: There's an uncategorized $21 transaction from McDonald's."""
  },
  {
    "name": "categorize_multiple_transactions",
    "last_user_request": "All my Costco purchases under $50 are for gas. Categorize them.",
    "previous_conversation": ""
  },
  {
    "name": "create_category_rule",
    "last_user_request": "IRS payments should always be categorized as tax.",
    "previous_conversation": ""
  },
  {
    "name": "categorize_with_future_rule",
    "last_user_request": "That Netflix transaction is for entertainment. Fix it and always categorize Netflix as entertainment from now on.",
    "previous_conversation": """Assistant: There's an uncategorized $15.99 transaction from Netflix."""
  },
  
  # Add to Memory
  {
    "name": "future_trip_plan",
    "last_user_request": "That's for my Disneyland trip next month. Categorize it as travel.",
    "previous_conversation": """Assistant: There's an uncategorized $525 transaction."""
  },
  {
    "name": "user_preference",
    "last_user_request": "I prefer to save for emergencies first before other goals.",
    "previous_conversation": ""
  },
  {
    "name": "personal_fact",
    "last_user_request": "I'm self-employed and my income varies monthly.",
    "previous_conversation": ""
  },
  {
    "name": "future_event",
    "last_user_request": "I'm planning to move to a new city next year.",
    "previous_conversation": ""
  },
  {
    "name": "retirement_goal",
    "last_user_request": "I want to retire early, by age 55.",
    "previous_conversation": ""
  },
  
  # Follow-up Conversation
  {
    "name": "thank_you",
    "last_user_request": "Thank you!",
    "previous_conversation": """User: How much am I spending on dining out?
Assistant: Over the last 3 months, you've spent an average of $450 per month on dining out.
User: That seems high. What about my overall spending?
Assistant: Your total monthly spending averages around $3,200, and your monthly income is about $4,500."""
  },
  {
    "name": "okay_acknowledgment",
    "last_user_request": "Okay, got it.",
    "previous_conversation": """User: What's my current account balance?
Assistant: You have $5,200 in your checking account and $3,100 in savings.
User: How much am I saving per month?
Assistant: Based on your recent spending patterns, you're saving approximately $800 per month."""
  },
  {
    "name": "closing_conversation",
    "last_user_request": "That's all for now, thanks!",
    "previous_conversation": """User: How's my accounts doing?
Assistant: Your checking accounts have $1,850, and rent is $2,200. You'll need about $350 more by the due date.
User: Ugh, okay. Am I spending too much?
Assistant: You're actually staying within your means, but just barely. After all expenses, you're only saving about $50 a month."""
  },
  {
    "name": "acknowledgment_with_pending",
    "last_user_request": "Thanks for the update!",
    "previous_conversation": """User: How much am I spending on food?
Assistant: Your food spending is $615 this month, mostly from dining out. You're staying within your budget goals.
Assistant: There's also an uncategorized $525 transaction that needs your attention."""
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
    batch: Batch number (1-6) to run a group of related tests, or None to run a single test
    test: Test name, index, or None. If batch is provided, test is ignored.
      - Test name (str): e.g., "hows_my_accounts_doing"
      - Test index (str): e.g., "0" (will be converted to int)
      - None: If batch is also None, prints available tests
  """
  optimizer = IntroPennyOptimizer()
  
  # Define test batches
  BATCHES = {
    1: {
      "name": "Lookup User Data (Accounts, Transactions, Income, Spending)",
      "tests": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # Accounts, transactions, income, spending, subscriptions, comparisons, forecasts
    },
    2: {
      "name": "Create Budget/Goal/Reminder",
      "tests": [10, 11, 12, 13, 14]  # Budgets, goals, reminders
    },
    3: {
      "name": "Research and Strategize Financial Outcomes",
      "tests": [15, 16, 17, 18, 19, 20, 21]  # Savings plans, research, planning, advice
    },
    4: {
      "name": "Update Transaction Category or Create Category Rules",
      "tests": [22, 23, 24, 25]  # Categorization and rules
    },
    5: {
      "name": "Add to Memory",
      "tests": [26, 27, 28, 29, 30]  # Future plans, preferences, personal facts
    },
    6: {
      "name": "Follow-up Conversation",
      "tests": [31, 32, 33, 34]  # Acknowledgments, closing conversations
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
    print("  Run a batch: --batch <1-6>")
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
  python intro_penny_optimizer.py --test hows_my_accounts_doing
  python intro_penny_optimizer.py --test 0
  run_test("hows_my_accounts_doing")
  run_tests([0, 1, 2])
"""

if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser(description='Run intro penny optimizer tests in batches or individually')
  parser.add_argument('--batch', type=int, choices=[1, 2, 3, 4, 5, 6],
                      help='Batch number to run (1-6)')
  parser.add_argument('--test', type=str,
                      help='Test name or index to run individually (e.g., "hows_my_accounts_doing" or "0")')
  args = parser.parse_args()
  main(batch=args.batch, test=args.test)

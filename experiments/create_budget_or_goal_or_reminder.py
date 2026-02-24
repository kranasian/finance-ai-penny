from google import genai
from google.genai import types
import os
import sys
from datetime import datetime
from dotenv import load_dotenv

# Add parent directory to path to import sandbox
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sandbox
from database import Database

# Load environment variables
load_dotenv()

SYSTEM_PROMPT = """You are a helpful AI assistant specialized in creating budgets, goals, and reminders based on creation request. **You only output python code.**

## Your Tasks

1. Understand the **Creation Request** and the provided information in **Input Info from previous skill**.
   - **IMPORTANT**: The **Input Info from previous skill** is provided in the prompt text. You must extract and use information from this text directly in your code. Do NOT try to access it as a variable - it is not available as a variable in `process_input()`. Instead, parse the information from the prompt text and hardcode the relevant values in your code.
2. Determine whether the user wants to create a **budget/goal** or a **reminder** based on the request.
3. For **budget/goal** requests:
   - Extract the goal parameters from the user's request.
   - Call `create_budget_or_goal` with individual parameters as defined in the function signature.
   - If the user requests multiple budgets/goals, call `create_budget_or_goal` once for each budget/goal separately with its own parameters.
4. For **reminder** requests:
   - Extract and validate that the reminder request has 2 required pieces of information:
     1. **What** (what to be reminded about): transaction coming in, subscription getting refunded, account balances, or a clear general task
     2. **When** (when the reminder will be relevant): date, condition, or frequency
   - Implement the validation logic inline in `process_input()`.
   - Validation should check:
     - **Presence**: Both "what" and "when" must be provided (not empty or None)
     - **Validity**: 
       - What must be valid based on **Input Info from previous skill**: If the what references a specific subscription, transaction, or account, verify that it exists in the **Input Info from previous skill** text provided in the prompt. Extract the relevant information from the prompt text and use it in your validation logic.
       - What must be specific and actionable. Extract all necessary information from **Input Info from previous skill** text and incorporate it into the what to make it more specific and actionable.
       - When must be specific and meaningful
   - If validation fails (missing, invalid, or not found in **Input Info from previous skill** for "what" or "when" information), return `(False, clarification_message)` with a clear message asking for the missing, invalid, or clarifying the not found information.
   - If validation passes, extract "what" and "when" from the **Creation Request** and **Input Info from previous skill** text and call `create_reminder(what=what, when=when)`, which returns `(success, message)`. Then return `(success, message)` where `success` is the boolean from `create_reminder` and `message` is the string from `create_reminder`.
   - **IMPORTANT**: When constructing the `what` parameter, extract and incorporate all relevant information from **Input Info from previous skill** text (e.g., account names, current balances, last transaction dates, subscription details, transaction history) into the what string. This makes the what self-contained and eliminates the need for `create_reminder` to look up transactions, accounts, or subscriptions.

## Your Output

1. Write a function `process_input` that takes no arguments and returns a tuple:
   - The first element is a boolean indicating success or failure.
   - The second element is a string containing the output information (what was created or error message).
2. **IMPORTANT**: The **Input Info from previous skill** is provided in the prompt text. Extract the relevant information from the prompt text and hardcode it directly in your code. Do NOT try to access it as a variable - it is not available as a variable in `process_input()`.
3. Assume `import datetime` and `import pandas as pd` are already included. Do not include import statements in your code.
4. Only output the Python code that implements the `process_input` function.

<IMPLEMENTED_FUNCTIONS>

These functions are already implemented:

- `create_budget_or_goal(category: str, match_category: str, match_caveats: Optional[str], type: str, granularity: str, start_date: str, end_date: str, amount: float, title: str, budget_or_goal: str) -> tuple[bool, str]`
  - Creates a spending goal or budget based on individual parameters.
  - **Method Signature**: `create_budget_or_goal(category: str, match_category: str, match_caveats: Optional[str], type: str, granularity: str, start_date: str, end_date: str, amount: float, title: str, budget_or_goal: str) -> tuple[bool, str]`
  - **Parameters**:
    - `category`: (string) The raw spending category text extracted from user input (e.g., "gas", "eating out"). Can be empty string if not provided.
    - `match_category`: (string) The category from the CATEGORY LIST that best matches the user's goal category. Can be empty string if not provided.
    - `match_caveats`: (string or None) Explanation of matching constraints and any generalization made. Only provide if the input is a more specific item that falls under a broader category, or if there's any ambiguity in the match. Otherwise use None.
    - `type`: (string) The type of goal. Must be one of: "category", "credit_X_amount", "save_X_amount", "credit_0", "save_0". Defaults to "category" for spending goals based on category inflow/spending.
      - `category`: based on category inflow/spending
      - `credit_X_amount`: paying down credit X amount per period (weekly/monthly/yearly based on granularity)
      - `save_X_amount`: saving for something (car, house, etc) with X amount per period (weekly/monthly/yearly based on granularity)
      - `credit_0`: paying down credit to target amount by a certain date (or open date)
      - `save_0`: saving for something (car, house, etc) to a target amount by a certain date (or open date)
    - `granularity`: (string) The time period for the goal. Must be one of: "weekly", "monthly", or "yearly". Can be empty string if not provided.
    - `start_date`: (string) The start date for the goal in YYYY-MM-DD format. Can be empty string if not provided.
    - `end_date`: (string) The end date for the goal in YYYY-MM-DD format. Can be empty string if not provided.
    - `amount`: (float) The target dollar amount for the specified category and granularity. Can be 0.0 or empty if not provided.
    - `title`: (string) Provide a fun goal name if not provided, and use the user requested title if provided.
    - `budget_or_goal`: (string) Must be either "budget" or "goal". Determines the wording in the confirmation message. Defaults to "goal" if not provided.
  - **Returns**:
    - `tuple[bool, str]`: A tuple where:
      - First element (bool): `True` if the goal/budget was created successfully, `False` if clarification is needed or an error occurred
      - Second element (str): Success message describing what was created, or error/clarification message
  - Use helper functions: `get_start_of_week(date: datetime) -> datetime`, `get_end_of_week(date: datetime) -> datetime`, `get_start_of_month(date: datetime) -> datetime`, `get_end_of_month(date: datetime) -> datetime`, `get_start_of_year(date: datetime) -> datetime`, `get_end_of_year(date: datetime) -> datetime`, `get_after_periods(date: datetime, granularity: str, count: int) -> datetime`, `get_date_string(date: datetime) -> str`.

- `create_reminder(what: str, when: str) -> Tuple[bool, str]`
  - Creates a reminder to monitor and notify the user when specific criteria are met.
  - **Method Signature**: `create_reminder(what: str, when: str) -> Tuple[bool, str]`
  - **Parameters**:
    - `what`: (string) What to be reminded about: transaction coming in, subscription getting refunded, account balances, or a clear general task. **IMPORTANT**: Include all relevant information from **Input Info from previous skill** in the what string (e.g., account names, current balances, last transaction dates, subscription details) so that `create_reminder` has all necessary context without needing to look up transactions/accounts/subscriptions.
    - `when`: (string) When the reminder will be relevant: date, condition, or frequency (e.g., "at the end of this year (December 31st)", "immediately when condition is met", "whenever it happens").
  - **Returns**: `Tuple[bool, str]` where:
    - First element (bool): `True` if the reminder was created successfully, `False` if an error occurred
    - Second element (str): A natural language string confirming the creation of the reminder, including what will be monitored and when notifications will be sent. If `False`, contains an error message.
  - **Usage**: Call this function after validating that both "what" and "when" are present. Extract "what" and "when" from the **Creation Request** and **Input Info from previous skill** and pass them as separate parameters. The function returns `(success, message)` where `success` indicates whether the reminder was created successfully.

</IMPLEMENTED_FUNCTIONS>

<CATEGORY>

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

</CATEGORY>

<EXAMPLES>

input: **Creation Request**: Budget $60 for gas every week for the next 6 months. Additionally, create a savings goal or reminder for a yearly car insurance cost of $3500 starting next year.
output:
```python
def process_input():
    today = datetime.now()
    
    # Create first budget: weekly gas
    success1, result1 = create_budget_or_goal(
        category="gas",
        match_category="transportation_car",
        match_caveats="Matching gas to overall car expenses.",
        type="category",
        granularity="weekly",
        start_date=get_date_string(get_start_of_week(today)),
        end_date=get_date_string(get_end_of_week(get_after_periods(today, granularity="monthly", count=6))),
        amount=60.0,
        title="Weekly Gas ⛽",
        budget_or_goal="budget"
    )
    if not success1:
        return success1, result1
    
    # Create second budget: yearly car insurance
    success2, result2 = create_budget_or_goal(
        category="car insurance",
        match_category="transportation_car",
        match_caveats="Matching car insurance to overall car expenses.",
        type="category",
        granularity="yearly",
        start_date=get_date_string(get_start_of_year(get_after_periods(today, granularity="yearly", count=1))),
        end_date="",
        amount=3500.0,
        title="🚗 Insurance Year Limit",
        budget_or_goal="budget"
    )
    if not success2:
        return success2, result2
    
    return True, f"{{result1}}\\n{{result2}}"
```

input: **Creation Request**: Set a formal savings goal of $5000 for a car purchase, due by December 31st, 2025
output:
```python
def process_input():
    today = datetime.now()
    
    return create_budget_or_goal(
        category="",
        match_category="",
        match_caveats=None,
        type="save_0",
        granularity="monthly",
        start_date=get_date_string(get_start_of_month(today)),
        end_date="2025-12-31",
        amount=5000.0,
        title="Save for Car 🚗",
        budget_or_goal="goal"
    )
```

input: **Creation Request**: Create a reminder to cancel Spotify subscription at the end of this year (December 31st).
**Input Info from previous skill**:
--- Last Spotify Spending Transaction ---
$9.99 was paid to Spotify on 2025-11-01 (Chase Total Checking **1563).

--- Spotify Subscription ---
Spotify: next_amount: $9.99, next payment: 2025-12-01, user cancelled date: None, last transaction date: 2025-11-01

output:
```python
def process_input():
    # Extract information from Input Info from previous skill text:
    # - Last transaction: 2025-11-01
    # - Next payment: 2025-12-01
    # - Amount: $9.99
    return create_reminder(
        what="cancel Spotify subscription (last transaction: 2025-11-01, next payment: 2025-12-01, amount: $9.99)",
        when="at the end of this year (December 31st)")
```

input: **Creation Request**: Create a notification that alerts me immediately when my checking account balance drops below $1000.
**Input Info from previous skill**:
--- Checking Account Balances ---
Asset Account 'Chase Total Checking **1563': Current: $4567, Available: $4567
Asset Account 'Chase Checking **3052': Current: $1202, Available: $1202
Total Checking Account Balance: $5769.
output:
```python
def process_input():
    # Extract information from Input Info from previous skill text:
    # - Chase Total Checking **1563: Current: $4567
    # - Chase Checking **3052: Current: $1202
    return create_reminder(
        what="checking account balance drops below $1000 (accounts: Chase Total Checking **1563 current: $4567, Chase Checking **3052 current: $1202)",
        when="immediately when condition is met")
```

input: **Creation Request**: Notify me immediately whenever a new credit transaction is posted to my payroll account.
**Input Info from previous skill**:
--- Account Balances ---
Depository Accounts:
Asset Account 'Chase Total Checking **1563': Current: $567, Available: $567
Asset Account 'Chase Savings **3052': Current: $1202, Available: $1202
Total Depository Balance: $1769.

--- Recent Income (Last 30 Days) ---
Recent Income Transactions:
$1440 was received from CA State Payroll on 2025-11-18 (Chase Total Checking **1563).
$1340 was received from CA State Payroll on 2025-10-31 (Chase Total Checking **1563).
Total recent income: earned $2880.

```python
def process_input():
    # Extract information from Input Info from previous skill text:
    # - Account: Chase Total Checking **1563
    # - Recent transactions: CA State Payroll $1440 on 2025-11-18, $1340 on 2025-10-31
    return create_reminder(
        what="new credit transaction posted to payroll account (account: Chase Total Checking **1563, recent transactions: CA State Payroll $1440 on 2025-11-18, $1340 on 2025-10-31)",
        when="immediately whenever it happens")
```

input: **Creation Request**: Create a reminder.
output:
```python
def process_input():
    return False, "I need more information to create a reminder. Please specify: (1) What should I remind you about? (2) When should I remind you?"
```

input: **Creation Request**: Remind me about something later.
output:
```python
def process_input():
    return False, "I need more specific information to create a reminder. Please specify: (1) What exactly should I remind you about? (2) When specifically should I remind you?"
```

input: **Creation Request**: Create a reminder to cancel Netflix subscription at the end of this year.
**Input Info from previous skill**:
--- Last 10 Netflix Spending Transactions ---
None

--- Nextflix Subscription ---
None

output:
```python
def process_input():
    return False, "Cannot create reminder: Netflix subscription or transactions not found in your account data. I found Spotify and Hulu subscriptions, but no Netflix. Please verify the service name or check if you have an active Netflix subscription."
```

</EXAMPLES>

Today's date is |TODAY_DATE|."""


class CreateBudgetOrGoalOrReminder:
  """Handles all Gemini API interactions for creating budgets, goals, and reminders"""
  
  def __init__(self, model_name="gemini-flash-lite-latest"):
    """Initialize the Gemini agent with API configuration"""
    # API Configuration
    self.client = genai.Client(api_key=os.getenv('GEMINI_API_KEY'))
    
    # Model Configuration
    self.thinking_budget = 0
    self.model_name = model_name
    
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

  def _get_today_date_string(self) -> str:
    """
    Get today's date formatted as "YYYY-MM-DD"
    
    Returns:
      String containing today's date in the specified format
    """
    today = datetime.now()
    return today.strftime("%Y-%m-%d")

  def generate_response(self, creation_request: str, input_info: str = None) -> str:
    """
    Generate a response using Gemini API for creating budgets, goals, or reminders.
    
    Args:
      creation_request: What needs to be created factoring in the information from input_info
      input_info: Optional input from another skill function
      
    Returns:
      Generated code as a string
    """
    # Get today's date
    today_date = self._get_today_date_string()
    
    # Replace placeholders in system prompt
    full_system_prompt = self.system_prompt.replace("|TODAY_DATE|", today_date)
    
    # Create request text
    input_info_text = f"\n\n**Input Info from previous skill**:\n{input_info}" if input_info else ""
    request_text = types.Part.from_text(text=f"""**Creation Request**: {creation_request}{input_info_text}

output:""")
    
    # Create content and configuration
    contents = [types.Content(role="user", parts=[request_text])]
    
    generate_content_config = types.GenerateContentConfig(
      temperature=self.temperature,
      top_p=self.top_p,
      max_output_tokens=self.max_output_tokens,
      safety_settings=self.safety_settings,
      system_instruction=[types.Part.from_text(text=full_system_prompt)],
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


def _get_heavy_data_user_id() -> int:
  """
  Get the user ID for HeavyDataUser from the database.
  
  Returns:
    The user ID for HeavyDataUser, or 1 if not found
  """
  try:
    db = Database()
    heavy_user = db.get_user("HeavyDataUser")
    if heavy_user and 'id' in heavy_user:
      return heavy_user['id']
    else:
      print("Warning: HeavyDataUser not found, using default user_id=1")
      return 1
  except Exception as e:
    print(f"Warning: Error getting HeavyDataUser: {e}, using default user_id=1")
    return 1


def _run_test_with_logging(creation_request: str, input_info: str = None, generator: CreateBudgetOrGoalOrReminder = None, user_id: int = None):
  """
  Internal helper function that runs a test with consistent logging.
  
  Args:
    creation_request: The creation request as a string
    input_info: Optional input info from previous skill
    generator: Optional CreateBudgetOrGoalOrReminder instance. If None, creates a new one.
    user_id: User ID for sandbox execution (default: HeavyDataUser ID from database)
    
  Returns:
    The generated response string
  """
  if generator is None:
    generator = CreateBudgetOrGoalOrReminder()
  
  # Get HeavyDataUser ID if not provided
  if user_id is None:
    user_id = _get_heavy_data_user_id()
  
  # Construct LLM input
  input_info_text = f"\n\n**Input Info from previous skill**:\n{input_info}" if input_info else ""
  llm_input = f"""**Creation Request**: {creation_request}{input_info_text}

output:"""
  
  # Print the input
  print("=" * 80)
  print("LLM INPUT:")
  print("=" * 80)
  print(llm_input)
  print("=" * 80)
  print()
  
  result = generator.generate_response(creation_request, input_info)
  
  # Print the output
  print("=" * 80)
  print("LLM OUTPUT:")
  print("=" * 80)
  print(result)
  print("=" * 80)
  print()
  
  # Execute the generated code in sandbox
  print("=" * 80)
  print("SANDBOX EXECUTION:")
  print("=" * 80)
  try:
    success, output_string, logs, goals_list = sandbox.execute_agent_with_tools(result, user_id)
    
    print(f"Success: {success}")
    print()
    print("Output:")
    print("-" * 80)
    print(output_string)
    print("-" * 80)
    print()
  except Exception as e:
    print(f"**Sandbox Execution Error**: {str(e)}")
    import traceback
    print(traceback.format_exc())
  print("=" * 80)
  print()
  
  return result


def test_create_gas_budget(generator: CreateBudgetOrGoalOrReminder = None):
  """
  Test method for creating a gas budget scenario.
  
  Args:
    generator: Optional CreateBudgetOrGoalOrReminder instance. If None, creates a new one.
    
  Returns:
    The generated response string
  """
  creation_request = "budget $60 for gas every week for the next 6 months and a yearly car insurance cost of 3500 starting next year"
  
  return _run_test_with_logging(creation_request, None, generator)


def test_create_dining_out_budget(generator: CreateBudgetOrGoalOrReminder = None):
  """
  Test method for creating a dining out budget scenario.
  
  Args:
    generator: Optional CreateBudgetOrGoalOrReminder instance. If None, creates a new one.
    
  Returns:
    The generated response string
  """
  creation_request = "create a monthly budget of $500 for dining out"
  
  return _run_test_with_logging(creation_request, None, generator)


def test_create_netflix_reminder(generator: CreateBudgetOrGoalOrReminder = None):
  """
  Test method for creating a Netflix cancellation reminder scenario.
  
  Args:
    generator: Optional CreateBudgetOrGoalOrReminder instance. If None, creates a new one.
    
  Returns:
    The generated response string
  """
  creation_request = "remind me to cancel Netflix on August 31st"
  
  return _run_test_with_logging(creation_request, None, generator)


def test_create_account_balance_reminder(generator: CreateBudgetOrGoalOrReminder = None):
  """
  Test method for creating an account balance reminder scenario.
  
  Args:
    generator: Optional CreateBudgetOrGoalOrReminder instance. If None, creates a new one.
    
  Returns:
    The generated response string
  """
  creation_request = "remind me when my checking account balance drops below $1000"
  
  return _run_test_with_logging(creation_request, None, generator)


def test_create_savings_goal(generator: CreateBudgetOrGoalOrReminder = None):
  """
  Test method for creating a savings goal scenario.
  
  Args:
    generator: Optional CreateBudgetOrGoalOrReminder instance. If None, creates a new one.
    
  Returns:
    The generated response string
  """
  creation_request = "I want to save $5000 in the next 6 months"
  input_info = "Based on your current spending patterns, you can save approximately $800 per month."
  
  return _run_test_with_logging(creation_request, input_info, generator)


def test_create_ambiguous_budget(generator: CreateBudgetOrGoalOrReminder = None):
  """
  Test method for creating a budget with ambiguous category that requires clarification.
  
  Args:
    generator: Optional CreateBudgetOrGoalOrReminder instance. If None, creates a new one.
    
  Returns:
    The generated response string
  """
  creation_request = "create a budget for subscriptions"
  
  return _run_test_with_logging(creation_request, None, generator)


def test_create_incomplete_budget(generator: CreateBudgetOrGoalOrReminder = None):
  """
  Test method for creating a budget with missing information that requires clarification.
  
  Args:
    generator: Optional CreateBudgetOrGoalOrReminder instance. If None, creates a new one.
    
  Returns:
    The generated response string
  """
  creation_request = "create a budget for $200"
  
  return _run_test_with_logging(creation_request, None, generator)


def test_create_transaction_reminder(generator: CreateBudgetOrGoalOrReminder = None):
  """
  Test method for creating a transaction reminder scenario.
  
  Args:
    generator: Optional CreateBudgetOrGoalOrReminder instance. If None, creates a new one.
    
  Returns:
    The generated response string
  """
  creation_request = "remind me when my paycheck from work hits my account"
  
  return _run_test_with_logging(creation_request, None, generator)


def test_create_subscription_reminder(generator: CreateBudgetOrGoalOrReminder = None):
  """
  Test method for creating a subscription renewal reminder scenario.
  
  Args:
    generator: Optional CreateBudgetOrGoalOrReminder instance. If None, creates a new one.
    
  Returns:
    The generated response string
  """
  creation_request = "remind me 3 days before my gym membership renews"
  
  return _run_test_with_logging(creation_request, None, generator)


def test_create_refund_reminder(generator: CreateBudgetOrGoalOrReminder = None):
  """
  Test method for creating a refund reminder scenario.
  
  Args:
    generator: Optional CreateBudgetOrGoalOrReminder instance. If None, creates a new one.
    
  Returns:
    The generated response string
  """
  creation_request = "remind me when my Amazon refund comes in"
  
  return _run_test_with_logging(creation_request, None, generator)


def test_create_immediate_reminder(generator: CreateBudgetOrGoalOrReminder = None):
  """
  Test method for creating a reminder that triggers immediately (condition already met).
  
  This test checks a scenario where the reminder condition is already satisfied,
  so the reminder should return a trigger message right away.
  
  Args:
    generator: Optional CreateBudgetOrGoalOrReminder instance. If None, creates a new one.
    
  Returns:
    The generated response string
  """
  creation_request = "remind me when my checking account balance drops below $2000"
  
  return _run_test_with_logging(creation_request, None, generator)


def test_with_inputs(creation_request: str, input_info: str = None, generator: CreateBudgetOrGoalOrReminder = None):
  """
  Convenient method to test the generator with custom inputs.
  
  Args:
    creation_request: The creation request as a string
    input_info: Optional input info from previous skill
    generator: Optional CreateBudgetOrGoalOrReminder instance. If None, creates a new one.
    
  Returns:
    The generated response string
  """
  return _run_test_with_logging(creation_request, input_info, generator)


def main(batch: int = 1):
  """
  Main function to test the create budget/goal/reminder generator
  
  Args:
    batch: Batch number (1, 2, 3, or 4) to determine which tests to run
  """
  print("Testing CreateBudgetOrGoalOrReminder\n")
  
  if batch == 1:
    # Basic budget/goal creation tests
    print("Test 1: Creating gas and car insurance budget")
    print("-" * 80)
    test_create_gas_budget()
    print("\n")
    
    print("Test 2: Creating dining out budget")
    print("-" * 80)
    test_create_dining_out_budget()
    print("\n")
    
    print("Test 3: Creating savings goal with input info")
    print("-" * 80)
    test_create_savings_goal()
    print("\n")
  elif batch == 2:
    # Budget edge cases
    print("Test 1: Creating budget with ambiguous category (requires followup)")
    print("-" * 80)
    test_create_ambiguous_budget()
    print("\n")
    
    print("Test 2: Creating budget with missing information (requires followup)")
    print("-" * 80)
    test_create_incomplete_budget()
    print("\n")
    
    print("Test 3: Creating account balance reminder")
    print("-" * 80)
    test_create_account_balance_reminder()
    print("\n")
  elif batch == 3:
    # Reminder tests part 1
    print("Test 1: Creating Netflix cancellation reminder")
    print("-" * 80)
    test_create_netflix_reminder()
    print("\n")
    
    print("Test 2: Creating transaction reminder")
    print("-" * 80)
    test_create_transaction_reminder()
    print("\n")
    
    print("Test 3: Creating subscription renewal reminder")
    print("-" * 80)
    test_create_subscription_reminder()
    print("\n")
  elif batch == 4:
    # Reminder tests part 2
    print("Test 1: Creating refund reminder")
    print("-" * 80)
    test_create_refund_reminder()
    print("\n")
    
    print("Test 2: Creating reminder that triggers immediately")
    print("-" * 80)
    test_create_immediate_reminder()
    print("\n")
  else:
    raise ValueError("batch must be 1, 2, 3, or 4")
  
  print("All tests completed!")


if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser(description='Run tests in batches')
  parser.add_argument('--batch', type=int, default=1, choices=[1, 2, 3, 4],
                      help='Batch number to run (1, 2, 3, or 4)')
  args = parser.parse_args()
  main(batch=args.batch)

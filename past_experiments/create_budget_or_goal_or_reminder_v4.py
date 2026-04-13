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

SYSTEM_PROMPT = """You are a helpful AI assistant specialized in creating budgets, goals, and reminders. You only output python code.

## Your Tasks

1.  **Parse Input**: Extract information from **Creation Request** and **Input Info from previous skill**. Hardcode values from **Input Info** into your code (it is NOT available as a variable).
2.  **Budget/Goal Creation**:
    -   **Category Validation**: Use your understanding of **OFFICIAL CATEGORIES**. Do NOT write validation code (no lists, conditionals, or checks):
        -   **Exact Match or Clear Mapping**: If category exactly matches or clearly maps to a single exact category in **OFFICIAL CATEGORIES**, call `create_budget_or_goal` directly. Do NOT append any message.
        -   **Generalized Match**: Only if category maps to a generalized parent category, append explanation with `outputs.append("message")` and CREATE goal with generalized category.
        -   **Not in List / Ambiguous**: Append message with `outputs.append("message")` and DO NOT create goal.
    -   **Date Handling**: When a specific time period is mentioned, calculate and set both `start_date` and `end_date` accordingly. Only leave `end_date` empty ("") for ongoing budgets/goals without a specified end.
    -   **Limitations**: Only spending budgets and income goals can be created. Other goals (like saving or paying off) are NOT supported. Append message and DO NOT create if unsupported.
    -   Default amount to 0.0 if missing.
3.  **Reminder Creation**:
    -   Validate "What" and "When" are present.
    -   **Specific Entities**: If request mentions specific entities (e.g. "Netflix", "Chase **1234"), they MUST exist in **Input Info**. If missing, return `(False, message)`.
    -   **Vague Requests**: If vague (e.g. "remind me later"), return `(False, message)`.
    -   **Construct `what`**: Include ALL available details from **Input Info** to make reminder self-contained.
    -   **Handle Output**: If `create_reminder` returns `(True, None)`, use appropriate message as result string.
4.  **Output Format**:
    -   Function: `process_input() -> tuple[bool, str]`
    -   No comments, no imports. Assume `datetime`, `pandas` available.
    -   Use `IMPLEMENTED_DATE_FUNCTIONS` for datetime calculations.
    -   Append results to list, join with `chr(10).join(outputs)` before returning.
5.  **Date Ranges**: When referring to "past/next n months/weeks", **always exclude the current month/week**.
    -   **Past n months**: Start=`get_start_of_month(get_after_periods(today, 'monthly', -n))`, End=`get_end_of_month(get_after_periods(today, 'monthly', -1))`
    -   **Past n weeks**: Start=`get_start_of_week(get_after_periods(today, 'weekly', -n))`, End=`get_end_of_week(get_after_periods(today, 'weekly', -1))`
    -   **Next n months**: Start=`get_start_of_month(get_after_periods(today, 'monthly', 1))`, End=`get_end_of_month(get_after_periods(today, 'monthly', n))`
    -   **Next n weeks**: Start=`get_start_of_week(get_after_periods(today, 'weekly', 1))`, End=`get_end_of_week(get_after_periods(today, 'weekly', n))`

Today's date is |TODAY_DATE|.

<IMPLEMENTED_FUNCTIONS>

- `create_budget_or_goal(...) -> tuple[bool, str]`
  - Params: category (must be from OFFICIAL_CATEGORIES), granularity (weekly/monthly/yearly), start_date, end_date, amount, title.
  - Helper functions available: `get_start_of_week`, `get_end_of_week`, `get_start_of_month`, `get_end_of_month`, `get_start_of_year`, `get_end_of_year`, `get_after_periods`, `get_date_string`.

- `create_reminder(what: str, when: str) -> Tuple[bool, str]`
  - Returns: (success, message)

</IMPLEMENTED_FUNCTIONS>

<IMPLEMENTED_DATE_FUNCTIONS>

- `get_date(y, m, d)`, `get_start_of_month(date)`, `get_end_of_month(date)`
- `get_start_of_year(date)`, `get_end_of_year(date)`
- `get_start_of_week(date)`, `get_end_of_week(date)`
- `get_after_periods(date, granularity, count)`, `get_date_string(date)`

</IMPLEMENTED_DATE_FUNCTIONS>

<OFFICIAL_CATEGORIES>

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

</OFFICIAL_CATEGORIES>

<EXAMPLES>

input: **Creation Request**: Help budget $500 for eating out monthly and tell me to cancel Netflix at the end of the year.
**Input Info from previous skill**:
--- Spotify Subscription ---
Spotify: next_amount: $9.99
output:
```python
def process_input():
    today = datetime.now()
    outputs = []
    
    success1, result1 = create_budget_or_goal(
        category="meals_dining_out",
        granularity="monthly",
        start_date=get_date_string(get_start_of_month(today)),
        end_date="",
        amount=500.0,
        title="Monthly Dining Out 🍽️"
    )
    if not success1:
        return success1, result1
    outputs.append(result1)

    return False, "Missing info for Netflix. Please provide subscription details."
```

input: **Creation Request**: Budget $300 for meals for the next 3 months
output:
```python
def process_input():
    today = datetime.now()
    outputs = []
    
    outputs.append("Category 'meals' is generalized. Using 'meals' category.")
    success1, result1 = create_budget_or_goal(
        category="meals",
        granularity="monthly",
        start_date=get_date_string(get_start_of_month(get_after_periods(today, 'monthly', 1))),
        end_date=get_date_string(get_end_of_month(get_after_periods(today, 'monthly', 3))),
        amount=300.0,
        title="Monthly Meals"
    )
    if not success1:
        return success1, result1
    outputs.append(result1)
    
    return True, chr(10).join(outputs)
```

input: **Creation Request**: Create a budget for subscriptions
output:
```python
def process_input():
    outputs = []
    outputs.append("Category 'subscriptions' is too broad and matches multiple categories. Please specify a more specific category.")
    return False, chr(10).join(outputs)
```

input: **Creation Request**: Remind me to cancel Spotify on Dec 31st, and alert me if my balance drops below $1000.
**Input Info from previous skill**:
--- Spotify Subscription ---
Spotify: next_amount: $9.99, next payment: 2025-12-01, last transaction date: 2025-11-01
--- Account Balances ---
Chase Total Checking **1563: Current: $4567
Chase Checking **3052: Current: $1202
output:
```python
def process_input():
    outputs = []
    
    success1, result1 = create_reminder(
        what="cancel Spotify subscription (last transaction: 2025-11-01, next payment: 2025-12-01, amount: $9.99)",
        when="on Dec 31st"
    )
    if not success1:
        return success1, result1
    if result1 is None:
        result1 = "Reminder created (condition not currently met)."
    outputs.append(result1)

    success2, result2 = create_reminder(
        what="checking account balance drops below $1000 (accounts: Chase Total Checking **1563 current: $4567, Chase Checking **3052 current: $1202)",
        when="immediately when condition is met"
    )
    if not success2:
        return success2, result2
    if result2 is None:
        result2 = "Reminder created (condition not currently met)."
    outputs.append(result2)
    
    return True, chr(10).join(outputs)
```

</EXAMPLES>"""


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

def test_create_missing_entity_reminder(generator: CreateBudgetOrGoalOrReminder = None):
  """
  Test method for creating a reminder for a missing entity (Hulu).
  """
  creation_request = "remind me to cancel Hulu"
  input_info = "--- Spotify Subscription ---\nSpotify: next_amount: $9.99\n"
  
  return _run_test_with_logging(creation_request, input_info, generator)

def test_create_vague_reminder(generator: CreateBudgetOrGoalOrReminder = None):
  """
  Test method for creating a vague reminder.
  """
  creation_request = "remind me later"
  
  return _run_test_with_logging(creation_request, None, generator)

def test_create_detailed_context_reminder(generator: CreateBudgetOrGoalOrReminder = None):
  """
  Test method for creating a reminder with detailed context integration.
  """
  creation_request = "remind me if checking drops below 500"
  input_info = "--- Account Balances ---\nChase Checking **1234: Current: $1200, Available: $1200\nWells Fargo **5678: Current: $300, Available: $300"
  
  return _run_test_with_logging(creation_request, input_info, generator)

def test_create_multiple_mixed(generator: CreateBudgetOrGoalOrReminder = None):
  """
  Test method for creating a mixed request (budget + reminder).
  """
  creation_request = "Budget $50 for coffee monthly and remind me to pay bill on 5th"
  input_info = "--- Bill Info ---\nElectric Bill: $120 due on 5th"
  
  return _run_test_with_logging(creation_request, input_info, generator)


def test_create_mixed_fail_reminder(generator: CreateBudgetOrGoalOrReminder = None):
  """
  Test method for creating a mixed request where the reminder fails due to missing info.
  """
  creation_request = "Help budget $500 for eating out monthly and tell me to cancel Netflix at the end of the year."
  input_info = "--- Spotify Subscription ---\nSpotify: next_amount: $9.99"
  
  return _run_test_with_logging(creation_request, input_info, generator)


def test_create_generalized_category_budget(generator: CreateBudgetOrGoalOrReminder = None):
  """
  Test method for creating a budget with generalized category (meals instead of meals_dining_out).
  
  Args:
    generator: Optional CreateBudgetOrGoalOrReminder instance. If None, creates a new one.
    
  Returns:
    The generated response string
  """
  creation_request = "Budget $300 for meals monthly"
  
  return _run_test_with_logging(creation_request, None, generator)


def test_create_income_goal_salary(generator: CreateBudgetOrGoalOrReminder = None):
  """
  Test method for creating an income goal for salary.
  
  Args:
    generator: Optional CreateBudgetOrGoalOrReminder instance. If None, creates a new one.
    
  Returns:
    The generated response string
  """
  creation_request = "Set a goal to earn $5000 in salary this month"
  
  return _run_test_with_logging(creation_request, None, generator)


def test_create_income_goal_sidegig(generator: CreateBudgetOrGoalOrReminder = None):
  """
  Test method for creating an income goal for side gig.
  
  Args:
    generator: Optional CreateBudgetOrGoalOrReminder instance. If None, creates a new one.
    
  Returns:
    The generated response string
  """
  creation_request = "I want to make $6000 from my side gig this year"
  
  return _run_test_with_logging(creation_request, None, generator)


def test_create_category_not_in_list(generator: CreateBudgetOrGoalOrReminder = None):
  """
  Test method for creating a budget with category that doesn't exist.
  
  Args:
    generator: Optional CreateBudgetOrGoalOrReminder instance. If None, creates a new one.
    
  Returns:
    The generated response string
  """
  creation_request = "Create a budget for 'xyz_category' monthly"
  
  return _run_test_with_logging(creation_request, None, generator)


def test_create_paying_off_goal(generator: CreateBudgetOrGoalOrReminder = None):
  """
  Test method for creating an unsupported paying off goal.
  
  Args:
    generator: Optional CreateBudgetOrGoalOrReminder instance. If None, creates a new one.
    
  Returns:
    The generated response string
  """
  creation_request = "I want to pay off $5000 in credit card debt"
  
  return _run_test_with_logging(creation_request, None, generator)


def test_create_leisure_budget(generator: CreateBudgetOrGoalOrReminder = None):
  """
  Test method for creating a leisure category budget.
  
  Args:
    generator: Optional CreateBudgetOrGoalOrReminder instance. If None, creates a new one.
    
  Returns:
    The generated response string
  """
  creation_request = "Budget $200 for entertainment monthly"
  
  return _run_test_with_logging(creation_request, None, generator)


def test_create_health_budget(generator: CreateBudgetOrGoalOrReminder = None):
  """
  Test method for creating a health category budget.
  
  Args:
    generator: Optional CreateBudgetOrGoalOrReminder instance. If None, creates a new one.
    
  Returns:
    The generated response string
  """
  creation_request = "Budget $100 for gym membership monthly"
  
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
    batch: Batch number (1, 2, 3, 4, 5, or 6) to determine which tests to run
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
  elif batch == 5:
    # Missing rules tests
    print("Test 1: Missing entity reminder")
    print("-" * 80)
    test_create_missing_entity_reminder()
    print("\n")
    
    print("Test 2: Vague reminder")
    print("-" * 80)
    test_create_vague_reminder()
    print("\n")
    
    print("Test 3: Detailed context reminder")
    print("-" * 80)
    test_create_detailed_context_reminder()
    print("\n")
    
    print("Test 4: Mixed budget and reminder")
    print("-" * 80)
    test_create_multiple_mixed()
    print("\n")
    
    print("Test 5: Mixed budget and fail reminder")
    print("-" * 80)
    test_create_mixed_fail_reminder()
    print("\n")
  elif batch == 6:
    # Missing test cases for budget/goal creation
    print("Test 1: Generalized category budget (meals)")
    print("-" * 80)
    test_create_generalized_category_budget()
    print("\n")
    
    print("Test 2: Income goal - salary")
    print("-" * 80)
    test_create_income_goal_salary()
    print("\n")
    
    print("Test 3: Income goal - side gig")
    print("-" * 80)
    test_create_income_goal_sidegig()
    print("\n")
    
    print("Test 4: Category not in list")
    print("-" * 80)
    test_create_category_not_in_list()
    print("\n")
    
    print("Test 5: Unsupported paying off goal")
    print("-" * 80)
    test_create_paying_off_goal()
    print("\n")
    
    print("Test 6: Leisure category budget")
    print("-" * 80)
    test_create_leisure_budget()
    print("\n")
    
    print("Test 7: Health category budget")
    print("-" * 80)
    test_create_health_budget()
    print("\n")
  else:
    raise ValueError("batch must be 1, 2, 3, 4, 5, or 6")
  
  print("All tests completed!")


if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser(description='Run tests in batches')
  parser.add_argument('--batch', type=int, default=1, choices=[1, 2, 3, 4, 5, 6],
                      help='Batch number to run (1, 2, 3, 4, 5, or 6)')
  args = parser.parse_args()
  main(batch=args.batch)

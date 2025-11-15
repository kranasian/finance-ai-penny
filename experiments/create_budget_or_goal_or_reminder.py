from google import genai
from google.genai import types
import sys
import os
from dotenv import load_dotenv
import datetime as dt_module
from datetime import datetime, timedelta, date
from typing import Tuple
import pandas as pd


# Add the parent directory to the path so we can import the tools
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
  sys.path.insert(0, parent_dir)

from penny.tool_funcs.create_budget_or_goal import create_budget_or_goal, VALID_GRANULARITIES, VALID_GOAL_TYPES
from penny.tool_funcs.date_utils import get_start_of_week, get_end_of_week, get_start_of_month, get_end_of_month, get_start_of_year, get_end_of_year, get_after_periods, get_date_string
from database import Database

# Load environment variables
load_dotenv()

SYSTEM_PROMPT = """You are a helpful AI assistant specialized in creating budgets, goals, and reminders based on user requests. **You only output python code.**

## Your Task and Main Rules

1. Write a function `process_input` that takes no arguments and returns a tuple:
   - The first element is a boolean indicating success or failure.
   - The second element is a string containing the output information (what was created or error message).
2. Assume `import datetime` and `import pandas as pd` are already included. Do not include import statements in your code.
3. Determine whether the user wants to create a **budget/goal** or a **reminder** based on the request.
4. For **budget/goal** requests:
   - Create a list of goal dictionaries based on the user's request.
   - Each dictionary should follow the structure described in `create_budget_or_goal` function documentation.
   - Call `create_budget_or_goal(goal_dicts)` with the list of dictionaries.
5. For **reminder** requests:
   - Generate validation code that checks if the reminder request has 2 required pieces of information:
     1. **What** to be reminded about: transaction coming in, subscription getting refunded, account balances, or a clear general task
     2. **When** the reminder will be relevant: date, condition, or frequency
   - Implement the validation logic inline in `process_input()`.
   - If validation fails (missing information), return `(False, clarification_message)` with a clear message asking for the missing information.
   - If validation passes, also generate code to check if the reminder condition is already met and should trigger immediately (should_remind logic).
   - The should_remind code should:
     - Check if the triggering condition has been met
     - Return a message if the condition is met, or None if not
     - Determine when to check next (next_check_date) if the condition might be met in the future
   - Save/store the should_remind code logic (as a string or in metadata) so it can be used later for periodic checking.
   - After generating and executing should_remind code:
     - The should_remind code should be saved/stored (as a string or in metadata) for later periodic checking.
     - If should_remind returns a trigger message (condition is met), include the trigger message in the response.
     - Include the next_check_date in the response if provided.
     - Build the response message describing what reminder was created. The message should be clear and descriptive, following the pattern: `Reminder "<reminder_description>" was successfully created.` where `<reminder_description>` is a natural description of what the reminder is for (e.g., "to cancel Netflix on August 31st", "when checking account balance drops below $1000"). You may also include additional context like when the last transaction/subscription occurred if relevant.
     - Return `(True, response_message)` with the reminder creation confirmation, trigger message (if any), next_check_date (if any), and the saved should_remind code.
   - The validation should check the dataframes to ensure the reminder can be created (e.g., subscription exists, account exists, etc.).
   - The should_remind check should determine if the reminder condition is already satisfied (e.g., account balance already below threshold, subscription already cancelled, transaction already occurred).

<IMPLEMENTED_FUNCTIONS>

These functions are already implemented:

- `create_budget_or_goal(goal_dicts: list[dict]) -> tuple[bool, str]`
  - Creates a spending goal or budget based on the list of goal dictionaries provided.
  - `goal_dicts`: A list of dictionaries, where each dictionary represents a single spending goal/budget with the following keys:
    - `category`: (string) The raw spending category text extracted from user input (e.g., "gas", "eating out"). Can be empty string if not provided.
    - `match_category`: (string) The category from the OFFICIAL CATEGORY LIST that best matches the user's goal category. Can be empty string if not provided or if clarification is needed.
    - `match_caveats`: (string or None) Explanation of matching constraints and any generalization made. Only provide if the input is a more specific item that falls under a broader category, or if there's any ambiguity in the match. Otherwise use None.
    - `clarification_needed`: (string or None) If the goal category is ambiguous, incomplete, or could belong to multiple categories, provide a string prompting the user for clarification. Otherwise use None.
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
  - Use helper functions: `get_start_of_week(date: datetime) -> datetime`, `get_end_of_week(date: datetime) -> datetime`, `get_start_of_month(date: datetime) -> datetime`, `get_end_of_month(date: datetime) -> datetime`, `get_start_of_year(date: datetime) -> datetime`, `get_end_of_year(date: datetime) -> datetime`, `get_after_periods(date: datetime, granularity: str, count: int) -> datetime`, `get_date_string(date: datetime) -> str`.

- `get_accounts_df() -> pd.DataFrame`
  - Retrieves Accounts dataframe with columns:
    - `account_id`: unique numeric identifier
    - `name`: Account name containing bank name, account name and last 4 digits
    - `account_type`: One of: deposit_savings, deposit_money_market, deposit_checking, credit_card, loan_home_equity, loan_line_of_credit, loan_mortgage, loan_auto
    - `balance_available`: amount withdrawable for deposit accounts
    - `balance_current`: loaned amount in loans/credit accounts, usable amount in savings/checking
    - `balance_limit`: credit limit of credit-type accounts
  - Use this function to retrieve account data for validation and should_remind checks in reminder requests.

- `get_transactions_df() -> pd.DataFrame`
  - Retrieves Transactions dataframe with columns:
    - `account_id`: account where this transaction is
    - `transaction_id`: unique numeric identifier
    - `datetime`: inflow or spending date
    - `name`: establishment or service name
    - `amount`: inflow amount will be negative, outflow/spending will be positive
    - `category`: transaction category from Official Category List
    - `output_category`: category used for display
  - Use this function to retrieve transaction data for validation and should_remind checks in reminder requests.

- `get_subscriptions_df() -> pd.DataFrame`
  - Retrieves Subscriptions dataframe with columns:
    - `name`: establishment or service name with the subscription
    - `next_amount`: likely upcoming amount
    - `next_likely_payment_date`: most likely upcoming payment date
    - `next_earliest_payment_date`: earliest possible payment date
    - `next_latest_payment_date`: latest possible payment date
    - `user_cancelled_date`: date when user cancelled subscription
    - `last_transaction_date`: date of the most recent transaction for this subscription
  - Use this function to retrieve subscription data for validation and should_remind checks in reminder requests.

  - **IMPORTANT**: For reminder requests, you should generate validation code inline in `process_input()`. The validation should check:
    - That the reminder has both "what" and "when" information
    - That the relevant data exists (subscription, account, transaction) in the dataframes
    - Return `(False, clarification_message)` if validation fails
    - Only proceed if validation passes
  - **IMPORTANT**: After validation passes, also generate should_remind code that:
    - Checks if the reminder condition is already met and should trigger immediately
    - Returns a tuple `(message, next_check_date)` where:
      - `message` is the message to send if triggering condition is met, None otherwise
      - `next_check_date` is the date to check again if condition might be met in future, None if no more checks needed
    - The should_remind code should be saved/stored (as a string or in metadata) for later periodic checking
    - If the condition is met, include the trigger message in the response
    - Examples of should_remind logic:
      - If checking account balance threshold: check if balance is already below the threshold
      - If checking for a transaction: check if the transaction has already occurred
      - If checking subscription renewal date: check if the date has already passed
      - For date-based reminders: use `datetime.now().date()` to compare with target dates
      - For recurring reminders: calculate next_check_date based on frequency (daily, weekly, monthly)
    - Return `(True, response_message)` with reminder creation confirmation, trigger message (if any), next_check_date (if any), and saved should_remind code

</IMPLEMENTED_FUNCTIONS>

<OFFICIAL_CATEGORY_LIST>

* `meals` for all types of food spending. *Includes:*
  * `meals_groceries` for supermarkets and other unprepared food marketplaces.
  * `meals_dining_out` for food prepared outside the home like restaurants and takeout.
  * `meals_delivered_food` for prepared food delivered to the doorstep like DoorDash.
* `leisure` for relaxation, recreation and travel activities. *Includes:*
  * `leisure_entertainment` for movies, concerts, cable and streaming services.
  * `leisure_travel` for flights, hotels, and other travel expenses.
* `bills` for essential payments for services and recurring costs. *Includes:*
  * `bills_connectivity` for internet and phone bills.
  * `bills_insurance` for life insurance and other insurance payments.
  * `bills_tax` for income, state tax and other payments.
  * `bills_service_fees` for payments for services rendered like professional fees or fees for a product.
* `shelter` for all housing-related expenses including rent, mortgage, property taxes and utilities. *Includes:*
  * `shelter_home` for rent, mortgage, property taxes.
  * `shelter_utilities` for electricity, water, gas and trash utility bills.
  * `shelter_upkeep` for maintenance and repair and improvement costs for the home.
* `education` for all learning spending including kids after care and activities. *Includes:*
  * `education_kids_activities` for after school activities, sports and camps.
  * `education_tuition` for school tuition, daycare and other education fees.
* `shopping` for discretionary spending on clothes, electronics, home goods, etc. *Includes:*
  * `shopping_clothing` for clothing, shoes, accessories and other wearable items.
  * `shopping_gadgets` for electronics, gadgets, computers and other tech items.
  * `shopping_kids` for kids clothing, toys, school supplies and other kid-related items.
  * `shopping_pets` for pet food, toys, grooming, vet bills and other pet-related items.
* `transportation` for public transportation, car payments, gas and maintenance and car insurance. *Includes:*
  * `transportation_public` for bus, train, subway and other public transportation.
  * `transportation_car` for car payments, gas, maintenance and car insurance.
* `health` for medical bills, pharmacy spending, insurance, gym memberships and personal care. *Includes:*
  * `health_medical_pharmacy` for doctor visits, hospital, meds and health insurance costs.
  * `health_gym_wellness` for gym memberships, personal training and spa services.
  * `health_personal_care` for haircuts, beauty products and beauty services.
* `donations_gifts` for charitable donations, gifts and other giving to friends and family.
* `income` for salary, bonuses, interest, side hussles and business. *Includes:*
  * `income_salary` for regular paychecks and bonuses.
  * `income_sidegig` for side hussles like Uber, Etsy and other gigs.
  * `income_business` for business income and spending.
  * `income_interest` for interest income from savings or investments.
* `uncategorized` for explicitly tagged as not yet categorized or unknown.
* `transfers` for moving money between accounts or paying off credit cards or loans.
* `miscellaneous` for explicitly tagged as miscellaneous.

</OFFICIAL_CATEGORY_LIST>

<EXAMPLES>

input: **Last User Request**: budget $60 for gas every week for the next 6 months and a yearly car insurance cost of 3500 starting next year
output:
```python
def process_input():
    today = datetime.now()
    goal_dicts = []
    
    goal_dicts.append({{
        "category": "gas",
        "match_category": "transportation_car",
        "match_caveats": "Matching gas to overall car expenses.",
        "clarification_needed": None,
        "type": "category",
        "granularity": "weekly",
        "start_date": get_date_string(get_start_of_week(today)),
        "end_date": get_date_string(get_end_of_week(get_after_periods(today, granularity="monthly", count=6))),
        "amount": 60.0,
        "title": "Weekly Gas ⛽",
        "budget_or_goal": "budget",
    }})
    
    goal_dicts.append({{
        "category": "car insurance",
        "match_category": "transportation_car",
        "match_caveats": "Matching car insurance to overall car expenses.",
        "clarification_needed": None,
        "type": "category",
        "granularity": "yearly",
        "start_date": get_date_string(get_start_of_year(get_after_periods(today, granularity="yearly", count=1))),
        "end_date": "",
        "amount": 3500.0,
        "title": "🚗 Insurance Year Limit",
        "budget_or_goal": "budget",
    }})
    
    return create_budget_or_goal(goal_dicts)
```

input: **Last User Request**: remind me to cancel Netflix on August 31st
output:
```python
def process_input():
    # Validate reminder request
    accounts_df = get_accounts_df()
    transactions_df = get_transactions_df()
    subscriptions_df = get_subscriptions_df()
    
    # Check if we have subscription data to validate Netflix subscription
    if subscriptions_df.empty and transactions_df.empty:
        return False, "Unable to create reminder: No subscription or transaction data available. Please ensure your accounts are connected."
    
    # Check if Netflix subscription exists
    netflix_subscriptions = subscriptions_df[subscriptions_df['name'].str.lower().str.contains('netflix', na=False)]
    netflix_transactions = transactions_df[transactions_df['name'].str.lower().str.contains('netflix', na=False)]
    
    if netflix_subscriptions.empty and netflix_transactions.empty:
        return False, "No Netflix subscription or transactions found. Please check if you have an active Netflix subscription."
    
    # Generate should_remind code
    should_remind_code = '''
def should_remind(get_accounts_df, get_transactions_df, get_subscriptions_df):
    # Check if reminder should trigger now and when to check next
    today = datetime.now().date()
    target_date = datetime(2024, 8, 31).date()
    
    subscriptions_df = get_subscriptions_df()
    netflix_subscriptions = subscriptions_df[subscriptions_df['name'].str.lower().str.contains('netflix', na=False)]
    
    if today >= target_date:
        active_subscriptions = netflix_subscriptions[
            (netflix_subscriptions['user_cancelled_date'].isna()) |
            (pd.to_datetime(netflix_subscriptions['user_cancelled_date']).dt.date > today)
        ]
        if not active_subscriptions.empty:
            return f"📅 REMINDER: Today is {{{{target_date.strftime('%B %d')}}}}! Don't forget to cancel your Netflix subscription.", None
        else:
            return None, None  # Already cancelled, no more checks needed
    else:
        # Not yet time, check again on target date
        return None, target_date
'''
    
    # Check if reminder should trigger now
    exec(should_remind_code)
    trigger_message, next_check_date = should_remind(get_accounts_df, get_transactions_df, get_subscriptions_df)
    
    # Save should_remind_code for later use (in a real system, this would be persisted to database)
    # Build response message with reminder description
    reminder_description = "cancel Netflix on August 31st"
    result_message = f"Reminder \\"{{{{reminder_description}}}}\\" was successfully created."
    if trigger_message:
        result_message += f"\\n\\n{{trigger_message}}"
    if next_check_date:
        result_message += f"\\nNext check scheduled for: {{next_check_date}}"
    result_message += f"\\n\\nShould remind code saved: {{should_remind_code}}"
    return True, result_message
```

input: **Last User Request**: create a monthly budget of $500 for dining out
output:
```python
def process_input():
    today = datetime.now()
    goal_dicts = []
    
    goal_dicts.append({{
        "category": "dining out",
        "match_category": "meals_dining_out",
        "match_caveats": None,
        "clarification_needed": None,
        "type": "category",
        "granularity": "monthly",
        "start_date": get_date_string(get_start_of_month(today)),
        "end_date": "",
        "amount": 500.0,
        "title": "Monthly Dining Out Budget",
        "budget_or_goal": "budget",
    }})
    
    return create_budget_or_goal(goal_dicts)
```

input: **Last User Request**: remind me when my checking account balance drops below $1000
output:
```python
def process_input():
    # Validate reminder request
    accounts_df = get_accounts_df()
    
    # Check if we have account data
    if accounts_df.empty:
        return False, "Unable to create reminder: No account data available. Please ensure your accounts are connected."
    
    # Check if checking account exists
    checking_accounts = accounts_df[
        (accounts_df['account_type'].str.contains('checking', case=False, na=False)) |
        (accounts_df['name'].str.lower().str.contains('checking', na=False))
    ]
    
    if checking_accounts.empty:
        return False, "No checking account found. Please specify which account you'd like to monitor, or ensure your checking account is connected."
    
    # Generate should_remind code
    should_remind_code = '''
def should_remind(get_accounts_df, get_transactions_df, get_subscriptions_df):
    # Check if reminder should trigger now and when to check next
    threshold = 1000.0
    accounts_df = get_accounts_df()
    checking_accounts = accounts_df[
        (accounts_df['account_type'].str.contains('checking', case=False, na=False)) |
        (accounts_df['name'].str.lower().str.contains('checking', na=False))
    ]
    
    accounts_below_threshold = checking_accounts[
        (checking_accounts['balance_current'].notna()) &
        (checking_accounts['balance_current'] < threshold)
    ]
    
    if not accounts_below_threshold.empty:
        account_names = accounts_below_threshold['name'].tolist()
        balances = accounts_below_threshold['balance_current'].tolist()
        message = f"⚠️ ALERT: Your checking account balance is already below ${{{{threshold:.2f}}}}:\\n"
        for name, balance in zip(account_names, balances):
            message += f"  - {{{{name}}}}: ${{{{balance:.2f}}}}\\n"
        # Check daily for balance changes
        next_check = datetime.now().date() + timedelta(days=1)
        return message, next_check
    else:
        # Check daily until balance drops below threshold
        next_check = datetime.now().date() + timedelta(days=1)
        return None, next_check
'''
    
    # Check if reminder should trigger now
    namespace = {{'datetime': datetime, 'timedelta': timedelta, 'pd': pd, 'get_accounts_df': get_accounts_df, 'get_transactions_df': get_transactions_df, 'get_subscriptions_df': get_subscriptions_df}}
    exec(should_remind_code, namespace)
    should_remind_func = namespace['should_remind']
    trigger_message, next_check_date = should_remind_func(get_accounts_df, get_transactions_df, get_subscriptions_df)
    
    # Save should_remind_code for later use (in a real system, this would be persisted to database)
    # Build response message with reminder description
    reminder_description = "notify when checking account balance drops below $1000"
    result_message = f"Reminder \\"{{{{reminder_description}}}}\\" was successfully created."
    if trigger_message:
        result_message += f"\\n\\n{{trigger_message}}"
    if next_check_date:
        result_message += f"\\nNext check scheduled for: {{next_check_date}}"
    result_message += f"\\n\\nShould remind code saved: {{should_remind_code}}"
    return True, result_message
```

input: **Last User Request**: I want to save $5000 for a car by December 31st, 2025
output:
```python
def process_input():
    today = datetime.now()
    goal_dicts = []
    
    goal_dicts.append({{
        "category": "",
        "match_category": "",
        "match_caveats": None,
        "clarification_needed": None,
        "type": "save_0",
        "granularity": "monthly",
        "start_date": get_date_string(get_start_of_month(today)),
        "end_date": "2025-12-31",
        "amount": 5000.0,
        "title": "Save for Car 🚗",
        "budget_or_goal": "goal",
    }})
    
    return create_budget_or_goal(goal_dicts)
```

</EXAMPLES>

Today's date is {today_date}.
"""


class CreateBudgetOrGoalOrReminder:
  """Handles all Gemini API interactions for creating budgets, goals, and reminders"""
  
  def __init__(self, model_name="gemini-2.0-flash"):
    """Initialize the Gemini agent with API configuration"""
    # API Configuration
    self.client = genai.Client(api_key=os.getenv('GEMINI_API_KEY'))
    
    # Model Configuration
    if "-thinking" in model_name:
      self.thinking_budget = 2048
      self.model_name = model_name.replace("-thinking", "")
    else:
      self.thinking_budget = 0
      self.model_name = model_name
    
    # Generation Configuration Constants
    self.temperature = 0.2
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
    today_date = datetime.now().strftime("%Y-%m-%d")
    self.system_prompt = SYSTEM_PROMPT.format(today_date=today_date)

  
  def generate_response(self, creation_request: str, input_info: str = None) -> str:
    """
    Generate a response using Gemini API for creating budgets, goals, or reminders.
    
    Args:
      creation_request: What needs to be created factoring in the information from input_info
      input_info: Optional input from another skill function
      
    Returns:
      Generated code as a string
    """
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




class _DateTimeNamespace:
  """Namespace that supports both datetime.datetime() and datetime.now() patterns."""
  def __init__(self, module, dt_class):
    self._module = module
    self._dt_class = dt_class
  
  def __getattr__(self, name):
    # Delegate to module for all attributes (datetime.datetime, datetime.date, etc.)
    return getattr(self._module, name)
  
  # Expose datetime class methods directly
  def now(self, tz=None):
    return self._dt_class.now(tz)
  
  def today(self):
    return self._dt_class.today()
  
  def utcnow(self):
    return self._dt_class.utcnow()
  
  def fromtimestamp(self, timestamp, tz=None):
    return self._dt_class.fromtimestamp(timestamp, tz)
  
  def fromordinal(self, ordinal):
    return self._dt_class.fromordinal(ordinal)
  
  def combine(self, date, time):
    return self._dt_class.combine(date, time)
  
  def strptime(self, date_string, format):
    return self._dt_class.strptime(date_string, format)
  
  def __call__(self, *args, **kwargs):
    # Allow datetime(...) to work as datetime.datetime(...)
    return self._dt_class(*args, **kwargs)


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


def create_budget_or_goal_or_reminder(
    creation_request: str, 
    input_info: str = None
) -> Tuple[bool, str]:
    """
    Create a budget, goal, or reminder based on the creation request.
    
    Args:
        creation_request: What needs to be created factoring in the information from input_info
        input_info: Optional input from another skill function
        
    Returns:
        Tuple[bool, str]: (success, output_info)
    """
    generator = CreateBudgetOrGoalOrReminder()
    code = generator.generate_response(creation_request, input_info)
    
    # Extract Python code from markdown code blocks if present
    code = extract_python_code(code)
    
    # Execute the generated code
    try:
      # Import pandas if available
      try:
        import pandas as pd
      except ImportError:
        pd = None
      
      # Create a namespace that supports both datetime.datetime() and datetime.now()
      datetime_ns = _DateTimeNamespace(dt_module, datetime)
      
      # Create functions for dataframe retrieval from test database
      def get_accounts_df():
        """Retrieve accounts from test database and format according to expected structure."""
        if pd is None:
          return None
        try:
          db = Database()
          user_id = 1  # Default test user
          accounts = db.get_accounts_by_user(user_id=user_id)
          if not accounts:
            return pd.DataFrame(columns=['account_id', 'name', 'account_type', 'balance_available', 'balance_current', 'balance_limit'])
          
          df = pd.DataFrame(accounts)
          # Combine account_name and account_mask into 'name' column
          if 'account_name' in df.columns and 'account_mask' in df.columns:
            df['name'] = df['account_name'] + ' *' + df['account_mask']
          elif 'account_name' in df.columns:
            df['name'] = df['account_name']
          else:
            df['name'] = ''
          
          # Select and reorder columns to match expected format
          expected_cols = ['account_id', 'name', 'account_type', 'balance_available', 'balance_current', 'balance_limit']
          available_cols = [col for col in expected_cols if col in df.columns]
          return df[available_cols]
        except Exception:
          return pd.DataFrame(columns=['account_id', 'name', 'account_type', 'balance_available', 'balance_current', 'balance_limit'])
      
      def get_transactions_df():
        """Retrieve transactions from test database and format according to expected structure."""
        if pd is None:
          return None
        try:
          db = Database()
          user_id = 1  # Default test user
          transactions = db.get_transactions_by_user(user_id=user_id)
          if not transactions:
            return pd.DataFrame(columns=['account_id', 'transaction_id', 'datetime', 'name', 'amount', 'category', 'output_category'])
          
          df = pd.DataFrame(transactions)
          # Rename 'date' to 'datetime' and 'transaction_name' to 'name'
          if 'date' in df.columns:
            df['datetime'] = pd.to_datetime(df['date'])
          if 'transaction_name' in df.columns:
            df['name'] = df['transaction_name']
          
          # Add output_category column (format category for display)
          if 'category' in df.columns:
            def format_category(cat):
              if not cat:
                return 'Unknown'
              # Replace underscores with spaces and title case
              formatted = cat.replace('_', ' ').title()
              # Remove common prefixes
              for prefix in ['meals ', 'income ', 'bills ', 'leisure ', 'shelter ']:
                if formatted.lower().startswith(prefix):
                  formatted = formatted[len(prefix):]
                  break
              return formatted
            df['output_category'] = df['category'].apply(format_category)
          
          # Select and reorder columns to match expected format
          expected_cols = ['account_id', 'transaction_id', 'datetime', 'name', 'amount', 'category', 'output_category']
          available_cols = [col for col in expected_cols if col in df.columns]
          return df[available_cols]
        except Exception:
          return pd.DataFrame(columns=['account_id', 'transaction_id', 'datetime', 'name', 'amount', 'category', 'output_category'])
      
      def get_subscriptions_df():
        """Retrieve subscriptions from test database and format according to expected structure."""
        if pd is None:
          return None
        try:
          db = Database()
          user_id = 1  # Default test user
          subscriptions = db.get_subscriptions(user_id=user_id)
          if not subscriptions:
            return pd.DataFrame(columns=['name', 'next_amount', 'next_likely_payment_date', 'next_earliest_payment_date', 'next_latest_payment_date', 'user_cancelled_date', 'last_transaction_date'])
          
          df = pd.DataFrame(subscriptions)
          
          # Convert date columns to datetime
          date_cols = ['next_likely_payment_date', 'next_earliest_payment_date', 'next_latest_payment_date', 'user_cancelled_date', 'last_transaction_date']
          for col in date_cols:
            if col in df.columns:
              df[col] = pd.to_datetime(df[col], errors='coerce')
          
          # Select and reorder columns to match expected format
          expected_cols = ['name', 'next_amount', 'next_likely_payment_date', 'next_earliest_payment_date', 'next_latest_payment_date', 'user_cancelled_date', 'last_transaction_date']
          available_cols = [col for col in expected_cols if col in df.columns]
          return df[available_cols]
        except Exception:
          return pd.DataFrame(columns=['name', 'next_amount', 'next_likely_payment_date', 'next_earliest_payment_date', 'next_latest_payment_date', 'user_cancelled_date', 'last_transaction_date'])
      
      # Create a namespace for execution with available functions
      namespace = {
        'datetime': datetime_ns,  # Supports both datetime.datetime() and datetime.now()
        'timedelta': timedelta,
        'date': date,
        'pd': pd,
        'create_budget_or_goal': create_budget_or_goal,
        'get_accounts_df': get_accounts_df,
        'get_transactions_df': get_transactions_df,
        'get_subscriptions_df': get_subscriptions_df,
        'list': list,
        'dict': dict,
        # Date helper functions
        'get_start_of_week': get_start_of_week,
        'get_end_of_week': get_end_of_week,
        'get_start_of_month': get_start_of_month,
        'get_end_of_month': get_end_of_month,
        'get_start_of_year': get_start_of_year,
        'get_end_of_year': get_end_of_year,
        'get_after_periods': get_after_periods,
        'get_date_string': get_date_string,
      }
      
      # Execute the code
      exec(code, namespace)
      
      # Call the process_input function
      if 'process_input' in namespace:
        success, output = namespace['process_input']()
        return success, output
      else:
        return False, "Generated code does not contain process_input function"
    except Exception as e:
      return False, f"Error executing generated code: {str(e)}"


def _run_test_with_logging(creation_request: str, input_info: str = None, generator: CreateBudgetOrGoalOrReminder = None):
  """
  Internal helper function that runs a test with consistent logging.
  
  Args:
    creation_request: The creation request as a string
    input_info: Optional input info from previous skill
    generator: Optional CreateBudgetOrGoalOrReminder instance. If None, creates a new one.
    
  Returns:
    The generated response string
  """
  if generator is None:
    generator = CreateBudgetOrGoalOrReminder()
  
  # Construct LLM input
  input_info_text = f"\n\n**Input Info from previous skill**:\n{input_info}" if input_info else ""
  llm_input = f"""**Last User Request**: {creation_request}{input_info_text}

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
  
  # Execute the generated code
  print("=" * 80)
  print("EXECUTION RESULT:")
  print("=" * 80)
  try:
    success, output_info = create_budget_or_goal_or_reminder(creation_request, input_info)
    print(f"Success: {success}")
    print(f"Output Info:")
    print(output_info)
  except Exception as e:
    print(f"Error during execution: {str(e)}")
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


def main():
  """Main function to test the create budget/goal/reminder generator"""
  print("Testing CreateBudgetOrGoalOrReminder\n")
  
  # ============================================================================
  # BUDGET/GOAL TESTS
  # ============================================================================
  
#   print("Test 1: Creating gas and car insurance budget")
#   print("-" * 80)
#   test_create_gas_budget()
#   print("\n")
  
#   print("Test 2: Creating dining out budget")
#   print("-" * 80)
#   test_create_dining_out_budget()
#   print("\n")
  
#   print("Test 3: Creating savings goal with input info")
#   print("-" * 80)
#   test_create_savings_goal()
#   print("\n")
  
#   print("Test 4: Creating budget with ambiguous category (requires followup)")
#   print("-" * 80)
#   test_create_ambiguous_budget()
#   print("\n")
  
#   print("Test 5: Creating budget with missing information (requires followup)")
#   print("-" * 80)
#   test_create_incomplete_budget()
#   print("\n")
  
  # ============================================================================
  # REMINDER TESTS
  # ============================================================================
  
#   print("Test 1: Creating Netflix cancellation reminder")
#   print("-" * 80)
#   test_create_netflix_reminder()
#   print("\n")
  
#   print("Test 2: Creating account balance reminder")
#   print("-" * 80)
#   test_create_account_balance_reminder()
#   print("\n")
  
#   print("Test 3: Creating transaction reminder")
#   print("-" * 80)
#   test_create_transaction_reminder()
#   print("\n")
  
#   print("Test 4: Creating subscription renewal reminder")
#   print("-" * 80)
#   test_create_subscription_reminder()
#   print("\n")
  
#   print("Test 5: Creating refund reminder")
#   print("-" * 80)
#   test_create_refund_reminder()
#   print("\n")
  
  print("Test 6: Creating reminder that triggers immediately")
  print("-" * 80)
  test_create_immediate_reminder()
  print("\n")
  
  print("All tests completed!")


if __name__ == "__main__":
  main()

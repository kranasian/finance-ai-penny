from google import genai
from google.genai import types
import sys
import os
from dotenv import load_dotenv
import datetime as dt_module
from datetime import datetime, timedelta, date
from typing import Tuple, Optional
import pandas as pd


# Add the parent directory to the path so we can import database and other modules
# From penny/tool_funcs/, we need to go up two levels to get to the root
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if parent_dir not in sys.path:
  sys.path.insert(0, parent_dir)

# Import from same package (tool_funcs) - use absolute imports to work both as module and when run directly
from penny.tool_funcs.create_budget_or_goal import create_budget_or_goal, VALID_GRANULARITIES, VALID_GOAL_TYPES
from penny.tool_funcs.date_utils import get_start_of_week, get_end_of_week, get_start_of_month, get_end_of_month, get_start_of_year, get_end_of_year, get_after_periods, get_date_string
from penny.tool_funcs.create_reminder import create_reminder
from database import Database

# Load environment variables
load_dotenv()

SYSTEM_PROMPT = """You are a helpful AI assistant specialized in creating budgets, goals, and reminders based on user requests. **You only output python code.**

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

- `create_budget_or_goal(category: str, match_category: str, match_caveats: str | None, type: str, granularity: str, start_date: str, end_date: str, amount: float, title: str, budget_or_goal: str) -> tuple[bool, str]`
  - Creates a spending goal or budget based on individual parameters.
  - **Method Signature**: `create_budget_or_goal(category: str, match_category: str, match_caveats: str | None, type: str, granularity: str, start_date: str, end_date: str, amount: float, title: str, budget_or_goal: str) -> tuple[bool, str]`
  - **Parameters**:
    - `category`: (string) The raw spending category text extracted from user input (e.g., "gas", "eating out"). Can be empty string if not provided.
    - `match_category`: (string) The category from the OFFICIAL CATEGORY LIST that best matches the user's goal category. Can be empty string if not provided.
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
        'create_reminder': create_reminder,
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
  llm_input = f"""{creation_request}{input_info_text}

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


def test_create_year_end_10k_savings_goal(generator: CreateBudgetOrGoalOrReminder = None):
  """
  Test method for creating a specific savings goal named 'Year End $10k Savings'.
  
  Args:
    generator: Optional CreateBudgetOrGoalOrReminder instance. If None, creates a new one.
    
  Returns:
    The generated response string
  """
  creation_request = "Create a specific savings goal named 'Year End $10k Savings' to save $10000 by the end of the year"
  
  return _run_test_with_logging(creation_request, None, generator)


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
  
  # print("Test 1: Creating gas and car insurance budget")
  # print("-" * 80)
  # test_create_gas_budget()
  # print("\n")
  
  # print("Test 2: Creating dining out budget")
  # print("-" * 80)
  # test_create_dining_out_budget()
  # print("\n")
  
#   print("Test 3: Creating savings goal with input info")
#   print("-" * 80)
#   test_create_savings_goal()
#   print("\n")
  
  # print("Test 3b: Creating Year End $10k Savings goal")
  # print("-" * 80)
  # test_create_year_end_10k_savings_goal()
  # print("\n")
  
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
  
  # print("Test 3: Creating transaction reminder")
  # print("-" * 80)
  # test_create_transaction_reminder()
  # print("\n")
  
  print("Test 4: Creating subscription renewal reminder")
  print("-" * 80)
  test_create_subscription_reminder()
  print("\n")
  
#   print("Test 5: Creating refund reminder")
#   print("-" * 80)
#   test_create_refund_reminder()
#   print("\n")
  
  # print("Test 6: Creating reminder that triggers immediately")
  # print("-" * 80)
  # test_create_immediate_reminder()
  # print("\n")
  
  # print("All tests completed!")


if __name__ == "__main__":
  main()

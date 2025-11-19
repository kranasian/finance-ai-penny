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

from database import Database

# Load environment variables
load_dotenv()

SYSTEM_PROMPT = """You are a helpful AI assistant specialized in creating reminders based on user requests. **You only output python code.**

## Your Tasks

1. Understand the **Reminder Request** which contains:
   - **What**: What to be reminded about (e.g., "cancel Netflix subscription", "checking account balance drops below $1000")
   - **When**: When the reminder will be relevant (e.g., "at the end of this year (December 31st)", "immediately when condition is met")

2. Generate reminder logic:
   - Generate the `should_remind_code` string directly inside `process_input()` based on the `what` and `when` parameters
   - The `should_remind_code` should be prefixed with `def should_remind(get_accounts_df, get_transactions_df, get_subscriptions_df):` and follow this pattern:
     - Use `reminder_messages = []` to accumulate messages
     - Use `base_date = date(...)` to hardcode the base date from "Today's date"
     - Use `current_date = date.today()` and calculate `trigger_date` based on the "when" parameter
     - Use `trigger_today = (current_date == trigger_date)` and `trigger_in_future = (current_date < trigger_date)`
     - Return `(message, next_check_date)` where:
       - If `trigger_in_future`: return `(None, trigger_date)` to check again on the trigger date
       - Otherwise: return `(None, None)` if the trigger date has passed
   - Execute the `should_remind_code` using a namespace dictionary: create `namespace = {{}}`, then `exec(should_remind_code, globals(), namespace)`, then access the function as `should_remind = namespace['should_remind']`, then call `should_remind(get_accounts_df, get_transactions_df, get_subscriptions_df)` to get `message` and `next_check_date`
   - Call `save_reminder_request(reminder_description, should_remind_code, next_check_date)` to persist the reminder (only if `next_check_date` is not None)
   - Return `(True, message)` where `message` is the confirmation message (use `message` if not None, otherwise create a confirmation message)

## Your Output

1. Write a function `process_input` that takes no arguments and returns a tuple:
   - The first element is a boolean indicating success
   - The second element is a string containing the confirmation message
2. Assume `import datetime` and `import pandas as pd` are already included. Do not include import statements in your code.
3. Only output the Python code that implements the `process_input` function.

<IMPLEMENTED_FUNCTIONS>

These functions are already implemented:

- **Generate `should_remind_code` directly**: You should generate the `should_remind_code` as a Python code string directly inside `process_input()`. The code should:
  - Be prefixed with `def should_remind(get_accounts_df, get_transactions_df, get_subscriptions_df):` function definition
  - Use `reminder_messages = []` to accumulate reminder messages
  - Use `base_date = date(...)` to hardcode the base date from "Today's date" in the prompt
  - Use `current_date = date.today()` and calculate `trigger_date` based on the "when" parameter
  - Use `trigger_today = (current_date == trigger_date)` and `trigger_in_future = (current_date < trigger_date)` to check date conditions
  - Use the function parameters `get_accounts_df`, `get_transactions_df`, and `get_subscriptions_df` to retrieve data
  - Use `datetime`, `timedelta`, and `date` for date calculations
  - The `should_remind_code` should be a multi-line string containing the function definition with the actual logic to check if the reminder condition is met
  - Execute the `should_remind_code` using a namespace dictionary: create `namespace = {{}}`, then `exec(should_remind_code, globals(), namespace)`, then access the function as `should_remind = namespace['should_remind']`, then call `should_remind(get_accounts_df, get_transactions_df, get_subscriptions_df)` to get `message` and `next_check_date`

- `get_accounts_df() -> pd.DataFrame`
  - Retrieves Accounts dataframe with columns:
    - `account_id`: unique numeric identifier
    - `name`: Account name containing bank name, account name and last 4 digits
    - `account_type`: One of: deposit_savings, deposit_money_market, deposit_checking, credit_card, loan_home_equity, loan_line_of_credit, loan_mortgage, loan_auto
    - `balance_available`: amount withdrawable for deposit accounts
    - `balance_current`: loaned amount in loans/credit accounts, usable amount in savings/checking
    - `balance_limit`: credit limit of credit-type accounts
  - Use this function to retrieve account data for reminder checking logic.

- `get_transactions_df() -> pd.DataFrame`
  - Retrieves Transactions dataframe with columns:
    - `account_id`: account where this transaction is
    - `transaction_id`: unique numeric identifier
    - `datetime`: inflow or spending date
    - `name`: establishment or service name
    - `amount`: inflow amount will be negative, outflow/spending will be positive
    - `category`: transaction category from Official Category List
    - `output_category`: category used for display
  - Use this function to retrieve transaction data for reminder checking logic.

- `get_subscriptions_df() -> pd.DataFrame`
  - Retrieves Subscriptions dataframe with columns:
    - `name`: establishment or service name with the subscription
    - `next_amount`: likely upcoming amount
    - `next_likely_payment_date`: most likely upcoming payment date
    - `next_earliest_payment_date`: earliest possible payment date
    - `next_latest_payment_date`: latest possible payment date
    - `user_cancelled_date`: date when user cancelled subscription
    - `last_transaction_date`: date of the most recent transaction for this subscription
  - Use this function to retrieve subscription data for reminder checking logic.

- `save_reminder_request(reminder_description: str, should_remind_code: str, next_check_date: date) -> None`
  - Saves a reminder request to the database for later periodic checking.
  - **Method Signature**: `save_reminder_request(reminder_description: str, should_remind_code: str, next_check_date: date) -> None`
  - **Parameters**:
    - `reminder_description`: (string) A natural description of what the reminder is for (e.g., "cancel Netflix subscription at the end of this year", "checking account balance drops below $1000")
    - `should_remind_code`: (string) The code string that checks if the reminder condition is met. This code will be executed periodically to determine if the reminder should trigger.
    - `next_check_date`: (date) The date when the reminder should be checked next. This is required and must be a valid date object.
  - **Returns**: None (function saves to database silently)
  - **Usage**: Call this function after generating and executing the should_remind code to persist the reminder to the database.

</IMPLEMENTED_FUNCTIONS>

<EXAMPLES>

input: **Reminder Request**: cancel Spotify subscription (last transaction: 2025-11-01, next payment: 2025-12-01, amount: $9.99) | at the end of this year (December 31st)
output:
```python
def process_input():
    what = "cancel Spotify subscription (last transaction: 2025-11-01, next payment: 2025-12-01, amount: $9.99)"
    when = "at the end of this year (December 31st)"
    
    # Generate should_remind code
    should_remind_code = """
def should_remind(get_accounts_df, get_transactions_df, get_subscriptions_df):
        # Accumulate reminder messages in reminder_messages
        reminder_messages = []
        
        # Hardcode Base Date from Today's date
        base_date = date(2025, 11, 1)
        
        current_date = date.today()
        trigger_date = date(2025, 12, 31)
        trigger_today = (current_date == trigger_date)
        trigger_in_future = (current_date < trigger_date)
        
        # Retrieve transactions data to check last Spotify transaction
        transactions_df = get_transactions_df()
        # Filter Spotify transactions
        spotify_transactions_df = transactions_df[
            transactions_df['name'].str.lower().str.contains('spotify', na=False)
        ]
        # Get the last Spotify transaction
        spotify_last_transaction = spotify_transactions_df.sort_values(by='datetime', ascending=False).iloc[0]
        # Correctly stating the amount and if it's a payment or a refund
        pay_or_refund = "payment" if spotify_last_transaction['amount'] >= 0 else "refund"
        reminder_messages.append(f"The last Spotify transaction was a {{pay_or_refund}} on {{spotify_last_transaction['datetime'].strftime('%B %d, %Y')}} for ${{abs(spotify_last_transaction['amount']):.0f}}.")
        
        # Retrieve subscriptions data to check Spotify status
        subscriptions_df = get_subscriptions_df()
        spotify_subscriptions_df = subscriptions_df[
            subscriptions_df['name'].str.lower().str.contains('spotify', na=False)
        ]
        
        # Check if the user marked Spotify subscription has been cancelled
        active_spotify_subscription_df = spotify_subscriptions_df[spotify_subscriptions_df['user_cancelled_date'].isnull()]
        if active_spotify_subscription_df.empty:
            cancelled_date = spotify_subscriptions_df['user_cancelled_date'].iloc[0]
            if pd.notna(cancelled_date):
                reminder_messages.append(f"You asked to remind you to cancel Spotify subscription today but looks like you already marked it as cancelled on {{cancelled_date.strftime('%B %d, %Y')}}.")
            else:
                reminder_messages.append(f"You asked to remind you to cancel Spotify subscription today but it appears to be cancelled (cancellation date not specified).")
        else:
            reminder_messages.append(f"You asked to be reminded to cancel your Spotify subscription today, December 31st, 2025.")
        
        # If December 31st, 2025 is today
        if trigger_today:
            # Return reminder and no date
            return chr(10).join(reminder_messages), None
        # If December 31st, 2025 is in the future
        elif trigger_in_future:
            # Return no reminder and return date
            return None, trigger_date
        # If December 31st, 2025 is in the past
        else:
            # Return neither reminder nor date
            return None, None
"""
    
    # Execute should_remind code to define the function
    namespace = {}
    exec(should_remind_code, globals(), namespace)
    should_remind = namespace['should_remind']
    # Call should_remind function to get message and next_check_date
    message, next_check_date = should_remind(get_accounts_df, get_transactions_df, get_subscriptions_df)
    
    # Create reminder description
    reminder_description = f"Remind me to {{what}} {{when}}"
    
    # Save reminder request
    save_reminder_request(reminder_description, should_remind_code, next_check_date)
    
    # Return success with message (use message if not None, otherwise create confirmation)
    if message is not None:
        return True, message
    else:
        return True, f"Reminder created: I will remind you to {{what}} {{when}}."
```

input: **Reminder Request**: checking account balance drops below $1000 (accounts: Chase Total Checking **1563 current: $4567, Chase Checking **3052 current: $1202) | immediately when condition is met
output:
```python
def process_input():
    what = "checking account balance drops below $1000 (accounts: Chase Total Checking **1563 current: $4567, Chase Checking **3052 current: $1202)"
    when = "immediately when condition is met"
    
    # Generate should_remind code
    should_remind_code = """
def should_remind(get_accounts_df, get_transactions_df, get_subscriptions_df):
        # Accumulate reminder messages in reminder_messages
        reminder_messages = []
        
        # Hardcode Base Date from Today's date
        base_date = date(2025, 11, 1)
        
        current_date = date.today()
        threshold = 1000.0
        
        # Retrieve accounts data to check checking account balances
        accounts_df = get_accounts_df()
        checking_accounts = accounts_df[accounts_df['account_type'].str.contains('checking', case=False, na=False)]
        
        if checking_accounts.empty:
            reminder_messages.append("No checking accounts found.")
            # Check again tomorrow
            return None, current_date + timedelta(days=1)
        
        # Get the minimum balance across all checking accounts
        min_balance = checking_accounts['balance_current'].min()
        
        # Check if balance has dropped below threshold
        if min_balance < threshold:
            reminder_messages.append(f"⚠️ ALERT: Your checking account balance has dropped below ${{threshold:.2f}}. Current minimum balance: ${{min_balance:.2f}}")
            # Return reminder immediately
            return chr(10).join(reminder_messages), None
        else:
            reminder_messages.append(f"Your checking account balance is currently ${{min_balance:.2f}}, which is above the threshold of ${{threshold:.2f}}.")
            # Check again tomorrow
            return None, current_date + timedelta(days=1)
f"""
    
    # Execute should_remind code to define the function
    namespace = {{}}
    exec(should_remind_code, globals(), namespace)
    should_remind = namespace['should_remind']
    # Call should_remind function to get message and next_check_date
    message, next_check_date = should_remind(get_accounts_df, get_transactions_df, get_subscriptions_df)
    
    # Create reminder description
    reminder_description = f"Notify me {{when}} {{what}}"
    
    # Save reminder request
    save_reminder_request(reminder_description, should_remind_code, next_check_date)
    
    # Return success with message (use message if not None, otherwise create confirmation)
    if message is not None:
        return True, message
    else:
        return True, f"Reminder created: I will notify you {{when}} {{what}}."
```

</EXAMPLES>

Today's date is {{today_date}}.
"""


class CreateReminder:
  """Handles all Gemini API interactions for creating reminders"""
  
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

  
  def generate_response(self, reminder_request: str) -> str:
    """
    Generate a response using Gemini API for creating reminders.
    
    Args:
      reminder_request: The reminder request containing "what | when" format
      
    Returns:
      Generated code as a string
    """
    # Create request text
    request_text = types.Part.from_text(text=f"""**Reminder Request**: {reminder_request}

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


def save_reminder_request(reminder_description: str, should_remind_code: str, next_check_date: date) -> None:
    """
    Save a reminder request to the database for later periodic checking.
    
    Dummy empty implementation - does nothing.
    
    Args:
        reminder_description: A natural description of what the reminder is for
        should_remind_code: The code string that checks if the reminder condition is met
        next_check_date: The date when the reminder should be checked next
    """
    pass


def create_reminder(what: str, when: str) -> Tuple[bool, str]:
    """
    Create a reminder based on the what and when parameters.
    
    Args:
        what: What to be reminded about: transaction coming in, subscription getting 
                refunded, account balances, or a clear general task 
                (e.g., "cancel Netflix subscription", "checking account balance drops below $1000").
        when: When the reminder will be relevant: date, condition, or frequency 
               (e.g., "at the end of this year (December 31st)", "immediately when condition is met").
    
    Returns:
        Tuple[bool, str]: (success, message) where success is True if the reminder was created successfully
    """
    generator = CreateReminder()
    
    # Format reminder request as "what | when"
    reminder_request = f"{what} | {when}"
    
    code = generator.generate_response(reminder_request)
    
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
        'get_accounts_df': get_accounts_df,
        'get_transactions_df': get_transactions_df,
        'get_subscriptions_df': get_subscriptions_df,
        'save_reminder_request': save_reminder_request,
        'list': list,
        'dict': dict,
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


def _run_test_with_logging(what: str, when: str, generator: CreateReminder = None):
  """
  Internal helper function that runs a test with consistent logging.
  
  Args:
    what: What to be reminded about
    when: When the reminder will be relevant
    generator: Optional CreateReminder instance. If None, creates a new one.
    
  Returns:
    The generated response string
  """
  if generator is None:
    generator = CreateReminder()
  
  # Format reminder request as "what | when"
  reminder_request = f"{what} | {when}"
  
  # Print the input
  print("=" * 80)
  print("LLM INPUT:")
  print("=" * 80)
  print(f"**Reminder Request**: {reminder_request}")
  print("=" * 80)
  print()
  
  result = generator.generate_response(reminder_request)
  
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
    success, output_info = create_reminder(what, when)
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


def test_create_hulu_cancellation_reminder(generator: CreateReminder = None):
  """
  Test method for creating a Hulu cancellation reminder scenario.
  
  Args:
    generator: Optional CreateReminder instance. If None, creates a new one.
    
  Returns:
    The generated response string
  """
  what = "cancel Hulu subscription (last transaction: 2025-10-15, next payment: 2025-11-15, amount: $15.99)"
  when = "on March 1st, 2026"
  
  return _run_test_with_logging(what, when, generator)


def test_create_credit_card_balance_reminder(generator: CreateReminder = None):
  """
  Test method for creating a credit card balance reminder scenario.
  
  Args:
    generator: Optional CreateReminder instance. If None, creates a new one.
    
  Returns:
    The generated response string
  """
  what = "credit card balance exceeds $2000 (account: Chase Sapphire **1234 current balance: $1500, limit: $5000)"
  when = "immediately when condition is met"
  
  return _run_test_with_logging(what, when, generator)


def test_create_apple_subscription_reminder(generator: CreateReminder = None):
  """
  Test method for creating an Apple subscription cancellation reminder scenario.
  
  Args:
    generator: Optional CreateReminder instance. If None, creates a new one.
    
  Returns:
    The generated response string
  """
  what = "cancel Apple Music subscription"
  when = "on January 15th, 2026"
  
  return _run_test_with_logging(what, when, generator)


def test_create_savings_goal_reminder(generator: CreateReminder = None):
  """
  Test method for creating a savings account goal reminder scenario.
  
  Args:
    generator: Optional CreateReminder instance. If None, creates a new one.
    
  Returns:
    The generated response string
  """
  what = "savings account balance exceeds $5000 (account: Chase Savings **3052 current: $3200)"
  when = "immediately when condition is met"
  
  return _run_test_with_logging(what, when, generator)


def test_create_income_transaction_reminder(generator: CreateReminder = None):
  """
  Test method for creating an income transaction reminder scenario.
  
  Args:
    generator: Optional CreateReminder instance. If None, creates a new one.
    
  Returns:
    The generated response string
  """
  what = "new income transaction posted to savings account (account: Chase Savings **3052, expected: Interest Payment)"
  when = "immediately whenever it happens"
  
  return _run_test_with_logging(what, when, generator)


def test_create_amazon_subscription_reminder(generator: CreateReminder = None):
  """
  Test method for creating an Amazon Prime subscription reminder scenario.
  
  Args:
    generator: Optional CreateReminder instance. If None, creates a new one.
    
  Returns:
    The generated response string
  """
  what = "Amazon Prime subscription renews"
  when = "7 days before renewal date"
  
  return _run_test_with_logging(what, when, generator)


def test_create_mortgage_payment_reminder(generator: CreateReminder = None):
  """
  Test method for creating a mortgage payment reminder scenario.
  
  Args:
    generator: Optional CreateReminder instance. If None, creates a new one.
    
  Returns:
    The generated response string
  """
  what = "mortgage payment due (account: Home Mortgage **5678, amount: $2500)"
  when = "3 days before payment date"
  
  return _run_test_with_logging(what, when, generator)


def test_create_rent_payment_reminder(generator: CreateReminder = None):
  """
  Test method for creating a rent payment reminder scenario.
  
  Args:
    generator: Optional CreateReminder instance. If None, creates a new one.
    
  Returns:
    The generated response string
  """
  what = "rent payment due (amount: $1800)"
  when = "on the 1st of each month"
  
  return _run_test_with_logging(what, when, generator)


def test_with_inputs(what: str, when: str, generator: CreateReminder = None):
  """
  Convenient method to test the generator with custom inputs.
  
  Args:
    what: What to be reminded about
    when: When the reminder will be relevant
    generator: Optional CreateReminder instance. If None, creates a new one.
    
  Returns:
    The generated response string
  """
  return _run_test_with_logging(what, when, generator)


def main():
  """Main function to test the create reminder generator"""
  print("Testing CreateReminder\n")
  
  # ============================================================================
  # REMINDER TESTS
  # ============================================================================
  
  # print("Test 1: Creating Hulu cancellation reminder")
  # print("-" * 80)
  # test_create_hulu_cancellation_reminder()
  # print("\n")
  
  print("Test 2: Creating credit card balance reminder")
  print("-" * 80)
  test_create_credit_card_balance_reminder()
  print("\n")
  
  # print("Test 3: Creating Apple subscription reminder")
  # print("-" * 80)
  # test_create_apple_subscription_reminder()
  # print("\n")
  
  # print("Test 4: Creating savings goal reminder")
  # print("-" * 80)
  # test_create_savings_goal_reminder()
  # print("\n")
  
  # print("Test 5: Creating income transaction reminder")
  # print("-" * 80)
  # test_create_income_transaction_reminder()
  # print("\n")
  
  # print("Test 6: Creating Amazon subscription reminder")
  # print("-" * 80)
  # test_create_amazon_subscription_reminder()
  # print("\n")
  
  # print("Test 7: Creating mortgage payment reminder")
  # print("-" * 80)
  # test_create_mortgage_payment_reminder()
  # print("\n")
  
  # print("Test 8: Creating rent payment reminder")
  # print("-" * 80)
  # test_create_rent_payment_reminder()
  # print("\n")
  
  # print("All tests completed!")


if __name__ == "__main__":
  main()

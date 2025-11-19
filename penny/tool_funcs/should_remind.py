from google import genai
from google.genai import types
import sys
import os
from dotenv import load_dotenv
import datetime as dt_module
from datetime import datetime, timedelta, date
from typing import Tuple, Optional, Callable, Callable
import pandas as pd


# Add the parent directory to the path so we can import database and other modules
# From penny/tool_funcs/, we need to go up two levels to get to the root
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if parent_dir not in sys.path:
  sys.path.insert(0, parent_dir)

from database import Database

# Load environment variables
load_dotenv()

SYSTEM_PROMPT = """You are a helpful AI assistant specialized in generating reminder checking functions. **You only output python code.**

## Your Input

1. **What**: What to be reminded about (e.g., "cancel Netflix subscription", "checking account balance drops below $1000")
2. **When**: When the reminder will be relevant (e.g., "at the end of this year (December 31st)", "immediately when condition is met")

## Your Task

1. Determine the reminder message and next check date based on the what and when parameters.
2. Generate a function `def should_remind(get_accounts_df, get_transactions_df, get_subscriptions_df):`.

## Your Output

1. Write only the `should_remind` function definition with its implementation.
2. Assume `import datetime` and `import pandas as pd` are already included. Do not include import statements in your code.
3. Only output the Python code that implements the `should_remind` function.

<IMPLEMENTED_FUNCTIONS>

These functions are available to use within `should_remind`:

- `get_accounts_df() -> pd.DataFrame`
  - Retrieves Accounts dataframe with columns:
    - `account_id`: unique numeric identifier
    - `name`: Account name containing bank name, account name and last 4 digits
    - `account_type`: One of: deposit_savings, deposit_money_market, deposit_checking, credit_card, loan_home_equity, loan_line_of_credit, loan_mortgage, loan_auto
    - `balance_available`: amount withdrawable for deposit accounts
    - `balance_current`: loaned amount in loans/credit accounts, usable amount in savings/checking
    - `balance_limit`: credit limit of credit-type accounts

- `get_transactions_df() -> pd.DataFrame`
  - Retrieves Transactions dataframe with columns:
    - `account_id`: account where this transaction is
    - `transaction_id`: unique numeric identifier
    - `datetime`: inflow or spending date
    - `name`: establishment or service name
    - `amount`: inflow amount will be negative, outflow/spending will be positive
    - `category`: transaction category from Official Category List
    - `output_category`: category used for display

- `get_subscriptions_df() -> pd.DataFrame`
  - Retrieves Subscriptions dataframe with columns:
    - `name`: establishment or service name with the subscription
    - `next_amount`: likely upcoming amount
    - `next_likely_payment_date`: most likely upcoming payment date
    - `next_earliest_payment_date`: earliest possible payment date
    - `next_latest_payment_date`: latest possible payment date
    - `user_cancelled_date`: date when user cancelled subscription
    - `last_transaction_date`: date of the most recent transaction for this subscription

</IMPLEMENTED_FUNCTIONS>

<EXAMPLES>

input: **What**: cancel Spotify subscription (last transaction: 2025-11-01, next payment: 2025-12-01, amount: $9.99)
**When**: at the end of this year (December 31st)
output: 
```python
def should_remind(get_accounts_df, get_transactions_df, get_subscriptions_df):
    # Accumulate reminder messages in reminder_messages
    reminder_messages = []
    
    today = date.today()
    trigger_date = date(2025, 12, 31)
    trigger_today = (today == trigger_date)
    trigger_in_future = (today < trigger_date)
    
    # Retrieve subscriptions data to check Spotify status
    subscriptions_df = get_subscriptions_df()
    spotify_subscriptions_df = subscriptions_df[
        subscriptions_df['name'].str.lower().str.contains('spotify', na=False)
    ]
    
    # Check if Spotify subscriptions exist
    if spotify_subscriptions_df.empty:
        if trigger_today:
            return None, None
        elif trigger_in_future:
            return None, trigger_date
        else:
            return None, None
    
    # Check if the user marked Spotify subscription has been cancelled
    active_spotify_subscription_df = spotify_subscriptions_df[spotify_subscriptions_df['user_cancelled_date'].isnull()]
    if active_spotify_subscription_df.empty:
        # All Spotify subscriptions are cancelled, get the cancelled date from the first one
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
```

input: **What**: checking account balance drops below $1000 (accounts: Chase Total Checking **1563 current: $4567, Chase Checking **3052 current: $1202)
**When**: immediately when condition is met for the next 3 months
output:
```python
def should_remind(get_accounts_df, get_transactions_df, get_subscriptions_df):
    # Accumulate reminder messages in reminder_messages
    reminder_messages = []
    
    # Hardcode Reminder Start Date from Today's date
    reminder_start_date = date(2025, 11, 1)
    reminder_end_date = reminder_start_date + timedelta(months=3)
    
    today = date.today()
    tomorrow = today + timedelta(days=1)
    threshold = 1000.0
    
    # Retrieve accounts data to check checking account balances
    accounts_df = get_accounts_df()
    checking_accounts = accounts_df[accounts_df['account_type'].str.contains('checking', case=False, na=False)]
    
    if checking_accounts.empty:
        reminder_messages.append("No checking accounts found.")
        if tomorrow <= reminder_end_date:
            return None, tomorrow
        else:
            return None, None
    
    # Get the minimum balance across all checking accounts
    min_balance = checking_accounts['balance_current'].min()
    
    # Check if balance has dropped below threshold
    if min_balance < threshold:
        reminder_messages.append(f"⚠️ ALERT: Your checking account balance has dropped below ${{threshold:.2f}}. Current minimum balance: ${{min_balance:.2f}}")
        # Return reminder immediately
        return chr(10).join(reminder_messages), None
    else:
        reminder_messages.append(f"Your checking account balance is currently ${{min_balance:.2f}}, which is above the threshold of ${{threshold:.2f}}.")
        if tomorrow <= reminder_end_date:
            return None, tomorrow
        else:
            return None, None
```

</EXAMPLES>

Today's date is {today_date}.
"""


class ShouldRemindGenerator:
  """Handles all Gemini API interactions for generating should_remind functions"""
  
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
    Generate a response using Gemini API for generating should_remind functions.
    
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


def should_remind(what: str, when: str, get_accounts_df=None, get_transactions_df=None, get_subscriptions_df=None, user_id: int = 1) -> Tuple[Optional[str], Optional[date], str]:
    """
    Generate and execute a should_remind function based on the what and when parameters.
    
    Args:
        what: What to be reminded about: transaction coming in, subscription getting 
                refunded, account balances, or a clear general task 
                (e.g., "cancel Netflix subscription", "checking account balance drops below $1000").
        when: When the reminder will be relevant: date, condition, or frequency 
               (e.g., "at the end of this year (December 31st)", "immediately when condition is met").
        get_accounts_df: Optional function to retrieve accounts dataframe. If None, creates from database.
        get_transactions_df: Optional function to retrieve transactions dataframe. If None, creates from database.
        get_subscriptions_df: Optional function to retrieve subscriptions dataframe. If None, creates from database.
        user_id: User ID for database queries (default: 1)
    
    Returns:
        Tuple[Optional[str], Optional[date], str]: (message, next_check_date, should_remind_code_string) where:
            - message is a string (or None) containing reminder messages
            - next_check_date is a date (or None) indicating when to check again
            - should_remind_code_string is the string representation of the function code
    """
    generator = ShouldRemindGenerator()
    
    # Format reminder request as "what | when"
    reminder_request = f"{what} | {when}"
    
    code = generator.generate_response(reminder_request)
    
    # Extract Python code from markdown code blocks if present
    should_remind_code = extract_python_code(code)
    
    # Import pandas if available
    try:
      import pandas as pd
    except ImportError:
      pd = None
      raise ImportError("pandas is required for database operations")
    
    # Create dataframe getter functions if not provided
    if get_accounts_df is None:
      def _get_accounts_df():
        """Retrieve accounts from database and format according to expected structure."""
        try:
          db = Database()
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
      get_accounts_df = _get_accounts_df
    
    if get_transactions_df is None:
      def _get_transactions_df():
        """Retrieve transactions from database and format according to expected structure."""
        try:
          db = Database()
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
      get_transactions_df = _get_transactions_df
    
    if get_subscriptions_df is None:
      def _get_subscriptions_df():
        """Retrieve subscriptions from database and format according to expected structure."""
        try:
          db = Database()
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
      get_subscriptions_df = _get_subscriptions_df
    
    # Execute the generated code to get the function
    try:
      # Create a namespace that supports both datetime.datetime() and datetime.now()
      datetime_ns = _DateTimeNamespace(dt_module, datetime)
      
      # Create a namespace for execution with available functions
      namespace = {
        'datetime': datetime_ns,  # Supports both datetime.datetime() and datetime.now()
        'timedelta': timedelta,
        'date': date,
        'pd': pd,
        'chr': chr,  # For chr(10) to join messages
      }
      
      # Execute the code to define the should_remind function
      exec(should_remind_code, namespace)
      
      # Get the should_remind function from the namespace
      if 'should_remind' in namespace:
        should_remind_func = namespace['should_remind']
        
        # Execute the function with the dataframe getters
        message, next_check_date = should_remind_func(get_accounts_df, get_transactions_df, get_subscriptions_df)
        
        return message, next_check_date, should_remind_code
      else:
        raise ValueError("Generated code does not contain should_remind function")
    except Exception as e:
      raise ValueError(f"Error executing generated code: {str(e)}")


def _run_test_with_logging(what: str, when: str, generator: ShouldRemindGenerator = None):
  """
  Internal helper function that runs a test with consistent logging.
  
  Args:
    what: What to be reminded about
    when: When the reminder will be relevant
    generator: Optional ShouldRemindGenerator instance. If None, creates a new one.
    
  Returns:
    The generated response string
  """
  if generator is None:
    generator = ShouldRemindGenerator()
  
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
  
  # Try to generate and execute the function
  print("=" * 80)
  print("GENERATED FUNCTION AND EXECUTION:")
  print("=" * 80)
  try:
    message, next_check_date, should_remind_code = should_remind(what, when)
    print("Successfully generated and executed should_remind function")
    print(f"\nMessage: {message}")
    print(f"Next check date: {next_check_date}")
    print("\nGenerated code:")
    print(should_remind_code)
  except Exception as e:
    print(f"Error generating/executing function: {str(e)}")
    import traceback
    print(traceback.format_exc())
  print("=" * 80)
  print()
  
  return result


# Test cases list
TEST_CASES = [
  {
    "name": "Spotify cancellation reminder",
    "what": "cancel Spotify subscription (last transaction: 2025-11-01, next payment: 2025-12-01, amount: $9.99)",
    "when": "at the end of this year (December 31st)"
  },
  {
    "name": "Checking account balance drops below threshold",
    "what": "checking account balance drops below $1000 (accounts: Chase Total Checking **1563 current: $4567, Chase Checking **3052 current: $1202)",
    "when": "immediately when condition is met"
  },
  {
    "name": "Credit card balance exceeds threshold",
    "what": "credit card balance exceeds $2000 (account: Chase Sapphire **1234 current balance: $1500, limit: $5000, utilization: 30%)",
    "when": "immediately when condition is met for the next 6 months"
  },
  {
    "name": "Savings account goal reminder",
    "what": "savings account balance exceeds $5000 (account: Chase Savings **3052 current: $3200, goal: $5000, remaining: $1800)",
    "when": "immediately when condition is met"
  },
  {
    "name": "Subscription renewal reminder",
    "what": "Netflix subscription renews (last transaction: 2025-10-15, next payment: 2025-11-15, amount: $15.99, subscription active: True)",
    "when": "7 days before renewal date"
  },
  {
    "name": "Multiple subscription cancellation",
    "what": "cancel all streaming subscriptions (Netflix: $15.99/month, Spotify: $9.99/month, Hulu: $7.99/month, all active)",
    "when": "on December 31st, 2025"
  },
  {
    "name": "Mortgage payment due reminder",
    "what": "mortgage payment due (account: Home Mortgage **5678, amount: $2500, next payment date: 2025-12-01, current balance: $245000)",
    "when": "3 days before payment date"
  },
  {
    "name": "Income transaction monitoring",
    "what": "new income transaction posted to savings account (account: Chase Savings **3052, expected: Interest Payment or CA State Payroll, last income: 2025-10-31)",
    "when": "immediately whenever it happens"
  },
  {
    "name": "Rent payment monthly reminder",
    "what": "rent payment due (amount: $1800, expected merchant: Property Management LLC, last payment: 2025-10-01)",
    "when": "on the 1st of each month"
  },
  {
    "name": "Credit card payment due reminder",
    "what": "credit card minimum payment due (account: Chase Sapphire **1234, current balance: $1500, minimum payment: $25, due date: 2025-11-25)",
    "when": "5 days before due date"
  },
  {
    "name": "Emergency fund threshold reminder",
    "what": "emergency fund balance drops below $10000 (account: Chase Savings **3052 current: $12500, target: $10000, monthly expenses: $3500)",
    "when": "immediately when condition is met"
  },
  {
    "name": "Loan balance payoff reminder",
    "what": "auto loan balance drops below $5000 (account: Auto Loan **7890, current balance: $8500, monthly payment: $350, remaining payments: 24)",
    "when": "immediately when condition is met"
  },
  {
    "name": "Large transaction alert",
    "what": "any transaction exceeds $500 on checking account (account: Chase Total Checking **1563, recent large transactions: Costco $450 on 2025-10-20)",
    "when": "immediately when condition is met"
  },
  {
    "name": "Subscription price increase reminder",
    "what": "Spotify subscription price increases above $10 (current: $9.99/month, last transaction: 2025-11-01, subscription active: True)",
    "when": "immediately when condition is met"
  },
  {
    "name": "Tax refund arrival reminder",
    "what": "tax refund deposit arrives in checking account (account: Chase Total Checking **1563, expected amount: $2500, expected date range: 2026-02-15 to 2026-03-15)",
    "when": "immediately when condition is met"
  },
  {
    "name": "Paycheck arrival reminder",
    "what": "paycheck not received by 5th of month (account: Chase Total Checking **1563, expected: CA State Payroll $1440, last received: 2025-10-31)",
    "when": "on the 5th of each month if not received"
  },
  {
    "name": "Account balance above threshold reminder",
    "what": "checking account balance exceeds $10000 (account: Chase Total Checking **1563 current: $4567, threshold: $10000, goal: move excess to savings)",
    "when": "immediately when condition is met"
  }
]


def get_test_case(test_name_or_index):
  """
  Get a test case by name or index.
  
  Args:
    test_name_or_index: Name of the test case or its index in TEST_CASES
    
  Returns:
    Dictionary with 'name', 'what', and 'when' keys, or None if not found
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


def run_test(test_name_or_index_or_dict, generator: ShouldRemindGenerator = None):
  """
  Run a single test case.
  
  Args:
    test_name_or_index_or_dict: Test case name, index, or a dictionary with 'what' and 'when' keys
    generator: Optional ShouldRemindGenerator instance. If None, creates a new one.
    
  Returns:
    The generated response string
  """
  if isinstance(test_name_or_index_or_dict, dict):
    what = test_name_or_index_or_dict.get("what")
    when = test_name_or_index_or_dict.get("when")
    name = test_name_or_index_or_dict.get("name", "Custom test")
  else:
    test_case = get_test_case(test_name_or_index_or_dict)
    if test_case is None:
      raise ValueError(f"Test case not found: {test_name_or_index_or_dict}")
    what = test_case["what"]
    when = test_case["when"]
    name = test_case["name"]
  
  print(f"Running test: {name}")
  return _run_test_with_logging(what, when, generator)


def run_tests(test_names_or_indices_or_dicts=None, generator: ShouldRemindGenerator = None):
  """
  Run multiple test cases.
  
  Args:
    test_names_or_indices_or_dicts: List of test case names, indices, or dictionaries. 
                                     If None, runs all tests.
    generator: Optional ShouldRemindGenerator instance. If None, creates a new one.
    
  Returns:
    List of generated response strings
  """
  if test_names_or_indices_or_dicts is None:
    test_names_or_indices_or_dicts = list(range(len(TEST_CASES)))
  
  results = []
  for test_input in test_names_or_indices_or_dicts:
    results.append(run_test(test_input, generator))
    print("\n")
  
  return results


def main():
  """Main function to test the should_remind generator"""
  print("Testing ShouldRemindGenerator\n")
  
  # ============================================================================
  # REMINDER TESTS
  # ============================================================================
  
  # Run a single test by name
  # run_test("Spotify cancellation reminder")
  
  # Run a single test by index
  # run_test(0)
  
  # Run a single test with custom input
  # run_test({
  #   "name": "Custom test",
  #   "what": "custom reminder what",
  #   "when": "custom reminder when"
  # })
  
  # Run multiple specific tests
  # run_tests(["Spotify cancellation reminder", "Checking account balance drops below threshold"])
  
  # Run multiple tests by indices
  # run_tests([0, 1, 2])
  
  # Run all tests
  # run_tests()
  
  # Example: Run first test
  print("Running first test case:")
  print("-" * 80)
  run_test(15)
  print("\n")
  
  print("All tests completed!")


if __name__ == "__main__":
  main()

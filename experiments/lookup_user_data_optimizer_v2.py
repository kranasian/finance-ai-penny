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


SYSTEM_PROMPT = """Your name is "Penny" and you are a helpful AI specialized in code generation. **You only output python code.**

## Task & Rules
1. **Output**: Write `process_input() -> tuple[bool, str]`. Keep code concise. Minimal comments.
2. **Tools**: Use `IMPLEMENTED_FUNCTIONS` & `IMPLEMENTED_DATE_FUNCTIONS`. `import datetime` is assumed.
3. **Matching**: Use partial case-insensitive matching for Account/Subscription names.
4. **Safety**: Always check `if df.empty: ...` before accessing data.
5. **Output Collection**: Accumulate all output strings in a list and join with newlines. Return the joined string as the second element of the tuple.

Today: |TODAY_DATE|.

<IMPLEMENTED_FUNCTIONS>

- `retrieve_depository_accounts() -> pd.DataFrame`: checking, savings. Cols: account_type, account_name, balance_available, balance_current
- `retrieve_credit_accounts() -> pd.DataFrame`: credit, loans. Cols: account_type, account_name, balance_available, balance_current, balance_limit
- `account_names_and_balances(df, template) -> str`: Format account list.
- `utter_account_totals(df, template) -> str`: Sum balances and format.
- `retrieve_income_transactions() -> pd.DataFrame`: Past income. Cols: date, transaction_name, amount, category, transaction_id, account_id
- `retrieve_spending_transactions() -> pd.DataFrame`: Past spending. Cols: date, transaction_name, amount, category, transaction_id, account_id
- `transaction_names_and_amounts(df, template) -> str`: List transactions. Placeholders: {amount_with_direction}, {transaction_id}, {account_id}, and any DataFrame column
- `utter_transaction_total(df, template) -> str`: Sum transactions (auto-detects income vs spending from category). Placeholders: {income_total_amount}, {spending_total_amount}. Example: "Total income: {income_total_amount}" or "Total spending: {spending_total_amount}"
- `compare_income_or_spending(df, template, metadata=None) -> str`: Compare 2 groups. Placeholders: {difference}, {more_label}, {more_amount}, {less_label}, {less_amount}
- `retrieve_spending_forecasts(granularity='monthly') -> pd.DataFrame`: Future spending. Cols: start_date, forecasted_amount, category
- `retrieve_income_forecasts(granularity='monthly') -> pd.DataFrame`: Future income. Cols: start_date, forecasted_amount, category
- `forecast_dates_and_amount(df, template) -> str`: List forecasts.
- `utter_forecast_amount(amount, template) -> str`: Format forecast amount. Placeholders: {income_total_amount}, {spending_total_amount}. Example: "Total Expected Income: {income_total_amount}" or "Spending: {spending_total_amount}"
- `utter_absolute_amount(amount, template) -> str`: Format absolute amount. Placeholders: {amount}, {amount_with_direction}
- `retrieve_subscriptions() -> pd.DataFrame`: Cols: transaction_name, amount, category, subscription_name, date
- `subscription_names_and_amounts(df, template) -> str`: List subscriptions.
- `utter_subscription_totals(df, template) -> str`: Sum subscriptions.

</IMPLEMENTED_FUNCTIONS>

<IMPLEMENTED_DATE_FUNCTIONS>

- `get_date(y, m, d)`, `get_start_of_month(date)`, `get_end_of_month(date)`
- `get_start_of_year(date)`, `get_end_of_year(date)`
- `get_start_of_week(date)`, `get_end_of_week(date)`
- `get_after_periods(date, granularity, count)`, `get_date_string(date)`

</IMPLEMENTED_DATE_FUNCTIONS>

<ACCOUNT_TYPE>

deposit_savings, deposit_money_market, deposit_checking, credit_card, loan_home_equity, loan_line_of_credit, loan_mortgage, loan_auto

</ACCOUNT_TYPE>

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

input: User: What is my net worth and checking balance?
output:
```python
def process_input():
    output_lines = []
    assets = retrieve_depository_accounts()
    credit = retrieve_credit_accounts()
    if assets.empty and credit.empty:
        return True, "No accounts."

    tot_a = assets['balance_current'].sum() if not assets.empty else 0
    tot_l = credit['balance_current'].sum() if not credit.empty else 0
    net_worth = tot_a - tot_l
    output_lines.append(f"Assets: {utter_absolute_amount(tot_a, '{amount_with_direction}')}")
    output_lines.append(f"Liabilities: {utter_absolute_amount(tot_l, '{amount_with_direction}')}")
    output_lines.append(f"Net Worth: {utter_absolute_amount(net_worth, '{amount_with_direction}')}")

    if not assets.empty:
        chk = assets[assets['account_type'] == 'deposit_checking']
        if not chk.empty:
            output_lines.append("Checking:")
            output_lines.append(account_names_and_balances(chk, "Account '{account_name}' has {balance_current} left with {balance_available} available now."))
    return True, chr(10).join(output_lines)
```

input: User: Can I afford $200 for concert tickets next week?
output:
```python
def process_input():
    output_lines = []
    liq = retrieve_depository_accounts()
    cash = liq['balance_available'].sum() if not liq.empty else 0
    
    today = datetime.now()
    sp = retrieve_spending_forecasts(granularity='weekly')
    
    nxt = get_start_of_week(get_after_periods(today, 'weekly', 1))
    sp = sp[sp['start_date'] == nxt] if not sp.empty else pd.DataFrame()
    obligations = sp['forecasted_amount'].sum() if not sp.empty else 0
    
    rem = cash - obligations - 200
    
    output_lines.append(f"Cash: {utter_absolute_amount(cash, '{amount_with_direction}')}")
    output_lines.append(f"Next Week Expenses: {utter_absolute_amount(obligations, '{amount}')}")
    
    if rem >= 0:
        output_lines.append(f"Yes. You'll have {utter_absolute_amount(rem, '{amount_with_direction}')} left.")
    else:
        output_lines.append(f"No. You'll be short by {utter_absolute_amount(rem, '{amount_with_direction}')}.")
    return True, chr(10).join(output_lines)
```

input: User: List subscriptions paid last month.
output:
```python
def process_input():
    output_lines = []
    subs = retrieve_subscriptions()
    if subs.empty:
        return True, "No subscriptions."
    
    start = get_start_of_month(get_after_periods(datetime.now(), 'monthly', -1))
    end = get_end_of_month(start)
    
    subs = subs[(subs['date'] >= start) & (subs['date'] <= end)]
    if subs.empty:
        return True, "No subscriptions paid last month."
        
    output_lines.append("Subscriptions paid last month:")
    output_lines.append(subscription_names_and_amounts(subs, "- {amount_with_direction} {subscription_name} on {date}."))
    output_lines.append(utter_subscription_totals(subs, "Total: {total_amount}"))
    return True, chr(10).join(output_lines)
```

input: User: How did my income compare to my spending last year?
output:
```python
def process_input():
    output_lines = []
    start = get_start_of_year(get_after_periods(datetime.now(), 'yearly', -1))
    end = get_end_of_year(start)
    
    inc = retrieve_income_transactions()
    sp = retrieve_spending_transactions()
    
    inc = inc[(inc['date'] >= start) & (inc['date'] <= end)] if not inc.empty else pd.DataFrame()
    sp = sp[(sp['date'] >= start) & (sp['date'] <= end)] if not sp.empty else pd.DataFrame()
    
    if inc.empty and sp.empty:
        return True, "No data last year."
        
    i_val = inc['amount'].sum() if not inc.empty else 0
    s_val = sp['amount'].sum() if not sp.empty else 0
    diff = i_val - s_val
    
    output_lines.append(f"Last Year ({start.year}):")
    output_lines.append(f"Income: {utter_absolute_amount(i_val, '{amount}')}")
    output_lines.append(f"Spending: {utter_absolute_amount(s_val, '{amount}')}")
    output_lines.append(f"Net: {utter_absolute_amount(diff, '{amount_with_direction}')}")
    return True, chr(10).join(output_lines)
```

input: User: Project my savings for the next 4 weeks.
output:
```python
def process_input():
    output_lines = []
    today = datetime.now()
    inc = retrieve_income_forecasts('weekly')
    sp = retrieve_spending_forecasts('weekly')
    
    if inc.empty and sp.empty:
        return True, "No forecasts."
    
    start_next = get_start_of_week(get_after_periods(today, 'weekly', 1))
    dates = [get_after_periods(start_next, 'weekly', i) for i in range(4)]
    
    total_sav = 0
    output_lines.append("Projected Savings (Next 4 Weeks):")
    
    for d in dates:
        i_wk = inc[inc['start_date'] == d]['forecasted_amount'].sum() if not inc.empty else 0
        s_wk = sp[sp['start_date'] == d]['forecasted_amount'].sum() if not sp.empty else 0
        sav = i_wk - s_wk
        total_sav += sav
        d_str = get_date_string(d)
        output_lines.append(f"- Wk of {d_str}: Income {utter_absolute_amount(i_wk, '{amount}')} - Spending {utter_absolute_amount(s_wk, '{amount}')} = Savings {utter_absolute_amount(sav, '{amount_with_direction}')}")
        
    output_lines.append(f"Total Projected Savings: {utter_absolute_amount(total_sav, '{amount_with_direction}')}")
    return True, chr(10).join(output_lines)
```

</EXAMPLES>

<ACCOUNT_NAMES>

|ACCOUNT_NAMES|

</ACCOUNT_NAMES>

<SUBSCRIPTION_NAMES>

|SUBSCRIPTION_NAMES|

</SUBSCRIPTION_NAMES>
"""


# - `retrieve_subscriptions() -> pd.DataFrame`: Cols: transaction_name, amount, category, subscription_name, date
# - `subscription_names_and_amounts(df, template) -> str`: List subscriptions.
# - `utter_subscription_totals(df, template) -> str`: Sum subscriptions.
# - `respond_to_app_inquiry(inquiry) -> str`: For general Qs.

class LookupUserDataOptimizer:
  """Handles all Gemini API interactions for financial planning and optimization"""
  
  def __init__(self, model_name="gemini-flash-lite-latest"):
    """Initialize the Gemini agent with API configuration for financial planning"""
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

  def _build_account_names_section(self, user_id: int) -> str:
    """
    Build the ACCOUNT_NAMES section dynamically based on the user's accounts.
    
    Args:
      user_id: The user ID to retrieve accounts for
      
    Returns:
      String containing the ACCOUNT_NAMES section
    """
    db = Database()
    accounts = db.get_accounts_by_user(user_id)
    
    if not accounts:
      return "    (No accounts found for this user)"
    
    account_names_list = []
    for account in accounts:
      account_name = account.get('account_name', '')
      if account_name:
        account_names_list.append(f"    - `{account_name}`")
    
    account_names_text = "\n".join(account_names_list) if account_names_list else "    (No account names found)"
    
    return account_names_text

  def _build_subscription_names_section(self, user_id: int) -> str:
    """
    Build the SUBSCRIPTION_NAMES section dynamically based on the user's subscriptions.
    
    Args:
      user_id: The user ID to retrieve subscriptions for
      
    Returns:
      String containing the SUBSCRIPTION_NAMES section
    """
    db = Database()
    # Get subscription transactions to extract unique subscription names
    subscription_transactions = db.get_subscription_transactions(user_id, confidence_score_bills_threshold=0.5)
    
    if not subscription_transactions:
      return "    (No subscriptions found for this user)"
    
    # Extract unique subscription names from transactions
    subscription_names_set = set()
    for transaction in subscription_transactions:
      subscription_name = transaction.get('subscription_name', '')
      if subscription_name:
        subscription_names_set.add(subscription_name)
    
    subscription_names_list = [f"    - `{name}`" for name in sorted(subscription_names_set)]
    
    subscription_names_text = "\n".join(subscription_names_list) if subscription_names_list else "    (No subscription names found)"
    
    return subscription_names_text

  def _get_today_date_string(self) -> str:
    """
    Get today's date formatted as "Day, Month DD, YYYY"
    
    Returns:
      String containing today's date in the specified format
    """
    today = datetime.now()
    return today.strftime("%A, %B %d, %Y")

  
  def generate_response(self, last_user_request: str, user_id: int = 1) -> str:
    """
    Generate a response using Gemini API for financial planning.
    
    Args:
      last_user_request: The last user request as a string
      user_id: User ID for building dynamic sections (default: 1)
      
    Returns:
      Generated code as a string
    """
    # Build dynamic ACCOUNT_NAMES section for this user
    account_names_section = self._build_account_names_section(user_id)
    # Build dynamic SUBSCRIPTION_NAMES section for this user
    subscription_names_section = self._build_subscription_names_section(user_id)
    # Get today's date
    today_date = self._get_today_date_string()
    
    # Replace placeholders in system prompt
    full_system_prompt = self.system_prompt.replace("|ACCOUNT_NAMES|", account_names_section)
    full_system_prompt = full_system_prompt.replace("|SUBSCRIPTION_NAMES|", subscription_names_section)
    full_system_prompt = full_system_prompt.replace("|TODAY_DATE|", today_date)
    
    # Create request text with Last User Request and Previous Conversation
    request_text = types.Part.from_text(text=f"""User: {last_user_request}
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


def _run_test_with_logging(last_user_request: str, lookup_data: LookupUserDataOptimizer = None, user_id: int = None):
  """
  Internal helper function that runs a test with consistent logging.
  
  Args:
    last_user_request: The last user request as a string
    lookup_data: Optional LookupUserDataOptimizer instance. If None, creates a new one.
    user_id: User ID for sandbox execution (default: HeavyDataUser ID from database)
    
  Returns:
    The generated response string
  """
  if lookup_data is None:
    lookup_data = LookupUserDataOptimizer()
  
  # Get HeavyDataUser ID if not provided
  if user_id is None:
    user_id = _get_heavy_data_user_id()
  
  # Construct LLM input
  llm_input = f"""User: {last_user_request}"""
  
  # Print the input
  print("=" * 80)
  print("LLM INPUT:")
  print("=" * 80)
  print(llm_input)
  print("=" * 80)
  print()
  
  result = lookup_data.generate_response(last_user_request, user_id)
  
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
    success, output_string, logs = sandbox.execute_agent_with_tools(result, user_id)
    
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
  
  return result


def test_hows_my_accounts_doing(lookup_data: LookupUserDataOptimizer = None):
  """
  Test method for "how's my accounts doing?" scenario.
  
  Args:
    lookup_data: Optional LookupUserDataOptimizer instance. If None, creates a new one.
    
  Returns:
    The generated response string
  """
  last_user_request = "how's my accounts doing?"
  return _run_test_with_logging(last_user_request, lookup_data)


def test_how_is_my_net_worth_doing_lately(lookup_data: LookupUserDataOptimizer = None):
  """
  Test method for "how is my net worth doing lately?" scenario with conversational distractions.
  
  Args:
    lookup_data: Optional LookupUserDataOptimizer instance. If None, creates a new one.
    
  Returns:
    The generated response string
  """
  last_user_request = "how is my net worth doing lately?"
  return _run_test_with_logging(last_user_request, lookup_data)


def test_analyze_income_and_spending_patterns_for_savings(lookup_data: LookupUserDataOptimizer = None):
  """
  Test method for analyzing recent income and spending patterns to identify areas where spending can be reduced.
  
  Args:
    lookup_data: Optional LookupUserDataOptimizer instance. If None, creates a new one.
    
  Returns:
    The generated response string
  """
  last_user_request = "Analyze recent income and spending patterns across all accounts to identify areas where spending can be immediately reduced to facilitate saving money."
  return _run_test_with_logging(last_user_request, lookup_data)


def test_compare_projected_income_across_months(lookup_data: LookupUserDataOptimizer = None):
  """
  Test method for comparing projected income this month to previous months.
  
  Args:
    lookup_data: Optional LookupUserDataOptimizer instance. If None, creates a new one.
    
  Returns:
    The generated response string
  """
  last_user_request = "Compare my projected income this month to my projected income last month, my projected income from two months ago, and my projected income from three months ago."
  return _run_test_with_logging(last_user_request, lookup_data)


def test_checking_account_sufficient_for_rent(lookup_data: LookupUserDataOptimizer = None):
  """
  Test method for checking if checking account has enough funds to pay rent next month.
  
  Args:
    lookup_data: Optional LookupUserDataOptimizer instance. If None, creates a new one.
    
  Returns:
    The generated response string
  """
  last_user_request = "Does my checking account have enough to pay for my rent next month?"
  return _run_test_with_logging(last_user_request, lookup_data)

def test_list_subscriptions_last_month(lookup_data: LookupUserDataOptimizer = None):
  """
  Test method for listing subscriptions paid last month (exercises new example).
  """
  last_user_request = "List my subscriptions from last month."
  return _run_test_with_logging(last_user_request, lookup_data)


def test_comprehensive_financial_health_overview(lookup_data: LookupUserDataOptimizer = None):
  """
  Test method for providing a comprehensive overview of current account balances, 
  recent income, and spending patterns to assess financial health.
  
  Args:
    lookup_data: Optional LookupUserDataOptimizer instance. If None, creates a new one.
    
  Returns:
    The generated response string
  """
  last_user_request = "Provide a comprehensive overview of current account balances, recent income, and spending patterns to assess financial health."
  return _run_test_with_logging(last_user_request, lookup_data)

def main(batch: int = 1):
  """
  Main function to test the lookup_data optimizer
  
  Args:
    batch: Batch number (1, 2, or 3) to determine which tests to run
  """
  if batch == 1:
    test_hows_my_accounts_doing()
    test_how_is_my_net_worth_doing_lately()
    test_analyze_income_and_spending_patterns_for_savings()
  elif batch == 2:
    test_compare_projected_income_across_months()
    test_checking_account_sufficient_for_rent()
  elif batch == 3:
    test_list_subscriptions_last_month()
    test_comprehensive_financial_health_overview()
  else:
    raise ValueError("batch must be 1, 2, or 3")


if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser(description='Run tests in batches')
  parser.add_argument('--batch', type=int, default=1, choices=[1, 2, 3],
                      help='Batch number to run (1, 2, or 3)')
  args = parser.parse_args()
  main(batch=args.batch)

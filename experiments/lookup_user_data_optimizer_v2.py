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
1. **Output**: Write `process_input() -> tuple[bool, dict]`. Keep code concise. Minimal comments.
2. **Tools**: Use `IMPLEMENTED_FUNCTIONS` & `IMPLEMENTED_DATE_FUNCTIONS`. `import datetime` is assumed.
3. **Matching**: Use partial case-insensitive matching for Account/Subscription names.
4. **Safety**: Always check `if df.empty: ...` before accessing data.
5. **Aggregations**: Use explicit `print()` loops for summarized data.

Today: |TODAY_DATE|.

<IMPLEMENTED_FUNCTIONS>

- `retrieve_depository_accounts() -> pd.DataFrame`: checking, savings. Cols: account_type, account_name, balance_available, balance_current
- `retrieve_credit_accounts() -> pd.DataFrame`: credit, loans. Cols: account_type, account_name, balance_available, balance_current, balance_limit
- `account_names_and_balances(df, template) -> tuple[str, list]`: Format account list.
- `utter_account_totals(df, template) -> str`: Sum balances and format.
- `utter_net_worth(total_assets, total_liabilities, template) -> str`: Format net worth. Placeholders: {net_worth_state_with_amount}, {total_asset_state_with_amount}, {total_liability_state_with_amount}
- `retrieve_income_transactions() -> pd.DataFrame`: Cols: date, transaction_name, amount, category
- `retrieve_spending_transactions() -> pd.DataFrame`: Cols: date, transaction_name, amount, category
- `transaction_names_and_amounts(df, template) -> tuple[str, list]`: List transactions. Placeholder: {amount_and_direction}
- `utter_spending_transaction_total(df, template) -> str`: Sum spending. Placeholder: {verb_and_total_amount}
- `utter_income_transaction_total(df, template) -> str`: Sum income. Placeholder: {verb_and_total_amount}
- `compare_spending(df, template, metadata=None) -> tuple[str, dict]`: Compare 2 groups. Placeholders: {difference}, {more_label}, {more_amount}, {less_label}, {less_amount}
- `retrieve_spending_forecasts(granularity='monthly') -> pd.DataFrame`: Cols: start_date, forecasted_amount, category
- `retrieve_income_forecasts(granularity='monthly') -> pd.DataFrame`: Cols: start_date, forecasted_amount, category
- `forecast_dates_and_amount(df, template) -> tuple[str, list]`: List forecasts.
- `utter_income_forecast_totals(df, template) -> str`: Sum income forecasts. Placeholders: {verb_and_total_amount}
- `utter_spending_forecast_totals(df, template) -> str`: Sum spending forecasts. Placeholders: {verb_and_total_amount}
- `utter_spending_forecast_amount(amount, template) -> str`: Format spending forecast.
- `utter_income_forecast_amount(amount, template) -> str`: Format income forecast.
- `utter_balance(amount, template) -> str`: Format balance/diff. Placeholder: {amount_with_direction}
- `utter_amount(amount, template) -> str`: Format absolute amount. Placeholder: {amount}
- `retrieve_subscriptions() -> pd.DataFrame`: Cols: transaction_name, amount, category, subscription_name, date
- `subscription_names_and_amounts(df, template) -> tuple[str, list]`: List subscriptions.
- `utter_subscription_totals(df, template) -> str`: Sum subscriptions.
- `respond_to_app_inquiry(inquiry) -> str`: For general Qs.

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

- `income` for salary, bonuses, interest, side hussles and business. Includes:
    - `income_salary` for regular paychecks and bonuses.
    - `income_sidegig` for side hussles like Uber, Etsy and other gigs.
    - `income_business` for business income and spending.
    - `income_interest` for interest income from savings or investments.
- `meals` for all types of food spending includes groceries, dining-out and delivered food.
    - `meals_groceries` for supermarkets and other unprepared food marketplaces.
    - `meals_dining_out` for food prepared outside the home like restaurants and takeout.
    - `meals_delivered_food` for prepared food delivered to the doorstep like DoorDash.
- `leisure` for all relaxation or recreation and travel activities.
    - `leisure_entertainment` for movies, concerts, cable and streaming services.
    - `leisure_travel`for flights, hotels, and other travel expenses.
- `bills` for essential payments for services and recurring costs.
    - `bills_connectivity` for internet and phone bills.
    - `bills_insurance` for life insurance and other insurance payments.
    - `bills_tax`  for income, state tax and other payments.
    - `bills_service_fees` for payments for services rendered like professional fees or fees for a product.
- `shelter` for all housing-related expenses including rent, mortgage, property taxes and utilities.
    - `shelter_home` for rent, mortgage, property taxes.
    - `shelter_utilities` for electricity, water, gas and trash utility bills.
    - `shelter_upkeep` for maintenance and repair and improvement costs for the home.
- `education` for all learning spending including kids after care and activities.
    - `education_kids_activities` for after school activities, sports and camps.
    - `education_tuition` for school tuition, daycare and other education fees.
- `shopping` for discretionary spending on clothes, electronics, home goods, etc.
    - `shopping_clothing` for clothing, shoes, accessories and other wearable items.
    - `shopping_gadgets` for electronics, gadgets, computers, software and other tech items.
    - `shopping_kids` for kids clothing, toys, school supplies and other kid-related items.
    - `shopping_pets` for pet food, toys, grooming, vet bills and other pet-related items.
- `transportation` for public transportation, car payments, gas and maintenance and car insurance.
    - `transportation_public` for bus, train, subway and other public transportation.
    - `transportation_car` for car payments, gas, maintenance and car insurance.
- `health` for medical bills, pharmacy spending, insurance, gym memberships and personal care.
    - `health_medical_pharmacy` for doctor visits, hospital, meds and health insurance costs.
    - `health_gym_wellness` for gym memberships, personal training and spa services.
    - `health_personal_care` for haircuts, beauty products and beuaty services.
- `donations_gifts` for charitable donations, gifts and other giving to friends and family.
- `uncategorized` for explicitly tagged as not yet categorized or unknown.
- `transfers` for moving money between accounts or paying off credit cards or loans.
- `miscellaneous` for explicitly tagged as miscellaneous.

</CATEGORY>

<EXAMPLES>
input: User: What is my net worth and checking balance?
output:
```python
def process_input():
    meta = {}
    assets = retrieve_depository_accounts()
    credit = retrieve_credit_accounts()
    if assets.empty and credit.empty: print("No accounts."); return True, meta

    tot_a = assets['balance_current'].sum() if not assets.empty else 0
    tot_l = credit['balance_current'].sum() if not credit.empty else 0
    print(utter_net_worth(tot_a, tot_l, "Net Worth: {net_worth_state_with_amount}"))

    if not assets.empty:
        chk = assets[assets['account_type'] == 'deposit_checking']
        if not chk.empty:
            print("Checking:")
            out, meta["accounts"] = account_names_and_balances(chk, "- {account_name}: {balance_available}")
            print(out)
    return True, meta
```

input: User: Compare dining out vs groceries spending last month.
output:
```python
def process_input():
    df = retrieve_spending_transactions()
    meta = {}
    if df.empty: print("No spending."); return True, meta

    today = datetime.now()
    start = get_start_of_month(get_after_periods(get_start_of_month(today), "monthly", -1))
    end = get_end_of_month(start)
    
    df = df[(df['date'] >= start) & (df['date'] <= end)]
    df = df[df['category'].isin(['meals_dining_out', 'meals_groceries'])]
    if df.empty: print("No relevant spending."); return True, meta
        
    res, meta = compare_spending(df, "Spent ${difference} more on {more_label} (${more_amount}) than {less_label} (${less_amount}).")
    print(res)
    return True, meta
```

input: User: Can I afford a $500 trip next month?
output:
```python
def process_input():
    meta = {}
    liq = retrieve_depository_accounts()
    cash = liq['balance_available'].sum() if not liq.empty else 0
    
    nxt = get_start_of_month(get_after_periods(get_start_of_month(datetime.now()), "monthly", 1))
    inc = retrieve_income_forecasts('monthly')
    sp = retrieve_spending_forecasts('monthly')
    
    inc_val = inc[inc['start_date'] == nxt]['forecasted_amount'].sum() if not inc.empty else 0
    sp_val = sp[sp['start_date'] == nxt]['forecasted_amount'].sum() if not sp.empty else 0
    
    surplus = inc_val - sp_val
    final = cash + surplus - 500
    
    c_str = utter_balance(cash, "{amount_with_direction}")
    s_str = utter_balance(surplus, "{amount_with_direction}")
    
    if final >= 0:
        print(f"Yes. Have {c_str}. Next month surplus {s_str}. After trip: ~${final:.0f}.")
    else:
        print(f"No. Have {c_str}. Next month surplus {s_str}. After trip: ${final:.0f}.")
    return True, meta
```

input: User: Savings last month vs this month?
output:
```python
def process_input():
    meta = {}
    today = datetime.now()
    cur = get_start_of_month(today)
    last = get_start_of_month(get_after_periods(cur, "monthly", -1))
    
    inc_act = retrieve_income_transactions()
    sp_act = retrieve_spending_transactions()
    
    i_last = inc_act[(inc_act['date'] >= last) & (inc_act['date'] <= get_end_of_month(last))]['amount'].sum() if not inc_act.empty else 0
    s_last = sp_act[(sp_act['date'] >= last) & (sp_act['date'] <= get_end_of_month(last))]['amount'].sum() if not sp_act.empty else 0
    sav_last = i_last - s_last
    
    inc_fc = retrieve_income_forecasts('monthly')
    sp_fc = retrieve_spending_forecasts('monthly')
    
    i_cur = inc_fc[inc_fc['start_date'] == cur]['forecasted_amount'].sum() if not inc_fc.empty else 0
    s_cur = sp_fc[sp_fc['start_date'] == cur]['forecasted_amount'].sum() if not sp_fc.empty else 0
    sav_cur = i_cur - s_cur
    
    print(f"Last month saved {utter_balance(sav_last, '{amount_with_direction}')}.")
    print(f"This month projected {utter_balance(sav_cur, '{amount_with_direction}')}.")
    print(f"Diff: {utter_balance(sav_cur - sav_last, '{amount_with_direction}')}.")
    return True, meta
```

</EXAMPLES>

<ACCOUNT_NAMES>

|ACCOUNT_NAMES|

</ACCOUNT_NAMES>

<SUBSCRIPTION_NAMES>

|SUBSCRIPTION_NAMES|

</SUBSCRIPTION_NAMES>
"""



class LookupUserDataOptimizer:
  """Handles all Gemini API interactions for financial planning and optimization"""
  
  def __init__(self, model_name="gemini-flash-lite-latest"):
    """Initialize the Gemini agent with API configuration for financial planning"""
    # API Configuration
    self.client = genai.Client(api_key=os.getenv('GEMINI_API_KEY'))
    
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
    success, captured_output, metadata, logs = sandbox.execute_agent_with_tools(result, user_id)
    
    print(f"Success: {success}")
    print()
    print("Captured Output:")
    print("-" * 80)
    print(captured_output)
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


def main():
  """Main function to test the lookup_data optimizer"""
  # test_hows_my_accounts_doing()
  # test_how_is_my_net_worth_doing_lately()
  test_analyze_income_and_spending_patterns_for_savings()
  test_compare_projected_income_across_months()
  test_checking_account_sufficient_for_rent()


if __name__ == "__main__":
  main()

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
1. Output only raw python code for `process_input() -> tuple[bool, str]`. Do not use markdown fences.
2. Use only `IMPLEMENTED_FUNCTIONS` and `IMPLEMENTED_DATE_FUNCTIONS`. `import datetime` is assumed.
3. Keep code concise and readable. Do not include code comments.
4. Use partial case-insensitive matching for account/subscription/merchant names when filtering by name.
5. Always guard DataFrame access with `if df.empty` checks.
6. Collect response lines in a list and return `chr(10).join(output_lines)`.
7. Whole-number fuzzy amount rule: if user references whole-number amount `X`, match `X - 0.50 <= amount <= X + 0.49` (inclusive). Keep amount sign; do not use absolute value.
8. Date range rule: for "past/next n months/weeks", always exclude current month/week.
9. Do not hardcode calendar dates. Use `today = datetime.now()` and derive ranges from date helpers.
10. For year-based ranges, use `get_after_periods(..., 'yearly', ...)` with `get_start_of_year`/`get_end_of_year`.
11. For discretionary spending, filter by `category` values starting with `meals`, `shopping`, `leisure`, plus exact `donations_gifts`.
12. Do not add unnecessary dataframe coercions (for example `pd.to_datetime`) unless required by explicit runtime errors.
13. After every retrieve_* call, handle `df.empty` before any column selection, filtering, or string operations.
14. Do not use `abs()` for amounts unless the user explicitly asks for absolute values.
15. Prefer helper formatters (`utter_transaction_total`, `utter_forecast_amount`, `utter_subscription_totals`) over manual amount-string formatting when applicable.
16. Build end-of-period dates from the computed start anchor when possible (for example `end = get_end_of_year(start)`).
17. Preserve requested granularity in responses (for example weekly query -> weekly breakdown, monthly query -> monthly breakdown).
18. If the user asks only for total spending/income, return totals with `utter_transaction_total` instead of manual amount strings.
19. If a user query closely matches an example intent, preserve the example’s filtering and date-range structure.
   - **Past n months**: Start = `get_start_of_month(get_after_periods(today, 'monthly', -n))`, End = `get_end_of_month(get_after_periods(today, 'monthly', -1))`.
   - **Past n weeks**: Start = `get_start_of_week(get_after_periods(today, 'weekly', -n))`, End = `get_end_of_week(get_after_periods(today, 'weekly', -1))`.
   - **Next n months**: Start = `get_start_of_month(get_after_periods(today, 'monthly', 1))`, End = `get_end_of_month(get_after_periods(today, 'monthly', n))`.
   - **Next n weeks**: Start = `get_start_of_week(get_after_periods(today, 'weekly', 1))`, End = `get_end_of_week(get_after_periods(today, 'weekly', n))`.

Today: |TODAY_DATE|.

<IMPLEMENTED_FUNCTIONS>

- `retrieve_depository_accounts() -> pd.DataFrame`: checking, savings. Cols: account_type, account_name, balance_available, balance_current
- `retrieve_credit_accounts() -> pd.DataFrame`: credit, loans. Cols: account_type, account_name, balance_available, balance_current, balance_limit
- `account_names_and_balances(df, template) -> str`: Format account list. Placeholders: any df col.  When listing accounts, include account_id in the template.
- `retrieve_income_transactions() -> pd.DataFrame`: Past income. Cols: date, transaction_name, amount, category, transaction_id, account_id
- `retrieve_spending_transactions() -> pd.DataFrame`: Past spending. Cols: date, transaction_name, amount, category, transaction_id, account_id
- `transaction_names_and_amounts(df, template) -> str`: List transactions. Placeholders: any df col.  When listing transactions, include account_id and transaction_id in the template.
- `utter_transaction_total(df, template) -> str`: Sum transactions then use correct placeholder for income or spending. Placeholders: {income_total_amount}, {spending_total_amount}. Example: "Total income: {income_total_amount}" or "Total spending: {spending_total_amount}"
- `retrieve_spending_forecasts(granularity='monthly') -> pd.DataFrame`: Future spending by week/month. Cols: start_date, forecasted_amount, category
- `retrieve_income_forecasts(granularity='monthly') -> pd.DataFrame`: Future income by week/month. Cols: start_date, forecasted_amount, category
- `forecast_dates_and_amount(df, template) -> str`: List forecasts. Placeholders: any df col.
- `utter_forecast_amount(amount, template) -> str`: Format forecast amount. Placeholders: {income_total_amount}, {spending_total_amount}. Example: "Total Expected Income: {income_total_amount}" or "Spending: {spending_total_amount}"
- `utter_absolute_amount(amount, template) -> str`: Format absolute amount. Placeholders: {amount}, {amount_with_direction}, {income_total_amount}, {spending_total_amount}
- `retrieve_subscriptions() -> pd.DataFrame`: Cols: transaction_name, amount, category, subscription_name, date
- `subscription_names_and_amounts(df, template) -> str`: List subscriptions. Placeholders: any df col.
- `utter_subscription_totals(df, template) -> str`: Sum subscriptions. Placeholders: {total_amount}. Example: "Total subscriptions: {total_amount}"

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
            output_lines.append(account_names_and_balances(chk, "Account '{account_name}' (account_id: {account_id}) has {balance_current} left with {balance_available} available now."))
    return True, chr(10).join(output_lines)
```

input: User: Can I afford $200 for concert tickets next week and what were my most recent fun expenses?
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
    output_lines.append(f"Next Week Expenses: {utter_absolute_amount(obligations, '{spending_total_amount}')}")
    
    if rem >= 0:
        output_lines.append(f"Yes. You'll have {utter_absolute_amount(rem, '{amount_with_direction}')} left.")
    else:
        output_lines.append(f"No. You'll be short by {utter_absolute_amount(rem, '{amount_with_direction}')}.")
    
    past_spending = retrieve_spending_transactions()
    if not past_spending.empty:
        fun_spending = past_spending[(past_spending['category'].str.startswith('leisure'))]
        if not fun_spending.empty:
            fun_spending = fun_spending.sort_values('date', ascending=False).head(5)
            output_lines.append("Recent fun spending:")
            output_lines.append(transaction_names_and_amounts(fun_spending, "- {transaction_name}: {amount} on {date} in {category} (account_id: {account_id}, transaction_id: {transaction_id})"))
        else:
            output_lines.append("No recent fun spending found.")
    else:
        output_lines.append("No past spending data available.")
    
    return True, chr(10).join(output_lines)
```

input: User: How many $50 AT&T payments have I had this year?
output:
```python
def process_input():
    output_lines = []
    start = get_start_of_year(datetime.now())
    end = get_end_of_year(start)

    sp = retrieve_spending_transactions()
    if sp.empty:
        return True, "No spending transactions found this year."

    sp = sp[(sp['date'] >= start) & (sp['date'] <= end)]
    if sp.empty:
        return True, "No spending transactions found this year."

    sp = sp[
        sp['transaction_name'].str.lower().str.contains('at&t', na=False) &
        (sp['amount'] >= 49.50) &
        (sp['amount'] <= 50.49)
    ]

    if sp.empty:
        return True, "There have been 0 $50 AT&T payments so far this year."

    output_lines.append(f"There have been {len(sp)} $50 AT&T payments so far this year.")
    output_lines.append(transaction_names_and_amounts(
        sp.sort_values('date'),
        "- {transaction_name}: {amount} on {date} (account_id: {account_id}, transaction_id: {transaction_id})",
    ))
    return True, chr(10).join(output_lines)
```

input: User: Compare my income and discretionary spending last year.
output:
```python
def process_input():
    start = get_start_of_year(get_after_periods(datetime.now(), 'yearly', -1))
    end = get_end_of_year(start)
    inc = retrieve_income_transactions()
    sp = retrieve_spending_transactions()
    inc = inc[(inc['date'] >= start) & (inc['date'] <= end)] if not inc.empty else pd.DataFrame()
    sp = sp[(sp['date'] >= start) & (sp['date'] <= end)] if not sp.empty else pd.DataFrame()
    sp = sp[sp['category'].str.startswith('meals') | sp['category'].str.startswith('shopping') | sp['category'].str.startswith('leisure') | (sp['category'] == 'donations_gifts')] if not sp.empty else pd.DataFrame()
    if inc.empty and sp.empty:
        return True, "No data last year."
    inc_total = inc['amount'].sum() if not inc.empty else 0
    sp_total = sp['amount'].sum() if not sp.empty else 0
    return True, chr(10).join([
        f"Income: {utter_absolute_amount(inc_total, '{amount}')}",
        f"Discretionary Spending: {utter_absolute_amount(sp_total, '{amount}')}",
        f"Net: {utter_absolute_amount(inc_total - sp_total, '{amount_with_direction}')}",
    ])
```
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
    self.top_k = 40
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

  
  def generate_response(self, last_user_request: str, user_id: int = 1) -> dict:
    """
    Generate a response using Gemini API for financial planning.
    
    Args:
      last_user_request: The last user request as a string
      user_id: User ID for building dynamic sections (default: 1)
      
    Returns:
      Dict with generated code and thought_summary
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
      top_k=self.top_k,
      temperature=self.temperature,
      top_p=self.top_p,
      max_output_tokens=self.max_output_tokens,
      safety_settings=self.safety_settings,
      system_instruction=[types.Part.from_text(text=full_system_prompt)],
      thinking_config=types.ThinkingConfig(
        thinking_budget=self.thinking_budget,
        include_thoughts=True,
      ),
    )

    # Generate response
    output_text = ""
    thought_summary = ""
    for chunk in self.client.models.generate_content_stream(
      model=self.model_name,
      contents=contents,
      config=generate_content_config,
    ):
      if chunk.text is not None:
        output_text += chunk.text
      if hasattr(chunk, "candidates") and chunk.candidates:
        for candidate in chunk.candidates:
          if hasattr(candidate, "content") and candidate.content:
            if hasattr(candidate.content, "parts") and candidate.content.parts:
              for part in candidate.content.parts:
                if hasattr(part, "thought") and part.thought:
                  if hasattr(part, "text") and part.text:
                    thought_summary += part.text

    return {
      "response": output_text,
      "thought_summary": thought_summary.strip(),
    }
  
  
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
  response_text = result["response"]
  thought_summary = result["thought_summary"]
  
  # Print the output
  print("=" * 80)
  print("LLM OUTPUT:")
  print("=" * 80)
  print(response_text)
  print("=" * 80)
  print()

  if thought_summary:
    print("-" * 80)
    print("THOUGHT SUMMARY:")
    print(thought_summary)
    print("-" * 80)
    print()
  
  # Execute the generated code in sandbox
  print("=" * 80)
  print("SANDBOX EXECUTION:")
  print("=" * 80)
  try:
    success, output_string, logs, goals_list = sandbox.execute_agent_with_tools(response_text, user_id)
    
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
  
  return response_text


TEST_CASES = [
  {
    "user": "List all my accounts.",
    "ideal_output": """def process_input():
    output_lines = []
    dep = retrieve_depository_accounts()
    cre = retrieve_credit_accounts()

    if dep.empty and cre.empty:
        return True, "No accounts found."

    if not dep.empty:
        output_lines.append("Depository Accounts:")
        output_lines.append(account_names_and_balances(dep, "- {account_name} (account_id: {account_id}): {balance_current}"))

    if not cre.empty:
        output_lines.append("Credit/Loan Accounts:")
        output_lines.append(account_names_and_balances(cre, "- {account_name} (account_id: {account_id}): {balance_current}"))

    return True, chr(10).join(output_lines)""",
  },
  {
    "user": "What is my net worth right now?",
    "ideal_output": """def process_input():
    output_lines = []
    dep = retrieve_depository_accounts()
    cre = retrieve_credit_accounts()

    if dep.empty and cre.empty:
        return True, "No accounts found."

    assets_total = dep['balance_current'].sum() if not dep.empty else 0
    liabilities_total = cre['balance_current'].sum() if not cre.empty else 0
    net_worth = assets_total - liabilities_total

    output_lines.append(f"Assets: {utter_absolute_amount(assets_total, '{amount_with_direction}')}")
    output_lines.append(f"Liabilities: {utter_absolute_amount(liabilities_total, '{amount_with_direction}')}")
    output_lines.append(f"Net Worth: {utter_absolute_amount(net_worth, '{amount_with_direction}')}")
    return True, chr(10).join(output_lines)""",
  },
  {
    "user": "How much did I spend on groceries in the past 2 months?",
    "ideal_output": """def process_input():
    start = get_start_of_month(get_after_periods(datetime.now(), 'monthly', -2))
    end = get_end_of_month(get_after_periods(datetime.now(), 'monthly', -1))
    sp = retrieve_spending_transactions()
    if sp.empty:
        return True, "No spending transactions found."

    sp = sp[(sp['date'] >= start) & (sp['date'] <= end)]
    sp = sp[sp['category'] == 'meals_groceries'] if not sp.empty else pd.DataFrame()
    if sp.empty:
        return True, "No grocery spending found in the past 2 months."
    return True, utter_transaction_total(sp, "Total grocery spending: {spending_total_amount}")""",
  },
  {
    "user": "Show my top 5 largest spending transactions from last month.",
    "ideal_output": """def process_input():
    start = get_start_of_month(get_after_periods(datetime.now(), 'monthly', -1))
    end = get_end_of_month(start)
    sp = retrieve_spending_transactions()
    if sp.empty:
        return True, "No spending transactions found."

    sp = sp[(sp['date'] >= start) & (sp['date'] <= end)]
    if sp.empty:
        return True, "No spending transactions found last month."

    top_sp = sp.sort_values('amount', ascending=False).head(5)
    output_lines = ["Top 5 largest spending transactions from last month:"]
    output_lines.append(transaction_names_and_amounts(top_sp, "- {transaction_name}: {amount} on {date} in {category} (account_id: {account_id}, transaction_id: {transaction_id})"))
    return True, chr(10).join(output_lines)""",
  },
  {
    "user": "How many $50 AT&T payments have I had this year?",
    "ideal_output": """def process_input():
    start = get_start_of_year(datetime.now())
    end = get_end_of_year(start)
    sp = retrieve_spending_transactions()
    if sp.empty:
        return True, "No spending transactions found."

    sp = sp[(sp['date'] >= start) & (sp['date'] <= end)]
    sp = sp[
        sp['transaction_name'].str.lower().str.contains('at&t', na=False) &
        (sp['amount'] >= 49.50) &
        (sp['amount'] <= 50.49)
    ] if not sp.empty else pd.DataFrame()
    return True, f"There have been {len(sp)} $50 AT&T payments so far this year." if not sp.empty else "There have been 0 $50 AT&T payments so far this year." """,
  },
  {
    "user": "List all subscriptions paid last month and total them.",
    "ideal_output": """def process_input():
    subs = retrieve_subscriptions()
    if subs.empty:
        return True, "No subscriptions found."

    start = get_start_of_month(get_after_periods(datetime.now(), 'monthly', -1))
    end = get_end_of_month(start)
    subs = subs[(subs['date'] >= start) & (subs['date'] <= end)]
    if subs.empty:
        return True, "No subscriptions paid last month."

    output_lines = ["Subscriptions paid last month:"]
    output_lines.append(subscription_names_and_amounts(subs, "- {subscription_name}: {amount} on {date}"))
    output_lines.append(utter_subscription_totals(subs, "Total: {total_amount}"))
    return True, chr(10).join(output_lines)""",
  },
  {
    "user": "What are my projected expenses for the next 3 months?",
    "ideal_output": """def process_input():
    sp = retrieve_spending_forecasts(granularity='monthly')
    if sp.empty:
        return True, "No spending forecasts found."

    today = datetime.now()
    start = get_start_of_month(get_after_periods(today, 'monthly', 1))
    end = get_end_of_month(get_after_periods(today, 'monthly', 3))
    sp = sp[(sp['start_date'] >= start) & (sp['start_date'] <= end)]
    if sp.empty:
        return True, "No spending forecasts for the next 3 months."

    output_lines = ["Projected expenses for the next 3 months:"]
    output_lines.append(forecast_dates_and_amount(sp.sort_values('start_date'), "- {start_date}: {forecasted_amount}"))
    output_lines.append(utter_forecast_amount(sp['forecasted_amount'].sum(), "Total projected spending: {spending_total_amount}"))
    return True, chr(10).join(output_lines)""",
  },
  {
    "user": "Project my savings for the next 4 weeks.",
    "ideal_output": """def process_input():
    output_lines = []
    today = datetime.now()
    inc = retrieve_income_forecasts('weekly')
    sp = retrieve_spending_forecasts('weekly')
    if inc.empty and sp.empty:
        return True, "No forecasts found."

    start_next = get_start_of_week(get_after_periods(today, 'weekly', 1))
    dates = [get_after_periods(start_next, 'weekly', i) for i in range(4)]
    total_savings = 0
    for d in dates:
        i_val = inc[inc['start_date'] == d]['forecasted_amount'].sum() if not inc.empty else 0
        s_val = sp[sp['start_date'] == d]['forecasted_amount'].sum() if not sp.empty else 0
        savings = i_val - s_val
        total_savings += savings
        output_lines.append(f"{get_date_string(d)}: {utter_absolute_amount(savings, '{amount_with_direction}')}")
    output_lines.append(f"Total projected savings: {utter_absolute_amount(total_savings, '{amount_with_direction}')}")
    return True, chr(10).join(output_lines)""",
  },
  {
    "user": "Compare my income and discretionary spending last year.",
    "ideal_output": """def process_input():
    start = get_start_of_year(get_after_periods(datetime.now(), 'yearly', -1))
    end = get_end_of_year(start)
    inc = retrieve_income_transactions()
    sp = retrieve_spending_transactions()

    inc = inc[(inc['date'] >= start) & (inc['date'] <= end)] if not inc.empty else pd.DataFrame()
    sp = sp[(sp['date'] >= start) & (sp['date'] <= end)] if not sp.empty else pd.DataFrame()
    if not sp.empty:
        sp = sp[sp['category'].str.startswith('meals') | sp['category'].str.startswith('shopping') | sp['category'].str.startswith('leisure') | (sp['category'] == 'donations_gifts')]
    if inc.empty and sp.empty:
        return True, "No data found last year."

    inc_total = inc['amount'].sum() if not inc.empty else 0
    sp_total = sp['amount'].sum() if not sp.empty else 0
    net = inc_total - sp_total
    return True, chr(10).join([
        f"Income: {utter_absolute_amount(inc_total, '{amount}')}",
        f"Discretionary Spending: {utter_absolute_amount(sp_total, '{amount}')}",
        f"Net: {utter_absolute_amount(net, '{amount_with_direction}')}",
    ])""",
  },
  {
    "user": "Do I have enough in checking to cover a $180 utility bill this week?",
    "ideal_output": """def process_input():
    dep = retrieve_depository_accounts()
    if dep.empty:
        return True, "No depository accounts found."

    chk = dep[dep['account_type'] == 'deposit_checking']
    if chk.empty:
        return True, "No checking account found."

    available = chk['balance_available'].sum()
    remaining = available - 180
    if remaining >= 0:
        return True, f"Yes. You can cover it and still have {utter_absolute_amount(remaining, '{amount_with_direction}')}"
    return True, f"No. You are short by {utter_absolute_amount(remaining, '{amount_with_direction}')}" """,
  },
  {
    "user": "Show my transportation spending in the past 4 weeks.",
    "ideal_output": """def process_input():
    start = get_start_of_week(get_after_periods(datetime.now(), 'weekly', -4))
    end = get_end_of_week(get_after_periods(datetime.now(), 'weekly', -1))
    sp = retrieve_spending_transactions()
    if sp.empty:
        return True, "No spending transactions found."

    sp = sp[(sp['date'] >= start) & (sp['date'] <= end)]
    sp = sp[sp['category'].str.startswith('transportation')] if not sp.empty else pd.DataFrame()
    if sp.empty:
        return True, "No transportation spending in the past 4 weeks."

    output_lines = ["Transportation spending in the past 4 weeks:"]
    output_lines.append(transaction_names_and_amounts(sp.sort_values('date', ascending=False), "- {transaction_name}: {amount} on {date} (account_id: {account_id}, transaction_id: {transaction_id})"))
    output_lines.append(utter_transaction_total(sp, "Total: {spending_total_amount}"))
    return True, chr(10).join(output_lines)""",
  },
  {
    "user": "Which subscriptions look like streaming services?",
    "ideal_output": """def process_input():
    subs = retrieve_subscriptions()
    if subs.empty:
        return True, "No subscriptions found."

    filtered = subs[subs['subscription_name'].str.lower().str.contains('netflix|disney|hulu|spotify|youtube|max|prime', na=False)]
    if filtered.empty:
        return True, "No streaming subscriptions found."

    output_lines = ["Streaming subscriptions:"]
    output_lines.append(subscription_names_and_amounts(filtered.sort_values('subscription_name'), "- {subscription_name}: {amount}"))
    return True, chr(10).join(output_lines)""",
  },
]


def _print_test_case_with_ideal_output(user_request: str, ideal_output: str):
  print("=" * 80)
  print(f"User: {user_request}")
  print("> Ideal_Output:")
  print(ideal_output)
  print("=" * 80)
  print()


def _run_catalog_case(test_case: dict, lookup_data: LookupUserDataOptimizer = None, user_id: int = None):
  _print_test_case_with_ideal_output(test_case["user"], test_case["ideal_output"])
  return _run_test_with_logging(test_case["user"], lookup_data, user_id)


def run_test_batch(batch: int = 1, lookup_data: LookupUserDataOptimizer = None, user_id: int = None):
  """
  Run test cases in batches to keep execution manageable.
  """
  batch_to_indices = {
    1: [0, 1, 2],
    2: [3, 4, 5],
    3: [6, 7, 8],
    4: [9, 10, 11],
  }
  if batch not in batch_to_indices:
    raise ValueError("batch must be 1, 2, 3, or 4")

  for idx in batch_to_indices[batch]:
    _run_catalog_case(TEST_CASES[idx], lookup_data, user_id)


def main(batch: int = 1):
  """
  Main function to test the lookup_data optimizer
  
  Args:
    batch: Batch number (1, 2, 3, or 4) to determine which tests to run
      - Batch 1: Basic account and net worth tests
      - Batch 2: Income comparison and rent checking tests
      - Batch 3: Subscription and comprehensive health tests
      - Batch 4: Date range tests (past/next months/weeks, excluding current)
  """
  run_test_batch(batch=batch)


if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser(description='Run tests in batches')
  parser.add_argument('--batch', type=int, default=1, choices=[1, 2, 3, 4],
                      help='Batch number to run (1, 2, 3, or 4)')
  args = parser.parse_args()
  main(batch=args.batch)

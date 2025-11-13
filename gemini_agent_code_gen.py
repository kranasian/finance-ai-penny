from google import genai
from google.genai import types
import os
import time
from typing import Dict, List, Optional
from dotenv import load_dotenv
from datetime import datetime
import sandbox
from database import Database

# Load environment variables
load_dotenv()

class GeminiAgentCodeGen:
  """Handles all Gemini API interactions for code generation"""
  
  def __init__(self, model_name="gemini-2.0-flash"):
    """Initialize the Gemini agent with API configuration for code generation"""
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
    self.thinking_budget_default = 2048
    
    # Safety Settings
    self.safety_settings = [
      types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"),
      types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"),
      types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"),
      types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="OFF")
    ]
    
    # System Prompt
    today_date = datetime.now().strftime("%Y-%m-%d")
    self.system_prompt = """Your name is "Penny" and you are a helpful AI specialized in code generation. **You only output python code.**
Write a function `process_input` that takes no arguments and print()s what to tell the user and returns a tuple:
  - The first element the boolean success or failure of the function.
  - The second element is the metadata for the entities created or retrieved.
- Assume `import datetime` is already included.  Use IMPLEMENTED_DATE_FUNCTIONS to compute for dates.
- When looking for `account_name` and `==`, look for other relevant variations to find more matches. Refer to <ACCOUNT_NAMES> for the list of account names.
- When looking for `subscription_name` and `==`, look for other relevant variations to find more matches. Refer to <SUBSCRIPTION_NAMES> for the list of subscription names.""" + f"""
- Today's date is {today_date}.""" + """

<AMOUNT_SIGN_CONVENTIONS>

**Amount sign conventions**:
- **Transactions and Forecasts**:
  - Income (money coming in): negative amounts = "received from"
  - Income outflow (refunds/returns): positive amounts = "returned to"
  - Spending (money going out): positive amounts = "paid to"
  - Spending inflow (refunds/returns): negative amounts = "refunded from"
- **Subscriptions** (always spending transactions):
  - Spending (money going out): positive amounts = "paid to"
  - Spending inflow (refunds/returns): negative amounts = "refunded from"

</AMOUNT_SIGN_CONVENTIONS>


<IMPLEMENTED_FUNCTIONS>

These functions are already implemented:

- `retrieve_depository_accounts(): pd.DataFrame`
  - retrieves depository accounts (checking, savings, money market) and returns a pandas DataFrame. It may be empty if no depository accounts exist.
  - DataFrame columns: `account_type` (str), `account_name` (str), `balance_available` (float), `balance_current` (float), `balance_limit` (float)
- `retrieve_credit_accounts(): pd.DataFrame`
  - retrieves credit card accounts and returns a pandas DataFrame. It may be empty if no credit card accounts exist.
  - DataFrame columns: `account_type` (str), `account_name` (str), `balance_available` (float), `balance_current` (float), `balance_limit` (float)
- `retrieve_loan_accounts(): pd.DataFrame`
  - retrieves loan accounts (mortgage, auto, home equity, line of credit) and returns a pandas DataFrame. It may be empty if no loan accounts exist.
  - DataFrame columns: `account_type` (str), `account_name` (str), `balance_available` (float), `balance_current` (float), `balance_limit` (float)
- `account_names_and_balances(df: pd.DataFrame, template: str) -> tuple[str, list]`
  - takes filtered `df` and generates a formatted string based on `template` and returns metadata.
  - Template placeholders: any column from the DataFrame
- `utter_account_totals(df: pd.DataFrame, template: str) -> str`
  - takes filtered `df` and calculates total balances and returns a formatted string based on `template`.
  - Template placeholders: any column from the DataFrame
- `utter_net_worth(total_assets: float, total_liabilities: float, template: str) -> str`
  - calculates net worth and returns a formatted string with state descriptions.
- `retrieve_income_transactions() -> pd.DataFrame`
  - retrieves income transactions and returns a pandas DataFrame. It may be empty if no income transactions exist.
  - DataFrame columns: `date` (datetime), `transaction_name` (str), `amount` (float), `category` (str)
- `retrieve_spending_transactions() -> pd.DataFrame`
  - retrieves spending transactions and returns a pandas DataFrame. It may be empty if no spending transactions exist.
  - DataFrame columns: `date` (datetime), `transaction_name` (str), `amount` (float), `category` (str)
- `transaction_names_and_amounts(df: pd.DataFrame, template: str) -> tuple[str, list]`
  - takes filtered `df` and generates a formatted string based on `template` and returns metadata.
- `utter_spending_transaction_total(df: pd.DataFrame, template: str) -> str`
  - takes filtered spending transaction `df` and `template` string, calculates total spending transaction amounts and returns a formatted string.
- `utter_income_transaction_total(df: pd.DataFrame, template: str) -> str`
  - takes filtered income transaction `df` and `template` string, calculates total income transaction amounts and returns a formatted string.
- `compare_spending(df: pd.DataFrame, template: str, metadata: dict = None) -> tuple[str, dict]`
  - compares spending between two categories or groups. If `df` has 'group' column, compares by groups; otherwise by category.
- `retrieve_spending_forecasts(granularity: str = 'monthly') -> pd.DataFrame`
  - retrieves spending forecasts from the database and returns a pandas DataFrame. May be empty if no forecasts exist.
  - `granularity` parameter can be 'monthly' or 'weekly' to specify forecast granularity.
  - DataFrame columns: `start_date` (datetime), `forecasted_amount` (float), `category` (str)
  - Returns only spending forecasts.
- `retrieve_income_forecasts(granularity: str = 'monthly') -> pd.DataFrame`
  - retrieves income forecasts from the database and returns a pandas DataFrame. May be empty if no forecasts exist.
  - `granularity` parameter can be 'monthly' or 'weekly' to specify forecast granularity.
  - DataFrame columns: `start_date` (datetime), `forecasted_amount` (float), `category` (str)
- `forecast_dates_and_amount(df: pd.DataFrame, template: str) -> tuple[str, list]`
  - takes filtered `df` and generates a formatted string based on `template` and returns metadata.
- `utter_income_forecast_totals(df: pd.DataFrame, template: str) -> str`
  - takes filtered income forecast `df` and calculates total income forecasted amounts and returns a formatted string based on `template`.
- `utter_spending_forecast_totals(df: pd.DataFrame, template: str) -> str`
  - takes filtered spending forecast `df` and calculates total spending forecasted amounts and returns a formatted string based on `template`.
- `retrieve_subscriptions() -> pd.DataFrame`
  - Returns a pandas DataFrame with subscription transaction data. May be empty if no subscription transactions exist.
  - DataFrame columns: `transaction_id` (int), `user_id` (int), `account_id` (int), `date` (datetime), `transaction_name` (str), `amount` (float), `category` (str), `subscription_name` (str), `confidence_score_bills` (float), `reviewer_bills` (str)
- `subscription_names_and_amounts(df: pd.DataFrame, template: str) -> tuple[str, list]`
  - takes filtered `df` and generates a formatted string based on `template` and returns metadata.
- `utter_subscription_totals(df: pd.DataFrame, template: str) -> str`
  - takes filtered `df` and `template` string, calculates total subscription transaction amounts and returns a formatted string.
  - The function automatically determines if transactions are income or spending based on the `category` column in the DataFrame.
- `respond_to_app_inquiry(inquiry: str) -> str`
  - accepts a string `inquiry` on how to categorize transactions, Penny's capabilities, or other app questions and returns a string response.

</IMPLEMENTED_FUNCTIONS>

<IMPLEMENTED_DATE_FUNCTIONS>

Only use these date functions that are already implemented.

- `get_date(year: int, month: int, day: int) -> datetime`
	- Returns a datetime object for the specified date.
- `get_start_of_month(date: datetime) -> datetime`
	- Returns the start of the month for a given date.
- `get_end_of_month(date: datetime) -> datetime`
	- Returns the end of the month for a given date.
- `get_start_of_year(date: datetime) -> datetime`
	- Returns the start of the year for a given date.
- `get_end_of_year(date: datetime) -> datetime`
	- Returns the end of the year for a given date.
- `get_start_of_week(date: datetime) -> datetime`
	- Returns the start of the week for a given date.
- `get_end_of_week(date: datetime) -> datetime`
	- Returns the end of the week for a given date.
- `get_after_periods(date: datetime, granularity: str, count: int) -> datetime`
	- Adds periods ("daily" | "weekly" | "monthly" | "yearly") and returns date.
- `get_date_string(date: datetime) -> str`
	- Returns date in "YYYY-MM-DD" format.

</IMPLEMENTED_DATE_FUNCTIONS>

<ACCOUNT_TYPE>

These are the valid `account_type` values.

- `deposit_savings`: savings, cash accounts
- `deposit_money_market`: money market accounts
- `deposit_checking`: checking, debit accounts
- `credit_card`: credit cards
- `loan_home_equity`: loans tied to home equity
- `loan_line_of_credit`: personal loans extended by banks
- `loan_mortgage`: home mortgage accounts
- `loan_auto`: auto loan accounts

</ACCOUNT_TYPE>

<CATEGORY>

These are the valid `category` values.

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
"""
    # - `retrieve_transactions(): pd.DataFrame`
    #     - retrieves all transactions componsed of income and expenses and returns a pandas DataFrame.  It may be empty if no transactions exist.
    #     - Panda's dataframe columns: `transaction_id`, `transaction_datetime`, `transaction_name`, `amount`, `category`, `account_id`
    #     - `transaction_datetime` is a datetime type.


    # - `create_reminder(title: str, reminder_datetime: str): tuple[str, dict | None]`
    #     - This function creates a reminder for `title` at `reminder_datetime`.
    #     - Returns a tuple:
    #       - The first element returns the failure message if the function fails, otherwise returns None.
    #       - The second element is the metadata for the entities created or retrieved.
    # - `update_reminder(reminder_id: int, new_title: str = None, new_reminder_datetime: str = None): tuple[str, dict | None]`
    #     - This function updates a reminder's title and/or reminder time.
    #     - Returns a tuple:
    #       - The first element returns the failure message if the function fails, otherwise returns None.
    #       - The second element is the metadata for the entities updated.
    # - `delete_reminder(reminder_id: int): tuple[str, dict | None]`
    #     - This function deletes a reminder by its ID.
    #     - Returns a tuple:
    #       - The first element returns the failure message if the function fails, otherwise returns None.
    #       - The second element is the metadata for the deleted reminder.

    # - `utter_delta_from_now(future_time: datetime) -> str`: Utter the time delta in a human-readable format
    # - `reminder_data(reminder: dict) -> dict`: Return reminders as a metadata formatted dict
    # - `turn_on_off_device(device_id: str, set_on: bool): tuple[str, dict | None]`
    #     - This function turns a home device (light or thermostat) on or off.
    #     - Returns a tuple:
    #       - The first element returns the failure message if the function fails, otherwise returns None.
    #       - The second element is the metadata for the device that was turned on/off.
    # - `retrieve_home_devices(): pd.DataFrame`
    #     - This function retrieves all home devices and returns a pandas DataFrame.  It may be empty if no devices exist.
    #     - Panda's dataframe columns: `device_id`, `name`, `device_type`, `room_name`, `user_id`, `is_on`, `brightness`, `color_name`, `target_temperature`
    #     - `device_id` is a str type, `is_on` is a boolean type, `brightness` is an integer (0-100), `target_temperature` is a float.

  
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
      return "<ACCOUNT_NAMES>\n  These are the `account_name` in the `accounts` table:\n    (No accounts found for this user)\n</ACCOUNT_NAMES>"
    
    account_names_list = []
    for account in accounts:
      account_name = account.get('account_name', '')
      if account_name:
        account_names_list.append(f"    - `{account_name}`")
    
    account_names_text = "\n".join(account_names_list) if account_names_list else "    (No account names found)"
    
    return f"""<ACCOUNT_NAMES>
  **REFERENCE ONLY** - These are example `account_name` values in the `accounts` table. This is NOT code - use `retrieve_depository_accounts()`, `retrieve_credit_accounts()`, or `retrieve_loan_accounts()` to get the actual DataFrame.
  Account names:
{account_names_text}
</ACCOUNT_NAMES>"""

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
      return "<SUBSCRIPTION_NAMES>\n  These are the `name` in the `user_recurring_transactions` table:\n    (No subscriptions found for this user)\n</SUBSCRIPTION_NAMES>"
    
    # Extract unique subscription names from transactions
    subscription_names_set = set()
    for transaction in subscription_transactions:
      subscription_name = transaction.get('subscription_name', '')
      if subscription_name:
        subscription_names_set.add(subscription_name)
    
    subscription_names_list = [f"    - `{name}`" for name in sorted(subscription_names_set)]
    
    subscription_names_text = "\n".join(subscription_names_list) if subscription_names_list else "    (No subscription names found)"
    
    return f"""<SUBSCRIPTION_NAMES>
  **REFERENCE ONLY** - These are example `name` values in the `user_recurring_transactions` table. This is NOT code - use `retrieve_subscriptions()` to get the actual DataFrame.
  Subscription names:
{subscription_names_text}
</SUBSCRIPTION_NAMES>"""

  def _create_few_shot_examples(self) -> str:
    """
    Create few-shot examples for code generation.
    
    Returns:
      List of example dictionaries with user input and expected code output
    """
    return """input: User: how much left in checking
output: ```python
def process_input():
    df = retrieve_depository_accounts()
    metadata = {}
    
    if df.empty:
      print("You have no depository accounts.")
      return True, metadata
    
    # Filter for checking accounts
    df = df[df['account_type'] == 'deposit_checking']
    
    if df.empty:
      print("You have no checking accounts.")
      return True, metadata
    
    print("Here are your checking account balances:")
    for_print, metadata["accounts"] = account_names_and_balances(df, "Account '{account_name}' has {balance_current} left with {balance_available} available now.")
    print(for_print)
    print(utter_account_totals(df, "Across all checking accounts, you have {balance_current} left."))
    
    return True, metadata
```

input: User: what is my net worth
output: ```python
def process_input():
    metadata = {}

    # Get depository accounts (assets)
    assets_df = retrieve_depository_accounts()
    
    # Get credit and loan accounts (liabilities)
    credit_df = retrieve_credit_accounts()
    loan_df = retrieve_loan_accounts()
    
    # Combine all accounts for checking if empty
    if assets_df.empty and credit_df.empty and loan_df.empty:
        print("You have no accounts to calculate net worth.")
        return True, metadata

    # Calculate totals
    total_assets = assets_df['balance_current'].sum() if not assets_df.empty else 0.0
    total_credit_liabilities = credit_df['balance_current'].sum() if not credit_df.empty else 0.0
    total_loan_liabilities = loan_df['balance_current'].sum() if not loan_df.empty else 0.0
    total_liabilities = total_credit_liabilities + total_loan_liabilities
    
    # Use utter_net_worth to format the message
    print(utter_net_worth(total_assets, total_liabilities, "You have a {net_worth_state_with_amount}, with a {total_asset_state_with_amount} and a {total_liability_state_with_amount}."))

    return True, metadata
```

input: User: did i spend more on dining out over groceries last month?
output: ```python
def process_input():
    df = retrieve_spending_transactions()
    metadata = {}
    
    if df.empty:
      print("You have no spending transactions.")
      return True, metadata
    
    # Filter for last month
    first_day_current_month = get_start_of_month(datetime.now())
    first_day_last_month = get_after_periods(first_day_current_month, granularity="monthly", count=-1)
    last_day_last_month = get_end_of_month(first_day_last_month)
    
    df = df[(df['date'] >= first_day_last_month) & (df['date'] <= last_day_last_month)]
    
    if df.empty:
      print("You have no spending transactions from last month.")
      return True, metadata
    
    # Filter for dining out and groceries categories
    df = df[df['category'].isin(['meals_dining_out', 'meals_groceries'])]
    
    if df.empty:
      print("You have no dining out or groceries transactions from last month.")
      return True, metadata
    
    categories = df['category'].unique()
    if len(categories) < 2:
      print(f"You only have transactions in one category: {categories[0]}")
      return True, metadata
    
    # Compare spending between categories
    result, metadata = compare_spending(df, 'You spent ${difference} more on {more_label} (${more_amount}, {more_count} transactions) over {less_label} (${less_amount}, {less_count} transactions).')
    print(result)
    
    return True, metadata
```

input: User: can I afford to pay a couple months of fun with what I have now
output: ```python
def process_input():
    metadata = {}
    
    # Get checking and savings account balances
    liquid_accounts_df = retrieve_depository_accounts()
    
    if liquid_accounts_df.empty:
      print("You have no checking or savings accounts.")
      return True, metadata
    
    # Calculate total available balance in liquid accounts
    current_balance = liquid_accounts_df['balance_available'].sum()
    
    # Get next 2 months
    first_day_current_month = get_start_of_month(datetime.now())
    next_month_start = get_start_of_month(get_after_periods(first_day_current_month, granularity="monthly", count=1))
    month_after_next_start = get_start_of_month(get_after_periods(first_day_current_month, granularity="monthly", count=2))
    
    # Get spending forecasts for the next couple of months
    spending_df = retrieve_spending_forecasts('monthly')
    
    # Filter for next 2 months if forecasts exist
    if not spending_df.empty:
      spending_df = spending_df[spending_df['start_date'].isin([next_month_start, month_after_next_start])]
      
      # Filter for fun/leisure categories
      fun_categories = ['leisure_entertainment', 'leisure_travel', 'leisure']
      fun_spending_df = spending_df[spending_df['category'].isin(fun_categories)]
      
      # Calculate total spending for fun activities in next 2 months
      total_spending = fun_spending_df['forecasted_amount'].sum() if not fun_spending_df.empty else 0.0
    else:
      total_spending = 0.0
    
    # Compare and determine affordability
    if current_balance < 0:
      if total_spending < 0:
        # Refund would reduce the deficit
        new_balance = current_balance - total_spending
        print(f"You have a negative balance of ${abs(current_balance):.0f} in your checking and savings accounts. However, your projected refunds of ${abs(total_spending):.0f} would reduce your deficit to ${abs(new_balance):.0f}.")
      else:
        # Additional spending would increase the deficit
        print(f"You have a negative balance of ${abs(current_balance):.0f} in your checking and savings accounts. You cannot afford additional spending. Your projected total spending is ${total_spending:.0f}, which would increase your deficit to ${abs(current_balance - total_spending):.0f}.")
    elif current_balance >= total_spending:
      print(f"You can afford a couple months of fun. Your checking and savings accounts have ${current_balance:.0f} available. Your projected total spending is ${total_spending:.0f}, leaving you with ${current_balance - total_spending:.0f} remaining.")
    else:
      print(f"You cannot afford a couple months of fun. Your checking and savings accounts have ${current_balance:.0f} available. However, your projected total spending is ${total_spending:.0f}, so you would need ${total_spending - current_balance:.0f} more.")
    
    return True, metadata
```

input: User: Have I been saving anything monthly in the past 4 months?
output: ```python
def process_input():
    metadata = {}
    
    # Get income and spending transactions
    income_df = retrieve_income_transactions()
    spending_df = retrieve_spending_transactions()
    
    if income_df.empty and spending_df.empty:
      print("You have no transactions.")
      return True, metadata
    
    # Get current month start
    first_day_current_month = get_start_of_month(datetime.now())
    
    # Calculate savings for each of the past 4 months
    months_with_savings = []
    months_without_savings = []
    
    for i in range(1, 5):  # Past 4 months (1, 2, 3, 4 months ago)
      month_start = get_start_of_month(get_after_periods(first_day_current_month, granularity="monthly", count=-i))
      month_end = get_end_of_month(month_start)
      
      # Filter transactions for this month
      month_income_df = income_df[(income_df['date'] >= month_start) & (income_df['date'] <= month_end)]
      month_expenses_df = spending_df[(spending_df['date'] >= month_start) & (spending_df['date'] <= month_end)]
      
      if month_income_df.empty and month_expenses_df.empty:
        continue
      
      # Calculate income and expenses for this month
      total_income = month_income_df['amount'].sum() if not month_income_df.empty else 0.0
      total_expenses = month_expenses_df['amount'].sum() if not month_expenses_df.empty else 0.0
      savings = total_income + total_expenses
      
      # Check if savings were positive (saving money)
      month_name = month_start.strftime('%B %Y')
      if savings < 0:  # Negative savings means positive net (income > expenses)
        months_with_savings.append((month_name, abs(savings)))
      else:
        months_without_savings.append((month_name, savings))
    
    # Format and print results
    if months_with_savings:
      savings_list = ", ".join([f"{month} (${amount:.0f})" for month, amount in months_with_savings])
      print(f"Yes, you have been saving monthly in {len(months_with_savings)} of the past 4 months: {savings_list}.")
    else:
      print("No, you have not been saving monthly in any of the past 4 months.")
    
    if months_without_savings:
      no_savings_list = ", ".join([f"{month} (spent ${amount:.0f} more than earned)" for month, amount in months_without_savings])
      print(f"Months without savings: {no_savings_list}.")
    
    return True, metadata
```

input: User: how much am i expected to save next week?
output: ```python
def process_input():
    metadata = {}
    
    # Get next week dates
    start_of_next_week = get_after_periods(datetime.now(), granularity="weekly", count=1)
    
    # Retrieve income and spending forecasts for next week
    income_df = retrieve_income_forecasts('weekly')
    spending_df = retrieve_spending_forecasts('weekly')
    
    if income_df.empty and spending_df.empty:
      print("You have no forecasts for next week.")
      return True, metadata
    
    # Filter for next week (start_date matches start of next week)
    if not income_df.empty:
      income_df = income_df[income_df['start_date'] == start_of_next_week]
    if not spending_df.empty:
      spending_df = spending_df[spending_df['start_date'] == start_of_next_week]
    if income_df.empty and spending_df.empty:
      print("You have no forecasts for next week.")
      return True, metadata
    
    # Calculate totals for expected savings
    total_income = income_df['forecasted_amount'].sum() if not income_df.empty else 0.0
    total_spending = spending_df['forecasted_amount'].sum() if not spending_df.empty else 0.0
    expected_savings = total_income - total_spending
    
    # Format messages using forecast totals
    income_msg = utter_income_forecast_totals(income_df, "${total_amount}")
    expenses_msg = utter_spending_forecast_totals(spending_df, "${total_amount}")
    
    # Format and print expected savings message
    if expected_savings > 0:
      print(f"You are expected to save ${expected_savings:.0f} next week. Your forecasted income is {income_msg} and your forecasted spending is {expenses_msg}.")
    elif expected_savings < 0:
      print(f"You are expected to spend ${abs(expected_savings):.0f} more than you earn next week. Your forecasted income is {income_msg} and your forecasted spending is {expenses_msg}.")
    else:
      print(f"You are expected to break even next week. Your forecasted income is {income_msg} and your forecasted spending is {expenses_msg}.")
    
    return True, metadata
```

input: User: Did i get any income in last last few weeks and what about upcoming weeks?
output: ```python
def process_input():
    metadata = {"transactions": [], "forecasts": []}
    
    # Check past few weeks (transactions)
    income_transactions_df = retrieve_income_transactions()
    
    if income_transactions_df.empty:
      print("You have no income transactions.")
    else:
      # Filter for past few weeks (last 3 weeks)
      start_of_current_week = get_start_of_week(datetime.now())
      start_of_three_weeks_ago = get_after_periods(start_of_current_week, granularity="weekly", count=-3)
      past_income_df = income_transactions_df[(income_transactions_df['date'] >= start_of_three_weeks_ago) & (income_transactions_df['date'] < start_of_current_week)]
      
      if past_income_df.empty:
        print("You did not receive any income in the past few weeks.")
      else:
        print("Here is your income from the past few weeks:")
        for_print, metadata["transactions"] = transaction_names_and_amounts(past_income_df, "{amount_and_direction} {transaction_name} on {date}.")
        print(for_print)
        print(utter_income_transaction_total(past_income_df, "In total, you {total_amount_and_verb} from the past few weeks."))
    
    # Check upcoming weeks (forecasts)
    print("\nUpcoming weeks:")
    income_forecasts_df = retrieve_income_forecasts('weekly')
    
    if income_forecasts_df.empty:
      print("You have no income forecasts for upcoming weeks.")
    else:
      # Filter for next few weeks (next 3 weeks)
      start_of_next_week = get_after_periods(datetime.now(), granularity="weekly", count=1)
      start_of_four_weeks_ahead = get_after_periods(datetime.now(), granularity="weekly", count=4)
      upcoming_income_df = income_forecasts_df[(income_forecasts_df['start_date'] >= start_of_next_week) & (income_forecasts_df['start_date'] < start_of_four_weeks_ahead)]
      
      if upcoming_income_df.empty:
        print("You have no income forecasts for the upcoming weeks.")
      else:
        print("Here is your forecasted income for upcoming weeks:")
        for_print, metadata["forecasts"] = forecast_dates_and_amount(upcoming_income_df, "{amount_and_direction} {category} on {start_date}.")
        print(for_print)
        print(utter_income_forecast_totals(upcoming_income_df, "In total, you are expected to {verb_and_total_amount} in upcoming weeks."))
    
    return True, metadata
```

input: User: check my checking account if i can afford paying my rent next month
output: ```python
def process_input():
    metadata = {}
    
    # Get checking account balance
    depository_df = retrieve_depository_accounts()
    
    if depository_df.empty:
      print("You have no depository accounts.")
      return True, metadata
    
    # Filter for checking account
    checking_df = depository_df[depository_df['account_type'] == 'deposit_checking']
    
    if checking_df.empty:
      print("You have no checking account.")
      return True, metadata
    
    # Get total available balance from checking accounts
    total_available = checking_df['balance_available'].sum()
    
    # Get next month date
    first_day_current_month = get_start_of_month(datetime.now())
    next_month_start_date = get_start_of_month(get_after_periods(first_day_current_month, granularity="monthly", count=1))
    
    # Retrieve spending forecasts for next month
    spending_df = retrieve_spending_forecasts('monthly')
    
    if spending_df.empty:
      print("You have no spending forecasts for next month.")
      return True, metadata
    
    # Filter for next month
    spending_df = spending_df[spending_df['start_date'] == next_month_start_date]
    
    if spending_df.empty:
      print("You have no spending forecasts for next month.")
      return True, metadata
    
    rent_df = spending_df[spending_df['category'] == 'shelter_home']
    
    if rent_df.empty:
      print("You have no rent forecast for next month.")
      return True, metadata
    
    # Calculate total forecasted rent
    total_rent = rent_df['forecasted_amount'].sum()
    
    # Compare and determine affordability
    if total_available >= total_rent:
      print(f"You can afford your rent next month. Your checking account has ${total_available:.0f} available, and your forecasted rent is ${total_rent:.0f}. You would have ${total_available - total_rent:.0f} remaining.")
    else:
      print(f"You cannot afford your rent next month. Your checking account has ${total_available:.0f} available, but your forecasted rent is ${total_rent:.0f}. You would need ${total_rent - total_available:.0f} more.")
    
    return True, metadata
```

input: User: list my subscriptions
output: ```python
def process_input():
    metadata = {"subscriptions": []}
    
    subscriptions_df = retrieve_subscriptions()
    
    if subscriptions_df.empty:
      print("You have no subscriptions.")
      return True, metadata
    
    for_print, metadata["subscriptions"] = subscription_names_and_amounts(subscriptions_df, '{amount_and_direction} {subscription_name} on {date}.')
    transaction_count = len(subscriptions_df)
    print(f"Your subscriptions ({transaction_count} transaction{'s' if transaction_count != 1 else ''}):")
    print(for_print)
    
    print(utter_subscription_totals(subscriptions_df, 'Total subscription transactions: ${total_amount:.0f} {direction}'))
    
    return True, metadata
```

input: User: list streaming subscriptions paid last month
output: ```python
def process_input():
    metadata = {"subscriptions": []}
    
    subscriptions_df = retrieve_subscriptions()
    
    if subscriptions_df.empty:
      print("You have no subscriptions.")
      return True, metadata
    
    # Filter for streaming subscriptions: use subscription_name AND category
    # Populate using relevant names from SUBSCRIPTION_NAMES
    streaming_names = []
    streaming_categories = ['leisure_entertainment']
    
    name_matches = subscriptions_df['subscription_name'].str.lower().isin(streaming_names)
    category_matches = subscriptions_df['category'].isin(streaming_categories)
    streaming_df = subscriptions_df[name_matches & category_matches]
    
    if streaming_df.empty:
      print("You have no streaming subscriptions.")
      return True, metadata
      
    # Filter for last month
    first_day_current_month = get_start_of_month(datetime.now())
    first_day_last_month = get_after_periods(first_day_current_month, granularity="monthly", count=-1)
    last_day_last_month = get_end_of_month(first_day_last_month)
    
    streaming_df = streaming_df[(streaming_df['date'] >= first_day_last_month) & (streaming_df['date'] <= last_day_last_month)]
    
    if streaming_df.empty:
      print("You have no streaming subscription payments last month.")
      return True, metadata
    
    for_print, metadata["subscriptions"] = subscription_names_and_amounts(streaming_df, '{amount_and_direction} {subscription_name} on {date}')
    transaction_count = len(streaming_df)
    print(f"Your streaming subscription payments last month ({transaction_count} transaction{'s' if transaction_count != 1 else ''}):")
    print(for_print)
    
    print(utter_subscription_totals(streaming_df, 'Total streaming subscription spending last month: ${total_amount:.0f} {direction}'))
    
    return True, metadata
```

"""




  def generate_response(self, messages: List[Dict], timing_data: Dict, user_id: int = 1) -> Dict:
    """
    Generate a response using Gemini API with timing tracking for code generation.
    Uses GenAI API to construct the prompt with few-shot examples.
    
    Args:
      messages: The user/assistant messages
      timing_data: Dictionary to store timing information
      user_id: User ID for sandbox execution
      
    Returns:
      Dictionary with response text and timing data
    """
    # Filter messages from the last 1 minute (60 seconds)
    current_time = time.time()
    recent_messages = []
    
    for msg in messages:
      # Check if message has request_time and is within 60 seconds
      if "request_time" in msg and (current_time - msg["request_time"]) <= 60:
        recent_messages.append(msg)
      # If no request_time, include it (backward compatibility)
      elif "request_time" not in msg:
        recent_messages.append(msg)
    
    # Format recent messages with role prefixes
    formatted_messages = []
    for msg in recent_messages:
      role = msg.get('role', 'user')
      content = msg.get('content', '')
      if role == 'user':
        formatted_messages.append(f"User: {content}")
      elif role == 'assistant':
        formatted_messages.append(f"Assistant: {content}")
    
    # Join all recent messages
    recent_conversation = "\n".join(formatted_messages)
    print(recent_conversation)
    
    gemini_start = time.time()
    
    # Create few-shot examples and request text
    few_shot_examples = self._create_few_shot_examples()
    request_text = types.Part.from_text(text=f"""<EXAMPLES>
{few_shot_examples}
</EXAMPLES>

input: {recent_conversation}
output:""")
    
    # Create content and configuration
    contents = [types.Content(role="user", parts=[request_text])]
    
    # Build dynamic ACCOUNT_NAMES section for this user
    account_names_section = self._build_account_names_section(user_id)
    # Build dynamic SUBSCRIPTION_NAMES section for this user
    subscription_names_section = self._build_subscription_names_section(user_id)
    # Combine system prompt with dynamic account names and subscription names
    full_system_prompt = self.system_prompt + "\n\n" + account_names_section + "\n\n" + subscription_names_section
    
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
    output_tokens = 0
    last_chunk = None
    for chunk in self.client.models.generate_content_stream(
      model=self.model_name,
      contents=contents,
      config=generate_content_config,
    ):
      if chunk.text is not None:
        output_text += chunk.text
      last_chunk = chunk
    
    # Extract usage metadata from the last chunk if available
    # The usage metadata is typically only available in the last chunk of a streaming response
    if last_chunk:
      # Try different ways to access usage metadata
      if hasattr(last_chunk, 'usage_metadata') and last_chunk.usage_metadata:
        output_tokens = getattr(last_chunk.usage_metadata, 'output_token_count', 0) or getattr(last_chunk.usage_metadata, 'candidates_token_count', 0)
      elif hasattr(last_chunk, 'candidates') and last_chunk.candidates:
        # Check if usage metadata is in candidates
        for candidate in last_chunk.candidates:
          if hasattr(candidate, 'usage_metadata') and candidate.usage_metadata:
            output_tokens = getattr(candidate.usage_metadata, 'output_token_count', 0) or getattr(candidate.usage_metadata, 'candidates_token_count', 0)
            break
    
    gemini_end = time.time()
    
    # Store output tokens in timing data
    timing_data['output_tokens'] = output_tokens
    
    # Execute the generated code in sandbox
    try:
      success, utter, metadata, logs = sandbox.execute_agent_with_tools(output_text, user_id)
    except Exception as e:
      # Extract logs from error message if available
      error_str = str(e)
      logs = ""
      if "Captured logs:" in error_str:
        logs = error_str.split("Captured logs:")[-1].strip()
      success = False
      utter = f"Error executing code: {error_str}"
      metadata = {"error": error_str}
    
    execution_end = time.time()
    
    # Record timing data
    timing_data['gemini_api_calls'].append({
      'call_number': 1,
      'start_time': gemini_start,
      'end_time': gemini_end,
      'duration_ms': (gemini_end - gemini_start) * 1000
    })
    timing_data['execution_time'].append({
      'call_number': 1,
      'start_time': gemini_end,
      'end_time': execution_end,
      'duration_ms': (execution_end - gemini_end) * 1000
    })
    
    return {
      'response': utter,
      'function_called': None,
      'execution_success': success,
      'execution_metadata': metadata,
      'code_generated': output_text,
      'logs': logs
    }

  
  def generate_response_with_function_calling(self, messages: List[Dict], timing_data: Dict, 
                                            function_handler: callable, user_id: int = 1) -> Dict:
    """
    Generate a response using Gemini API for code generation.
    Note: This method doesn't use function calling for code generation.
    
    Args:
      messages: The user/assistant messages
      timing_data: Dictionary to store timing information
      function_handler: Function to handle function calls (not used in code gen)
      user_id: User ID for sandbox execution
      
    Returns:
      Dictionary with response text and timing data
    """
    return self.generate_response(messages, timing_data, user_id)
  
  def get_available_models(self) -> List[str]:
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


def create_gemini_agent_code_gen(model_name="gemini-2.0-flash"):
  """Create a new Gemini agent code gen instance with the specified model"""
  return GeminiAgentCodeGen(model_name)

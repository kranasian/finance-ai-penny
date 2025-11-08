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
    self.system_prompt = """Your name is "Penny" and you are a helpful AI specialized in code generation. **You only output python code.**
Write a function `process_input` that takes no arguments and print()s what to tell the user and returns a tuple:
  - The first element the boolean success or failure of the function.
  - The second element is the metadata for the entities created or retrieved.
- Compute for dates using `datetime` package.  Assume `import datetime` is already included.
- When looking for `account_name` and `==`, look for other relevant variations to find more matches. Refer to <ACCOUNT_NAMES> for the list of account names.
- When looking for `subscription_name` and `==`, look for other relevant variations to find more matches. Refer to <SUBSCRIPTION_NAMES> for the list of subscription names.
- When creating goals that require an `account_id` (credit_X_amount, save_X_amount, credit_0, save_0), if multiple accounts of the same type exist and the user didn't specify which account or said "all", handle as follows:
  - If user explicitly mentions "all" (e.g., "all my credit cards", "all credit cards"), create separate goals for each matching account.
  - If multiple accounts exist but user didn't specify which one and didn't say "all", set `clarification_needed` in the goal dict to ask which account they want, and set `account_id` to None.
  - If only one account of that type exists, use that account's `account_id`.
- Today's date is {today_date}.

<AMOUNT_SIGN_CONVENTIONS>
  **Amount sign conventions** (applies to all transaction, forecast, and subscription functions):
  - Income (money coming in): negative amounts = "earned"
  - Income outflow (refunds/returns): positive amounts = "outflow" (for subscriptions) or "refunded" (for transactions/forecasts)
  - Spending (money going out): positive amounts = "spent"
  - Spending inflow (refunds/returns): negative amounts = "inflow" (for subscriptions) or "received" (for transactions/forecasts)
</AMOUNT_SIGN_CONVENTIONS>

<IMPLEMENTED_FUNCTIONS>
  These functions are already implemented:
    - `retrieve_accounts(): pd.DataFrame`
        - retrieves all accounts and returns a pandas DataFrame.  It may be empty if no accounts exist.
        - DataFrame columns: `account_id` (int), `account_type` (str), `account_name` (str), `balance_available` (float), `balance_current` (float), `balance_limit` (float)
        - **Note**: `account_id` is for metadata only (e.g., filtering, joining), not for display to users. Use `account_name` for user-facing output.
    - `account_names_and_balances(df: pd.DataFrame, template: str) -> tuple[str, list]`
        - takes filtered `df` and generates a formatted string based on `template` and returns metadata.
        - Template placeholders: any column from the DataFrame (e.g., `[account_name]`, `[balance_current]`, `[balance_available]`, `[account_type]`)
    - `utter_account_totals(df: pd.DataFrame, template: str) -> str`
        - takes filtered `df` and calculates total balances and returns a formatted string based on `template`.
        - Template placeholders: any column from the DataFrame (e.g., `[balance_current]`, `[balance_available]`)
    - `retrieve_transactions() -> pd.DataFrame`
        - retrieves all transactions and returns a pandas DataFrame.  It may be empty if no transactions exist.
        - DataFrame columns: `transaction_id` (int), `user_id` (int), `account_id` (int), `date` (datetime), `transaction_name` (str), `amount` (float), `category` (str), `ai_category_id` (int)
        - **Note**: `transaction_id` and `account_id` are for metadata only (e.g., filtering, joining, returning in metadata), not for display to users. Use `transaction_name` and `account_name` (from accounts DataFrame) for user-facing output.
        - **Category definitions**: 
          - "Spending" refers to all transactions with non-income categories (expense categories), regardless of whether the amount is positive or negative.
          - "Earning" (or "Income") refers to all transactions with income categories, regardless of whether the amount is positive or negative.
    - `transaction_names_and_amounts(df: pd.DataFrame, template: str) -> tuple[str, list]`
        - takes filtered `df` and generates a formatted string based on `template` and returns metadata.
        - Template placeholders: any column from the DataFrame (e.g., `[transaction_name]`, `[amount]`, `[date]`, `[category]`, `[direction]`, `[account_name]`)
        - `[direction]` already contains the verb: "earned" or "refunded" for income, "spent" or "received" for expenses.
    - `utter_transaction_totals(df: pd.DataFrame, template: str) -> str`
        - takes filtered `df` and `template` string, calculates total transaction amounts and returns a formatted string.
        - The function automatically determines if transactions are income or spending based on the `category` column in the DataFrame.
        - Template placeholders: `[total_amount]`, `[direction]`
        - `[direction]` already contains the verb: "earned" or "refunded" for income, "spent" or "received" for expenses.
    - `compare_spending(df: pd.DataFrame, template: str, metadata: dict = None) -> tuple[str, dict]`
        - compares spending between two categories or groups. If `df` has 'group' column, compares by groups; otherwise by category.
    - `retrieve_spending_forecasts(granularity: str = 'monthly') -> pd.DataFrame`
        - retrieves spending forecasts from the database and returns a pandas DataFrame. May be empty if no forecasts exist.
        - `granularity` parameter can be 'monthly' or 'weekly' to specify forecast granularity.
        - DataFrame columns: `user_id` (int), `ai_category_id` (int), `month_date` (datetime, if monthly) or `sunday_date` (datetime, if weekly), `forecasted_amount` (float), `category` (str)
        - `month_date` is in YYYY-MM-DD format with day always 01 (first day of the month).
        - `sunday_date` is the Sunday (start) date of the week.
        - Returns only spending forecasts (excludes income category IDs: 36, 37, 38, 39).
    - `retrieve_income_forecasts(granularity: str = 'monthly') -> pd.DataFrame`
        - retrieves income forecasts from the database and returns a pandas DataFrame. May be empty if no forecasts exist.
        - `granularity` parameter can be 'monthly' or 'weekly' to specify forecast granularity.
        - DataFrame columns: `user_id` (int), `ai_category_id` (int), `month_date` (datetime, if monthly) or `sunday_date` (datetime, if weekly), `forecasted_amount` (float), `category` (str)
        - `month_date` is in YYYY-MM-DD format with day always 01 (first day of the month).
        - `sunday_date` is the Sunday (start) date of the week.
    - `forecast_dates_and_amount(df: pd.DataFrame, template: str) -> tuple[str, list]`
        - takes filtered `df` and generates a formatted string based on `template` and returns metadata.
        - Template placeholders: any column from the DataFrame (e.g., `[date]`, `[amount]`, `[forecasted_amount]`, `[direction]`, `[category]`, `[ai_category_id]`, `[month_date]`, `[sunday_date]`)
        - `[direction]` already contains the verb: "earned" or "refunded" for income categories, "spent" or "received" for expense categories. Do NOT add verbs before `[direction]` in templates.
    - `utter_forecasts(df: pd.DataFrame, template: str) -> str`
        - takes filtered `df` and calculates total forecasted amounts and returns a formatted string based on `template`.
        - Template placeholders: `[total_amount]`, `[amount]`, `[forecasted_amount]`, `[direction]`, `[month_date]`, `[sunday_date]`, `[category_count]`, `[categories]`
        - `[direction]` already contains the verb: "earned" or "refunded" for income categories, "spent" or "received" for expense categories. Do NOT add verbs before `[direction]` in templates.
    - `retrieve_subscriptions() -> pd.DataFrame`
        - Returns a pandas DataFrame with subscription transaction data. May be empty if no subscription transactions exist.
        - DataFrame columns: `transaction_id` (int), `user_id` (int), `account_id` (int), `date` (datetime), `transaction_name` (str), `amount` (float), `category` (str), `subscription_name` (str), `confidence_score_bills` (float), `reviewer_bills` (str)
    - `subscription_names_and_amounts(df: pd.DataFrame, template: str) -> tuple[str, list]`
        - takes filtered `df` and generates a formatted string based on `template` and returns metadata.
        - Template placeholders: any column from the DataFrame (e.g., `[subscription_name]`, `[transaction_name]`, `[amount]`, `[date]`, `[category]`, `[direction]`)
        - `[direction]` already contains the verb: "earned" or "refunded" for income categories, "spent" or "received" for expense categories.
    - `utter_subscription_totals(df: pd.DataFrame, template: str) -> str`
        - takes filtered `df` and `template` string, calculates total subscription transaction amounts and returns a formatted string.
        - The function automatically determines if transactions are income or spending based on the `category` column in the DataFrame.
        - Template placeholders: `[total_amount]`, `[direction]`
        - `[direction]` will be blank (empty string) for "earned" or "spent", and will show "(inflow)" or "(outflow)" for refunds/returns.
    - `respond_to_app_inquiry(inquiry: str) -> str`
        - accepts a string `inquiry` on how to categorize transactions, Penny's capabilities, or other app questions and returns a string response.
    - `create_goal(goals: list[dict]) -> tuple[str, dict]`
        - creates spending budgets or goals. Accepts a list of goal dictionaries.
        - Each goal dict should contain:
          - `type`: "category", "credit_X_amount", "save_X_amount", "credit_0", or "save_0"
          - `granularity`: "weekly", "monthly", or "yearly"
          - `title`: Goal title/name
          - `amount`: Target dollar amount (>= 0)
          - `start_date`: Start date in YYYY-MM-DD format
          - `end_date`: End date in YYYY-MM-DD format (defaults to "2099-12-31")
          - `description`: Goal description string
          - `category`: Raw category text (for type="category")
          - `match_category`: Official category name (for type="category")
          - `account_id`: Account ID integer (for credit_X_amount/save_X_amount/credit_0/save_0)
          - `percent`: Target percent 0-100 (only valid for credit_X_amount/save_X_amount)
          - `match_caveats`: Matching constraints explanation
          - `clarification_needed`: Clarification prompt if needed
        - Goal types: "category" (requires match_category), "credit_X_amount"/"save_X_amount" (requires account_id), "credit_0"/"save_0" (requires account_id).
        - Returns tuple: (str response message with caveats, success message, and goal descriptions, dict metadata with goals list).
</IMPLEMENTED_FUNCTIONS>

<IMPLEMENTED_DATE_FUNCTIONS>
  These functions are already implemented:
    - `get_date(year: int, month: int, day: int) -> datetime`: Returns a datetime object for the specified date.
    - `get_start_of_month(date: datetime) -> datetime`: Returns the start of the month for a given date.
    - `get_end_of_month(date: datetime) -> datetime`: Returns the end of the month for a given date.
    - `get_start_of_year(date: datetime) -> datetime`: Returns the start of the year for a given date.
    - `get_end_of_year(date: datetime) -> datetime`: Returns the end of the year for a given date.
    - `get_start_of_week(date: datetime) -> datetime`: Returns the start of the week for a given date.
    - `get_end_of_week(date: datetime) -> datetime`: Returns the end of the week for a given date.
    - `get_after_periods(date: datetime, granularity: str, count: int) -> datetime`: Adds periods ("daily" | "weekly" | "monthly" | "yearly") and returns date.
    - `get_date_string(date: datetime) -> str`: Returns date in "YYYY-MM-DD" format.
</IMPLEMENTED_DATE_FUNCTIONS>

<ACCOUNT_TYPE>
  These are the `account_type` in the `accounts` table:
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
  These are the `category` in the `transactions` table:
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
  **REFERENCE ONLY** - These are example `account_name` values in the `accounts` table. This is NOT code - use `retrieve_accounts()` to get the actual DataFrame.
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
    df = retrieve_accounts()
    metadata = {"accounts": []}
    
    if df.empty:
      print("You have no accounts.")
    else:
      # Filter for checking accounts
      df = df[df['account_type'] == 'deposit_checking']
      
      if df.empty:
        print("You have no checking accounts.")
      else:
        print("Here are your checking account balances:")
        for_print, metadata["accounts"] = account_names_and_balances(df, "Account '[account_name]' has [balance_current] left with [balance_available] available now.")
        print(for_print)
        print(utter_account_totals(df, "Across all checking accounts, you have [balance_current] left."))
    
    return True, metadata
```

input: User: what is my net worth
output: ```python
def process_input():
    accounts_df = retrieve_accounts()
    metadata = {}

    if accounts_df.empty:
        print("You have no accounts to calculate net worth.")
    else:
        # List of asset account types for net worth calculation
        asset_types = ['deposit_savings', 'deposit_money_market', 'deposit_checking']
        liability_types = ['credit_card', 'loan_home_equity', 'loan_line_of_credit', 'loan_mortgage', 'loan_auto']

        # Filter for assets and liabilities
        assets_df = accounts_df[accounts_df['account_type'].isin(asset_types)]
        liabilities_df = accounts_df[accounts_df['account_type'].isin(liability_types)]

        total_assets = assets_df['balance_current'].sum()
        total_liabilities = liabilities_df['balance_current'].sum()
        # net worth is the sum of assets minus liabilities
        net_worth = total_assets - total_liabilities
        print(f"You have a net worth of ${net_worth:,.0f} with assets of ${total_assets:,.0f} and liabilities of ${total_liabilities:,.0f}.")

    return True, metadata
```

input: User: did i spend more on dining out over groceries last month?
output: ```python
def process_input():
    df = retrieve_transactions()
    metadata = {}
    
    if df.empty:
      print("You have no transactions.")
    else:
      # Filter for last month
      first_day_current_month = get_start_of_month(datetime.now())
      first_day_last_month = get_after_periods(first_day_current_month, granularity="monthly", count=-1)
      last_day_last_month = get_end_of_month(first_day_last_month)
      
      df = df[(df['date'] >= first_day_last_month) & (df['date'] <= last_day_last_month)]
      
      if df.empty:
        print("You have no transactions from last month.")
      else:
        # Filter for dining out and groceries categories
        df = df[df['category'].isin(['meals_dining_out', 'meals_groceries'])]
        
        if df.empty:
          print("You have no dining out or groceries transactions from last month.")
        else:
          categories = df['category'].unique()
          if len(categories) < 2:
            print(f"You only have transactions in one category: {categories[0]}")
          else:
            # Compare spending between categories
            result, metadata = compare_spending(df, 'You spent $[difference] more on [more_label] ($[more_amount], [more_count] transactions) over [less_label] ($[less_amount], [less_count] transactions).')
            print(result)
    
    return True, metadata
```

input: User: check my checking account if i can afford paying my dining out last month
output: ```python
def process_input():
    metadata = {"accounts": []}
    
    # Get checking account balance
    accounts_df = retrieve_accounts()
    checking_df = accounts_df[accounts_df['account_type'] == 'deposit_checking']
    
    if checking_df.empty:
      print("You have no checking accounts.")
      return True, metadata
    
    for_print, metadata["accounts"] = account_names_and_balances(checking_df, "Account '[account_name]' has [balance_current] left with [balance_available] available now.")
    
    # Calculate total available balance in checking accounts
    total_available = checking_df['balance_available'].sum()
    
    # Get dining out transactions from last month
    transactions_df = retrieve_transactions()
    
    if transactions_df.empty:
      print("You have no transactions.")
      return True, metadata
    
    # Filter for last month
    first_day_current_month = get_start_of_month(datetime.now())
    first_day_last_month = get_after_periods(first_day_current_month, granularity="monthly", count=-1)
    last_day_last_month = get_end_of_month(first_day_last_month)
    
    transactions_df = transactions_df[(transactions_df['date'] >= first_day_last_month) & (transactions_df['date'] <= last_day_last_month)]
    
    if transactions_df.empty:
      print("You have no transactions from last month.")
      return True, metadata
    
    # Filter for dining out (spending is negative)
    dining_out_df = transactions_df[transactions_df['category'] == 'meals_dining_out']
    
    if dining_out_df.empty:
      print("You have no dining out transactions from last month.")
      return True, metadata
    
    total_dining_out = dining_out_df['amount'].sum()
    
    # Compare and determine affordability
    if total_available >= total_dining_out:
      print(f"You can afford your dining out expenses from last month. Your checking account has ${total_available:,.2f} available, and your dining out spending was ${total_dining_out:,.2f}. You would have ${total_available - total_dining_out:,.2f} remaining.")
    else:
      print(f"You cannot afford your dining out expenses from last month. Your checking account has ${total_available:,.2f} available, but your dining out spending was ${total_dining_out:,.2f}. You would need ${total_dining_out - total_available:,.2f} more.")
    
    return True, metadata
```

input: User: how much did i save last month?
output: ```python
def process_input():
    metadata = {}
    
    # Get transactions from last month
    transactions_df = retrieve_transactions()
    
    if transactions_df.empty:
      print("You have no transactions.")
      return True, metadata
    
    # Filter for last month
    first_day_current_month = get_start_of_month(datetime.now())
    first_day_last_month = get_after_periods(first_day_current_month, granularity="monthly", count=-1)
    last_day_last_month = get_end_of_month(first_day_last_month)
    
    transactions_df = transactions_df[(transactions_df['date'] >= first_day_last_month) & (transactions_df['date'] <= last_day_last_month)]
    
    if transactions_df.empty:
      print("You have no transactions from last month.")
      return True, metadata
    
    # Calculate income (filter by income categories)
    income_categories = ['income_salary', 'income_sidegig', 'income_business', 'income_interest']
    income_df = transactions_df[transactions_df['category'].isin(income_categories)]
    total_income = income_df['amount'].sum()
    
    # Calculate expenses (all non-income transactions, use absolute value)
    expenses_df = transactions_df[~transactions_df['category'].isin(income_categories)]
    total_expenses = expenses_df['amount'].sum()
    
    # Calculate savings
    savings = total_income - total_expenses
    
    # Get formatted income and spending messages using utter_transaction_totals
    income_msg = utter_transaction_totals(income_df, "[direction] [total_amount:,.2f]") if not income_df.empty else "$0.00"
    expenses_msg = utter_transaction_totals(expenses_df, "[direction] [total_amount:,.2f]") if not expenses_df.empty else "$0.00"
    
    # Format and print savings message
    if savings < 0:
      print(f"You saved ${abs(savings):,.2f} last month. Your income was {income_msg} and your expenses were {expenses_msg}.")
    elif savings > 0:
      print(f"You spent ${savings:,.2f} more than you earned last month. Your income was {income_msg} and your expenses were {expenses_msg}.")
    else:
      print(f"You broke even last month. Your income was {income_msg} and your expenses were {expenses_msg}.")
    
    return True, metadata
```

input: User: list income past 2 weeks
output: ```python
def process_input():
    metadata = {"transactions": []}
    
    # Get transactions
    df = retrieve_transactions()
    
    if df.empty:
      print("You have no transactions.")
      return True, metadata
    
    # Filter for income categories
    income_categories = ['income_salary', 'income_sidegig', 'income_business', 'income_interest']
    df = df[df['category'].isin(income_categories)]
    
    if df.empty:
      print("You have no income transactions.")
      return True, metadata
    
    # Filter for past 2 weeks (from 2 weeks ago to now)
    start_of_current_week = get_start_of_week(datetime.now())
    start_of_two_weeks_ago = get_after_periods(start_of_current_week, granularity="weekly", count=-2)
    df = df[(df['date'] >= start_of_two_weeks_ago) & (df['date'] < start_of_current_week)]
    
    if df.empty:
      print("You have no income transactions from the past 2 weeks.")
      return True, metadata
    
    print("Here are your income transactions from the past 2 weeks:")
    for_print, metadata["transactions"] = transaction_names_and_amounts(df, "[transaction_name]: [direction] $[amount:,.2f] on [date]")
    print(for_print)
    print(utter_transaction_totals(df, "In total, you [direction] [total_amount] from the past 2 weeks."))
    
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
    
    # Filter for next week (sunday_date matches start of next week)
    if not income_df.empty:
      income_df = income_df[income_df['sunday_date'] == start_of_next_week]
    if not spending_df.empty:
      spending_df = spending_df[spending_df['sunday_date'] == start_of_next_week]
    if income_df.empty and spending_df.empty:
      print("You have no forecasts for next week.")
      return True, metadata
    
    # Calculate expected savings
    income_total = income_df['forecasted_amount'].sum() if not income_df.empty else 0.0
    spending_total = spending_df['forecasted_amount'].sum() if not spending_df.empty else 0.0
    expected_savings = income_total - spending_total
    
    # Get formatted income and spending messages using utter_forecasts
    income_msg = utter_forecasts(income_df, "[total_amount:,.2f]") if not income_df.empty else "$0.00"
    expenses_msg = utter_forecasts(spending_df, "[total_amount:,.2f]") if not spending_df.empty else "$0.00"
    
    # Format and print expected savings message
    if expected_savings > 0:
      print(f"You are expected to save ${expected_savings:,.2f} next week. Your forecasted income is {income_msg} and your forecasted spending is {expenses_msg}.")
    elif expected_savings < 0:
      print(f"You are expected to spend ${abs(expected_savings):,.2f} more than you earn next week. Your forecasted income is {income_msg} and your forecasted spending is {expenses_msg}.")
    else:
      print(f"You are expected to break even next week. Your forecasted income is {income_msg} and your forecasted spending is {expenses_msg}.")
    
    return True, metadata
```

input: User: how much am i expected to save next month?
output: ```python
def process_input():
    metadata = {}
    
    # Get next month date
    first_day_current_month = get_start_of_month(datetime.now())
    first_day_next_month = get_after_periods(first_day_current_month, granularity="monthly", count=1)
    next_month_date = first_day_next_month.replace(day=1)
    
    # Retrieve income and spending forecasts for next month
    income_df = retrieve_income_forecasts('monthly')
    spending_df = retrieve_spending_forecasts('monthly')
    
    if income_df.empty and spending_df.empty:
      print("You have no forecasts for next month.")
      return True, metadata
    
    # Filter for next month
    if not income_df.empty:
      income_df = income_df[income_df['month_date'] == next_month_date]
    if not spending_df.empty:
      spending_df = spending_df[spending_df['month_date'] == next_month_date]
    if income_df.empty and spending_df.empty:
      print("You have no forecasts for next month.")
      return True, metadata
    
    # Calculate expected savings
    income_total = income_df['forecasted_amount'].sum() if not income_df.empty else 0.0
    spending_total = spending_df['forecasted_amount'].sum() if not spending_df.empty else 0.0
    expected_savings = income_total - spending_total
    
    # Get formatted income and spending messages using utter_forecasts
    income_msg = utter_forecasts(income_df, "[total_amount:,.2f]") if not income_df.empty else "$0.00"
    expenses_msg = utter_forecasts(spending_df, "[total_amount:,.2f]") if not spending_df.empty else "$0.00"
    
    # Format and print expected savings message
    if expected_savings > 0:
      print(f"You are expected to save ${expected_savings:,.2f} next month. Your forecasted income is {income_msg} and your forecasted spending is {expenses_msg}.")
    elif expected_savings < 0:
      print(f"You are expected to spend ${abs(expected_savings):,.2f} more than you earn next month. Your forecasted income is {income_msg} and your forecasted spending is {expenses_msg}.")
    else:
      print(f"You are expected to break even next month. Your forecasted income is {income_msg} and your forecasted spending is {expenses_msg}.")
    
    return True, metadata
```

input: User: check my checking account if i can afford paying my rent next month
output: ```python
def process_input():
    metadata = {}
    
    # Get checking account balance
    accounts_df = retrieve_accounts()
    
    if accounts_df.empty:
      print("You have no accounts.")
      return True, metadata
    
    # Filter for checking account
    checking_df = accounts_df[accounts_df['account_type'] == 'deposit_checking']
    
    if checking_df.empty:
      print("You have no checking account.")
      return True, metadata
    
    # Get total available balance from checking accounts
    total_available = checking_df['balance_available'].sum()
    
    # Get next month date
    first_day_current_month = get_start_of_month(datetime.now())
    first_day_next_month = get_after_periods(first_day_current_month, granularity="monthly", count=1)
    next_month_date = first_day_next_month.replace(day=1)
    
    # Retrieve spending forecasts for next month
    spending_df = retrieve_spending_forecasts('monthly')
    
    if spending_df.empty:
      print("You have no spending forecasts for next month.")
      return True, metadata
    
    # Filter for next month
    spending_df = spending_df[spending_df['month_date'] == next_month_date]
    
    if spending_df.empty:
      print("You have no spending forecasts for next month.")
      return True, metadata
    
    rent_df = spending_df[spending_df['ai_category_id'] == 14]
    
    if rent_df.empty:
      print("You have no rent forecast for next month.")
      return True, metadata
    
    # Calculate total forecasted rent
    total_rent = rent_df['forecasted_amount'].sum()
    
    # Compare and determine affordability
    if total_available >= total_rent:
      print(f"You can afford your rent next month. Your checking account has ${total_available:,.2f} available, and your forecasted rent is ${total_rent:,.2f}. You would have ${total_available - total_rent:,.2f} remaining.")
    else:
      print(f"You cannot afford your rent next month. Your checking account has ${total_available:,.2f} available, but your forecasted rent is ${total_rent:,.2f}. You would need ${total_rent - total_available:,.2f} more.")
    
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
    
    for_print, metadata["subscriptions"] = subscription_names_and_amounts(subscriptions_df, '[subscription_name]: [direction] $[amount:,.2f] on [date]')
    transaction_count = len(subscriptions_df)
    print(f"Your subscriptions ({transaction_count} transaction{'s' if transaction_count != 1 else ''}):")
    print(for_print)
    
    print(utter_subscription_totals(subscriptions_df, 'Total subscription transactions: $[total_amount:,.2f] [direction]'))
    
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
    
    for_print, metadata["subscriptions"] = subscription_names_and_amounts(streaming_df, '[subscription_name]: [direction] $[amount:,.2f] on [date]')
    transaction_count = len(streaming_df)
    print(f"Your streaming subscription payments last month ({transaction_count} transaction{'s' if transaction_count != 1 else ''}):")
    print(for_print)
    
    print(utter_subscription_totals(streaming_df, 'Total streaming subscription spending last month: $[total_amount:,.2f] [direction]'))
    
    return True, metadata
```

input: User: create a budget for $60 gas every week for the next 6 months
output: ```python
def process_input():
    metadata = {"goals": []}
    
    # Get start of week today
    start_date = get_start_of_week(datetime.now())
    
    # Calculate end date: 6 months from start, then get Saturday of that week
    end_date = get_after_periods(start_date, granularity="monthly", count=6)
    end_date = get_end_of_week(end_date)
    
    goals = [{
        "type": "category",
        "granularity": "weekly",
        "title": "Weekly Gas ⛽",
        "amount": 60.0,
        "start_date": get_date_string(start_date),
        "end_date": get_date_string(end_date),
        "category": "gas",
        "match_category": "transportation_car",
        "match_caveats": "Matching gas to overall car expenses.",
        "clarification_needed": None,
        "description": "Created $60 Weekly Gas ⛽ from 2025-01-05 to 2025-07-05."
    }]
    
    response, goal_metadata = create_goal(goals)
    
    print(response)
    
    if goal_metadata and isinstance(goal_metadata, dict) and "goals" in goal_metadata:
        metadata["goals"] = goal_metadata["goals"]
    
    return True, metadata
```

input: User: Pay $200 weekly on my BoFa credit card
output: ```python
def process_input():
    metadata = {"goals": []}
    
    # Retrieve all accounts to check for credit cards
    accounts_df = retrieve_accounts()
    
    # Filter for credit cards
    credit_cards_df = accounts_df[accounts_df['account_type'] == 'credit_card']
    
    if credit_cards_df.empty:
        print("You don't have any credit cards.")
        return True, metadata
    
    # Populate using relevant names from ACCOUNT_NAMES
    bofa_names = []

    bofa_cards_df = credit_cards_df[
        credit_cards_df['account_name'].str.lower().isin(bofa_names)
    ]
    
    if bofa_cards_df.empty:
        print("You don't have any BoFa credit cards.")
        return True, metadata
    
    # Multiple Amex credit cards exist but user didn't specify which one
    # Check if user mentioned "all" in their request - if not, ask for clarification
    if len(bofa_cards_df) > 1:
        card_names = bofa_cards_df['account_name'].tolist()
        goals = [{
            "type": "credit_X_amount",
            "granularity": "weekly",
            "title": "Weekly Credit Card Payment",
            "amount": 200.0,
            "start_date": get_date_string(get_start_of_week(datetime.now())),
            "end_date": "2099-12-31",
            "account_id": None,
            "clarification_needed": f"You have multiple BoFa credit cards: {', '.join(card_names)}. Which BoFa credit card would you like to pay $200 weekly on?",
            "description": None
        }]
        
        response, goal_metadata = create_goal(goals)
        print(response)
        return True, metadata
    
    # Single BoFa credit card found
    card = bofa_cards_df.iloc[0]
    start_date = get_start_of_week(datetime.now())
    title = f"Weekly Payment - {card['account_name']}"
    amount = 200.0
    
    goals = [{
        "type": "credit_X_amount",
        "granularity": "weekly",
        "title": title,
        "amount": amount,
        "start_date": get_date_string(start_date),
        "end_date": "2099-12-31",
        "account_id": int(card['account_id']),
        "description": f"Created ${amount:.0f} {title} from {get_date_string(start_date)} to 2099-12-31."
    }]
    
    response, goal_metadata = create_goal(goals)
    print(response)
    
    if goal_metadata and isinstance(goal_metadata, dict) and "goals" in goal_metadata:
        metadata["goals"] = goal_metadata["goals"]
    
    return True, metadata
```

input: User: Pay $200 weekly on all my credit cards
output: ```python
def process_input():
    metadata = {"goals": []}
    
    # Retrieve all accounts to find credit cards
    accounts_df = retrieve_accounts()
    
    # Filter for credit cards
    credit_cards_df = accounts_df[accounts_df['account_type'] == 'credit_card']
    
    if credit_cards_df.empty:
        print("You don't have any credit cards.")
        return True, metadata
    
    # Create goals for all credit cards
    goals = []
    start_date = get_start_of_week(datetime.now())
    
    for _, card in credit_cards_df.iterrows():
        title = f"Weekly Payment - {card['account_name']}"
        amount = 200.0
        goals.append({
            "type": "credit_X_amount",
            "granularity": "weekly",
            "title": title,
            "amount": amount,
            "start_date": get_date_string(start_date),
            "end_date": "2099-12-31",
            "account_id": int(card['account_id']),
            "description": f"Created ${amount:.0f} {title} from {get_date_string(start_date)} to 2099-12-31."
        })
    
    response, goal_metadata = create_goal(goals)
    print(response)
    
    if goal_metadata and isinstance(goal_metadata, dict) and "goals" in goal_metadata:
        metadata["goals"] = goal_metadata["goals"]
    
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
    # Add today's date to the system prompt
    today_date = datetime.now().strftime("%Y-%m-%d")
    system_prompt_with_date = self.system_prompt.format(today_date=today_date)
    # Combine system prompt with dynamic account names and subscription names
    full_system_prompt = system_prompt_with_date + "\n\n" + account_names_section + "\n\n" + subscription_names_section
    
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

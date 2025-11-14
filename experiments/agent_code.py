#!/usr/bin/env python3
"""
Agent Code Experiment - Demonstrates usage of create_reminder and retrieve_reminder tools
"""

import sys
import os
import datetime
from datetime import datetime, timedelta
import pandas as pd


# Add the parent directory to the path so we can import the tools
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
  sys.path.insert(0, parent_dir)

from database import Database
from user_seeder import seed_users
from penny.tool_funcs.retrieve_accounts import (
    retrieve_depository_accounts_function_code_gen,
    retrieve_credit_accounts_function_code_gen,
    account_names_and_balances,
    utter_account_totals,
    utter_net_worth
)
from penny.tool_funcs.retrieve_transactions import (
    retrieve_income_transactions_function_code_gen,
    retrieve_spending_transactions_function_code_gen,
    transaction_names_and_amounts,
    utter_spending_transaction_total,
    utter_income_transaction_total
)
from penny.tool_funcs.compare_spending import compare_spending
from penny.tool_funcs.retrieve_forecasts import retrieve_spending_forecasts_function_code_gen, retrieve_income_forecasts_function_code_gen
from penny.tool_funcs.forecast_utils import utter_amount, utter_income_forecast_totals, utter_spending_forecast_totals, forecast_dates_and_amount, utter_spending_forecast_amount, utter_income_forecast_amount, utter_balance
from penny.tool_funcs.retrieve_subscriptions import retrieve_subscriptions_function_code_gen, subscription_names_and_amounts, utter_subscription_totals
from penny.tool_funcs.date_utils import get_start_of_month, get_end_of_month, get_start_of_week, get_end_of_week, get_after_periods, get_date_string
user_id = 1

def retrieve_depository_accounts() -> pd.DataFrame:
  global user_id
  return retrieve_depository_accounts_function_code_gen(user_id=user_id)


def retrieve_credit_accounts() -> pd.DataFrame:
  global user_id
  return retrieve_credit_accounts_function_code_gen(user_id=user_id)


def retrieve_income_transactions() -> pd.DataFrame:
  global user_id
  return retrieve_income_transactions_function_code_gen(user_id=user_id)


def retrieve_spending_transactions() -> pd.DataFrame:
  global user_id
  return retrieve_spending_transactions_function_code_gen(user_id=user_id)


def retrieve_spending_forecasts(granularity: str = 'monthly') -> pd.DataFrame:
  global user_id
  return retrieve_spending_forecasts_function_code_gen(user_id=user_id, granularity=granularity)


def retrieve_income_forecasts(granularity: str = 'monthly') -> pd.DataFrame:
  global user_id
  return retrieve_income_forecasts_function_code_gen(user_id=user_id, granularity=granularity)


def retrieve_subscriptions() -> pd.DataFrame:
  global user_id
  return retrieve_subscriptions_function_code_gen(user_id=user_id)


def utter_delta_from_now(future_time: datetime) -> str:
  """Utter the time delta in a human-readable format"""
  time_delta = future_time - datetime.now()
  # support singular and plural for days, hours, and minutes
  if time_delta.days == 1:
    return f"a day"
  elif time_delta.days > 1:
    return f"{time_delta.days} days"
  elif time_delta.total_seconds() >= 3600:  # 1 hour or more
    hours = int(time_delta.total_seconds() // 3600)
    if hours == 1:
      return f"an hour"
    else:
      return f"{hours} hours"
  elif time_delta.total_seconds() >= 60:  # 1 minute or more
    minutes = int(time_delta.total_seconds() // 60)
    if minutes == 1:
      return f"a minute"
    else:
      return f"{minutes} minutes"
  else:
    return f"soon"


def process_input_how_much_left_in_checking():
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


def process_input_what_is_my_net_worth():
    metadata = {}

    # Get depository accounts (assets)
    assets_df = retrieve_depository_accounts()
    
    # Get credit and loan accounts (liabilities)
    credit_df = retrieve_credit_accounts()
    
    # Check if empty
    if assets_df.empty and credit_df.empty:
        print("You have no accounts to calculate net worth.")
        return True, metadata

    # Calculate totals
    total_assets = assets_df['balance_current'].sum() if not assets_df.empty else 0.0
    total_liabilities = credit_df['balance_current'].sum() if not credit_df.empty else 0.0
    
    # Use utter_net_worth to format the message
    print(utter_net_worth(total_assets, total_liabilities, "You have a {net_worth_state_with_amount}, with a {total_asset_state_with_amount} and a {total_liability_state_with_amount}."))

    return True, metadata

def process_input_did_i_spend_more_on_dining_out_over_groceries_last_month():
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
    result, _ = compare_spending(df, 'You spent ${difference} more on {more_label} (${more_amount}, {more_count} transactions) over {less_label} (${less_amount}, {less_count} transactions).')
    print(result)
    
    return True, metadata


def process_input_can_i_afford_to_pay_a_couple_months_of_fun_with_what_i_have_now():
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
    
    # Calculate remaining balance after spending
    remaining_balance = current_balance - total_spending
    
    # Format amounts using utter_balance
    current_balance_str = utter_balance(current_balance, "{amount_with_direction}")
    total_spending_str = utter_spending_forecast_amount(total_spending, "{amount_and_direction}")
    remaining_balance_str = utter_balance(remaining_balance, "{amount_with_direction}")
    
    # Determine affordability and format message
    if current_balance < 0:
      if total_spending < 0:
        # Refund would reduce the deficit
        print(f"You have {current_balance_str} in your checking and savings accounts. However, your projected refunds of {total_spending_str} would reduce your deficit to {remaining_balance_str}.")
      else:
        # Additional spending would increase the deficit
        deficit_after = abs(remaining_balance)
        print(f"You have {current_balance_str} in your checking and savings accounts. You cannot afford additional spending. Your projected total spending is {total_spending_str}, which would increase your deficit to ${deficit_after:.0f}.")
    elif current_balance >= total_spending:
      print(f"You can afford a couple months of fun. Your checking and savings accounts have {current_balance_str} available. Your projected total spending is {total_spending_str}, leaving you with {remaining_balance_str} remaining.")
    else:
      # Need more money
      shortfall = total_spending - current_balance
      shortfall_str = utter_balance(shortfall, "{amount_with_direction}")
      print(f"You cannot afford a couple months of fun. Your checking and savings accounts have {current_balance_str} available. However, your projected total spending is {total_spending_str}, so you would need {shortfall_str} more.")
    
    return True, metadata

def process_input_have_i_been_saving_anything_monthly_in_the_past_4_months():
    metadata = {}
    
    # Get income and spending transactions
    income_df = retrieve_income_transactions()
    spending_df = retrieve_spending_transactions()
    
    if income_df.empty and spending_df.empty:
      print("You have no transactions.")
      return True, metadata
    
    # Get current month start
    first_day_current_month = get_start_of_month(datetime.now())
    
    # Calculate start dates for the past 4 months
    month_4_ago_start = get_start_of_month(get_after_periods(first_day_current_month, granularity="monthly", count=-4))
    month_1_ago_end = get_after_periods(first_day_current_month, granularity="daily", count=-1)
    
    # Filter transactions for the past 4 months
    past_4_months_income_df = income_df[
      (income_df['date'] >= month_4_ago_start) & (income_df['date'] <= month_1_ago_end)
    ]
    
    past_4_months_spending_df = spending_df[
      (spending_df['date'] >= month_4_ago_start) & (spending_df['date'] <= month_1_ago_end)
    ]
    
    if past_4_months_income_df.empty and past_4_months_spending_df.empty:
      print("You have no transactions for the past 4 months.")
      return True, metadata

    # Calculate totals for the period
    total_income = past_4_months_income_df['amount'].sum() if not past_4_months_income_df.empty else 0.0
    total_spending = past_4_months_spending_df['amount'].sum() if not past_4_months_spending_df.empty else 0.0
    transaction_total = total_income + total_spending
    transaction_total_str = utter_amount(transaction_total, "{amount}")
    
    # Format messages
    income_msg = utter_income_transaction_total(past_4_months_income_df, "Total income: {verb_and_total_amount}") if not past_4_months_income_df.empty else "No income transactions found."
    spending_msg = utter_spending_transaction_total(past_4_months_spending_df, "Total spending: {verb_and_total_amount}") if not past_4_months_spending_df.empty else "No spending transactions found."
    
    print(f"Checking monthly savings over the past 4 months:")
    print(f"  {income_msg}")
    print(f"  {spending_msg}")
    
    if transaction_total < 0:  # Negative transaction_total means positive savings (income > expenses)
      print(f"Yes, you have been saving! Your net savings over the last 4 months is {transaction_total_str}.")
    elif transaction_total > 0:  # Positive transaction_total means spending exceeded income
      print(f"No, you have spent more than you earned over the last 4 months. Your net result is {transaction_total_str}.")
    else:
      print("No, you have broken even over the last 4 months.")
      
    return True, metadata


def process_input_how_much_am_i_expected_to_save_next_week():
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


def process_input_did_i_get_any_income_in_last_few_weeks_and_what_about_upcoming_weeks():
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
        print(utter_income_transaction_total(past_income_df, "In total, you {verb_and_total_amount} from the past few weeks."))
    
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


def process_input_check_my_checking_account_if_i_can_afford_paying_my_rent_next_month():
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


def process_input_what_is_my_forecasted_discretionary_spending_breakdown_for_the_next_three_months():
    metadata = {}
    
    # Get current month start
    first_day_current_month = get_start_of_month(datetime.now())
    
    # Calculate next 3 months start dates
    next_month_start = get_start_of_month(get_after_periods(first_day_current_month, granularity="monthly", count=1))
    month_after_next_start = get_start_of_month(get_after_periods(first_day_current_month, granularity="monthly", count=2))
    third_month_start = get_start_of_month(get_after_periods(first_day_current_month, granularity="monthly", count=3))
    
    # Retrieve spending forecasts for next 3 months
    spending_df = retrieve_spending_forecasts('monthly')
    
    if spending_df.empty:
      print("You have no spending forecasts for the next 3 months.")
      return True, metadata
    
    # Filter for next 3 months
    spending_df = spending_df[spending_df['start_date'].isin([next_month_start, month_after_next_start, third_month_start])]
    
    if spending_df.empty:
      print("You have no spending forecasts for the next 3 months.")
      return True, metadata
    
    leisure_categories = ['leisure_entertainment', 'leisure_travel']
    shopping_categories = ['shopping_clothing', 'shopping_gadgets', 'shopping_kids', 'shopping_pets']
    gifts_categories = ['donations_gifts']
    
    leisure_df = spending_df[spending_df['category'].isin(leisure_categories)]
    shopping_df = spending_df[spending_df['category'].isin(shopping_categories)]
    gifts_df = spending_df[spending_df['category'].isin(gifts_categories)]
    
    if leisure_df.empty and shopping_df.empty and gifts_df.empty:
      print("You have no discretionary spending forecasts for the next 3 months.")
      return True, metadata
    
    # Calculate totals by category group
    total_leisure = leisure_df['forecasted_amount'].sum() if not leisure_df.empty else 0.0
    total_shopping = shopping_df['forecasted_amount'].sum() if not shopping_df.empty else 0.0
    total_gifts = gifts_df['forecasted_amount'].sum() if not gifts_df.empty else 0.0
    total_discretionary = total_leisure + total_shopping + total_gifts
    
    # Format and print breakdown
    print(f"Your forecasted discretionary spending breakdown for the next 3 months:")
    print(f"  Leisure: {utter_spending_forecast_amount(total_leisure, '{amount_and_direction}')}")
    print(f"  Shopping: {utter_spending_forecast_amount(total_shopping, '{amount_and_direction}')}")
    print(f"  Gifts: {utter_spending_forecast_amount(total_gifts, '{amount_and_direction}')}")
    print(f"  Total: {utter_spending_forecast_amount(total_discretionary, '{amount_and_direction}')}")
    
    return True, metadata


def process_input_list_streaming_subscriptions_paid_last_month():
    metadata = {"subscriptions": []}
    
    subscriptions_df = retrieve_subscriptions()
    
    if subscriptions_df.empty:
      print("You have no subscriptions.")
      return True, metadata
    
    # Filter for streaming subscriptions: use subscription_name AND category
    # Populate using relevant names from SUBSCRIPTION_NAMES
    streaming_names = []
    streaming_categories = ['leisure_entertainment']
    
    name_matches = subscriptions_df['subscription_name'].str.contains('|'.join(streaming_names), case=False, regex=True, na=False)
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


def process_input_whats_the_total_across_account_types():
    depository_df = retrieve_depository_accounts()
    credit_df = retrieve_credit_accounts()
    metadata = {"accounts": []}
    
    # Combine all accounts
    df = pd.concat([depository_df, credit_df], ignore_index=True) if not (depository_df.empty and credit_df.empty) else pd.DataFrame()
    
    if df.empty:
      print("You have no accounts.")
      return True, metadata
    
    print("Here are all your account balances:")
    for_print, metadata["accounts"] = account_names_and_balances(df, "Account \"{account_name}\" ({account_type}) has {balance_current} left with {balance_available} available now.")
    print(for_print)
    
    # Calculate totals for different account types
    account_types = df['account_type'].unique()
    
    for acc_type in account_types:
      type_df = df[df['account_type'] == acc_type]
      if not type_df.empty:
        print(utter_account_totals(type_df, f"Total for {acc_type.replace('_', ' ')} accounts: ${{balance_current:,.2f}} left."))
    
    return True, metadata


def process_input_how_much_eating_out_have_I_done():
    df = retrieve_spending_transactions()
    metadata = {"transactions": []}
    
    if df.empty:
      print("You have no spending transactions.")
      return True, metadata
    
    # Filter for eating out categories
    eating_out_categories = ['meals_dining_out', 'meals_delivered_food']
    df = df[df['category'].isin(eating_out_categories)]
    
    if df.empty:
      print("You have no eating out transactions.")
      return True, metadata
    
    print("Here are your eating out transactions:")
    for_print, metadata["transactions"] = transaction_names_and_amounts(df, "{amount_and_direction} {transaction_name} on {date}.")
    print(for_print)
    print(utter_spending_transaction_total(df, "In total, you {verb_and_total_amount} on eating out."))
    
    return True, metadata


def process_input_provide_a_comprehensive_summary_of_my_financial_situation():
    metadata = {}
    
    # --- 1. Comprehensive Account Balances ---
    print("--- Account Balances Summary ---")
    
    # Depository Accounts (Assets)
    assets_df = retrieve_depository_accounts()
    if not assets_df.empty:
        print("\nDepository Accounts:")
        _, metadata["depository_balances"] = account_names_and_balances(assets_df, "Account '{account_name}' ({account_type}): {balance_current} current.")
        print(utter_account_totals(assets_df, "Total Assets: {balance_current}."))
    else:
        print("No depository accounts found.")

    # Credit/Loan Accounts (Liabilities)
    credit_df = retrieve_credit_accounts()
    if not credit_df.empty:
        print("\nCredit/Loan Accounts:")
        _, metadata['credit_liabilities'] = account_names_and_balances(credit_df, "Account '{account_name}' ({account_type}): {balance_current} owed.")
        print(utter_account_totals(credit_df, "Total Liabilities: {balance_current}."))
    else:
        print("No credit or loan accounts found.")
        
    # Net Worth
    total_assets = assets_df['balance_current'].sum() if not assets_df.empty else 0.0
    total_liabilities = credit_df['balance_current'].sum() if not credit_df.empty else 0.0
    print(utter_net_worth(total_assets, total_liabilities, "Net Worth: {net_worth_state_with_amount}."))

    # --- 2. Income vs Spending Comparison (Last Month) ---
    print("\n--- Income vs Spending Comparison (Last Month) ---")
    
    income_df = retrieve_income_transactions()
    spending_df = retrieve_spending_transactions()
    
    if income_df.empty and spending_df.empty:
        print("No recent transactions to compare.")
    else:
        # Filter for last month
        first_day_current_month = get_start_of_month(datetime.now())
        first_day_last_month = get_after_periods(first_day_current_month, granularity="monthly", count=-1)
        last_day_last_month = get_end_of_month(first_day_last_month)
        
        last_month_income_df = income_df[(income_df['date'] >= first_day_last_month) & (income_df['date'] <= last_day_last_month)]
        last_month_spending_df = spending_df[(spending_df['date'] >= first_day_last_month) & (spending_df['date'] <= last_day_last_month)]
        
        total_income_last_month = last_month_income_df['amount'].sum() if not last_month_income_df.empty else 0.0
        total_spending_last_month = last_month_spending_df['amount'].sum() if not last_month_spending_df.empty else 0.0
        
        income_msg = utter_income_transaction_total(last_month_income_df, "Total Income: {verb_and_total_amount}")
        spending_msg = utter_spending_transaction_total(last_month_spending_df, "Total Spending: {verb_and_total_amount}")
        
        net_flow = total_income_last_month + total_spending_last_month # Since spending amounts are positive (outflow)
        
        print(income_msg)
        print(spending_msg)
        
        if net_flow > 0:
            print(f"Net result: You saved/had a surplus of ${net_flow:.2f} last month.")
        elif net_flow < 0:
            print(f"Net result: You spent/had a deficit of ${abs(net_flow):.2f} last month.")
        else:
            print("Net result: You broke even last month.")
        
        metadata['last_month_income_total'] = total_income_last_month
        metadata['last_month_spending_total'] = total_spending_last_month


    # --- 3. Recent Significant Spending (Travel & Dining Out) ---
    print("\n--- Recent Significant Spending (Travel & Dining Out) ---")
    
    if spending_df.empty:
        print("No spending transactions available for detailed review.")
    else:
        # Define categories of interest
        travel_categories = ['leisure_travel']
        dining_categories = ['meals_dining_out', 'meals_delivered_food']
        
        # Filter for recent spending (e.g., last 30 days)
        thirty_days_ago = datetime.now() - timedelta(days=30)
        recent_spending_df = spending_df[spending_df['date'] >= thirty_days_ago]
        
        travel_df = recent_spending_df[recent_spending_df['category'].isin(travel_categories)]
        dining_df = recent_spending_df[recent_spending_df['category'].isin(dining_categories)]
        
        if travel_df.empty and dining_df.empty:
            print("No recent travel or dining out spending found in the last 30 days.")
        else:
            # Travel Summary
            if not travel_df.empty:
                total_travel = travel_df['amount'].sum()
                print(f"Travel Spending (Last 30 Days): {utter_spending_forecast_amount(total_travel, '{amount_and_direction}')}")
                _, metadata['recent_travel'] = transaction_names_and_amounts(travel_df, " - {amount_and_direction} {transaction_name} on {date}.")
            else:
                print("No recent travel spending.")

            # Dining Summary
            if not dining_df.empty:
                total_dining = dining_df['amount'].sum()
                print(f"Dining/Food Spending (Last 30 Days): {utter_spending_forecast_amount(total_dining, '{amount_and_direction}')}")
                _, metadata['recent_dining'] = transaction_names_and_amounts(dining_df, " - {amount_and_direction} {transaction_name} on {date}.")
            else:
                print("No recent dining/food spending.")
                
    print("\n--- Summary Complete ---")
    return True, metadata


def main():
  """Main function that demonstrates the reminder tools"""
  global user_id
  print("🤖 Agent Code Experiment - Reminder Tools Demo")
  print("=" * 50)
  
  # Initialize database and seed users (same as flask_app.py)
  print("🔧 Setting up database and seeding users...")
  seed_users()
  db = Database()
  
  # Get or create a test user (same pattern as flask_app.py)
  username = "test_user"
  user = db.get_user(username)
  if not user:
    user_id = db.create_user(username, f"{username}@example.com")
    user = db.get_user(username)
    print(f"✅ Created new user: {username} (ID: {user['id']})")
  else:
    print(f"✅ Using existing user: {username} (ID: {user['id']})")
  
  user_id = 3

  # Test the checking account balance function
  print("\n💰 Testing checking account balance function...")
  success, metadata = process_input_how_much_left_in_checking()
  print(f"Success: {success}")
  print(f"Metadata: {metadata}")
  
  print("\n💰 Testing net worth function...")
  success, metadata = process_input_what_is_my_net_worth()
  print(f"Success: {success}")
  print(f"Metadata: {metadata}")
  
  print("\n🍽️  Testing eating out spending function...")
  success, metadata = process_input_how_much_eating_out_have_I_done()
  print(f"Success: {success}")
  print(f"Metadata: {metadata}")
  
  print("\n🍽️  Testing dining out vs groceries comparison...")
  success, metadata = process_input_did_i_spend_more_on_dining_out_over_groceries_last_month()
  print(f"Success: {success}")
  print(f"Metadata: {metadata}")
  
  print("\n💰 Testing checking account affordability for fun spending...")
  success, metadata = process_input_can_i_afford_to_pay_a_couple_months_of_fun_with_what_i_have_now()
  print(f"Success: {success}")
  print(f"Metadata: {metadata}")
  
  print("\n💰 Testing savings calculation for past 4 months...")
  success, metadata = process_input_have_i_been_saving_anything_monthly_in_the_past_4_months()
  print(f"Success: {success}")
  print(f"Metadata: {metadata}")
  
  print("\n💰 Did I get any income in last last few weeks and what about upcoming weeks?")
  success, metadata = process_input_did_i_get_any_income_in_last_few_weeks_and_what_about_upcoming_weeks()
  print(f"Success: {success}")
  print(f"Metadata: {metadata}")
  
  print("\n📊 Testing expected savings next week...")
  success, metadata = process_input_how_much_am_i_expected_to_save_next_week()
  print(f"Success: {success}")
  print(f"Metadata: {metadata}")
  
  print("\n🏠 Testing checking account affordability for rent next month...")
  success, metadata = process_input_check_my_checking_account_if_i_can_afford_paying_my_rent_next_month()
  print(f"Success: {success}")
  print(f"Metadata: {metadata}")
  
  print("\n💰 Testing forecasted discretionary spending breakdown for next 3 months...")
  success, metadata = process_input_what_is_my_forecasted_discretionary_spending_breakdown_for_the_next_three_months()
  print(f"Success: {success}")
  print(f"Metadata: {metadata}")
  
  print("\n📋 Testing list streaming subscriptions last month...")
  success, metadata = process_input_list_streaming_subscriptions_paid_last_month()
  print(f"Success: {success}")
  print(f"Metadata: {metadata}")
  
  print("\n📋 Testing provide a comprehensive summary of my financial situation...")
  success, metadata = process_input_provide_a_comprehensive_summary_of_my_financial_situation()
  print(f"Success: {success}")
  print(f"Metadata: {metadata}")



if __name__ == "__main__":
  main()

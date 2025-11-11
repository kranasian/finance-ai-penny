#!/usr/bin/env python3
"""
Agent Code Experiment - Demonstrates usage of create_reminder and retrieve_reminder tools
"""

import sys
import os
from datetime import datetime, timedelta
import pandas as pd


# Add the parent directory to the path so we can import the tools
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
  sys.path.insert(0, parent_dir)

from database import Database
from user_seeder import seed_users
from penny.tool_funcs.retrieve_accounts import retrieve_accounts_function_code_gen, account_names_and_balances, utter_account_totals
from penny.tool_funcs.retrieve_transactions import retrieve_transactions_function_code_gen, transaction_names_and_amounts, utter_transaction_totals
from penny.tool_funcs.compare_spending import compare_spending
from penny.tool_funcs.retrieve_forecasts import retrieve_spending_forecasts_function_code_gen, retrieve_income_forecasts_function_code_gen
from penny.tool_funcs.forecast_utils import utter_forecasts
from penny.tool_funcs.retrieve_subscriptions import retrieve_subscriptions_function_code_gen, subscription_names_and_amounts, utter_subscription_totals
from penny.tool_funcs.create_goal import create_goal_function_code_gen
from penny.tool_funcs.date_utils import get_start_of_month, get_end_of_month, get_start_of_week, get_end_of_week, get_after_periods, get_date_string
user_id = 1

def retrieve_accounts() -> pd.DataFrame:
  global user_id
  return retrieve_accounts_function_code_gen(user_id=user_id)


def retrieve_transactions() -> pd.DataFrame:
  global user_id
  return retrieve_transactions_function_code_gen(user_id=user_id)


def retrieve_spending_forecasts(granularity: str = 'monthly') -> pd.DataFrame:
  global user_id
  return retrieve_spending_forecasts_function_code_gen(user_id=user_id, granularity=granularity)


def retrieve_income_forecasts(granularity: str = 'monthly') -> pd.DataFrame:
  global user_id
  return retrieve_income_forecasts_function_code_gen(user_id=user_id, granularity=granularity)


def retrieve_subscriptions() -> pd.DataFrame:
  global user_id
  return retrieve_subscriptions_function_code_gen(user_id=user_id)


def create_goal(goals: list[dict]) -> tuple[str, list]:
  global user_id
  return create_goal_function_code_gen(goals, user_id=user_id)


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
    df = retrieve_accounts()
    metadata = {"accounts": []}
    
    if df.empty:
      print("You have no accounts.")
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
    accounts_df = retrieve_accounts()
    metadata = {}

    if accounts_df.empty:
        print("You have no accounts to calculate net worth.")
        return True, metadata

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
    print(f"You have a net worth of ${net_worth:.0f} with assets of ${total_assets:.0f} and liabilities of ${total_liabilities:.0f}.")

    return True, metadata


def process_input_how_much_eating_out_have_I_done():
    df = retrieve_transactions()
    metadata = {"transactions": []}
    
    if df.empty:
      print("You have no transactions.")
      return True, metadata
    
    # Filter for eating out categories
    eating_out_categories = ['meals_dining_out', 'meals_delivered_food']
    df = df[df['category'].isin(eating_out_categories)]
    
    if df.empty:
      print("You have no eating out transactions.")
      return True, metadata
    
    print("Here are your eating out transactions:")
    for_print, metadata["transactions"] = transaction_names_and_amounts(df, "{transaction_name} on {date}: {amount}")
    print(for_print)
    print(utter_transaction_totals(df, "In total, you have spent {total_amount} on eating out."))
    
    return True, metadata


def process_input_did_i_spend_more_on_dining_out_over_groceries_last_month():
    df = retrieve_transactions()
    metadata = {}
    
    if df.empty:
      print("You have no transactions.")
      return True, metadata
    
    # Filter for last month
    first_day_current_month = get_start_of_month(datetime.now())
    first_day_last_month = get_after_periods(first_day_current_month, granularity="monthly", count=-1)
    last_day_last_month = get_end_of_month(first_day_last_month)
    
    df = df[(df['date'] >= first_day_last_month) & (df['date'] <= last_day_last_month)]
    
    if df.empty:
      print("You have no transactions from last month.")
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


def process_input_check_my_checking_account_if_i_can_afford_paying_my_dining_out_last_month():
    metadata = {"accounts": []}
    
    # Get checking account balance
    accounts_df = retrieve_accounts()
    checking_df = accounts_df[accounts_df['account_type'] == 'deposit_checking']
    
    if checking_df.empty:
      print("You have no checking accounts.")
      return True, metadata
    
    for_print, metadata["accounts"] = account_names_and_balances(checking_df, "Account '{account_name}' has {balance_current} left with {balance_available} available now.")
    print(for_print)
    
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
      print(f"You can afford your dining out expenses from last month. Your checking account has ${total_available:.0f} available, and your dining out spending was ${total_dining_out:.0f}. You would have ${total_available - total_dining_out:.0f} remaining.")
    else:
      print(f"You cannot afford your dining out expenses from last month. Your checking account has ${total_available:.0f} available, but your dining out spending was ${total_dining_out:.0f}. You would need ${total_dining_out - total_available:.0f} more.")
    
    return True, metadata


def process_input_how_much_did_i_save_last_month():
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
    income_msg = utter_transaction_totals(income_df, "{direction} {total_amount:.0f}") if not income_df.empty else "$0.00"
    expenses_msg = utter_transaction_totals(expenses_df, "{direction} {total_amount:.0f}") if not expenses_df.empty else "$0.00"
    
    # Format and print savings message
    if savings < 0:
      print(f"You saved ${abs(savings):.0f} last month. Income: {income_msg} and expenses: {expenses_msg}.")
    elif savings > 0:
      print(f"You spent ${savings:.0f} more than you earned last month. Income: {income_msg} and expenses: {expenses_msg}.")
    else:
      print(f"You broke even last month. Income: {income_msg} and expenses: {expenses_msg}.")
    
    return True, metadata


def process_input_list_income_past_2_weeks():
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
    for_print, metadata["transactions"] = transaction_names_and_amounts(df, "{transaction_name}: {direction} ${amount:.0f} on {date}")
    print(for_print)
    print(utter_transaction_totals(df, "In total, you {direction} {total_amount} from the past 2 weeks."))
    
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
    
    # Calculate totals
    total_income = income_df['forecasted_amount'].sum() if not income_df.empty else 0.0
    total_spending = spending_df['forecasted_amount'].sum() if not spending_df.empty else 0.0
    
    # Calculate expected savings
    expected_savings = total_income - total_spending
    
    # Get formatted income and spending messages using utter_forecasts
    income_msg = utter_forecasts(income_df, "{direction} {total_amount:.0f}") if not income_df.empty else "$0.00"
    expenses_msg = utter_forecasts(spending_df, "{direction} {total_amount:.0f}") if not spending_df.empty else "$0.00"
    
    # Format and print expected savings message
    if expected_savings > 0:
      print(f"You are expected to save ${expected_savings:.0f} next week. Your forecasted income is {income_msg} and your forecasted spending is {expenses_msg}.")
    elif expected_savings < 0:
      print(f"You are expected to spend ${abs(expected_savings):.0f} more than you earn next week. Your forecasted income is {income_msg} and your forecasted spending is {expenses_msg}.")
    else:
      print(f"You are expected to break even next week. Your forecasted income is {income_msg} and your forecasted spending is {expenses_msg}.")
    
    return True, metadata


def process_input_how_much_am_i_expected_to_save_next_month():
    metadata = {}
    
    # Get next month date
    first_day_current_month = get_start_of_month(datetime.now())
    next_month_start_date = get_start_of_month(get_after_periods(first_day_current_month, granularity="monthly", count=1))
    
    # Retrieve income and spending forecasts for next month
    income_df = retrieve_income_forecasts('monthly')
    spending_df = retrieve_spending_forecasts('monthly')
    
    if income_df.empty and spending_df.empty:
      print("You have no forecasts for next month.")
      return True, metadata
    
    # Filter for next month
    if not income_df.empty:
      income_df = income_df[income_df['start_date'] == next_month_start_date]
    if not spending_df.empty:
      spending_df = spending_df[spending_df['start_date'] == next_month_start_date]
    if income_df.empty and spending_df.empty:
      print("You have no forecasts for next month.")
      return True, metadata
    
    # Calculate totals
    total_income = income_df['forecasted_amount'].sum() if not income_df.empty else 0.0
    total_spending = spending_df['forecasted_amount'].sum() if not spending_df.empty else 0.0
    
    # Calculate expected savings
    expected_savings = total_income - total_spending
    
    # Get formatted income and spending messages using utter_forecasts
    income_msg = utter_forecasts(income_df, "{direction} {total_amount:.0f}") if not income_df.empty else "$0.00"
    expenses_msg = utter_forecasts(spending_df, "{direction} {total_amount:.0f}") if not spending_df.empty else "$0.00"
    
    # Format and print expected savings message
    if expected_savings > 0:
      print(f"You are expected to save ${expected_savings:.0f} next month. Your forecasted income is {income_msg} and your forecasted spending is {expenses_msg}.")
    elif expected_savings < 0:
      print(f"You are expected to spend ${abs(expected_savings):.0f} more than you earn next month. Your forecasted income is {income_msg} and your forecasted spending is {expenses_msg}.")
    else:
      print(f"You are expected to break even next month. Your forecasted income is {income_msg} and your forecasted spending is {expenses_msg}.")
    
    return True, metadata


def process_input_check_my_checking_account_if_i_can_afford_paying_my_rent_next_month():
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


def process_input_list_my_subscriptions():
    metadata = {"subscriptions": []}
    
    subscriptions_df = retrieve_subscriptions()
    
    if subscriptions_df.empty:
      print("You have no subscriptions.")
      return True, metadata
    
    for_print, metadata["subscriptions"] = subscription_names_and_amounts(subscriptions_df, '{subscription_name}: {direction} ${amount:.0f} on {date}')
    transaction_count = len(subscriptions_df)
    print(f"Your subscriptions ({transaction_count} transaction{'s' if transaction_count != 1 else ''}):")
    print(for_print)
    
    print(utter_subscription_totals(subscriptions_df, 'Total subscription transactions: ${total_amount:.0f} {direction}'))
    
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
    
    name_matches = subscriptions_df['subscription_name'].str.lower().isin(streaming_names)
    category_matches = subscriptions_df['category'].isin(streaming_categories)
    streaming_df = subscriptions_df[name_matches & category_matches]
    
    if streaming_df.empty:
      print("You have no streaming subscriptions.")
      return True, metadata
      
    # Filter for last month
    today = datetime.now()
    first_day_current_month = get_start_of_month(today)
    first_day_last_month = get_after_periods(first_day_current_month, granularity="monthly", count=-1)
    last_day_last_month = get_end_of_month(first_day_last_month)
    
    streaming_df = streaming_df[(streaming_df['date'] >= first_day_last_month) & (streaming_df['date'] <= last_day_last_month)]
    
    if streaming_df.empty:
      print("You have no streaming subscription payments last month.")
      return True, metadata
    
    for_print, metadata["subscriptions"] = subscription_names_and_amounts(streaming_df, '{subscription_name}: {direction} ${amount:.0f} on {date}')
    transaction_count = len(streaming_df)
    print(f"Your streaming subscription payments last month ({transaction_count} transaction{'s' if transaction_count != 1 else ''}):")
    print(for_print)
    
    print(utter_subscription_totals(streaming_df, 'Total streaming subscription spending last month: ${total_amount:.0f} {direction}'))
    
    return True, metadata


def process_input_create_a_budget_for_60_gas_every_week_for_the_next_6_months():
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
        "description": f"Created $60 Weekly Gas ⛽ from {get_date_string(start_date)} to {get_date_string(end_date)}."
    }]
    
    response, goals_list = create_goal(goals)
    metadata["goals"] = goals_list
    
    print(response)
    
    return True, metadata


def process_input_pay_200_weekly_on_my_bofa_credit_card():
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
        
        response, goals_list = create_goal(goals)
        metadata["goals"] = goals_list
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
    
    response, metadata["goals"] = create_goal(goals)
    print(response)
    
    return True, metadata


def process_input_pay_200_weekly_on_all_my_credit_cards():
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
    
    response, goals_list = create_goal(goals)
    metadata["goals"] = goals_list
    print(response)
    
    return True, metadata
  
def process_input_whats_the_total_across_account_types():
    df = retrieve_accounts()
    metadata = {"accounts": []}
    
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
  
  print("\n💰 Testing checking account affordability for dining out...")
  success, metadata = process_input_check_my_checking_account_if_i_can_afford_paying_my_dining_out_last_month()
  print(f"Success: {success}")
  print(f"Metadata: {metadata}")
  
  print("\n💰 Testing savings calculation for last month...")
  success, metadata = process_input_how_much_did_i_save_last_month()
  print(f"Success: {success}")
  print(f"Metadata: {metadata}")
  
  print("\n💰 Testing list income past 2 weeks...")
  success, metadata = process_input_list_income_past_2_weeks()
  print(f"Success: {success}")
  print(f"Metadata: {metadata}")
  
  print("\n📊 Testing expected savings next week...")
  success, metadata = process_input_how_much_am_i_expected_to_save_next_week()
  print(f"Success: {success}")
  print(f"Metadata: {metadata}")
  
  print("\n📊 Testing expected savings next month...")
  success, metadata = process_input_how_much_am_i_expected_to_save_next_month()
  print(f"Success: {success}")
  print(f"Metadata: {metadata}")
  
  print("\n💰 Testing checking account affordability for rent next month...")
  success, metadata = process_input_check_my_checking_account_if_i_can_afford_paying_my_rent_next_month()
  print(f"Success: {success}")
  print(f"Metadata: {metadata}")
  
  print("\n📋 Testing list subscriptions...")
  success, metadata = process_input_list_my_subscriptions()
  print(f"Success: {success}")
  print(f"Metadata: {metadata}")
  
  print("\n📋 Testing list streaming subscriptions last month...")
  success, metadata = process_input_list_streaming_subscriptions_paid_last_month()
  print(f"Success: {success}")
  print(f"Metadata: {metadata}")
  
  print("\n🎯 Testing create gas budget...")
  success, metadata = process_input_create_a_budget_for_60_gas_every_week_for_the_next_6_months()
  print(f"Success: {success}")
  print(f"Metadata: {metadata}")
  
  print("\n💳 Testing pay BoFa credit card weekly...")
  success, metadata = process_input_pay_200_weekly_on_my_bofa_credit_card()
  print(f"Success: {success}")
  print(f"Metadata: {metadata}")
  
  print("\n💳 Testing pay all credit cards weekly...")
  success, metadata = process_input_pay_200_weekly_on_all_my_credit_cards()
  print(f"Success: {success}")
  print(f"Metadata: {metadata}")

  print("\n💰 Testing checking account affordability for rent next month...")
  success, metadata = process_input_check_my_checking_account_if_i_can_afford_paying_my_rent_next_month()
  print(f"Success: {success}")
  print(f"Metadata: {metadata}")


if __name__ == "__main__":
  main()

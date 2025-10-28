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
from tools.retrieve_accounts import retrieve_accounts_function_code_gen, account_names_and_balances, utter_account_totals
user_id = 1

def retrieve_accounts() -> pd.DataFrame:
  global user_id
  return retrieve_accounts_function_code_gen(user_id=user_id)


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
    else:
      # Filter for checking accounts
      df = df[df['account_type'] == 'deposit_checking']
      
      if df.empty:
        print("You have no checking accounts.")
      else:
        print("Here are your checking account balances:")
        for_print, metadata["accounts"] = account_names_and_balances(df, "Account \"{name}\" has {current_balance} left with {available_balance} available now.")
        print(for_print)
        print(utter_account_totals(df, "Across all checking accounts, you have {current_balance} left."))
    
    return True, metadata


def process_input_what_is_my_net_worth():
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
  
  user_id = user['id']
  
  # Create a checking account for the test user
  print("\n🏦 Creating checking account for test user...")
  account_id = db.create_account(
    user_id=user_id,
    account_type='deposit_checking',
    balance_available=1500.00,
    balance_current=0.00,
    account_name='Test Checking Account',
    account_mask='1234'
  )
  print(f"Created checking account with ID: {account_id}")
  
  # Test the checking account balance function
  print("\n💰 Testing checking account balance function...")
  success, metadata = process_input_how_much_left_in_checking()
  print(f"Success: {success}")
  print(f"Metadata: {metadata}")
  
  success, metadata = process_input_what_is_my_net_worth()
  print(f"Success: {success}")
  print(f"Metadata: {metadata}")
  # success, metadata = process_input_simple_tell_me_to_wash_dishes_at_11_00()
  # print(f"Success: {success}")
  # print(f"Metadata: {metadata}")
  

if __name__ == "__main__":
  main()

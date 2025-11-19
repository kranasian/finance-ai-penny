from typing import Tuple
import random


def lookup_user_accounts_transactions_income_and_spending_patterns(
    lookup_request: str, 
    input_info: str = None
) -> Tuple[bool, str]:
    """
    Lookup user accounts, transactions, income and spending patterns.
    
    This is a dummy implementation that returns sample data.
    
    Args:
        lookup_request: The detailed information requested, written in natural language to lookup 
                       about the user's accounts, transactions including income and spending, 
                       subscriptions and compare them.
        input_info: Optional input from another skill function
        
    Returns:
        Tuple[bool, str]: (success, output_info)
    """
    # Randomly choose between general format or Spotify-specific format
    # if random.choice([True, False]):
    if True:
        # General format
        sample_output = """--- Account Balances ---
Depository Accounts:
- Asset Account 'Chase Total Checking **1563': Current: $567, Available: $567
- Asset Account 'Chase Savings **3052': Current: $1202, Available: $1202
Total Depository Balance: $1769.

--- Recent Income (Last 30 Days) ---
Recent Income Transactions:
- $1440 was received from CA State Payroll on 2025-11-18 (Chase Total Checking **1563).
- $2 was received from Savings Interest on 2025-11-01 (Chase Savings **3052).
- $1340 was received from CA State Payroll on 2025-10-31 (Chase Total Checking **1563).
Total recent income: earned $2882.

--- Recent Spending Patterns (Last 30 Days) ---
Recent Spending Transactions:
- $2311 was paid to Credit Card Payment on 2025-11-17 (Chase Total Checking **1563).
- $368 was paid to Texas Roadhouse on 2025-11-17 (Chase Total Checking **1563).
- $150 was paid to Geico on 2025-11-17 (Chase Total Checking **1563).
- $56 was paid to Costco Gas on 2025-10-17 (Chase Total Checking **1563).
- $150 was paid to Target on 2025-10-05 (Chase Total Checking **1563).
Total recent spending: spent $2435.
"""
    else:
        # Spotify-specific format
        sample_output = """
--- Last 10 Spotify Spending Transactions ---
- $9.99 was paid to Spotify on 2025-11-01 (Chase Total Checking **1563).
- $9.99 was paid to Spotify on 2025-10-01 (Chase Total Checking **1563).
- $9.99 was paid to Spotify on 2025-09-01 (Chase Total Checking **1563).
- $9.99 was paid to Spotify on 2025-08-01 (Chase Total Checking **1563).
- $9.99 was paid to Spotify on 2025-07-01 (Chase Total Checking **1563).
- $9.99 was paid to Spotify on 2025-06-01 (Chase Total Checking **1563).
- $9.99 was paid to Spotify on 2025-05-01 (Chase Total Checking **1563).
- $9.99 was paid to Spotify on 2025-04-01 (Chase Total Checking **1563).
- $9.99 was paid to Spotify on 2025-03-01 (Chase Total Checking **1563).
- $9.99 was paid to Spotify on 2025-02-01 (Chase Total Checking **1563).

--- Spotify Subscription ---
name: Spotify
next_amount: 9.99
next_likely_payment_date: 2025-12-01
next_earliest_payment_date: 2025-12-01
next_latest_payment_date: 2025-12-05
user_cancelled_date: None
last_transaction_date: 2025-11-01
"""
    
    return True, sample_output


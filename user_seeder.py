import sqlite3
import os
import random
import uuid
from datetime import datetime, timedelta
from database import Database

def reset_database(db_path: str = "chatbot.db"):
  """Reset the database by dropping and recreating all tables"""
  # Remove existing database file
  if os.path.exists(db_path):
    os.remove(db_path)
    print(f"Removed existing database: {db_path}")
  
  # Initialize new database
  db = Database(db_path)
  print("Database reset and reinitialized")
  return db

def create_sample_accounts(user_id: int, account_count: int) -> list:
  """Create sample accounts for a user"""
  db = Database()
  account_ids = []
  
  # Account templates for different types
  account_templates = [
    {
      'account_type': 'deposit_checking',
      'account_name': 'Chase Total Checking Account',
      'balance_available': 2500.00,
      'balance_current': 2500.00,
      'balance_limit': 0.00
    },
    {
      'account_type': 'deposit_savings',
      'account_name': 'Ally Bank Online Savings Account',
      'balance_available': 15000.00,
      'balance_current': 15000.00,
      'balance_limit': 0.00
    },
    {
      'account_type': 'deposit_savings',
      'account_name': 'Marcus by Goldman Sachs High-Yield Savings',
      'balance_available': 20000.00,
      'balance_current': 20000.00,
      'balance_limit': 0.00
    },
    {
      'account_type': 'deposit_savings',
      'account_name': 'American Express High Yield Savings',
      'balance_available': 12000.00,
      'balance_current': 12000.00,
      'balance_limit': 0.00
    },
    {
      'account_type': 'deposit_money_market',
      'account_name': 'Capital One Money Market Account',
      'balance_available': 25000.00,
      'balance_current': 25000.00,
      'balance_limit': 0.00
    },
    {
      'account_type': 'credit_card',
      'account_name': 'Chase Freedom Unlimited Credit Card',
      'balance_available': 3800.00,
      'balance_current': 1200.00,
      'balance_limit': 5000.00  # Credit limit
    },
    {
      'account_type': 'credit_card',
      'account_name': 'American Express Gold Card',
      'balance_available': 7500.00,
      'balance_current': 2500.00,
      'balance_limit': 10000.00  # Credit limit
    },
    {
      'account_type': 'credit_card',
      'account_name': 'Capital One Venture Rewards Credit Card',
      'balance_available': 8500.00,
      'balance_current': 1500.00,
      'balance_limit': 10000.00  # Credit limit
    },
    {
      'account_type': 'credit_card',
      'account_name': 'Citi Double Cash Card',
      'balance_available': 5200.00,
      'balance_current': 800.00,
      'balance_limit': 6000.00  # Credit limit
    },
    {
      'account_type': 'credit_card',
      'account_name': 'Discover it Cash Back',
      'balance_available': 3400.00,
      'balance_current': 600.00,
      'balance_limit': 4000.00  # Credit limit
    },
    {
      'account_type': 'loan_line_of_credit',
      'account_name': 'Bank of America Personal Line of Credit',
      'balance_available': 10000.00,
      'balance_current': 5000.00,
      'balance_limit': 15000.00  # Credit line limit
    },
    {
      'account_type': 'loan_home_equity',
      'account_name': 'Wells Fargo Home Equity Line of Credit',
      'balance_available': 38000.00,
      'balance_current': 12000.00,
      'balance_limit': 50000.00  # Credit line limit
    },
    {
      'account_type': 'loan_mortgage',
      'account_name': 'Quicken Loans Mortgage',
      'balance_available': 0.00,
      'balance_current': 250000.00,
      'balance_limit': 250000.00  # Original loan amount
    },
    {
      'account_type': 'loan_auto',
      'account_name': 'Toyota Financial Services Auto Loan',
      'balance_available': 0.00,
      'balance_current': 18000.00,
      'balance_limit': 22000.00  # Original loan amount
    }
  ]
  
  # Select accounts based on count
  if account_count == 1:
    # SmallDataUser: 1 checking account
    selected_accounts = [account_templates[0]]  # deposit_checking
  elif account_count == 3:
    # MediumDataUser: 3 accounts (checking, credit card, auto loan)
    selected_accounts = [
      account_templates[0],  # deposit_checking
      account_templates[5],  # credit_card (Chase Freedom)
      account_templates[13]  # loan_auto
    ]
  else:
    # HeavyDataUser: 13 accounts (all base account types + 2 additional savings + 3 additional credit cards)
    selected_accounts = [
      account_templates[0],  # deposit_checking
      account_templates[1],  # deposit_savings (Ally)
      account_templates[2],  # deposit_savings (Marcus) - additional
      account_templates[3],  # deposit_savings (Amex) - additional
      account_templates[4],  # deposit_money_market
      account_templates[5],  # credit_card (Chase Freedom)
      account_templates[6],  # credit_card (Amex Gold) - additional
      account_templates[7],  # credit_card (Capital One) - additional
      account_templates[8],  # credit_card (Citi) - additional
      account_templates[9],  # credit_card (Discover) - additional
      account_templates[10], # loan_line_of_credit
      account_templates[11], # loan_home_equity
      account_templates[12], # loan_mortgage
      account_templates[13]  # loan_auto
    ]
  
  for account_template in selected_accounts:
    account_mask = f"{random.randint(1000, 9999)}"
    
    account_id = db.create_account(
      user_id=user_id,
      account_type=account_template['account_type'],
      balance_available=account_template['balance_available'],
      balance_current=account_template['balance_current'],
      account_name=account_template['account_name'],
      account_mask=account_mask,
      balance_limit=account_template.get('balance_limit', None)
    )
    account_ids.append(account_id)
  
  return account_ids

def create_sample_transactions(user_id: int, account_ids: list, transaction_count: int, months: int) -> list:
  """Create sample transactions for a user across their accounts"""
  db = Database()
  transaction_ids = []
  
  # Transaction templates organized by category
  transaction_templates = {
    'income_salary': [
      ('Direct Deposit - Salary', 'PAYROLL DEPOSIT COMPANY INC', 3500.00),
      ('Monthly Bonus', 'BONUS PAYMENT COMPANY CORP', 500.00),
      ('Overtime Pay', 'OVERTIME PAYMENT COMPANY', 200.00)
    ],
    'income_sidegig': [
      ('Uber Earnings', 'UBER TECHNOLOGIES INC', 150.00),
      ('Etsy Sales', 'ETSY INC', 75.00),
      ('Freelance Payment', 'FREELANCE CLIENT LLC', 300.00)
    ],
    'meals_groceries': [
      ('Whole Foods', 'WHOLE FOODS MARKET', -85.50),
      ('Safeway', 'SAFEWAY STORE', -120.30),
      ('Trader Joes', 'TRADER JOES', -65.75),
      ('Costco', 'COSTCO WHOLESALE', -180.00)
    ],
    'meals_dining_out': [
      ('McDonalds', 'MCDONALDS RESTAURANT', -12.50),
      ('Starbucks', 'STARBUCKS COFFEE', -5.75),
      ('Local Restaurant', 'DOWNTOWN BISTRO', -45.00),
      ('Pizza Palace', 'PIZZA PALACE INC', -28.90)
    ],
    'meals_delivered_food': [
      ('DoorDash', 'DOORDASH INC', -35.00),
      ('Uber Eats', 'UBER EATS', -42.50),
      ('Grubhub', 'GRUBHUB INC', -38.75)
    ],
    'leisure_entertainment': [
      ('Netflix', 'NETFLIX INC', -15.99),
      ('Spotify', 'SPOTIFY USA', -9.99),
      ('Movie Theater', 'AMC THEATERS', -25.00),
      ('Concert Tickets', 'TICKETMASTER', -85.00)
    ],
    'leisure_travel': [
      ('Airline Ticket', 'DELTA AIRLINES', -450.00),
      ('Hotel Booking', 'MARRIOTT HOTELS', -180.00),
      ('Car Rental', 'ENTERPRISE RENTAL', -95.00),
      ('Airbnb', 'AIRBNB INC', -120.00)
    ],
    'bills_connectivity': [
      ('Internet Bill', 'COMCAST CABLE', -79.99),
      ('Phone Bill', 'VERIZON WIRELESS', -85.00),
      ('Cable TV', 'SPECTRUM CABLE', -65.00)
    ],
    'bills_insurance': [
      ('Car Insurance', 'STATE FARM INS', -125.00),
      ('Health Insurance', 'BLUE CROSS BLUE', -200.00),
      ('Home Insurance', 'ALLSTATE INSURANCE', -85.00)
    ],
    'bills_service_fees': [
      ('Bank Fee', 'BANK OF AMERICA', -12.00),
      ('ATM Fee', 'ATM TRANSACTION', -3.50),
      ('Service Charge', 'MONTHLY SERVICE', -8.00)
    ],
    'shelter_home': [
      ('Rent Payment', 'APARTMENT COMPLEX', -1200.00),
      ('Mortgage Payment', 'MORTGAGE COMPANY', -1800.00),
      ('Property Tax', 'COUNTY TREASURER', -300.00)
    ],
    'shelter_utilities': [
      ('Electric Bill', 'PGE ELECTRIC', -85.00),
      ('Water Bill', 'CITY WATER DEPT', -45.00),
      ('Gas Bill', 'NATURAL GAS CO', -65.00),
      ('Trash Service', 'WASTE MANAGEMENT', -25.00)
    ],
    'shelter_upkeep': [
      ('Home Depot', 'HOME DEPOT INC', -150.00),
      ('Lowe\'s', 'LOWES HOME IMPROVEMENT', -95.00),
      ('Plumber Service', 'ABC PLUMBING', -200.00),
      ('HVAC Repair', 'COOL AIR SYSTEMS', -350.00)
    ],
    'education_tuition': [
      ('School Tuition', 'UNIVERSITY NAME', -1200.00),
      ('Daycare Payment', 'KIDS CARE CENTER', -800.00),
      ('Textbook Purchase', 'CAMPUS BOOKSTORE', -150.00)
    ],
    'shopping_clothing': [
      ('Target', 'TARGET STORE', -75.00),
      ('Amazon', 'AMAZON.COM', -45.00),
      ('Nike Store', 'NIKE INC', -120.00),
      ('Macy\'s', 'MACYS DEPARTMENT', -85.00)
    ],
    'shopping_gadgets': [
      ('Best Buy', 'BEST BUY STORE', -299.99),
      ('Apple Store', 'APPLE INC', -1299.00),
      ('Microsoft Store', 'MICROSOFT CORP', -199.00)
    ],
    'transportation_public': [
      ('Metro Card', 'METRO TRANSIT', -50.00),
      ('Bus Pass', 'CITY BUS SYSTEM', -30.00),
      ('Train Ticket', 'AMTRAK', -45.00)
    ],
    'transportation_car': [
      ('Gas Station', 'SHELL OIL', -45.00),
      ('Gas Station', 'CHEVRON', -52.00),
      ('Car Wash', 'CAR WASH EXPRESS', -15.00),
      ('Auto Parts', 'AUTOZONE INC', -85.00)
    ],
    'health_medical_pharmacy': [
      ('CVS Pharmacy', 'CVS PHARMACY', -35.00),
      ('Doctor Visit', 'MEDICAL CENTER', -150.00),
      ('Walgreens', 'WALGREENS STORE', -25.00),
      ('Dental Office', 'SMILES DENTAL', -200.00)
    ],
    'health_gym_wellness': [
      ('Gym Membership', 'FITNESS CENTER', -49.99),
      ('Personal Trainer', 'FITNESS TRAINER', -80.00),
      ('Spa Service', 'RELAX SPA', -120.00)
    ],
    'donations_gifts': [
      ('Charity Donation', 'RED CROSS', -50.00),
      ('Birthday Gift', 'GIFT PURCHASE', -75.00),
      ('Wedding Gift', 'WEDDING REGISTRY', -100.00)
    ],
    'transfers': [
      ('Transfer to Savings', 'ACCOUNT TRANSFER', -500.00),
      ('Credit Card Payment', 'CREDIT CARD PAYMENT', -800.00),
      ('Loan Payment', 'LOAN PAYMENT', -400.00)
    ],
    'miscellaneous': [
      ('ATM Withdrawal', 'ATM CASH WITHDRAWAL', -100.00),
      ('Cash Back', 'CASH BACK PURCHASE', -20.00),
      ('Refund', 'MERCHANT REFUND', 25.00)
    ]
  }
  
  # Generate transactions over the specified time period
  start_date = datetime.now() - timedelta(days=months * 30)
  
  for i in range(transaction_count):
    # Select random category and transaction template
    category = random.choice(list(transaction_templates.keys()))
    name, raw_name, amount = random.choice(transaction_templates[category])
    
    # Select random account
    account_id = random.choice(account_ids)
    
    # Generate random date within the time period
    random_days = random.randint(0, months * 30)
    transaction_date = start_date + timedelta(days=random_days)
    
    # Generate unique transaction ID
    transaction_id = f"TXN_{uuid.uuid4().hex[:8].upper()}"
    
    # Create merged transaction name
    transaction_name = f"{name} [{raw_name}]"
    
    # Create transaction
    db.create_transaction(
      user_id=user_id,
      account_id=account_id,
      transaction_id=transaction_id,
      date=transaction_date.strftime("%Y-%m-%d"),
      transaction_name=transaction_name,
      amount=amount,
      category=category
    )
    transaction_ids.append(transaction_id)
  
  return transaction_ids

def seed_users():
  """Seed the database with test users, accounts, and transactions"""
  print("Starting user seeding process...")
  
  # Reset database
  db = reset_database()
  
  # Create SmallDataUser with 1 account and 100 transactions over 2 months
  small_user_id = db.create_user("SmallDataUser", "small@example.com")
  small_accounts = create_sample_accounts(small_user_id, 1)
  small_transactions = create_sample_transactions(small_user_id, small_accounts, 100, 2)
  print(f"Created SmallDataUser with ID: {small_user_id}")
  print(f"  - {len(small_accounts)} account(s)")
  print(f"  - {len(small_transactions)} transactions over 2 months")
  
  # Create MediumDataUser with 3 accounts and 400 transactions over 3 months
  medium_user_id = db.create_user("MediumDataUser", "medium@example.com")
  medium_accounts = create_sample_accounts(medium_user_id, 3)
  medium_transactions = create_sample_transactions(medium_user_id, medium_accounts, 400, 3)
  print(f"Created MediumDataUser with ID: {medium_user_id}")
  print(f"  - {len(medium_accounts)} account(s)")
  print(f"  - {len(medium_transactions)} transactions over 3 months")
  
  # Create HeavyDataUser with 13 accounts and 1000 transactions over 6 months
  heavy_user_id = db.create_user("HeavyDataUser", "heavy@example.com")
  heavy_accounts = create_sample_accounts(heavy_user_id, 13)
  heavy_transactions = create_sample_transactions(heavy_user_id, heavy_accounts, 1000, 6)
  print(f"Created HeavyDataUser with ID: {heavy_user_id}")
  print(f"  - {len(heavy_accounts)} account(s)")
  print(f"  - {len(heavy_transactions)} transactions over 6 months")
  
  print("\nUser seeding completed successfully!")
  
  # Verify the seeding
  print("\nVerification:")
  small_user = db.get_user("SmallDataUser")
  medium_user = db.get_user("MediumDataUser")
  heavy_user = db.get_user("HeavyDataUser")
  
  print(f"SmallDataUser:")
  print(f"  - Accounts: {len(db.get_accounts_by_user(small_user['id']))}")
  print(f"  - Transactions: {len(db.get_transactions_by_user(small_user['id']))}")
  
  print(f"MediumDataUser:")
  print(f"  - Accounts: {len(db.get_accounts_by_user(medium_user['id']))}")
  print(f"  - Transactions: {len(db.get_transactions_by_user(medium_user['id']))}")
  
  print(f"HeavyDataUser:")
  print(f"  - Accounts: {len(db.get_accounts_by_user(heavy_user['id']))}")
  print(f"  - Transactions: {len(db.get_transactions_by_user(heavy_user['id']))}")
  
  print(f"\nTotal accounts in database: {len(db.get_all_accounts())}")
  print(f"Total transactions in database: {len(db.get_all_transactions())}")
  
  # Show account breakdown by user
  print("\nAccount Breakdown by User:")
  for user in [small_user, medium_user, heavy_user]:
    user_accounts = db.get_accounts_by_user(user['id'])
    print(f"\n{user['username']}:")
    if user_accounts:
      for account in user_accounts:
        print(f"  - {account['account_name']} ({account['account_type']}) - Available: ${account['balance_available']:.2f}, Current: ${account['balance_current']:.2f}")
    else:
      print("  No accounts")
  
  # Show transaction category breakdown
  print("\nTransaction Category Breakdown:")
  all_transactions = db.get_all_transactions()
  category_counts = {}
  for transaction in all_transactions:
    category = transaction['category']
    category_counts[category] = category_counts.get(category, 0) + 1
  
  for category, count in sorted(category_counts.items()):
    print(f"  {category}: {count} transactions")

if __name__ == "__main__":
  seed_users()
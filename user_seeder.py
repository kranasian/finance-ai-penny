import sqlite3
import os
import random
import uuid
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
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

def create_sample_transactions(user_id: int, account_ids: list, transaction_count: int, months: int, start_date: datetime = None) -> list:
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
  if start_date is None:
    start_date = datetime.now() - timedelta(days=months * 30)
  
  end_date = start_date + timedelta(days=months * 30)
  
  for i in range(transaction_count):
    # Select random category and transaction template
    category = random.choice(list(transaction_templates.keys()))
    name, raw_name, amount = random.choice(transaction_templates[category])
    
    # Select random account
    account_id = random.choice(account_ids)
    
    # Generate random date within the time period
    random_days = random.randint(0, months * 30)
    transaction_date = start_date + timedelta(days=random_days)
    
    # Ensure transaction date is within the range
    if transaction_date > end_date:
      transaction_date = end_date
    
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

def create_sample_forecasts(user_id: int) -> list:
  """Create sample monthly and weekly forecasts for a user"""
  db = Database()
  forecast_count = 0
  
  # Category ID mapping (from category_masterlist)
  category_map = {
    'income_salary': 36,
    'income_sidegig': 37,
    'income_business': 38,
    'income_interest': 39,
    'meals_groceries': 4,
    'meals_dining_out': 2,
    'meals_delivered_food': 3,
    'leisure_entertainment': 6,
    'leisure_travel': 7,
    'bills_connectivity': 10,
    'bills_insurance': 11,
    'shelter_home': 15,
    'shelter_utilities': 16,
    'shelter_upkeep': 17,
    'transportation_car': 26,
    'health_gym_wellness': 30,
  }
  
  # Get current date and calculate dates for next 12 months
  today = datetime.now()
  
  # Generate monthly forecasts for next 12 months
  for month_offset in range(1, 13):
    forecast_date = today + relativedelta(months=month_offset)
    # Set to first day of month (YYYY-MM-01)
    month_date = forecast_date.replace(day=1).strftime("%Y-%m-%d")
    
    # Monthly income forecasts
    db.create_monthly_forecast(
      user_id=user_id,
      ai_category_id=category_map['income_salary'],
      month_date=month_date,
      forecasted_amount=3500.00
    )
    forecast_count += 1
    
    # Occasional side gig income (every 3 months)
    if month_offset % 3 == 0:
      db.create_monthly_forecast(
        user_id=user_id,
        ai_category_id=category_map['income_sidegig'],
        month_date=month_date,
        forecasted_amount=200.00
      )
      forecast_count += 1
    
    # Monthly spending forecasts
    monthly_spending = {
      category_map['shelter_home']: 1200.00,
      category_map['shelter_utilities']: 200.00,
      category_map['bills_connectivity']: 80.00,
      category_map['bills_insurance']: 150.00,
      category_map['transportation_car']: 100.00,
      category_map['health_gym_wellness']: 50.00,
      category_map['meals_groceries']: 480.00,  # 4 weeks * 120
      category_map['leisure_entertainment']: 25.00,
    }
    
    for ai_category_id, amount in monthly_spending.items():
      db.create_monthly_forecast(
        user_id=user_id,
        ai_category_id=ai_category_id,
        month_date=month_date,
        forecasted_amount=amount
      )
      forecast_count += 1
  
  # Generate weekly forecasts for next 12 weeks
  for week_offset in range(0, 12):
    # Calculate Sunday date for the week
    forecast_date = today + timedelta(days=week_offset * 7)
    # Find the Sunday of that week (Sunday is weekday 6)
    days_since_sunday = forecast_date.weekday()
    if days_since_sunday == 6:  # Already Sunday
      sunday_date = forecast_date
    else:
      # Go back to previous Sunday
      sunday_date = forecast_date - timedelta(days=days_since_sunday + 1)
    sunday_date_str = sunday_date.strftime("%Y-%m-%d")
    
    # Weekly spending forecasts
    weekly_spending = {
      category_map['meals_groceries']: 120.00,
      category_map['meals_dining_out']: random.uniform(50.00, 150.00),
      category_map['transportation_car']: 25.00,
    }
    
    for ai_category_id, amount in weekly_spending.items():
      db.create_weekly_forecast(
        user_id=user_id,
        ai_category_id=ai_category_id,
        sunday_date=sunday_date_str,
        forecasted_amount=amount
      )
      forecast_count += 1
    
    # Occasional weekly income (side gig every 4 weeks)
    if week_offset % 4 == 0:
      db.create_weekly_forecast(
        user_id=user_id,
        ai_category_id=category_map['income_sidegig'],
        sunday_date=sunday_date_str,
        forecasted_amount=50.00
      )
      forecast_count += 1
  
  return forecast_count

def create_sample_subscriptions(user_id: int) -> int:
  """Create sample subscription data for a user"""
  from database import Database
  from datetime import datetime, timedelta
  import random
  
  db = Database()
  subscription_count = 0
  
  today = datetime.now()
  
  # Sample subscriptions with different frequencies and types
  subscriptions = [
    {
      'name': 'netflix',
      'recurrence_json': {'min': 28, 'mean': 30, 'max': 31},
      'confidence_score_bills': 0.95,
      'next_amount': 15.99,
      'frequency': 'monthly',
      'next_likely_payment_date': (today + timedelta(days=30)).strftime('%Y-%m-%d')
    },
    {
      'name': 'spotify',
      'recurrence_json': {'min': 28, 'mean': 30, 'max': 31},
      'confidence_score_bills': 0.92,
      'next_amount': 9.99,
      'frequency': 'monthly',
      'next_likely_payment_date': (today + timedelta(days=28)).strftime('%Y-%m-%d')
    },
    {
      'name': 'gym membership',
      'recurrence_json': {'min': 28, 'mean': 30, 'max': 31},
      'confidence_score_bills': 0.88,
      'next_amount': 49.99,
      'frequency': 'monthly',
      'next_likely_payment_date': (today + timedelta(days=15)).strftime('%Y-%m-%d')
    },
    {
      'name': 'amazon prime',
      'recurrence_json': {'min': 360, 'mean': 365, 'max': 366},
      'confidence_score_bills': 0.90,
      'next_amount': 139.00,
      'frequency': 'yearly',
      'next_likely_payment_date': (today + timedelta(days=180)).strftime('%Y-%m-%d')
    },
    {
      'name': 'salary',
      'recurrence_json': {'min': 14, 'mean': 14, 'max': 15},
      'confidence_score_salary': 0.98,
      'next_amount': 3500.00,
      'frequency': 'biweekly',
      'next_likely_payment_date': (today + timedelta(days=14)).strftime('%Y-%m-%d')
    },
    {
      'name': 'side gig',
      'recurrence_json': {'min': 28, 'mean': 30, 'max': 31},
      'confidence_score_sidegig': 0.85,
      'next_amount': 200.00,
      'frequency': 'monthly',
      'next_likely_payment_date': (today + timedelta(days=25)).strftime('%Y-%m-%d')
    },
    {
      'name': 'municipal water and sewer',
      'recurrence_json': {'min': 28, 'mean': 30, 'max': 31},
      'confidence_score_bills': 0.95,
      'next_amount': 150.00,
      'frequency': 'monthly',
      'next_likely_payment_date': (today + timedelta(days=12)).strftime('%Y-%m-%d')
    },
    {
      'name': 'mid-carolina electric cooperative',
      'recurrence_json': {'min': 28, 'mean': 30, 'max': 31},
      'confidence_score_bills': 0.92,
      'next_amount': 120.00,
      'frequency': 'monthly',
      'next_likely_payment_date': (today + timedelta(days=15)).strftime('%Y-%m-%d')
    },
    {
      'name': 'phone bill',
      'recurrence_json': {'min': 28, 'mean': 30, 'max': 31},
      'confidence_score_bills': 0.93,
      'next_amount': 80.00,
      'frequency': 'monthly',
      'next_likely_payment_date': (today + timedelta(days=20)).strftime('%Y-%m-%d')
    }
  ]
  
  for sub in subscriptions:
    db.create_subscription(
      user_id=user_id,
      name=sub['name'],
      recurrence_json=sub['recurrence_json'],
      confidence_score_bills=sub.get('confidence_score_bills'),
      confidence_score_salary=sub.get('confidence_score_salary'),
      confidence_score_sidegig=sub.get('confidence_score_sidegig'),
      next_amount=sub.get('next_amount'),
      frequency=sub.get('frequency'),
      next_likely_payment_date=sub.get('next_likely_payment_date')
    )
    subscription_count += 1
  
  return subscription_count


def create_subscription_transactions(user_id: int, account_ids: list, months: int = 6, start_date: datetime = None) -> int:
  """Create subscription transactions that match subscription names"""
  from database import Database
  from datetime import datetime, timedelta
  from dateutil.relativedelta import relativedelta
  import random
  import uuid
  
  db = Database()
  
  # Get subscriptions for this user
  subscriptions = db.get_subscriptions(user_id)
  if not subscriptions:
    return 0
  
  # Subscription name to transaction mapping
  subscription_transaction_map = {
    'netflix': ('Netflix', -15.99, 'leisure_entertainment'),
    'spotify': ('Spotify', -9.99, 'leisure_entertainment'),
    'gym membership': ('Gym Membership', -49.99, 'health_gym_wellness'),
    'amazon prime': ('Amazon Prime', -139.00, 'shopping_gadgets'),
    'salary': ('Salary', 3500.00, 'income_salary'),
    'side gig': ('Side Gig', 200.00, 'income_sidegig'),
    'municipal water and sewer': ('Municipal Water And Sewer', -150.00, 'shelter_utilities'),
    'mid-carolina electric cooperative': ('Mid-Carolina Electric Cooperative', -120.00, 'shelter_utilities'),
    'phone bill': ('Phone Bill', -80.00, 'bills_connectivity')
  }
  
  # Determine date range - ensure we cover all calendar months from start_date to today
  if start_date is None:
    today = datetime.now()
    # Start from months ago
    start_date = today - relativedelta(months=months)
  else:
    today = datetime.now()
  
  # Ensure start_date is the first day of that month
  start_date = start_date.replace(day=1)
  end_date = today
  
  transaction_count = 0
  
  for subscription in subscriptions:
    sub_name = subscription.get('name', '').lower()
    if sub_name not in subscription_transaction_map:
      continue
    
    transaction_name, amount, category = subscription_transaction_map[sub_name]
    
    # Determine frequency based on subscription
    if sub_name in ['netflix', 'spotify', 'gym membership', 'municipal water and sewer', 'mid-carolina electric cooperative', 'phone bill', 'side gig']:
      # Monthly subscriptions - create one transaction per calendar month
      # Start from the first day of the start month
      current_month = start_date.replace(day=1)
      end_month = end_date.replace(day=1)
      
      # Iterate through each calendar month from start to end (inclusive)
      while current_month <= end_month:
        # For the current month, only create transactions up to today's date
        # For past months, create transactions throughout the month
        if current_month.month == end_date.month and current_month.year == end_date.year:
          # Current month - only create transactions up to today
          max_day = min(28, end_date.day)  # Use today's day or 28, whichever is smaller
          random_day = random.randint(1, max_day)
        else:
          # Past months - can create transactions throughout the month
          random_day = random.randint(1, 28)
        
        transaction_date = current_month.replace(day=random_day)
        
        # Only create if transaction date is not in the future and within our range
        if transaction_date <= end_date and transaction_date >= start_date:
          # Select random account
          account_id = random.choice(account_ids)
          
          # Generate unique transaction ID
          transaction_id = f"TXN_{uuid.uuid4().hex[:8].upper()}"
          
          # Create transaction with exact subscription name (lowercase) so it matches
          db.create_transaction(
            user_id=user_id,
            account_id=account_id,
            transaction_id=transaction_id,
            date=transaction_date.strftime("%Y-%m-%d"),
            transaction_name=transaction_name.lower(),  # Use lowercase to match subscription name
            amount=amount,
            category=category
          )
          transaction_count += 1
        
        # Move to next month
        current_month = current_month + relativedelta(months=1)
        
    elif sub_name == 'salary':
      # Biweekly salary - create one transaction every 14 days
      current_date = start_date
      while current_date <= end_date:
        # Select random account
        account_id = random.choice(account_ids)
        
        # Generate unique transaction ID
        transaction_id = f"TXN_{uuid.uuid4().hex[:8].upper()}"
        
        # Create transaction
        db.create_transaction(
          user_id=user_id,
          account_id=account_id,
          transaction_id=transaction_id,
          date=current_date.strftime("%Y-%m-%d"),
          transaction_name=transaction_name.lower(),
          amount=amount,
          category=category
        )
        transaction_count += 1
        
        # Move to next occurrence (14 days)
        current_date += timedelta(days=14)
        
    elif sub_name == 'amazon prime':
      # Yearly subscription - create one transaction per year within the range
      current_date = start_date.replace(day=1)
      while current_date <= end_date:
        # Create one transaction per year
        random_day = random.randint(1, 28)
        transaction_date = current_date.replace(day=random_day)
        
        if transaction_date <= end_date and transaction_date >= start_date:
          account_id = random.choice(account_ids)
          transaction_id = f"TXN_{uuid.uuid4().hex[:8].upper()}"
          
          db.create_transaction(
            user_id=user_id,
            account_id=account_id,
            transaction_id=transaction_id,
            date=transaction_date.strftime("%Y-%m-%d"),
            transaction_name=transaction_name.lower(),
            amount=amount,
            category=category
          )
          transaction_count += 1
        
        # Move to next year
        current_date = current_date + relativedelta(years=1)
  
  return transaction_count

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
  
  # Add transactions from last year (12 months ago)
  last_year_start = datetime.now() - relativedelta(years=1)
  months_diff = 12
  heavy_transactions_last_year = create_sample_transactions(heavy_user_id, heavy_accounts, 500, months_diff, start_date=last_year_start)
  
  # Create forecast data for HeavyDataUser
  forecast_count = create_sample_forecasts(heavy_user_id)
  
  # Create subscription data for HeavyDataUser
  subscription_count = create_sample_subscriptions(heavy_user_id)
  
  # Create subscription transactions that match subscription names
  # Create subscription transactions for the last 6 months (calendar months) including current month
  today = datetime.now()
  six_months_ago = today - relativedelta(months=6)
  subscription_transaction_count = create_subscription_transactions(heavy_user_id, heavy_accounts, months=6, start_date=six_months_ago.replace(day=1))
  
  # Also create subscription transactions for last year (12 months ago, calendar months)
  last_year_start = datetime.now() - relativedelta(years=1)
  subscription_transaction_count_last_year = create_subscription_transactions(heavy_user_id, heavy_accounts, months=12, start_date=last_year_start.replace(day=1))
  
  print(f"Created HeavyDataUser with ID: {heavy_user_id}")
  print(f"  - {len(heavy_accounts)} account(s)")
  print(f"  - {len(heavy_transactions)} transactions over 6 months")
  print(f"  - {len(heavy_transactions_last_year)} transactions from last year")
  print(f"  - {forecast_count} forecasts (monthly and weekly)")
  print(f"  - {subscription_count} subscriptions")
  print(f"  - {subscription_transaction_count} subscription transactions (last 6 months)")
  print(f"  - {subscription_transaction_count_last_year} subscription transactions (last year)")
  
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
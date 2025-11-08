import sqlite3
import os
from datetime import datetime
from typing import List, Dict, Optional

class Database:
  def __init__(self, db_path: str = "chatbot.db"):
    self.db_path = db_path
    self.init_database()
  
  def init_database(self):
    """Initialize the database with required tables"""
    conn = sqlite3.connect(self.db_path)
    cursor = conn.cursor()
    
    # Create users table
    cursor.execute('''
      CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        email TEXT UNIQUE NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
      )
    ''')

    # Create accounts table
    cursor.execute('''
      CREATE TABLE IF NOT EXISTS accounts (
        account_id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        account_type TEXT NOT NULL CHECK (account_type IN ('deposit_savings', 'deposit_money_market', 'deposit_checking', 'credit_card', 'loan_home_equity', 'loan_line_of_credit', 'loan_mortgage', 'loan_auto')),
        balance_available REAL DEFAULT 0,
        balance_current REAL DEFAULT 0,
        balance_limit REAL DEFAULT 0,
        account_name TEXT NOT NULL,
        account_mask TEXT NOT NULL,
        FOREIGN KEY (user_id) REFERENCES users (id)
      )
    ''')
    
    # Create transactions table
    cursor.execute('''
      CREATE TABLE IF NOT EXISTS transactions (
        transaction_id INTEGER PRIMARY KEY,
        user_id INTEGER NOT NULL,
        account_id INTEGER NOT NULL,
        date DATE NOT NULL,
        transaction_name TEXT NOT NULL,
        amount REAL NOT NULL,
        category TEXT NOT NULL,
        FOREIGN KEY (user_id) REFERENCES users (id),
        FOREIGN KEY (account_id) REFERENCES accounts (account_id)
      )
    ''')
    
    # Create ai_monthly_forecasts table
    cursor.execute('''
      CREATE TABLE IF NOT EXISTS ai_monthly_forecasts (
        user_id INTEGER NOT NULL,
        ai_category_id INTEGER NOT NULL,
        month_date DATE NOT NULL,
        forecasted_amount REAL NOT NULL,
        PRIMARY KEY (user_id, ai_category_id, month_date),
        FOREIGN KEY (user_id) REFERENCES users (id)
      )
    ''')
    
    # Create ai_weekly_forecasts table
    cursor.execute('''
      CREATE TABLE IF NOT EXISTS ai_weekly_forecasts (
        user_id INTEGER NOT NULL,
        ai_category_id INTEGER NOT NULL,
        sunday_date DATE NOT NULL,
        forecasted_amount REAL NOT NULL,
        PRIMARY KEY (user_id, ai_category_id, sunday_date),
        FOREIGN KEY (user_id) REFERENCES users (id)
      )
    ''')
    
    # Create user_recurring_transactions table
    cursor.execute('''
      CREATE TABLE IF NOT EXISTS user_recurring_transactions (
        user_id INTEGER NOT NULL,
        name TEXT NOT NULL,
        last_reviewed_datetime TIMESTAMP,
        recurrence_json TEXT NOT NULL,
        confidence_score_bills REAL,
        confidence_score_salary REAL,
        confidence_score_sidegig REAL,
        reviewer_bills INTEGER DEFAULT 0,
        reviewer_salary INTEGER DEFAULT 0,
        reviewer_sidegig INTEGER DEFAULT 0,
        user_not_bills INTEGER,
        user_not_salary INTEGER,
        user_not_sidegig INTEGER,
        user_need_level TEXT CHECK (user_need_level IN ('need', 'want', 'trial')),
        necessity TEXT CHECK (necessity IN ('need', 'want', 'trial')),
        next_user_remind_datetime TIMESTAMP,
        user_remind_last_response TEXT,
        user_cancelled_date DATE,
        next_likely_payment_date DATE,
        next_earliest_payment_date DATE,
        next_latest_payment_date DATE,
        next_payment_date_debug TEXT,
        next_amount REAL,
        last_transaction_date TIMESTAMP,
        awaiting_new_transaction INTEGER,
        necessity_update_date TIMESTAMP,
        necessity_prompt_date TIMESTAMP,
        frequency TEXT CHECK (frequency IN ('weekly', 'biweekly', 'monthly', 'quarterly', 'biannual', 'yearly', 'irregular', 'no_pattern')),
        PRIMARY KEY (user_id, name),
        FOREIGN KEY (user_id) REFERENCES users (id)
      )
    ''')
    
    conn.commit()
    conn.close()
  
  def create_user(self, username: str, email: str) -> int:
    """Create a new user and return user ID"""
    conn = sqlite3.connect(self.db_path)
    cursor = conn.cursor()
    
    try:
      cursor.execute(
        "INSERT INTO users (username, email) VALUES (?, ?)",
        (username, email)
      )
      user_id = cursor.lastrowid
      conn.commit()
      return user_id
    except sqlite3.IntegrityError:
      # User already exists, get their ID
      cursor.execute("SELECT id FROM users WHERE username = ?", (username,))
      result = cursor.fetchone()
      return result[0] if result else None
    finally:
      conn.close()
  
  def get_user(self, username: str) -> Optional[Dict]:
    """Get user by username"""
    conn = sqlite3.connect(self.db_path)
    cursor = conn.cursor()
    
    cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
    result = cursor.fetchone()
    conn.close()
    
    if result:
      return {
        'id': result[0],
        'username': result[1],
        'email': result[2],
        'created_at': result[3]
      }
    return None
  
  def get_all_users(self) -> List[Dict]:
    """Get all users from the database"""
    conn = sqlite3.connect(self.db_path)
    cursor = conn.cursor()
    
    cursor.execute("SELECT * FROM users ORDER BY username")
    results = cursor.fetchall()
    conn.close()
    
    users = []
    for result in results:
      users.append({
        'id': result[0],
        'username': result[1],
        'email': result[2],
        'created_at': result[3]
      })
    
    return users

  # Account management methods
  def create_account(self, user_id: int, account_type: str, 
                    balance_available: float, balance_current: float, 
                    account_name: str, account_mask: str, balance_limit: Optional[float] = None) -> int:
    """Create a new account and return account ID"""
    conn = sqlite3.connect(self.db_path)
    cursor = conn.cursor()
    
    cursor.execute(
      "INSERT INTO accounts (user_id, account_type, balance_available, balance_current, balance_limit, account_name, account_mask) VALUES (?, ?, ?, ?, ?, ?, ?)",
      (user_id, account_type, balance_available, balance_current, balance_limit, account_name, account_mask)
    )
    account_id = cursor.lastrowid
    conn.commit()
    conn.close()
    
    return account_id

  def get_account(self, account_id: int) -> Optional[Dict]:
    """Get account by ID"""
    conn = sqlite3.connect(self.db_path)
    cursor = conn.cursor()
    
    cursor.execute("SELECT account_id, user_id, account_type, balance_available, balance_current, balance_limit, account_name, account_mask FROM accounts WHERE account_id = ?", (account_id,))
    result = cursor.fetchone()
    conn.close()
    
    if result:
      return {
        'account_id': result[0],
        'user_id': result[1],
        'account_type': result[2],
        'balance_available': result[3],
        'balance_current': result[4],
        'balance_limit': result[5],
        'account_name': result[6],
        'account_mask': result[7]
      }
    return None

  def get_accounts_by_user(self, user_id: int) -> List[Dict]:
    """Get all accounts for a specific user"""
    conn = sqlite3.connect(self.db_path)
    cursor = conn.cursor()
    
    cursor.execute("SELECT account_id, user_id, account_type, balance_available, balance_current, balance_limit, account_name, account_mask FROM accounts WHERE user_id = ? ORDER BY account_type, account_name", (user_id,))
    results = cursor.fetchall()
    conn.close()
    
    accounts = []
    for result in results:
      accounts.append({
        'account_id': result[0],
        'user_id': result[1],
        'account_type': result[2],
        'balance_available': result[3],
        'balance_current': result[4],
        'balance_limit': result[5],
        'account_name': result[6],
        'account_mask': result[7]
      })
    
    return accounts

  def get_all_accounts(self) -> List[Dict]:
    """Get all accounts from the database"""
    conn = sqlite3.connect(self.db_path)
    cursor = conn.cursor()
    
    cursor.execute("SELECT account_id, user_id, account_type, balance_available, balance_current, balance_limit, account_name, account_mask FROM accounts ORDER BY user_id, account_type, account_name")
    results = cursor.fetchall()
    conn.close()
    
    accounts = []
    for result in results:
      accounts.append({
        'account_id': result[0],
        'user_id': result[1],
        'account_type': result[2],
        'balance_available': result[3],
        'balance_current': result[4],
        'balance_limit': result[5],
        'account_name': result[6],
        'account_mask': result[7]
      })
    
    return accounts

  # Transaction management methods
  def create_transaction(self, user_id: int, account_id: int, transaction_id: int,
                        date: str, transaction_name: str, amount: float, category: str) -> int:
    """Create a new transaction and return transaction ID"""
    conn = sqlite3.connect(self.db_path)
    cursor = conn.cursor()
    
    cursor.execute(
      "INSERT INTO transactions (transaction_id, user_id, account_id, date, transaction_name, amount, category) VALUES (?, ?, ?, ?, ?, ?, ?)",
      (transaction_id, user_id, account_id, date, transaction_name, amount, category)
    )
    conn.commit()
    conn.close()
    
    return transaction_id

  def get_transaction(self, transaction_id: int) -> Optional[Dict]:
    """Get transaction by ID"""
    conn = sqlite3.connect(self.db_path)
    cursor = conn.cursor()
    
    cursor.execute("SELECT * FROM transactions WHERE transaction_id = ?", (transaction_id,))
    result = cursor.fetchone()
    conn.close()
    
    if result:
      return {
        'transaction_id': result[0],
        'user_id': result[1],
        'account_id': result[2],
        'date': result[3],
        'transaction_name': result[4],
        'amount': result[5],
        'category': result[6]
      }
    return None

  def get_transactions_by_user(self, user_id: int) -> List[Dict]:
    """Get all transactions for a specific user"""
    # Category name to ID mapping
    _CATEGORY_NAME_TO_ID = {
      'meals': 1,
      'meals_groceries': 4,
      'meals_dining_out': 2,
      'meals_delivered_food': 3,
      'leisure': 5,
      'leisure_entertainment': 6,
      'leisure_travel': 7,
      'bills': 9,
      'bills_connectivity': 10,
      'bills_insurance': 11,
      'bills_tax': 12,
      'bills_service_fees': 13,
      'shelter': 14,
      'shelter_home': 15,
      'shelter_utilities': 16,
      'shelter_upkeep': 17,
      'education': 18,
      'education_kids_activities': 19,
      'education_tuition': 20,
      'shopping': 21,
      'shopping_clothing': 22,
      'shopping_gadgets': 23,
      'shopping_kids': 24,
      'shopping_pets': 8,
      'transportation': 25,
      'transportation_public': 27,
      'transportation_car': 26,
      'health': 28,
      'health_medical_pharmacy': 29,
      'health_gym_wellness': 30,
      'health_personal_care': 31,
      'donations_gifts': 32,
      'income': 47,
      'income_salary': 36,
      'income_sidegig': 37,
      'income_business': 38,
      'income_interest': 39,
      'uncategorized': -1,
      'transfers': 45,
      'miscellaneous': 33,
    }
    
    conn = sqlite3.connect(self.db_path)
    cursor = conn.cursor()
    
    cursor.execute("SELECT transaction_id, user_id, account_id, date, transaction_name, amount, category FROM transactions WHERE user_id = ? ORDER BY date DESC", (user_id,))
    results = cursor.fetchall()
    conn.close()
    
    transactions = []
    for result in results:
      category_name = result[6]
      ai_category_id = _CATEGORY_NAME_TO_ID.get(category_name, -1)  # Default to uncategorized if not found
      transactions.append({
        'transaction_id': result[0],
        'user_id': result[1],
        'account_id': result[2],
        'date': result[3],
        'transaction_name': result[4],
        'amount': result[5],
        'category': category_name,
        'ai_category_id': ai_category_id
      })
    
    return transactions

  def get_transactions_by_account(self, account_id: int) -> List[Dict]:
    """Get all transactions for a specific account"""
    conn = sqlite3.connect(self.db_path)
    cursor = conn.cursor()
    
    cursor.execute("SELECT transaction_id, user_id, account_id, date, transaction_name, amount, category FROM transactions WHERE account_id = ? ORDER BY date DESC", (account_id,))
    results = cursor.fetchall()
    conn.close()
    
    transactions = []
    for result in results:
      transactions.append({
        'transaction_id': result[0],
        'user_id': result[1],
        'account_id': result[2],
        'date': result[3],
        'transaction_name': result[4],
        'amount': result[5],
        'category': result[6]
      })
    
    return transactions

  def get_all_transactions(self) -> List[Dict]:
    """Get all transactions from the database"""
    conn = sqlite3.connect(self.db_path)
    cursor = conn.cursor()
    
    cursor.execute("SELECT transaction_id, user_id, account_id, date, transaction_name, amount, category FROM transactions ORDER BY date DESC")
    results = cursor.fetchall()
    conn.close()
    
    transactions = []
    for result in results:
      transactions.append({
        'transaction_id': result[0],
        'user_id': result[1],
        'account_id': result[2],
        'date': result[3],
        'transaction_name': result[4],
        'amount': result[5],
        'category': result[6]
      })
    
    return transactions

  # AI Monthly Forecasts management methods
  def create_monthly_forecast(self, user_id: int, ai_category_id: int, month_date: str, forecasted_amount: float) -> None:
    """Create or update a monthly forecast"""
    conn = sqlite3.connect(self.db_path)
    cursor = conn.cursor()
    
    cursor.execute(
      "INSERT OR REPLACE INTO ai_monthly_forecasts (user_id, ai_category_id, month_date, forecasted_amount) VALUES (?, ?, ?, ?)",
      (user_id, ai_category_id, month_date, forecasted_amount)
    )
    conn.commit()
    conn.close()

  def get_monthly_forecasts_by_user(self, user_id: int) -> List[Dict]:
    """Get all monthly forecasts for a specific user"""
    conn = sqlite3.connect(self.db_path)
    cursor = conn.cursor()
    
    cursor.execute(
      "SELECT user_id, ai_category_id, month_date, forecasted_amount FROM ai_monthly_forecasts WHERE user_id = ? ORDER BY month_date ASC, ai_category_id ASC",
      (user_id,)
    )
    results = cursor.fetchall()
    conn.close()
    
    forecasts = []
    for result in results:
      forecasts.append({
        'user_id': result[0],
        'ai_category_id': result[1],
        'month_date': result[2],
        'forecasted_amount': result[3]
      })
    
    return forecasts

  def get_all_monthly_forecasts(self) -> List[Dict]:
    """Get all monthly forecasts from the database"""
    conn = sqlite3.connect(self.db_path)
    cursor = conn.cursor()
    
    cursor.execute(
      "SELECT user_id, ai_category_id, month_date, forecasted_amount FROM ai_monthly_forecasts ORDER BY month_date ASC, ai_category_id ASC"
    )
    results = cursor.fetchall()
    conn.close()
    
    forecasts = []
    for result in results:
      forecasts.append({
        'user_id': result[0],
        'ai_category_id': result[1],
        'month_date': result[2],
        'forecasted_amount': result[3]
      })
    
    return forecasts

  # AI Weekly Forecasts management methods
  def create_weekly_forecast(self, user_id: int, ai_category_id: int, sunday_date: str, forecasted_amount: float) -> None:
    """Create or update a weekly forecast"""
    conn = sqlite3.connect(self.db_path)
    cursor = conn.cursor()
    
    cursor.execute(
      "INSERT OR REPLACE INTO ai_weekly_forecasts (user_id, ai_category_id, sunday_date, forecasted_amount) VALUES (?, ?, ?, ?)",
      (user_id, ai_category_id, sunday_date, forecasted_amount)
    )
    conn.commit()
    conn.close()

  def get_weekly_forecasts_by_user(self, user_id: int) -> List[Dict]:
    """Get all weekly forecasts for a specific user"""
    conn = sqlite3.connect(self.db_path)
    cursor = conn.cursor()
    
    cursor.execute(
      "SELECT user_id, ai_category_id, sunday_date, forecasted_amount FROM ai_weekly_forecasts WHERE user_id = ? ORDER BY sunday_date ASC, ai_category_id ASC",
      (user_id,)
    )
    results = cursor.fetchall()
    conn.close()
    
    forecasts = []
    for result in results:
      forecasts.append({
        'user_id': result[0],
        'ai_category_id': result[1],
        'sunday_date': result[2],
        'forecasted_amount': result[3]
      })
    
    return forecasts

  def get_all_weekly_forecasts(self) -> List[Dict]:
    """Get all weekly forecasts from the database"""
    conn = sqlite3.connect(self.db_path)
    cursor = conn.cursor()
    
    cursor.execute(
      "SELECT user_id, ai_category_id, sunday_date, forecasted_amount FROM ai_weekly_forecasts ORDER BY sunday_date ASC, ai_category_id ASC"
    )
    results = cursor.fetchall()
    conn.close()
    
    forecasts = []
    for result in results:
      forecasts.append({
        'user_id': result[0],
        'ai_category_id': result[1],
        'sunday_date': result[2],
        'forecasted_amount': result[3]
      })
    
    return forecasts

  # Subscription management methods
  def create_subscription(self, user_id: int, name: str, recurrence_json: dict,
                         confidence_score_bills: Optional[float] = None,
                         confidence_score_salary: Optional[float] = None,
                         confidence_score_sidegig: Optional[float] = None,
                         next_amount: Optional[float] = None,
                         frequency: Optional[str] = None,
                         next_likely_payment_date: Optional[str] = None) -> None:
    """Create or update a subscription"""
    conn = sqlite3.connect(self.db_path)
    cursor = conn.cursor()
    
    import json
    recurrence_json_str = json.dumps(recurrence_json)
    
    cursor.execute('''
      INSERT OR REPLACE INTO user_recurring_transactions 
      (user_id, name, recurrence_json, confidence_score_bills, confidence_score_salary, 
       confidence_score_sidegig, next_amount, frequency, next_likely_payment_date)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (user_id, name.lower(), recurrence_json_str, confidence_score_bills,
          confidence_score_salary, confidence_score_sidegig, next_amount, frequency, next_likely_payment_date))
    
    conn.commit()
    conn.close()

  def get_subscription_transactions(self, user_id: int, confidence_score_bills_threshold: float = 0.5) -> List[Dict]:
    """Get subscription transactions by joining transactions with user_recurring_transactions"""
    conn = sqlite3.connect(self.db_path)
    cursor = conn.cursor()
    
    cursor.execute('''
      SELECT DISTINCT t.transaction_id, t.user_id, t.account_id, t.date, t.transaction_name, t.amount, t.category,
             urt.name as subscription_name, urt.confidence_score_bills, urt.reviewer_bills
      FROM transactions t
      INNER JOIN user_recurring_transactions urt
          ON lower(t.transaction_name) = urt.name
      WHERE t.user_id = ?
          AND t.user_id = urt.user_id
          AND ((urt.confidence_score_bills > ?)
          OR (urt.reviewer_bills = 1))
      ORDER BY t.date DESC
    ''', (user_id, confidence_score_bills_threshold))
    
    results = cursor.fetchall()
    conn.close()
    
    transactions = []
    for result in results:
      transactions.append({
        'transaction_id': result[0],
        'user_id': result[1],
        'account_id': result[2],
        'date': result[3],
        'transaction_name': result[4],
        'amount': result[5],
        'category': result[6],
        'subscription_name': result[7],
        'confidence_score_bills': result[8],
        'reviewer_bills': bool(result[9]) if result[9] is not None else None
      })
    
    return transactions

  def get_subscriptions(self, user_id: int) -> List[Dict]:
    """Get all subscriptions for a specific user"""
    conn = sqlite3.connect(self.db_path)
    cursor = conn.cursor()
    
    cursor.execute('''
      SELECT user_id, name, last_reviewed_datetime, recurrence_json, 
             confidence_score_bills, confidence_score_salary, confidence_score_sidegig,
             reviewer_bills, reviewer_salary, reviewer_sidegig,
             user_not_bills, user_not_salary, user_not_sidegig,
             user_need_level, necessity, next_user_remind_datetime,
             user_remind_last_response, user_cancelled_date,
             next_likely_payment_date, next_earliest_payment_date, next_latest_payment_date,
             next_payment_date_debug, next_amount, last_transaction_date,
             awaiting_new_transaction, necessity_update_date, necessity_prompt_date
      FROM user_recurring_transactions 
      WHERE user_id = ? 
      ORDER BY name ASC
    ''', (user_id,))
    
    results = cursor.fetchall()
    conn.close()
    
    import json
    subscriptions = []
    for result in results:
      recurrence_json = json.loads(result[3]) if result[3] else {}
      subscriptions.append({
        'user_id': result[0],
        'name': result[1],
        'last_reviewed_datetime': result[2],
        'recurrence_json': recurrence_json,
        'recurrence_min': recurrence_json.get('min'),
        'recurrence_mean': recurrence_json.get('mean'),
        'recurrence_max': recurrence_json.get('max'),
        'confidence_score_bills': result[4],
        'confidence_score_salary': result[5],
        'confidence_score_sidegig': result[6],
        'reviewer_bills': bool(result[7]) if result[7] is not None else None,
        'reviewer_salary': bool(result[8]) if result[8] is not None else None,
        'reviewer_sidegig': bool(result[9]) if result[9] is not None else None,
        'user_not_bills': bool(result[10]) if result[10] is not None else None,
        'user_not_salary': bool(result[11]) if result[11] is not None else None,
        'user_not_sidegig': bool(result[12]) if result[12] is not None else None,
        'user_need_level': result[13],
        'necessity': result[14],
        'next_user_remind_datetime': result[15],
        'user_remind_last_response': result[16],
        'user_cancelled_date': result[17],
        'next_likely_payment_date': result[18],
        'next_earliest_payment_date': result[19],
        'next_latest_payment_date': result[20],
        'next_payment_date_debug': result[21],
        'next_amount': result[22],
        'last_transaction_date': result[23],
        'awaiting_new_transaction': bool(result[24]) if result[24] is not None else None,
        'necessity_update_date': result[25],
        'necessity_prompt_date': result[26]
      })
    
    return subscriptions


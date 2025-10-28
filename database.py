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
        account_name TEXT NOT NULL,
        account_mask TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users (id)
      )
    ''')
    
    # Create transactions table
    cursor.execute('''
      CREATE TABLE IF NOT EXISTS transactions (
        transaction_id TEXT PRIMARY KEY UNIQUE,
        user_id INTEGER NOT NULL,
        account_id INTEGER NOT NULL,
        date DATE NOT NULL,
        transaction_name TEXT NOT NULL,
        amount REAL NOT NULL,
        category TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users (id),
        FOREIGN KEY (account_id) REFERENCES accounts (account_id)
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
                    account_name: str, account_mask: str) -> int:
    """Create a new account and return account ID"""
    conn = sqlite3.connect(self.db_path)
    cursor = conn.cursor()
    
    cursor.execute(
      "INSERT INTO accounts (user_id, account_type, balance_available, balance_current, account_name, account_mask) VALUES (?, ?, ?, ?, ?, ?)",
      (user_id, account_type, balance_available, balance_current, account_name, account_mask)
    )
    account_id = cursor.lastrowid
    conn.commit()
    conn.close()
    
    return account_id

  def get_account(self, account_id: int) -> Optional[Dict]:
    """Get account by ID"""
    conn = sqlite3.connect(self.db_path)
    cursor = conn.cursor()
    
    cursor.execute("SELECT * FROM accounts WHERE account_id = ?", (account_id,))
    result = cursor.fetchone()
    conn.close()
    
    if result:
      return {
        'account_id': result[0],
        'user_id': result[1],
        'account_type': result[2],
        'balance_available': result[3],
        'balance_current': result[4],
        'account_name': result[5],
        'account_mask': result[6],
        'created_at': result[7]
      }
    return None

  def get_accounts_by_user(self, user_id: int) -> List[Dict]:
    """Get all accounts for a specific user"""
    conn = sqlite3.connect(self.db_path)
    cursor = conn.cursor()
    
    cursor.execute("SELECT * FROM accounts WHERE user_id = ? ORDER BY account_type, account_name", (user_id,))
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
        'account_name': result[5],
        'account_mask': result[6],
        'created_at': result[7]
      })
    
    return accounts

  def get_all_accounts(self) -> List[Dict]:
    """Get all accounts from the database"""
    conn = sqlite3.connect(self.db_path)
    cursor = conn.cursor()
    
    cursor.execute("SELECT * FROM accounts ORDER BY user_id, account_type, account_name")
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
        'account_name': result[5],
        'account_mask': result[6],
        'created_at': result[7]
      })
    
    return accounts

  # Transaction management methods
  def create_transaction(self, user_id: int, account_id: int, transaction_id: str,
                        date: str, transaction_name: str, amount: float, category: str) -> str:
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

  def get_transaction(self, transaction_id: str) -> Optional[Dict]:
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
        'category': result[6],
        'created_at': result[7]
      }
    return None

  def get_transactions_by_user(self, user_id: int) -> List[Dict]:
    """Get all transactions for a specific user"""
    conn = sqlite3.connect(self.db_path)
    cursor = conn.cursor()
    
    cursor.execute("SELECT * FROM transactions WHERE user_id = ? ORDER BY date DESC", (user_id,))
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
        'created_at': result[7]
      })
    
    return transactions

  def get_transactions_by_account(self, account_id: int) -> List[Dict]:
    """Get all transactions for a specific account"""
    conn = sqlite3.connect(self.db_path)
    cursor = conn.cursor()
    
    cursor.execute("SELECT * FROM transactions WHERE account_id = ? ORDER BY date DESC", (account_id,))
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
        'created_at': result[7]
      })
    
    return transactions

  def get_all_transactions(self) -> List[Dict]:
    """Get all transactions from the database"""
    conn = sqlite3.connect(self.db_path)
    cursor = conn.cursor()
    
    cursor.execute("SELECT * FROM transactions ORDER BY date DESC")
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
        'created_at': result[7]
      })
    
    return transactions

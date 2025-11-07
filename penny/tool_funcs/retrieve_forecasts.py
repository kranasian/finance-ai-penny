from database import Database
import pandas as pd
from penny.tool_funcs.sandbox_logging import log

# Category ID to name mapping (from categories.py)
_CATEGORY_ID_TO_NAME = {
    -1: 'Uncategorized',
    1: 'Meals',
    2: 'Dining Out',
    3: 'Delivered Food',
    4: 'Groceries',
    5: 'Leisure',
    6: 'Entertainment',
    7: 'Travel & Vacations',
    8: 'Pets',
    9: 'Bills',
    10: 'Connectivity',
    11: 'Insurance',
    12: 'Taxes',
    13: 'Service Fees',
    14: 'Shelter',
    15: 'Home',
    16: 'Utilities',
    17: 'Upkeep',
    18: 'Education',
    19: 'Kids Activities',
    20: 'Tuition',
    21: 'Shopping',
    22: 'Clothing',
    23: 'Gadgets',
    24: 'Kids',
    25: 'Transport',
    26: 'Car & Fuel',
    27: 'Public Transit',
    28: 'Health',
    29: 'Medical & Pharmacy',
    30: 'Gym & Wellness',
    31: 'Personal Care',
    32: 'Donations & Gifts',
    33: 'Miscellaneous',
    36: 'Salary',
    37: 'Side-Gig',
    38: 'Business',
    39: 'Interest',
    41: 'Food',
    42: 'Others',
    43: 'Bills',
    44: 'Shopping',
    45: 'Transfer',
    46: 'Income',
    47: 'Income',
}


def retrieve_spending_forecasts_function_code_gen(user_id: int = 1, granularity: str = 'monthly') -> pd.DataFrame:
  """Function to retrieve spending forecasts from the database for a specific user"""
  db = Database()
  
  if granularity not in ['monthly', 'weekly']:
    log(f"**Error**: Invalid granularity '{granularity}'. Must be 'monthly' or 'weekly'.")
    return pd.DataFrame()
  
  # Income category IDs (36: salary, 37: sidegig, 38: business, 39: interest)
  income_category_ids = [36, 37, 38, 39]
  
  if granularity == 'monthly':
    forecasts = db.get_monthly_forecasts_by_user(user_id=user_id)
    date_col = 'month_date'
  else:  # weekly
    forecasts = db.get_weekly_forecasts_by_user(user_id=user_id)
    date_col = 'sunday_date'
  
  if not forecasts:
    log(f"**Retrieved Spending Forecasts** of `U-{user_id}` (granularity: {granularity}): No forecasts found")
    if granularity == 'monthly':
      return pd.DataFrame(columns=['user_id', 'ai_category_id', 'month_date', 'forecasted_amount', 'category'])
    else:
      return pd.DataFrame(columns=['user_id', 'ai_category_id', 'sunday_date', 'forecasted_amount', 'category'])
  
  df = pd.DataFrame(forecasts)
  
  # Filter out income category IDs (keep only spending)
  df = df[~df['ai_category_id'].isin(income_category_ids)]
  
  if df.empty:
    log(f"**Retrieved Spending Forecasts** of `U-{user_id}` (granularity: {granularity}): No spending forecasts found")
    if granularity == 'monthly':
      return pd.DataFrame(columns=['user_id', 'ai_category_id', 'month_date', 'forecasted_amount', 'category'])
    else:
      return pd.DataFrame(columns=['user_id', 'ai_category_id', 'sunday_date', 'forecasted_amount', 'category'])
  
  # Add category names by mapping ai_category_id
  df['category'] = df['ai_category_id'].map(_CATEGORY_ID_TO_NAME).fillna('Unknown')
  
  # Convert date column to datetime for proper comparisons
  if date_col in df.columns and len(df) > 0:
    df[date_col] = pd.to_datetime(df[date_col])
  
  log(f"**Retrieved Spending Forecasts** of `U-{user_id}` (granularity: {granularity}): `df: {df.shape}` w/ **cols**:\n  - `{'`, `'.join(df.columns)}`")
  return df


def retrieve_income_forecasts_function_code_gen(user_id: int = 1, granularity: str = 'monthly') -> pd.DataFrame:
  """Function to retrieve income forecasts from the database for a specific user"""
  db = Database()
  
  if granularity not in ['monthly', 'weekly']:
    log(f"**Error**: Invalid granularity '{granularity}'. Must be 'monthly' or 'weekly'.")
    return pd.DataFrame()
  
  # Income category IDs (36: salary, 37: sidegig, 38: business, 39: interest)
  income_category_ids = [36, 37, 38, 39]
  
  if granularity == 'monthly':
    forecasts = db.get_monthly_forecasts_by_user(user_id=user_id)
    date_col = 'month_date'
  else:  # weekly
    forecasts = db.get_weekly_forecasts_by_user(user_id=user_id)
    date_col = 'sunday_date'
  
  if not forecasts:
    log(f"**Retrieved Income Forecasts** of `U-{user_id}` (granularity: {granularity}): No forecasts found")
    if granularity == 'monthly':
      return pd.DataFrame(columns=['user_id', 'ai_category_id', 'month_date', 'forecasted_amount', 'category'])
    else:
      return pd.DataFrame(columns=['user_id', 'ai_category_id', 'sunday_date', 'forecasted_amount', 'category'])
  
  df = pd.DataFrame(forecasts)
  
  # Filter for income category IDs only
  df = df[df['ai_category_id'].isin(income_category_ids)]
  
  if df.empty:
    log(f"**Retrieved Income Forecasts** of `U-{user_id}` (granularity: {granularity}): No income forecasts found")
    if granularity == 'monthly':
      return pd.DataFrame(columns=['user_id', 'ai_category_id', 'month_date', 'forecasted_amount', 'category'])
    else:
      return pd.DataFrame(columns=['user_id', 'ai_category_id', 'sunday_date', 'forecasted_amount', 'category'])
  
  # Add category names by mapping ai_category_id
  df['category'] = df['ai_category_id'].map(_CATEGORY_ID_TO_NAME).fillna('Unknown')
  
  # Convert date column to datetime for proper comparisons
  if date_col in df.columns and len(df) > 0:
    df[date_col] = pd.to_datetime(df[date_col])
  
  log(f"**Retrieved Income Forecasts** of `U-{user_id}` (granularity: {granularity}): `df: {df.shape}` w/ **cols**:\n  - `{'`, `'.join(df.columns)}`")
  return df


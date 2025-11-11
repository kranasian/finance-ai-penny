from database import Database
import pandas as pd
from penny.tool_funcs.sandbox_logging import log
from penny.tools.utils import to_all_category_name


def retrieve_spending_forecasts_function_code_gen(user_id: int = 1, granularity: str = 'monthly') -> pd.DataFrame:
  """Function to retrieve spending forecasts from the database for a specific user"""
  db = Database()
  
  if granularity not in ['monthly', 'weekly']:
    log(f"**Error**: Invalid granularity '{granularity}'. Must be 'monthly' or 'weekly'.")
    return pd.DataFrame()
  
  # Income category IDs (46: income, 47: income, 36: salary, 37: sidegig, 38: business, 39: interest)
  income_category_ids = [46, 47, 36, 37, 38, 39]
  
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
  
  df["category"] = df["ai_category_id"].apply(to_all_category_name)
  # Change category of top_income to income and top_bills to bills, etc
  df["category"] = df["category"].apply(lambda x: x.replace("top_", "") if x.startswith("top_") else x)
  
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
  
  df["category"] = df["ai_category_id"].apply(to_all_category_name)
  # Change category of top_income to income and top_bills to bills, etc
  df["category"] = df["category"].apply(lambda x: x.replace("top_", "") if x.startswith("top_") else x)
  
  # Convert date column to datetime for proper comparisons
  if date_col in df.columns and len(df) > 0:
    df[date_col] = pd.to_datetime(df[date_col])
  
  log(f"**Retrieved Income Forecasts** of `U-{user_id}` (granularity: {granularity}): `df: {df.shape}` w/ **cols**:\n  - `{'`, `'.join(df.columns)}`")
  return df


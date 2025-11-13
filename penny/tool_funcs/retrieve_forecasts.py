from database import Database
import pandas as pd
from penny.tool_funcs.sandbox_logging import log
from penny.tools.utils import to_all_category_name
from categories import get_parents_with_leaves_as_dict_categories

def _adjust_parent_forecasts(df: pd.DataFrame) -> pd.DataFrame:
  """Post-process forecasts: adjust parent category forecasts to be difference of original minus sum of children.
  
  For each parent category, the forecast is adjusted to:
  parent_forecast = original_forecast - sum(children_forecasts)
  
  This means the parent represents the "other" spending in that category not accounted for by children.
  """
  if df.empty:
    return df
  
  # Get parent-to-leaf mapping from categories.py
  parent_to_leaf_categories = get_parents_with_leaves_as_dict_categories()
  
  # Get all parent category IDs
  parent_category_ids = set(parent_to_leaf_categories.keys())
  
  # Create a copy to avoid modifying the original
  df = df.copy()
  
  # Group by start_date to process each date separately
  for start_date, date_group in df.groupby('start_date'):
    # For each parent category that exists in this date group
    for parent_id in parent_category_ids:
      # Get children category IDs (excluding the parent itself)
      children_ids = [cid for cid in parent_to_leaf_categories[parent_id] if cid != parent_id]
      
      # Sum forecasts of all children categories for this date
      children_rows = date_group[date_group['ai_category_id'].isin(children_ids)]
      children_sum = children_rows['forecasted_amount'].sum() if not children_rows.empty else 0.0
      
      # Adjust parent forecast: original - sum of children
      parent_mask = (df['start_date'] == start_date) & (df['ai_category_id'] == parent_id)
      if parent_mask.any():
        original_forecast = df.loc[parent_mask, 'forecasted_amount'].iloc[0]
        adjusted_forecast = original_forecast - children_sum
        df.loc[parent_mask, 'forecasted_amount'] = adjusted_forecast
        log(f"**Adjusted Parent Forecast**: Category {parent_id} on {start_date}: {original_forecast:.2f} -> {adjusted_forecast:.2f} (original - sum of children: {children_sum:.2f})")
  
  return df


def retrieve_spending_forecasts_function_code_gen(user_id: int = 1, granularity: str = 'monthly') -> pd.DataFrame:
  """Function to retrieve spending forecasts from the database for a specific user"""
  db = Database()
  
  if granularity not in ['monthly', 'weekly']:
    log(f"**Error**: Invalid granularity '{granularity}'. Must be 'monthly' or 'weekly'.")
    return pd.DataFrame()
  
  # Income category IDs (46: income, 47: income, 36: salary, 37: sidegig, 38: business, 39: interest)
  income_category_ids = [46, 47, 36, 37, 38, 39]
  
  if granularity == 'monthly':
    df = db.get_monthly_forecasts_by_user(user_id=user_id)
  else:  # weekly
    df = db.get_weekly_forecasts_by_user(user_id=user_id)
  
  if df.empty:
    log(f"**Retrieved Spending Forecasts** of `U-{user_id}` (granularity: {granularity}): No forecasts found")
    return pd.DataFrame()
  
  # Filter out income category IDs (keep only spending)
  df = df[~df['ai_category_id'].isin(income_category_ids)]
  
  if df.empty:
    log(f"**Retrieved Spending Forecasts** of `U-{user_id}` (granularity: {granularity}): No spending forecasts found")
    return pd.DataFrame()

  df["category"] = df["ai_category_id"].apply(to_all_category_name)
  # Change category of top_income to income and top_bills to bills, etc
  df["category"] = df["category"].apply(lambda x: x.replace("top_", "") if x.startswith("top_") else x)
  
  # Post-process: adjust parent category forecasts to equal sum of children
  df = _adjust_parent_forecasts(df)
  
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
    df = db.get_monthly_forecasts_by_user(user_id=user_id)
  else:  # weekly
    df = db.get_weekly_forecasts_by_user(user_id=user_id)
  
  if df.empty:
    log(f"**Retrieved Income Forecasts** of `U-{user_id}` (granularity: {granularity}): No forecasts found")
    return pd.DataFrame()
  
  # Filter for income category IDs only
  df = df[df['ai_category_id'].isin(income_category_ids)]
  
  if df.empty:
    log(f"**Retrieved Income Forecasts** of `U-{user_id}` (granularity: {granularity}): No income forecasts found")
    return pd.DataFrame()

  df["category"] = df["ai_category_id"].apply(to_all_category_name)
  # Change category of top_income to income and top_bills to bills, etc
  df["category"] = df["category"].apply(lambda x: x.replace("top_", "") if x.startswith("top_") else x)
  
  # Post-process: adjust parent category forecasts to equal sum of children
  df = _adjust_parent_forecasts(df)
  
  log(f"**Retrieved Income Forecasts** of `U-{user_id}` (granularity: {granularity}): `df: {df.shape}` w/ **cols**:\n  - `{'`, `'.join(df.columns)}`")
  return df


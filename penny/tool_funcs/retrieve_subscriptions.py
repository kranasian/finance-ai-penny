from database import Database
import json
import pandas as pd
import re
from penny.tool_funcs.sandbox_logging import log


def retrieve_subscriptions_function_code_gen(user_id: int = 1) -> pd.DataFrame:
  """Function to retrieve subscription transactions by joining transactions with user_recurring_transactions"""
  db = Database()
  subscription_transactions = db.get_subscription_transactions(user_id=user_id, confidence_score_bills_threshold=0.5)
  
  if not subscription_transactions:
    log(f"**Retrieved Subscription Transactions** of `U-{user_id}`: No subscription transactions found")
    return pd.DataFrame(columns=['transaction_id', 'user_id', 'account_id', 'date', 'transaction_name', 'amount', 'category', 'subscription_name', 'confidence_score_bills', 'reviewer_bills'])
  
  df = pd.DataFrame(subscription_transactions)
  
  # Convert date column to datetime for proper comparisons
  if 'date' in df.columns and len(df) > 0:
    df['date'] = pd.to_datetime(df['date'])
  
  # Add output_category column (format category for display)
  if 'category' in df.columns:
    def format_category(cat):
      if not cat:
        return 'Unknown'
      # Replace underscores with spaces and title case
      formatted = cat.replace('_', ' ').title()
      # Remove common prefixes
      for prefix in ['meals ', 'income ', 'bills ', 'leisure ', 'shelter ']:
        if formatted.lower().startswith(prefix):
          formatted = formatted[len(prefix):]
          break
      return formatted
    
    df['output_category'] = df['category'].apply(format_category)
  
  log(f"**Retrieved Subscription Transactions** of `U-{user_id}`: `df: {df.shape}` w/ **cols**:\n  - `{'`, `'.join(df.columns)}`")
  return df


def subscription_names_and_amounts(df: pd.DataFrame, template: str) -> tuple[str, list]:
  """Generate a formatted string describing subscription transaction names and amounts using the provided template and return metadata"""
  log(f"**Subscription Names/Amounts**: `df: {df.shape}` w/ **cols**:\n  - `{'`, `'.join(df.columns)}`")
  
  if df.empty:
    log("- **`df` is empty**, returning empty string and metadata.")
    return "", []
  
  # Check if required columns exist (subscription transaction columns)
  required_columns = ['transaction_name', 'amount', 'date', 'category', 'transaction_id']
  missing_columns = [col for col in required_columns if col not in df.columns]
  if missing_columns:
    error_msg = f"DataFrame is missing required columns: {missing_columns}. Available columns: {list(df.columns)}"
    log(error_msg)
    raise ValueError(error_msg)
  
  utterances = []
  metadata = []
  
  log(f"**Listing Individual Subscription Transactions**: Processing {len(df)} items.")
  for i in range(len(df)):
    transaction = df.iloc[i]
    transaction_name = transaction.get('transaction_name', 'Unknown Transaction')
    subscription_name = transaction.get('subscription_name', transaction_name)  # Use subscription_name if available, fallback to transaction_name
    amount = transaction.get('amount', 0.0)
    date = transaction.get('date', 'Unknown Date')
    category = transaction.get('category', 'Unknown Category')
    transaction_id = transaction.get('transaction_id', None)
    
    amount_log = f"${abs(amount):,.0f}" if amount else "Unknown"
    log(f"  - `T-{transaction_id}`]  **Name**: `{transaction_name}`  |  **Subscription**: `{subscription_name}`  |  **Amount**: `{amount_log}`  |  **Date**: `{date}`  |  **Category**: `{category}`")
    
    # Determine direction and preposition based on category and amount sign
    # First check if category is income category
    income_categories = ['income_salary', 'income_sidegig', 'income_business', 'income_interest', 'income']
    is_income = category in income_categories if category else False
    
    # Determine direction based on category and amount sign
    # For income categories: negative = earned (inflow), positive = refunded (outflow)
    # For expense categories: positive = spent (outflow), negative = received (inflow/refund)
    if is_income:
      if amount < 0:
        direction = "earned"
        preposition = "from"
      else:  # amount > 0
        direction = "refunded"
        preposition = "to"
    else:
      # Expense categories (most subscriptions)
      if amount > 0:
        direction = "spent"
        preposition = "on"
      else:  # amount < 0
        direction = "received"
        preposition = "from"
    
    # Check if template has format specifiers for amount (like {amount:,.0f})
    amount_format_pattern = r'\{amount:([^}]+)\}'
    has_amount_format_specifier = bool(re.search(amount_format_pattern, template))
    
    # Check if template has dollar sign - if it does, format amount without $, otherwise with $
    has_dollar_sign = '$' in template
    
    # Check if template has date format specifiers (like {date:%%Y-%%m-%%d})
    date_format_pattern = r'\{date:([^}]+)\}'
    date_format_match = re.search(date_format_pattern, template)
    date_str = date
    temp_template = template
    
    if date_format_match:
      # Extract the format specifier (e.g., "%%Y-%%m-%%d")
      format_spec = date_format_match.group(1)
      # Replace %% with % for strftime format
      strftime_format = format_spec.replace('%%', '%')
      # Replace the format specifier in template with simple {date}
      temp_template = re.sub(date_format_pattern, '{date}', template)
      
      # Format the date if it's a datetime object, otherwise use as-is
      if hasattr(date, 'strftime'):
        date_str = date.strftime(strftime_format)
      elif isinstance(date, str):
        # If date is already a string, try to parse and reformat if needed
        try:
          from datetime import datetime
          # Try common date formats
          for fmt in ['%Y-%m-%d', '%Y-%m-%d %H:%M:%S', '%Y/%m/%d']:
            try:
              parsed_date = datetime.strptime(date, fmt)
              date_str = parsed_date.strftime(strftime_format)
              break
            except ValueError:
              continue
        except (ValueError, TypeError):
          # If parsing fails, use date as-is
          date_str = date
      else:
        date_str = str(date)
    else:
      # No date format specifier - format as date only (YYYY-MM-DD)
      if hasattr(date, 'strftime'):
        # pandas Timestamp or datetime object - format as date only
        date_str = date.strftime('%Y-%m-%d')
      elif isinstance(date, str):
        # If date string includes time, extract just the date part
        if ' ' in date:
          date_str = date.split(' ')[0]
        else:
          date_str = date
      else:
        # Convert to string and extract date part if it includes time
        date_str = str(date)
        if ' ' in date_str:
          date_str = date_str.split(' ')[0]
    
    # Handle amount formatting based on template format specifiers
    if has_amount_format_specifier:
      # Template has format specifiers, replace ALL of them with simple {amount}
      temp_template = re.sub(amount_format_pattern, '{amount}', temp_template)
    
    # Format amount string based on template requirements
    if has_amount_format_specifier:
      # Template originally had format specifiers, format with $ if template doesn't have $
      if has_dollar_sign:
        # Template has $, format without $ prefix
        amount_str = f"{abs(amount):,.0f}"
      else:
        # Template has NO $, format WITH $ prefix
        amount_str = f"${abs(amount):,.0f}"
    else:
      # No format specifiers - check if template has dollar signs
      if has_dollar_sign:
        # Template has dollar signs, format without $ prefix
        amount_str = f"{abs(amount):,.0f}"
      else:
        # Template has NO dollar signs, format WITH $ prefix
        amount_str = f"${abs(amount):,.0f}"
    
    # Build format dictionary with subscription transaction fields
    format_dict = {
      'name': transaction_name,
      'transaction_name': transaction_name,
      'subscription_name': subscription_name,
      'amount': amount_str,
      'date': date_str,
      'category': category,
      'direction': direction,
      'preposition': preposition
    }
    
    # Check for any additional columns in the DataFrame that might be referenced in the template
    # Extract all placeholder names from the template (e.g., {direction_amount}, {custom_column})
    template_placeholders = re.findall(r'\{([^}:]+)', temp_template)
    for placeholder in template_placeholders:
      # Remove any format specifiers (e.g., "amount:,.0f" -> "amount")
      placeholder_name = placeholder.split(':')[0]
      # If the placeholder is not already in format_dict and exists as a column in the DataFrame
      if placeholder_name not in format_dict and placeholder_name in df.columns:
        format_dict[placeholder_name] = transaction.get(placeholder_name, '')
    
    try:
      utterance = temp_template.format(**format_dict)
    except (ValueError, KeyError) as e:
      # If formatting fails, try with raw numeric amount (only if template had format specifiers)
      if has_amount_format_specifier:
        # Try with raw number (format specifier was already replaced)
        format_dict_raw = format_dict.copy()
        format_dict_raw['amount'] = abs(amount)
        try:
          utterance = temp_template.format(**format_dict_raw)
        except Exception as e2:
          # Fall back to formatted string
          utterance = temp_template.format(**format_dict)
      else:
        # No format specifiers, so the error is unexpected - re-raise
        raise
    
    utterances.append(utterance)
    
    # Convert date to string for JSON serialization
    date_for_metadata = date
    if hasattr(date, 'strftime'):
      # pandas Timestamp or datetime object
      date_for_metadata = date.strftime('%Y-%m-%d')
    elif not isinstance(date, str):
      # Convert to string if not already
      date_for_metadata = str(date)
    
    metadata.append({
      "transaction_id": int(transaction_id),
      "transaction_name": transaction_name
    })
  
  log(f"**Returning** {len(utterances)} utterances and {len(metadata)} metadata entries.")
  log(f"**Utterances**:\n  - `{'`\n  - `'.join(utterances)}`")
  log(f"**Metadata**:\n```json\n{json.dumps(metadata, indent=2)}\n```")
  return "\n".join(utterances), metadata


def utter_subscription_totals(df: pd.DataFrame, template: str) -> str:
  """Calculate total subscription transaction amounts and return formatted string"""
  log(f"**Subscription Totals**: `df: {df.shape}` w/ **cols**:\n  - `{'`, `'.join(df.columns)}`")
  
  if df.empty:
    log("- **`df` is empty**, returning empty string.")
    return ""
  
  # Check if required columns exist
  required_columns = ['amount']
  missing_columns = [col for col in required_columns if col not in df.columns]
  if missing_columns:
    error_msg = f"- **`df` is missing required columns**: `{', '.join(missing_columns)}`. Available columns: `{', '.join(df.columns)}`"
    log(error_msg)
    raise ValueError(error_msg)
  
  # Calculate total, excluding None values
  total_amount = df['amount'].dropna().sum()
  transaction_count = len(df)
  log(f"**Calculated Total**: **Amount**: `${abs(total_amount):,.0f}` | **Count**: `{transaction_count}`")
  
  # Determine if transactions are income or spending based on categories
  is_income = False
  if len(df) > 0:
    if 'category' in df.columns:
      income_categories = ['income_salary', 'income_sidegig', 'income_business', 'income_interest', 'income']
      is_income = df['category'].isin(income_categories).any()
  
  # Determine direction and verb based on category and amount sign
  # Always use abs() for display amount
  display_amount = abs(total_amount)
  
  # Determine direction based on category and amount sign
  # For income categories: negative = earned (inflow), positive = refunded (outflow)
  # For expense categories: positive = spent (outflow), negative = received (inflow/refund)
  if is_income:
    # Income categories
    if total_amount < 0:
      # Income inflow (negative amount = money coming in)
      direction = "earned"
    else:  # total_amount > 0
      # Income outflow (positive amount = money going out, refund)
      direction = "outflow"
  else:
    # Expense categories (most subscriptions)
    if total_amount > 0:
      # Spending outflow (positive amount = money going out)
      direction = "spent"
    else:  # total_amount < 0
      # Spending inflow (negative amount = money coming in, refund)
      direction = "inflow"
  
  # Only show direction for inflow/outflow, not for earned/spent
  # If direction is "earned" or "spent", set to empty string so it won't appear in templates
  # Otherwise, format as "(inflow)" or "(outflow)"
  if direction in ["earned", "spent"]:
    direction_display = ""
  else:
    direction_display = f"({direction})"
  
  # Check if template has format specifiers for total_amount or amount (like {total_amount:,.0f} or {amount:,.0f})
  total_amount_format_pattern = r'\{total_amount:([^}]+)\}'
  amount_format_pattern = r'\{amount:([^}]+)\}'
  has_total_amount_format_specifier = bool(re.search(total_amount_format_pattern, template))
  has_amount_format_specifier = bool(re.search(amount_format_pattern, template))
  
  # Check if template has dollar sign - if it does, format amount without $, otherwise with $
  has_dollar_sign = '$' in template
  
  # Handle format specifiers - replace with simple placeholders
  temp_template = template
  if has_total_amount_format_specifier:
    temp_template = re.sub(total_amount_format_pattern, '{total_amount}', temp_template)
  if has_amount_format_specifier:
    temp_template = re.sub(amount_format_pattern, '{total_amount}', temp_template)
  
  # Also replace any plain {amount} with {total_amount}
  temp_template = temp_template.replace('{amount}', '{total_amount}')
  
  # Format amount string based on template requirements
  has_any_format_specifier = has_total_amount_format_specifier or has_amount_format_specifier
  
  if has_any_format_specifier:
    if has_dollar_sign:
      total_amount_str = f"{display_amount:,.0f}"
    else:
      total_amount_str = f"${display_amount:,.0f}"
  else:
    if has_dollar_sign:
      total_amount_str = f"{display_amount:,.0f}"
    else:
      total_amount_str = f"${display_amount:,.0f}"
  
  format_dict = {
    'total_amount': total_amount_str,
    'direction': direction_display,  # Use direction_display which is empty for earned/spent
  }
  
  try:
    result = temp_template.format(**format_dict)
  except (ValueError, KeyError) as e:
    # If formatting fails, try with raw numeric amount
    if has_any_format_specifier:
      try:
        format_dict_raw = {
          'total_amount': display_amount,
          'direction': direction_display,
        }
        result = temp_template.format(**format_dict_raw)
      except Exception as e2:
        result = temp_template.format(total_amount=total_amount_str)
    else:
      raise
  
  # Clean up empty parentheses if direction was empty (e.g., "$100.00 ()" -> "$100.00")
  result = re.sub(r'\s*\(\)', '', result)
  
  log(f"**Subscription Totals**: Template formatted successfully")
  return result

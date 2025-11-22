from database import Database
import json
import pandas as pd
import re
from penny.tool_funcs.sandbox_logging import log

# Maximum number of subscriptions to return in subscription_names_and_amounts
MAX_SUBSCRIPTIONS = 10


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


def subscription_names_and_amounts(df: pd.DataFrame, template: str) -> str:
  """Generate a formatted string describing subscription transaction names and amounts using the provided template.
  
  Returns:
    str: Newline-separated subscription descriptions (max MAX_SUBSCRIPTIONS). 
      If there are more subscriptions, appends a message like "n more subscriptions."
  """
  log(f"**Subscription Names/Amounts**: `df: {df.shape}` w/ **cols**:\n  - `{'`, `'.join(df.columns)}`")
  
  if df.empty:
    log("- **`df` is empty**, returning empty string.")
    return ""
  
  # Check if required columns exist (subscription transaction columns)
  required_columns = ['transaction_name', 'amount', 'date', 'category', 'transaction_id']
  missing_columns = [col for col in required_columns if col not in df.columns]
  if missing_columns:
    error_msg = f"DataFrame is missing required columns: {missing_columns}. Available columns: {list(df.columns)}"
    log(error_msg)
    raise ValueError(error_msg)
  
  # Limit both utterances and metadata to MAX_SUBSCRIPTIONS
  total_count = len(df)
  has_more = total_count > MAX_SUBSCRIPTIONS
  
  utterances = []
  metadata = []
  
  log(f"**Listing Individual Subscription Transactions**: Processing up to {MAX_SUBSCRIPTIONS} items (out of {total_count} total).")
  for i in range(len(df)):
    transaction = df.iloc[i]
    transaction_name = transaction.get('transaction_name', 'Unknown Transaction')
    subscription_name = transaction.get('subscription_name', transaction_name)  # Use subscription_name if available, fallback to transaction_name
    amount = transaction.get('amount', 0.0)
    date = transaction.get('date', 'Unknown Date')
    category = transaction.get('category', 'Unknown Category')
    transaction_id = transaction.get('transaction_id', None)
    
    amount_log = f"${abs(amount):.0f}" if amount else "Unknown"
    log(f"  - `T-{transaction_id}`]  **Name**: `{transaction_name}`  |  **Subscription**: `{subscription_name}`  |  **Amount**: `{amount_log}`  |  **Date**: `{date}`  |  **Category**: `{category}`")
    
    # Determine amount_and_direction based on amount sign
    # Subscriptions are always spending transactions:
    # - Spending (Outflow): amount > 0 → "$X.XX was paid to"
    # - Spending (Inflow/Refund): amount < 0 → "$X.XX was refunded from"
    
    # Format amount as positive for display
    amount_abs = abs(amount)
    amount_str_for_direction = f"${amount_abs:.0f}"
    
    # Determine the verb phrase based on amount sign
    if amount > 0:
      # Spending (Outflow): amount > 0 → "was paid to"
      verb_phrase = "was paid to"
    else:  # amount < 0
      # Spending (Inflow/Refund): amount < 0 → "was refunded from"
      verb_phrase = "was refunded from"
    
    # Build amount_and_direction string
    amount_and_direction = f"{amount_str_for_direction} {verb_phrase}"
    
    temp_template = template
    
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
    
    # Build format dictionary with subscription transaction fields
    format_dict = {
      'name': transaction_name,
      'transaction_name': transaction_name,
      'subscription_name': subscription_name,
      'amount_and_direction': amount_and_direction,
      'date': date_str,
      'category': category
    }
    
    # Check for any additional columns in the DataFrame that might be referenced in the template
    # Extract all placeholder names from the template (e.g., {direction_amount}, {custom_column})
    template_placeholders = re.findall(r'\{([^}:]+)', temp_template)
    for placeholder in template_placeholders:
      # Remove any format specifiers (e.g., "amount:.0f" -> "amount")
      placeholder_name = placeholder.split(':')[0]
      # If the placeholder is not already in format_dict and exists as a column in the DataFrame
      if placeholder_name not in format_dict and placeholder_name in df.columns:
        format_dict[placeholder_name] = transaction.get(placeholder_name, '')
    
    try:
      utterance = temp_template.format(**format_dict)
    except (ValueError, KeyError) as e:
      # If formatting fails, log error and re-raise
      log(f"**Template Formatting Error**: {e}. Template: {temp_template}, Format dict keys: {list(format_dict.keys())}")
      raise
    
    # Convert date to string for JSON serialization
    date_for_metadata = date
    if hasattr(date, 'strftime'):
      # pandas Timestamp or datetime object
      date_for_metadata = date.strftime('%Y-%m-%d')
    elif not isinstance(date, str):
      # Convert to string if not already
      date_for_metadata = str(date)
    
    # Add to metadata only if under the limit (max MAX_SUBSCRIPTIONS)
    if len(metadata) < MAX_SUBSCRIPTIONS:
      metadata.append({
        "transaction_id": int(transaction_id),
        "transaction_name": transaction_name
      })
    
    # Add to utterances only if under the limit (max MAX_SUBSCRIPTIONS)
    if len(utterances) < MAX_SUBSCRIPTIONS:
      utterances.append(utterance)
    
    # Break early if we've reached both limits
    if len(metadata) >= MAX_SUBSCRIPTIONS and len(utterances) >= MAX_SUBSCRIPTIONS:
      break
  
  # Add message about remaining subscriptions if there are more
  utterance_text = "\n".join(utterances)
  if has_more:
    remaining_count = total_count - MAX_SUBSCRIPTIONS
    utterance_text += f"\n{remaining_count} more subscription{'s' if remaining_count != 1 else ''}."
  
  log(f"**Returning** {len(utterances)} utterances. Has more: {has_more}")
  log(f"**Utterances**:\n  - `{'`\n  - `'.join(utterances)}`")
  return utterance_text


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
  log(f"**Calculated Total**: **Amount**: `${abs(total_amount):.0f}` | **Count**: `{transaction_count}`")
  
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
  
  # Check if template has format specifiers for total_amount or amount (like {total_amount:.0f} or {amount:.0f})
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
      total_amount_str = f"{display_amount:.0f}"
    else:
      total_amount_str = f"${display_amount:.0f}"
  else:
    if has_dollar_sign:
      total_amount_str = f"{display_amount:.0f}"
    else:
      total_amount_str = f"${display_amount:.0f}"
  
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

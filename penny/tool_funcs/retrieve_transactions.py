from database import Database
import json
import pandas as pd
import re
from penny.tool_funcs.sandbox_logging import log

# Maximum number of transactions to return in transaction_names_and_amounts
MAX_TRANSACTIONS = 10


def retrieve_transactions_function_code_gen(user_id: int = 1) -> pd.DataFrame:
  """Function to retrieve transactions from the database for a specific user"""
  db = Database()
  transactions = db.get_transactions_by_user(user_id=user_id)
  df = pd.DataFrame(transactions)
  
  # Convert date column to datetime for proper comparisons
  if 'date' in df.columns and len(df) > 0:
    df['date'] = pd.to_datetime(df['date'])
  
  log(f"**Retrieved All Transactions** of `U-{user_id}`: `df: {df.shape}` w/ **cols**:\n  - `{'`, `'.join(df.columns)}`")
  return df


def retrieve_income_transactions_function_code_gen(user_id: int = 1) -> pd.DataFrame:
  """Function to retrieve income transactions from the database for a specific user"""
  df = retrieve_transactions_function_code_gen(user_id=user_id)
  
  if df.empty:
    log(f"**Retrieved Income Transactions** of `U-{user_id}`: empty DataFrame")
    return df
  
  # Filter for income categories
  income_categories = ['income_salary', 'income_sidegig', 'income_business', 'income_interest', 'income']
  income_df = df[df['category'].isin(income_categories)]
  
  log(f"**Retrieved Income Transactions** of `U-{user_id}`: `df: {income_df.shape}` w/ **cols**:\n  - `{'`, `'.join(income_df.columns)}`")
  return income_df


def retrieve_spending_transactions_function_code_gen(user_id: int = 1) -> pd.DataFrame:
  """Function to retrieve spending transactions from the database for a specific user"""
  df = retrieve_transactions_function_code_gen(user_id=user_id)
  
  if df.empty:
    log(f"**Retrieved Spending Transactions** of `U-{user_id}`: empty DataFrame")
    return df
  
  # Filter for spending categories (exclude income categories)
  income_categories = ['income_salary', 'income_sidegig', 'income_business', 'income_interest', 'income']
  spending_df = df[~df['category'].isin(income_categories)]
  
  log(f"**Retrieved Spending Transactions** of `U-{user_id}`: `df: {spending_df.shape}` w/ **cols**:\n  - `{'`, `'.join(spending_df.columns)}`")
  return spending_df


def transaction_names_and_amounts(df: pd.DataFrame, template: str) -> tuple[str, list]:
  """Generate a formatted string describing transaction names and amounts using the provided template and return metadata.
  
  Returns:
    tuple[str, list]: (formatted string, metadata list)
      - formatted string: Newline-separated transaction descriptions (max MAX_TRANSACTIONS). 
        If there are more transactions, appends a message like "n more transactions."
      - metadata list: List of transaction metadata dictionaries (only MAX_TRANSACTIONS transactions included)
  """
  
  log(f"**Transaction Names/Amounts**: `df: {df.shape}` w/ **cols**:\n  - `{'`, `'.join(df.columns)}`")
  
  if df.empty:
    log("- **`df` is empty**, returning empty string and metadata.")
    return "", []
  
  # Check if required columns exist (only transaction_name and amount are required)
  required_columns = ['transaction_name', 'amount']
  missing_columns = [col for col in required_columns if col not in df.columns]
  if missing_columns:
    error_msg = f"DataFrame is missing required columns: {missing_columns}. Available columns: {list(df.columns)}"
    log(error_msg)
    raise ValueError(error_msg)
  
  # Limit both utterances and metadata to MAX_TRANSACTIONS
  total_count = len(df)
  has_more = total_count > MAX_TRANSACTIONS
  
  utterances = []
  metadata = []
  
  log(f"**Listing Individual Transactions**: Processing up to {MAX_TRANSACTIONS} items (out of {total_count} total).")
  for i in range(len(df)):
    transaction = df.iloc[i]
    transaction_name = transaction.get('transaction_name', 'Unknown Transaction')
    # Clean up transaction name by removing text inside brackets (e.g., "Local Restaurant [DOWNTOWN BISTRO]" -> "Local Restaurant")
    transaction_name_cleaned = re.sub(r'\s*\[.*?\]\s*', '', transaction_name).strip()
    amount = transaction.get('amount', 0.0)
    date = transaction.get('date', 'Unknown Date')
    category = transaction.get('category', 'Unknown Category')
    transaction_id = transaction.get('transaction_id', None)
    account_id = transaction.get('account_id', None)
    
    amount_log = f"${abs(amount):.0f}" if amount else "Unknown"
    log(f"  - `T-{transaction_id}`]  **Name**: `{transaction_name}`  |  **Amount**: `{amount_log}`  |  **Date**: `{date}`  |  **Category**: `{category}`  |  **Account ID**: `{account_id}`")
    
    # Determine amount_and_direction based on amount sign and category
    # Rules:
    # - Spending (Outflow): amount > 0 → "$X.XX was paid to"
    # - Income (Inflow): amount < 0, category is income → "$X.XX was received from"
    # - Spending (Inflow): amount < 0, category is spending → "$X.XX was refunded from"
    # - Income Outflow (Refund): amount > 0, category is income → "$X.XX was returned to"
    income_categories = ['income_salary', 'income_sidegig', 'income_business', 'income_interest', 'income']
    is_income = category in income_categories
    
    # Format amount as positive for display
    amount_abs = abs(amount)
    amount_str_for_direction = f"${amount_abs:.0f}"
    
    # Determine the verb phrase based on amount sign and category
    if is_income:
      if amount > 0:
        # Income Outflow (Refund): amount > 0, category is income → "was returned to"
        verb_phrase = "was returned to"
      else:  # amount < 0
        # Income (Inflow): amount < 0, category is income → "was received from"
        verb_phrase = "was received from"
    else:  # spending categories
      if amount > 0:
        # Spending (Outflow): amount > 0 → "was paid to"
        verb_phrase = "was paid to"
      else:  # amount < 0
        # Spending (Inflow): amount < 0, category is spending → "was refunded from"
        verb_phrase = "was refunded from"
    
    # Build amount_and_direction string
    amount_and_direction = f"{amount_str_for_direction} {verb_phrase}"
    
    temp_template = template
    
    # Check if template has date format specifiers (like {date:%%Y-%%m-%%d})
    # If so, replace them with simple {date} and format the date accordingly
    date_format_pattern = r'\{date:([^}]+)\}'
    date_format_match = re.search(date_format_pattern, temp_template)
    date_str = date
    
    if date_format_match:
      # Extract the format specifier (e.g., "%%Y-%%m-%%d")
      format_spec = date_format_match.group(1)
      # Replace %% with % for strftime format
      strftime_format = format_spec.replace('%%', '%')
      # Replace the format specifier in template with simple {date}
      temp_template = re.sub(date_format_pattern, '{date}', temp_template)
      
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
    
    # Build format dictionary with available columns
    format_dict = {
      'name': transaction_name,
      'transaction_name': transaction_name_cleaned,
      'amount_and_direction': amount_and_direction,
      'date': date_str,
      'category': category,
      'transaction_id': transaction_id if transaction_id is not None else None,
      'account_id': account_id if account_id is not None else None
    }
    
    # Check for any additional columns in the DataFrame that might be referenced in the template
    # Extract all placeholder names from the template (e.g., {account_id}, {transaction_id})
    template_placeholders = re.findall(r'\{([^}:]+)', temp_template)
    for placeholder in template_placeholders:
      # Remove any format specifiers (e.g., "amount:.0f" -> "amount")
      placeholder_name = placeholder.split(':')[0]
      
      # If the placeholder is not already in format_dict and exists as a column in the DataFrame
      if placeholder_name not in format_dict and placeholder_name in df.columns:
        value = transaction.get(placeholder_name, '')
        # Convert to native Python type for JSON serialization
        if pd.isna(value):
          format_dict[placeholder_name] = None
        elif isinstance(value, (pd.Timestamp, pd.DatetimeTZDtype)):
          format_dict[placeholder_name] = value.strftime('%Y-%m-%d') if hasattr(value, 'strftime') else str(value)
        else:
          format_dict[placeholder_name] = value
    
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
    
    # Add to metadata only if under the limit (max MAX_TRANSACTIONS)
    if len(metadata) < MAX_TRANSACTIONS:
      metadata.append({
        "transaction_id": int(transaction_id) if transaction_id is not None else None,
        "account_id": int(account_id) if account_id is not None else None,
        "transaction_name": transaction_name_cleaned
      })
    
    # Add to utterances only if under the limit (max MAX_TRANSACTIONS)
    if len(utterances) < MAX_TRANSACTIONS:
      utterances.append(utterance)
    
    # Break early if we've reached both limits
    if len(metadata) >= MAX_TRANSACTIONS and len(utterances) >= MAX_TRANSACTIONS:
      break
  
  # Add message about remaining transactions if there are more
  utterance_text = "\n".join(utterances)
  if has_more:
    remaining_count = total_count - MAX_TRANSACTIONS
    utterance_text += f"\n{remaining_count} more transaction{'s' if remaining_count != 1 else ''}."
  
  log(f"**Returning** {len(utterances)} utterances and {len(metadata)} metadata entries. Has more: {has_more}")
  log(f"**Utterances**:\n  - `{'`\n  - `'.join(utterances)}`")
  log(f"**Metadata**:\n```json\n{json.dumps(metadata, indent=2)}\n```")
  return utterance_text, metadata


def utter_spending_transaction_total(df: pd.DataFrame, template: str) -> str:
  """Calculate total spending transaction amounts and return formatted string.
  
  Args:
    df: DataFrame with spending transactions (must have 'amount' column)
    template: Template string with {verb_and_total_amount} placeholder.
      Example: "In total, you {verb_and_total_amount}."
  
  Returns:
    Formatted string with verb automatically determined and inserted.
    Only {verb_and_total_amount} placeholder is supported.
  """
  
  log(f"**Spending Transaction Total**: `df: {df.shape}` w/ **cols**:\n  - `{'`, `'.join(df.columns)}`")
  
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
  
  total_amount = df['amount'].sum()
  log(f"**Calculated Total**: **Amount**: `${abs(total_amount):.0f}`")
  
  # Determine verb/phrase based on amount sign for spending transactions
  # Expense categories: positive = spent (outflow), negative = received (inflow/refund)
  display_amount = abs(total_amount)
  
  if total_amount > 0:
    # Spending outflow (positive amount = money going out)
    verb = "spent"
  else:  # total_amount < 0
    # Spending inflow (negative amount = money coming in, refund)
    verb = "received"
  
  # Check if template has format specifiers for verb_and_total_amount (like {verb_and_total_amount:.0f})
  verb_and_total_amount_format_pattern = r'\{verb_and_total_amount:([^}]+)\}'
  has_format_specifier = bool(re.search(verb_and_total_amount_format_pattern, template))
  
  # Check if template has dollar sign - if it does, format amount without $, otherwise with $
  has_dollar_sign = '$' in template
  
  # Handle format specifiers - replace with simple placeholder
  temp_template = template
  if has_format_specifier:
    # Template has format specifiers, replace them with simple {verb_and_total_amount}
    temp_template = re.sub(verb_and_total_amount_format_pattern, '{verb_and_total_amount}', temp_template)
  
  # Format amount string based on template requirements
  if has_format_specifier:
    # Template originally had format specifiers, format with $ if template doesn't have $
    if has_dollar_sign:
      # Template has $, format without $ prefix
      total_amount_str = f"{display_amount:.0f}"
    else:
      # Template has NO $, format WITH $ prefix
      total_amount_str = f"${display_amount:.0f}"
  else:
    # No format specifiers - check if template has dollar signs
    if has_dollar_sign:
      # Template has dollar signs, format without $ prefix
      total_amount_str = f"{display_amount:.0f}"
    else:
      # Template has NO dollar signs, format WITH $ prefix
      total_amount_str = f"${display_amount:.0f}"
  
  # Create verb_and_total_amount string (e.g., "spent $500" or "received $500")
  verb_and_total_amount_str = f"{verb} {total_amount_str}"
  
  # Replace placeholder in template
  result = temp_template.replace('{verb_and_total_amount}', verb_and_total_amount_str)
  
  log(f"**Spending Transaction Total Utterance**: `{result}`")
  return result


def utter_income_transaction_total(df: pd.DataFrame, template: str) -> str:
  """Calculate total income transaction amounts and return formatted string.
  
  Args:
    df: DataFrame with income transactions (must have 'amount' column)
    template: Template string with {verb_and_total_amount} placeholder.
      Example: "In total, you {verb_and_total_amount}."
  
  Returns:
    Formatted string with verb automatically determined and inserted.
    Only {verb_and_total_amount} placeholder is supported.
  """
  
  log(f"**Income Transaction Total**: `df: {df.shape}` w/ **cols**:\n  - `{'`, `'.join(df.columns)}`")
  
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
  
  total_amount = df['amount'].sum()
  log(f"**Calculated Total**: **Amount**: `${abs(total_amount):.0f}`")
  
  # Determine verb/phrase based on amount sign for income transactions
  # Income categories: negative = earned (inflow), positive = refunded (outflow)
  display_amount = abs(total_amount)
  
  if total_amount < 0:
    # Income inflow (negative amount = money coming in)
    verb = "earned"
  else:  # total_amount > 0
    # Income outflow (positive amount = money going out, refund)
    verb = "were refunded"
  
  # Check if template has format specifiers for verb_and_total_amount (like {verb_and_total_amount:.0f})
  verb_and_total_amount_format_pattern = r'\{verb_and_total_amount:([^}]+)\}'
  has_format_specifier = bool(re.search(verb_and_total_amount_format_pattern, template))
  
  # Check if template has dollar sign - if it does, format amount without $, otherwise with $
  has_dollar_sign = '$' in template
  
  # Handle format specifiers - replace with simple placeholder
  temp_template = template
  if has_format_specifier:
    # Template has format specifiers, replace them with simple {verb_and_total_amount}
    temp_template = re.sub(verb_and_total_amount_format_pattern, '{verb_and_total_amount}', temp_template)
  
  # Format amount string based on template requirements
  if has_format_specifier:
    # Template originally had format specifiers, format with $ if template doesn't have $
    if has_dollar_sign:
      # Template has $, format without $ prefix
      total_amount_str = f"{display_amount:.0f}"
    else:
      # Template has NO $, format WITH $ prefix
      total_amount_str = f"${display_amount:.0f}"
  else:
    # No format specifiers - check if template has dollar signs
    if has_dollar_sign:
      # Template has dollar signs, format without $ prefix
      total_amount_str = f"{display_amount:.0f}"
    else:
      # Template has NO dollar signs, format WITH $ prefix
      total_amount_str = f"${display_amount:.0f}"
  
  # Create verb_and_total_amount string (e.g., "earned $500" or "were refunded $500")
  verb_and_total_amount_str = f"{verb} {total_amount_str}"
  
  # Replace placeholder in template
  result = temp_template.replace('{verb_and_total_amount}', verb_and_total_amount_str)
  
  log(f"**Income Transaction Total Utterance**: `{result}`")
  return result



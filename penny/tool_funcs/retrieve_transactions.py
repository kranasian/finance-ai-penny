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
  
  cols_str = "`, `".join(df.columns)
  log(f"**Retrieved All Transactions** of `U-{user_id}`: `df: {df.shape}` w/ **cols**:\n  - `{cols_str}`")
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
  
  # Flip amount for income transactions
  income_df['amount'] = income_df['amount'] * -1
  
  cols_str = "`, `".join(income_df.columns)
  log(f"**Retrieved Income Transactions** of `U-{user_id}`: `df: {income_df.shape}` w/ **cols**:\n  - `{cols_str}`")
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
  
  cols_str = "`, `".join(spending_df.columns)
  log(f"**Retrieved Spending Transactions** of `U-{user_id}`: `df: {spending_df.shape}` w/ **cols**:\n  - `{cols_str}`")
  return spending_df

def transaction_category_grouped(df: pd.DataFrame, template: str) -> str:
  """Generate a formatted string describing transaction categories and amounts using the provided template.
  
  Args:
    df: DataFrame with transactions (must have 'category' and 'amount' columns)
    template: Template string with placeholders like {category}, {amount}, etc.
  
  Returns:
    str: Newline-separated category descriptions (max MAX_TRANSACTIONS). 
      If there are more categories, appends a message like "n more transactions."
  """
  
  cols_str = "`, `".join(df.columns)
  log(f"**Transaction Category Grouped**: `df: {df.shape}` w/ **cols**:\n  - `{cols_str}`")
  
  if df.empty:
    log("- **`df` is empty**, returning empty string.")
    return ""
  
  # Check if required columns exist (only category and amount are required)
  required_columns = ['category', 'amount']
  missing_columns = [col for col in required_columns if col not in df.columns]
  if missing_columns:
    error_msg = f"DataFrame is missing required columns: {missing_columns}. Available columns: {list(df.columns)}"
    log(error_msg)
    raise ValueError(error_msg)
  
  # Group by category and sum amounts
  grouped = df.groupby('category')['amount'].sum().reset_index()
  grouped = grouped.sort_values('amount', key=abs, ascending=False)  # Sort by absolute amount
  
  # Limit to MAX_TRANSACTIONS categories
  total_count = len(grouped)
  has_more = total_count > MAX_TRANSACTIONS
  grouped = grouped.head(MAX_TRANSACTIONS)
  
  utterances = []
  
  log(f"**Listing Category Groups**: Processing up to {MAX_TRANSACTIONS} categories (out of {total_count} total).")
  for i in range(len(grouped)):
    category_row = grouped.iloc[i]
    category = category_row.get('category', 'Unknown Category')
    amount = category_row.get('amount', 0.0)
    
    amount_log = f"${abs(amount):.0f}" if amount else "Unknown"
    log(f"  - **Category**: `{category}`  |  **Total Amount**: `{amount_log}`")
    
    # Determine category based on amount sign and category
    income_categories = ['income_salary', 'income_sidegig', 'income_business', 'income_interest', 'income']
    is_income = category in income_categories
    
    # Format amount as positive for display
    amount_abs = abs(amount)
    
    # Determine the verb phrase based on amount sign and category
    if is_income:
      verb_phrase = "earned" if amount >= 0 else "lost"
    else:  # spending categories
      verb_phrase = "spent" if amount >= 0 else "received"

    if amount >= 0:
      income_verb = "earned"
      spending_verb = "spent"
    else:  # amount < 0
      income_verb = "lost"
      spending_verb = "received"

    # Build amount_with_direction string
    amount_str = f"${amount_abs:.0f}"
    amount_with_direction = f"{amount_str} {verb_phrase}"
    income_amount_str = f"{income_verb} {amount_str}"
    spending_amount_str = f"{spending_verb} {amount_str}"
    
    temp_template = template
    
    # Build format dictionary with available columns
    format_dict = {
      'category': category,
      'amount_with_direction': amount_with_direction,
      'amount': amount_str,
      'income_amount': income_amount_str,
      'spending_amount': spending_amount_str,
      'income_total_amount': income_amount_str,
      'spending_total_amount': spending_amount_str,
    }
    
    # Check for any additional columns in the DataFrame that might be referenced in the template
    # Extract all placeholder names from the template (e.g., {category}, {amount})
    template_placeholders = re.findall(r'\{([^}:]+)', temp_template)
    for placeholder in template_placeholders:
      # Remove any format specifiers (e.g., "amount:.0f" -> "amount")
      placeholder_name = placeholder.split(':')[0]
      
      # If the placeholder is not already in format_dict and exists as a column in the original DataFrame
      if placeholder_name not in format_dict and placeholder_name in df.columns:
        # For grouped data, we can't use individual transaction values
        # But we can provide aggregate values if needed
        log(f"  - **Warning**: Placeholder `{placeholder_name}` requested but not available for grouped data.")
    
    try:
      utterance = temp_template.format(**format_dict)
    except (ValueError, KeyError) as e:
      # If formatting fails, log error and re-raise
      log(f"**Template Formatting Error**: {e}. Template: {temp_template}, Format dict keys: {list(format_dict.keys())}")
      raise
    
    utterances.append(utterance)
  
  # Add message about remaining categories if there are more
  utterance_text = "\n".join(utterances)
  if has_more:
    remaining_count = total_count - MAX_TRANSACTIONS
    utterance_text += f"\n{remaining_count} more transaction{'s' if remaining_count != 1 else ''}."
  
  log(f"**Returning** {len(utterances)} category groups. Has more: {has_more}")
  utterances_str = "`\n  - `".join(utterances)
  log(f"**Utterances**:\n  - `{utterances_str}`")
  return utterance_text


def transaction_names_and_amounts(df: pd.DataFrame, template: str) -> str:
  """Generate a formatted string describing transaction names and amounts using the provided template.
  
  Returns:
    str: Newline-separated transaction descriptions (max MAX_TRANSACTIONS). 
      If there are more transactions, appends a message like "n more transactions."
  """
  
  cols_str = "`, `".join(df.columns)
  log(f"**Transaction Names/Amounts**: `df: {df.shape}` w/ **cols**:\n  - `{cols_str}`")
  
  if df.empty:
    log("- **`df` is empty**, returning empty string.")
    return ""
  
  # Check if required columns exist (only transaction_name and amount are required)
  required_columns = ['transaction_name', 'amount']
  missing_columns = [col for col in required_columns if col not in df.columns]
  if missing_columns:
    ## Attempt to check if has category and amount columns
    if 'category' in df.columns and 'amount' in df.columns:
      return transaction_category_grouped(df, template)
    else:
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
    
    # Determine amount_with_direction based on amount sign and category
    # Rules:
    # - Spending (Outflow): amount > 0 → "$X.XX was paid to"
    # - Income (Inflow): amount < 0, category is income → "$X.XX was received from"
    # - Spending (Inflow): amount < 0, category is spending → "$X.XX was refunded from"
    # - Income Outflow (Refund): amount > 0, category is income → "$X.XX was returned to"
    income_categories = ['income_salary', 'income_sidegig', 'income_business', 'income_interest', 'income']
    is_income = category in income_categories
    
    # Format amount as positive for display
    amount_abs = abs(amount)
    
    # Determine the verb phrase based on amount sign and category
    if is_income:
      verb_phrase = "earned" if amount >= 0 else "lost"
    else:  # spending categories
      verb_phrase = "spent" if amount >= 0 else "received"

    if amount >= 0:
      income_verb = "earned"
      spending_verb = "spent"
    else:  # amount < 0
      income_verb = "lost"
      spending_verb = "received"

    # Build amount_with_direction string
    amount_str = f"${amount_abs:.0f}"
    amount_with_direction = f"{amount_str} {verb_phrase}"
    income_amount_str = f"{income_verb} {amount_str}"
    spending_amount_str = f"{spending_verb} {amount_str}"
    
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
      'amount_with_direction': amount_with_direction,
      'date': date_str,
      'category': category,
      'transaction_id': transaction_id if transaction_id is not None else None,
      'account_id': account_id if account_id is not None else None,
      'amount': amount_str,
      'income_total_amount': income_amount_str,
      'income_amount': income_amount_str,
      'spending_total_amount': spending_amount_str,
      'spending_amount': spending_amount_str,
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
  
  log(f"**Returning** {len(utterances)} utterances. Has more: {has_more}")
  utterances_str = "`\n  - `".join(utterances)
  log(f"**Utterances**:\n  - `{utterances_str}`")
  return utterance_text


def utter_transaction_total(df: pd.DataFrame, template: str) -> str:
  """Calculate total transaction amounts and return formatted string.
  
  Provides both {income_total_amount} and {spending_total_amount} placeholders.
  For income: negative amounts = earned, positive = refunded.
  For spending: positive amounts = spent, negative = received/refunded.
  The template should use the appropriate placeholder based on transaction type.
  
  Args:
    df: DataFrame with transactions (must have 'amount' column)
    template: Template string with {income_total_amount} or {spending_total_amount} placeholder.
      Example: "Total income: {income_total_amount}" or "Total spending: {spending_total_amount}"
      The placeholder automatically includes the verb (e.g., "earned $500" or "spent $500").
  
  Returns:
    Formatted string with verb and total amount inserted.
  """
  
  cols_str = "`, `".join(df.columns)
  log(f"**Transaction Total**: `df: {df.shape}` w/ **cols**:\n  - `{cols_str}`")
  
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
  log(f"**Calculated Total**: **Amount**: `${total_amount:.0f}`")
  
  # Format amount with dollar sign (use absolute for display)
  display_amount = abs(total_amount)
  total_amount_str = f"${display_amount:.0f}"
  
  # Determine verb/phrase for income: positive = earned (inflow), negative = refunded (outflow)
  # Determine verb/phrase for spending: positive = spent (outflow), negative = received (inflow/refund)
  if total_amount >= 0:
    income_verb = "earned"
    spending_verb = "spent"
  else:  # total_amount > 0
    income_verb = "lost"
    spending_verb = "received"
  income_total_amount_str = f"{income_verb} {total_amount_str}"
  spending_total_amount_str = f"{spending_verb} {total_amount_str}"
  
  # Build format dictionary - always provide both placeholders
  format_dict = {
    'income_total_amount': income_total_amount_str,
    'spending_total_amount': spending_total_amount_str,
    'total_amount': f"${total_amount:.0f}",
  }
  
  # Handle format specifiers in template (like {income_total_amount:.0f})
  # Extract and normalize placeholders
  temp_template = template
  placeholder_pattern = r'\{([^}:]+)(:[^}]+)?\}'
  
  # Replace format specifiers with simple placeholders
  def replace_format_specifier(match):
    placeholder_name = match.group(1)
    return f'{{{placeholder_name}}}'
  
  temp_template = re.sub(placeholder_pattern, replace_format_specifier, temp_template)
  
  try:
    result = temp_template.format(**format_dict)
  except (ValueError, KeyError) as e:
    log(f"**Template Formatting Error**: {e}. Template: {temp_template}, Format dict keys: {list(format_dict.keys())}")
    raise
  
  log(f"**Transaction Total Utterance**: `{result}`")
  return result


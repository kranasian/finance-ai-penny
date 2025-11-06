from database import Database
import json
import pandas as pd
import re
from sandbox_logging import log


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


def transaction_names_and_amounts(df: pd.DataFrame, template: str) -> tuple[str, list]:
  """Generate a formatted string describing transaction names and amounts using the provided template and return metadata"""
  log(f"**Transaction Names/Amounts**: `df: {df.shape}` w/ **cols**:\n  - `{'`, `'.join(df.columns)}`")
  
  if df.empty:
    log("- **`df` is empty**, returning empty string and metadata.")
    return "", []
  
  # Check if required columns exist
  required_columns = ['transaction_name', 'amount', 'date', 'category', 'transaction_id']
  missing_columns = [col for col in required_columns if col not in df.columns]
  if missing_columns:
    error_msg = f"DataFrame is missing required columns: {missing_columns}. Available columns: {list(df.columns)}"
    log(error_msg)
    raise ValueError(error_msg)
  
  utterances = []
  metadata = []
  
  log(f"**Listing Individual Transactions**: Processing {len(df)} items.")
  for i in range(len(df)):
    transaction = df.iloc[i]
    transaction_name = transaction.get('transaction_name', 'Unknown Transaction')
    amount = transaction.get('amount', 0.0)
    date = transaction.get('date', 'Unknown Date')
    category = transaction.get('category', 'Unknown Category')
    transaction_id = transaction.get('transaction_id', None)
    
    amount_log = f"${abs(amount):,.2f}" if amount else "Unknown"
    log(f"  - `T-{transaction_id}`]  **Name**: `{transaction_name}`  |  **Amount**: `{amount_log}`  |  **Date**: `{date}`  |  **Category**: `{category}`")
    
    # Determine direction and preposition based on amount sign
    # Negative amount = spending/outflow = "spent" / "on"
    # Positive amount = receiving/inflow = "received" / "from"
    if amount < 0:
      direction = "spent"
      preposition = "on"
    else:
      direction = "received"
      preposition = "from"
    
    # Check if template has format specifiers for amount (like {amount:,.2f})
    amount_format_pattern = r'\{amount:([^}]+)\}'
    has_amount_format_specifier = bool(re.search(amount_format_pattern, template))
    
    # Check if template has dollar sign - if it does, format amount without $, otherwise with $
    has_dollar_sign = '$' in template
    
    # Check if template has date format specifiers (like {date:%%Y-%%m-%%d})
    # If so, replace them with simple {date} and format the date accordingly
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
    # Apply amount format replacement to temp_template (after date format replacement)
    if has_amount_format_specifier:
      # Template has format specifiers, replace ALL of them with simple {amount}
      temp_template = re.sub(amount_format_pattern, '{amount}', temp_template)
    
    # Format amount string based on template requirements
    if has_amount_format_specifier:
      # Template originally had format specifiers, format with $ if template doesn't have $
      if has_dollar_sign:
        # Template has $, format without $ prefix
        amount_str = f"{abs(amount):,.2f}"
      else:
        # Template has NO $, format WITH $ prefix
        amount_str = f"${abs(amount):,.2f}"
    else:
      # No format specifiers - check if template has dollar signs
      if has_dollar_sign:
        # Template has dollar signs, format without $ prefix
        amount_str = f"{abs(amount):,.2f}"
      else:
        # Template has NO dollar signs, format WITH $ prefix
        amount_str = f"${abs(amount):,.2f}"
    
    # Build format dictionary with available columns
    format_dict = {
      'name': transaction_name,
      'transaction_name': transaction_name,
      'amount': amount_str,
      'date': date_str,
      'category': category,
      'direction': direction,
      'preposition': preposition
    }
    
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
      "transaction_id": transaction_id,
      "transaction_name": transaction_name,
      "amount": amount,
      "date": date_for_metadata,
      "category": category
    })
  
  log(f"**Returning** {len(utterances)} utterances and {len(metadata)} metadata entries.")
  log(f"**Utterances**:\n  - `{'`\n  - `'.join(utterances)}`")
  log(f"**Metadata**:\n```json\n{json.dumps(metadata, indent=2)}\n```")
  return "\n".join(utterances), metadata


def utter_transaction_totals(df: pd.DataFrame, is_spending: bool, template: str) -> str:
  """Calculate total transaction amounts and return formatted string"""
  log(f"**Transaction Totals**: `df: {df.shape}` w/ **cols**:\n  - `{'`, `'.join(df.columns)}`")
  
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
  log(f"**Calculated Total**: **Amount**: `${abs(total_amount):,.2f}`")
  
  # Determine verb/phrase based on is_spending and sign of total_amount
  # Always use abs() for display amount
  display_amount = abs(total_amount)
  
  if is_spending:
    if total_amount < 0:
      verb = "spent"
      amount_suffix = None
    else:  # total_amount > 0
      verb = "received"
      amount_suffix = None
  else:  # not spending (income)
    if total_amount < 0:
      verb = "earned"
      amount_suffix = None
    else:  # total_amount > 0
      verb = "had a"
      amount_suffix = "outflow"
  
  # Check if template has format specifiers for total_amount or amount (like {total_amount:,.2f} or {amount:,.2f})
  total_amount_format_pattern = r'\{total_amount:([^}]+)\}'
  amount_format_pattern = r'\{amount:([^}]+)\}'
  has_total_amount_format_specifier = bool(re.search(total_amount_format_pattern, template))
  has_amount_format_specifier = bool(re.search(amount_format_pattern, template))
  
  # Check if template has dollar sign - if it does, format amount without $, otherwise with $
  has_dollar_sign = '$' in template
  
  # Handle format specifiers - replace with simple placeholders
  temp_template = template
  if has_total_amount_format_specifier:
    # Template has format specifiers, replace them with simple {total_amount}
    temp_template = re.sub(total_amount_format_pattern, '{total_amount}', temp_template)
  if has_amount_format_specifier:
    # Template uses {amount} instead of {total_amount}, replace format specifiers with {total_amount}
    temp_template = re.sub(amount_format_pattern, '{total_amount}', temp_template)
  
  # Also replace any plain {amount} with {total_amount}
  temp_template = temp_template.replace('{amount}', '{total_amount}')
  
  # Format amount string based on template requirements
  has_any_format_specifier = has_total_amount_format_specifier or has_amount_format_specifier
  if has_any_format_specifier:
    # Template originally had format specifiers, format with $ if template doesn't have $
    if has_dollar_sign:
      # Template has $, format without $ prefix
      total_amount_str = f"{display_amount:,.2f}"
    else:
      # Template has NO $, format WITH $ prefix
      total_amount_str = f"${display_amount:,.2f}"
  else:
    # No format specifiers - check if template has dollar signs
    if has_dollar_sign:
      # Template has dollar signs, format without $ prefix
      total_amount_str = f"{display_amount:,.2f}"
    else:
      # Template has NO dollar signs, format WITH $ prefix
      total_amount_str = f"${display_amount:,.2f}"
  
  # Extract category if available and template needs it
  category_value = None
  target_category_value = None
  if 'category' in df.columns:
    if len(df) > 0:
      category_value = df['category'].iloc[0]  # Use first category as representative
      # Format category for display (e.g., "meals_dining_out" -> "dining out")
      if category_value:
        # Replace underscores with spaces and title case
        target_category_value = category_value.replace('_', ' ').title()
        # Remove common prefixes like "meals ", "income ", etc.
        for prefix in ['meals ', 'income ', 'bills ', 'leisure ', 'shelter ']:
          if target_category_value.lower().startswith(prefix):
            target_category_value = target_category_value[len(prefix):]
            break
  
  # Prepare format dictionary with verb, amount_suffix, total_amount, category, and target_category
  format_dict = {
    'total_amount': total_amount_str,
    'verb': verb,
  }
  if amount_suffix:
    format_dict['amount_suffix'] = amount_suffix
  if category_value is not None:
    format_dict['category'] = category_value
  if target_category_value is not None:
    format_dict['target_category'] = target_category_value
  
  try:
    # If template has {verb}, {amount_suffix}, {category}, or {target_category}, use them; otherwise use total_amount
    if '{verb}' in temp_template or '{amount_suffix}' in temp_template or '{category}' in temp_template or '{target_category}' in temp_template:
      result = temp_template.format(**format_dict)
    else:
      # Fall back to just total_amount for backward compatibility
      result = temp_template.format(total_amount=total_amount_str)
  except (ValueError, KeyError) as e:
    # If formatting fails, try with raw numeric amount (only if template had format specifiers)
    if has_any_format_specifier:
      # Try with raw number (format specifier was already replaced)
      try:
        if '{verb}' in temp_template or '{amount_suffix}' in temp_template or '{category}' in temp_template or '{target_category}' in temp_template:
          format_dict_raw = {'total_amount': display_amount, 'verb': verb}
          if amount_suffix:
            format_dict_raw['amount_suffix'] = amount_suffix
          if category_value is not None:
            format_dict_raw['category'] = category_value
          if target_category_value is not None:
            format_dict_raw['target_category'] = target_category_value
          result = temp_template.format(**format_dict_raw)
        else:
          result = temp_template.format(
            total_amount=display_amount
          )
      except Exception as e2:
        # Fall back to formatted string
        result = temp_template.format(
          total_amount=total_amount_str
        )
    else:
      # No format specifiers, so the error is unexpected - re-raise
      raise
  
  return result


def get_income_msg(amount: float) -> str:
  """
  Format income amount with inflow/outflow label.
  
  Args:
    amount: Sum of income amounts (negative = inflow/income, positive = outflow/refunds)
  
  Returns:
    Formatted string like "$100.00" or "$50.00 (outflow)"
  """
  if amount < 0:
    return f"${abs(amount):,.2f}"
  else:
    return f"${amount:,.2f} (outflow)"


def get_spending_msg(amount: float) -> str:
  """
  Format spending/expenses amount with inflow/outflow label.
  
  Args:
    amount: Sum of expense amounts (negative = outflow/expenses, positive = inflow/refunds)
  
  Returns:
    Formatted string like "$100.00" or "$50.00 (inflow)"
  """
  if amount < 0:
    return f"${abs(amount):,.2f}"
  else:
    return f"${amount:,.2f} (inflow)"


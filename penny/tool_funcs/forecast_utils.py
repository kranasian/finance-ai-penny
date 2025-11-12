import re
import pandas as pd
from penny.tool_funcs.sandbox_logging import log


def forecast_dates_and_amount(df: pd.DataFrame, template: str) -> tuple[str, list]:
  """Format forecast dates and amounts using the provided template"""
  
  if df.empty:
    log(f"**Forecast Dates and Amounts**: Empty DataFrame")
    return "", []
  
  # Check if DataFrame has start_date column
  if 'start_date' not in df.columns:
    log(f"**Forecast Dates and Amounts**: DataFrame must have 'start_date' column")
    return "", []
  
  utterances = []
  metadata = []
  
  # Detect format specifiers in template
  format_specifiers = re.findall(r'\{([^}]+):[^}]+\}', template)
  has_format_specifiers = len(format_specifiers) > 0
  
  # Check if template has dollar sign
  has_dollar_sign = '$' in template
  
  # Check if template has format specifiers for amount (we'll handle these specially)
  amount_has_format_specifier = bool(re.search(r'\{amount:[^}]+\}', template)) or bool(re.search(r'\{forecasted_amount:[^}]+\}', template))
  
  # Replace format specifiers with simple placeholders (except for amount/forecasted_amount)
  temp_template = template
  if has_format_specifiers:
    for placeholder in format_specifiers:
      # Don't replace format specifiers for amount/forecasted_amount - let Python handle them
      if placeholder not in ['amount', 'forecasted_amount']:
        # Replace {placeholder:format} with {placeholder}
        temp_template = re.sub(
          r'\{' + re.escape(placeholder) + r':[^}]+\}',
          '{' + placeholder + '}',
          temp_template
        )
  
  # Income categories for determining direction
  income_categories = ['income_salary', 'income_sidegig', 'income_business', 'income_interest']
  
  # Process each forecast row
  for _, row in df.iterrows():
    forecasted_amount = row['forecasted_amount']
    ai_category_id = row.get('ai_category_id', 0)
    category_name = row['category']  # category is always available
    
    # Determine if this is an income category
    is_income = category_name in income_categories
    
    # Determine amount_and_direction based on category and amount sign
    # Rules:
    # 1. Spending (Outflow): amount > 0, not income → "pay for $X.XX"
    # 2. Income (Inflow): amount < 0, income → "receive $X.XX from"
    # 3. Spending Refund (Inflow): amount < 0, not income → "be refunded $X.XX from"
    # 4. Income Adjustment (Outflow): amount > 0, income → "return $X.XX to"
    
    # Format amount as positive for display
    amount_abs = abs(forecasted_amount)
    amount_str_for_direction = f"${amount_abs:.0f}"
    
    # Determine the verb phrase and preposition based on amount sign and category
    if is_income:
      if forecasted_amount < 0:
        # Income (Inflow): amount < 0, income → "receive $X.XX from"
        verb_phrase = "receive"
        preposition = "from"
      else:  # forecasted_amount > 0
        # Income Adjustment (Outflow): amount > 0, income → "return $X.XX to"
        verb_phrase = "return"
        preposition = "to"
    else:  # expense categories
      if forecasted_amount > 0:
        # Spending (Outflow): amount > 0, not income → "pay for $X.XX"
        verb_phrase = "pay for"
        preposition = "for"
      else:  # forecasted_amount < 0
        # Spending Refund (Inflow): amount < 0, not income → "be refunded $X.XX from"
        verb_phrase = "be refunded"
        preposition = "from"
    
    # Build amount_and_direction string: "{verb_phrase} ${amount} {preposition}"
    amount_and_direction = f"{verb_phrase} {amount_str_for_direction} {preposition}"
    
    # Keep direction for backward compatibility
    if is_income:
      if forecasted_amount < 0:
        direction = "earned"
      else:  # forecasted_amount > 0
        direction = "refunded"
    else:  # expense categories
      if forecasted_amount > 0:
        direction = "spent"
      else:  # forecasted_amount < 0
        direction = "received"
    
    # Get date from start_date column (already a datetime object)
    date_dt = row['start_date']
    
    # Format date - check if template has date format specifier
    date_format_pattern = r'\{date:([^}]+)\}'
    start_date_format_pattern = r'\{start_date:([^}]+)\}'
    date_format_match = re.search(date_format_pattern, template)
    start_date_format_match = re.search(start_date_format_pattern, template)
    
    if date_format_match:
      # Extract format specifier (e.g., %%Y-%%m-%%d)
      date_format = date_format_match.group(1).replace('%%', '%')
      # Replace {date:%%Y-%%m-%%d} with {date}
      temp_template = re.sub(date_format_pattern, '{date}', temp_template)
      date_str = date_dt.strftime(date_format)
    elif start_date_format_match:
      # Extract format specifier for start_date (e.g., %%B %%Y for "December 2025")
      date_format = start_date_format_match.group(1).replace('%%', '%')
      # Replace {start_date:%%B %%Y} with {start_date}
      temp_template = re.sub(start_date_format_pattern, '{start_date}', temp_template)
      date_str = date_dt.strftime(date_format)
    else:
      # Default date format YYYY-MM-DD
      date_str = date_dt.strftime('%Y-%m-%d')
    
    # Format amount
    if amount_has_format_specifier:
      # Template has format specifier like {amount:.0f}, provide raw numeric value
      # Python's format will apply the specifier
      amount_value = abs(forecasted_amount)
      amount_str = str(amount_value)  # Fallback string representation
    else:
      # No format specifier, format as string
      if has_dollar_sign:
        amount_str = f"{abs(forecasted_amount):.0f}"
      else:
        amount_str = f"${abs(forecasted_amount):.0f}"
      amount_value = amount_str  # For consistency
    
    # Build format dictionary
    format_dict = {
      'date': date_str,
      'start_date': date_str,
      'direction': direction,
      'amount_and_direction': amount_and_direction,
      'ai_category_id': str(int(ai_category_id)),
      'category': category_name,
    }
    
    # Add amount values - use numeric if format specifier present, otherwise use string
    if amount_has_format_specifier:
      format_dict['amount'] = amount_value
      format_dict['forecasted_amount'] = amount_value
    else:
      format_dict['amount'] = amount_str
      format_dict['forecasted_amount'] = amount_str
    
    # Format the template
    try:
      utterance = temp_template.format(**format_dict)
    except KeyError as e:
      log(f"**Forecast Dates and Amounts**: Missing placeholder in template: {e}")
      utterance = template
    
    # Ensure dollar sign is present if needed
    if not has_dollar_sign and '$' not in utterance and forecasted_amount != 0:
      # Try to add $ before numbers
      utterance = re.sub(r'(\d+[.,]\d+)', r'$\1', utterance)
    
    utterances.append(utterance)
    
    # Add to metadata
    metadata_entry = {
      'ai_category_id': int(ai_category_id),
      'forecasted_amount': float(forecasted_amount),
    }
    metadata_entry['start_date'] = date_str
    
    metadata.append(metadata_entry)
  
  result = "\n".join(utterances)
  
  log(f"**Forecast Dates and Amounts**: Returning {len(utterances)} utterances and {len(metadata)} metadata entries.")
  log(f"**Utterances**:\n  - `{'`\n  - `'.join(utterances)}`")
  
  return result, metadata


def utter_forecast_totals(df: pd.DataFrame, template: str) -> str:
  """Calculate total forecasted amounts and return formatted string"""
  
  log(f"**Forecast Totals**: `df: {df.shape}` w/ **cols**:\n  - `{'`, `'.join(df.columns)}`")
  
  if df.empty:
    log("- **`df` is empty**, returning $0.00.")
    return "$0.00"
  
  # Check if required columns exist
  required_columns = ['forecasted_amount']
  missing_columns = [col for col in required_columns if col not in df.columns]
  if missing_columns:
    error_msg = f"- **`df` is missing required columns**: `{', '.join(missing_columns)}`. Available columns: `{', '.join(df.columns)}`"
    log(error_msg)
    raise ValueError(error_msg)
  
  total_amount = df['forecasted_amount'].sum()
  log(f"**Calculated Total**: **Amount**: `${abs(total_amount):.0f}`")
  
  # Determine if forecasts are income or spending based on categories
  income_categories = ['income_salary', 'income_sidegig', 'income_business', 'income_interest']
  is_income = False
  if len(df) > 0:
    if 'category' in df.columns:
      is_income = df['category'].isin(income_categories).any()
  
  # Determine direction based on category and total_amount sign
  # For income categories: negative = earned (inflow), positive = refunded (outflow)
  # For expense categories: positive = spent (outflow), negative = received (inflow)
  if is_income:
    if total_amount < 0:
      # Income inflow (negative amount = money coming in)
      direction = "earned"
    else:  # total_amount > 0
      # Income outflow (positive amount = money going out, refund)
      direction = "outflow"
  else:
    # Expense categories
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
  
  # Format amount string - always use .0f format, with or without $ based on template
  display_amount = abs(total_amount)
  if has_dollar_sign:
    # Template has $, format without $ prefix
    total_amount_str = f"{display_amount:.0f}"
  else:
    # Template doesn't have $, format with $ prefix
    total_amount_str = f"${display_amount:.0f}"
  
  format_dict = {
    'total_amount': total_amount_str,
    'direction': direction_display,  # Use direction_display which is empty for earned/spent
  }
  
  try:
    result = temp_template.format(**format_dict)
  except (ValueError, KeyError) as e:
    # If formatting fails, try with raw numeric amount
    try:
      format_dict_raw = {
        'total_amount': display_amount,
        'direction': direction_display,
      }
      result = temp_template.format(**format_dict_raw)
    except Exception as e2:
      result = temp_template.format(total_amount=total_amount_str)
  
  # Clean up empty parentheses if direction was empty (e.g., "$100.00 ()" -> "$100.00")
  result = re.sub(r'\s*\(\)', '', result)
  
  log(f"**Forecast Totals**: Template formatted successfully")
  return result

import re
import pandas as pd
from penny.tool_funcs.sandbox_logging import log


def forecast_dates_and_amount(df: pd.DataFrame, template: str) -> tuple[str, list]:
  """Format forecast dates and amounts using the provided template"""
  
  if df.empty:
    log(f"**Forecast Dates and Amounts**: Empty DataFrame")
    return "", []
  
  # Check if this is monthly or weekly forecasts (date columns are optional)
  is_monthly = 'month_date' in df.columns
  is_weekly = 'sunday_date' in df.columns
  
  utterances = []
  metadata = []
  
  # Detect format specifiers in template
  format_specifiers = re.findall(r'\{([^}]+):[^}]+\}', template)
  has_format_specifiers = len(format_specifiers) > 0
  
  # Check if template has dollar sign
  has_dollar_sign = '$' in template
  
  # Replace format specifiers with simple placeholders
  temp_template = template
  if has_format_specifiers:
    for placeholder in format_specifiers:
      # Replace {amount:,.0f} with {amount}
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
    
    # Determine direction based on category and amount sign
    # For income categories: negative = earned (inflow), positive = refunded (outflow)
    # For expense categories: positive = spent (outflow), negative = received (inflow)
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
    
    # Get date based on forecast type (date is always available)
    if is_monthly:
      date_value = row['month_date']
      date_col = 'month_date'
    else:  # is_weekly
      date_value = row['sunday_date']
      date_col = 'sunday_date'
    
    # Format date - check if template has date format specifier
    date_format_pattern = r'\{date:([^}]+)\}'
    date_format_match = re.search(date_format_pattern, template)
    
    if date_format_match:
      # Extract format specifier (e.g., %%Y-%%m-%%d)
      date_format = date_format_match.group(1).replace('%%', '%')
      # Replace {date:%%Y-%%m-%%d} with {date}
      temp_template = re.sub(date_format_pattern, '{date}', temp_template)
      date_str = pd.to_datetime(date_value).strftime(date_format)
    else:
      # Default date format YYYY-MM-DD
      date_str = pd.to_datetime(date_value).strftime('%Y-%m-%d')
    
    # Format amount
    if has_format_specifiers:
      # If template has format specifiers, format the number
      if has_dollar_sign:
        # Template has $, so format without $
        amount_str = f"{abs(forecasted_amount):,.0f}"
      else:
        # Template doesn't have $, so format with $
        amount_str = f"${abs(forecasted_amount):,.0f}"
    else:
      # No format specifiers, just format as string
      if has_dollar_sign:
        amount_str = f"{abs(forecasted_amount):,.0f}"
      else:
        amount_str = f"${abs(forecasted_amount):,.0f}"
    
    # Build format dictionary
    format_dict = {
      'date': date_str,
      'forecasted_amount': amount_str,
      'amount': amount_str,
      'direction': direction,
      'ai_category_id': str(int(ai_category_id)),
      'category': category_name,
    }
    
    if is_monthly:
      format_dict['month_date'] = date_str
    elif is_weekly:
      format_dict['sunday_date'] = date_str
    
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
    if is_monthly:
      metadata_entry['month_date'] = date_str
    elif is_weekly:
      metadata_entry['sunday_date'] = date_str
    
    metadata.append(metadata_entry)
  
  result = "\n".join(utterances)
  
  log(f"**Forecast Dates and Amounts**: Returning {len(utterances)} utterances and {len(metadata)} metadata entries.")
  log(f"**Utterances**:\n  - `{'`\n  - `'.join(utterances)}`")
  
  return result, metadata


def utter_forecasts(df: pd.DataFrame, template: str) -> str:
  """Format forecast totals using the provided template"""
  
  if df.empty:
    log(f"**Utter Forecasts**: Empty DataFrame")
    return ""
  
  # Check if this is monthly or weekly forecasts
  is_monthly = 'month_date' in df.columns
  is_weekly = 'sunday_date' in df.columns
  
  if not is_monthly and not is_weekly:
    log(f"**Utter Forecasts**: DataFrame must have 'month_date' or 'sunday_date' column")
    return ""
  
  # Calculate total forecasted amount
  total_amount = df['forecasted_amount'].sum()
  
  # Determine if forecasts are income or spending based on categories
  income_categories = ['income_salary', 'income_sidegig', 'income_business', 'income_interest']
  if 'category' in df.columns:
    # Check if any category is an income category
    is_income = df['category'].isin(income_categories).any()
  else:
    # Fallback: assume spending if no category info
    is_income = False
  
  # Determine direction based on category and total_amount sign
  # For income categories: negative = earned (inflow), positive = refunded (outflow)
  # For expense categories: positive = spent (outflow), negative = received (inflow)
  if is_income:
    if total_amount < 0:
      direction = "earned"
    else:  # total_amount > 0
      direction = "outflow"
  else:  # expense categories
    if total_amount > 0:
      direction = "spent"
    else:  # total_amount < 0
      direction = "inflow"
  
  # Only show direction for inflow/outflow, not for earned/spent
  # If direction is "earned" or "spent", set to empty string so it won't appear in templates
  if direction in ["earned", "spent"]:
    direction_display = ""
  else:
    direction_display = direction
  
  # Check if template has format specifiers for total_amount or amount (like {total_amount:,.0f} or {amount:,.0f})
  total_amount_format_pattern = r'\{total_amount:([^}]+)\}'
  amount_format_pattern = r'\{amount:([^}]+)\}'
  has_total_amount_format_specifier = bool(re.search(total_amount_format_pattern, template))
  has_amount_format_specifier = bool(re.search(amount_format_pattern, template))
  
  # Check if template has dollar sign
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
  display_amount = abs(total_amount)
  
  if has_any_format_specifier:
    # Template originally had format specifiers, format with $ if template doesn't have $
    if has_dollar_sign:
      # Template has $, format without $ prefix
      total_amount_str = f"{display_amount:,.0f}"
    else:
      # Template has NO $, format WITH $ prefix
      total_amount_str = f"${display_amount:,.0f}"
  else:
    # No format specifiers - check if template has dollar signs
    if has_dollar_sign:
      # Template has dollar signs, format without $ prefix
      total_amount_str = f"{display_amount:,.0f}"
    else:
      # Template has NO dollar signs, format WITH $ prefix
      total_amount_str = f"${display_amount:,.0f}"
  
  # Build format dictionary
  format_dict = {
    'forecasted_amount': total_amount_str,
    'total': total_amount_str,
    'total_amount': total_amount_str,
    'amount': total_amount_str,
    'direction': direction_display,  # Use direction_display which is empty for earned/spent
  }
  
  # Add date column based on type
  if is_monthly:
    # Get unique month dates
    month_dates = df['month_date'].unique()
    if len(month_dates) == 1:
      format_dict['month_date'] = pd.to_datetime(month_dates[0]).strftime('%Y-%m-%d')
    else:
      format_dict['month_date'] = f"{len(month_dates)} months"
  elif is_weekly:
    # Get unique Sunday dates
    sunday_dates = df['sunday_date'].unique()
    if len(sunday_dates) == 1:
      format_dict['sunday_date'] = pd.to_datetime(sunday_dates[0]).strftime('%Y-%m-%d')
    else:
      format_dict['sunday_date'] = f"{len(sunday_dates)} weeks"
  
  # Add category count
  unique_categories = df['ai_category_id'].nunique()
  format_dict['category_count'] = str(unique_categories)
  format_dict['categories'] = str(unique_categories)
  
  # Format the template
  try:
    # If template has {direction} or {total_amount}, use them; otherwise use forecasted_amount for backward compatibility
    if '{direction}' in temp_template or '{total_amount}' in temp_template:
      result = temp_template.format(**format_dict)
    else:
      # Fall back to just forecasted_amount for backward compatibility
      result = temp_template.format(forecasted_amount=total_amount_str)
  except (ValueError, KeyError) as e:
    # If formatting fails, try with raw numeric amount (only if template had format specifiers)
    if has_any_format_specifier:
      # Try with raw number (format specifier was already replaced)
      try:
        if '{direction}' in temp_template or '{total_amount}' in temp_template:
          format_dict_raw = {'total_amount': display_amount, 'direction': direction_display}
          result = temp_template.format(**format_dict_raw)
        else:
          result = temp_template.format(forecasted_amount=display_amount)
      except Exception as e2:
        # Fall back to formatted string
        result = temp_template.format(forecasted_amount=total_amount_str)
    else:
      # No format specifiers, so the error is unexpected - re-raise
      raise
  
  # Ensure dollar sign is present if needed
  if not has_dollar_sign and '$' not in result and total_amount != 0:
    # Try to add $ before numbers
    result = re.sub(r'(\d+[.,]\d+)', r'$\1', result)
  
  log(f"**Utter Forecasts**: Template formatted successfully")
  return result


import re
import pandas as pd
from penny.tool_funcs.sandbox_logging import log
from categories import get_all_parent_categories, get_parents_with_leaves_as_dict_categories, get_name

MAX_FORECASTS = 10


def _consolidate_parent_categories(df: pd.DataFrame) -> tuple[pd.DataFrame, set[int]]:
  """Consolidate child categories into parent categories when all children are present.
  
  This function will replace child category rows with their parent category row when
  all children for the same start_date are present. The parent forecast amount is the
  sum of all children (original forecast).
  
  If the DataFrame has more than 10 rows, consolidation happens automatically.
  If the DataFrame has 10 or fewer rows, consolidation still happens but only when
  all children of a parent are present.
  
  Args:
    df: DataFrame with forecast data
    
  Returns:
    tuple: (consolidated DataFrame, set of consolidated parent category IDs)
  """
  total_count = len(df)
  consolidated_parent_ids = set()
  
  parent_to_leaf_categories = get_parents_with_leaves_as_dict_categories()
  df_consolidated = df.copy()
  rows_to_remove = []
  rows_to_add = []
  
  # Group by start_date to process each date separately
  for start_date, date_group in df.groupby('start_date'):
    # For each parent category, check if all children are present
    for parent_id, children_ids in parent_to_leaf_categories.items():
      # Get children category IDs (excluding the parent itself)
      child_ids = [cid for cid in children_ids if cid != parent_id]
      
      if not child_ids:
        continue
      
      # Check if all children are present in this date group
      children_in_group = date_group[date_group['ai_category_id'].isin(child_ids)]
      if len(children_in_group) == len(child_ids):
        # All children are present - consolidate into parent
        # Only consolidate if df has more than 10 rows, OR if this is a filtered query
        # (we can detect filtered queries by checking if parent is not in the original df)
        parent_in_df = date_group[date_group['ai_category_id'] == parent_id]
        should_consolidate = total_count > 10 or parent_in_df.empty
        
        if should_consolidate:
          # Use original_forecasted_amount from parent if available, otherwise sum children
          if not parent_in_df.empty and 'original_forecasted_amount' in parent_in_df.columns:
            parent_forecast = parent_in_df['original_forecasted_amount'].iloc[0]
          else:
            parent_forecast = children_in_group['forecasted_amount'].sum()
          
          # Get parent category name
          parent_category_name = get_name(parent_id)
          if not parent_category_name:
            continue
          
          # Mark child rows for removal
          child_indices = children_in_group.index.tolist()
          rows_to_remove.extend(child_indices)
          
          # Create parent row
          parent_row = children_in_group.iloc[0].copy()
          parent_row['ai_category_id'] = parent_id
          parent_row['category'] = parent_category_name
          parent_row['forecasted_amount'] = parent_forecast
          # Set original_forecasted_amount if it exists
          if 'original_forecasted_amount' in parent_row:
            parent_row['original_forecasted_amount'] = parent_forecast
          rows_to_add.append(parent_row)
          consolidated_parent_ids.add(parent_id)
          
          log(f"**Consolidating Parent Category**: Replacing {len(child_ids)} children of parent {parent_id} ({parent_category_name}) with parent forecast ${parent_forecast:.0f} for {start_date}")
  
  # Remove child rows and add parent rows
  if rows_to_remove:
    df_consolidated = df_consolidated.drop(index=rows_to_remove)
    if rows_to_add:
      parent_df = pd.DataFrame(rows_to_add)
      df_consolidated = pd.concat([df_consolidated, parent_df], ignore_index=True)
  
  return df_consolidated, consolidated_parent_ids


def _format_date_for_metadata(date_dt, template: str, temp_template: str) -> tuple[str, str]:
  """
  Convert a date object to a JSON-serializable string for metadata.
  
  Args:
    date_dt: The date object (datetime, Timestamp, etc.)
    template: The original template string
    temp_template: The template string being modified (may have format specifiers removed)
  
  Returns:
    tuple: (formatted_date_string, modified_temp_template)
  """
  # Ensure date_dt is always converted to a string for JSON serialization
  try:
    if date_dt is None or (hasattr(pd, 'NaT') and pd.isna(date_dt)):
      return '', temp_template
    elif hasattr(date_dt, 'strftime'):
      # Format date - check if template has date format specifier
      date_format_pattern = r'\{date:([^}]+)\}'
      start_date_format_pattern = r'\{start_date:([^}]+)\}'
      date_format_match = re.search(date_format_pattern, template)
      start_date_format_match = re.search(start_date_format_pattern, template)
      
      if date_format_match:
        # Extract format specifier (e.g., %%Y-%%m-%%d)
        date_format = date_format_match.group(1).replace('%%', '%')
        # Replace {date:%%Y-%%m-%%d} with {date}
        modified_template = re.sub(date_format_pattern, '{date}', temp_template)
        date_str = date_dt.strftime(date_format)
        return date_str, modified_template
      elif start_date_format_match:
        # Extract format specifier for start_date (e.g., %%B %%Y for "December 2025")
        date_format = start_date_format_match.group(1).replace('%%', '%')
        # Replace {start_date:%%B %%Y} with {start_date}
        modified_template = re.sub(start_date_format_pattern, '{start_date}', temp_template)
        date_str = date_dt.strftime(date_format)
        return date_str, modified_template
      else:
        # Default date format YYYY-MM-DD
        date_str = date_dt.strftime('%Y-%m-%d')
        return date_str, temp_template
    else:
      # Fallback: convert to string if it's not a datetime-like object
      date_str = str(date_dt)
      return date_str, temp_template
  except Exception as e:
    # If anything goes wrong, convert to string as fallback
    log(f"**Forecast Dates and Amounts**: Error formatting date {date_dt}: {e}, converting to string")
    if date_dt is None:
      return '', temp_template
    elif isinstance(date_dt, pd.Timestamp):
      # Convert pandas Timestamp to ISO format string
      date_str = date_dt.strftime('%Y-%m-%d')
      return date_str, temp_template
    else:
      # Try to convert to string, but ensure it's JSON-serializable
      try:
        date_str = str(date_dt)
        return date_str, temp_template
      except:
        return '', temp_template


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
  
  # Limit both utterances and metadata to MAX_FORECASTS
  total_count = len(df)
  has_more = total_count > MAX_FORECASTS
  
  log(f"**Listing Individual Forecasts**: Processing up to {MAX_FORECASTS} items (out of {total_count} total).")
  
  # Get parent category IDs to filter them out
  parent_category_ids = set(get_all_parent_categories())
  
  # Consolidate child categories into parent categories when all children are present
  df, consolidated_parent_ids = _consolidate_parent_categories(df)
  
  # Process each forecast row
  for _, row in df.iterrows():
    ai_category_id = row.get('ai_category_id', 0)
    category_name = row['category']  # category is always available
    
    # Skip parent category forecasts (unless they were added through consolidation)
    if ai_category_id in parent_category_ids and ai_category_id not in consolidated_parent_ids:
      continue
    
    # Use original_forecasted_amount for parent categories if available, otherwise use forecasted_amount
    if ai_category_id in parent_category_ids and 'original_forecasted_amount' in row:
      forecasted_amount = row['original_forecasted_amount']
    else:
      forecasted_amount = row['forecasted_amount']
    
    # Determine if this is an income category
    is_income = category_name in income_categories
    
    # Determine amount_and_direction based on category and amount sign
    # Rules:
    # 1. Spending (Outflow): amount > 0, not income → "spend $X.XX for"
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
        # Spending (Outflow): amount > 0, not income → "spend $X.XX for"
        verb_phrase = "spend"
        preposition = "for"
      else:  # forecasted_amount < 0
        # Spending Refund (Inflow): amount < 0, not income → "be refunded $X.XX from"
        verb_phrase = "be refunded"
        preposition = "from"
    
    # Build amount_and_direction string: "{verb_phrase} ${amount} {preposition}"
    if preposition:
      amount_and_direction = f"{verb_phrase} {amount_str_for_direction} {preposition}"
    else:
      amount_and_direction = f"{verb_phrase} {amount_str_for_direction}"
    
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
    
    # Format date for metadata (ensures JSON serialization)
    date_str, temp_template = _format_date_for_metadata(date_dt, template, temp_template)
    
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
    
    # Add to utterances only if under the limit (max MAX_FORECASTS)
    if len(utterances) < MAX_FORECASTS:
      utterances.append(utterance)
    
    # Add to metadata only if under the limit (max MAX_FORECASTS)
    if len(metadata) < MAX_FORECASTS:
      metadata_entry = {
        'ai_category_id': int(ai_category_id),
        'forecasted_amount': float(forecasted_amount),
      }
      metadata_entry['start_date'] = date_str
      metadata.append(metadata_entry)
    
    # Break early if we've reached both limits
    if len(metadata) >= MAX_FORECASTS and len(utterances) >= MAX_FORECASTS:
      break
  
  # Add message about remaining forecasts if there are more
  result = "\n".join(utterances)
  if has_more:
    remaining_count = total_count - MAX_FORECASTS
    result += f"\n{remaining_count} more forecast{'s' if remaining_count != 1 else ''}."
  
  log(f"**Forecast Dates and Amounts**: Returning {len(utterances)} utterances and {len(metadata)} metadata entries. Has more: {has_more}")
  log(f"**Utterances**:\n  - `{'`\n  - `'.join(utterances)}`")
  
  return result, metadata


def _format_forecast_total(df: pd.DataFrame, template: str, is_income: bool) -> str:
  """Helper function to format forecast totals for income or spending"""
  
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
  
  # Determine verb and direction based on category and total_amount sign
  # For income categories: negative = earned (inflow), positive = returned (outflow)
  # For expense categories: positive = spent (outflow), negative = received (inflow)
  if is_income:
    if total_amount < 0:
      # Income inflow (negative amount = money coming in)
      verb = "earn"
      direction = "earned"
    else:  # total_amount > 0
      # Income outflow (positive amount = money going out, return/refund)
      verb = "return"
      direction = "outflow"
  else:
    # Expense categories
    if total_amount > 0:
      # Spending outflow (positive amount = money going out)
      verb = "spend"
      direction = "spent"
    else:  # total_amount < 0
      # Spending inflow (negative amount = money coming in, refund)
      verb = "receive"
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
  verb_and_total_amount_format_pattern = r'\{verb_and_total_amount:([^}]+)\}'
  has_total_amount_format_specifier = bool(re.search(total_amount_format_pattern, template))
  has_amount_format_specifier = bool(re.search(amount_format_pattern, template))
  has_verb_and_total_amount_format_specifier = bool(re.search(verb_and_total_amount_format_pattern, template))
  
  # Check if template has dollar sign - if it does, format amount without $, otherwise with $
  has_dollar_sign = '$' in template
  
  # Handle format specifiers - replace with simple placeholders
  temp_template = template
  if has_total_amount_format_specifier:
    temp_template = re.sub(total_amount_format_pattern, '{total_amount}', temp_template)
  if has_amount_format_specifier:
    temp_template = re.sub(amount_format_pattern, '{total_amount}', temp_template)
  if has_verb_and_total_amount_format_specifier:
    temp_template = re.sub(verb_and_total_amount_format_pattern, '{verb_and_total_amount}', temp_template)
  
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
  
  # Create combined verb_and_total_amount string (e.g., "earn $5000" or "spend $3000")
  verb_and_total_amount_str = f"{verb} {total_amount_str}"
  
  format_dict = {
    'total_amount': total_amount_str,
    'direction': direction_display,  # Use direction_display which is empty for earned/spent
    'verb': verb,  # Verb for use in templates like "earn", "spend", "receive", "return"
    'verb_and_total_amount': verb_and_total_amount_str,  # Combined verb and amount (e.g., "earn $5000")
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


def utter_income_forecast_totals(df: pd.DataFrame, template: str) -> str:
  """Calculate total income forecasted amounts and return formatted string.
  
  Args:
    df: DataFrame with income forecasts (must have 'forecasted_amount' column)
    template: Template string with placeholders:
      - {verb_and_total_amount}: Combined verb and amount (e.g., "earn $5000" or "return $500")
      - {verb}: Verb only (e.g., "earn", "return")
      - {total_amount}: Amount only (e.g., "$5000")
      - {direction}: Direction display (empty for earned/returned, or "(outflow)")
  
  Returns:
    Formatted string with income forecast totals.
    Returns "$0.00" if DataFrame is empty.
  """
  return _format_forecast_total(df, template, is_income=True)


def utter_spending_forecast_totals(df: pd.DataFrame, template: str) -> str:
  """Calculate total spending forecasted amounts and return formatted string.
  
  Args:
    df: DataFrame with spending forecasts (must have 'forecasted_amount' column)
    template: Template string with placeholders:
      - {verb_and_total_amount}: Combined verb and amount (e.g., "spend $3000" or "receive $500")
      - {verb}: Verb only (e.g., "spend", "receive")
      - {total_amount}: Amount only (e.g., "$3000")
      - {direction}: Direction display (empty for spent/received, or "(inflow)")
  
  Returns:
    Formatted string with spending forecast totals.
    Returns "$0.00" if DataFrame is empty.
  """
  return _format_forecast_total(df, template, is_income=False)


def utter_spending_forecast_amount(amount: float, template: str) -> str:
  """Format a spending forecast amount with appropriate verb and direction.
  
  Args:
    amount: Spending forecast amount (positive = outflow/spent, negative = inflow/received)
    template: Template string with placeholders:
      - {verb_and_amount}: Combined verb and amount (e.g., "spend $3000" or "receive $500")
      - {amount_and_direction}: Combined amount and direction (e.g., "$3000" or "$500 (inflow)")
      - {verb}: Verb only (e.g., "spend", "receive")
      - {amount}: Amount only (e.g., "$3000")
      - {direction}: Direction display (empty for spent, or "(inflow)" for received)
  
  Returns:
    Formatted string with spending forecast amount.
  """
  log(f"**Spending Forecast Amount**: Amount: ${amount:.0f}")
  
  # Determine verb and direction based on amount sign
  # Positive = spent (outflow), negative = received (inflow)
  if amount > 0:
    # Spending outflow (positive amount = money going out)
    verb = "spend"
    direction = "spent"
  elif amount < 0:
    # Spending inflow (negative amount = money coming in, refund)
    verb = "receive"
    direction = "inflow"
  else:
    # Zero amount
    verb = "spend"
    direction = "spent"
  
  # Only show direction for inflow, not for spent
  if direction == "spent":
    direction_display = ""
  else:
    direction_display = f"({direction})"
  
  # Format amount string
  display_amount = abs(amount)
  amount_str = f"${display_amount:.0f}"
  
  # Create combined verb_and_amount string
  verb_and_amount_str = f"{verb} {amount_str}"
  
  # Create combined amount_and_direction string
  amount_and_direction_str = f"{amount_str} {direction_display}".strip()
  
  format_dict = {
    'amount': amount_str,
    'direction': direction_display,
    'verb': verb,
    'verb_and_amount': verb_and_amount_str,
    'amount_and_direction': amount_and_direction_str,
  }
  
  result = template.format(**format_dict)
  log(f"**Spending Forecast Amount Utterance**: `{result}`")
  return result


def utter_income_forecast_amount(amount: float, template: str) -> str:
  """Format an income forecast amount with appropriate verb and direction.
  
  Args:
    amount: Income forecast amount (negative = inflow/earned, positive = outflow/returned)
    template: Template string with placeholders:
      - {verb_and_amount}: Combined verb and amount (e.g., "earn $5000" or "return $500")
      - {amount_and_direction}: Combined amount and direction (e.g., "$5000" or "$500 (outflow)")
      - {verb}: Verb only (e.g., "earn", "return")
      - {amount}: Amount only (e.g., "$5000")
      - {direction}: Direction display (empty for earned, or "(outflow)" for returned)
  
  Returns:
    Formatted string with income forecast amount.
  """
  log(f"**Income Forecast Amount**: Amount: ${amount:.0f}")
  
  # Determine verb and direction based on amount sign
  # Negative = earned (inflow), positive = returned (outflow)
  if amount < 0:
    # Income inflow (negative amount = money coming in)
    verb = "earn"
    direction = "earned"
  elif amount > 0:
    # Income outflow (positive amount = money going out, return/refund)
    verb = "return"
    direction = "outflow"
  else:
    # Zero amount
    verb = "earn"
    direction = "earned"
  
  # Only show direction for outflow, not for earned
  if direction == "earned":
    direction_display = ""
  else:
    direction_display = f"({direction})"
  
  # Format amount string
  display_amount = abs(amount)
  amount_str = f"${display_amount:.0f}"
  
  # Create combined verb_and_amount string
  verb_and_amount_str = f"{verb} {amount_str}"
  
  # Create combined amount_and_direction string
  amount_and_direction_str = f"{amount_str} {direction_display}".strip()
  
  format_dict = {
    'amount': amount_str,
    'direction': direction_display,
    'verb': verb,
    'verb_and_amount': verb_and_amount_str,
    'amount_and_direction': amount_and_direction_str,
  }
  
  result = template.format(**format_dict)
  log(f"**Income Forecast Amount Utterance**: `{result}`")
  return result


def utter_balance(amount: float, template: str) -> str:
  """Format a balance amount (positive or negative) with appropriate sign and direction.
  
  Use this method for:
  - Account balances (balance_available, balance_current, remaining_balance)
  - Balance differences (shortfall, deficit_after)
  - Any financial amount representing a balance or difference that can be positive or negative
  
  Args:
    amount: Balance value (positive or negative)
    template: Template string with placeholder:
      - {amount_with_direction}: Amount with direction indicator (e.g., "$1000" or "$500 deficit")
  
  Returns:
    Formatted string with balance amount.
  """
  log(f"**Balance**: Amount: ${amount:.0f}")
  
  # Determine if negative
  is_negative = amount < 0
  
  # Format amount strings
  amount_abs = abs(amount)
  amount_str = f"${amount_abs:.0f}"
  
  # Amount with direction: "$X deficit" for negative, "$X" for positive
  if is_negative:
    amount_with_direction_str = f"{amount_str} deficit"
  else:
    amount_with_direction_str = amount_str
  
  format_dict = {
    'amount_with_direction': amount_with_direction_str,
  }
  
  result = template.format(**format_dict)
  log(f"**Balance Utterance**: `{result}`")
  return result


def utter_amount(amount: float, template: str) -> str:
  """Format a transaction total amount as a positive number.
  
  Use this method for:
  - Transaction totals (always displays as positive, no sign indicators)
  - Any financial amount that should be displayed as a simple positive number
  
  Args:
    amount: Transaction amount (positive or negative, will be displayed as absolute value)
    template: Template string with placeholder:
      - {amount}: Amount formatted as positive number. If template doesn't include "$", dollar sign will be added automatically.
  
  Returns:
    Formatted string with amount (always positive, no commas, 0 decimals). Dollar sign added if not in template.
  """
  log(f"**Amount**: Amount: ${amount:.0f}")
  
  # Format as positive number (absolute value), no commas, 0 decimals
  amount_abs = abs(amount)
  amount_str = f"{amount_abs:.0f}"
  
  # Add dollar sign if template doesn't already include it
  if "$" not in template:
    amount_str = f"${amount_str}"
  
  format_dict = {
    'amount': amount_str,
  }
  
  result = template.format(**format_dict)
  log(f"**Amount Utterance**: `{result}`")
  return result

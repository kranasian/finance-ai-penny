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


def forecast_dates_and_amount(df: pd.DataFrame, template: str) -> str:
  """Format forecast dates and amounts using the provided template"""
  
  if df.empty:
    log(f"**Forecast Dates and Amounts**: Empty DataFrame")
    return ""
  
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
  
  log(f"**Forecast Dates and Amounts**: Returning {len(utterances)} utterances. Has more: {has_more}")
  utterances_str = "`\n  - `".join(utterances)
  log(f"**Utterances**:\n  - `{utterances_str}`")
  
  return result


def utter_forecast_amount(amount: float, template: str) -> str:
  """Format a forecast amount with appropriate verb.
  
  Provides both {income_total_amount} and {spending_total_amount} placeholders.
  For income: positive = earned, negative = lost.
  For spending: positive = spent, negative = received.
  The template should use the appropriate placeholder based on forecast type.
  
  Args:
    amount: Forecast amount (float). Sum DataFrame's forecasted_amount column before calling.
    template: Template string with {income_total_amount} or {spending_total_amount} placeholder.
      Example: "Income: {income_total_amount}" or "Spending: {spending_total_amount}"
      The placeholder automatically includes the verb (e.g., "earn $5000" or "spend $3000").
  
  Returns:
    Formatted string with verb and amount inserted.
  """
  log(f"**Forecast Amount**: Amount: ${amount:.0f}")
  
  # Format amount with dollar sign (use absolute for display)
  display_amount = abs(amount)
  amount_str = f"${display_amount:.0f}"
  
  # Determine verb/phrase for income: positive = earned, negative = lost
  # Determine verb/phrase for spending: positive = spent, negative = received
  if amount >= 0:
    income_verb = "earn"
    spending_verb = "spend"
  else:  # amount < 0
    income_verb = "lose"
    spending_verb = "receive"
  
  income_total_amount_str = f"{income_verb} {amount_str}"
  spending_total_amount_str = f"{spending_verb} {amount_str}"
  
  # Build format dictionary - always provide both placeholders
  format_dict = {
    'income_total_amount': income_total_amount_str,
    'spending_total_amount': spending_total_amount_str,
    'amount': amount_str,
    'total_amount': amount_str,
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
  
  log(f"**Forecast Amount Utterance**: `{result}`")
  return result


def utter_absolute_amount(amount: float, template: str) -> str:
  """Format an absolute amount with support for both simple and directional formatting.
  
  Use this method for:
  - Account balances (balance_available, balance_current, remaining_balance)
  - Balance differences (shortfall, deficit_after)
  - Transaction totals (always displays as positive, no sign indicators)
  - Any financial amount that should be displayed as an absolute value
  
  Args:
    amount: Amount value (positive or negative, will be displayed as absolute value)
    template: Template string with placeholders:
      - {amount}: Amount formatted as positive number. If template doesn't include "$", dollar sign will be added automatically.
      - {amount_with_direction}: Amount with direction indicator (e.g., "$1000" or "$500 deficit")
  
  Returns:
    Formatted string with amount. Supports both {amount} and {amount_with_direction} placeholders.
  """
  log(f"**Absolute Amount**: Amount: ${amount:.0f}")
  
  # Determine if negative
  is_negative = amount < 0
  
  # Format as positive number (absolute value), no commas, 0 decimals
  amount_abs = abs(amount)
  amount_str = f"{amount_abs:.0f}"
  
  # Add dollar sign if template doesn't already include it (for {amount} placeholder)
  if "$" not in template:
    amount_str_with_dollar = f"${amount_str}"
  else:
    amount_str_with_dollar = amount_str
  
  # Amount with direction: "$X deficit" for negative, "$X" for positive
  if is_negative:
    amount_with_direction_str = f"{amount_str_with_dollar} deficit"
    income_total_amount_str = f"{amount_str_with_dollar} lost"
    spending_total_amount_str = f"{amount_str_with_dollar} received"
  else:
    amount_with_direction_str = amount_str_with_dollar
    income_total_amount_str = f"{amount_str_with_dollar} earned"
    spending_total_amount_str = f"{amount_str_with_dollar} spent"
  
  format_dict = {
    'amount': amount_with_direction_str,
    'income_total_amount': income_total_amount_str,
    'spending_total_amount': spending_total_amount_str,
    "total_amount": amount_with_direction_str,
    'amount_with_direction': amount_with_direction_str,
  }
  
  result = template.format(**format_dict)
  log(f"**Absolute Amount Utterance**: `{result}`")
  return result

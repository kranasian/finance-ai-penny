import json
import pandas as pd
import re
from sandbox_logging import log


def compare_spending(df: pd.DataFrame, template: str) -> tuple[str, list]:
  """Compare spending data between categories or time periods and return formatted string with metadata"""
  log(f"**Compare Spending**: `df: {df.shape}` w/ **cols**:\n  - `{'`, `'.join(df.columns)}`")
  
  if df.empty:
    log("- **`df` is empty**, returning empty string and metadata.")
    return "", []
  
  # Check if required columns exist
  if 'category' not in df.columns:
    error_msg = f"- **`df` is missing required 'category' column**. Available columns: `{', '.join(df.columns)}`"
    log(error_msg)
    raise ValueError(error_msg)
  
  # If 'group' column exists, use it for grouping (allows grouping multiple categories)
  # Otherwise, group by category
  if 'group' in df.columns:
    groups = df['group'].dropna().unique()
    if len(groups) == 0:
      error_msg = f"- **`df` has 'group' column but no groups found after filtering**. Available columns: `{', '.join(df.columns)}`"
      log(error_msg)
      raise ValueError(error_msg)
    if len(groups) == 1:
      # Only one group present - return spending for that single group
      group_name = str(groups[0])
      group_df = df[df['group'] == groups[0]]
      # Use absolute value to handle both spending (negative) and earnings (positive)
      total = abs(group_df['amount'].sum())
      count = len(group_df)
      
      # Format category name for display
      def format_category_name(cat: str) -> str:
        if '_' in cat:
          return ' '.join(word.capitalize() for word in cat.split('_'))
        return cat.capitalize()
      
      group_label = format_category_name(group_name)
      
      # Handle format specifiers in template (e.g., {amount:,.2f})
      temp_template = template
      temp_template = re.sub(r'\{amount:[^}]+\}', '{amount}', temp_template)
      temp_template = re.sub(r'\{total:[^}]+\}', '{total}', temp_template)
      temp_template = re.sub(r'\{count:[^}]+\}', '{count}', temp_template)
      
      # Check if template has dollar sign
      has_dollar_sign = '$' in template
      if has_dollar_sign:
        amount_str = f"{total:,.2f}"
      else:
        amount_str = f"${total:,.2f}"
      
      # Try to use template, fallback to default message
      try:
        result = temp_template.format(
          group_name=group_name,
          group_label=group_label,
          amount=amount_str,
          total=total,
          count=count
        )
      except (KeyError, ValueError):
        # Default message if template doesn't have the right placeholders
        result = f"You only have transactions in {group_label} this month. You spent {amount_str} on {group_label} ({count} transactions)."
      
      metadata = [{
        "comparison_type": "single_group",
        "group": {
          "name": group_name,
          "label": group_label,
          "total": float(total),
          "count": int(count)
        }
      }]
      
      log(f"**Returning** single group result and metadata.")
      log(f"**Result**: `{result}`")
      log(f"**Metadata**:\n```json\n{json.dumps(metadata, indent=2)}\n```")
      
      return result, metadata
    elif len(groups) != 2:
      error_msg = f"- **`df` must have exactly 1 or 2 groups for comparison**. Found {len(groups)} groups: {list(groups)}"
      log(error_msg)
      raise ValueError(error_msg)
    
    group1_name = str(groups[0])
    group2_name = str(groups[1])
    df1 = df[df['group'] == groups[0]]
    df2 = df[df['group'] == groups[1]]
  else:
    # Group by category
    categories = df['category'].dropna().unique()
    if len(categories) == 0:
      error_msg = f"- **`df` has no categories found after filtering**. Available columns: `{', '.join(df.columns)}`"
      log(error_msg)
      raise ValueError(error_msg)
    if len(categories) == 1:
      # Only one category present - return spending for that single category
      category_name = str(categories[0])
      category_df = df[df['category'] == categories[0]]
      # Use absolute value to handle both spending (negative) and earnings (positive)
      total = abs(category_df['amount'].sum())
      count = len(category_df)
      
      # Format category name for display
      def format_category_name(cat: str) -> str:
        if '_' in cat:
          return ' '.join(word.capitalize() for word in cat.split('_'))
        return cat.capitalize()
      
      category_label = format_category_name(category_name)
      
      # Handle format specifiers in template (e.g., {amount:,.2f})
      temp_template = template
      temp_template = re.sub(r'\{amount:[^}]+\}', '{amount}', temp_template)
      temp_template = re.sub(r'\{total:[^}]+\}', '{total}', temp_template)
      temp_template = re.sub(r'\{count:[^}]+\}', '{count}', temp_template)
      
      # Check if template has dollar sign
      has_dollar_sign = '$' in template
      if has_dollar_sign:
        amount_str = f"{total:,.2f}"
      else:
        amount_str = f"${total:,.2f}"
      
      # Try to use template, fallback to default message
      try:
        result = temp_template.format(
          category_name=category_name,
          category_label=category_label,
          amount=amount_str,
          total=total,
          count=count
        )
      except (KeyError, ValueError):
        # Default message if template doesn't have the right placeholders
        result = f"You only have transactions in {category_label} this month. You spent {amount_str} on {category_label} ({count} transactions)."
      
      metadata = [{
        "comparison_type": "single_category",
        "category": {
          "name": category_name,
          "label": category_label,
          "total": float(total),
          "count": int(count)
        }
      }]
      
      log(f"**Returning** single category result and metadata.")
      log(f"**Result**: `{result}`")
      log(f"**Metadata**:\n```json\n{json.dumps(metadata, indent=2)}\n```")
      
      return result, metadata
    elif len(categories) != 2:
      error_msg = f"- **`df` must have exactly 1 or 2 categories for comparison**. Found {len(categories)} categories: {list(categories)}. To compare groups of categories, create a 'group' column that maps categories to two groups."
      log(error_msg)
      raise ValueError(error_msg)
    
    group1_name = str(categories[0])
    group2_name = str(categories[1])
    df1 = df[df['category'] == categories[0]]
    df2 = df[df['category'] == categories[1]]
  
  # Ensure df1 and df2 are not empty
  if df1.empty or df2.empty:
    empty_group = "first" if df1.empty else "second"
    error_msg = f"- **`df` has empty {empty_group} group after filtering**. This comparison cannot be performed."
    log(error_msg)
    raise ValueError(error_msg)
  
  # Calculate totals for each group (using absolute value for spending)
  total1 = abs(df1['amount'].sum())
  total2 = abs(df2['amount'].sum())
  
  log(f"**Comparison**: Group 1 (`{group1_name}`): `${total1:,.2f}` | Group 2 (`{group2_name}`): `${total2:,.2f}`")
  
  # Format amounts based on template
  # Check if template has format specifiers for any placeholders
  # Replace all format specifiers with simple placeholders to avoid format errors
  temp_template = template
  
  # Replace format specifiers for all numeric placeholders
  temp_template = re.sub(r'\{first_amount:[^}]+\}', '{first_amount}', temp_template)
  temp_template = re.sub(r'\{second_amount:[^}]+\}', '{second_amount}', temp_template)
  temp_template = re.sub(r'\{first_total:[^}]+\}', '{first_total}', temp_template)
  temp_template = re.sub(r'\{second_total:[^}]+\}', '{second_total}', temp_template)
  temp_template = re.sub(r'\{difference:[^}]+\}', '{difference}', temp_template)
  temp_template = re.sub(r'\{first_count:[^}]+\}', '{first_count}', temp_template)
  temp_template = re.sub(r'\{second_count:[^}]+\}', '{second_count}', temp_template)
  temp_template = re.sub(r'\{count_difference:[^}]+\}', '{count_difference}', temp_template)
  temp_template = re.sub(r'\{more_amount:[^}]+\}', '{more_amount}', temp_template)
  temp_template = re.sub(r'\{less_amount:[^}]+\}', '{less_amount}', temp_template)
  temp_template = re.sub(r'\{more_total:[^}]+\}', '{more_total}', temp_template)
  temp_template = re.sub(r'\{less_total:[^}]+\}', '{less_total}', temp_template)
  temp_template = re.sub(r'\{more_count:[^}]+\}', '{more_count}', temp_template)
  temp_template = re.sub(r'\{less_count:[^}]+\}', '{less_count}', temp_template)
  
  # Check if template has dollar sign
  has_dollar_sign = '$' in template
  
  # Format amounts as strings (always format as strings, not numbers)
  if has_dollar_sign:
    first_amount_str = f"{total1:,.2f}"
    second_amount_str = f"{total2:,.2f}"
  else:
    first_amount_str = f"${total1:,.2f}"
    second_amount_str = f"${total2:,.2f}"
  
  # Calculate difference in amounts
  difference = total1 - total2
  difference_abs = abs(difference)
  difference_str = f"${difference_abs:,.2f}"
  if has_dollar_sign:
    difference_str = f"{difference_abs:,.2f}"
  
  # Calculate difference in counts
  count1 = len(df1)
  count2 = len(df2)
  count_difference = count1 - count2
  count_difference_abs = abs(count_difference)
  
  # Format category names for display (convert snake_case to Title Case)
  def format_category_name(cat: str) -> str:
    if '_' in cat:
      return ' '.join(word.capitalize() for word in cat.split('_'))
    return cat.capitalize()
  
  first_label = format_category_name(group1_name)
  second_label = format_category_name(group2_name)
  
  # Determine which category has more/less spending
  if total1 > total2:
    more_label = first_label
    more_amount_str = first_amount_str
    more_total = total1
    more_count = count1
    less_label = second_label
    less_amount_str = second_amount_str
    less_total = total2
    less_count = count2
  elif total2 > total1:
    more_label = second_label
    more_amount_str = second_amount_str
    more_total = total2
    more_count = count2
    less_label = first_label
    less_amount_str = first_amount_str
    less_total = total1
    less_count = count1
  else:
    # Equal amounts
    more_label = first_label
    more_amount_str = first_amount_str
    more_total = total1
    more_count = count1
    less_label = second_label
    less_amount_str = second_amount_str
    less_total = total2
    less_count = count2
  
  # Format the result string - pass all values as strings/numeric (no format specifiers)
  try:
    result = temp_template.format(
      first_amount=first_amount_str,
      second_amount=second_amount_str,
      first_label=first_label,
      second_label=second_label,
      difference=difference_str,
      first_total=total1,  # Numeric value
      second_total=total2,  # Numeric value
      first_count=count1,  # Numeric value
      second_count=count2,  # Numeric value
      count_difference=count_difference_abs,  # Numeric value (absolute difference)
      more_label=more_label,
      more_amount=more_amount_str,
      more_total=more_total,  # Numeric value
      more_count=more_count,  # Numeric value
      less_label=less_label,
      less_amount=less_amount_str,
      less_total=less_total,  # Numeric value
      less_count=less_count  # Numeric value
    )
  except (KeyError, ValueError) as e:
    # If template doesn't have all placeholders, try with minimal set
    try:
      result = temp_template.format(
        first_amount=first_amount_str,
        second_amount=second_amount_str,
        first_label=first_label,
        second_label=second_label
      )
    except (KeyError, ValueError) as e2:
      # If that also fails, use the original template with minimal formatting
      result = f"{first_label}: {first_amount_str} | {second_label}: {second_amount_str}"
  
  # Create metadata
  metadata = [{
    "comparison_type": "group" if 'group' in df.columns else "category",
    "first_group": {
      "name": group1_name,
      "label": first_label,
      "total": float(total1),
      "count": int(count1)
    },
    "second_group": {
      "name": group2_name,
      "label": second_label,
      "total": float(total2),
      "count": int(count2)
    },
    "difference": float(difference),
    "difference_abs": float(difference_abs),
    "count_difference": int(count_difference),
    "count_difference_abs": int(count_difference_abs)
  }]
  
  log(f"**Returning** comparison result and metadata.")
  log(f"**Result**: `{result}`")
  log(f"**Metadata**:\n```json\n{json.dumps(metadata, indent=2)}\n```")
  
  # Ensure we always return a tuple
  if result is None:
    result = f"{first_label}: {first_amount_str} | {second_label}: {second_amount_str}"
  if metadata is None:
    metadata = []
  
  return result, metadata


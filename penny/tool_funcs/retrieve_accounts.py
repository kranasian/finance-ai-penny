import json
import re
from database import Database
import pandas as pd
from penny.tool_funcs.sandbox_logging import log


ACCOUNT_TYPE_TO_STRING = {
  'deposit_savings': 'savings',
  'deposit_money_market': 'money market',
  'deposit_checking': 'checking',
  'credit_card': 'credit card',
  'loan_home_equity': 'home equity loan',
  'loan_line_of_credit': 'line of credit',
  'loan_mortgage': 'mortgage',
  'loan_auto': 'auto loan',
}
ACCOUNT_STRING_TO_TYPE = {v: k for k, v in ACCOUNT_TYPE_TO_STRING.items()}


def retrieve_accounts_function_code_gen(user_id: int = 1) -> pd.DataFrame:
  """Function to retrieve accounts from the database for a specific user"""
  db = Database()
  accounts = db.get_accounts_by_user(user_id=user_id)
  df = pd.DataFrame(accounts)
  
  if 'account_type' in df.columns and 'account_subtype' in df.columns:
    # Mapping dictionary: (account_type, account_subtype) -> standardized_account_type
    ACCOUNT_TYPE_MAPPING = {
      ('loan', 'home equity'): 'loan_home_equity',
      ('depository', 'savings'): 'deposit_savings',
      ('depository', 'depository'): 'deposit_savings',
      ('loan', 'line of credit'): 'loan_line_of_credit',
      ('credit', 'credit card'): 'credit_card',
      ('loan', 'mortgage'): 'loan_mortgage',
      ('depository', 'money market'): 'deposit_money_market',
      ('loan', 'loan'): 'loan_home_equity',
      ('loan', 'auto'): 'loan_auto',
      ('depository', 'checking'): 'deposit_checking',
    }
    
    def map_account_type(row):
      """Map account_type and account_subtype to standardized format"""
      key = (row['account_type'], row['account_subtype'])
      return ACCOUNT_TYPE_MAPPING.get(key, f"{row['account_type']}_{row['account_subtype']}")
    
    df['account_type'] = df.apply(map_account_type, axis=1)
  
  log(f"**Retrieved All Accounts** of `U-{user_id}`: `df: {df.shape}` w/ **cols**:\n  - `{'`, `'.join(df.columns)}`")
  return df


def account_names_and_balances(df: pd.DataFrame, template: str) -> tuple[str, list]:
  """Generate a formatted string describing account names and balances using the provided template and return metadata"""
  log(f"**Account Names/Balances**: `df: {df.shape}` w/ **cols**:\n  - `{'`, `'.join(df.columns)}`")
  
  if df.empty:
    log("- **`df` is empty**, returning empty string and metadata.")
    return "", []
  
  # Check if we have balance columns
  has_balance_available = 'balance_available' in df.columns
  has_balance_current = 'balance_current' in df.columns
  
  if not has_balance_available and not has_balance_current:
    log("- **`df` has no balance columns**.")
    raise ValueError(f"DataFrame must have at least one balance column. Available columns: {list(df.columns)}")
  
  # Determine if this is individual accounts or aggregated data
  has_account_details = 'account_name' in df.columns
  
  utterances = []
  metadata = []
  
  if has_account_details:
    # Individual accounts - format each account
    log(f"**Listing Individual Accounts**: Processing {len(df)} items.")
    for i in range(len(df)):
      account = df.iloc[i]
      account_name = account['account_name']
      account_type = account.get('account_type', None)
      account_type_string = ACCOUNT_TYPE_TO_STRING.get(account_type, None)
      account_mask = account.get('account_mask', None)
      account_id = account.get('account_id', None)
      balance_available = account.get('balance_available', None)
      balance_current = account.get('balance_current', None)
      balance_limit = account.get('balance_limit', None)
      
      # Handle None values for logging
      balance_available_log = "None" if balance_available is None else f"${balance_available:.0f}"
      balance_current_log = "None" if balance_current is None else f"${balance_current:.0f}"
      balance_limit_log = "None" if balance_limit is None else f"${balance_limit:.0f}"
      log(f"  - `A-{account_id}`]  **Name**: `{account_name}`  |  **A**: `{balance_available_log}`  |  **C**: `{balance_current_log}`  |  **L**: `{balance_limit_log}`")
      
      # Check if template has format specifiers - if so, use numeric values
      has_format_specifiers = bool(re.search(r'\{balance_(available|current|limit):[^}]+\}', template))
      has_dollar_sign = bool(re.search(r'\$\{balance_(available|current|limit):', template)) if has_format_specifiers else False
      # Check for dollar signs anywhere in template (for templates without format specifiers)
      has_dollar_sign_anywhere = '$' in template
      
      
      # Try formatting - handle both cases: with and without format specifiers
      try:
        if has_format_specifiers and has_dollar_sign:
          # Template has format specifiers WITH $ sign, pass raw numeric values
          balance_available_val = 0.0 if balance_available is None else balance_available
          balance_current_val = 0.0 if balance_current is None else balance_current
          balance_limit_val = 0.0 if balance_limit is None else balance_limit
          utterance = template.format(
            name=account_name,
            account_name=account_name,
            account_type=account_type_string,
            account_mask=account_mask,
            account_id=account_id,
            balance_available=balance_available_val,
            balance_current=balance_current_val,
            balance_limit=balance_limit_val
          )
        elif has_format_specifiers and not has_dollar_sign:
          # Template has format specifiers but NO $ sign, format with $ included
          available_balance_str = "Unknown" if balance_available is None else f"${balance_available:.0f}"
          current_balance_str = "Unknown" if balance_current is None else f"${balance_current:.0f}"
          limit_balance_str = "Unknown" if balance_limit is None else f"${balance_limit:.0f}"
          # Replace ALL format specifiers with simple placeholders for formatting
          temp_template = re.sub(r'\{balance_(available|current|limit):[^}]+\}', r'{\1}', template)
          temp_template = temp_template.replace('{available}', '{balance_available}')
          temp_template = temp_template.replace('{current}', '{balance_current}')
          temp_template = temp_template.replace('{limit}', '{balance_limit}')
          log(f"**Formatting with $ signs**: Template: `{template[:150]}...` -> Replaced: `{temp_template[:150]}...`")
          log(f"**Values**: available={available_balance_str}, current={current_balance_str}, limit={limit_balance_str}")
          # Use format_map to ensure proper substitution
          format_dict = {
            'name': account_name,
            'account_name': account_name,
            'account_type': account_type_string,
            'account_mask': account_mask,
            'account_id': account_id,
            'balance_available': available_balance_str,
            'balance_current': current_balance_str,
            'balance_limit': limit_balance_str
          }
          utterance = temp_template.format(**format_dict)
          # Post-process to ensure dollar signs are present for formatted numbers
          # This is a safety net in case format() does something unexpected
          if balance_current is not None:
            # Look for numbers like "2,500" without dollar sign and add it
            current_num = f"{balance_current:.0f}"
            if current_num in utterance and f"${current_num}" not in utterance:
              utterance = utterance.replace(f" {current_num}", f" ${current_num}", 1)
              log(f"**Fixed current balance**: Added $ to {current_num}")
          if balance_available is not None:
            available_num = f"{balance_available:.0f}"
            if available_num in utterance and f"${available_num}" not in utterance:
              utterance = utterance.replace(f" {available_num}", f" ${available_num}", 1)
              log(f"**Fixed available balance**: Added $ to {available_num}")
          if balance_limit is not None:
            limit_num = f"{balance_limit:.0f}"
            if limit_num in utterance and f"${limit_num}" not in utterance:
              utterance = utterance.replace(f" {limit_num}", f" ${limit_num}", 1)
              log(f"**Fixed limit balance**: Added $ to {limit_num}")
          
        else:
          # No format specifiers - check if template has dollar signs
          if has_dollar_sign_anywhere:
            # Template has dollar signs, format without $ prefix
            available_balance_str = "Unknown" if balance_available is None else f"{balance_available:.0f}"
            current_balance_str = "Unknown" if balance_current is None else f"{balance_current:.0f}"
            limit_balance_str = "Unknown" if balance_limit is None else f"{balance_limit:.0f}"
          else:
            # Template has NO dollar signs, format WITH $ prefix
            available_balance_str = "Unknown" if balance_available is None else f"${balance_available:.0f}"
            current_balance_str = "Unknown" if balance_current is None else f"${balance_current:.0f}"
            limit_balance_str = "Unknown" if balance_limit is None else f"${balance_limit:.0f}"
          utterance = template.format(
            name=account_name,
            account_name=account_name,
            account_type=account_type_string,
            account_mask=account_mask,
            account_id=account_id,
            balance_available=available_balance_str,
            balance_current=current_balance_str,
            balance_limit=limit_balance_str
          )
      except (ValueError, KeyError) as e:
        # If formatting with numeric values fails, try with formatted strings
        # Check if template has format specifiers - if so, replace them and add $ signs
        log(f"**Exception caught**: {e}, checking template for format specifiers")
        has_format_specifiers = bool(re.search(r'\{balance_(available|current|limit):[^}]+\}', template))
        if has_format_specifiers:
          # Format with $ signs and replace format specifiers
          available_balance_str = "Unknown" if balance_available is None else f"${balance_available:.0f}"
          current_balance_str = "Unknown" if balance_current is None else f"${balance_current:.0f}"
          limit_balance_str = "Unknown" if balance_limit is None else f"${balance_limit:.0f}"
          temp_template = re.sub(r'\{balance_(available|current|limit):[^}]+\}', r'{\1}', template)
          temp_template = temp_template.replace('{available}', '{balance_available}')
          temp_template = temp_template.replace('{current}', '{balance_current}')
          temp_template = temp_template.replace('{limit}', '{balance_limit}')
          log(f"**Exception handler**: Template: `{template[:150]}...` -> Replaced: `{temp_template[:150]}...`")
          utterance = temp_template.format(
            name=account_name,
            account_name=account_name,
            account_type=account_type_string,
            account_mask=account_mask,
            account_id=account_id,
            balance_available=available_balance_str,
            balance_current=current_balance_str,
            balance_limit=limit_balance_str
          )
        else:
          # No format specifiers - check if template has dollar signs
          has_dollar_sign_anywhere = '$' in template
          if has_dollar_sign_anywhere:
            # Template has dollar signs, format without $ prefix
            available_balance_str = "Unknown" if balance_available is None else f"{balance_available:.0f}"
            current_balance_str = "Unknown" if balance_current is None else f"{balance_current:.0f}"
            limit_balance_str = "Unknown" if balance_limit is None else f"{balance_limit:.0f}"
          else:
            # Template has NO dollar signs, format WITH $ prefix
            available_balance_str = "Unknown" if balance_available is None else f"${balance_available:.0f}"
            current_balance_str = "Unknown" if balance_current is None else f"${balance_current:.0f}"
            limit_balance_str = "Unknown" if balance_limit is None else f"${balance_limit:.0f}"
          utterance = template.format(
            name=account_name,
            account_name=account_name,
            account_type=account_type_string,
            account_mask=account_mask,
            account_id=account_id,
            balance_available=available_balance_str,
            balance_current=current_balance_str,
            balance_limit=limit_balance_str
          )
      utterances.append(utterance)
      
      metadata.append({
        "account_id": int(account_id),
        "account_name": account_name,
      })
  else:
    # Aggregated data - format the totals
    log(f"**Aggregated Totals**: Processing {len(df)} items.")
    total_available = df['balance_available'].sum() if has_balance_available else None
    total_current = df['balance_current'].sum() if has_balance_current else None
    
    log(f"Aggregated totals: total_available={total_available}, total_current={total_current}")
    
    # Format balances, using "Unknown" for None values
    available_balance_str = "Unknown" if total_available is None else f"${total_available:.0f}"
    current_balance_str = "Unknown" if total_current is None else f"${total_current:.0f}"
    
    # Use "Total" or similar as the name for aggregated results
    aggregate_name = "Total"
    utterance = template.format(
      name=aggregate_name,
      balance_available=available_balance_str,
      balance_current=current_balance_str
    )
    utterances.append(utterance)
    
    # For aggregated data, we may not have individual account metadata
    metadata.append({
      "aggregate": True,
      "total_available": total_available,
      "total_current": total_current,
    })
  
  log(f"**Returning** {len(utterances)} utterances and {len(metadata)} metadata entries.")
  log(f"**Utterances**:\n  - `{'`\n  - `'.join(utterances)}`")
  log(f"**Metadata**:\n```json\n{json.dumps(metadata, indent=2)}\n```")
  return "\n".join(utterances), metadata


def utter_account_totals(df: pd.DataFrame, template: str) -> str:
  """Calculate total balances and return formatted string"""
  log(f"**Account Totals**: `df: {df.shape}` w/ **cols**:\n  - `{'`, `'.join(df.columns)}`")
  
  if df.empty:
    log("- **`df` is empty**, returning empty string.")
    return ""
  
  # Check if required columns exist
  required_columns = ['balance_available', 'balance_current']
  missing_columns = [col for col in required_columns if col not in df.columns]
  if missing_columns:
    log(f"- **`df` is missing required columns**")
    raise ValueError(f"- **`df` is missing required columns**: `{', '.join(missing_columns)}`. Available columns: `{', '.join(df.columns)}`")
  
  # Calculate separate totals for savings and credit accounts if account_type column exists
  savings_available = None
  savings_current = None
  credit_available = None
  credit_current = None
  
  if 'account_type' in df.columns:
    # Define account types
    savings_types = ['deposit_savings', 'deposit_checking', 'deposit_money_market']
    credit_types = ['credit_card', 'loan_home_equity', 'loan_line_of_credit', 'loan_mortgage', 'loan_auto']
    
    # Filter savings accounts
    savings_df = df[df['account_type'].isin(savings_types)]
    if not savings_df.empty:
      savings_available = savings_df['balance_available'].sum()
      savings_current = savings_df['balance_current'].sum()
    
    # Filter credit accounts
    credit_df = df[df['account_type'].isin(credit_types)]
    if not credit_df.empty:
      credit_available = credit_df['balance_available'].sum()
      credit_current = credit_df['balance_current'].sum()
    
    savings_avail_str = f"{savings_available:.0f}" if savings_available is not None else "0"
    savings_curr_str = f"{savings_current:.0f}" if savings_current is not None else "0"
    credit_avail_str = f"{credit_available:.0f}" if credit_available is not None else "0"
    credit_curr_str = f"{credit_current:.0f}" if credit_current is not None else "0"
    log(f"**Separated Totals**: STA: `${savings_avail_str}` | STC: `${savings_curr_str}` | CTA: `${credit_avail_str}` | CTC: `${credit_curr_str}`")
  
  # Check if template uses separated totals placeholders
  uses_separated_totals = bool(re.search(r"\{savings_balance_|credit_balance_", template))
  
  # Check if template uses overall totals (balance_available or balance_current without savings_/credit_ prefix)
  # Match {balance_available} or {balance_current} but not {savings_balance_available} or {credit_balance_available}
  uses_overall_totals = bool(re.search(r"\{balance_(available|current)\}", template)) and not bool(re.search(r"\{savings_balance_|credit_balance_", template))
  
  # If we have separated totals and template doesn't use them, automatically format with separated totals
  has_both_account_types = (savings_available is not None or savings_current is not None) and (credit_available is not None or credit_current is not None)
  
  if has_both_account_types and not uses_separated_totals:
    # Automatically format with separated totals
    has_dollar_sign_anywhere = '$' in template
    if has_dollar_sign_anywhere:
      savings_curr_str = f"{savings_current:.0f}" if savings_current is not None else "0"
      credit_curr_str = f"{credit_current:.0f}" if credit_current is not None else "0"
      savings_avail_str = f"{savings_available:.0f}" if savings_available is not None else "0"
      credit_avail_str = f"{credit_available:.0f}" if credit_available is not None else "0"
    else:
      savings_curr_str = f"${savings_current:.0f}" if savings_current is not None else "$0"
      credit_curr_str = f"${credit_current:.0f}" if credit_current is not None else "$0"
      savings_avail_str = f"${savings_available:.0f}" if savings_available is not None else "$0"
      credit_avail_str = f"${credit_available:.0f}" if credit_available is not None else "$0"
    
    result = f"Your savings accounts (checking, savings, money market) have a total current balance of {savings_curr_str} and available balance of {savings_avail_str}. Your credit accounts (credit cards, loans) have a total current balance of {credit_curr_str} and available balance of {credit_avail_str}."
    log(f"**Auto-separated Utterance**: `{result}`")
    return result
  
  # Detect if template uses format specifiers like {balance_available:.0f} or {savings_balance_available:.0f}
  has_format_specifiers = bool(re.search(r"\{(balance_available|balance_current|savings_balance_available|savings_balance_current|credit_balance_available|credit_balance_current):[^}]+\}", template))
  has_dollar_sign_anywhere = '$' in template
  
  # Calculate overall totals if template uses them (either with or without format specifiers)
  total_available = None
  total_current = None
  if uses_overall_totals or (has_format_specifiers and bool(re.search(r"\{balance_(available|current)", template))):
    total_available = df['balance_available'].sum()
    total_current = df['balance_current'].sum()
    log(f"**Calculated Totals**: **TA**: `${total_available:.0f}`  |  **TC**: `${total_current:.0f}`")
  
  if has_format_specifiers:
    # Pass raw numbers so the template's specifiers can format them
    format_dict = {
      'balance_available': 0.0 if total_available is None else total_available,
      'balance_current': 0.0 if total_current is None else total_current,
      'savings_balance_available': 0.0 if savings_available is None else savings_available,
      'savings_balance_current': 0.0 if savings_current is None else savings_current,
      'credit_balance_available': 0.0 if credit_available is None else credit_available,
      'credit_balance_current': 0.0 if credit_current is None else credit_current,
    }
    result = template.format(**format_dict)
  else:
    # No format specifiers - check if template has dollar signs
    if has_dollar_sign_anywhere:
      # Template has dollar signs, format without $ prefix
      available_str = "Unknown" if total_available is None else f"{total_available:.0f}"
      current_str = "Unknown" if total_current is None else f"{total_current:.0f}"
      savings_available_str = "Unknown" if savings_available is None else f"{savings_available:.0f}"
      savings_current_str = "Unknown" if savings_current is None else f"{savings_current:.0f}"
      credit_available_str = "Unknown" if credit_available is None else f"{credit_available:.0f}"
      credit_current_str = "Unknown" if credit_current is None else f"{credit_current:.0f}"
    else:
      # Template has NO dollar signs, format WITH $ prefix
      available_str = "Unknown" if total_available is None else f"${total_available:.0f}"
      current_str = "Unknown" if total_current is None else f"${total_current:.0f}"
      savings_available_str = "Unknown" if savings_available is None else f"${savings_available:.0f}"
      savings_current_str = "Unknown" if savings_current is None else f"${savings_current:.0f}"
      credit_available_str = "Unknown" if credit_available is None else f"${credit_available:.0f}"
      credit_current_str = "Unknown" if credit_current is None else f"${credit_current:.0f}"
    
    format_dict = {
      'balance_available': available_str,
      'balance_current': current_str,
      'savings_balance_available': savings_available_str,
      'savings_balance_current': savings_current_str,
      'credit_balance_available': credit_available_str,
      'credit_balance_current': credit_current_str,
    }
    result = template.format(**format_dict)
  log(f"**Utterance**: `{result}`")
  return result

# Note: handle_function_call is now centralized in tools/tool_defs.py
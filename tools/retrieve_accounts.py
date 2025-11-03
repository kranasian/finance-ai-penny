import json
from database import Database
import pandas as pd
from sandbox_logging import log


ACCOUNT_TYPE_TO_STRING = {
  'deposit_savings': 'savings',
  'deposit_money_market': 'money market',
  'deposit_checking': 'checking',
  'credit_card': 'credit card',
  'loan_home_equity': 'home equity loan',
  'loan_line_of_credit': 'line of credit',
  'loan_mortgage': 'mortgage',
}
ACCOUNT_STRING_TO_TYPE = {v: k for k, v in ACCOUNT_TYPE_TO_STRING.items()}


def retrieve_accounts_function_code_gen(user_id: int = 1) -> pd.DataFrame:
  """Function to retrieve accounts from the database for a specific user"""
  db = Database()
  accounts = db.get_accounts_by_user(user_id=user_id)
  df = pd.DataFrame(accounts)
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
      
      log(f"  - `A-{account_id}`]  **Name**: `{account_name}`  |  **A**: `${balance_available:,.0f}`  |  **C**: `${balance_current:,.0f}`")
      
      # Format balances, using "Unknown" for None values
      available_balance_str = "Unknown" if balance_available is None else f"${balance_available:,.0f}"
      current_balance_str = "Unknown" if balance_current is None else f"${balance_current:,.0f}"
      
      utterance = template.format(
        account_name=account_name,
        account_type=account_type_string,
        account_mask=account_mask,
        account_id=account_id,
        balance_available=available_balance_str,
        balance_current=current_balance_str
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
    available_balance_str = "Unknown" if total_available is None else f"${total_available:,.0f}"
    current_balance_str = "Unknown" if total_current is None else f"${total_current:,.0f}"
    
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
  
  total_available = df['balance_available'].sum()
  total_current = df['balance_current'].sum()
  log(f"**Calculated Totals**: **TA**: `${total_available:,.0f}`  |  **TC**: `${total_current:,.0f}`")
  
  result = template.format(
    balance_available=f"${total_available:,.0f}",
    balance_current=f"${total_current:,.0f}"
  )
  log(f"**Utterance**: `{result}`")
  return result

# Note: handle_function_call is now centralized in tools/tool_defs.py
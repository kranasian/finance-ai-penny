from database import Database
import pandas as pd


def retrieve_accounts_function_code_gen(user_id: int = 1) -> pd.DataFrame:
  """Function to retrieve accounts from the database for a specific user"""
  db = Database()
  df = pd.DataFrame(db.get_accounts_by_user(user_id=user_id))
  return df


def account_names_and_balances(df: pd.DataFrame, template: str) -> tuple[str, list]:
  """Generate a formatted string describing account names and balances using the provided template and return metadata"""
  if df.empty:
    return "", []
  
  utterances = []
  metadata = []
  
  for i in range(len(df)):
    account = df.iloc[i]
    utterance = template.format(
      name=account['account_name'],
      available_balance=f"${account['balance_available']:,.0f}",
      current_balance=f"${account['balance_current']:,.0f}"
    )
    utterances.append(utterance)
    
    metadata.append({
      "account_id": account['account_id'],
      "account_name": account['account_name'],
    })
  
  return "\n".join(utterances), metadata


def utter_account_totals(df: pd.DataFrame, template: str) -> str:
  """Calculate total balances and return formatted string"""
  if df.empty:
    return ""
  
  total_available = sum(account['balance_available'] for account in df.iloc)
  total_current = sum(account['balance_current'] for account in df.iloc)
  
  return template.format(
    available_balance=f"${total_available:,.0f}",
    current_balance=f"${total_current:,.0f}"
  )

# Note: handle_function_call is now centralized in tools/tool_defs.py
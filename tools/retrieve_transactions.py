from database import Database
import pandas as pd
from sandbox_logging import log


def retrieve_transactions_function_code_gen(user_id: int = 1) -> pd.DataFrame:
  """Function to retrieve transactions from the database for a specific user"""
  log(f"retrieve_transactions_function_code_gen called with user_id={user_id}")
  db = Database()
  transactions = db.get_transactions_by_user(user_id=user_id)
  log(f"Retrieved {len(transactions)} transactions from database for user_id={user_id}")
  df = pd.DataFrame(transactions)
  log(f"Created DataFrame with shape {df.shape} and columns: {list(df.columns)}")
  return df


def transaction_names_and_amounts(df: pd.DataFrame, template: str) -> tuple[str, list]:
  """Generate a formatted string describing transaction names and amounts using the provided template and return metadata"""
  log(f"transaction_names_and_amounts called with df.shape={df.shape}, columns={list(df.columns)}")
  
  if df.empty:
    log("DataFrame is empty, returning empty string and metadata")
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
  
  log(f"Processing {len(df)} transactions")
  for i in range(len(df)):
    transaction = df.iloc[i]
    transaction_name = transaction.get('transaction_name', 'Unknown Transaction')
    amount = transaction.get('amount', 0.0)
    date = transaction.get('date', 'Unknown Date')
    category = transaction.get('category', 'Unknown Category')
    transaction_id = transaction.get('transaction_id', None)
    
    log(f"Processing transaction {i+1}/{len(df)}: transaction_id={transaction_id}, transaction_name={transaction_name}, "
        f"amount={amount}, date={date}, category={category}")
    
    utterance = template.format(
      name=transaction_name,
      amount=f"${abs(amount):,.2f}",
      date=date,
      category=category
    )
    utterances.append(utterance)
    
    metadata.append({
      "transaction_id": transaction_id,
      "transaction_name": transaction_name,
      "amount": amount,
      "date": date,
      "category": category
    })
  
  log(f"Returning {len(utterances)} utterances and {len(metadata)} metadata entries")
  return "\n".join(utterances), metadata


def utter_transaction_totals(df: pd.DataFrame, template: str) -> str:
  """Calculate total transaction amounts and return formatted string"""
  log(f"utter_transaction_totals called with df.shape={df.shape}, columns={list(df.columns)}")
  
  if df.empty:
    log("DataFrame is empty, returning empty string")
    return ""
  
  # Check if required columns exist
  required_columns = ['amount']
  missing_columns = [col for col in required_columns if col not in df.columns]
  if missing_columns:
    error_msg = f"DataFrame is missing required columns: {missing_columns}. Available columns: {list(df.columns)}"
    log(error_msg)
    raise ValueError(error_msg)
  
  total_amount = df['amount'].sum()
  log(f"Calculated total: total_amount={total_amount}")
  
  result = template.format(
    total_amount=f"${abs(total_amount):,.2f}"
  )
  log(f"Formatted result: {result}")
  return result


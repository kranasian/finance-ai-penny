from typing import Tuple
from AccessControl.ZopeGuards import guarded_filter, guarded_reduce, guarded_max, guarded_min, guarded_map, guarded_zip, guarded_getitem, guarded_hasattr
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from RestrictedPython import compile_restricted
from RestrictedPython.Guards import safe_builtins, safe_globals, full_write_guard, guarded_iter_unpack_sequence, guarded_unpack_sequence
from RestrictedPython.Limits import limited_builtins
from RestrictedPython.Utilities import utility_builtins
from RestrictedPython import PrintCollector
import datetime as dt
import dateutil
import pandas as pd
import traceback
import json
from tools.retrieve_accounts import retrieve_accounts_function_code_gen, account_names_and_balances, utter_account_totals
from tools.retrieve_transactions import retrieve_transactions_function_code_gen, transaction_names_and_amounts, utter_transaction_totals
from tools.retrieve_forecasts import retrieve_spending_forecasts_function_code_gen, retrieve_income_forecasts_function_code_gen
from tools.retrieve_subscriptions import retrieve_subscriptions_function_code_gen, subscription_names_and_amounts, utter_subscription_totals
from tools.forecast_utils import forecast_dates_and_amount, utter_forecasts
from tools.compare_spending import compare_spending
from tools.respond_to_app_inquiry import respond_to_app_inquiry
from tools.create_goal import create_goal
from tools.date_utils import (
    get_today_date,
    get_date,
    get_start_of_month,
    get_end_of_month,
    get_start_of_year,
    get_end_of_year,
    get_start_of_week,
    get_end_of_week,
    get_after_periods,
    get_date_string
)
from sandbox_logging import log as sandbox_log, clear_logs as clear_sandbox_logs, get_logs_as_string


def _write_(obj):
  """Guard for writing"""
  return obj


def _is_json_serializable(obj):
  """Check if an object is JSON serializable"""
  try:
    json.dumps(obj)
    return True
  except (TypeError, ValueError):
    return False

def _getitem_(obj, key):
  """Custom guard for item getting that allows pandas objects"""
  try:
    # Allow pandas objects (DataFrame, Series, etc.)
    if hasattr(obj, '__class__') and obj.__class__.__module__.startswith('pandas'):
      # For pandas objects, let them handle their own errors naturally
      return obj[key]
    
    # Handle callable objects (functions, etc.)
    if callable(obj):
      # Special handling for type constructors that can be subscripted directly
      # These are: list, dict, tuple, str, bytes, range, type
      type_constructors = {list, dict, tuple, str, bytes, range, type}
      if obj in type_constructors:
        # These can be subscripted directly (e.g., list[int], dict[str, int])
        return obj[key]
      
      # For other functions that return subscriptable objects, call them first
      try:
        result = obj()
        if hasattr(result, '__getitem__'):
          return result[key]
      except Exception:
        pass
      # If calling the function doesn't work, raise a clear error
      func_name = getattr(obj, '__name__', 'unknown')
      raise TypeError(f"'_getitem_ guard: Attempted to subscript function object {type(obj).__name__} with key {key}. Function name: {func_name}")
    
    # For other objects, try to use subscript notation
    return obj[key]
  except (TypeError, KeyError, IndexError) as e:
    # For pandas objects, let the original error propagate naturally
    if hasattr(obj, '__class__') and obj.__class__.__module__.startswith('pandas'):
      raise e
    # For other objects, provide more context
    raise TypeError(f"_getitem_ guard: Failed to access {type(obj).__name__} with key {key}. Object: {obj}, Error: {e}") from e
  except Exception as e:
    raise TypeError(f"_getitem_ guard: Unexpected error accessing {type(obj).__name__} with key {key}. Object: {obj}, Error: {e}") from e

def _hasattr_(obj, name):
  """Custom guard for hasattr that allows pandas objects"""
  try:
    # Allow pandas objects (DataFrame, Series, etc.)
    if hasattr(obj, '__class__') and obj.__class__.__module__.startswith('pandas'):
      return hasattr(obj, name)
    # For function objects and other objects, use standard hasattr
    return hasattr(obj, name)
  except Exception as e:
    return False

class _DataFrameGuard:
  """Guard class for DataFrame operations"""
  @staticmethod
  def _getattr_(obj, name):
    """Guard for attribute access"""
    try:
      # Allow pandas objects (DataFrame, Series, etc.)
      if hasattr(obj, '__class__') and obj.__class__.__module__.startswith('pandas'):
        return getattr(obj, name)
      # Allow access to PrintHandler's _call_print method
      if obj.__class__.__name__ == 'PrintHandler' and name == '_call_print':
        return getattr(obj, name)
      # Special handling for datetime - check if obj is the datetime module or class
      if obj is dt or obj is datetime:
        # If obj is the datetime module (dt), allow access to its attributes (including datetime class)
        if obj is dt:
          return getattr(dt, name)
        # If obj is the datetime class, allow access to its methods
        # If accessing .datetime on the class, return the class itself (for compatibility with datetime.datetime.now())
        if name == 'datetime':
          return datetime
        # If accessing .timedelta on the class, return timedelta (from datetime module)
        if name == 'timedelta':
          return timedelta
        try:
          return getattr(datetime, name)
        except AttributeError:
          pass
      # Check if obj is a datetime class type (but not the module)
      if type(obj).__name__ == 'type' and hasattr(obj, '__name__') and obj.__name__ == 'datetime' and obj is not dt:
        if name == 'datetime':
          return datetime
        try:
          return getattr(datetime, name)
        except AttributeError:
          pass
      # Handle method_descriptor objects that might be datetime-related
      if type(obj).__name__ == 'method_descriptor' and name in ('today', 'now', 'utcnow', 'fromtimestamp', 'fromordinal'):
        # Try to get the attribute from datetime class
        try:
          return getattr(datetime, name)
        except AttributeError:
          pass
      # For other objects, use a more restrictive approach
      if name.startswith('_'):
        raise AttributeError(f"_getattr_ guard: Access to private attribute '{name}' on {type(obj).__name__} is not allowed")
      return getattr(obj, name)
    except AttributeError as e:
      # If it's a datetime-related attribute error, try to get it from datetime class
      if name in ('today', 'now', 'utcnow', 'fromtimestamp', 'fromordinal'):
        try:
          return getattr(datetime, name)
        except AttributeError:
          pass
      raise AttributeError(f"_getattr_ guard: Failed to access attribute '{name}' on {type(obj).__name__}. Error: {e}") from e
    except Exception as e:
      raise AttributeError(f"_getattr_ guard: Failed to access attribute '{name}' on {type(obj).__name__}. Error: {e}") from e
  
  @staticmethod
  def _getiter_(obj):
    """Guard for iteration"""
    try:
      # Allow pandas objects (DataFrame, Series, etc.)
      if hasattr(obj, '__class__') and obj.__class__.__module__.startswith('pandas'):
        return iter(obj)
      # For other objects, use a more restrictive approach
      if hasattr(obj, '__iter__'):
        return iter(obj)
      raise TypeError(f"_getiter_ guard: Object of type {type(obj)} is not iterable")
    except Exception as e:
      raise TypeError(f"_getiter_ guard: Failed to iterate over {type(obj).__name__}. Error: {e}") from e
  
  ALLOWED_MODULES = {
    "datetime",
    "math",
    "time",
    "pandas",
    "pd",
    "dateutil",
    "relativedelta",
    "dateutil.relativedelta",
    "_strptime",
    "_datetime"
  }
  @staticmethod
  def restricted_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name in _DataFrameGuard.ALLOWED_MODULES:
        return __import__(name, globals, locals, fromlist, level)
    raise ImportError(f"Module '{name}' is not allowed")
  
  @staticmethod
  def _inplacevar_(op, left, right):
    """Handles in-place operations like +=, -=, *= within RestrictedPython."""
    if op[0] == '+':
      return left + right
    elif op[0] == '-':
      return left - right
    elif op[0] == '*':
      return left * right
    elif op[0] == '/':
      return left / right
    elif op[0] == '%':
      return left % right
    elif op[0] == '&':
      return left & right
    elif op[0] == '|':
      return left | right
    elif op[0] == '^':
      return left ^ right
    elif op == '//=':
      return left // right
    elif op == '**=':
      return left ** right
    elif op == '<<=':
      return left << right
    elif op == '>>=':
      return left >> right
    raise Exception(f"InPlaceVar Failure: {left} {op} {right}")


def _get_safe_globals(user_id,use_full_datetime=False):
  """Create a safe globals dictionary with limited functionality"""
  all_builtins = safe_builtins.copy()
  all_builtins.update(safe_globals)
  all_builtins.update(utility_builtins)
  all_builtins.update(limited_builtins)
  all_builtins["__import__"] = _DataFrameGuard.restricted_import
  
  # Add additional safe built-ins that are commonly needed
  additional_builtins = {
    "sum": sum,
    "len": len,
    "range": range,
    "enumerate": enumerate,
    "sorted": sorted,
    "reversed": reversed,
    "any": any,
    "all": all,
    "abs": abs,
    "round": round,
    "pow": pow,
    "divmod": divmod,
    "isinstance": isinstance,
    "type": type,
    "str": str,
    "int": int,
    "float": float,
    "bool": bool,
    "list": list,
    "tuple": tuple,
    "set": set,
    "dict": dict,
  }
  

  all_builtins.update(additional_builtins)
  
  # Create a wrapper function for retrieve_accounts that uses the provided user_id
  def retrieve_accounts_wrapper():
    return retrieve_accounts(user_id)
  
  # Create wrapper functions for account utility functions
  def account_names_and_balances_wrapper(df: pd.DataFrame, template: str):
    return account_names_and_balances(df, template)
  
  def utter_account_totals_wrapper(df: pd.DataFrame, template: str):
    return utter_account_totals(df, template)
  
  # Create a wrapper function for retrieve_transactions that uses the provided user_id
  def retrieve_transactions_wrapper():
    return retrieve_transactions(user_id)
  
  # Create wrapper functions for forecast retrieval
  def retrieve_spending_forecasts_wrapper(granularity: str = 'monthly'):
    return retrieve_spending_forecasts(user_id, granularity)
  
  def retrieve_income_forecasts_wrapper(granularity: str = 'monthly'):
    return retrieve_income_forecasts(user_id, granularity)
  
  # Create wrapper function for subscriptions
  def retrieve_subscriptions_wrapper():
    return retrieve_subscriptions(user_id)
  
  # Create wrapper functions for subscription utility functions
  def subscription_names_and_amounts_wrapper(df: pd.DataFrame, template: str):
    return subscription_names_and_amounts(df, template)
  
  def utter_subscription_totals_wrapper(df: pd.DataFrame, template: str):
    return utter_subscription_totals(df, template)
  
  # Create wrapper functions for transaction utility functions
  def transaction_names_and_amounts_wrapper(df: pd.DataFrame, template: str):
    return transaction_names_and_amounts(df, template)
  
  def utter_transaction_totals_wrapper(df: pd.DataFrame, is_spending: bool, template: str):
    return utter_transaction_totals(df, is_spending, template)
  
  def utter_forecasts_wrapper(df: pd.DataFrame, template: str):
    return utter_forecasts(df, template)
  
  def forecast_dates_and_amount_wrapper(df: pd.DataFrame, template: str):
    return forecast_dates_and_amount(df, template)
  
  def compare_spending_wrapper(df: pd.DataFrame, template: str, metadata: dict = None):
    return compare_spending(df, template, metadata)
  
  def respond_to_app_inquiry_wrapper(inquiry: str):
    return respond_to_app_inquiry(inquiry)
  
  def create_goal_wrapper(goals: list[dict]):
    return create_goal(goals)
  
  safe_globals_dict = {
    "__builtins__": all_builtins,
    "pd": pd,
    "pandas": pd,
    "dateutil": dateutil,
    "datetime": dt if use_full_datetime else datetime,
    "relativedelta": relativedelta,
    "_getitem_": _getitem_,
    "_hasattr_": _hasattr_,
    "_write_": _write_,
    "_iter_unpack_sequence_": guarded_iter_unpack_sequence,
    "_unpack_sequence_": guarded_unpack_sequence,
    "_getattr_": _DataFrameGuard._getattr_,
    "_getiter_": _DataFrameGuard._getiter_,
    "_inplacevar_": _DataFrameGuard._inplacevar_,
    "_print_": get_print_collector,
    "filter": guarded_filter,
    "reduce": guarded_reduce,
    "max": guarded_max,
    "min": guarded_min,
    "map": guarded_map,
    "timedelta": timedelta,
    "zip": guarded_zip,
    "retrieve_accounts": retrieve_accounts_wrapper,
    "account_names_and_balances": account_names_and_balances_wrapper,
    "utter_account_totals": utter_account_totals_wrapper,
    "retrieve_transactions": retrieve_transactions_wrapper,
    "retrieve_spending_forecasts": retrieve_spending_forecasts_wrapper,
    "retrieve_income_forecasts": retrieve_income_forecasts_wrapper,
    "retrieve_subscriptions": retrieve_subscriptions_wrapper,
    "subscription_names_and_amounts": subscription_names_and_amounts_wrapper,
    "utter_subscription_totals": utter_subscription_totals_wrapper,
    "transaction_names_and_amounts": transaction_names_and_amounts_wrapper,
    "utter_transaction_totals": utter_transaction_totals_wrapper,
    "utter_forecasts": utter_forecasts_wrapper,
    "forecast_dates_and_amount": forecast_dates_and_amount_wrapper,
    "compare_spending": compare_spending_wrapper,
    "respond_to_app_inquiry": respond_to_app_inquiry_wrapper,
    "create_goal": create_goal_wrapper,
    "utter_delta_from_now": utter_delta_from_now,
    "reminder_data": reminder_data,
    "log": sandbox_log,
    "get_today_date": get_today_date,
    "get_date": get_date,
    "get_start_of_month": get_start_of_month,
    "get_end_of_month": get_end_of_month,
    "get_start_of_year": get_start_of_year,
    "get_end_of_year": get_end_of_year,
    "get_start_of_week": get_start_of_week,
    "get_end_of_week": get_end_of_week,
    "get_after_periods": get_after_periods,
    "get_date_string": get_date_string,
  }
  return safe_globals_dict

def _check_code_for_full_datetime(code_str: str) -> bool:
  """Check if the code contains datetime.timedelta"""
  return " datetime.date(" in code_str or " datetime.time(" in code_str

# Global PrintCollector instance for capturing print outputs
_print_collector = None

def get_print_collector(_getattr_=None):
  """Get or create the global PrintCollector instance"""
  global _print_collector
  if _print_collector is None:
    _print_collector = PrintCollector(_getattr_=_getattr_ or _DataFrameGuard._getattr_)
  return _print_collector

def get_captured_print_output():
  """Get the accumulated print output from the PrintCollector"""
  global _print_collector
  if _print_collector is not None:
    return _print_collector()
  return ""

def clear_captured_print_output():
  """Clear the accumulated print output by creating a new PrintCollector"""
  global _print_collector
  _print_collector = PrintCollector(_getattr_=_DataFrameGuard._getattr_)


def retrieve_accounts(user_id: int = 1):
  """Internal function to retrieve accounts - available to executed code"""
  return retrieve_accounts_function_code_gen(user_id)


def retrieve_transactions(user_id: int = 1):
  """Internal function to retrieve transactions - available to executed code"""
  return retrieve_transactions_function_code_gen(user_id)


def retrieve_spending_forecasts(user_id: int = 1, granularity: str = 'monthly'):
  """Internal function to retrieve spending forecasts - available to executed code"""
  return retrieve_spending_forecasts_function_code_gen(user_id, granularity)


def retrieve_income_forecasts(user_id: int = 1, granularity: str = 'monthly'):
  """Internal function to retrieve income forecasts - available to executed code"""
  return retrieve_income_forecasts_function_code_gen(user_id, granularity)


def retrieve_subscriptions(user_id: int = 1):
  """Internal function to retrieve subscriptions - available to executed code"""
  return retrieve_subscriptions_function_code_gen(user_id)


def _create_restricted_process_input(code_str: str, user_id: int = 1) -> callable:
  """Compile and create a restricted function from a string"""    
  # Compile the code with restrictions
  byte_code = compile_restricted(
    code_str,
    filename="<inline>",
    mode="exec"
  )
  # Create namespace for execution
  safe_locals = {}
  safe_globals = _get_safe_globals(user_id=user_id, use_full_datetime=_check_code_for_full_datetime(code_str))
  # Execute the compiled code in restricted environment
  exec(byte_code, safe_globals, safe_locals)
  # Return the compiled function
  return safe_locals["process_input"]


def _run_sandbox_process_input(code_str: str, user_id: int) -> tuple[bool, str, dict | None, str]:
  """
  Run the provided code in a restricted sandbox environment
  Returns (success, captured_output, metadata, logs)
  """
  # Clear any previous captured print output and logs
  clear_captured_print_output()
  clear_sandbox_logs()
  
  success = None
  captured_logs = ""
  try:
    # Create the restricted function
    restricted_func = _create_restricted_process_input(code_str, user_id)
  except Exception as e:
    captured_output = f"**Compilation Error**: `{str(e)}`\n{traceback.format_exc()}"
    success = False
    metadata = {"error": traceback.format_exc()}
    
  # If the function was created successfully, run it
  if success is None:
    # Run the function - if it fails, capture logs before exception propagates
    try:
      success, metadata = restricted_func()
      captured_output = get_captured_print_output()
    except Exception as e:
      captured_output = f"**Execution Error**: `{str(e)}`\n{traceback.format_exc()}"
      success = False
      metadata = {"error": traceback.format_exc()}
  
    # Get the captured print output and logs
    captured_logs = get_logs_as_string()
  
  # Validate result is a bool
  if not isinstance(success, bool):
    raise ValueError(f"Restricted code must return a bool. Type: {type(success)}")
  
  if not (isinstance(metadata, dict) or metadata is None):
    if isinstance(metadata, list):
      metadata = {
        # TODO: In case the wrapping is wrong, sometimes its a list of things and no outside wrapping.
      }
    else:
      raise ValueError(f"Restricted code must return a dict. Type: {type(metadata)}")
  
  # Filter out non-serializable objects from metadata
  if metadata is not None:
    serializable_metadata = {}
    removed_keys = []
    
    for key, value in metadata.items():
      if _is_json_serializable(value):
        serializable_metadata[key] = value
      else:
        removed_keys.append(key)
        print(f"Removed non-serializable key '{key}' (type: {type(value).__name__})")
    
    if removed_keys:
      print(f"Removed {len(removed_keys)} non-serializable keys: {removed_keys}")
    
    metadata = serializable_metadata
  
  return success, captured_output, metadata, captured_logs

# Function to process DataFrame
def execute_agent_with_tools(code_str: str, user_id: int) -> Tuple[bool, str, dict | None, str]:
  """
  Extract Python code from generated response and execute it in restricted Python sandbox
  Returns: (success, utter, metadata, logs)
  """
  # Extract Python code from the response (look for ```python blocks)
  code_start = code_str.find("```python")
  if code_start != -1:
    code_start += len("```python")
    code_end = code_str.find("```", code_start)
    if code_end != -1:
      sandboxed_code = code_str[code_start:code_end].strip()
    else:
      # No closing ``` found, use the entire response as code
      sandboxed_code = code_str[code_start:].strip()
  else:
    # No ```python found, try to use the entire response as code
    sandboxed_code = code_str.strip()
  
  # Preprocess the code to replace _print_ with print (if needed)
  sandboxed_code = sandboxed_code.replace('_print_', 'print')
  
  return _run_sandbox_process_input(sandboxed_code, user_id)


# Helper functions for agent code
def utter_delta_from_now(future_time: datetime) -> str:
  """Utter the time delta in a human-readable format"""
  time_delta = future_time - datetime.now()
  # support singular and plural for days, hours, and minutes
  if time_delta.days == 1:
    return f"a day"
  elif time_delta.days > 1:
    return f"{time_delta.days} days"
  elif time_delta.total_seconds() >= 3600:  # 1 hour or more
    hours = int(time_delta.total_seconds() // 3600)
    if hours == 1:
      return f"an hour"
    else:
      return f"{hours} hours"
  elif time_delta.total_seconds() >= 60:  # 1 minute or more
    minutes = int(time_delta.total_seconds() // 60)
    if minutes == 1:
      return f"a minute"
    else:
      return f"{minutes} minutes"
  else:
    return f"soon"


def reminder_data(reminder) -> dict:
  # Handle both dict and pandas Series
  if hasattr(reminder, 'to_dict'):
    # It's a pandas Series
    reminder_dict = reminder.to_dict()
  else:
    # It's already a dict
    reminder_dict = reminder
  
  return {
    "id": reminder_dict.get('id', None),
    "title": reminder_dict.get('title', None),
    "reminder_datetime": reminder_dict.get('reminder_datetime', reminder_dict.get('reminder_time'))
  }
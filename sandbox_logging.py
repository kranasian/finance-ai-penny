"""
Sandbox logging module for capturing logs during restricted code execution.
This module provides a log() function that can be used within sandboxed code.
"""

from typing import List
import threading

# Thread-local storage for log messages
_local_storage = threading.local()


def _get_log_list() -> List[str]:
  """Get or create the thread-local log list"""
  if not hasattr(_local_storage, 'logs'):
    _local_storage.logs = []
  return _local_storage.logs


def log(message: str):
  """
  Log a message that will be captured and returned separately from sandbox execution.
  
  Args:
    message: The log message to capture
  """
  _get_log_list().append(message)


def get_logs() -> List[str]:
  """Get all captured logs for the current thread"""
  return _get_log_list().copy()


def clear_logs():
  """Clear all captured logs for the current thread"""
  if hasattr(_local_storage, 'logs'):
    _local_storage.logs.clear()


def get_logs_as_string() -> str:
  """Get all captured logs as a single string, one log per line"""
  return "\n\n".join(get_logs())


def get_logs_count() -> int:
  """Get the count of captured logs"""
  return len(_get_log_list())


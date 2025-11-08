"""Utility functions for tool_funcs"""
import re


def convert_brackets_to_braces(template: str) -> str:
  """Convert bracket placeholders [placeholder] to brace placeholders {placeholder} for Python format() compatibility"""
  # Convert [placeholder] to {placeholder}
  # Handle format specifiers like [amount:,.2f] -> {amount:,.2f}
  return re.sub(r'\[([^\]]+)\]', r'{\1}', template)


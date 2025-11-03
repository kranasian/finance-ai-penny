import re
import streamlit as st


def decode_escaped_string(text):
  """Convert escaped strings like '\\n' to actual newlines"""
  if not text:
    return text
  # Replace escaped newlines, tabs, etc.
  return (text
    .replace('\\n', '\n')
    .replace('\\t', '\t')
    .replace('\\r', '\r')
    .replace('\\"', '"')
    .replace("\\'", "'"))


def parse_tuple_error(error_text):
  """Parse tuple-formatted errors like: ('Sandbox execution failed: ...', KeyError('account_name'))"""
  error_str = str(error_text)
  
  # Look for the tuple ending pattern: , ErrorType('value'))
  # This is more reliable than trying to match the entire tuple
  tuple_end_pattern = r",\s*(\w+)\(['\"]([^'\"]*)['\"]\)\)\s*$"
  match = re.search(tuple_end_pattern, error_str)
  if match:
    error_type = match.group(1)
    error_value = match.group(2)
    # Extract the message part (everything before the comma)
    # Find the last comma before the error type
    comma_pos = error_str.rfind(',')
    if comma_pos > 0:
      message = error_str[:comma_pos].strip()
      # Remove leading ( if present
      if message.startswith('('):
        message = message[1:].strip()
      # Remove surrounding quotes if present
      message = message.strip().strip("'\"")
      return message, error_type, error_value
  
  return None, None, None


def extract_traceback(text):
  """Extract traceback from error text, handling escaped newlines"""
  if not text:
    return None
  
  # First decode escaped sequences
  decoded = decode_escaped_string(str(text))
  
  # Look for "Traceback:" pattern
  traceback_patterns = [
    r'(Traceback:.*?)(?:\n\n|$)',
    r'(Traceback \(most recent call last\):.*?)(?:\n\n|$)',
  ]
  
  for pattern in traceback_patterns:
    match = re.search(pattern, decoded, re.DOTALL)
    if match:
      traceback = match.group(1).strip()
      # Clean up any remaining escape sequences
      traceback = decode_escaped_string(traceback)
      return traceback
  
  return None


def extract_error_from_traceback(traceback_text):
  """Extract the final error type and message from traceback"""
  if not traceback_text:
    return None, None
  
  # Look for final error pattern at the end of traceback
  # Patterns like: KeyError: 'account_name'
  error_patterns = [
    r'(\w+Error|\w+Exception):\s*[\'\"]([^\'\"]+)[\'\"]\s*$',
    r'(\w+Error|\w+Exception):\s*([^\n\)]+)\s*$',
  ]
  
  for pattern in error_patterns:
    match = re.search(pattern, traceback_text.strip(), re.MULTILINE)
    if match:
      error_type = match.group(1)
      error_message = match.group(2).strip().strip("'\"")
      return error_type, error_message
  
  return None, None


def format_error_message(error_text):
  """Parse and format error messages for better readability"""
  if not error_text:
    return None, None
  
  error_str = str(error_text)
  main_error = None
  traceback_text = None
  
  # Step 1: Try to parse as tuple format first
  message, error_type, error_value = parse_tuple_error(error_str)
  if error_type and error_value:
    main_error = f"**{error_type}**: `{error_value}`"
    # Decode the message part to handle escaped sequences
    decoded_message = decode_escaped_string(message)
    # Extract traceback from the decoded message part
    traceback_text = extract_traceback(decoded_message)
    if not traceback_text:
      # If no traceback in message, try the full error string (decoded)
      decoded_error = decode_escaped_string(error_str)
      traceback_text = extract_traceback(decoded_error)
  else:
    # Step 2: Extract traceback directly (decode first to handle escaped sequences)
    decoded_error = decode_escaped_string(error_str)
    traceback_text = extract_traceback(decoded_error)
    
    # Step 3: Extract error from traceback if found
    if traceback_text:
      error_type, error_message = extract_error_from_traceback(traceback_text)
      if error_type and error_message:
        main_error = f"**{error_type}**: `{error_message}`"
    
    # Step 4: If still no error, try direct error patterns in the text
    if not main_error:
      error_patterns = [
        r"(KeyError|ValueError|TypeError|AttributeError|RuntimeError|IndexError|NameError):\s*['\"]([^'\"]+)['\"]",
        r"(KeyError|ValueError|TypeError|AttributeError|RuntimeError|IndexError|NameError):\s*([^\n\)]+)",
      ]
      
      for pattern in error_patterns:
        matches = list(re.finditer(pattern, error_str))
        if matches:
          match = matches[-1]  # Use last match
          error_type = match.group(1)
          error_message = match.group(2).strip().strip("'\"")
          main_error = f"**{error_type}**: `{error_message}`"
          break
  
  # Step 5: Fallback - find any error-like line
  if not main_error:
    decoded = decode_escaped_string(error_str)
    lines = decoded.split('\n')
    for line in lines:
      if re.search(r'\b(KeyError|ValueError|TypeError|AttributeError|RuntimeError|Exception|Failed)\b', line, re.IGNORECASE):
        line = line.strip()
        if line and len(line) < 200:
          main_error = line
          break
  
  # Step 6: Final fallback - use simplified version
  if not main_error:
    cleaned = decode_escaped_string(error_str)
    # Remove tuple wrapper if present
    if cleaned.startswith("('") or cleaned.startswith('("'):
      match = re.search(r"'([^']+)'|\"([^\"]+)\"", cleaned)
      if match:
        cleaned = match.group(1) or match.group(2)
    
    # Truncate if too long
    if len(cleaned) > 200:
      cleaned = cleaned[:197] + "..."
    main_error = cleaned
  
  return main_error, traceback_text


def display_error(error_text, error_title="⚠️ **Error occurred**"):
  """Display a formatted error in Streamlit"""
  if not error_text:
    return
  
  main_error, traceback_text = format_error_message(error_text)
  
  st.error(error_title)
  if main_error:
    st.markdown(main_error)
  
  if traceback_text:
    with st.expander("🔍 View Full Error Details", expanded=False):
      st.code(traceback_text, language="python")


def is_error_response(content):
  """Check if content contains error information"""
  if not content:
    return False
  
  error_indicators = [
    "Sorry, I encountered an error:",
    "Error executing code:",
    "Sandbox execution failed",
    "KeyError",
    "ValueError",
    "TypeError",
    "AttributeError",
    "RuntimeError",
  ]
  
  content_str = str(content)
  return any(indicator in content_str for indicator in error_indicators)


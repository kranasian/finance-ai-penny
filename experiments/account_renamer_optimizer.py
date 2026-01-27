from google import genai
from google.genai import types
import os
import sys
import json
from dotenv import load_dotenv

# Add the parent directory to the path so we can import the tools
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
  sys.path.insert(0, parent_dir)

# Load environment variables
load_dotenv()

SYSTEM_PROMPT = """Transform account names to concise, readable format. Output JSON array: each object has "id" and "account_name".

**Step 1: Process each account to create preliminary name**
1. Select source: Use `long_account_name` if more descriptive, else `account_name`.
2. Remove words: Delete (case-insensitive) "Free", "Visa", "Mastercard", "Discover", "Signature Card", "Card", "Account".
3. Shorten bank: Convert to common consumer name (e.g., "Technology Credit Union" → "Tech CU", "Citibank" → "Citi").
4. Build: [Shortened Bank] + " " + [Cleaned Name]
5. Title case: Apply Title Case (preserve acronyms).
6. Truncate: If >35 chars, remove words from right at whole-word boundaries. Never truncate bank prefix.

**Step 2: After ALL preliminary names are created, handle duplicates**
7. Detect duplicates: Compare all names. If any appears 2+ times, mark all instances.
8. Add masks: Append " **" + 4-digit `mask` ONLY to duplicates. Format: "Name **1234" (space before **, no space between ** and digits).

Preserve all input `id`s. Output valid JSON array.
"""

class AccountRenamerOptimizer:
  """Handles all Gemini API interactions for Account Renamer"""
  
  def __init__(self, model_name="gemini-flash-lite-latest"):
    """Initialize the Gemini agent with API configuration for Account Renamer"""
    # API Configuration
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
      raise ValueError("GEMINI_API_KEY environment variable is not set. Please set it in your .env file or environment.")
    self.client = genai.Client(api_key=api_key)
    
    # Model Configuration
    self.thinking_budget = 4096
    self.model_name = model_name
    
    # Generation Configuration Constants (from accounts_rename in gen_ai_lib.py)
    self.temperature = 0.3
    self.top_p = 0.95
    self.top_k = 40
    self.max_output_tokens = 4098
    
    # Safety Settings
    self.safety_settings = [
      types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"),
      types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"),
      types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"),
      types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="OFF")
    ]
    
    # System Prompt
    self.system_prompt = SYSTEM_PROMPT
  
  def generate_response(self, account_list: list) -> dict:
    """
    Generate a response using Gemini API for Account Renamer.
    
    Args:
      account_list: List of account dictionaries with id, account_name, long_account_name, bank_name, mask
      
    Returns:
      Dictionary containing response JSON and thought_summary
    """
    # Create request text with account list as JSON
    request_text = types.Part.from_text(text=f"input: {json.dumps(account_list, indent=0)}\n\noutput:")
    
    # Create content and configuration
    contents = [types.Content(role="user", parts=[request_text])]
    
    generate_content_config = types.GenerateContentConfig(
      temperature=self.temperature,
      top_p=self.top_p,
      top_k=self.top_k,
      max_output_tokens=self.max_output_tokens,
      safety_settings=self.safety_settings,
      system_instruction=[types.Part.from_text(text=self.system_prompt)],
      thinking_config=types.ThinkingConfig(
        thinking_budget=self.thinking_budget,
        include_thoughts=True
      ),
      response_mime_type="application/json",
    )

    # Generate response
    output_text = ""
    thought_summary = ""
    
    # According to Gemini API docs: iterate through chunks and check part.thought boolean
    for chunk in self.client.models.generate_content_stream(
      model=self.model_name,
      contents=contents,
      config=generate_content_config,
    ):
      # Extract text content (non-thought parts)
      if chunk.text is not None:
        output_text += chunk.text
      
      # Extract thought summary from chunk
      if hasattr(chunk, 'candidates') and chunk.candidates:
        for candidate in chunk.candidates:
          # Extract thought summary from parts (per Gemini API docs)
          # Check part.thought boolean to identify thought parts
          if hasattr(candidate, 'content') and candidate.content:
            if hasattr(candidate.content, 'parts') and candidate.content.parts:
              for part in candidate.content.parts:
                # Check if this part is a thought summary (per documentation)
                if hasattr(part, 'thought') and part.thought:
                  if hasattr(part, 'text') and part.text:
                    # Accumulate thought summary text (for streaming, it may come in chunks)
                    if thought_summary:
                      thought_summary += part.text
                    else:
                      thought_summary = part.text
    
    # Parse JSON response
    response_json = None
    try:
      response_json = json.loads(output_text) if output_text else None
    except json.JSONDecodeError as e:
      print(f"Error parsing JSON response: {str(e)}")
      print(f"Raw output: {output_text}")
    
    if thought_summary:
      print("-" * 80)
      print("THOUGHT SUMMARY:")
      print(thought_summary.strip())
      print("-" * 80)
    
    return {
      "response": response_json,
      "thought_summary": thought_summary.strip() if thought_summary else ""
    }
  
  def get_available_models(self):
    """
    Get list of available Gemini models.
    
    Returns:
      List of available model names
    """
    try:
      models = genai.list_models()
      return [m.name for m in models if 'generateContent' in m.supported_generation_methods]
    except Exception as e:
      raise Exception(f"Failed to get models: {str(e)}")


def _run_test_with_logging(account_list: list, optimizer: AccountRenamerOptimizer = None):
  """
  Internal helper function that runs a test with consistent logging.
  
  Args:
    account_list: List of account dictionaries
    optimizer: Optional AccountRenamerOptimizer instance. If None, creates a new one.
    
  Returns:
    The generated response dictionary
  """
  if optimizer is None:
    optimizer = AccountRenamerOptimizer()
  
  # Print the input
  print("LLM INPUT:")
  print("-" * 80)
  print(json.dumps(account_list, indent=2))
  print("-" * 80)
  
  result = optimizer.generate_response(account_list)
  
  # Print the output
  print("LLM OUTPUT:")
  print("-" * 80)
  print(json.dumps(result["response"], indent=2) if result["response"] else "No response")
  print("-" * 80)
  
  return result


TEST_CASES = [
  {
    "name": "basic_word_removal_and_bank_prepending",
    "account_list": [
      {
        "id": 15,
        "account_name": "FREE CHECKING",
        "long_account_name": "FREE CHECKING",
        "bank_name": "Patelco",
        "mask": 2231
      },
      {
        "id": 20,
        "account_name": "MONEY MARKET PLUS",
        "long_account_name": "MONEY MARKET PLUS",
        "bank_name": "Patelco",
        "mask": 9595
      },
      {
        "id": 28,
        "account_name": "Sapphire Preferred",
        "long_account_name": "Chase Sapphire Preferred",
        "bank_name": "Chase",
        "mask": 4933
      },
      {
        "id": 23,
        "account_name": "MONEY MARKET SELECT",
        "long_account_name": "MONEY MARKET SELECT",
        "bank_name": "Patelco",
        "mask": 6087
      },
      {
        "id": 2425,
        "account_name": "Money Market Plus",
        "long_account_name": "Money Market Plus",
        "bank_name": "Technology Credit Union",
        "mask": 8941
      }
    ],
    "ideal_response": """Expected output:
[
  {"id": 15, "account_name": "Patelco Checking"},
  {"id": 20, "account_name": "Patelco Money Market Plus"},
  {"id": 28, "account_name": "Chase Sapphire"},
  {"id": 23, "account_name": "Patelco Money Market Select"},
  {"id": 2425, "account_name": "Tech CU Money Market Plus"}
]
Key validations:
- Remove "FREE" from account 15
- Prepend bank names to all accounts
- Use long_account_name for account 28 (more descriptive)
- Shorten "Technology Credit Union" to "Tech CU"
- All names unique, no masks needed"""
  },
  {
    "name": "credit_cards_word_removal_and_long_name",
    "account_list": [
      {
        "id": 62,
        "account_name": "Quicksilver Cash Rewards",
        "long_account_name": "Capital One Quicksilver Cash Rewards",
        "bank_name": "Capital One",
        "mask": 9077
      },
      {
        "id": 34,
        "account_name": "CREDIT CARD",
        "long_account_name": "Amazon Prime Rewards Visa Signature Card",
        "bank_name": "Chase",
        "mask": 1309
      },
      {
        "id": 75,
        "account_name": "CREDIT CARD",
        "long_account_name": "UnitedSM Explorer Card",
        "bank_name": "Chase",
        "mask": 5972
      },
      {
        "id": 301,
        "account_name": "DOUBLE CASH",
        "long_account_name": "Citi Double Cash",
        "bank_name": "Citibank",
        "mask": 6055
      }
    ],
    "ideal_response": """Expected output:
[
  {"id": 62, "account_name": "Capital One Quicksilver"},
  {"id": 34, "account_name": "Chase Amazon Prime Rewards"},
  {"id": 75, "account_name": "Chase UnitedSM Explorer"},
  {"id": 301, "account_name": "Citi Double Cash"}
]
Key validations:
- Use long_account_name when more descriptive (accounts 34, 75)
- Remove "Visa Signature Card" from account 34
- Remove "Card" from account 75
- Truncate "Capital One Quicksilver Cash Rewards" to fit 35 chars
- All names unique, no masks needed"""
  },
  {
    "name": "duplicate_detection_and_masking",
    "account_list": [
      {
        "id": 76,
        "account_name": "Chase Total Checking",
        "long_account_name": "Chase Total Checking",
        "bank_name": "Chase",
        "mask": 4227
      },
      {
        "id": 322,
        "account_name": "Cashback Visa Signature",
        "long_account_name": "Cashback Visa Signature",
        "bank_name": "Alliant",
        "mask": 9236
      },
      {
        "id": 63,
        "account_name": "Chase Total Checking",
        "long_account_name": "Chase Total Checking",
        "bank_name": "Chase",
        "mask": 8493
      },
      {
        "id": 311,
        "account_name": "Checking",
        "long_account_name": "Checking",
        "bank_name": "Wells Fargo",
        "mask": 9598
      },
      {
        "id": 544,
        "account_name": "Savings Account",
        "long_account_name": "Savings Account",
        "bank_name": "Wells Fargo",
        "mask": 2460
      },
      {
        "id": 692,
        "account_name": "Savings Account",
        "long_account_name": "Savings Account",
        "bank_name": "Wells Fargo",
        "mask": 8123
      }
    ],
    "ideal_response": """Expected output:
[
  {"id": 76, "account_name": "Chase Total Checking **4227"},
  {"id": 322, "account_name": "Alliant Cashback"},
  {"id": 63, "account_name": "Chase Total Checking **8493"},
  {"id": 311, "account_name": "Wells Fargo Checking"},
  {"id": 544, "account_name": "Wells Fargo Savings **2460"},
  {"id": 692, "account_name": "Wells Fargo Savings **8123"}
]
Key validations:
- Accounts 76 and 63: Both become "Chase Total Checking" (duplicate) -> add masks
- Account 322: Remove "Visa Signature" -> "Alliant Cashback"
- Account 311: Remove "Account" word, prepend bank -> "Wells Fargo Checking"
- Accounts 544 and 692: Both become "Wells Fargo Savings" (duplicate) -> add masks
- Mask format: " **" (space + two asterisks) followed by 4-digit mask, no space before mask"""
  },
]


def get_test_case(test_name_or_index):
  """
  Get a test case by name or index.
  
  Args:
    test_name_or_index: Test case name (str) or index (int)
    
  Returns:
    Test case dict or None if not found
  """
  if isinstance(test_name_or_index, int):
    if 0 <= test_name_or_index < len(TEST_CASES):
      return TEST_CASES[test_name_or_index]
    return None
  elif isinstance(test_name_or_index, str):
    for test_case in TEST_CASES:
      if test_case["name"] == test_name_or_index:
        return test_case
    return None
  return None


def run_test(test_name_or_index_or_dict, optimizer: AccountRenamerOptimizer = None):
  """
  Run a single test by name, index, or by passing test data directly.
  
  Args:
    test_name_or_index_or_dict: One of:
      - Test case name (str): e.g., "example_1_patelco_and_chase"
      - Test case index (int): e.g., 0
      - Test data dict: {"account_list": [...], "name": "..."}
    optimizer: Optional AccountRenamerOptimizer instance. If None, creates a new one.
    
  Returns:
    The generated response dictionary, or None if test not found
  """
  # Check if it's a dict with test data
  if isinstance(test_name_or_index_or_dict, dict):
    if "account_list" in test_name_or_index_or_dict:
      test_name = test_name_or_index_or_dict.get("name", "custom_test")
      print(f"\n{'='*80}")
      print(f"Running test: {test_name}")
      print(f"{'-'*80}\n")
      
      result = _run_test_with_logging(
        test_name_or_index_or_dict["account_list"],
        optimizer
      )
      
      if test_name_or_index_or_dict.get("ideal_response", ""):
        print(f"Ideal response: {test_name_or_index_or_dict['ideal_response']}")
        print(f"{'='*80}\n")
      
      return result
    else:
      print(f"Invalid test dict: must contain 'account_list' key.")
      return None
  
  # Otherwise, treat it as a test name or index
  test_case = get_test_case(test_name_or_index_or_dict)
  if test_case is None:
    print(f"Test case '{test_name_or_index_or_dict}' not found.")
    return None
  
  print(f"\n{'='*80}")
  print(f"Running test: {test_case['name']}")
  print(f"{'='*80}\n")
  
  result = _run_test_with_logging(
    test_case["account_list"],
    optimizer
  )
  
  if test_case.get("ideal_response", ""):
    print(f"Ideal response: {test_case['ideal_response']}")
    print(f"{'='*80}\n")
  
  return result


def run_tests(test_names_or_indices_or_dicts, optimizer: AccountRenamerOptimizer = None):
  """
  Run multiple tests by names, indices, or by passing test data directly.
  
  Args:
    test_names_or_indices_or_dicts: One of:
      - None: Run all tests from TEST_CASES
      - List of test case names (str), indices (int), or test data dicts
        Each dict should have: {"account_list": [...], "name": "..."}
    optimizer: Optional AccountRenamerOptimizer instance. If None, creates a new one.
    
  Returns:
    List of generated response dictionaries
  """
  if test_names_or_indices_or_dicts is None:
    # Run all tests
    test_names_or_indices_or_dicts = list(range(len(TEST_CASES)))
  
  results = []
  for test_item in test_names_or_indices_or_dicts:
    result = run_test(test_item, optimizer)
    results.append(result)
  
  return results


def test_with_inputs(account_list: list, optimizer: AccountRenamerOptimizer = None):
  """
  Convenient method to test the account renamer optimizer with custom inputs.
  
  Args:
    account_list: List of account dictionaries
    optimizer: Optional AccountRenamerOptimizer instance. If None, creates a new one.
    
  Returns:
    The generated response dictionary
  """
  return _run_test_with_logging(account_list, optimizer)


def main(batch: int = None, test: str = None):
  """
  Main function to test the account renamer optimizer
  
  Args:
    batch: Batch number (1) to run all tests, or None to run a single test
    test: Test name, index, or None. If batch is provided, test is ignored.
      - Test name (str): e.g., "example_1_patelco_and_chase"
      - Test index (str): e.g., "0" (will be converted to int)
      - None: If batch is also None, prints available tests
  """
  optimizer = AccountRenamerOptimizer()
  
  # Define test batches
  BATCHES = {
    1: {
      "name": "Account Renamer Test Cases",
      "tests": [0, 1, 2]  # All 3 test cases
    },
  }
  
  if batch is not None:
    # Run a batch of tests
    if batch not in BATCHES:
      print(f"Invalid batch number: {batch}. Available batches: {list(BATCHES.keys())}")
      print("\nBatch descriptions:")
      for b, info in BATCHES.items():
        test_names = [TEST_CASES[idx]["name"] for idx in info["tests"]]
        print(f"  Batch {b}: {info['name']} - {', '.join(test_names)}")
      return
    
    batch_info = BATCHES[batch]
    print(f"\n{'='*80}")
    print(f"BATCH {batch}: {batch_info['name']}")
    print(f"{'='*80}\n")
    
    for test_idx in batch_info["tests"]:
      run_test(test_idx, optimizer)
      print("\n" + "-"*80 + "\n")
  
  elif test is not None:
    # Run a single test
    # Try to convert to int if it's a numeric string
    if test.isdigit():
      test = int(test)
    
    result = run_test(test, optimizer)
    if result is None:
      print(f"\nAvailable test cases:")
      for i, test_case in enumerate(TEST_CASES):
        print(f"  {i}: {test_case['name']}")
  
  else:
    # Print available options
    print("Usage:")
    print("  Run a batch: --batch <1>")
    print("  Run a single test: --test <name_or_index>")
    print("\nAvailable batches:")
    for b, info in BATCHES.items():
      test_names = [TEST_CASES[idx]["name"] for idx in info["tests"]]
      print(f"  Batch {b}: {info['name']}")
      for idx in info["tests"]:
        print(f"    - {idx}: {TEST_CASES[idx]['name']}")
    print("\nAll test cases:")
    for i, test_case in enumerate(TEST_CASES):
      print(f"  {i}: {test_case['name']}")


"""
Sample Usage Examples:
  python account_renamer_optimizer.py --batch 1
  python account_renamer_optimizer.py --test example_1_patelco_and_chase
  python account_renamer_optimizer.py --test 0
  run_test("example_1_patelco_and_chase")
  run_tests([0, 1, 2])
"""

if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser(description='Run account renamer optimizer tests in batches or individually')
  parser.add_argument('--batch', type=int, choices=[1],
                      help='Batch number to run (1)')
  parser.add_argument('--test', type=str,
                      help='Test name or index to run individually (e.g., "example_1_patelco_and_chase" or "0")')
  args = parser.parse_args()
  main(batch=args.batch, test=args.test)

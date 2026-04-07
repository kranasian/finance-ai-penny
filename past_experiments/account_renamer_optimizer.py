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

SYSTEM_PROMPT = """Transform a list of financial accounts into a concise, standardized JSON array of objects with keys "id" and "account_name".

### Logic Process (Must be applied to the ENTIRE BATCH at once)

**Step 1: Generate Preliminary Names (for each account)**
- Start with `long_account_name` (or `account_name` if long is missing).
- **Clean**:
  - Remove filler words: "Free", "Visa", "Mastercard", "Signature Card", "Signature", "Card", "Account".
  - Strip special symbols (®, ™, ©, *, etc.). Keep hyphens (-) and forward slashes (/) if part of the name.
  - Apply Title Case. Keep acronyms (HELOC, CD, SM) in ALL CAPS.
  - Replace bank names with short forms: Technology Credit Union -> "Tech CU", Citibank -> "Citi", Bank of America -> "BofA".
- **Bank Context**:
  - If ALL accounts in the batch are from the SAME bank: **REMOVE** the bank name from the string (e.g., "Truist Enjoy Cash" -> "Enjoy Cash").
  - If accounts are from DIFFERENT banks: **PREPEND** the bank short name if missing (e.g., "Sapphire" -> "Chase Sapphire").

**Step 2: Global Deduplication & Masking**
- Collect ALL preliminary names from Step 1.
- Identify names that appear **more than once** (exact string match).
- **For Duplicates**: Append " **" + `mask` (or `account_mask`) to ALL instances (e.g., "Checking **1234").
- **For Unique Names**: Leave the name AS IS. **NEVER** add a mask to a unique name.

### Constraints
- Output ONLY a valid JSON array.
- Max length 35 chars. Truncate from the right if needed, but keep bank prefix.
- Mask format: Space + `**` + Mask (e.g., ` **1234`). No space between `**` and number.
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
    self.thinking_budget = 2048
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
    "name": "mixed_banks_small_batch",
    "account_list": [
      {
        "id": 101,
        "account_name": "TOTAL CHECKING",
        "long_account_name": "Chase Total Checking",
        "bank_name": "Chase",
        "mask": 1234
      },
      {
        "id": 102,
        "account_name": "WAY2SAVE SAVINGS",
        "long_account_name": "Way2Save Savings",
        "bank_name": "Wells Fargo",
        "mask": 5678
      },
      {
        "id": 103,
        "account_name": "DOUBLE CASH",
        "long_account_name": "Citi Double Cash Card",
        "bank_name": "Citibank",
        "mask": 9012
      }
    ],
    "ideal_response": """Expected output:
[
  {"id": 101, "account_name": "Chase Total Checking"},
  {"id": 102, "account_name": "Wells Fargo Way2Save Savings"},
  {"id": 103, "account_name": "Citi Double Cash"}
]
Key validations:
- Prepend bank names (different banks)
- Clean up "Card" from Citi
- Standardize capitalization"""
  },
  {
    "name": "single_bank_many_products",
    "account_list": [
      {
        "id": 201,
        "account_name": "Adv Plus Banking",
        "long_account_name": "Bank of America Advantage Plus Banking",
        "bank_name": "Bank of America",
        "mask": 1111
      },
      {
        "id": 202,
        "account_name": "Adv Relationship Banking",
        "long_account_name": "Bank of America Advantage Relationship Banking",
        "bank_name": "Bank of America",
        "mask": 2222
      },
      {
        "id": 203,
        "account_name": "Customized Cash",
        "long_account_name": "Bank of America Customized Cash Rewards",
        "bank_name": "Bank of America",
        "mask": 3333
      },
      {
        "id": 204,
        "account_name": "Travel Rewards",
        "long_account_name": "Bank of America Travel Rewards Credit Card",
        "bank_name": "Bank of America",
        "mask": 4444
      },
      {
        "id": 205,
        "account_name": "Premium Rewards",
        "long_account_name": "Bank of America Premium Rewards Credit Card",
        "bank_name": "Bank of America",
        "mask": 5555
      }
    ],
    "ideal_response": """Expected output:
[
  {"id": 201, "account_name": "Advantage Plus Banking"},
  {"id": 202, "account_name": "Advantage Relationship Banking"},
  {"id": 203, "account_name": "Customized Cash Rewards"},
  {"id": 204, "account_name": "Travel Rewards"},
  {"id": 205, "account_name": "Premium Rewards"}
]
Key validations:
- All same bank (Bank of America) -> Remove bank name
- Remove "Credit Card"
- Use long names where appropriate"""
  },
  {
    "name": "duplicate_names_stress_test",
    "account_list": [
      {
        "id": 301,
        "account_name": "TOTAL CHECKING",
        "long_account_name": "Chase Total Checking",
        "bank_name": "Chase",
        "mask": 1001
      },
      {
        "id": 302,
        "account_name": "TOTAL CHECKING",
        "long_account_name": "Chase Total Checking",
        "bank_name": "Chase",
        "mask": 1002
      },
      {
        "id": 303,
        "account_name": "TOTAL CHECKING",
        "long_account_name": "Chase Total Checking",
        "bank_name": "Chase",
        "mask": 1003
      },
      {
        "id": 304,
        "account_name": "360 Performance Savings",
        "long_account_name": "Capital One 360 Performance Savings",
        "bank_name": "Capital One",
        "mask": 2001
      },
      {
        "id": 305,
        "account_name": "360 Performance Savings",
        "long_account_name": "Capital One 360 Performance Savings",
        "bank_name": "Capital One",
        "mask": 2002
      }
    ],
    "ideal_response": """Expected output:
[
  {"id": 301, "account_name": "Chase Total Checking **1001"},
  {"id": 302, "account_name": "Chase Total Checking **1002"},
  {"id": 303, "account_name": "Chase Total Checking **1003"},
  {"id": 304, "account_name": "Capital One 360 Savings **2001"},
  {"id": 305, "account_name": "Capital One 360 Savings **2002"}
]
Key validations:
- Detect duplicates across multiple instances
- Apply masks to ALL duplicates
- Shorten "Performance Savings" if needed or keep if fits"""
  },
  {
    "name": "complex_names_cleanup",
    "account_list": [
      {
        "id": 401,
        "account_name": "Amazon Prime",
        "long_account_name": "Amazon Prime Rewards Visa Signature Card",
        "bank_name": "Chase",
        "mask": 4001
      },
      {
        "id": 402,
        "account_name": "United Club",
        "long_account_name": "United Club Infinite Card",
        "bank_name": "Chase",
        "mask": 4002
      },
      {
        "id": 403,
        "account_name": "Costco Visa",
        "long_account_name": "Costco Anywhere Visa® Card by Citi",
        "bank_name": "Citibank",
        "mask": 4003
      },
      {
        "id": 404,
        "account_name": "Delta Gold",
        "long_account_name": "Delta SkyMiles® Gold American Express Card",
        "bank_name": "American Express",
        "mask": 4004
      },
      {
        "id": 405,
        "account_name": "Marriott",
        "long_account_name": "Marriott Bonvoy Boundless® Credit Card",
        "bank_name": "Chase",
        "mask": 4005
      }
    ],
    "ideal_response": """Expected output:
[
  {"id": 401, "account_name": "Chase Amazon Prime Rewards"},
  {"id": 402, "account_name": "Chase United Club Infinite"},
  {"id": 403, "account_name": "Citi Costco Anywhere"},
  {"id": 404, "account_name": "Amex Delta SkyMiles Gold"},
  {"id": 405, "account_name": "Chase Marriott Bonvoy Boundless"}
]
Key validations:
- Remove "Visa Signature", "Card", "Credit Card"
- Remove symbols like ®
- Prepend bank names (Chase, Citi, Amex)"""
  },
  {
    "name": "credit_unions_and_regionals",
    "account_list": [
      {
        "id": 501,
        "account_name": "Free Checking",
        "long_account_name": "Golden 1 Free Checking",
        "bank_name": "Golden 1 Credit Union",
        "mask": 5001
      },
      {
        "id": 502,
        "account_name": "Money Market",
        "long_account_name": "SchoolsFirst FCU Money Market",
        "bank_name": "SchoolsFirst FCU",
        "mask": 5002
      },
      {
        "id": 503,
        "account_name": "Virtual Wallet",
        "long_account_name": "PNC Virtual Wallet",
        "bank_name": "PNC Bank",
        "mask": 5003
      },
      {
        "id": 504,
        "account_name": "Simply Free",
        "long_account_name": "Truist Simply Free Checking",
        "bank_name": "Truist",
        "mask": 5004
      }
    ],
    "ideal_response": """Expected output:
[
  {"id": 501, "account_name": "Golden 1 Checking"},
  {"id": 502, "account_name": "SchoolsFirst Money Market"},
  {"id": 503, "account_name": "PNC Virtual Wallet"},
  {"id": 504, "account_name": "Truist Simply Free Checking"}
]
Key validations:
- Handle less common banks
- Remove "Free" filler
- Prepend bank names"""
  },
  {
    "name": "similar_names_different_banks",
    "account_list": [
      {
        "id": 601,
        "account_name": "Platinum",
        "long_account_name": "The Platinum Card®",
        "bank_name": "American Express",
        "mask": 6001
      },
      {
        "id": 602,
        "account_name": "Platinum",
        "long_account_name": "Capital One Platinum Credit Card",
        "bank_name": "Capital One",
        "mask": 6002
      },
      {
        "id": 603,
        "account_name": "Gold",
        "long_account_name": "American Express® Gold Card",
        "bank_name": "American Express",
        "mask": 6003
      },
      {
        "id": 604,
        "account_name": "Sapphire",
        "long_account_name": "Chase Sapphire Preferred",
        "bank_name": "Chase",
        "mask": 6004
      }
    ],
    "ideal_response": """Expected output:
[
  {"id": 601, "account_name": "Amex Platinum"},
  {"id": 602, "account_name": "Capital One Platinum"},
  {"id": 603, "account_name": "Amex Gold"},
  {"id": 604, "account_name": "Chase Sapphire Preferred"}
]
Key validations:
- Distinguish "Platinum" by bank name
- Remove "Card", "The", symbols
- Shorten "American Express" to "Amex" """
  },
  {
    "name": "missing_long_names",
    "account_list": [
      {
        "id": 701,
        "account_name": "Checking",
        "long_account_name": "Checking",
        "bank_name": "Chase",
        "mask": 7001
      },
      {
        "id": 702,
        "account_name": "Savings",
        "long_account_name": "Savings",
        "bank_name": "Wells Fargo",
        "mask": 7002
      },
      {
        "id": 703,
        "account_name": "Credit Card",
        "long_account_name": "Credit Card",
        "bank_name": "Citibank",
        "mask": 7003
      }
    ],
    "ideal_response": """Expected output:
[
  {"id": 701, "account_name": "Chase Checking"},
  {"id": 702, "account_name": "Wells Fargo Savings"},
  {"id": 703, "account_name": "Citi Credit Card"}
]
Key validations:
- Fallback to account_name when long_name is identical/useless
- Prepend bank name
- Note: "Credit Card" might be generic, but with bank name it's acceptable if no other info exists"""
  },
  {
    "name": "large_mixed_batch",
    "account_list": [
      { "id": 801, "account_name": "Checking", "long_account_name": "Chase Total Checking", "bank_name": "Chase", "mask": 8001 },
      { "id": 802, "account_name": "Savings", "long_account_name": "Chase Savings", "bank_name": "Chase", "mask": 8002 },
      { "id": 803, "account_name": "Sapphire", "long_account_name": "Chase Sapphire Reserve", "bank_name": "Chase", "mask": 8003 },
      { "id": 804, "account_name": "Freedom", "long_account_name": "Chase Freedom Unlimited", "bank_name": "Chase", "mask": 8004 },
      { "id": 805, "account_name": "Checking", "long_account_name": "Wells Fargo Everyday Checking", "bank_name": "Wells Fargo", "mask": 8005 },
      { "id": 806, "account_name": "Active Cash", "long_account_name": "Wells Fargo Active Cash Card", "bank_name": "Wells Fargo", "mask": 8006 },
      { "id": 807, "account_name": "Gold", "long_account_name": "Amex Gold Card", "bank_name": "American Express", "mask": 8007 },
      { "id": 808, "account_name": "Platinum", "long_account_name": "Amex Platinum Card", "bank_name": "American Express", "mask": 8008 },
      { "id": 809, "account_name": "BCE", "long_account_name": "Blue Cash Everyday", "bank_name": "American Express", "mask": 8009 },
      { "id": 810, "account_name": "Discover", "long_account_name": "Discover It Cash Back", "bank_name": "Discover", "mask": 8010 },
      { "id": 811, "account_name": "Checking", "long_account_name": "Capital One 360 Checking", "bank_name": "Capital One", "mask": 8011 },
      { "id": 812, "account_name": "Savor", "long_account_name": "Capital One SavorOne", "bank_name": "Capital One", "mask": 8012 }
    ],
    "ideal_response": """Expected output:
[
  {"id": 801, "account_name": "Chase Total Checking"},
  {"id": 802, "account_name": "Chase Savings"},
  {"id": 803, "account_name": "Chase Sapphire Reserve"},
  {"id": 804, "account_name": "Chase Freedom Unlimited"},
  {"id": 805, "account_name": "Wells Fargo Everyday Checking"},
  {"id": 806, "account_name": "Wells Fargo Active Cash"},
  {"id": 807, "account_name": "Amex Gold"},
  {"id": 808, "account_name": "Amex Platinum"},
  {"id": 809, "account_name": "Amex Blue Cash Everyday"},
  {"id": 810, "account_name": "Discover It Cash Back"},
  {"id": 811, "account_name": "Capital One 360 Checking"},
  {"id": 812, "account_name": "Capital One SavorOne"}
]
Key validations:
- Handle large batch with multiple banks
- Ensure correct bank prefixes for all
- Ensure uniqueness without unnecessary masks"""
  }
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
      print(f"\\n{'='*80}")
      print(f"Running test: {test_name}")
      print(f"{'-'*80}\\n")
      
      result = _run_test_with_logging(
        test_name_or_index_or_dict["account_list"],
        optimizer
      )
      
      if test_name_or_index_or_dict.get("ideal_response", ""):
        print(f"Ideal response: {test_name_or_index_or_dict['ideal_response']}")
        print(f"{'='*80}\\n")
      
      return result
    else:
      print(f"Invalid test dict: must contain 'account_list' key.")
      return None
  
  # Otherwise, treat it as a test name or index
  test_case = get_test_case(test_name_or_index_or_dict)
  if test_case is None:
    print(f"Test case '{test_name_or_index_or_dict}' not found.")
    return None
  
  print(f"\\n{'='*80}")
  print(f"Running test: {test_case['name']}")
  print(f"{'='*80}\\n")
  
  result = _run_test_with_logging(
    test_case["account_list"],
    optimizer
  )
  
  if test_case.get("ideal_response", ""):
    print(f"Ideal response: {test_case['ideal_response']}")
    print(f"{'='*80}\\n")
  
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
      "name": "All Test Cases",
      "tests": list(range(len(TEST_CASES)))
    },
    2: {
      "name": "Basic Scenarios",
      "tests": [0, 1]
    },
    3: {
      "name": "Complex Logic (Dedupe & Cleanup)",
      "tests": [2, 3, 5]
    },
    4: {
      "name": "Edge Cases (Regional, Missing Data)",
      "tests": [4, 6]
    },
    5: {
      "name": "Large Scale",
      "tests": [7]
    },
  }
  
  if batch is not None:
    # Run a batch of tests
    if batch not in BATCHES:
      print(f"Invalid batch number: {batch}. Available batches: {list(BATCHES.keys())}")
      print("\\nBatch descriptions:")
      for b, info in BATCHES.items():
        test_names = [TEST_CASES[idx]["name"] for idx in info["tests"]]
        print(f"  Batch {b}: {info['name']} - {', '.join(test_names)}")
      return
    
    batch_info = BATCHES[batch]
    print(f"\\n{'='*80}")
    print(f"BATCH {batch}: {batch_info['name']}")
    print(f"{'='*80}\\n")
    
    for test_idx in batch_info["tests"]:
      run_test(test_idx, optimizer)
      print("\\n" + "-"*80 + "\\n")
  
  elif test is not None:
    # Run a single test
    # Try to convert to int if it's a numeric string
    if test.isdigit():
      test = int(test)
    
    result = run_test(test, optimizer)
    if result is None:
      print(f"\\nAvailable test cases:")
      for i, test_case in enumerate(TEST_CASES):
        print(f"  {i}: {test_case['name']}")
  
  else:
    # Print available options
    print("Usage:")
    print("  Run a batch: --batch <1>")
    print("  Run a single test: --test <name_or_index>")
    print("\\nAvailable batches:")
    for b, info in BATCHES.items():
      test_names = [TEST_CASES[idx]["name"] for idx in info["tests"]]
      print(f"  Batch {b}: {info['name']}")
      for idx in info["tests"]:
        print(f"    - {idx}: {TEST_CASES[idx]['name']}")
    print("\\nAll test cases:")
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
  parser.add_argument('--batch', type=int, choices=[1, 2, 3, 4, 5],
                      help='Batch number to run (1-5)')
  parser.add_argument('--test', type=str,
                      help='Test name or index to run individually (e.g., "example_1_patelco_and_chase" or "0")')
  args = parser.parse_args()
  main(batch=args.batch, test=args.test)

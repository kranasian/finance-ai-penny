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

SYSTEM_PROMPT = """Transform a list of financial accounts into a concise, distinguishable, and standardized format. 

### Output Format Requirement
Return ONLY a valid JSON array of objects. Each object must have exactly two keys: "id" and "account_name". No preamble, no postscript, and no parent wrapping object.

### Phase 1: Strategic Batch Analysis (Analyze BEFORE processing individual accounts)
1. **Bank Necessity**: 
   - Identify the primary bank for every account. 
   - If ALL accounts in the input batch belong to the SAME bank, bank names are REDUNDANT. Global Flag: `Include_Bank = False`.
   - If accounts belong to DIFFERENT banks, the bank name is REQUIRED for distinction. Global Flag: `Include_Bank = True`.
2. **Redundancy Identification**: Scan for words that appear in every single record (e.g., "Account", "Checking"). These are candidates for removal to maximize brevity.

### Phase 2: Preliminary Name Generation (Apply to each record)
1. **Selection**: Select the most descriptive source between `long_account_name` and `account_name` (e.g., "High Yield" is better than "Savings").
2. **Aggressive Cleaning**: 
   - Strip all special symbols (®, ™, ©, *, etc.).
   - Delete common filler words (case-insensitive): "Free", "Visa", "Mastercard", "Discover", "Signature Card", "Signature", "Card", "Account". 
   - **MANDATORY**: "Free" MUST be removed.
3. **Internal Redundancy Filter**: If the bank's name (or variants like "Amex", "Citi", "BofA") exists in the cleaned string from the step above, remove it to prevent duplicates (e.g., "Chase Chase Checking").
4. **Assembly**:
   - If `Include_Bank` is True: Result = [Bank Short Name] + " " + [Cleaned Name].
   - If `Include_Bank` is False: Result = [Cleaned Name] only.
   - **Bank Short Names**: Technology Credit Union -> "Tech CU", Citibank -> "Citi", Bank of America -> "BofA". Others like Amex, Chase, Truist, Patelco, Alliant stay as-is.
   - **Fallback**: If the name becomes empty or non-descriptive, use the original cleaned `account_name`.
5. **Formatting**: Apply Title Case. Use ALL CAPS for acronyms (SM, CU, CD, HELOC).
6. **Character Limit**: Limit to 35 chars. If longer, remove words from the RIGHT end. NEVER truncate the bank prefix.

### Phase 3: Strategic Masking Logic (Batch-Level)
This phase MUST be performed by comparing the results of ALL records from Phase 2.

**MASKING DECISION ALGORITHM:**
1.  **Count occurrences** of each preliminary name string (case-insensitive).
2.  **IF Count == 1**: Output the name exactly as it is. **DO NOT ADD A MASK.**
3.  **IF Count >= 2**: Append the mask to **EVERY** instance of that specific name.
    - **Format**: "Name **1234" (Exactly one space before **, no space after).

### CRITICAL MASKING CONSTRAINTS
- **THE TWIN RULE**: If an `account_name` includes a mask, there **MUST** be at least one other `account_name` in the batch that is identical word-for-word (excluding the mask).
- **THE ZERO-MASK RULE**: If a name is unique in the batch, adding a mask is a CRITICAL FAILURE. Never mask a unique name.

### Formatting Rule for Acronyms
- **ALWAYS use ALL CAPS for acronyms.** This includes SM, CU, CD, HELOC, AMEX, BofA, etc. (e.g., "UnitedSM" should be "UnitedSM" or "United SM").

### Final Self-Correction Step
Before finalizing the JSON, perform this check for every account:
"Is there any other account in this batch with the exact same name (excluding the mask)?"
- If **NO**: You MUST remove the mask.
- If **YES**: You MUST ensure both have masks.
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
    self.thinking_budget = 1024
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
  {
    "name": "same_bank_multiple_account_types",
    "account_list": [
      {
        "id": 101,
        "account_name": "FREE CHECKING",
        "long_account_name": "Chase Total Checking Account",
        "bank_name": "Chase",
        "mask": 1234
      },
      {
        "id": 102,
        "account_name": "SAVINGS ACCOUNT",
        "long_account_name": "Chase Premier Savings",
        "bank_name": "Chase",
        "mask": 5678
      },
      {
        "id": 103,
        "account_name": "CREDIT CARD",
        "long_account_name": "Chase Sapphire Preferred Visa Signature Card",
        "bank_name": "Chase",
        "mask": 9012
      },
      {
        "id": 104,
        "account_name": "MONEY MARKET",
        "long_account_name": "Chase Money Market Select",
        "bank_name": "Chase",
        "mask": 3456
      },
      {
        "id": 105,
        "account_name": "INVESTMENT ACCOUNT",
        "long_account_name": "Chase You Invest Brokerage Account",
        "bank_name": "Chase",
        "mask": 7890
      }
    ],
    "ideal_response": """Expected output:
[
  {"id": 101, "account_name": "Chase Total Checking"},
  {"id": 102, "account_name": "Chase Premier Savings"},
  {"id": 103, "account_name": "Chase Sapphire Preferred"},
  {"id": 104, "account_name": "Chase Money Market Select"},
  {"id": 105, "account_name": "Chase You Invest Brokerage"}
]
Key validations:
- All accounts from same bank (Chase)
- Remove "FREE" from account 101
- Remove "Account" from accounts 101, 102, 105
- Remove "Visa Signature Card" from account 103
- Use long_account_name when more descriptive
- All names unique, no masks needed"""
  },
  {
    "name": "different_banks_each_account",
    "account_list": [
      {
        "id": 201,
        "account_name": "CHECKING",
        "long_account_name": "Chase Total Checking Account",
        "bank_name": "Chase",
        "mask": 1111
      },
      {
        "id": 202,
        "account_name": "SAVINGS",
        "long_account_name": "Bank of America Advantage Savings",
        "bank_name": "Bank of America",
        "mask": 2222
      },
      {
        "id": 203,
        "account_name": "CREDIT CARD",
        "long_account_name": "Wells Fargo Cash Wise Visa Signature Card",
        "bank_name": "Wells Fargo",
        "mask": 3333
      },
      {
        "id": 204,
        "account_name": "MONEY MARKET",
        "long_account_name": "Capital One 360 Money Market Account",
        "bank_name": "Capital One",
        "mask": 4444
      },
      {
        "id": 205,
        "account_name": "INVESTMENT",
        "long_account_name": "Citi Self Invest Account",
        "bank_name": "Citibank",
        "mask": 5555
      }
    ],
    "ideal_response": """Expected output:
[
  {"id": 201, "account_name": "Chase Total Checking"},
  {"id": 202, "account_name": "BofA Advantage Savings"},
  {"id": 203, "account_name": "Wells Fargo Cash Wise"},
  {"id": 204, "account_name": "Capital One 360 Money Market"},
  {"id": 205, "account_name": "Citi Self Invest"}
]
Key validations:
- Each account from different bank
- Remove "Account" from accounts 201, 204, 205
- Remove "Visa Signature Card" from account 203
- Shorten "Bank of America" to "BofA"
- Shorten "Citibank" to "Citi"
- Use long_account_name when more descriptive
- All names unique, no masks needed"""
  },
  {
    "name": "test_example_1",
    "account_list": [
      {
        "id": 8525,
        "account_name": "Amex Checking",
        "bank_name": None,
        "long_account_name": "Amex Checking",
        "account_mask": "4507"
      },
      {
        "id": 8526,
        "account_name": "Citi® Checking",
        "bank_name": None,
        "long_account_name": "Citi® Checking",
        "account_mask": "3686"
      },
      {
        "id": 8527,
        "account_name": "Amex Gold",
        "bank_name": None,
        "long_account_name": "Amex Gold",
        "account_mask": "1587"
      },
      {
        "id": 8528,
        "account_name": "Citi Double Cash®",
        "bank_name": None,
        "long_account_name": "Citi Double Cash®",
        "account_mask": "3297"
      },
      {
        "id": 8529,
        "account_name": "Amex High Yield Savings",
        "bank_name": None,
        "long_account_name": "Amex High Yield Savings",
        "account_mask": "1676"
      },
      {
        "id": 8530,
        "account_name": "Citi® Savings",
        "bank_name": None,
        "long_account_name": "Citi® Savings",
        "account_mask": "2977"
      }
    ],
    "ideal_response": """Expected output:
[
  {"id": 8525, "account_name": "Amex Checking"},
  {"id": 8526, "account_name": "Citi Checking"},
  {"id": 8527, "account_name": "Amex Gold"},
  {"id": 8528, "account_name": "Citi Double Cash"},
  {"id": 8529, "account_name": "Amex High Yield Savings"},
  {"id": 8530, "account_name": "Citi Savings"}
]
Key validations:
- Remove special characters like ®
- All names unique, no masks needed"""
  },
  {
    "name": "test_example_2",
    "account_list": [
      {
        "id": 6799,
        "account_name": "Alliant Checking",
        "bank_name": None,
        "long_account_name": "Alliant Checking",
        "account_mask": "3149"
      },
      {
        "id": 6800,
        "account_name": "Alliant Credit",
        "bank_name": None,
        "long_account_name": "Alliant Credit",
        "account_mask": "4080"
      },
      {
        "id": 6801,
        "account_name": "Alliant Checking",
        "bank_name": None,
        "long_account_name": "Alliant Checking",
        "account_mask": "2182"
      },
      {
        "id": 6802,
        "account_name": "Alliant Credit",
        "bank_name": None,
        "long_account_name": "Alliant Credit",
        "account_mask": "5972"
      }
    ],
    "ideal_response": """Expected output:
[
  {"id": 6799, "account_name": "Alliant Checking **3149"},
  {"id": 6800, "account_name": "Alliant Credit **4080"},
  {"id": 6801, "account_name": "Alliant Checking **2182"},
  {"id": 6802, "account_name": "Alliant Credit **5972"}
]
Key validations:
- Duplicate names "Alliant Checking" and "Alliant Credit" -> add masks to all"""
  },
  {
    "name": "test_example_3",
    "account_list": [
      {
        "id": 4816,
        "account_name": "Visa Card 2746",
        "bank_name": "Truist",
        "long_account_name": "Truist Enjoy Cash - 3/2/1",
        "account_mask": "2746"
      },
      {
        "id": 4817,
        "account_name": "Mortgage 4503",
        "bank_name": "Truist",
        "long_account_name": "Mortgage",
        "account_mask": "4503"
      },
      {
        "id": 4818,
        "account_name": "HELOC Fixed Draw *5001",
        "bank_name": "Truist",
        "long_account_name": "HOME EQUITY LINE FIXED DRAW",
        "account_mask": "5001"
      },
      {
        "id": 4819,
        "account_name": "Home Equity Line *5998",
        "bank_name": "Truist",
        "long_account_name": "HOME EQUITY LINE OF CREDIT",
        "account_mask": "5998"
      },
      {
        "id": 4820,
        "account_name": "Home Equity Line Summary",
        "bank_name": "Truist",
        "long_account_name": "HOME EQUITY LINE OF CREDIT",
        "account_mask": "5998"
      }
    ],
    "ideal_response": """Expected output:
[
  {"id": 4816, "account_name": "Enjoy Cash - 3/2/1"},
  {"id": 4817, "account_name": "Mortgage"},
  {"id": 4818, "account_name": "Home Equity Line Fixed Draw"},
  {"id": 4819, "account_name": "Home Equity Line **5998"},
  {"id": 4820, "account_name": "Home Equity Line **5998"}
]
Key validations:
- All from same bank (Truist) -> omit bank name
- "Home Equity Line" is duplicate (from long_account_name) -> add masks
- Note: id 4820 has same mask as 4819, which is fine as they are the same account or just share the mask"""
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
      "name": "All Test Cases - Comprehensive",
      "tests": [0, 1, 2, 3, 4, 5, 6, 7]  # All 8 test cases
    },
    2: {
      "name": "Same Bank Scenarios",
      "tests": [3, 7]  # same_bank_multiple_account_types, test_example_3
    },
    3: {
      "name": "Different Banks Scenarios",
      "tests": [0, 4, 5]  # Tests with multiple banks
    },
    4: {
      "name": "Duplicate Handling",
      "tests": [2, 6, 7]  # duplicate_detection_and_masking, test_example_2, test_example_3
    },
    5: {
      "name": "Word Removal and Truncation",
      "tests": [1, 5]  # credit_cards_word_removal, test_example_1
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

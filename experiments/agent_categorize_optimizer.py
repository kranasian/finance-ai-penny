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

SYSTEM_PROMPT = """Map category descriptions to category IDs. Return JSON: {"category_id": integer} or {"category_id": -1} if no match.

**Matching Rules** (try in order):
1. Exact match (case-insensitive)
2. Word match: Check if any input word (length > 3) matches a category name or word within it. Always prefer more specific categories over general ones (e.g., "Entertainment" over "Leisure", "Upkeep" over "Shelter").
3. Handle variations:
   - Plural/singular forms (salaries→salary, businesses→business)
   - "and" matches "&" in category names ("car and fuel" → "Car & Fuel")
   - Hyphens match spaces ("side gig" → "Side-Gig")
4. Semantic match: Ignore descriptive words like "stores", "expenses", "payments" that don't change category meaning.
5. If no reasonable match: return -1

**Category List** (ID: Name):
-1: Uncategorized | 1: Meals | 2: Dining Out | 3: Delivered Food | 4: Groceries | 5: Leisure | 6: Entertainment | 7: Travel & Vacations
9: Bills | 10: Connectivity | 11: Insurance | 12: Taxes | 13: Service Fees | 14: Shelter | 15: Home | 16: Utilities | 17: Upkeep
18: Education | 19: Kids Activities | 20: Tuition | 21: Shopping | 22: Clothing | 23: Gadgets | 24: Kids | 8: Pets
25: Transport | 26: Car & Fuel | 27: Public Transit | 28: Health | 29: Medical & Pharmacy | 30: Gym & Wellness | 31: Personal Care
32: Donations & Gifts | 33: Miscellaneous | 45: Transfer | 36: Salary | 37: Side-Gig | 38: Business | 39: Interest
"""

class CategoryGrounderOptimizer:
  """Handles all Gemini API interactions for Category Grounding"""
  
  def __init__(self, model_name="gemini-flash-lite-latest"):
    """Initialize the Gemini agent with API configuration for Category Grounding"""
    # API Configuration
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
      raise ValueError("GEMINI_API_KEY environment variable is not set. Please set it in your .env file or environment.")
    self.client = genai.Client(api_key=api_key)
    
    # Model Configuration
    self.thinking_budget = 4096
    self.model_name = model_name
    
    # Generation Configuration Constants (matching account_renamer_optimizer.py)
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
  
  def generate_response(self, category_description: str) -> dict:
    """
    Generate a response using Gemini API for Category Grounding.
    
    Args:
      category_description: String description of the category to ground
      
    Returns:
      Dictionary containing response JSON and thought_summary
    """
    # Create request text with category description
    request_text = types.Part.from_text(text=f"input: {category_description}\n\noutput:")
    
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


def _run_test_with_logging(category_description: str, optimizer: CategoryGrounderOptimizer = None):
  """
  Internal helper function that runs a test with consistent logging.
  
  Args:
    category_description: Category description string to ground
    optimizer: Optional CategoryGrounderOptimizer instance. If None, creates a new one.
    
  Returns:
    The generated response dictionary
  """
  if optimizer is None:
    optimizer = CategoryGrounderOptimizer()
  
  # Print the input
  print("LLM INPUT:")
  print("-" * 80)
  print(f"Category Description: {category_description}")
  print("-" * 80)
  
  result = optimizer.generate_response(category_description)
  
  # Print the output
  print("LLM OUTPUT:")
  print("-" * 80)
  print(json.dumps(result["response"], indent=2) if result["response"] else "No response")
  print("-" * 80)
  
  return result


TEST_CASES = [
  {
    "name": "exact_and_word_match",
    "category_description": "shelter upkeep",
    "ideal_response": """Expected output:
{
  "category_id": 17
}
Key validations:
- "shelter upkeep" contains the word "upkeep" which matches category "Upkeep" (ID: 17)
- Should not match "Shelter" (ID: 14) as "Upkeep" is more specific"""
  },
  {
    "name": "semantic_matching_and_variations",
    "category_description": "home improvement stores",
    "ideal_response": """Expected output:
{
  "category_id": 17
}
Key validations:
- "home improvement" semantically relates to "Upkeep" (home maintenance/repair)
- "stores" is a descriptor that doesn't change the core category meaning"""
  },
  {
    "name": "plural_singular_and_multiword",
    "category_description": "groceries",
    "ideal_response": """Expected output:
{
  "category_id": 4
}
Key validations:
- "groceries" exactly matches "Groceries" (ID: 4) case-insensitively
- Should handle plural form correctly"""
  },
  {
    "name": "complex_phrase_and_synonyms",
    "category_description": "medical and pharmacy expenses",
    "ideal_response": """Expected output:
{
  "category_id": 29
}
Key validations:
- "medical and pharmacy" matches "Medical & Pharmacy" (ID: 29)
- Should handle "and" matching "&" in category name
- "expenses" is a descriptor that doesn't change the category"""
  },
  {
    "name": "special_characters_and_conversion",
    "category_description": "car and fuel",
    "ideal_response": """Expected output:
{
  "category_id": 26
}
Key validations:
- "car and fuel" should match "Car & Fuel" (ID: 26) exactly
- Should handle "and" → "&" conversion in matching
- Both words "car" and "fuel" are present in the category name"""
  },
  {
    "name": "income_category_matching",
    "category_description": "side gig",
    "ideal_response": """Expected output:
{
  "category_id": 37
}
Key validations:
- "side gig" should match "Side-Gig" (ID: 37)
- Should handle hyphen vs space variations ("Side-Gig" vs "side gig")
- Should not match general "Income" (ID: 46/47) when specific match exists"""
  },
  {
    "name": "invalid_unknown_category",
    "category_description": "xyz123unknown",
    "ideal_response": """Expected output:
{
  "category_id": -1
}
Key validations:
- Invalid/nonsensical category should return category_id: -1
- Should not attempt to force a match to any category"""
  },
  {
    "name": "disambiguation_specificity",
    "category_description": "entertainment",
    "ideal_response": """Expected output:
{
  "category_id": 6
}
Key validations:
- "entertainment" exactly matches "Entertainment" (ID: 6)
- Should prefer specific "Entertainment" over general "Leisure" (ID: 5)
- When multiple categories could match, choose the most specific one"""
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


def run_test(test_name_or_index_or_dict, optimizer: CategoryGrounderOptimizer = None):
  """
  Run a single test by name, index, or by passing test data directly.
  
  Args:
    test_name_or_index_or_dict: One of:
      - Test case name (str): e.g., "exact_and_word_match"
      - Test case index (int): e.g., 0
      - Test data dict: {"category_description": "...", "name": "..."}
    optimizer: Optional CategoryGrounderOptimizer instance. If None, creates a new one.
    
  Returns:
    The generated response dictionary, or None if test not found
  """
  # Check if it's a dict with test data
  if isinstance(test_name_or_index_or_dict, dict):
    if "category_description" in test_name_or_index_or_dict:
      test_name = test_name_or_index_or_dict.get("name", "custom_test")
      print(f"\n{'='*80}")
      print(f"Running test: {test_name}")
      print(f"{'-'*80}\n")
      
      result = _run_test_with_logging(
        test_name_or_index_or_dict["category_description"],
        optimizer
      )
      
      if test_name_or_index_or_dict.get("ideal_response", ""):
        print(f"Ideal response: {test_name_or_index_or_dict['ideal_response']}")
        print(f"{'='*80}\n")
      
      return result
    else:
      print(f"Invalid test dict: must contain 'category_description' key.")
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
    test_case["category_description"],
    optimizer
  )
  
  if test_case.get("ideal_response", ""):
    print(f"Ideal response: {test_case['ideal_response']}")
    print(f"{'='*80}\n")
  
  return result


def run_tests(test_names_or_indices_or_dicts, optimizer: CategoryGrounderOptimizer = None):
  """
  Run multiple tests by names, indices, or by passing test data directly.
  
  Args:
    test_names_or_indices_or_dicts: One of:
      - None: Run all tests from TEST_CASES
      - List of test case names (str), indices (int), or test data dicts
        Each dict should have: {"category_description": "...", "name": "..."}
    optimizer: Optional CategoryGrounderOptimizer instance. If None, creates a new one.
    
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


def test_with_inputs(category_description: str, optimizer: CategoryGrounderOptimizer = None):
  """
  Convenient method to test the category grounder optimizer with custom inputs.
  
  Args:
    category_description: Category description string to ground
    optimizer: Optional CategoryGrounderOptimizer instance. If None, creates a new one.
    
  Returns:
    The generated response dictionary
  """
  return _run_test_with_logging(category_description, optimizer)


def main(batch: int = None, test: str = None):
  """
  Main function to test the category grounder optimizer
  
  Args:
    batch: Batch number (1) to run all tests, or None to run a single test
    test: Test name, index, or None. If batch is provided, test is ignored.
      - Test name (str): e.g., "exact_and_word_match"
      - Test index (str): e.g., "0" (will be converted to int)
      - None: If batch is also None, prints available tests
  """
  optimizer = CategoryGrounderOptimizer()
  
  # Define test batches
  BATCHES = {
    1: {
      "name": "Category Grounder Test Cases - Batch 1",
      "tests": [0, 1, 2, 3]  # First 4 test cases
    },
    2: {
      "name": "Category Grounder Test Cases - Batch 2",
      "tests": [4, 5, 6, 7]  # Second 4 test cases
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
  python agent_categorize_optimizer.py --batch 1
  python agent_categorize_optimizer.py --test exact_and_word_match
  python agent_categorize_optimizer.py --test 0
  run_test("exact_and_word_match")
  run_tests([0, 1, 2, 3])
"""

if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser(description='Run category grounder optimizer tests in batches or individually')
  parser.add_argument('--batch', type=int, choices=[1, 2],
                      help='Batch number to run (1 or 2)')
  parser.add_argument('--test', type=str,
                      help='Test name or index to run individually (e.g., "exact_and_word_match" or "0")')
  args = parser.parse_args()
  main(batch=args.batch, test=args.test)

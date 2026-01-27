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

SYSTEM_PROMPT = """Generate concise descriptions of what each establishment sells or what services they provide. Focus exclusively on what customers pay for.

Format: Start with action verb ("sells", "offers", "provides") or service type description ("streaming service that offers", "fast-food restaurant that sells"), then list specific products/services customers purchase.

Guidelines:
- Concise: max 50 words, no unnecessary words or elaboration
- Specific: List concrete products/services first, use catch-all phrases ("and more", "and other X", "various X") only after listing key items
- Direct: Use factual statements only, avoid marketing language or promotional details
- Individual: Process each establishment separately
- Preserve: Copy the id exactly to output

<EXAMPLES>
input: [{"id": 387, "establishment": "Upwork"}, {"id": 24, "establishment": "Tesla Supercharger"}, {"id": 109, "establishment": "Chick-fil-A"}, {"id": 255, "establishment": "Little Lucca"}, {"id": 233, "establishment": "Steamgames.com"}]
output: [{"id": 387, "description": "platform for hiring freelancers in writing, design, development, marketing, and other professional services"}, {"id": 24, "description": "sells fast-charging stations for electric vehicles and subscriptions to in-car services"}, {"id": 109, "description": "fast-food restaurant that sells chicken dishes, breakfast items, salads, waffle fries, drinks, and desserts"}, {"id": 255, "description": "sells sandwiches, salads, breakfast, lunch, and dinner"}, {"id": 233, "description": "sells video games, downloadable content, in-game items, and subscriptions"}]

input: [{"id": 891, "establishment": "WM Supercenter"}, {"id": 52, "establishment": "Lowe's"}, {"id": 872, "establishment": "Girl Scouts"}, {"id": 234, "establishment": "Precision Auto Glass"}, {"id": 498, "establishment": "Salvation Army"}]
output: [{"id": 891, "description": "sells wide variety of goods including groceries, electronics, clothing, furniture, and more"}, {"id": 52, "description": "sells home improvement products such as tools, appliances, building materials, and garden supplies"}, {"id": 872, "description": "sells Girl Scout Cookies through local troops as a fundraiser"}, {"id": 234, "description": "installs auto glass for various types of vehicles"}, {"id": 498, "description": "sells donated clothing, furniture, and household items"}]
</EXAMPLES>
"""

class EstablishmentDescriptionOptimizer:
  """Handles all Gemini API interactions for Establishment Description"""
  
  def __init__(self, model_name="gemini-3-pro-preview"):
    """Initialize the Gemini agent with API configuration for Establishment Description"""
    # API Configuration
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
      raise ValueError("GEMINI_API_KEY environment variable is not set. Please set it in your .env file or environment.")
    self.client = genai.Client(api_key=api_key)
    
    # Model Configuration
    self.thinking_budget = 4096
    self.model_name = model_name
    
    # Generation Configuration Constants (from establishment_descriptions in establishment_description.py)
    self.temperature = 0.3
    self.top_p = 0.95
    self.top_k = 40
    self.max_output_tokens = 2048
    
    # Safety Settings
    self.safety_settings = [
      types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"),
      types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"),
      types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"),
      types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="OFF")
    ]
    
    # System Prompt
    self.system_prompt = SYSTEM_PROMPT
  
  def generate_response(self, establishment_list: list) -> dict:
    """
    Generate a response using Gemini API for Establishment Description.
    
    Args:
      establishment_list: List of establishment dictionaries with id and establishment
      
    Returns:
      Dictionary containing response JSON and thought_summary
    """
    # Create request text with establishment list as JSON
    request_text = types.Part.from_text(text=f"input: {json.dumps(establishment_list, indent=0)}\n\noutput:")
    
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


def _run_test_with_logging(establishment_list: list, optimizer: EstablishmentDescriptionOptimizer = None):
  """
  Internal helper function that runs a test with consistent logging.
  
  Args:
    establishment_list: List of establishment dictionaries
    optimizer: Optional EstablishmentDescriptionOptimizer instance. If None, creates a new one.
    
  Returns:
    The generated response dictionary
  """
  if optimizer is None:
    optimizer = EstablishmentDescriptionOptimizer()
  
  # Print the input
  print("LLM INPUT:")
  print("-" * 80)
  print(json.dumps(establishment_list, indent=2))
  print("-" * 80)
  
  result = optimizer.generate_response(establishment_list)
  
  # Print the output
  print("LLM OUTPUT:")
  print("-" * 80)
  print(json.dumps(result["response"], indent=2) if result["response"] else "No response")
  print("-" * 80)
  
  return result


TEST_CASES = [
  {
    "name": "basic_establishments",
    "establishment_list": [
      {
        "id": 1,
        "establishment": "Starbucks"
      },
      {
        "id": 2,
        "establishment": "Amazon"
      },
      {
        "id": 3,
        "establishment": "Target"
      },
      {
        "id": 4,
        "establishment": "McDonald's"
      }
    ],
    "ideal_response": """Expected output:
[
  {"id": 1, "description": "sells coffee, espresso drinks, teas, pastries, sandwiches, and other beverages and food items"},
  {"id": 2, "description": "sells wide variety of products online including electronics, books, clothing, home goods, and more"},
  {"id": 3, "description": "sells clothing, electronics, home goods, groceries, and various consumer products"},
  {"id": 4, "description": "fast-food restaurant that sells burgers, fries, chicken nuggets, breakfast items, drinks, and desserts"}
]
Key validations:
- Each description should be concise (under 50 words)
- Descriptions should list what customers pay for
- Each establishment treated individually
- IDs preserved in output"""
  },
  {
    "name": "online_services_and_education",
    "establishment_list": [
      {
        "id": 101,
        "establishment": "Netflix"
      },
      {
        "id": 102,
        "establishment": "Coursera"
      },
      {
        "id": 103,
        "establishment": "Spotify"
      },
      {
        "id": 104,
        "establishment": "LinkedIn Premium"
      }
    ],
    "ideal_response": """Expected output:
[
  {"id": 101, "description": "streaming service that offers movies, TV shows, documentaries, and original content subscriptions"},
  {"id": 102, "description": "online learning platform that offers courses, specializations, and degrees from universities and companies"},
  {"id": 103, "description": "music streaming service that offers songs, podcasts, playlists, and premium subscriptions"},
  {"id": 104, "description": "professional networking platform that offers premium subscriptions for job searching, networking, and career development"}
]
Key validations:
- Online services should describe subscription or service offerings
- Education platforms should mention courses and learning content
- Each service treated individually with specific offerings"""
  },
  {
    "name": "retail_and_specialty_stores",
    "establishment_list": [
      {
        "id": 201,
        "establishment": "Home Depot"
      },
      {
        "id": 202,
        "establishment": "Sephora"
      },
      {
        "id": 203,
        "establishment": "Best Buy"
      },
      {
        "id": 204,
        "establishment": "Whole Foods Market"
      }
    ],
    "ideal_response": """Expected output:
[
  {"id": 201, "description": "sells home improvement products such as tools, appliances, building materials, and garden supplies"},
  {"id": 202, "description": "sells cosmetics, skincare products, fragrances, makeup, and beauty accessories"},
  {"id": 203, "description": "sells electronics, computers, appliances, mobile devices, and tech accessories"},
  {"id": 204, "description": "sells organic groceries, prepared foods, natural products, and specialty items"}
]
Key validations:
- Retail stores should list product categories
- Specialty stores should mention their specific focus
- Descriptions should be diverse and specific"""
  },
  {
    "name": "services_and_utilities",
    "establishment_list": [
      {
        "id": 301,
        "establishment": "Uber"
      },
      {
        "id": 302,
        "establishment": "PG&E"
      },
      {
        "id": 303,
        "establishment": "AT&T"
      },
      {
        "id": 304,
        "establishment": "YMCA"
      }
    ],
    "ideal_response": """Expected output:
[
  {"id": 301, "description": "ride-sharing and food delivery service that connects passengers with drivers and restaurants"},
  {"id": 302, "description": "utility company that provides electricity and natural gas services for residential and commercial customers"},
  {"id": 303, "description": "telecommunications company that provides mobile phone, internet, and TV services"},
  {"id": 304, "description": "community organization that offers gym memberships, fitness classes, swimming, childcare, and recreational programs"}
]
Key validations:
- Service companies should describe what services they provide
- Utilities should mention specific utility types
- Community organizations should list programs and services"""
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


def run_test(test_name_or_index_or_dict, optimizer: EstablishmentDescriptionOptimizer = None):
  """
  Run a single test by name, index, or by passing test data directly.
  
  Args:
    test_name_or_index_or_dict: One of:
      - Test case name (str): e.g., "basic_establishments"
      - Test case index (int): e.g., 0
      - Test data dict: {"establishment_list": [...], "name": "..."}
    optimizer: Optional EstablishmentDescriptionOptimizer instance. If None, creates a new one.
    
  Returns:
    The generated response dictionary, or None if test not found
  """
  # Check if it's a dict with test data
  if isinstance(test_name_or_index_or_dict, dict):
    if "establishment_list" in test_name_or_index_or_dict:
      test_name = test_name_or_index_or_dict.get("name", "custom_test")
      print(f"\n{'='*80}")
      print(f"Running test: {test_name}")
      print(f"{'-'*80}\n")
      
      result = _run_test_with_logging(
        test_name_or_index_or_dict["establishment_list"],
        optimizer
      )
      
      if test_name_or_index_or_dict.get("ideal_response", ""):
        print(f"Ideal response: {test_name_or_index_or_dict['ideal_response']}")
        print(f"{'='*80}\n")
      
      return result
    else:
      print(f"Invalid test dict: must contain 'establishment_list' key.")
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
    test_case["establishment_list"],
    optimizer
  )
  
  if test_case.get("ideal_response", ""):
    print(f"Ideal response: {test_case['ideal_response']}")
    print(f"{'='*80}\n")
  
  return result


def run_tests(test_names_or_indices_or_dicts, optimizer: EstablishmentDescriptionOptimizer = None):
  """
  Run multiple tests by names, indices, or by passing test data directly.
  
  Args:
    test_names_or_indices_or_dicts: One of:
      - None: Run all tests from TEST_CASES
      - List of test case names (str), indices (int), or test data dicts
        Each dict should have: {"establishment_list": [...], "name": "..."}
    optimizer: Optional EstablishmentDescriptionOptimizer instance. If None, creates a new one.
    
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


def test_with_inputs(establishment_list: list, optimizer: EstablishmentDescriptionOptimizer = None):
  """
  Convenient method to test the establishment description optimizer with custom inputs.
  
  Args:
    establishment_list: List of establishment dictionaries
    optimizer: Optional EstablishmentDescriptionOptimizer instance. If None, creates a new one.
    
  Returns:
    The generated response dictionary
  """
  return _run_test_with_logging(establishment_list, optimizer)


def main(batch: int = None, test: str = None):
  """
  Main function to test the establishment description optimizer
  
  Args:
    batch: Batch number (1) to run all tests, or None to run a single test
    test: Test name, index, or None. If batch is provided, test is ignored.
      - Test name (str): e.g., "basic_establishments"
      - Test index (str): e.g., "0" (will be converted to int)
      - None: If batch is also None, prints available tests
  """
  optimizer = EstablishmentDescriptionOptimizer()
  
  # Define test batches
  BATCHES = {
    1: {
      "name": "Establishment Description Test Cases",
      "tests": [0, 1, 2, 3]  # All 4 test cases
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
  python establishment_description_optimizer.py --batch 1
  python establishment_description_optimizer.py --test basic_establishments
  python establishment_description_optimizer.py --test 0
  run_test("basic_establishments")
  run_tests([0, 1, 2, 3])
"""

if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser(description='Run establishment description optimizer tests in batches or individually')
  parser.add_argument('--batch', type=int, choices=[1],
                      help='Batch number to run (1)')
  parser.add_argument('--test', type=str,
                      help='Test name or index to run individually (e.g., "basic_establishments" or "0")')
  args = parser.parse_args()
  main(batch=args.batch, test=args.test)

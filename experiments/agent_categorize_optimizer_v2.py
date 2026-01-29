from google import genai
from google.genai import types
import os
import sys
from dotenv import load_dotenv

# Add the parent directory to the path so we can import the tools
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
  sys.path.insert(0, parent_dir)

from penny.tool_funcs.update_transaction_category_or_create_category_rules import create_category_rules
from penny.tool_funcs.date_utils import (
  get_date,
  get_start_of_month,
  get_end_of_month,
  get_start_of_year,
  get_end_of_year,
  get_start_of_week,
  get_end_of_week,
  get_after_periods,
  get_date_string,
)
from datetime import datetime

# Load environment variables
load_dotenv()

SYSTEM_PROMPT = """You generate Python code only. Output format: exactly one ```python block containing only `def execute_plan() -> tuple[bool, str]:` and its body (valid Python, no parse errors). No other text. For any rule-like request, output the code block; do not respond with reasoning only.

Input: **Last User Request** and **Past Conversation** (context only).

Contract: Define exactly one function: `def execute_plan() -> tuple[bool, str]:` and return (bool, str). Single rule -> return create_category_rules(...); multiple -> outputs=[]; append each msg; return (all(ok), chr(10).join(outputs)); unintelligible only -> return (False, "brief guidance"). The str must be create_category_rules' message(s) or one brief guidance; no extra commentary.

Rules:
1. Extract from Last User Request (and Past Conversation): merchant/name, amounts, dates, target category.
2. Multiple merchants or rule clauses: one create_category_rules per merchant/clause; append msg to outputs; return (all(ok), chr(10).join(outputs)). No early-exit.
3. rules_dict keys (AND): name_contains OR name_eq (not both in one rule), date_greater_than_or_equal_to, date_less_than_or_equal_to (get_date_string for YYYY-MM-DD), amount_greater_than_or_equal_to, amount_less_than_or_equal_to.
4. Map category wording to OFFICIAL_CATEGORIES slug; if not exact, map to closest (e.g. aviation -> transportation_public, explosions -> leisure_entertainment). Do not return (False, "Invalid category").
5. Unintelligible only when no discernible rule (e.g. pure symbols): return (False, "Brief guidance.").

Date helpers: datetime, get_date(y,m,d), get_start_of_month, get_end_of_month, get_start_of_year, get_end_of_year, get_start_of_week, get_end_of_week, get_after_periods(date, granularity, count) (granularity: "monthly"|"weekly"|"yearly", count e.g. -1), get_date_string(date) -> "YYYY-MM-DD".

<OFFICIAL_CATEGORIES>
income: income_salary, income_sidegig, income_business, income_interest
meals: meals_groceries, meals_dining_out, meals_delivered_food
leisure: leisure_entertainment, leisure_travel
bills: bills_connectivity, bills_insurance, bills_tax, bills_service_fees
shelter: shelter_home, shelter_utilities, shelter_upkeep
education: education_kids_activities, education_tuition
shopping: shopping_clothing, shopping_gadgets, shopping_kids, shopping_pets
transportation: transportation_public, transportation_car
health: health_medical_pharmacy, health_gym_wellness, health_personal_care
donations_gifts, uncategorized, transfers, miscellaneous
</OFFICIAL_CATEGORIES>

Only callable: create_category_rules(rules_dict: dict, new_category: str) -> tuple[bool, str]

<EXAMPLES>

Last User Request: Safeway purchases should be groceries
Past Conversation: None
```python
def execute_plan() -> tuple[bool, str]:
    rules_dict = {"name_contains": "safeway"}
    return create_category_rules(rules_dict=rules_dict, new_category="meals_groceries")
```

Last User Request: Shell fuel from last 3 months under $50 are for car
Past Conversation: None
```python
def execute_plan() -> tuple[bool, str]:
    today = datetime.now()
    start = get_date_string(get_after_periods(today, granularity="monthly", count=-3))
    rules_dict = {"name_contains": "shell", "date_greater_than_or_equal_to": start, "amount_less_than_or_equal_to": 50}
    return create_category_rules(rules_dict=rules_dict, new_category="transportation_car")
```

Last User Request: Kroger and Trader Joe's should be groceries
Past Conversation: None
```python
def execute_plan() -> tuple[bool, str]:
    outputs = []
    cat = "meals_groceries"
    ok1, msg1 = create_category_rules(rules_dict={"name_contains": "kroger"}, new_category=cat)
    outputs.append(msg1)
    ok2, msg2 = create_category_rules(rules_dict={"name_contains": "trader joe"}, new_category=cat)
    outputs.append(msg2)
    return (ok1 and ok2, chr(10).join(outputs))
```

Last User Request: tea in name should be travel and yoga* after 6/1 should be wellness
Past Conversation: None
```python
def execute_plan() -> tuple[bool, str]:
    outputs = []
    ok1, msg1 = create_category_rules(rules_dict={"name_contains": "tea"}, new_category="leisure_travel")
    outputs.append(msg1)
    date_61 = get_date_string(get_date(2025, 6, 1))
    ok2, msg2 = create_category_rules(
        rules_dict={"name_contains": "yoga", "date_greater_than_or_equal_to": date_61},
        new_category="health_gym_wellness"
    )
    outputs.append(msg2)
    return (ok1 and ok2, chr(10).join(outputs))
```

Last User Request: !!!@@@ gibberish
Past Conversation: None
```python
def execute_plan() -> tuple[bool, str]:
    return False, "Provide a valid categorization rule (e.g. 'X purchases should be Y')."
```
</EXAMPLES>
"""

class CategoryGrounderOptimizer:
  """Handles all Gemini API interactions for Categorization Rule Creation"""
  
  def __init__(self, model_name="gemini-flash-lite-latest"):
    """Initialize the Gemini agent with API configuration for categorization rule creation"""
    # API Configuration
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
      raise ValueError("GEMINI_API_KEY environment variable is not set. Please set it in your .env file or environment.")
    self.client = genai.Client(api_key=api_key)
    
    # Model Configuration
    self.thinking_budget = 0
    self.model_name = model_name
    
    # Generation Configuration Constants (matching goal_agent_optimizer.py)
    self.temperature = 0.6
    self.top_p = 0.95
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

  
  def generate_response(self, last_user_request: str, previous_conversation: str) -> str:
    """
    Generate a response using Gemini API for categorization rule creation planning.
    
    Args:
      last_user_request: The last user request as a string
      previous_conversation: The previous conversation as a string
      
    Returns:
      Generated code as a string
    """
    # Create request text with Last User Request and Past Conversation
    request_text = types.Part.from_text(text=f"""**Last User Request**: {last_user_request}

**Past Conversation**:

{previous_conversation}

output:""")
    
    # Create content and configuration
    contents = [types.Content(role="user", parts=[request_text])]
    
    generate_content_config = types.GenerateContentConfig(
      temperature=self.temperature,
      top_p=self.top_p,
      max_output_tokens=self.max_output_tokens,
      safety_settings=self.safety_settings,
      system_instruction=[types.Part.from_text(text=self.system_prompt)],
      thinking_config=types.ThinkingConfig(
        thinking_budget=self.thinking_budget,
        include_thoughts=False
      ),
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
      
    #   # Extract thought summary from chunk
    #   if hasattr(chunk, 'candidates') and chunk.candidates:
    #     for candidate in chunk.candidates:
    #       # Extract thought summary from parts (per Gemini API docs)
    #       # Check part.thought boolean to identify thought parts
    #       if hasattr(candidate, 'content') and candidate.content:
    #         if hasattr(candidate.content, 'parts') and candidate.content.parts:
    #           for part in candidate.content.parts:
    #             # Check if this part is a thought summary (per documentation)
    #             if hasattr(part, 'thought') and part.thought:
    #               if hasattr(part, 'text') and part.text:
    #                 # Accumulate thought summary text (for streaming, it may come in chunks)
    #                 if thought_summary:
    #                   thought_summary += part.text
    #                 else:
    #                   thought_summary = part.text
    
    # Display thought summary if available
    if thought_summary:
      print("\n" + "-" * 80)
      print("THOUGHT SUMMARY:")
      print("-" * 80)
      print(thought_summary.strip())
      print("-" * 80 + "\n")
    
    return output_text
  
  
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


def extract_python_code(text: str) -> str:
    """Extract Python code from generated response (look for ```python blocks).
    
    Args:
        text: The generated response containing Python code
        
    Returns:
        str: Extracted Python code
    """
    code_start = text.find("```python")
    if code_start != -1:
        code_start += len("```python")
        code_end = text.find("```", code_start)
        if code_end != -1:
            return text[code_start:code_end].strip()
        else:
            # No closing ``` found, use the entire response as code
            return text[code_start:].strip()
    else:
        # No ```python found, try to use the entire response as code
        return text.strip()


def _run_test_with_logging(last_user_request: str, previous_conversation: str, optimizer: CategoryGrounderOptimizer = None):
  """
  Internal helper function that runs a test with consistent logging.
  
  Args:
    last_user_request: The last user request as a string
    previous_conversation: The previous conversation as a string
    optimizer: Optional CategoryGrounderOptimizer instance. If None, creates a new one.
    
  Returns:
    The generated response string
  """
  if optimizer is None:
    optimizer = CategoryGrounderOptimizer()
  
  # Construct LLM input
  llm_input = f"""**Last User Request**: {last_user_request}

**Past Conversation**:

{previous_conversation}

output:"""
  
  # Print the input
  print("=" * 80)
  print("LLM INPUT:")
  print("=" * 80)
  print(llm_input)
  print("=" * 80)
  print()
  
  result = optimizer.generate_response(last_user_request, previous_conversation)
  
  # Print the output
  print("=" * 80)
  print("LLM OUTPUT:")
  print("=" * 80)
  print(result)
  print("=" * 80)
  print()
  
  # Extract and execute the generated code
  code = extract_python_code(result)
  
  if code:
    print("=" * 80)
    print("EXECUTING GENERATED CODE:")
    print("=" * 80)
    try:
      # Create wrapper for create_category_rules; provide date helpers and datetime
      def wrapped_create_category_rules(rules_dict, new_category):
        print(f"\n[FUNCTION CALL] create_category_rules")
        print(f"  rules_dict: {rules_dict}")
        print(f"  new_category: {new_category}")
        result = create_category_rules(rules_dict=rules_dict, new_category=new_category)
        print(f"  [RETURN] success: {result[0]}")
        print(f"  [RETURN] output: {result[1]}")
        return result  # Returns tuple[bool, str]

      namespace = {
        'create_category_rules': wrapped_create_category_rules,
        'datetime': datetime,
        'get_date': get_date,
        'get_start_of_month': get_start_of_month,
        'get_end_of_month': get_end_of_month,
        'get_start_of_year': get_start_of_year,
        'get_end_of_year': get_end_of_year,
        'get_start_of_week': get_start_of_week,
        'get_end_of_week': get_end_of_week,
        'get_after_periods': get_after_periods,
        'get_date_string': get_date_string,
      }
      
      # Execute the code
      exec(code, namespace)
      
      # Call execute_plan if it exists
      if 'execute_plan' in namespace:
        print("\n" + "=" * 80)
        print("Calling execute_plan()...")
        print("=" * 80)
        execution_result = namespace['execute_plan']()
        print("\n" + "=" * 80)
        print("EXECUTION RESULT:")
        print("=" * 80)
        print(f"  success: {execution_result[0]}")
        print(f"  output: {execution_result[1]}")
        print("=" * 80)
        print()
      else:
        print("Warning: execute_plan() function not found in generated code")
        print("=" * 80)
        execution_result = None
    except Exception as e:
      print(f"Error executing generated code: {str(e)}")
      import traceback
      print(traceback.format_exc())
      print("=" * 80)
      execution_result = None
  else:
    execution_result = None
  
  return result


# Test cases list - add new tests here instead of creating new functions
TEST_CASES = [
  {
    "name": "simple_categorization_rule",
    "last_user_request": "Walmart purchases should be groceries",
    "previous_conversation": "",
    "ideal_response": """Expected output:
```python
def execute_plan() -> tuple[bool, str]:
    rules_dict = {"name_contains": "walmart"}
    return create_category_rules(rules_dict=rules_dict, new_category="meals_groceries")
```
Key validations:
- Creates rule with name_contains "walmart", new_category meals_groceries"""
  },
  {
    "name": "amount_filter_rule",
    "last_user_request": "schneider greater than $400 to wall construction",
    "previous_conversation": "",
    "ideal_response": """Expected output:
```python
def execute_plan() -> tuple[bool, str]:
    rules_dict = {"name_contains": "schneider", "amount_greater_than_or_equal_to": 400}
    return create_category_rules(rules_dict=rules_dict, new_category="shelter_upkeep")
```
Key validations:
- name_contains schneider, amount_gte 400, new_category shelter_upkeep"""
  },
  {
    "name": "multiple_merchants_rule",
    "last_user_request": "John's, Henries and moms and pops needs to all be in supermarkets",
    "previous_conversation": "",
    "ideal_response": """Expected output:
```python
def execute_plan() -> tuple[bool, str]:
    outputs = []
    cat = "meals_groceries"
    ok1, msg1 = create_category_rules(rules_dict={"name_contains": "john's"}, new_category=cat)
    outputs.append(msg1)
    ok2, msg2 = create_category_rules(rules_dict={"name_contains": "henries"}, new_category=cat)
    outputs.append(msg2)
    ok3, msg3 = create_category_rules(rules_dict={"name_contains": "moms and pops"}, new_category=cat)
    outputs.append(msg3)
    return (ok1 and ok2 and ok3, chr(10).join(outputs))
```
Key validations:
- Creates 3 rules: one per merchant (John's, Henries, moms and pops). new_category meals_groceries for supermarkets. No early exit; return (all(successes), chr(10).join(outputs))."""
  },
  {
    "name": "date_filter_rule",
    "last_user_request": "Barnes & Noble and Bed Bath & Beyond before last week must be put on shopping",
    "previous_conversation": "",
    "ideal_response": """Expected output:
```python
def execute_plan() -> tuple[bool, str]:
    outputs = []
    today = datetime.now()
    last_week = get_date_string(get_after_periods(today, granularity="weekly", count=-1))
    cat = "shopping_clothing"
    ok1, msg1 = create_category_rules(rules_dict={"name_contains": "barnes", "date_less_than_or_equal_to": last_week}, new_category=cat)
    outputs.append(msg1)
    ok2, msg2 = create_category_rules(rules_dict={"name_contains": "bed bath", "date_less_than_or_equal_to": last_week}, new_category=cat)
    outputs.append(msg2)
    return (ok1 and ok2, chr(10).join(outputs))
```
Key validations:
- Creates 2 rules: one per merchant (Barnes & Noble, Bed Bath & Beyond), same date_lte (before last week) and new_category shopping_clothing. No early exit; return (all(successes), chr(10).join(outputs))."""
  },
  {
    "name": "unintelligible_input",
    "last_user_request": "+&-$|(){}#@!%^&*_=[]:;,.<>/?`~",
    "previous_conversation": "",
    "ideal_response": """Expected output:
```python
def execute_plan() -> tuple[bool, str]:
    return False, "The input appears to be unintelligible. Please provide a valid categorization rule request (e.g., 'Walmart purchases should be groceries' or 'always categorize Costco as groceries')."
```
Key validations:
- Returns False with clarification message
- Does not attempt to create rules from unintelligible input
- Provides helpful guidance on valid request format"""
  },
  {
    "name": "regex_pattern_rule",
    "last_user_request": "coffee in the establishment name should be in aviation and if business like cricket* after 8/24 will be in explosions",
    "previous_conversation": "",
    "ideal_response": """Expected output:
```python
def execute_plan() -> tuple[bool, str]:
    outputs = []
    # Rule 1: coffee in name -> aviation (map to transportation_public)
    ok1, msg1 = create_category_rules(rules_dict={"name_contains": "coffee"}, new_category="transportation_public")
    outputs.append(msg1)
    # Rule 2: cricket in name, on or after 8/24 -> explosions (map to leisure_entertainment)
    date_824 = get_date_string(get_date(2025, 8, 24))
    ok2, msg2 = create_category_rules(
        rules_dict={"name_contains": "cricket", "date_greater_than_or_equal_to": date_824},
        new_category="leisure_entertainment"
    )
    outputs.append(msg2)
    return (ok1 and ok2, chr(10).join(outputs))
```
Key validations:
- Creates 2 rules: (1) name_contains coffee -> aviation mapped to transportation_public; (2) name_contains cricket + date_gte 8/24 -> explosions mapped to leisure_entertainment. No early exit; return (all(successes), chr(10).join(outputs))."""
  },
  {
    "name": "complex_multi_condition_rule",
    "last_user_request": "Costco purchases over $100 and under $500 from last month should be groceries",
    "previous_conversation": "",
    "ideal_response": """Expected output:
```python
def execute_plan() -> tuple[bool, str]:
    today = datetime.now()
    start_last_month = get_date_string(get_after_periods(today, granularity="monthly", count=-1))
    rules_dict = {
        "name_contains": "costco",
        "amount_greater_than_or_equal_to": 100,
        "amount_less_than_or_equal_to": 500,
        "date_greater_than_or_equal_to": start_last_month
    }
    return create_category_rules(rules_dict=rules_dict, new_category="meals_groceries")
```
Key validations:
- name_contains, amount_gte, amount_lte, date_gte, new_category meals_groceries"""
  },
  {
    "name": "income_category_rule",
    "last_user_request": "PayPal transfers from employer are salary",
    "previous_conversation": "",
    "ideal_response": """Expected output:
```python
def execute_plan() -> tuple[bool, str]:
    rules_dict = {"name_contains": "paypal"}
    return create_category_rules(rules_dict=rules_dict, new_category="income_salary")
```
Key validations:
- name_contains paypal, new_category income_salary"""
  },
  {
    "name": "rule_creation_with_context",
    "last_user_request": "Categorize all my Starbucks transactions as dining out",
    "previous_conversation": """User: How much am I spending at Starbucks?
Assistant: You've spent $120 at Starbucks over the last 3 months across 15 transactions.""",
    "ideal_response": """Expected output:
```python
def execute_plan() -> tuple[bool, str]:
    rules_dict = {"name_contains": "starbucks"}
    return create_category_rules(rules_dict=rules_dict, new_category="meals_dining_out")
```
Key validations:
- Uses Past Conversation for context; name_contains starbucks, new_category meals_dining_out"""
  },
  {
    "name": "clarification_answer_rule",
    "last_user_request": "Yes, create a rule for all Target purchases to be shopping",
    "previous_conversation": """User: I want to categorize Target purchases
Assistant: I need more information to create the categorization rule. What category should Target purchases be assigned to? For example, are they groceries, shopping, or another category?""",
    "ideal_response": """Expected output:
```python
def execute_plan() -> tuple[bool, str]:
    rules_dict = {"name_contains": "target"}
    return create_category_rules(rules_dict=rules_dict, new_category="shopping_clothing")
```
Key validations:
- Responds to clarification from Past Conversation; name_contains target, new_category shopping (shopping_clothing)"""
  },
  {
    "name": "always_keyword_rule",
    "last_user_request": "Always categorize Amazon purchases as shopping",
    "previous_conversation": "",
    "ideal_response": """Expected output:
```python
def execute_plan() -> tuple[bool, str]:
    rules_dict = {"name_contains": "amazon"}
    return create_category_rules(rules_dict=rules_dict, new_category="shopping_clothing")
```
Key validations:
- name_contains amazon, new_category shopping (shopping_clothing). "Always" implies create_category_rules (future rule)."""
  },
  {
    "name": "amount_range_and_date_rule",
    "last_user_request": "Gas station purchases between $20 and $80 from this year should be transportation",
    "previous_conversation": "",
    "ideal_response": """Expected output:
```python
def execute_plan() -> tuple[bool, str]:
    today = datetime.now()
    start_of_year = get_date_string(get_start_of_year(today))
    rules_dict = {
        "name_contains": "gas",
        "amount_greater_than_or_equal_to": 20,
        "amount_less_than_or_equal_to": 80,
        "date_greater_than_or_equal_to": start_of_year
    }
    return create_category_rules(rules_dict=rules_dict, new_category="transportation_car")
```
Key validations:
- name_contains gas, amount range 20-80, date_gte start of year, new_category transportation_car"""
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
      - Test case name (str): e.g., "simple_categorization_rule"
      - Test case index (int): e.g., 0
      - Test data dict: {"last_user_request": "...", "previous_conversation": "..."}
    optimizer: Optional CategoryGrounderOptimizer instance. If None, creates a new one.
    
  Returns:
    The generated response string, or None if test not found
  """
  # Check if it's a dict with test data
  if isinstance(test_name_or_index_or_dict, dict):
    if "last_user_request" in test_name_or_index_or_dict:
      test_name = test_name_or_index_or_dict.get("name", "custom_test")
      print(f"\n{'='*80}")
      print(f"Running test: {test_name}")
      print(f"{'-'*80}\n")
      
      result = _run_test_with_logging(
        test_name_or_index_or_dict["last_user_request"],
        test_name_or_index_or_dict.get("previous_conversation", ""),
        optimizer
      )
      
      # Display ideal response if available
      if test_name_or_index_or_dict.get("ideal_response", ""):
        print("\n" + "=" * 80)
        print("IDEAL RESPONSE:")
        print("=" * 80)
        print(test_name_or_index_or_dict["ideal_response"])
        print("=" * 80 + "\n")
      
      return result
    else:
      print(f"Invalid test dict: must contain 'last_user_request' key.")
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
    test_case["last_user_request"],
    test_case["previous_conversation"],
    optimizer
  )
  
  # Display ideal response if available
  if test_case.get("ideal_response", ""):
    print("\n" + "=" * 80)
    print("IDEAL RESPONSE:")
    print("=" * 80)
    print(test_case["ideal_response"])
    print("=" * 80 + "\n")
  
  return result


def run_tests(test_names_or_indices_or_dicts, optimizer: CategoryGrounderOptimizer = None):
  """
  Run multiple tests by names, indices, or by passing test data directly.
  
  Args:
    test_names_or_indices_or_dicts: One of:
      - None: Run all tests from TEST_CASES
      - List of test case names (str), indices (int), or test data dicts
        Each dict should have: {"last_user_request": "...", "previous_conversation": "..."}
    optimizer: Optional CategoryGrounderOptimizer instance. If None, creates a new one.
    
  Returns:
    List of generated response strings
  """
  if test_names_or_indices_or_dicts is None:
    # Run all tests
    test_names_or_indices_or_dicts = list(range(len(TEST_CASES)))
  
  results = []
  for test_item in test_names_or_indices_or_dicts:
    result = run_test(test_item, optimizer)
    results.append(result)
  
  return results


def test_with_inputs(last_user_request: str, previous_conversation: str, optimizer: CategoryGrounderOptimizer = None):
  """
  Convenient method to test the categorization rule creation optimizer with custom inputs.
  
  Args:
    last_user_request: The last user request as a string
    previous_conversation: The previous conversation as a string
    optimizer: Optional CategoryGrounderOptimizer instance. If None, creates a new one.
    
  Returns:
    The generated response string
  """
  return _run_test_with_logging(last_user_request, previous_conversation, optimizer)


def main(batch: int = None, test: str = None):
  """
  Main function to test the categorization rule creation optimizer
  
  Args:
    batch: Batch number (1 or 2) to run all tests, or None to run a single test
    test: Test name, index, or None. If batch is provided, test is ignored.
      - Test name (str): e.g., "simple_categorization_rule"
      - Test index (str): e.g., "0" (will be converted to int)
      - None: If batch is also None, prints available tests
  """
  optimizer = CategoryGrounderOptimizer()
  
  # Define test batches
  BATCHES = {
    1: {
      "name": "Categorization Rule Creation Test Cases - Batch 1",
      "tests": [0, 1, 2, 3]  # Simple rule, amount filter, multiple merchants, date filter
    },
    2: {
      "name": "Categorization Rule Creation Test Cases - Batch 2",
      "tests": [4, 5, 6, 7]  # Unintelligible, regex pattern, complex multi-condition, income category
    },
    3: {
      "name": "Categorization Rule Creation Test Cases - Batch 3",
      "tests": [8, 9, 10, 11]  # Rule with context, clarification answer, always keyword, amount range and date
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
    print("  Run a batch: --batch <1, 2, or 3>")
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
  python agent_categorize_optimizer.py --test simple_categorization_rule
  python agent_categorize_optimizer.py --test 0
  run_test("simple_categorization_rule")
  run_tests([0, 1, 2, 3])
"""

if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser(description='Run categorization rule creation optimizer tests in batches or individually')
  parser.add_argument('--batch', type=int, choices=[1, 2, 3],
                      help='Batch number to run (1, 2, or 3)')
  parser.add_argument('--test', type=str,
                      help='Test name or index to run individually (e.g., "simple_categorization_rule" or "0")')
  args = parser.parse_args()
  main(batch=args.batch, test=args.test)

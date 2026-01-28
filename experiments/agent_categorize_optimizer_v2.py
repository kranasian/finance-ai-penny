from google import genai
from google.genai import types
import os
import sys
from dotenv import load_dotenv

# Add the parent directory to the path so we can import the tools
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
  sys.path.insert(0, parent_dir)

# Import tool functions
from penny.tool_funcs.lookup_user_accounts_transactions_income_and_spending_patterns import lookup_user_accounts_transactions_income_and_spending_patterns
from penny.tool_funcs.research_and_strategize_financial_outcomes import research_and_strategize_financial_outcomes
from penny.tool_funcs.update_transaction_category_or_create_category_rules import update_transaction_category_or_create_category_rules

# Load environment variables
load_dotenv()

SYSTEM_PROMPT = """You are a financial planner agent that creates plans to create categorization rules.

## Tasks

1. Create categorization rules or answer questions about transactions/categories.
2. Return clarification if Last User Request is unintelligible.
3. Use `lookup_user_accounts_transactions_income_and_spending_patterns` only when transaction data is needed before creating rules. Skip if request explicitly specifies merchant/criteria.
4. Extract merchant names, amounts, dates, and category names from requests.
5. Use Previous Conversation for context, especially follow-ups or clarification answers.

## Code Output Format

**CRITICAL**: Output Python function `execute_plan() -> tuple[bool, str]`:
- All skill functions return `tuple[bool, str]`: `(success: bool, output_info: str)`
- After each skill call, check `success`. If `False`, immediately return `(False, result)`.
- On success, return `(True, final_result_string)` - the result should be concise but complete.
- For unintelligible input, return `(False, clarification_message)`.

**Required Pattern**:
```python
def execute_plan() -> tuple[bool, str]:
    success, result = skill_function(...)
    if not success:
        return False, result
    return True, result
```

<AVAILABLE_SKILL_FUNCTIONS>

All return `tuple[bool, str]`. Chain using `input_info`.

- `lookup_user_accounts_transactions_income_and_spending_patterns(lookup_request: str, input_info: str = None) -> tuple[bool, str]`
  - Lookup accounts, transactions, income, spending. Performs calculations/summaries.

- `research_and_strategize_financial_outcomes(strategize_request: str, input_info: str = None) -> tuple[bool, str]`
  - Research external data or complex financial planning. Not for user's own data.

- `update_transaction_category_or_create_category_rules(categorize_request: str, input_info: str = None) -> tuple[bool, str]`
  - Creates categorization rules. `categorize_request` describes rule criteria and category.

</AVAILABLE_SKILL_FUNCTIONS>

<EXAMPLES>

input: **Last User Request**: Walmart purchases should be groceries
**Previous Conversation**:
None
output:
```python
def execute_plan() -> tuple[bool, str]:
    success, categorize_result = update_transaction_category_or_create_category_rules(
        categorize_request="Walmart purchases should be groceries"
    )
    if not success:
        return False, categorize_result
    
    return True, categorize_result
```

input: **Last User Request**: Categorize all my Starbucks transactions as dining out
**Previous Conversation**:
User: How much am I spending at Starbucks?
Assistant: You've spent $120 at Starbucks over the last 3 months across 15 transactions.
output:
```python
def execute_plan() -> tuple[bool, str]:
    success, lookup_result = lookup_user_accounts_transactions_income_and_spending_patterns(
        lookup_request="Find all Starbucks transactions to understand patterns before creating rule."
    )
    if not success:
        return False, lookup_result

    success, categorize_result = update_transaction_category_or_create_category_rules(
        categorize_request="Create a rule to categorize all Starbucks transactions as dining out",
        input_info=lookup_result
    )
    if not success:
        return False, categorize_result
    
    return True, categorize_result
```

input: **Last User Request**: +&-$|(){}#@!%^&*_=[]:;,.<>/?`~
**Previous Conversation**:
None
output:
```python
def execute_plan() -> tuple[bool, str]:
    return False, "The input appears to be unintelligible. Please provide a valid categorization rule request (e.g., 'Walmart purchases should be groceries' or 'always categorize Costco as groceries')."
```

</EXAMPLES>
"""

class CategoryGrounderOptimizer:
  """Handles all Gemini API interactions for Categorization Rule Creation"""
  
  def __init__(self, model_name="gemini-3-pro-preview"):
    """Initialize the Gemini agent with API configuration for categorization rule creation"""
    # API Configuration
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
      raise ValueError("GEMINI_API_KEY environment variable is not set. Please set it in your .env file or environment.")
    self.client = genai.Client(api_key=api_key)
    
    # Model Configuration
    self.thinking_budget = 4096
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
    # Create request text with Last User Request and Previous Conversation
    request_text = types.Part.from_text(text=f"""**Last User Request**: {last_user_request}

**Previous Conversation**:

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
        include_thoughts=True
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

**Previous Conversation**:

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
      # Create wrapper functions that print their returns and handle return types
      def wrapped_lookup(*args, **kwargs):
        print(f"\n[FUNCTION CALL] lookup_user_accounts_transactions_income_and_spending_patterns")
        print(f"  args: {args}")
        print(f"  kwargs: {kwargs}")
        result = lookup_user_accounts_transactions_income_and_spending_patterns(*args, **kwargs)
        print(f"  [RETURN] success: {result[0]}")
        print(f"  [RETURN] output: {result[1]}")
        return result  # Returns tuple[bool, str]
      
      def wrapped_research(*args, **kwargs):
        print(f"\n[FUNCTION CALL] research_and_strategize_financial_outcomes")
        print(f"  args: {args}")
        print(f"  kwargs: {kwargs}")
        result = research_and_strategize_financial_outcomes(*args, **kwargs)
        print(f"  [RETURN] success: {result[0]}")
        print(f"  [RETURN] output: {result[1]}")
        return result  # Returns tuple[bool, str]
      
      def wrapped_categorize(*args, **kwargs):
        print(f"\n[FUNCTION CALL] update_transaction_category_or_create_category_rules")
        print(f"  args: {args}")
        print(f"  kwargs: {kwargs}")
        result = update_transaction_category_or_create_category_rules(*args, **kwargs)
        print(f"  [RETURN] success: {result[0]}")
        print(f"  [RETURN] output: {result[1]}")
        return result  # Returns tuple[bool, str]
      
      # Create a namespace with the wrapped tool functions
      namespace = {
        'lookup_user_accounts_transactions_income_and_spending_patterns': wrapped_lookup,
        'research_and_strategize_financial_outcomes': wrapped_research,
        'update_transaction_category_or_create_category_rules': wrapped_categorize,
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
    success, categorize_result = update_transaction_category_or_create_category_rules(
        categorize_request="Walmart purchases should be groceries"
    )
    if not success:
        return False, categorize_result
    
    return True, categorize_result
```
Key validations:
- Creates a categorization rule for Walmart purchases
- Grounds "groceries" to appropriate category
- No lookup needed as request is explicit"""
  },
  {
    "name": "amount_filter_rule",
    "last_user_request": "schneider greater than $400 to wall construction",
    "previous_conversation": "",
    "ideal_response": """Expected output:
```python
def execute_plan() -> tuple[bool, str]:
    success, categorize_result = update_transaction_category_or_create_category_rules(
        categorize_request="schneider greater than $400 to wall construction"
    )
    if not success:
        return False, categorize_result
    
    return True, categorize_result
```
Key validations:
- Creates rule with amount filter (> $400)
- Grounds "wall construction" to appropriate category (likely Upkeep)
- Includes both merchant name and amount criteria"""
  },
  {
    "name": "multiple_merchants_rule",
    "last_user_request": "John's, Henries and moms and pops needs to all be in supermarkets",
    "previous_conversation": "",
    "ideal_response": """Expected output:
```python
def execute_plan() -> tuple[bool, str]:
    success, categorize_result = update_transaction_category_or_create_category_rules(
        categorize_request="John's, Henries and moms and pops needs to all be in supermarkets"
    )
    if not success:
        return False, categorize_result
    
    return True, categorize_result
```
Key validations:
- Creates categorization rules for multiple merchants (John's, Henries, moms and pops)
- Grounds "supermarkets" to Groceries category
- Handles multiple merchant names in single request"""
  },
  {
    "name": "date_filter_rule",
    "last_user_request": "Barnes & Noble and Bed Bath & Beyond before last week must be put on shopping",
    "previous_conversation": "",
    "ideal_response": """Expected output:
```python
def execute_plan() -> tuple[bool, str]:
    success, categorize_result = update_transaction_category_or_create_category_rules(
        categorize_request="Barnes & Noble and Bed Bath & Beyond before last week must be put on shopping"
    )
    if not success:
        return False, categorize_result
    
    return True, categorize_result
```
Key validations:
- Creates rules with date filter (before last week)
- Multiple merchants (Barnes & Noble, Bed Bath & Beyond)
- Grounds "shopping" to Shopping category
- Handles date relative references"""
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
    success, categorize_result = update_transaction_category_or_create_category_rules(
        categorize_request="coffee in the establishment name should be in aviation and if business like cricket* after 8/24 will be in explosions"
    )
    if not success:
        return False, categorize_result
    
    return True, categorize_result
```
Key validations:
- Creates rules with regex pattern (cricket*)
- Handles multiple rules in one request (coffee -> aviation, cricket* -> explosions)
- Includes date filter (after 8/24)
- Grounds categories appropriately (or returns -1 if no match)"""
  },
  {
    "name": "complex_multi_condition_rule",
    "last_user_request": "Costco purchases over $100 and under $500 from last month should be groceries",
    "previous_conversation": "",
    "ideal_response": """Expected output:
```python
def execute_plan() -> tuple[bool, str]:
    success, categorize_result = update_transaction_category_or_create_category_rules(
        categorize_request="Costco purchases over $100 and under $500 from last month should be groceries"
    )
    if not success:
        return False, categorize_result
    
    return True, categorize_result
```
Key validations:
- Creates rule with multiple AND conditions (merchant + amount range + date)
- Amount greater than $100 AND less than $500
- Date filter (from last month)
- Grounds "groceries" to Groceries category"""
  },
  {
    "name": "income_category_rule",
    "last_user_request": "PayPal transfers from employer are salary",
    "previous_conversation": "",
    "ideal_response": """Expected output:
```python
def execute_plan() -> tuple[bool, str]:
    success, categorize_result = update_transaction_category_or_create_category_rules(
        categorize_request="PayPal transfers from employer are salary"
    )
    if not success:
        return False, categorize_result
    
    return True, categorize_result
```
Key validations:
- Creates rule for income category (salary)
- Handles income transactions (PayPal from employer)
- Grounds "salary" to Salary category
- Tests income category grounding (different from spending categories)"""
  },
  {
    "name": "rule_creation_with_lookup",
    "last_user_request": "Categorize all my Starbucks transactions as dining out",
    "previous_conversation": """User: How much am I spending at Starbucks?
Assistant: You've spent $120 at Starbucks over the last 3 months across 15 transactions.""",
    "ideal_response": """Expected output:
```python
def execute_plan() -> tuple[bool, str]:
    success, lookup_result = lookup_user_accounts_transactions_income_and_spending_patterns(
        lookup_request="Find all Starbucks transactions to understand patterns before creating rule."
    )
    if not success:
        return False, lookup_result

    success, categorize_result = update_transaction_category_or_create_category_rules(
        categorize_request="Create a rule to categorize all Starbucks transactions as dining out",
        input_info=lookup_result
    )
    if not success:
        return False, categorize_result
    
    return True, categorize_result
```
Key validations:
- Uses lookup to gather transaction data before creating rule
- Incorporates lookup result as input_info
- Creates rule for Starbucks transactions
- Grounds "dining out" to Dining Out category"""
  },
  {
    "name": "clarification_answer_rule",
    "last_user_request": "Yes, create a rule for all Target purchases to be shopping",
    "previous_conversation": """User: I want to categorize Target purchases
Assistant: I need more information to create the categorization rule. What category should Target purchases be assigned to? For example, are they groceries, shopping, or another category?""",
    "ideal_response": """Expected output:
```python
def execute_plan() -> tuple[bool, str]:
    success, categorize_result = update_transaction_category_or_create_category_rules(
        categorize_request="Create a rule for all Target purchases to be shopping"
    )
    if not success:
        return False, categorize_result
    
    return True, categorize_result
```
Key validations:
- Responds to clarification question from previous conversation
- Creates rule based on user's answer
- Grounds "shopping" to Shopping category
- Uses context from previous conversation to understand intent"""
  },
  {
    "name": "always_keyword_rule",
    "last_user_request": "Always categorize Amazon purchases as shopping",
    "previous_conversation": "",
    "ideal_response": """Expected output:
```python
def execute_plan() -> tuple[bool, str]:
    success, categorize_result = update_transaction_category_or_create_category_rules(
        categorize_request="Always categorize Amazon purchases as shopping"
    )
    if not success:
        return False, categorize_result
    
    return True, categorize_result
```
Key validations:
- Uses "always" keyword indicating permanent rule
- Creates rule for Amazon purchases
- Grounds "shopping" to Shopping category
- Explicit future/permanent intent"""
  },
  {
    "name": "amount_range_and_date_rule",
    "last_user_request": "Gas station purchases between $20 and $80 from this year should be transportation",
    "previous_conversation": "",
    "ideal_response": """Expected output:
```python
def execute_plan() -> tuple[bool, str]:
    success, categorize_result = update_transaction_category_or_create_category_rules(
        categorize_request="Gas station purchases between $20 and $80 from this year should be transportation"
    )
    if not success:
        return False, categorize_result
    
    return True, categorize_result
```
Key validations:
- Creates rule with amount range ($20-$80)
- Includes date filter (from this year)
- Grounds "transportation" to appropriate category (likely Car & Fuel)
- Multiple AND conditions in single rule"""
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
      "tests": [8, 9, 10, 11]  # Rule with lookup, clarification answer, always keyword, amount range and date
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
  python agent_categorize_optimizer_v2.py --batch 1
  python agent_categorize_optimizer_v2.py --test simple_categorization_rule
  python agent_categorize_optimizer_v2.py --test 0
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

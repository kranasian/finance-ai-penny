from google import genai
from google.genai import types
import os
import sys
from datetime import datetime
from dotenv import load_dotenv

# Add parent directory to path to import sandbox
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sandbox
from database import Database
from penny.tool_funcs.update_transaction_category_or_create_category_rules import (
    update_single_transaction_category,
    create_category_rules,
    update_multiple_transaction_categories_matching_rules
)
from penny.tool_funcs.date_utils import (
    get_date, get_start_of_month, get_end_of_month,
    get_start_of_year, get_end_of_year,
    get_start_of_week, get_end_of_week,
    get_after_periods, get_date_string
)

# Load environment variables
load_dotenv()


SYSTEM_PROMPT = """You are a helpful AI specialized in code generation and trying to categorize transactions. **You only output python code.**

## Task & Rules
1.  **Analyze Input**: Read the **Categorize Request** and **Input Info from previous skill** provided in the user message.
2.  **Match Transaction ID**:
    -   Check **Input Info** text. If a transaction line matches the request (amount, date, merchant), **EXTRACT AND USE ITS ID** with `update_single_transaction_category`.
    -   *Example*: Request "$20 at McD", Input "ID 123: $20 McDonald's" -> Use ID 123.
3.  **Rule Logic (No ID found)**:
    -   If no ID matches, use `update_multiple_transaction_categories_matching_rules`.
    -   **Rules vs Updates**:
        -   **DEFAULT**: Update *existing* transactions only.
        -   **EXCEPTION**: Only call `create_category_rules` if user explicitly writes: "future", "always", "any time", "from now on", "rule".
        -   "Should be", "is", "was" implies updating existing transactions, NOT creating new rules.
4.  **Map Categories**: Use **OFFICIAL_CATEGORIES**.
5.  **Output**: `process_input() -> tuple[bool, str]`. No comments.

Today: |TODAY_DATE|.

<IMPLEMENTED_FUNCTIONS>
- `update_single_transaction_category(transaction_id: int, new_category: str) -> tuple[bool, str]`
- `create_category_rules(rules_dict: dict, new_category: str) -> tuple[bool, str]`
- `update_multiple_transaction_categories_matching_rules(rules_dict: dict, new_category: str) -> tuple[bool, str]`
</IMPLEMENTED_FUNCTIONS>

<RULES_DICT>

Multiple rules are **AND** together, should all be true to match.

**Keys for Name Matching**:
- `name_contains`, `name_startswith`, `name_endswith`:
- `name_eq`: transaction name matches the value exactly.  Only use this if exact is requested specifically.

**Keys for Date Matching**:
- `date_greater_than_or_equal_to`, `date_less_than_or_equal_to`, `date_equal_to`, `date_greater_than`, `date_less_than`

**Keys for Amount Matching**:
- `amount_greater_than_or_equal_to`, `amount_less_than_or_equal_to`, `amount_equal_to`, `amount_greater_than`, `amount_less_than`

**Keys for Account ID Matching**:
- `account_id_eq`: match exact account id specified in the **Input Info from previous skill**.

</RULES_DICT>

<IMPLEMENTED_DATE_FUNCTIONS>

- `get_date(y, m, d)`, `get_start_of_month(date)`, `get_end_of_month(date)`
- `get_start_of_year(date)`, `get_end_of_year(date)`
- `get_start_of_week(date)`, `get_end_of_week(date)`
- `get_after_periods(date, granularity, count)`, `get_date_string(date)`

</IMPLEMENTED_DATE_FUNCTIONS>

<OFFICIAL_CATEGORIES>

- `income`: salary, bonuses, interest, side hussles. (`income_salary`, `income_sidegig`, `income_business`, `income_interest`)
- `meals`: food spending. (`meals_groceries`, `meals_dining_out`, `meals_delivered_food`)
- `leisure`: recreation/travel. (`leisure_entertainment`, `leisure_travel`)
- `bills`: recurring costs. (`bills_connectivity`, `bills_insurance`, `bills_tax`, `bills_service_fees`)
- `shelter`: housing. (`shelter_home`, `shelter_utilities`, `shelter_upkeep`)
- `education`: learning/kids. (`education_kids_activities`, `education_tuition`)
- `shopping`: discretionary. (`shopping_clothing`, `shopping_gadgets`, `shopping_kids`, `shopping_pets`)
- `transportation`: car/public. (`transportation_public`, `transportation_car`)
- `health`: medical/wellness. (`health_medical_pharmacy`, `health_gym_wellness`, `health_personal_care`)
- `donations_gifts`: charity/gifts.
- `uncategorized`: unknown.
- `transfers`: internal movements.
- `miscellaneous`: other.

</OFFICIAL_CATEGORIES>

<EXAMPLES>

input: **Categorize Request**: $21 yesterday was eating out
**Input Info from previous skill**:
Transaction ID 32343: 2025-11-26 $21 McDonald's (Account ID: 20)
output:
```python
def process_input():
    output_lines = []
    success, result = update_single_transaction_category(transaction_id=32343, new_category='meals_dining_out')
    output_lines.append(result)
    if not success:
        return False, chr(10).join(output_lines)
    return True, chr(10).join(output_lines)
```

input: **Categorize Request**: Costco purchases from last 3 months under $50 is for gas.
**Input Info from previous skill**:
Transaction ID 1234: 2025-10-15 $85 Costco (Account ID: 100)
output:
```python
def process_input():
    output_lines = []
    today = datetime.now()
    start_date = get_date_string(get_after_periods(today, granularity="monthly", count=-3))
    costco_rules_dict = {
        'name_contains': 'costco',
        'date_greater_than_or_equal_to': start_date,
        'amount_less_than': 50
    }
    success, result1 = update_multiple_transaction_categories_matching_rules(rules_dict=costco_rules_dict, new_category='transportation_car')
    output_lines.append(result1)
    if not success:
        return False, chr(10).join(output_lines)
    return True, chr(10).join(output_lines)
```

input: **Categorize Request**: IRS payments are ALWAYS tax.
output:
```python
def process_input():
    output_lines = []
    irs_rules_dict = {
        'name_contains': 'irs'
    }
    success, result1 = update_multiple_transaction_categories_matching_rules(rules_dict=irs_rules_dict, new_category='bills_tax')
    output_lines.append(result1)
    if not success:
        return False, chr(10).join(output_lines)
    success, result2 = create_category_rules(rules_dict=irs_rules_dict, new_category='bills_tax')
    output_lines.append(result2)
    if not success:
        return False, chr(10).join(output_lines)
    return True, chr(10).join(output_lines)
```

</EXAMPLES>
"""


class UpdateTransactionCategoryOptimizer:
  """Handles all Gemini API interactions for updating transaction categories"""
  
  def __init__(self, model_name="gemini-flash-lite-latest"):
    """Initialize the Gemini agent with API configuration for updating transaction categories"""
    # API Configuration
    self.client = genai.Client(api_key=os.getenv('GEMINI_API_KEY'))
    
    # Model Configuration
    self.thinking_budget = 0
    self.model_name = model_name
    
    # Generation Configuration Constants
    self.temperature = 0.6
    self.top_p = 0.95
    self.max_output_tokens = 4096
    
    # Safety Settings
    self.safety_settings = [
      types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"),
      types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"),
      types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"),
      types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="OFF")
    ]
    
    # System Prompt
    self.system_prompt = SYSTEM_PROMPT

  def _get_today_date_string(self) -> str:
    """
    Get today's date formatted as "Day, Month DD, YYYY"
    
    Returns:
      String containing today's date in the specified format
    """
    today = datetime.now()
    return today.strftime("%A, %B %d, %Y")

  
  def generate_response(self, categorize_request: str, user_id: int = 1) -> str:
    """
    Generate a response using Gemini API for updating transaction categories.
    
    Args:
      categorize_request: The categorization request as a string
      user_id: User ID for building dynamic sections (default: 1)
      
    Returns:
      Generated code as a string
    """
    # Get today's date
    today_date = self._get_today_date_string()
    
    # Replace placeholders in system prompt
    full_system_prompt = self.system_prompt.replace("|TODAY_DATE|", today_date)
    
    # Create request text with categorization request
    request_text = types.Part.from_text(text=f"""**Categorize Request**: {categorize_request}

**Input Info from previous skill**:
output:""")
    
    # Create content and configuration
    contents = [types.Content(role="user", parts=[request_text])]
    
    generate_content_config = types.GenerateContentConfig(
      temperature=self.temperature,
      top_p=self.top_p,
      max_output_tokens=self.max_output_tokens,
      safety_settings=self.safety_settings,
      system_instruction=[types.Part.from_text(text=full_system_prompt)],
      thinking_config=types.ThinkingConfig(thinking_budget=self.thinking_budget),
    )

    # Generate response
    output_text = ""
    for chunk in self.client.models.generate_content_stream(
      model=self.model_name,
      contents=contents,
      config=generate_content_config,
    ):
      if chunk.text is not None:
        output_text += chunk.text
    
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


def _get_heavy_data_user_id() -> int:
  """
  Get the user ID for HeavyDataUser from the database.
  
  Returns:
    The user ID for HeavyDataUser, or 1 if not found
  """
  try:
    db = Database()
    heavy_user = db.get_user("HeavyDataUser")
    if heavy_user and 'id' in heavy_user:
      return heavy_user['id']
    else:
      print("Warning: HeavyDataUser not found, using default user_id=1")
      return 1
  except Exception as e:
    print(f"Warning: Error getting HeavyDataUser: {e}, using default user_id=1")
    return 1


def _run_test_with_logging(categorize_request: str, input_info: str = None, optimizer: UpdateTransactionCategoryOptimizer = None, user_id: int = None):
  """
  Internal helper function that runs a test with consistent logging.
  
  Args:
    categorize_request: The categorization request as a string
    optimizer: Optional UpdateTransactionCategoryOptimizer instance. If None, creates a new one.
    user_id: User ID for sandbox execution (default: HeavyDataUser ID from database)
    
  Returns:
    The generated response string
  """
  if optimizer is None:
    optimizer = UpdateTransactionCategoryOptimizer()

  input_info_text = f"\n**Input Info from previous skill**:\n{input_info}" if input_info else ""
  
  # Get HeavyDataUser ID if not provided
  if user_id is None:
    user_id = _get_heavy_data_user_id()
  
  # Construct LLM input
  llm_input = f"""**Categorize Request**: {categorize_request}{input_info_text}"""
  
  # Print the input
  print("=" * 80)
  print("LLM INPUT:")
  print("=" * 80)
  print(llm_input)
  print("=" * 80)
  print()
  
  result = optimizer.generate_response(categorize_request, user_id)
  
  # Print the output
  print("=" * 80)
  print("LLM OUTPUT:")
  print("=" * 80)
  print(result)
  print("=" * 80)
  print()
  
  # Execute the generated code in sandbox
  print("=" * 80)
  print("EXECUTION RESULT:")
  print("=" * 80)
  try:
    # Import the three functions that are only available in update_transaction_category_or_create_category_rules namespace
    from penny.tool_funcs.update_transaction_category_or_create_category_rules import (
      update_single_transaction_category,
      create_category_rules,
      update_multiple_transaction_categories_matching_rules
    )
    # Add them to the execution namespace for testing
    additional_namespace = {
      'update_single_transaction_category': update_single_transaction_category,
      'create_category_rules': create_category_rules,
      'update_multiple_transaction_categories_matching_rules': update_multiple_transaction_categories_matching_rules,
    }
    # Execute code using sandbox (it will extract code from ```python blocks automatically)
    success, output, logs, _ = sandbox.execute_agent_with_tools(result, user_id, additional_namespace)
    
    print(f"Success: {success}")
    print(f"Output: {output}")
    if logs:
      print(f"\nLogs:\n{logs}")
      
  except Exception as e:
    print(f"**Execution Error**: {str(e)}")
    import traceback
    print(traceback.format_exc())
  print("=" * 80)
  print()
  
  return result


def test_transaction_id_update(optimizer: UpdateTransactionCategoryOptimizer = None):
  categorize_request = "$21 yesterday was eating out"
  input_info = """Transaction ID 32342: 2025-11-22 $23 Chipotle (Account ID: 20)
Transaction ID 32343: 2025-11-26 $21 McDonald's (Account ID: 20)"""
  return _run_test_with_logging(categorize_request, input_info, optimizer)


def test_costco_purchases_last_3_months(optimizer: UpdateTransactionCategoryOptimizer = None):
  """
  Test method for updating Costco purchases from last 3 months to gas category.
  
  Args:
    optimizer: Optional UpdateTransactionCategoryOptimizer instance. If None, creates a new one.
    
  Returns:
    The generated response string
  """
  categorize_request = "costco purchases from the last 3 months should be gas"
  return _run_test_with_logging(categorize_request, optimizer)


def test_multiple_conditions(optimizer: UpdateTransactionCategoryOptimizer = None):
  """
  Test method for multiple categorization conditions.
  
  Args:
    optimizer: Optional UpdateTransactionCategoryOptimizer instance. If None, creates a new one.
    
  Returns:
    The generated response string
  """
  categorize_request = "purchases over $400 last week was taxes and spending over 60 from Chase are to keep kids busy"
  input_info = """Account Chase Checking (account_id: 20) has $1000 balance.
Account Discover Credit Card (account_id: 21) has $3233 balance."""
  return _run_test_with_logging(categorize_request, input_info, optimizer)


def test_date_equal_to(optimizer: UpdateTransactionCategoryOptimizer = None):
  """Test date_equal_to matching"""
  categorize_request = "transactions on November 26, 2025 were groceries"
  input_info = """Transaction ID 32343: 2025-11-26 $21 McDonald's (Account ID: 20)
Transaction ID 32344: 2025-11-26 $45 Whole Foods (Account ID: 20)"""
  return _run_test_with_logging(categorize_request, input_info, optimizer)


def test_amount_equal_to(optimizer: UpdateTransactionCategoryOptimizer = None):
  """Test amount_equal_to matching"""
  categorize_request = "all $50 transactions are subscriptions"
  return _run_test_with_logging(categorize_request, optimizer=optimizer)


def test_name_startswith(optimizer: UpdateTransactionCategoryOptimizer = None):
  """Test name_startswith matching"""
  categorize_request = "all transactions starting with 'Amazon' are shopping"
  return _run_test_with_logging(categorize_request, optimizer=optimizer)


def test_name_endswith(optimizer: UpdateTransactionCategoryOptimizer = None):
  """Test name_endswith matching"""
  categorize_request = "transactions ending with 'Gas' are transportation"
  return _run_test_with_logging(categorize_request, optimizer=optimizer)


def test_name_exact_match(optimizer: UpdateTransactionCategoryOptimizer = None):
  """Test name_eq (exact match) matching"""
  categorize_request = "exactly 'Starbucks' transactions are dining out"
  return _run_test_with_logging(categorize_request, optimizer=optimizer)


def test_account_id_matching(optimizer: UpdateTransactionCategoryOptimizer = None):
  """Test account_id_eq matching"""
  categorize_request = "all transactions from account 20 are personal expenses"
  input_info = """Account Chase Checking (account_id: 20) has $1000 balance."""
  return _run_test_with_logging(categorize_request, input_info, optimizer)


def test_create_category_rules(optimizer: UpdateTransactionCategoryOptimizer = None):
  """Test creating category rules for future transactions"""
  categorize_request = "from now on, all IRS payments should always be categorized as taxes"
  return _run_test_with_logging(categorize_request, optimizer=optimizer)


def test_date_range_combination(optimizer: UpdateTransactionCategoryOptimizer = None):
  """Test combination of date_greater_than and date_less_than"""
  categorize_request = "transactions between November 1 and November 15 that are over $100 are bills"
  return _run_test_with_logging(categorize_request, optimizer=optimizer)


def test_amount_range_combination(optimizer: UpdateTransactionCategoryOptimizer = None):
  """Test combination of amount_greater_than and amount_less_than"""
  categorize_request = "transactions between $50 and $200 from last month are shopping"
  return _run_test_with_logging(categorize_request, optimizer=optimizer)


def test_multiple_name_conditions(optimizer: UpdateTransactionCategoryOptimizer = None):
  """Test multiple name matching conditions"""
  categorize_request = "transactions containing 'Target' or starting with 'Walmart' are groceries"
  return _run_test_with_logging(categorize_request, optimizer=optimizer)


def test_complex_combination(optimizer: UpdateTransactionCategoryOptimizer = None):
  """Test complex combination of multiple rule types"""
  categorize_request = "Costco purchases over $100 from account 20 in the last 2 months are home expenses"
  input_info = """Account Chase Checking (account_id: 20) has $1000 balance."""
  return _run_test_with_logging(categorize_request, input_info, optimizer)


def test_single_transaction_by_amount_and_date(optimizer: UpdateTransactionCategoryOptimizer = None):
  """Test finding single transaction by amount and date"""
  categorize_request = "$21 transaction on November 26 was eating out"
  input_info = """Transaction ID 32342: 2025-11-22 $23 Chipotle (Account ID: 20)
Transaction ID 32343: 2025-11-26 $21 McDonald's (Account ID: 20)"""
  return _run_test_with_logging(categorize_request, input_info, optimizer)


def main():
  """Main function to test the update transaction category optimizer"""
  # print("=" * 80)
  # print("BASIC TESTS")
  # print("=" * 80)
  # test_transaction_id_update()
  # test_costco_purchases_last_3_months()
  # test_multiple_conditions()
  
  # print("\n" + "=" * 80)
  # print("DATE MATCHING TESTS")
  # print("=" * 80)
  # test_date_equal_to()
  # test_date_range_combination()
  
  print("\n" + "=" * 80)
  print("AMOUNT MATCHING TESTS")
  print("=" * 80)
  test_amount_equal_to()
  test_amount_range_combination()
  
  # print("\n" + "=" * 80)
  # print("NAME MATCHING TESTS")
  # print("=" * 80)
  # test_name_startswith()
  # test_name_endswith()
  # test_name_exact_match()
  # test_multiple_name_conditions()
  
  # print("\n" + "=" * 80)
  # print("ACCOUNT MATCHING TESTS")
  # print("=" * 80)
  # test_account_id_matching()
  
  # print("\n" + "=" * 80)
  # print("RULE CREATION TESTS")
  # print("=" * 80)
  # test_create_category_rules()
  
  # print("\n" + "=" * 80)
  # print("COMPLEX COMBINATION TESTS")
  # print("=" * 80)
  # test_complex_combination()
  # test_single_transaction_by_amount_and_date()


if __name__ == "__main__":
  main()


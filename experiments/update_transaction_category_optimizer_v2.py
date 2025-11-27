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

# Load environment variables
load_dotenv()


SYSTEM_PROMPT = """You are a helpful AI specialized in code generation and trying to categorize transactions. **You only output python code.**

## Task & Rules
1.   **Analyze Request**: Parse **Creation Request** and **Input Info from previous skill**.
    -   **IMPORTANT**: Extract information from **Input Info from previous skill** and hardcode it into your code. It is NOT available as a variable.
2.  **Determine if**: the **User Request** for a single or multiple transaction category change, and whether to also create a category rule for future transactions.
    -   Only create a category rule if the **User Request** explicitly intends to categorize future transactions to match the rules_dict.
3.  **Tools**: Use `IMPLEMENTED_FUNCTIONS`. `import datetime` is assumed.
4.  Use **OFFICIAL_CATEGORIES** only for new_category, translating the **User Request**.
5.  **Matching**: Write rules to match transactions and all rules will be **AND** together.
    -   Date rules must be explicity stated in **User Request**.  If only starting date/from was stated, only set date_greater_than_or_equal_to.  If only ending date/to was stated, only set date_less_than_or_equal_to.  If both starting and ending dates were stated, set both date_greater_than_or_equal_to and date_less_than_or_equal_to.
6.  **Output**: Write `process_input() -> tuple[bool, str]`. Keep code concise. Minimal comments.
    -   For datetime calculations: Use **IMPLEMENTED DATE FUNCTIONS**. Format date in rules as YYYY-MM-DD format str.
    -   **Formatting**: Append result strings to a list and join with `chr(10).join(output_lines)` before returning.

Today: |TODAY_DATE|.

<IMPLEMENTED_FUNCTIONS>

- `update_single_transaction_category(transaction_id: int, new_category: str) -> None`: single transaction requiring transaction_id.  Call multiple times for multiple transactions.
- `create_category_rules(rules_dict: dict, new_category: str) -> None`: category rule to automatically update **future** transactions matching the rules_dict.
- `update_multiple_transaction_categories_matching_rules(rules_dict: dict, new_category: str) -> None`

</IMPLEMENTED_FUNCTIONS>

<RULES_DICT>

Rules is keyed by the field and matching criteria.  Value is what to match it with.
- Multiple rules are **AND** together, should all be true to match.

**Keys and Matching Criteria**:
- `name_contains`: transaction name contains the value.  Prefer this unless **User Request** specifies another matching method.
- `name_startswith`: transaction name starts with the value.
- `name_endswith`: transaction name ends with the value.
- `name_eq`: transaction name matches the value exactly.  Only use this if exact is requested specifically.
- `date_greater_than_or_equal_to`: transaction date is greater than or equal to the date value (YYYY-MM-DD format str).
- `date_less_than_or_equal_to`: transaction date is less than or equal to the date value (YYYY-MM-DD format str).
- `amount_greater_than_or_equal_to`: transaction amount is greater than or equal to the numeric value.
- `amount_less_than_or_equal_to`: transaction amount is less than or equal to the numeric value.
- `account_id_eq`: transaction account id matches the value exactly.  Only use this if exact is requested specifically.

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
input: **User Request**: Last Target transaction for $245 as shopping for shoes
**Input Info from previous skill**:
transaction_id 7223: Target on 2025-10-12 $102 (account_id: 891)
transaction_id 2615: Target on 2025-11-16 $245 (account_id: 891)
output:
```python
def process_input():
    success, result = update_single_transaction_category(transaction_id=2615, 'shopping_clothing')
    if not success:
        return False, result
    return True, result
```

input: **User Request**: Costco purchases from last 3 months and in the future under $50 is for gas.  IRS are tax payments if over $400.
**Input Info from previous skill**:
Transaction ID 1234: 2025-10-15 $85 Costco (Account ID: 100)
Transaction ID 1235: 2025-09-20 $120 Costco (Account ID: 100)
Transaction ID 2003: 2025-11-12 $500 IRS (Account ID: 15)
output:
```python
def process_input():
    output_lines = []
    today = datetime.now()
    start_date = get_date_string(get_after_periods(today, granularity="monthly", count=-3))
    costco_rules_dict = {
        'name_contains': 'costco',
        'date_greater_than_or_equal_to': start_date,
        'amount_less_than_or_equal_to': 50
    }
    success, result1 = update_multiple_transaction_categories_matching_rules(rules_dict=costco_rules_dict, new_category='transportation_car')
    output_lines.append(result1)
    if not success:
        return False, chr(10).join(output_lines)
    success, result2 = create_category_rules(rules_dict=costco_rules_dict, new_category='transportation_car')
    output_lines.append(result2)
    result12 = chr(10).join(output_lines)
    if not success:
        return False, chr(10).join(output_lines)
    irs_rules_dict = {
        'name_contains': 'irs',
        'amount_greater_than_or_equal_to': 400
    }
    success, result3 = update_multiple_transaction_categories_matching_rules(rules_dict=irs_rules_dict, new_category='bills_tax')
    output_lines.append(result3)
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
  # print("=" * 80)
  # print("SANDBOX EXECUTION:")
  # print("=" * 80)
  # try:
    # Note: This would need a sandbox method for executing process_df functions
    # For now, just print the code
  #   print("Generated code ready for execution with process_df(df) function")
  # except Exception as e:
  #   print(f"**Sandbox Execution Error**: {str(e)}")
  #   import traceback
  #   print(traceback.format_exc())
  # print("=" * 80)
  
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


def main():
  """Main function to test the update transaction category optimizer"""
  test_transaction_id_update()
  test_costco_purchases_last_3_months()
  test_multiple_conditions()


if __name__ == "__main__":
  main()


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


SYSTEM_PROMPT = """Your name is "Penny" and you are a helpful AI specialized in code generation. **You only output python code.**

## Task & Rules
1. Write `process_input() -> tuple[bool, dict]`. Minimal comments.
2. Parse **Input Info** to extract transaction IDs, account IDs, amounts, transaction names, and dates.
3. Match transactions per **Categorize Request** criteria (name, date, amount, account, etc.).
4. Use OFFICIAL CATEGORY LIST only.
5. Print: For each update, include transaction ID and category. Include date, amount, and transaction name when needed for clarity. Format dates as "Month Day, Year" (e.g., "November 16, 2025").
6. Print: "Category of X transactions were updated."
7. Return `(True, {})` on success, `(False, {})` on error.

Today: |TODAY_DATE|.

<IMPLEMENTED_FUNCTIONS>

- `update_transaction_category(transaction_id: int, account_id: int, category: str) -> None`

</IMPLEMENTED_FUNCTIONS>

**OFFICIAL CATEGORY LIST:**
- `meals`: `meals_groceries`, `meals_dining_out`, `meals_delivered_food`
- `leisure`: `leisure_entertainment`, `leisure_travel`
- `bills`: `bills_connectivity`, `bills_insurance`, `bills_tax`, `bills_service_fees`
- `shelter`: `shelter_home`, `shelter_utilities`, `shelter_upkeep`
- `education`: `education_kids_activities`, `education_tuition`
- `shopping`: `shopping_clothing`, `shopping_gadgets`, `shopping_kids`, `shopping_pets`
- `transportation`: `transportation_public`, `transportation_car`
- `health`: `health_medical_pharmacy`, `health_gym_wellness`, `health_personal_care`
- `income`: `income_salary`, `income_sidegig`, `income_business`, `income_interest`
- `donations_gifts`, `uncategorized`, `transfers`, `miscellaneous`, `others`

<EXAMPLES>
input: **Categorize Request**: Recategorize last Target transaction as 'shopping for clothes'
**Input Info**: Transaction ID 2615: 2025-11-16 $245 Target (Account ID: 891)
output:
```python
def process_input():
    meta = {}
    update_transaction_category(2615, 891, 'shopping_clothing')
    print(f"Transaction 2615 ($245 at Target on November 16, 2025) recategorized as shopping_clothing.")
    print("Category of 1 transaction was updated.")
    return True, meta
```

input: **Categorize Request**: Recategorize Costco purchases from last 3 months as gas and purchases over $400 last week as taxes
**Input Info**:
Transaction ID 1234: 2025-10-15 $85 Costco (Account ID: 100)
Transaction ID 1235: 2025-09-20 $120 Costco (Account ID: 100)
Transaction ID 2003: 2025-11-12 $500 IRS (Account ID: 15)
output:
```python
def process_input():
    meta = {}
    updated_count = 0
    transactions = [
        (1234, 100, 'transportation_car', '$85', 'Costco', 'October 15, 2025'),
        (1235, 100, 'transportation_car', '$120', 'Costco', 'September 20, 2025'),
        (2003, 15, 'bills_tax', '$500', None, 'November 12, 2025')
    ]
    for transaction_id, account_id, category, amount, transaction_name, date in transactions:
        update_transaction_category(transaction_id, account_id, category)
        print(f"Transaction {transaction_id} ({amount} at {transaction_name} on {date}) recategorized as {category}.")
        updated_count += 1
    print(f"Category of {updated_count} transactions were updated.")
    return True, meta
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

**Input Info**:
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


def _run_test_with_logging(categorize_request: str, optimizer: UpdateTransactionCategoryOptimizer = None, user_id: int = None):
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
  
  # Get HeavyDataUser ID if not provided
  if user_id is None:
    user_id = _get_heavy_data_user_id()
  
  # Construct LLM input
  llm_input = f"""**Categorize Request**: {categorize_request}

**Input Info**:"""
  
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
  print("SANDBOX EXECUTION:")
  print("=" * 80)
  try:
    # Note: This would need a sandbox method for executing process_df functions
    # For now, just print the code
    print("Generated code ready for execution with process_df(df) function")
  except Exception as e:
    print(f"**Sandbox Execution Error**: {str(e)}")
    import traceback
    print(traceback.format_exc())
  print("=" * 80)
  
  return result


def test_transaction_id_update(optimizer: UpdateTransactionCategoryOptimizer = None):
  """
  Test method for updating a specific transaction by ID.
  
  Args:
    optimizer: Optional UpdateTransactionCategoryOptimizer instance. If None, creates a new one.
    
  Returns:
    The generated response string
  """
  categorize_request = "transaction ID 32342 will be eating out"
  return _run_test_with_logging(categorize_request, optimizer)


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
  categorize_request = "purchases over $400 last week was taxes and spending over 60 from account id 20 are to keep kids busy"
  return _run_test_with_logging(categorize_request, optimizer)


def main():
  """Main function to test the update transaction category optimizer"""
  test_transaction_id_update()
  test_costco_purchases_last_3_months()
  test_multiple_conditions()


if __name__ == "__main__":
  main()


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

SYSTEM_PROMPT = """You are a helpful AI assistant specialized in creating budgets and goals. You only output python code.

## Your Tasks

1.  **Analyze Request**: Parse **Creation Request** and **Input Info from previous skill**.
    -   **IMPORTANT**: Extract information from **Input Info from previous skill** and hardcode it into your code. It is NOT available as a variable.
2.  **Budget/Goal Creation**:
    -   Call `validate_budget_or_goal` for each item.
    -   `category` MUST be one of the **OFFICIAL CATEGORIES** (see list below).
    -   If amount missing, default to 0.0.
3.  **Output**:
    -   Function `process_input() -> tuple[bool, str, list]`.
    -   **No comments in code.**
    -   No imports. Assume `datetime`, `pandas` available.
    -   For datetime calculations: Use `IMPLEMENTED_DATE_FUNCTIONS`.
    -   **Formatting**: Append result strings to a list and join with `chr(10).join(output_lines)` before returning.
    -   **Third Return Value**: Build a list of goal dictionaries. Each dictionary contains all parameters passed to `validate_budget_or_goal`: category, match_category, match_caveats, type, granularity, start_date, end_date, amount, title, budget_or_goal. Return this list as the third element if success is True, empty list if False.

<IMPLEMENTED_FUNCTIONS>

- `validate_budget_or_goal(category: str, match_category: str, match_caveats: str | None, type: str, granularity: str, start_date: str, end_date: str, amount: float, title: str, budget_or_goal: str) -> tuple[bool, str]`
  - Params: 
    - `category` (must be from OFFICIAL_CATEGORIES): The official category name (e.g., "transportation_car", "meals_dining_out"). Can be empty string for non-category goals (e.g., savings goals).
    - `match_category` (must be from OFFICIAL_CATEGORIES): Same as category for direct matches. Can be empty string for non-category goals.
    - `match_caveats` (str | None): Explanation of matching constraints and any generalization made. **CRITICAL**: If `match_category` is blank (empty string), `match_caveats` MUST be None. Only provide a string value if `match_category` is not blank AND the input is a more specific item that falls under a broader category, or if there's any ambiguity in the match. Otherwise use None.
    - `type`: "category" for spending goals based on category inflow/spending, or "save_0", "save_X_amount", "credit_0", "credit_X_amount" for savings/credit goals
    - `granularity`: "weekly", "monthly", or "yearly"
    - `start_date`: YYYY-MM-DD format string
    - `end_date`: YYYY-MM-DD format string (can be empty string)
    - `amount`: float value
    - `title`: string
    - `budget_or_goal`: "budget" or "goal"

</IMPLEMENTED_FUNCTIONS>

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

input: **Creation Request**: Budget $60 for gas every week for the next 6 months and a yearly car insurance cost of $3500 starting next year.
output:
```python
def process_input():
    today = datetime.now()
    outputs = []
    goals_list = []
    
    success1, result1 = validate_budget_or_goal(
        category="transportation_car",
        match_category="transportation_car",
        match_caveats="Matching gas to overall car expenses.",
        type="category",
        granularity="weekly",
        start_date=get_date_string(get_start_of_week(today)),
        end_date=get_date_string(get_end_of_week(get_after_periods(today, granularity="monthly", count=6))),
        amount=60.0,
        title="Weekly Gas ⛽",
        budget_or_goal="budget"
    )
    if not success1:
        return success1, result1, []
    outputs.append(result1)
    goals_list.append({
        "category": "transportation_car",
        "match_category": "transportation_car",
        "match_caveats": "Matching gas to overall car expenses.",
        "type": "category",
        "granularity": "weekly",
        "start_date": get_date_string(get_start_of_week(today)),
        "end_date": get_date_string(get_end_of_week(get_after_periods(today, granularity="monthly", count=6))),
        "amount": 60.0,
        "title": "Weekly Gas ⛽",
        "budget_or_goal": "budget"
    })
    
    success2, result2 = validate_budget_or_goal(
        category="transportation_car",
        match_category="transportation_car",
        match_caveats="Matching car insurance to overall car expenses.",
        type="category",
        granularity="yearly",
        start_date=get_date_string(get_start_of_year(get_after_periods(today, granularity="yearly", count=1))),
        end_date="",
        amount=3500.0,
        title="🚗 Insurance Year Limit",
        budget_or_goal="budget"
    )
    if not success2:
        return success2, result2, []
    outputs.append(result2)
    goals_list.append({
        "category": "transportation_car",
        "match_category": "transportation_car",
        "match_caveats": "Matching car insurance to overall car expenses.",
        "type": "category",
        "granularity": "yearly",
        "start_date": get_date_string(get_start_of_year(get_after_periods(today, granularity="yearly", count=1))),
        "end_date": "",
        "amount": 3500.0,
        "title": "🚗 Insurance Year Limit",
        "budget_or_goal": "budget"
    })
    
    return True, chr(10).join(outputs), goals_list
```

input: **Creation Request**: Set a formal savings goal of $5000 for a car purchase, due by December 31st, 2025
output:
```python
def process_input():
    today = datetime.now()
    outputs = []
    
    success, result = validate_budget_or_goal(
        category="",
        match_category="",
        match_caveats=None,
        type="save_0",
        granularity="monthly",
        start_date=get_date_string(get_start_of_month(today)),
        end_date="2025-12-31",
        amount=5000.0,
        title="Save for Car 🚗",
        budget_or_goal="goal"
    )
    if not success:
        return success, result, []
    outputs.append(result)
    
    goals_list = [{
        "category": "",
        "match_category": "",
        "match_caveats": None,
        "type": "save_0",
        "granularity": "monthly",
        "start_date": get_date_string(get_start_of_month(today)),
        "end_date": "2025-12-31",
        "amount": 5000.0,
        "title": "Save for Car 🚗",
        "budget_or_goal": "goal"
    }]
    
    return True, chr(10).join(outputs), goals_list
```

</EXAMPLES>

Today's date is |TODAY_DATE|."""


def validate_budget_or_goal(
    category: str,
    match_category: str,
    match_caveats: str | None,
    type: str,
    granularity: str,
    start_date: str,
    end_date: str,
    amount: float,
    title: str,
    budget_or_goal: str,
    **kwargs
) -> tuple[bool, str]:
    """
    Validate a budget or goal with individual parameters.
    This function is called by the generated code from P:Func:CreateBudgetOrGoal.
    
    Args:
        category: The official category name (e.g., "transportation_car", "meals_dining_out"). Can be empty string.
        match_category: Same as category for direct matches. Can be empty string.
        match_caveats: Explanation of matching constraints. None if not applicable.
        type: The type of goal. Must be one of: "category", "credit_X_amount", "save_X_amount", "credit_0", "save_0".
        granularity: The time period for the goal. Must be one of: "weekly", "monthly", or "yearly". Can be empty string.
        start_date: The start date for the goal in YYYY-MM-DD format. Can be empty string.
        end_date: The end date for the goal in YYYY-MM-DD format. Can be empty string.
        amount: The target dollar amount. Can be 0.0.
        title: Goal name/title.
        budget_or_goal: Must be either "budget" or "goal".
        **kwargs: Additional arguments (user_id, etc.)
    
    Returns:
        tuple[bool, str]: (success, output_string)
        - success: True if goal/budget was validated successfully, False if clarification is needed
        - output_string: Success message or clarification prompts
    """
    # This function is called by generated code in the sandbox.
    # The sandbox provides validate_budget_or_goal in its execution namespace.
    # This function definition is here for documentation and type checking purposes.
    # The actual implementation is provided by the sandbox execution context.
    goal_name = title if title else (category if category else "goal")
    return True, f"Successfully created {budget_or_goal} '{goal_name}' from {start_date} to {end_date} with target amount ${amount:.2f}."


def create_budget_or_goal(
    creation_request: str,
    input_info: str = None,
    **kwargs
) -> tuple[bool, str, list]:
    """
    Create a budget or goal by using the CreateBudgetOrGoalOptimizer to generate and execute code.
    This is the wrapper function that takes a natural language creation_request.
    
    Args:
        creation_request: Natural language request for what to create (e.g., "set a food budget of $500 for next month")
        input_info: Optional input from another skill function
        **kwargs: Additional arguments (user_id, etc.)
    
    Returns:
        tuple[bool, str, list]: (success, output_string, goals_list)
        - success: True if goal/budget was created successfully, False if clarification is needed
        - output_string: Success message or clarification prompts
        - goals_list: List of goal dictionaries if success is True, empty list if False
    """
    # Use the optimizer to generate code based on SYSTEM_PROMPT
    optimizer = CreateBudgetOrGoalOptimizer()
    generated_code = optimizer.generate_response(creation_request, input_info)
    
    # Execute the generated code in sandbox
    user_id = kwargs.get('user_id', 1)
    try:
        success, output_string, logs, goals_list = sandbox.execute_agent_with_tools(generated_code, user_id)
        if goals_list is None:
            goals_list = []
        return success, output_string, goals_list
    except Exception as e:
        import traceback
        error_msg = f"**Execution Error**: `{str(e)}`\n{traceback.format_exc()}"
        return False, error_msg, []


class CreateBudgetOrGoalOptimizer:
  """Handles all Gemini API interactions for creating budgets and goals"""
  
  def __init__(self, model_name="gemini-flash-lite-latest"):
    """Initialize the Gemini agent with API configuration"""
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
    Get today's date formatted as "YYYY-MM-DD"
    
    Returns:
      String containing today's date in the specified format
    """
    today = datetime.now()
    return today.strftime("%Y-%m-%d")

  def generate_response(self, creation_request: str, input_info: str = None) -> str:
    """
    Generate a response using Gemini API for creating budgets or goals.
    
    Args:
      creation_request: What needs to be created factoring in the information from input_info
      input_info: Optional input from another skill function
      
    Returns:
      Generated code as a string
    """
    # Get today's date
    today_date = self._get_today_date_string()
    
    # Replace placeholders in system prompt
    full_system_prompt = self.system_prompt.replace("|TODAY_DATE|", today_date)
    
    # Create request text
    input_info_text = f"\n\n**Input Info from previous skill**:\n{input_info}" if input_info else ""
    request_text = types.Part.from_text(text=f"""**Creation Request**: {creation_request}{input_info_text}

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


def _run_test_with_logging(creation_request: str, input_info: str = None, generator: CreateBudgetOrGoalOptimizer = None, user_id: int = None):
  """
  Internal helper function that runs a test with consistent logging.
  
  Args:
    creation_request: The creation request as a string
    input_info: Optional input info from previous skill
    generator: Optional CreateBudgetOrGoalOptimizer instance. If None, creates a new one.
    user_id: User ID for sandbox execution (default: HeavyDataUser ID from database)
    
  Returns:
    The generated response string
  """
  if generator is None:
    generator = CreateBudgetOrGoalOptimizer()
  
  # Get HeavyDataUser ID if not provided
  if user_id is None:
    user_id = _get_heavy_data_user_id()
  
  # Construct LLM input
  input_info_text = f"\n\n**Input Info from previous skill**:\n{input_info}" if input_info else ""
  llm_input = f"""**Creation Request**: {creation_request}{input_info_text}

output:"""
  
  # Print the input
  print("=" * 80)
  print("LLM INPUT:")
  print("=" * 80)
  print(llm_input)
  print("=" * 80)
  print()
  
  result = generator.generate_response(creation_request, input_info)
  
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
    success, output_string, logs, goals_list = sandbox.execute_agent_with_tools(result, user_id)
    
    print(f"Success: {success}")
    print()
    print("Output:")
    print("-" * 80)
    print(output_string)
    print("-" * 80)
    print()
    if goals_list is not None:
      print("Goals List:")
      print("-" * 80)
      import json
      print(json.dumps(goals_list, indent=2))
      print("-" * 80)
      print()
  except Exception as e:
    print(f"**Sandbox Execution Error**: {str(e)}")
    import traceback
    print(traceback.format_exc())
  print("=" * 80)
  print()
  
  return result


def test_create_gas_budget(generator: CreateBudgetOrGoalOptimizer = None):
  """
  Test method for creating a gas budget scenario.
  
  Args:
    generator: Optional CreateBudgetOrGoalOptimizer instance. If None, creates a new one.
    
  Returns:
    The generated response string
  """
  creation_request = "budget $60 for gas every week for the next 6 months and a yearly car insurance cost of 3500 starting next year"
  
  return _run_test_with_logging(creation_request, None, generator)


def test_create_dining_out_budget(generator: CreateBudgetOrGoalOptimizer = None):
  """
  Test method for creating a dining out budget scenario.
  
  Args:
    generator: Optional CreateBudgetOrGoalOptimizer instance. If None, creates a new one.
    
  Returns:
    The generated response string
  """
  creation_request = "create a monthly budget of $500 for dining out"
  
  return _run_test_with_logging(creation_request, None, generator)


def test_create_savings_goal(generator: CreateBudgetOrGoalOptimizer = None):
  """
  Test method for creating a savings goal scenario.
  
  Args:
    generator: Optional CreateBudgetOrGoalOptimizer instance. If None, creates a new one.
    
  Returns:
    The generated response string
  """
  creation_request = "I want to save $5000 in the next 6 months"
  input_info = "Based on your current spending patterns, you can save approximately $800 per month."
  
  return _run_test_with_logging(creation_request, input_info, generator)


def test_create_ambiguous_budget(generator: CreateBudgetOrGoalOptimizer = None):
  """
  Test method for creating a budget with ambiguous category that requires clarification.
  
  Args:
    generator: Optional CreateBudgetOrGoalOptimizer instance. If None, creates a new one.
    
  Returns:
    The generated response string
  """
  creation_request = "create a budget for subscriptions"
  
  return _run_test_with_logging(creation_request, None, generator)


def test_create_incomplete_budget(generator: CreateBudgetOrGoalOptimizer = None):
  """
  Test method for creating a budget with missing information that requires clarification.
  
  Args:
    generator: Optional CreateBudgetOrGoalOptimizer instance. If None, creates a new one.
    
  Returns:
    The generated response string
  """
  creation_request = "create a budget for $200"
  
  return _run_test_with_logging(creation_request, None, generator)


def test_with_inputs(creation_request: str, input_info: str = None, generator: CreateBudgetOrGoalOptimizer = None):
  """
  Convenient method to test the generator with custom inputs.
  
  Args:
    creation_request: The creation request as a string
    input_info: Optional input info from previous skill
    generator: Optional CreateBudgetOrGoalOptimizer instance. If None, creates a new one.
    
  Returns:
    The generated response string
  """
  return _run_test_with_logging(creation_request, input_info, generator)


def main(batch: int = 1):
  """
  Main function to test the create budget/goal generator
  
  Args:
    batch: Batch number (1 or 2) to determine which tests to run
  """
  print("Testing CreateBudgetOrGoalOptimizer\n")
  
  if batch == 1:
    # Basic budget/goal creation tests
    print("Test 1: Creating gas and car insurance budget")
    print("-" * 80)
    test_create_gas_budget()
    print("\n")
    
    print("Test 2: Creating dining out budget")
    print("-" * 80)
    test_create_dining_out_budget()
    print("\n")
    
    print("Test 3: Creating savings goal with input info")
    print("-" * 80)
    test_create_savings_goal()
    print("\n")
  elif batch == 2:
    # Budget edge cases
    print("Test 1: Creating budget with ambiguous category (requires followup)")
    print("-" * 80)
    test_create_ambiguous_budget()
    print("\n")
    
    print("Test 2: Creating budget with missing information (requires followup)")
    print("-" * 80)
    test_create_incomplete_budget()
    print("\n")
  else:
    raise ValueError("batch must be 1 or 2")
  
  print("All tests completed!")


if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser(description='Run tests in batches')
  parser.add_argument('--batch', type=int, default=1, choices=[1, 2],
                      help='Batch number to run (1 or 2)')
  args = parser.parse_args()
  main(batch=args.batch)


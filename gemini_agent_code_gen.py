from google import genai
from google.genai import types
import os
import time
from typing import Dict, List, Optional
from dotenv import load_dotenv
from datetime import datetime
import sandbox
from database import Database

# Load environment variables
load_dotenv()

class GeminiAgentCodeGen:
  """Handles all Gemini API interactions for code generation"""
  
  def __init__(self, model_name="gemini-2.0-flash"):
    """Initialize the Gemini agent with API configuration for code generation"""
    # API Configuration
    self.client = genai.Client(api_key=os.getenv('GEMINI_API_KEY'))
    
    # Model Configuration
    if "-thinking" in model_name:
      self.thinking_budget = 2048
      self.model_name = model_name.replace("-thinking", "")
    else:
      self.thinking_budget = 0
      self.model_name = model_name
    
    # Generation Configuration Constants
    self.temperature = 0.2
    self.top_p = 0.95
    self.max_output_tokens = 1024
    self.thinking_budget_default = 2048
    
    # Safety Settings
    self.safety_settings = [
      types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"),
      types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"),
      types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"),
      types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="OFF")
    ]
    
    # System Prompt
    self.system_prompt = """Your name is "Penny" and you are a helpful AI specialized in code generation. **You only output python code.**
Write a function `process_input` that takes no arguments and print()s what to tell the user and returns a tuple:
  - The first element the boolean success or failure of the function.
  - The second element is the metadata for the entities created or retrieved.
- Compute for dates using `datetime` package.  Assume `import datetime` is already included.
- When looking for `account_name` and `==`, look for other relevant variations like fuzzy search to find more matches. Refer to <ACCOUNT_NAMES> for the list of account names.

<IMPLEMENTED_FUNCTIONS>
  These functions are already implemented:
    - `retrieve_accounts(): pd.DataFrame`
        - retrieves all accounts and returns a pandas DataFrame.  It may be empty if no accounts exist.
        - Panda's dataframe columns: `account_id`, `account_type`, `account_name`, `balance_available`, `balance_current`, `balance_limit`
        - `account_id` is a integer type, `account_type` and `account_name` are str types, `balance_available`, `balance_current` and `balance_limit` are float types.
    - `account_names_and_balances(df: pd.DataFrame, template: str) -> tuple[str, list]`
        - takes filtered `df` and generates a formatted string based on `template` and returns metadata.
    - `utter_account_totals(df: pd.DataFrame, template: str) -> str`
        - takes filtered `df` and calculates total balances and returns a formatted string based on `template`.
    - `retrieve_transactions() -> pd.DataFrame`
        - retrieves all transactions and returns a pandas DataFrame.  It may be empty if no transactions exist.
        - Panda's dataframe columns: `transaction_id`, `user_id`, `account_id`, `date`, `transaction_name`, `amount`, `category`
        - `transaction_id`, `account_id`, and `user_id` are integer types, `transaction_name` and `category` are str types, `amount` is a float type, `date` is a pandas datetime64 (pd.Timestamp) type.
        - `amount` values are negative for spending/outflow, positive for receiving/inflow.
    - `transaction_names_and_amounts(df: pd.DataFrame, template: str) -> tuple[str, list]`
        - takes filtered `df` and generates a formatted string based on `template` and returns metadata.
    - `utter_transaction_totals(df: pd.DataFrame, is_spending: bool, template: str) -> str`
        - takes filtered `df`, `is_spending` flag (True for spending categories, False for income), and `template` string, calculates total transaction amounts and returns a formatted string.
</IMPLEMENTED_FUNCTIONS>

<ACCOUNT_TYPE>
  These are the `account_type` in the `accounts` table:
    - `deposit_savings`: savings, cash accounts
    - `deposit_money_market`: money market accounts
    - `deposit_checking`: checking, debit accounts
    - `credit_card`: credit cards
    - `loan_home_equity`: loans tied to home equity
    - `loan_line_of_credit`: personal loans extended by banks
    - `loan_mortgage`: home mortgage accounts
    - `loan_auto`: auto loan accounts
</ACCOUNT_TYPE>

<CATEGORY>
  These are the `category` in the `transactions` table:
    - `income` for salary, bonuses, interest, side hussles and business.
        - `income_salary` for regular paychecks and bonuses.
        - `income_sidegig` for side hussles like Uber, Etsy and other gigs.
        - `income_business` for business income and spending.
        - `income_interest` for interest income from savings or investments.
    - `meals` for all types of food spending includes groceries, dining-out and delivered food.
        - `meals_groceries` for supermarkets and other unprepared food marketplaces.
        - `meals_dining_out` for food prepared outside the home like restaurants and takeout.
        - `meals_delivered_food` for prepared food delivered to the doorstep like DoorDash.
    - `leisure` for all relaxation or recreation and travel activities.
        - `leisure_entertainment` for movies, concerts, cable and streaming services.
        - `leisure_travel`for flights, hotels, and other travel expenses.
    - `bills` for essential payments for services and recurring costs.
        - `bills_connectivity` for internet and phone bills.
        - `bills_insurance` for life insurance and other insurance payments.
        - `bills_tax`  for income, state tax and other payments.
        - `bills_service_fees` for payments for services rendered like professional fees or fees for a product.
    - `shelter` for all housing-related expenses including rent, mortgage, property taxes and utilities.
        - `shelter_home` for rent, mortgage, property taxes.
        - `shelter_utilities` for electricity, water, gas and trash utility bills.
        - `shelter_upkeep` for maintenance and repair and improvement costs for the home.
    - `education` for all learning spending including kids after care and activities.
        - `education_kids_activities` for after school activities, sports and camps.
        - `education_tuition` for school tuition, daycare and other education fees.
    - `shopping` for discretionary spending on clothes, electronics, home goods, etc.
        - `shopping_clothing` for clothing, shoes, accessories and other wearable items.
        - `shopping_gadgets` for electronics, gadgets, computers, software and other tech items.
        - `shopping_kids` for kids clothing, toys, school supplies and other kid-related items.
        - `shopping_pets` for pet food, toys, grooming, vet bills and other pet-related items.
    - `transportation` for public transportation, car payments, gas and maintenance and car insurance.
        - `transportation_public` for bus, train, subway and other public transportation.
        - `transportation_car` for car payments, gas, maintenance and car insurance.
    - `health` for medical bills, pharmacy spending, insurance, gym memberships and personal care.
        - `health_medical_pharmacy` for doctor visits, hospital, meds and health insurance costs.
        - `health_gym_wellness` for gym memberships, personal training and spa services.
        - `health_personal_care` for haircuts, beauty products and beuaty services.
    - `donations_gifts` for charitable donations, gifts and other giving to friends and family.
    - `uncategorized` for explicitly tagged as not yet categorized or unknown.
    - `transfers` for moving money between accounts or paying off credit cards or loans.
    - `miscellaneous` for explicitly tagged as miscellaneous.
</CATEGORY>
"""
    # - `retrieve_transactions(): pd.DataFrame`
    #     - retrieves all transactions componsed of income and expenses and returns a pandas DataFrame.  It may be empty if no transactions exist.
    #     - Panda's dataframe columns: `transaction_id`, `transaction_datetime`, `transaction_name`, `amount`, `category`, `account_id`
    #     - `transaction_datetime` is a datetime type.


    # - `create_reminder(title: str, reminder_datetime: str): tuple[str, dict | None]`
    #     - This function creates a reminder for `title` at `reminder_datetime`.
    #     - Returns a tuple:
    #       - The first element returns the failure message if the function fails, otherwise returns None.
    #       - The second element is the metadata for the entities created or retrieved.
    # - `update_reminder(reminder_id: int, new_title: str = None, new_reminder_datetime: str = None): tuple[str, dict | None]`
    #     - This function updates a reminder's title and/or reminder time.
    #     - Returns a tuple:
    #       - The first element returns the failure message if the function fails, otherwise returns None.
    #       - The second element is the metadata for the entities updated.
    # - `delete_reminder(reminder_id: int): tuple[str, dict | None]`
    #     - This function deletes a reminder by its ID.
    #     - Returns a tuple:
    #       - The first element returns the failure message if the function fails, otherwise returns None.
    #       - The second element is the metadata for the deleted reminder.

    # - `utter_delta_from_now(future_time: datetime) -> str`: Utter the time delta in a human-readable format
    # - `reminder_data(reminder: dict) -> dict`: Return reminders as a metadata formatted dict
    # - `turn_on_off_device(device_id: str, set_on: bool): tuple[str, dict | None]`
    #     - This function turns a home device (light or thermostat) on or off.
    #     - Returns a tuple:
    #       - The first element returns the failure message if the function fails, otherwise returns None.
    #       - The second element is the metadata for the device that was turned on/off.
    # - `retrieve_home_devices(): pd.DataFrame`
    #     - This function retrieves all home devices and returns a pandas DataFrame.  It may be empty if no devices exist.
    #     - Panda's dataframe columns: `device_id`, `name`, `device_type`, `room_name`, `user_id`, `is_on`, `brightness`, `color_name`, `target_temperature`
    #     - `device_id` is a str type, `is_on` is a boolean type, `brightness` is an integer (0-100), `target_temperature` is a float.

  
  def _build_account_names_section(self, user_id: int) -> str:
    """
    Build the ACCOUNT_NAMES section dynamically based on the user's accounts.
    
    Args:
      user_id: The user ID to retrieve accounts for
      
    Returns:
      String containing the ACCOUNT_NAMES section
    """
    db = Database()
    accounts = db.get_accounts_by_user(user_id)
    
    if not accounts:
      return "<ACCOUNT_NAMES>\n  These are the `account_name` in the `accounts` table:\n    (No accounts found for this user)\n</ACCOUNT_NAMES>"
    
    account_names_list = []
    for account in accounts:
      account_name = account.get('account_name', '')
      if account_name:
        account_names_list.append(f"    - `{account_name}`")
    
    account_names_text = "\n".join(account_names_list) if account_names_list else "    (No account names found)"
    
    return f"""<ACCOUNT_NAMES>
  **REFERENCE ONLY** - These are example `account_name` values in the `accounts` table. This is NOT code - use `retrieve_accounts()` to get the actual DataFrame.
  Account names:
{account_names_text}
</ACCOUNT_NAMES>"""

  def _create_few_shot_examples(self) -> str:
    """
    Create few-shot examples for code generation.
    
    Returns:
      List of example dictionaries with user input and expected code output
    """
    # todays_date = datetime.now().strftime("%Y-%m-%d")
    return """input: User: how much left in checking
output: ```python
def process_input():
    df = retrieve_accounts()
    metadata = {"accounts": []}
    
    if df.empty:
      print("You have no accounts.")
    else:
      # Filter for checking accounts
      df = df[df['account_type'] == 'deposit_checking']
      
      if df.empty:
        print("You have no checking accounts.")
      else:
        print("Here are your checking account balances:")
        for_print, metadata["accounts"] = account_names_and_balances(df, "Account \\"{name}\\" has {balance_current} left with {balance_available} available now.")
        print(for_print)
        print(utter_account_totals(df, "Across all checking accounts, you have {balance_current} left."))
    
    return True, metadata
```

input: User: what is my net worth
output: ```python
def process_input():
    accounts_df = retrieve_accounts()
    metadata = {}

    if accounts_df.empty:
        print("You have no accounts to calculate net worth.")
    else:
        # List of asset account types for net worth calculation
        asset_types = ['deposit_savings', 'deposit_money_market', 'deposit_checking']
        liability_types = ['credit_card', 'loan_home_equity', 'loan_line_of_credit', 'loan_mortgage', 'loan_auto']

        # Filter for assets and liabilities
        assets_df = accounts_df[accounts_df['account_type'].isin(asset_types)]
        liabilities_df = accounts_df[accounts_df['account_type'].isin(liability_types)]

        total_assets = assets_df['balance_current'].sum()
        total_liabilities = liabilities_df['balance_current'].sum()
        # net worth is the sum of assets minus liabilities
        net_worth = total_assets - total_liabilities
        print(f"You have a net worth of ${net_worth:,.0f} with assets of ${total_assets:,.0f} and liabilities of ${total_liabilities:,.0f}.")

    return True, metadata
```

input: User: how much did i spend on streaming last month?
output: ```python
def process_input():
    from datetime import datetime
    import pandas as pd
    
    df = retrieve_transactions()
    metadata = {"transactions": []}
    
    if df.empty:
      print("You have no transactions.")
    else:
      # Filter for streaming/entertainment category
      df = df[df['category'] == 'leisure_entertainment']
      
      if df.empty:
        print("You have no streaming/entertainment transactions.")
      else:
        # Filter for last month
        now = datetime.now()
        # First day of last month
        first_day_last_month = (now.replace(day=1) - pd.DateOffset(months=1))
        # Last day of last month
        last_day_last_month = now.replace(day=1) - pd.DateOffset(days=1)
        
        # Filter transactions from last month
        df = df[(df['date'] >= first_day_last_month) & (df['date'] <= last_day_last_month)]
        
        if df.empty:
          print("You have no streaming/entertainment transactions from last month.")
        else:
          # Calculate and display total spending
          total_str = utter_transaction_totals(df, True, "You {verb} ${total_amount:,.2f} on streaming/entertainment last month.")
          print(total_str)
    
    return True, metadata
```"""




  def generate_response(self, messages: List[Dict], timing_data: Dict, user_id: int = 1) -> Dict:
    """
    Generate a response using Gemini API with timing tracking for code generation.
    Uses GenAI API to construct the prompt with few-shot examples.
    
    Args:
      messages: The user/assistant messages
      timing_data: Dictionary to store timing information
      user_id: User ID for sandbox execution
      
    Returns:
      Dictionary with response text and timing data
    """
    # Filter messages from the last 1 minute (60 seconds)
    current_time = time.time()
    recent_messages = []
    
    for msg in messages:
      # Check if message has request_time and is within 60 seconds
      if "request_time" in msg and (current_time - msg["request_time"]) <= 60:
        recent_messages.append(msg)
      # If no request_time, include it (backward compatibility)
      elif "request_time" not in msg:
        recent_messages.append(msg)
    
    # Format recent messages with role prefixes
    formatted_messages = []
    for msg in recent_messages:
      role = msg.get('role', 'user')
      content = msg.get('content', '')
      if role == 'user':
        formatted_messages.append(f"User: {content}")
      elif role == 'assistant':
        formatted_messages.append(f"Assistant: {content}")
    
    # Join all recent messages
    recent_conversation = "\n".join(formatted_messages)
    print(recent_conversation)
    
    gemini_start = time.time()
    
    # Create few-shot examples and request text
    few_shot_examples = self._create_few_shot_examples()
    request_text = types.Part.from_text(text=f"""<EXAMPLES>
{few_shot_examples}
</EXAMPLES>

input: {recent_conversation}
output:""")
    
    # Create content and configuration
    contents = [types.Content(role="user", parts=[request_text])]
    
    # Build dynamic ACCOUNT_NAMES section for this user
    account_names_section = self._build_account_names_section(user_id)
    # Combine system prompt with dynamic account names
    full_system_prompt = self.system_prompt + "\n\n" + account_names_section
    
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
      output_text += chunk.text
    gemini_end = time.time()
    
    # Execute the generated code in sandbox
    try:
      success, utter, metadata, logs = sandbox.execute_agent_with_tools(output_text, user_id)
    except Exception as e:
      # Extract logs from error message if available
      error_str = str(e)
      logs = ""
      if "Captured logs:" in error_str:
        logs = error_str.split("Captured logs:")[-1].strip()
      success = False
      utter = f"Error executing code: {error_str}"
      metadata = {"error": error_str}
    
    execution_end = time.time()
    
    # Record timing data
    timing_data['gemini_api_calls'].append({
      'call_number': 1,
      'start_time': gemini_start,
      'end_time': gemini_end,
      'duration_ms': (gemini_end - gemini_start) * 1000
    })
    timing_data['execution_time'].append({
      'call_number': 1,
      'start_time': gemini_end,
      'end_time': execution_end,
      'duration_ms': (execution_end - gemini_end) * 1000
    })
    
    return {
      'response': utter,
      'function_called': None,
      'execution_success': success,
      'execution_metadata': metadata,
      'code_generated': output_text,
      'logs': logs
    }

  
  def generate_response_with_function_calling(self, messages: List[Dict], timing_data: Dict, 
                                            function_handler: callable, user_id: int = 1) -> Dict:
    """
    Generate a response using Gemini API for code generation.
    Note: This method doesn't use function calling for code generation.
    
    Args:
      messages: The user/assistant messages
      timing_data: Dictionary to store timing information
      function_handler: Function to handle function calls (not used in code gen)
      user_id: User ID for sandbox execution
      
    Returns:
      Dictionary with response text and timing data
    """
    return self.generate_response(messages, timing_data, user_id)
  
  def get_available_models(self) -> List[str]:
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


def create_gemini_agent_code_gen(model_name="gemini-2.0-flash"):
  """Create a new Gemini agent code gen instance with the specified model"""
  return GeminiAgentCodeGen(model_name)

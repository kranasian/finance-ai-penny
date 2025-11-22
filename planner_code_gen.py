from google import genai
from google.genai import types
import os
import time
from typing import Dict, List, Optional
from dotenv import load_dotenv
import sandbox
from database import Database

# Load environment variables
load_dotenv()

# Import planner skill functions for sandbox
from penny.tool_funcs.lookup_user_accounts_transactions_income_and_spending_patterns import lookup_user_accounts_transactions_income_and_spending_patterns
from penny.tool_funcs.create_budget_or_goal_or_reminder import create_budget_or_goal_or_reminder
from penny.tool_funcs.research_and_strategize_financial_outcomes import research_and_strategize_financial_outcomes
from penny.tool_funcs.update_transaction_category_or_create_category_rules import update_transaction_category_or_create_category_rules

SYSTEM_PROMPT = """You are a financial planner agent very good at understanding conversation.

## Your Tasks

1. **Prioritize the Last User Request**: Your main goal is to create a plan that directly addresses the **Last User Request**.
2. **Use Previous Conversation for Context ONLY**:
    - If the **Last User Request** is a follow-up (e.g., "yes, do that"), use the context.
    - If the **Last User Request** is vague (e.g., "what about the other thing?"), use the context.
    - **If the Last User Request is a new, general question (e.g., "how's my accounts doing?"), DO NOT use specific details from the Previous Conversation in your plan.**
3. **Gather Data When Useful**: When calling `create_budget_or_goal_or_reminder`, `research_and_strategize_financial_outcomes`, or `update_transaction_category_or_create_category_rules` would benefit from financial data (accounts, transactions, subscriptions, spending patterns), first call `lookup_user_accounts_transactions_income_and_spending_patterns` and pass the result as `input_info`. Only call lookup when the data would be useful for the subsequent function.
4. **Create a Focused Plan**: The steps in your plan should only be for achieving the **Last User Request**. Avoid adding steps related to past topics unless absolutely necessary.
5. **Output Python Code**: The plan must be written as a Python function `execute_plan`.

Write a python function `execute_plan` that takes no arguments:
  - Express actionable steps as **calls to skill functions**, passing in a natural language request and optionally another input from another skill.
  - Do not use other python functions, just available skill functions, conditional operations and string concatenations.

<AVAILABLE_SKILL_FUNCTIONS>

These are the **available skills** that can be stacked and sequenced using `input_info`.
- All of these skills can accept **multiple requests**, written as multiple sentences in their request parameters.
- All **skill functions** return a `tuple[bool, str]`.
	- The first element is `success` boolean.  `True` if information was found and the request was achieved and `False` if information not available, an error occured or more information needed from the user.
	- The second element is `output_info` string.  It contains the output of the **skill function** that can be used as `input_info` in a following skill function call.

### List of all Skill Functions

- `lookup_user_accounts_transactions_income_and_spending_patterns(lookup_request: str, input_info: str = None) -> tuple[bool, str]`
  - `lookup_request` is the detailed information requested, written in natural language to lookup about the user's accounts, transactions including income and spending, subscriptions and compare them.
  - Lookup request can also be about expected and future weekly/monthly income or spending.  Lookup request must phrase the best natural language output needed towards the plan to answer the user.
- `create_budget_or_goal_or_reminder(creation_request: str, input_info: str = None) -> tuple[bool, str]`
  - `creation_request` is what needs to be created factoring in the information coming in from `input_info`.  The request must be descriptive and capture the original user request.
  - Function output `str` is the detail of what was created.
  - If more information is needed from the user, `success` will be `False` and the information needed will be in the output `str` second element.
- `research_and_strategize_financial_outcomes(strategize_request: str, input_info: str = None) -> tuple[bool, str]`
  - `strategize_request` is what needs to thought out, planned or strategized.  It can contain research information like "average dining out for a couple in Chicago, Illinois" and factoring in information from `input_info`.
  - This skill can financially plan for the future, lookup feasibility and overall provide assessment of different simulated outcomes with finances.
- `update_transaction_category_or_create_category_rules(categorize_request: str, input_info: str = None) -> tuple[bool, str]`
  - `categorize_request` is a description of the category rule that needs to be created, or the description of the transaction that needs to be recategorized.  This can be a single transaction, or a group of transactions with a criteria.
  - If user hints at doing this in the future as well, specify that a category rule needs to be created on top of updating transaction categories.

</AVAILABLE_SKILL_FUNCTIONS>

<EXAMPLES>

input: **Last User Request**: what can I do? definitely kill that Netflix which is for my entertainment btw.  Please fix categories of that  and Tell me what's the best plan to get to save $5K.
**Previous Conversation**:
User: Will I be able to pay for rent?
Assistant: Not yet but close.  Your checking accounts only total $840 but rent is expected to be $980.
User: Have I been spending more than what I earn?
Assistant: Yep, slightly.  You're spending $230 more than you earn.
output:
```python
def execute_plan() -> tuple[bool,  str]:
    # Goal: Get the necessary data for the savings plan.
    success, lookup_result = lookup_user_accounts_transactions_income_and_spending_patterns(
        lookup_request="Find all 'Netflix' transactions. Get a summary of monthly income and spending for the last 3 months."
    )
    if not success:
        return False, lookup_result

    # Goal: Correct the categorization of Netflix transactions.
    success, category_result = update_transaction_category_or_create_category_rules(
        categorize_request="Recategorize all 'Netflix' transactions as 'Entertainment' and create a rule for future ones.",
        input_info=lookup_result
    )
    if not success:
        pass 

    # Goal: Develop a strategy to save $5,000.
    return research_and_strategize_financial_outcomes(
        strategize_request="Create a detailed savings plan to save $5,000. Specify a timeline and a monthly savings target, accounting for the canceled Netflix subscription.",
        input_info=lookup_result
    )
```

input: **Last User Request**: need to save up to fix my car for $2000
**Previous Conversation**:
User: I've just been to Disneyland. Add up the total damage.  
Assistant: Yeah, its significant but manageable for a family. For the past 2 weeks, travel spending was $2,344 and dining out was $844.
output:
```python
def execute_plan() -> tuple[bool,  str]:
    # Goal: Find where to save money for the car repair.
    success, lookup_result = lookup_user_accounts_transactions_income_and_spending_patterns(
        lookup_request="Analyze discretionary spending from the last 3 months to find areas to cut back for a $2000 car repair."
    )
    if not success:
        return False, lookup_result

    # Goal: Create a realistic savings plan.
    success, goals_plan = research_and_strategize_financial_outcomes(
        strategize_request="Develop a savings plan to reach $2000. The plan should propose a timeline and suggest specific spending reductions.",
        input_info=lookup_result
    )
    if not success:
        return False, goals_plan
    
    output_message = f"I have developed a savings plan for you: {goals_plan}\n"

    # Goal: Implement the savings plan by creating a budget.
    success, budget_result = create_budget_or_goal_or_reminder(
        creation_request="Create a budget based on the new savings plan to track your progress.",
        input_info=f"{lookup_result}\n\n{goals_plan}"
    )
    if not success:
        output_message += f"I was unable to create the budget automatically: {budget_result}"
        return False, output_message
    
    output_message += f"To help you stay on track, I have created this budget: {budget_result}"
    return True, output_message
```

input: **Last User Request**: remind me to cancel Spotify subscription at the end of this year.
**Previous Conversation**:

output:
```python
def execute_plan() -> tuple[bool, str]:
    # Goal: Gather all necessary information about accounts, transactions, and subscriptions.
    success, lookup_result = lookup_user_accounts_transactions_income_and_spending_patterns(
        lookup_request="Find all Spotify subscriptions and transactions. Get account information and subscription details."
    )
    if not success:
        return False, lookup_result
    
    # Goal: Create a reminder with all the necessary data already gathered.
    return create_budget_or_goal_or_reminder(
        creation_request="Create a reminder to cancel Spotify subscription at the end of this year (December 31st).",
        input_info=lookup_result
    )
```

</EXAMPLES>
"""

class PlannerCodeGen:
  """Handles all Gemini API interactions for planner-based code generation"""
  
  def __init__(self, model_name="gemini-flash-lite-latest"):
    """Initialize the Gemini agent with API configuration for planner code generation"""
    # API Configuration
    self.client = genai.Client(api_key=os.getenv('GEMINI_API_KEY'))
    
    # Model Configuration
    if "-thinking" in model_name:
      self.thinking_budget = 4096
      self.model_name = model_name.replace("-thinking", "")
    else:
      self.thinking_budget = 4096
      self.model_name = model_name
    
    # Generation Configuration Constants
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

  def _format_conversation_for_planner(self, messages: List[Dict]) -> tuple[str, str]:
    """
    Format messages into Last User Request and Previous Conversation format.
    
    Args:
      messages: List of message dictionaries with 'role' and 'content'
      
    Returns:
      Tuple of (last_user_request, previous_conversation)
    """
    if not messages:
      return "", ""
    
    # Find the last user message
    last_user_request = ""
    previous_conversation_parts = []
    
    for msg in messages:
      role = msg.get('role', '')
      content = msg.get('content', '')
      
      if role == 'user':
        last_user_request = content
        # Build previous conversation up to (but not including) the last user message
        previous_conversation_parts = []
        for prev_msg in messages[:messages.index(msg)]:
          prev_role = prev_msg.get('role', '')
          prev_content = prev_msg.get('content', '')
          if prev_role == 'user':
            previous_conversation_parts.append(f"User: {prev_content}")
          elif prev_role == 'assistant':
            previous_conversation_parts.append(f"Assistant: {prev_content}")
    
    previous_conversation = "\n".join(previous_conversation_parts)
    
    return last_user_request, previous_conversation

  def _extract_python_code(self, text: str) -> str:
    """Extract Python code from generated response (look for ```python blocks)."""
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


  def generate_response(self, messages: List[Dict], timing_data: Dict, user_id: int = 1) -> Dict:
    """
    Generate a response using Gemini API with timing tracking for planner code generation.
    
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
    
    # Format messages for planner
    last_user_request, previous_conversation = self._format_conversation_for_planner(recent_messages)
    
    if not last_user_request:
      # No user message found, use the last message content
      if recent_messages:
        last_user_request = recent_messages[-1].get('content', '')
    
    gemini_start = time.time()
    
    # Create request text in planner format
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
      thinking_config=types.ThinkingConfig(thinking_budget=self.thinking_budget),
    )

    # Generate response
    output_text = ""
    output_tokens = 0
    last_chunk = None
    for chunk in self.client.models.generate_content_stream(
      model=self.model_name,
      contents=contents,
      config=generate_content_config,
    ):
      if chunk.text is not None:
        output_text += chunk.text
      last_chunk = chunk
    
    # Extract usage metadata from the last chunk if available
    if last_chunk:
      if hasattr(last_chunk, 'usage_metadata') and last_chunk.usage_metadata:
        output_tokens = getattr(last_chunk.usage_metadata, 'output_token_count', 0) or getattr(last_chunk.usage_metadata, 'candidates_token_count', 0)
      elif hasattr(last_chunk, 'candidates') and last_chunk.candidates:
        for candidate in last_chunk.candidates:
          if hasattr(candidate, 'usage_metadata') and candidate.usage_metadata:
            output_tokens = getattr(candidate.usage_metadata, 'output_token_count', 0) or getattr(candidate.usage_metadata, 'candidates_token_count', 0)
            break
    
    gemini_end = time.time()
    
    # Store output tokens in timing data
    timing_data['output_tokens'] = output_tokens
    
    # Execute the generated code in sandbox
    # Note: execute_planner_with_tools will extract code from markdown if needed,
    # but we've already wrapped it, so we pass the wrapped code directly
    try:
      success, message, captured_output, logs = sandbox.execute_planner_with_tools(output_text, user_id)
    except Exception as e:
      # Extract logs from error message if available
      error_str = str(e)
      logs = ""
      if "Captured logs:" in error_str:
        logs = error_str.split("Captured logs:")[-1].strip()
      success = False
      message = f"Error executing code: {error_str}"  
    
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
      'response': message,
      'function_called': None,
      'execution_success': success,
      'execution_message': message,
      'execution_captured_output': captured_output,
      'code_generated': output_text,
      'logs': logs
    }

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


def create_planner_code_gen(model_name="gemini-flash-lite-latest"):
  """Create a new Planner code gen instance with the specified model"""
  return PlannerCodeGen(model_name)


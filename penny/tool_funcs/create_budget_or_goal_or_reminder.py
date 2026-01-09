from typing import Tuple
import os
import sys

# Add parent directories to path to import CreateBudgetOrGoalOrReminder and sandbox
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

experiments_dir = os.path.join(parent_dir, 'experiments')
if experiments_dir not in sys.path:
    sys.path.insert(0, experiments_dir)

from create_budget_or_goal_or_reminder import CreateBudgetOrGoalOrReminder
import sandbox


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


def create_budget_or_goal_or_reminder(
    creation_request: str,
    input_info: str = None,
    **kwargs
) -> Tuple[bool, str]:
    """
    Create a budget, goal, or reminder based on a natural language request.
    
    This is a wrapper function that uses CreateBudgetOrGoalOrReminder to generate
    and execute code for creating budgets, goals, or reminders.
    
    Args:
        creation_request: Natural language request for what to create
                          (e.g., "set a food budget of $500 for next month",
                          "remind me to cancel Netflix on November 30th")
        input_info: Optional input from another skill function
        **kwargs: Additional arguments (user_id, etc.)
        
    Returns:
        Tuple[bool, str]: (success, output_info)
        - success: True if creation was successful, False if clarification is needed
        - output_info: Success message or clarification prompts
    """
    # Use the optimizer to generate code
    generator = CreateBudgetOrGoalOrReminder()
    generated_code = generator.generate_response(creation_request, input_info)
    
    # Extract Python code from the response
    code = extract_python_code(generated_code)
    
    # Execute the generated code in sandbox
    user_id = kwargs.get('user_id', 1)
    try:
        success, output_string, logs, goals_list = sandbox.execute_agent_with_tools(code, user_id)
        return success, output_string
    except Exception as e:
        import traceback
        error_msg = f"**Execution Error**: `{str(e)}`\n{traceback.format_exc()}"
        return False, error_msg

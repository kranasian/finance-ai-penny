from google import genai
from google.genai import types
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

SYSTEM_PROMPT = """You are a financial planner agent very good at understanding conversation.

## Your Tasks

1. **Prioritize the Last User Request**: Your main goal is to create a plan that directly addresses the **Last User Request**.
2. **Use Previous Conversation for Context ONLY**:
    - If the **Last User Request** is a follow-up (e.g., "yes, do that"), use the context.
    - If the **Last User Request** is vague (e.g., "what about the other thing?"), use the context.
    - **If the Last User Request is a new, general question (e.g., "how's my accounts doing?"), DO NOT use specific details from the Previous Conversation in your plan.**
3. **Create a Focused Plan**: The steps in your plan should only be for achieving the **Last User Request**. Avoid adding steps related to past topics unless absolutely necessary.
4. **Output Python Code**: The plan must be written as a Python function `execute_plan`.

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
    # Step 1: Lookup current income and spending needed to build the plan and find Netflix transaction details
    success, lookup_result = lookup_user_accounts_transactions_income_and_spending_patterns(
        lookup_request="Provide my last 3 months of monthly income.  Categorized breakdown of the key categories of monthly spending in the last 3 months to validate the $230 of overspending. List transactions  matching 'netflix' across all my transactions."
    )
    if not success:
        return False, lookup_result

    # Step 2: First fix categorization of Netflix to entertainment and make a category rule
    success, category_result = update_transaction_category_or_create_category_rules(
        categorize_request="Update categorization of 'netflix' to be entertainment for all transactions.  Create a category rule that matches the netflix transaction name.",
        input_info=lookup_result
    )
    if not success:
        # Even if categorization fails, need to proceed and just tell the user why in the final result
        pass 

    # Step 3: Develop the strategy to save $5,000, incorporating the elimination of the deficit and after the Netflix categorization and future cancellation.
    return research_and_strategize_financial_outcomes(
        strategize_request="Develop a concrete financial plan detailing the timeline and required monthly surplus needed to save $5,000. This plan must account for eliminating the existing monthly deficit and factoring in the recurring savings achieved by canceling the Netflix subscription.",
        # The result of categorization is irrelevant so only use lookup_result.
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
    # Step 1: Lookup discretionary monthly spending in the last 3 months to assess feasibility of saving $2000.
    success, lookup_result = lookup_user_accounts_transactions_income_and_spending_patterns(
        lookup_request="Retrieve last 3 months of discretionary monthly spending patterns to identify how many months is realistically achievable to save $2000."
    )
    if not success:
        return False, lookup_result

    # Step 2: Develop the strategy to save $2000 and propose a reasonable timeline, factoring in current monthly spending patterns.
    success, goals_plan = research_and_strategize_financial_outcomes(
        strategize_request="Looking at the discretionary spending, develop a feasible, realistic spending plan for the discretionary categories to be able to save $2000 and propose reasonable timeline.",
        input_info=lookup_result
    )
    if not success:
        return False, goals_plan
    
    output_message = f"The proposed spending plan is: {goals_plan}\n"

    # Step 3: Create the budget across the proposed spending plan in the reasonable timeframe.
    success, budget_result = create_budget_or_goal_or_reminder(
        strategize_request="Create the budgets in the proposed spending plan in accordance with the proposed timeline.",
        input_info=goals_plan
    )
    if not success:
        output_message += f"But: {budget_result}"
        return False, output_message
    
    output_message += f"I have created these spending budgets: {budget_result}"
    return True, output_message
```

</EXAMPLES>
"""

class PlannerOptimizer:
  """Handles all Gemini API interactions for financial planning and optimization"""
  
  def __init__(self, model_name="gemini-flash-lite-latest"):
    """Initialize the Gemini agent with API configuration for financial planning"""
    # API Configuration
    self.client = genai.Client(api_key=os.getenv('GEMINI_API_KEY'))
    
    # Model Configuration
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

  
  def generate_response(self, last_user_request: str, previous_conversation: str) -> str:
    """
    Generate a response using Gemini API for financial planning.
    
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


def main():
  """Main function to test the planner optimizer with a sample input"""
  # Create planner optimizer instance
  planner = PlannerOptimizer()
  
  # Last user request
  last_user_request = "how's my accounts doing?"
  
  # Previous conversation as a string
  previous_conversation = """User: Will I be able to pay for rent?
Assistant: Not yet but close.  Your checking accounts only total $603 but rent is expected to be $2233.
User: Have I been spending more than what I earn?
Assistant: No, but its close.  You only save about $33.
User: I've just been to Europe.  How was my spending?
Assistant: For the last week, travel was $1,344 and eating out was $443."""
  
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
  
  # Generate response
  print("=" * 80)
  print("Sending request to LLM...")
  print("=" * 80)
  print()
  
  result = planner.generate_response(last_user_request, previous_conversation)
  
  # Print the output
  print("=" * 80)
  print("LLM OUTPUT:")
  print("=" * 80)
  print(result)
  print("=" * 80)


if __name__ == "__main__":
  main()

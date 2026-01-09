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
from penny.tool_funcs.create_budget_or_goal_or_reminder import create_budget_or_goal_or_reminder, CreateBudgetOrGoalOrReminder
from penny.tool_funcs.create_budget_or_goal_or_reminder import extract_python_code
from penny.tool_funcs.research_and_strategize_financial_outcomes import research_and_strategize_financial_outcomes
from penny.tool_funcs.update_transaction_category_or_create_category_rules import update_transaction_category_or_create_category_rules
from penny.tool_funcs.add_to_memory import add_to_memory
from penny.tool_funcs.follow_up_conversation import follow_up_conversation

# Load environment variables
load_dotenv()

SYSTEM_PROMPT = """You are a financial planner agent very good at understanding conversation.

## Your Tasks

1. **Prioritize the Last User Request**: Your main goal is to create a plan that directly addresses the **Last User Request**.
2. **Use Previous Conversation for Context ONLY**:
    - If the **Last User Request** is a follow-up (e.g., "yes, do that"), use the context.
    - If the **Last User Request** is vague (e.g., "what about the other thing?"), use the context.
    - **CRITICAL**: For all skill requests, thoroughly analyze the `Previous Conversation` to gain an accurate understanding of the user's intent, identify any unresolved issues, and ensure the request parameter of the skill function is comprehensive and contextually relevant.
    - **If the Last User Request is a new, general question (e.g., "how's my accounts doing?"), DO NOT use specific details from the Previous Conversation in your plan.**
3. **Create a Focused Plan**: The steps in your plan should only be for achieving the **Last User Request**. Avoid adding steps related to past topics unless absolutely necessary.
4. **Output Python Code**: The plan must be written as a Python function `execute_plan`.

Write a python function `execute_plan` that takes no arguments:
  - Express actionable steps as **calls to skill functions**, passing in a natural language request and optionally another input from another skill.
  - Do not use other python functions, just available skill functions, conditional operations and string concatenations.
  - **CRITICAL**: Always check the `success` boolean returned by each skill function call. If `success` is `False`, handle the error appropriately (e.g., return early with the error message, or handle gracefully). Do NOT ignore the success status - it indicates whether the operation completed successfully or needs user clarification.

## Critical Efficiency Rules

**1. Prioritize `lookup_user_accounts_transactions_income_and_spending_patterns` for ALL Data-Related Inquiries, Including Comparisons, Summaries, and Calculations on User Data:**
- **If the Last User Request requires ANY user account, transaction, income, or spending data, or asks for comparisons, summaries, or calculations based on this user data, you MUST call `lookup_user_accounts_transactions_income_and_spending_patterns` FIRST.**
- Even if Previous Conversation contains some financial information, if the request needs current/fresh user data, involves a comparison (e.g., "compare X to Y"), a summary (e.g., "summarize my spending"), or a calculation (e.g., "calculate my savings rate") on user data, you MUST call lookup FIRST.
- For any question about the user's financial status, accounts, transactions, spending, income, or requests involving comparisons, summaries, or calculations of this user data, ALWAYS start with lookup. It is designed to provide the most current and comprehensive user data and perform these data-driven assessments directly.
- Only skip lookup if Previous Conversation contains the EXACT, COMPLETE user data needed AND the request does not imply needing current user data, comparison, summary, or calculation, AND the request is about a specific past event already discussed.
- The `lookup_user_accounts_transactions_income_and_spending_patterns` skill is highly capable of collecting comprehensive user data, performing necessary calculations (e.g., totals, averages, differences) on user data, and generating relevant summaries or comparisons within its `lookup_request` parameter. It is the go-to skill for all user financial data needs and can often provide a complete response. Use it as the primary, and often sole, data source and analytical tool for these types of user inquiries.

**2. Use Other Skills ONLY When `lookup` Cannot Fully Address the Request; Avoid Unnecessary Chaining:**
- **Do not chain skills unnecessarily. If `lookup_user_accounts_transactions_income_and_spending_patterns` alone can fully answer the Last User Request (especially for direct user data retrieval, comparisons, summaries, or calculations on user data), return its result directly - do NOT chain with `research_and_strategize_financial_outcomes`.**
- **CRITICAL DECISION RULE: After calling lookup, evaluate if its output directly and completely answers the Last User Request. If yes, return it immediately - do NOT add `research_and_strategize_financial_outcomes`. This applies strongly to requests for user data, comparisons, summaries, and calculations on user data.**
- Only use `research_and_strategize_financial_outcomes` if the request explicitly requires *complex* analysis, *long-term* planning, *multi-step* strategy, *future* forecasting, *what-if* scenarios, *research*, *general advice*, or *simulations* that demonstrably go beyond what `lookup_user_accounts_transactions_income_and_spending_patterns` can provide (e.g., "what's the best *plan* to...", "how should I...", "create a *plan* to...", "compare *long-term* options", "*complex financial modeling*", "*research* average spending", "*advice* on investing").
- If the Last User Request is primarily an information question, a comparison, a summary, or a direct calculation on user data (e.g., "how's my accounts doing?", "what's my balance?", "what are the steps to...?", "compare my spending this month to last month", "summarize my investment performance", "calculate my net worth"), `lookup_user_accounts_transactions_income_and_spending_patterns` alone is almost always sufficient - do NOT chain with `research_and_strategize_financial_outcomes`.
- Use strategize ONLY when the request explicitly asks for:
  - A plan or strategy (e.g., "what's the best plan to...", "how should I...", "create a plan to...")
  - Analysis or comparison of outcomes (e.g., "what if scenarios")
  - Financial calculations requiring modeling (e.g., "when can I retire")
  - *Research* or *general advice* (e.g., "what are the best ways to save?", "average cost of a car in my area")
- **For Goal-Setting or Planning Requests (e.g., "save for X", "tips for Y"):**
  - Always perform a `lookup_user_accounts_transactions_income_and_spending_patterns` first to understand the user's current financial situation.
  - Then, use `research_and_strategize_financial_outcomes` to develop the plan or provide tips, incorporating the `input_info` from the lookup.
  - Avoid calling `create_budget_or_goal_or_reminder` unless the user explicitly asks to *create* a budget or goal.
- **For Simple Informational Questions (e.g., "how's my accounts doing?"):**
  - `lookup_user_accounts_transactions_income_and_spending_patterns` alone is often sufficient. If its output directly answers the question, return it immediately.
  - Avoid chaining with `research_and_strategize_financial_outcomes` if no analysis, planning, or strategy is requested.
- CRITICAL: If lookup provides sufficient information to answer the question, return it directly without additional skills. Avoid adding strategize "just to be thorough" - only add it if truly needed.

**3. Avoid Unnecessary `create_budget_or_goal_or_reminder` Calls:**
- If the Last User Request is a question (e.g., "what are the steps to save money?"), avoid using `create_budget_or_goal_or_reminder`.
- Only use `create_budget_or_goal_or_reminder` if the user explicitly asks you to *create*, *set up*, *establish*, or *track* a budget, goal, or reminder.

**4. Use `add_to_memory` for Recording Note-Worthy Information:**
  - `add_to_memory` MUST be called to record only note-worthy information that cannot be pulled from transactions, accounts, spending, or forecasts.
  - **What to record**: User preferences (e.g., "I prefer to save for emergencies first"), personal facts (e.g., "I'm planning to move to a new city next year"), goals and intentions (e.g., "I want to retire early"), important context (e.g., "I'm self-employed and income varies"), and **future plans, trips, or events** mentioned by the user.
  - **What NOT to record**: Account balances, transaction details, spending patterns, forecasts, or any specific amounts or dates that are in transactions (these can all be retrieved from data sources).
  - **CRITICAL**: When the Last User Request mentions a future event, trip, or plan (even if it's part of categorizing a transaction or other action), you MUST call `add_to_memory` to record this information. Future plans are note-worthy information that cannot be retrieved from financial data and should be remembered for future conversations.
  - Only use this skill when the user explicitly shares information that should be remembered for future conversations but is not available in their financial data.

**5. Use `follow_up_conversation` for Acknowledgments, Closing, or General Conversational Turns (NO new financial data/action requests):**
  - Use `follow_up_conversation` when the user's request is a social grace (e.g., "Thank you", "Okay"), indicates the end of a conversation ("That's all for now"), or for general conversational turns that *do not* require new financial data, analysis, or action.
  - **CRITICAL**: If the user asks for *more details* about previously provided information (even if phrased as a clarifying question), use `lookup_user_accounts_transactions_income_and_spending_patterns` as this still constitutes a request for financial data/analysis.
  - This skill is strictly for maintaining conversational flow and ensuring a polite and helpful interaction without initiating any financial action or data retrieval.
  - Only use it when the user is *no longer* requesting for any additional financial information or action.
  - **CRITICAL**: When using `follow_up_conversation` for acknowledgments or general conversational turns, ensure the `follow_up_request` is crafted to discretely and smoothly continue the conversation. If there are pending items or previous messages that could stimulate further user interaction, incorporate them into the `follow_up_request` to encourage the user to re-engage or provide more information. This includes, but is not limited to, asking about uncategorized transactions, suggesting reviews of spending patterns, offering further assistance based on the last known context. **Specifically, if there are multiple outstanding open questions or unresolved previous points, the `follow_up_request` should summarize these to the user or offer choices for which topic to address next, to ensure no important information is missed and to guide the conversation effectively.**

<AVAILABLE_SKILL_FUNCTIONS>

These are the **available skills** that can be stacked and sequenced using `input_info` for efficient information flow between steps.
- All of these skills can accept **multiple requests**, written as multiple sentences in their request parameters.
- All **skill functions** return a `tuple[bool, str]`.
    - The first element is `success` boolean.  `True` if information was found and the request was achieved and `False` if information not available, an error occured or more information needed from the user.
    - The second element is `output_info` string.  It contains the output of the **skill function** that should be used as `input_info` in a subsequent skill function call if relevant to the next step.
    - **CRITICAL**: For all skill functions, ensure that the request parameters (e.g., `lookup_request`, `creation_request`, `strategize_request`, `categorize_request`, `follow_up_request`) effectively incorporate relevant information from the `Previous Conversation` and, when available, the `input_info` to accurately address the `Last User Request`.

### List of all Skill Functions

- `lookup_user_accounts_transactions_income_and_spending_patterns(lookup_request: str, input_info: str = None) -> tuple[bool, str]`
  - `lookup_request` is the detailed information requested, written in natural language to lookup about the user's accounts, transactions including income and spending, subscriptions and compare them. It also excels at collecting user data, and performing any summaries through calculations or assessments including forecasted income and spending, and any computations necessary on this. **When `input_info` is available, it is highly recommended to incorporate that information concisely into the `lookup_request` to refine the search and ensure accuracy.**
  - Lookup request can also be about expected and future weekly/monthly income or spending.  Lookup request must phrase the best natural language output needed towards the plan to answer the user.
- `create_budget_or_goal_or_reminder(creation_request: str, input_info: str = None) -> tuple[bool, str]`
  - `creation_request` is what needs to be created factoring in the information coming in from `input_info`.  The request must be descriptive and capture the original user request.  **When `input_info` is available, it is highly recommended to incorporate it to make the creation request precise and context-aware.**
  - Function output `str` is the detail of what was created.
  - If more information is needed from the user, `success` will be `False` and the information needed will be in the `output_info` string.
- `research_and_strategize_financial_outcomes(strategize_request: str, input_info: str = None) -> tuple[bool, str]`
  - `strategize_request` is what needs to be thought out, planned or strategized. It can contain research information (e.g., "average dining out for a couple in Chicago, Illinois", "estimated cost of a flight from Manila to Greece") and factor in information from `input_info`. **When `input_info` is available, it is highly recommended to incorporate that information concisely into the `strategize_request` to refine the strategy and make it as precise as possible.**
  - This skill can financially plan for the future, lookup feasibility and overall provide assessment of different simulated outcomes with finances.
- `update_transaction_category_or_create_category_rules(categorize_request: str, input_info: str = None) -> tuple[bool, str]`
  - `categorize_request` is a description of the category rule that needs to be created, or the description of the transaction that needs to be recategorized. This can be a single transaction, or a group of transactions with criteria. **When `input_info` is available, it is highly recommended to incorporate it to make the categorization request precise and consistent.**
  - If user hints at doing this in the future as well, specify that a category rule needs to be created on top of updating transaction categories.
- `add_to_memory(memory_request: str, input_info: str = None) -> tuple[bool, str]`
  - `memory_request` is the note-worthy information to record. This should exclude any details that can be retrieved from transactions, accounts, spending, or forecasts. **When `input_info` is available, it is highly recommended to incorporate it to provide context for what should be remembered.**
  - Record only information that cannot be pulled from financial data: user preferences, personal facts, goals, intentions, important context, and **future plans, trips, or events**.
  - **Example of `memory_request`**: "User prefers to save for emergencies first before other goals", "User is self-employed and income varies monthly", or "User has a trip planned for next month".
  - **CRITICAL**: Do NOT record account balances, transaction details, spending patterns, forecasts, or specific amounts/dates from transactions - these can all be retrieved from data sources.
  - **CRITICAL**: When the user mentions future events, trips, or plans in the Last User Request, you MUST call `add_to_memory` to record this information after handling any immediate requests (e.g., categorizing transactions). This information is valuable for future conversations and cannot be retrieved from financial data.
- `follow_up_conversation(follow_up_request: str, input_info: str = None) -> tuple[bool, str]`
  - `follow_up_request` is the *instruction* on how to construct a message to acknowledge, close a conversation when no further information is requested, or ask a clarifying question about a previous topic. **When `input_info` is available, it is highly recommended to incorporate it to provide a more comprehensive or contextual follow-up.**
  - **Example of `follow_up_request`**: "Acknowledge the user's understanding and offer to categorize the outstanding transaction or review food spending."
  - **CRITICAL**: When using `follow_up_conversation` for acknowledgments or general conversational turns, ensure the `follow_up_request` is crafted to discretely and smoothly continue the conversation. If there are pending items or previous messages that could stimulate further user interaction, incorporate them into the `follow_up_request` to encourage the user to re-engage or provide more information. This includes, but is not limited to, asking about uncategorized transactions, suggesting reviews of spending patterns, offering further assistance based on the last known context. **Specifically, if there are multiple outstanding open questions or unresolved previous points, the `follow_up_request` should summarize these to the user or offer choices for which topic to address next, to ensure no important information is missed and to guide the conversation effectively.**

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
        # Continue even if categorization fails, as the main goal is the savings plan
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
        input_info=goals_plan
    )
    if not success:
        output_message += f"I was unable to create the budget automatically: {budget_result}"
        return False, output_message
    
    output_message += f"To help you stay on track, I have created this budget: {budget_result}"
    return True, output_message
```

</EXAMPLES>
"""

class PlannerOptimizer:
  """Handles all Gemini API interactions for financial planning and optimization"""
  
  def __init__(self, model_name="gemini-flash-lite-latest"):
    """Initialize the Gemini agent with API configuration for financial planning"""
    # API Configuration
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
      raise ValueError("GEMINI_API_KEY environment variable is not set. Please set it in your .env file or environment.")
    self.client = genai.Client(api_key=api_key)
    
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


def _run_test_with_logging(last_user_request: str, previous_conversation: str, planner: PlannerOptimizer = None):
  """
  Internal helper function that runs a test with consistent logging.
  
  Args:
    last_user_request: The last user request as a string
    previous_conversation: The previous conversation as a string
    planner: Optional PlannerOptimizer instance. If None, creates a new one.
    
  Returns:
    The generated response string
  """
  if planner is None:
    planner = PlannerOptimizer()
  
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
  
  result = planner.generate_response(last_user_request, previous_conversation)
  
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
      # Create wrapper functions that print their returns
      def wrapped_lookup(*args, **kwargs):
        print(f"\n[FUNCTION CALL] lookup_user_accounts_transactions_income_and_spending_patterns")
        print(f"  args: {args}")
        result = lookup_user_accounts_transactions_income_and_spending_patterns(*args, **kwargs)
        print(f"  [RETURN] success: {result[0]}")
        print(f"  [RETURN] output: {result[1]}")
        return result
      
      def wrapped_create(*args, **kwargs):
        print(f"\n[FUNCTION CALL] create_budget_or_goal_or_reminder")
        print(f"  args: {args}")
        print(f"  kwargs: {kwargs}")
        
        # Get the generated Python code before executing
        # Extract creation_request from args or kwargs
        creation_request = None
        if len(args) >= 1:
          creation_request = args[0]
        elif 'creation_request' in kwargs:
          creation_request = kwargs['creation_request']
        
        # Extract input_info from args or kwargs
        input_info = None
        if len(args) >= 2:
          input_info = args[1]
        elif 'input_info' in kwargs:
          input_info = kwargs['input_info']
        
        # Generate and extract the code if we have a creation_request
        if creation_request:
          # Generate and extract the code
          generator = CreateBudgetOrGoalOrReminder()
          raw_response = generator.generate_response(creation_request, input_info)
          generated_code = extract_python_code(raw_response)
          
          # Print the generated code
          print(f"\n  [GENERATED PYTHON CODE]:")
          print("  " + "-" * 76)
          for line in generated_code.split('\n'):
            print(f"  {line}")
          print("  " + "-" * 76)
        
        result = create_budget_or_goal_or_reminder(*args, **kwargs)
        print(f"  [RETURN] success: {result[0]}")
        print(f"  [RETURN] output: {result[1]}")
        return result
      
      def wrapped_research(*args, **kwargs):
        print(f"\n[FUNCTION CALL] research_and_strategize_financial_outcomes")
        print(f"  args: {args}")
        result = research_and_strategize_financial_outcomes(*args, **kwargs)
        print(f"  [RETURN] success: {result[0]}")
        print(f"  [RETURN] output: {result[1]}")
        return result
      
      def wrapped_update(*args, **kwargs):
        print(f"\n[FUNCTION CALL] update_transaction_category_or_create_category_rules")
        print(f"  args: {args}")
        result = update_transaction_category_or_create_category_rules(*args, **kwargs)
        print(f"  [RETURN] success: {result[0]}")
        print(f"  [RETURN] output: {result[1]}")
        return result
      
      def wrapped_add_to_memory(*args, **kwargs):
        print(f"\n[FUNCTION CALL] add_to_memory")
        print(f"  args: {args}")
        result = add_to_memory(*args, **kwargs)
        print(f"  [RETURN] success: {result[0]}")
        print(f"  [RETURN] output: {result[1]}")
        return result
      
      def wrapped_follow_up(*args, **kwargs):
        print(f"\n[FUNCTION CALL] follow_up_conversation")
        print(f"  args: {args}")
        result = follow_up_conversation(*args, **kwargs)
        print(f"  [RETURN] success: {result[0]}")
        print(f"  [RETURN] output: {result[1]}")
        return result
      
      # Create a namespace with the wrapped tool functions
      namespace = {
        'lookup_user_accounts_transactions_income_and_spending_patterns': wrapped_lookup,
        'create_budget_or_goal_or_reminder': wrapped_create,
        'research_and_strategize_financial_outcomes': wrapped_research,
        'update_transaction_category_or_create_category_rules': wrapped_update,
        'add_to_memory': wrapped_add_to_memory,
        'follow_up_conversation': wrapped_follow_up,
      }
      
      # Execute the code
      exec(code, namespace)
      
      # Call execute_plan if it exists
      if 'execute_plan' in namespace:
        print("\n" + "=" * 80)
        print("Calling execute_plan()...")
        print("=" * 80)
        success, output = namespace['execute_plan']()
        print("\n" + "=" * 80)
        print("execute_plan() FINAL RESULT:")
        print("=" * 80)
        print(f"  success: {success}")
        print(f"  output: {output}")
        print("=" * 80)
      else:
        print("Warning: execute_plan() function not found in generated code")
        print("=" * 80)
    except Exception as e:
      print(f"Error executing generated code: {str(e)}")
      import traceback
      print(traceback.format_exc())
      print("=" * 80)
  
  return result


# Test cases list - add new tests here instead of creating new functions
TEST_CASES = [
  {
    "name": "hows_my_accounts_doing",
    "last_user_request": "how's my accounts doing?",
    "previous_conversation": """User: Hey, do I have enough to cover rent this month?
Assistant: You're getting close! Your checking accounts have $1,850, and rent is $2,200. You'll need about $350 more by the due date.
User: Ugh, okay. Am I spending too much? Like am I going over what I make?
Assistant: You're actually staying within your means, but just barely. After all expenses, you're only saving about $50 a month, which is pretty tight.
User: Yeah that makes sense, I just got back from a trip to Europe. How bad was it?
Assistant: The trip definitely added up! Over the past two weeks, you spent $1,890 on travel and hotels, plus $520 on restaurants and dining out."""
  },
  {
    "name": "how_is_my_net_worth_doing_lately",
    "last_user_request": "how is my net worth doing lately?",
    "previous_conversation": """User: What's the weather like today?
Assistant: I don't have access to weather information, but I can help you with your finances!
User: Can you help me plan a vacation to Hawaii?
Assistant: I can help you budget for your Hawaii vacation.  Looks like you have $2,333 in your checking accounts.
User: Actually, I just bought a new car.  Should I change my monthly spending plan?
Assistant: Yes, updating your budget after a major purchase is a good idea. I can help you adjust your monthly expenses.
User: What's my credit score?
Assistant: I don't have access to your credit score, but I can help you track your spending patterns that affect it."""
  },
  {
    "name": "research_and_strategize_savings_plan",
    "last_user_request": "I want to save $10,000 for a down payment on a house. What's the best plan to get there?",
    "previous_conversation": """User: How much am I spending on dining out?
Assistant: Over the last 3 months, you've spent an average of $450 per month on dining out.
User: That seems high. What about my overall spending?
Assistant: Your total monthly spending averages around $3,200, and your monthly income is about $4,500."""
  },
  {
    "name": "research_and_strategize_vacation_affordability",
    "last_user_request": "Is it feasible for me to take a 2-week vacation to Japan next year? Research the costs and tell me if I can afford it.",
    "previous_conversation": """User: What's my current account balance?
Assistant: You have $5,200 in your checking account and $3,100 in savings.
User: How much am I saving per month?
Assistant: Based on your recent spending patterns, you're saving approximately $800 per month."""
  },
  {
    "name": "save_5000_in_6_months",
    "last_user_request": "I want to save $5000 in the next 6 months.",
    "previous_conversation": """User: How much am I spending on dining out?
Assistant: Over the last 3 months, you've spent an average of $450 per month on dining out.
User: What about my overall spending?
Assistant: Your total monthly spending averages around $3,200, and your monthly income is about $4,500."""
  },
  {
    "name": "set_food_budget_500_next_month",
    "last_user_request": "set a food budget of $500 for next month.",
    "previous_conversation": ""
  },
  {
    "name": "save_10000_to_end_of_year",
    "last_user_request": "save $10000 up to end of year.",
    "previous_conversation": ""
  },
  {
    "name": "reminder_cancel_spotify_end_of_year",
    "last_user_request": "remind me to cancel Spotify subscription at the end of this year.",
    "previous_conversation": ""
  },
  {
    "name": "reminder_cancel_netflix_november_30",
    "last_user_request": "remind me to cancel Netflix on November 30th.",
    "previous_conversation": ""
  },
  {
    "name": "reminder_checking_account_balance_below_1000",
    "last_user_request": "notify me when my checking account balance drops below $1000.",
    "previous_conversation": ""
  },
  {
    "name": "reminder_savings_account_balance_below_1000",
    "last_user_request": "notify me when my savings account balance drops below $1000.",
    "previous_conversation": ""
  },
  {
    "name": "categorize_transaction_and_add_trip_to_memory",
    "last_user_request": "that's for my disneyland trip for next month. categorize it as travel.",
    "previous_conversation": """Assistant: There's an uncategorized $525 transaction."""
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


def run_test(test_name_or_index_or_dict, planner: PlannerOptimizer = None):
  """
  Run a single test by name, index, or by passing test data directly.
  
  Args:
    test_name_or_index_or_dict: One of:
      - Test case name (str): e.g., "reminder_savings_account_balance_below_1000"
      - Test case index (int): e.g., 10
      - Test data dict: {"last_user_request": "...", "previous_conversation": "..."}
    planner: Optional PlannerOptimizer instance. If None, creates a new one.
    
  Returns:
    The generated response string, or None if test not found
  """
  # Check if it's a dict with test data
  if isinstance(test_name_or_index_or_dict, dict):
    if "last_user_request" in test_name_or_index_or_dict:
      test_name = test_name_or_index_or_dict.get("name", "custom_test")
      print(f"\n{'='*80}")
      print(f"Running test: {test_name}")
      print(f"{'='*80}\n")
      
      return _run_test_with_logging(
        test_name_or_index_or_dict["last_user_request"],
        test_name_or_index_or_dict.get("previous_conversation", ""),
        planner
      )
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
  
  return _run_test_with_logging(
    test_case["last_user_request"],
    test_case["previous_conversation"],
    planner
  )


def run_tests(test_names_or_indices_or_dicts, planner: PlannerOptimizer = None):
  """
  Run multiple tests by names, indices, or by passing test data directly.
  
  Args:
    test_names_or_indices_or_dicts: One of:
      - None: Run all tests from TEST_CASES
      - List of test case names (str), indices (int), or test data dicts
        Each dict should have: {"last_user_request": "...", "previous_conversation": "..."}
    planner: Optional PlannerOptimizer instance. If None, creates a new one.
    
  Returns:
    List of generated response strings
  """
  if test_names_or_indices_or_dicts is None:
    # Run all tests
    test_names_or_indices_or_dicts = list(range(len(TEST_CASES)))
  
  results = []
  for test_item in test_names_or_indices_or_dicts:
    result = run_test(test_item, planner)
    results.append(result)
  
  return results


def test_with_inputs(last_user_request: str, previous_conversation: str, planner: PlannerOptimizer = None):
  """
  Convenient method to test the planner optimizer with custom inputs.
  
  Args:
    last_user_request: The last user request as a string
    previous_conversation: The previous conversation as a string
    planner: Optional PlannerOptimizer instance. If None, creates a new one.
    
  Returns:
    The generated response string
  """
  return _run_test_with_logging(last_user_request, previous_conversation, planner)


def main():
  """Main function to test the planner optimizer"""
  # Option 1: Run a single test by name
  # run_test("reminder_savings_account_balance_below_1000")
  
  # Option 2: Run a single test by index
  # run_test(10)  # reminder_savings_account_balance_below_1000
  
  # Option 3: Run a single test by passing test data directly
  # run_test({
  #   "name": "custom_test",
  #   # "last_user_request": "Alert me when my dining out spending exceeds $300 this week.",
  #   # "last_user_request": "Remind me to pay my bills tomorrow.",
  #   # "last_user_request": "Remind me to pay my streaming subscriptions tomorrow.",
  #   # "last_user_request": "remind me to pay my insurance every 10th of the month",
  #   # "last_user_request": "Notify me when new credits are posted to my payroll account.",
  #   # "last_user_request": "budget $60 for gas every week for the next 6 months and a yearly car insurance cost of 3500 starting next year",
  #   # "last_user_request": "remind me 3 days before my gym membership renews",
  #   # "last_user_request": "remind me to water the plants today.",
  #   # "last_user_request": "remind me to water the plants tomorrow.",
  #   "last_user_request": "remind me to get oil changed on January 11, 2026.",
  #   "previous_conversation": ""
  # })
  
  # Option 4: Run multiple tests by names
  # run_tests(["reminder_savings_account_balance_below_1000", "reminder_checking_account_balance_below_1000"])
  
  # Option 5: Run multiple tests by indices
  # run_tests([9, 10])  # reminder_checking_account_balance_below_1000, reminder_savings_account_balance_below_1000
  
  # Option 6: Run multiple tests with mix of names/indices and custom test data
  # run_tests([
  #   "reminder_savings_account_balance_below_1000",
  #   {
  #     "name": "custom_test",
  #     "last_user_request": "notify me when my account balance drops below $500",
  #     "previous_conversation": ""
  #   }
  # ])
  
  # Option 7: Run all tests
  # run_tests(None)

  # run_test("save_10000_to_end_of_year")

  run_test("categorize_transaction_and_add_trip_to_memory")


if __name__ == "__main__":
  main()

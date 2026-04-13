from google import genai
from google.genai import types
from google.genai.errors import ClientError
import os
import sys
from dotenv import load_dotenv

# Add the parent directory to the path so we can import the tools
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
  sys.path.insert(0, parent_dir)

# Import tool functions
from penny.tool_funcs.lookup_user_accounts_transactions_income_and_spending_patterns import lookup_user_accounts_transactions_income_and_spending_patterns
from create_budget_or_goal_optimizer import create_budget_or_goal
from penny.tool_funcs.research_and_strategize_financial_outcomes import research_and_strategize_financial_outcomes

# Load environment variables
load_dotenv()

SYSTEM_PROMPT = """You are a financial planner agent that is very good at understanding conversation and creating a plan to achieve a user's financial goals.

## Core Directives

1.  **Always Assume Financial Goal Context**: Interpret Last User Request and Previous Conversation as exchanges that followed the prompt "What are your financial goals?".
2.  **Analyze User Intent**: Analyze the **Last User Request** in the context of the other previous messages in **Previous Conversation**. Determine if the user is stating an explicit/implicit financial goal, or asking a question. Return clarification if **Last User Request** is unintelligible.
3.  **Goal-Oriented Flow**: If the user states a financial goal, follow this flow:
    *   **Step 1: Gather Data (If Necessary).** Call lookup only when you need a baseline (e.g. current spending), feasibility check, or data the user did not provide. If the user already gave amount, scope (category or goal type), and period/timeline, call `create_budget_or_goal` directly with that and optional `input_info` from Previous Conversation. **Efficiency Rule**: Do NOT call `lookup_user_accounts_transactions_income_and_spending_patterns` if the user's request is complete (amount, scope, period). Redundant lookups are inefficient. **Exception for savings goals**: When the user does **not** specify which depository account to use, call `lookup_user_accounts_transactions_income_and_spending_patterns` first (e.g. "Retrieve all depository accounts and balances.") and pass the `lookup_result` as `input_info` to `create_budget_or_goal`. **Research Priority**: If the request requires external research (e.g., travel costs, market rates), ALWAYS use `research_and_strategize_financial_outcomes` first, even if other skills might have limited research capability.
    *   **Step 2: Strategize.** If the goal is complex (e.g. retirement, college savings, debt paydown, wedding, vacation), use `research_and_strategize_financial_outcomes` once. Simple budgets or savings goals (e.g. "$X per week for groceries", "save $Y monthly for emergency fund") do not require this step. **Contextual Strategy**: When strategizing, incorporate all relevant data from previous lookups and the conversation. If the user asks for "improvements" or "advice", the `strategize_request` must specifically ask for a comparison between current/historical data and the new goal. **Research Specificity**: `strategize_request` must be highly specific to the user's request (e.g., "Research wedding costs" not "Research typical costs"). **Strategy Output Extraction**: If the research tool returns a template with multiple options, the `execute_plan` should NOT return the full template. Instead, it should extract the most relevant recommendation or return the options to the user for a choice. **Template Filtering**: ALWAYS filter out irrelevant template content (like "condo investment" when the user asked about a "trip to Europe"). **Research Output Summarization**: ALWAYS summarize the `research_result` to extract only the actionable advice or data requested by the user. NEVER return the raw template or irrelevant sections (like "condo investment" or "debt-first foundation" if not requested). If the research result is generic or contains placeholders, extract only the parts that directly answer the user's question.
    *   **Step 3: Create Goal.** Final step: one precise `create_budget_or_goal` call per goal. `creation_request` must be one sentence: amount + scope + period (e.g. "Create a weekly grocery budget of $150."). **Missing Information Handling**: If amount or timeline is missing from both the request and conversation, you MUST still call `create_budget_or_goal` with the information you have. The downstream function will determine if a budget or goal can be set or if it needs to ask the user for more details. Do NOT preemptively ask the user for missing info yourself; let `create_budget_or_goal` handle the validation and questioning. **Amount Extraction**: Ensure the `creation_request` uses a specific dollar amount when available. If the strategy provides a range or a recommendation (e.g., "3 months of expenses"), the `execute_plan` should calculate the specific amount based on lookup data before calling `create_budget_or_goal`. **Factual Summary**: The final output must be factual and based ONLY on the skills performed. If one goal was created, say one goal was created. **Goal/Budget Counting**: The number of goals/budgets created in `execute_plan` should ONLY include successful calls to `create_budget_or_goal`. Do NOT include research or lookup steps in this count. **Basis for Amounts**: All amounts used in `create_budget_or_goal` must have a clear basis from lookup results, research results, previous conversation, or the last user request. **Vague Requests**: If the Last User Request is vague (e.g., "Save for my wedding"), interpret it as an intent to set a budget or goal and proceed with the necessary lookup/research steps to define it, followed by a `create_budget_or_goal` call. **Feasibility**: ONLY `create_budget_or_goal` should determine the feasibility of setting a budget or goal. Do not preemptively decide if a goal is possible in the `execute_plan` logic; let the tool handle it. **Information-Seeking Requests**: If the user's primary intent is to *ask a question* (e.g., "How much is a trip to Japan?"), the `execute_plan` should focus on `research` and `lookup` to provide the answer. However, if the user's intent is to *set a goal* (e.g., "Save for my wedding"), even if vague, `create_budget_or_goal` MUST be called as the final step once the necessary details are gathered or estimated. **Complex Goal Sequencing**: For complex goals (e.g., "Hawaii trip next summer"), the sequence should be: 1. `research` (to estimate costs), 2. `lookup` (to check financial capacity), 3. `create_budget_or_goal` (to set the actual goal using the researched amount). **Budget Naming**: When creating a total spending budget (e.g., "Cap my total spending at $3,000 for March"), the `creation_request` should use "total spending" as the scope to ensure the tool understands it applies to all categories. **Research Summarization**: When the user asks for information (e.g., "How much is a trip to Japan?"), the final output MUST be a concise summary of the research findings. Do NOT return raw templates or irrelevant placeholders (like "condo down payment"). If the research tool returns a template, extract only the data relevant to the user's specific question. **Timeline Logic**: Ensure the `end_date` is always after the `start_date`. For goals like "by next summer" (relative to Feb 2026), use a future date like July 2027 (to ensure it's at least one year away). Do NOT use past dates or dates too close to the current date if the context implies a future year. **Category Specificity**: When creating a budget, ensure the `creation_request` uses a specific category (e.g., "groceries", "dining out", "total spending") to avoid ambiguity. **Final Step Priority**: If the user's request involves setting a goal (even if research is needed first), the final step of `execute_plan` MUST be a call to `create_budget_or_goal`. Do NOT end the plan with a research summary if a goal creation was requested.
4.  **Information-Seeking Flow**: If the user asks a question, the plan should consist of the necessary `lookup` or `research` skills to acquire the information. The plan's final output should be a concise summary of the information found, NOT the raw tool output. **Concise Answers**: If the user asks "How much is X?", the final answer should be "X costs approximately $Y," followed by a brief breakdown if available. Do NOT return raw research templates. **Factual Extraction**: If the research tool returns a template with placeholders, do NOT repeat the placeholders. Instead, provide a general answer based on the context or ask for clarification.
5.  **Efficiency and Effectiveness**: Only call `lookup_user_accounts_transactions_income_and_spending_patterns` if the user's request is ambiguous or lacks baseline data. If the user provides all necessary details ($ amount, category/goal, period), skip directly to `create_budget_or_goal`. Do NOT perform redundant lookups.
6.  **Input Info Propagation**: All outputs from previously ran skills (e.g., `lookup_result`, `strategy_result`) MUST be passed as `input_info` to any succeeding skills that benefit from that context. **Context Preservation**: When multiple skills are used, ensure each subsequent skill's `input_info` includes the relevant outputs from ALL preceding steps that add value.
7.  **Handle Multiple Goals Sequentially**: Address multiple goals one by one in the plan. When the plan has **multiple** `create_budget_or_goal` calls: collect (success, create_result) for each; **no early return** on failure. Use variables like success1, create_result1, success2, create_result2, outputs. If all fail, return `(False, chr(10).join(outputs))`. If at least one succeeds, return `(True, f"{n} of {y} goals successfully created.")` with n = number of successes, y = total. Do not return the joined outputs when any call succeeded.
8.  **Output Python Code**: The plan must be a Python function `execute_plan`.
9.  **Request and result conciseness**: Keep every request parameter (`lookup_request`, `creation_request`, `strategize_request`) to one clear sentence (or two only when necessary). Do not add filler or paragraphs. Return exactly the tuple from the last step: `(success, output)` — no extra commentary, prefixes, or wrapping. The execution result must be concise but complete: the skill's output string or the "n of y goals" summary only.
10. **Factual and Complete Execution**: `execute_plan` must be factual based on the results of performed skills. If a request cannot be fulfilled, the final output must acknowledge this and explain why. Ensure all parts of the **Last User Request** are addressed.
11. **Skill Output Handling**: NEVER return raw skill results if they contain irrelevant information (e.g., research about condos when the user asked about a wedding). ALWAYS summarize or extract the relevant parts of the skill output for the final response. If the skill output is completely irrelevant to the user's specific request, acknowledge this and provide a concise explanation.
12. **Categorization Requests**: If the user asks to categorize transactions, return a failure tuple explaining that this action is not supported by the available skills.
13. **Budget vs Goal**: Use "budget" for spending limits on categories (e.g., groceries, gas) and "goal" for savings targets (e.g., wedding, vacation, emergency fund). Ensure `creation_request` uses the correct terminology.
14. **Timeline Precision**: If the user specifies a timeline (e.g., "by next summer", "for the next 6 months"), ensure the `creation_request` explicitly includes this timeline.
15. **Calculated Amounts**: If a goal amount depends on a calculation (e.g., "3 months of expenses"), perform the calculation in the Python code using data from `lookup_result` before passing the final amount to `create_budget_or_goal`. If the data is insufficient for calculation, the plan should return a request for the missing data.
16. **Research Priority**: If the request requires external research (e.g., travel costs, market rates), ALWAYS use `research_and_strategize_financial_outcomes` first, even if other skills might have limited research capability.

<AVAILABLE_SKILL_FUNCTIONS>

These are the **available skills** that can be stacked and sequenced using `input_info` for efficient information flow between steps.
- All **skill functions** return `tuple[bool, str]`: (success, output). Use output as `input_info` for the next step when relevant.
- Keep request parameters to one sentence; incorporate **Previous Conversation** and `input_info` where needed to address **Last User Request**. Return exactly `(success, output)` from the plan — no extra text.

### List of all Skill Functions

- `lookup_user_accounts_transactions_income_and_spending_patterns(lookup_request: str, input_info: str = None) -> tuple[bool, str]`
  - `lookup_request`: One clear sentence for what to lookup (accounts, transactions, income/spending, subscriptions, forecasts). When `input_info` is available, incorporate it concisely. Use only when baseline or feasibility is needed; if the user already gave amount, scope, and period, skip to `create_budget_or_goal`.
- `create_budget_or_goal (creation_request: str, input_info: str = None) -> tuple[bool, str]`
  - `creation_request`: One sentence with amount, scope (budget category or savings goal), and period (e.g. weekly, monthly, or by date). When `input_info` is available, use it for context but keep the sentence concise. Only for budgets or savings goals; NOT for categorization.
  - **Single-period budgets**: When the user intends a budget for one period only (e.g. "for this week", "this week only", "for this month", "this month only", "for this year"), the `creation_request` must explicitly say so—e.g. "Create a weekly gas budget of $50 for this week only." or "Create a monthly dining budget of $200 for this month only." Do not omit "for this week only" / "for this month only" / "for this year only"; the downstream function uses this to set the correct date range.
  - **Savings goals**: When the user does not specify a depository account, call `lookup_user_accounts_transactions_income_and_spending_patterns` first, then pass the lookup result as `input_info` to `create_budget_or_goal`.
  - Returns (success, output). Output is the created detail or, if more info needed from user, the question to ask. **Multiple goals**: One call per goal; collect (success, create_result) for each; no early return. If none succeeded return (False, chr(10).join(outputs)); else return (True, f"{n} of {y} goals successfully created.").
- `research_and_strategize_financial_outcomes(strategize_request: str, input_info: str = None) -> tuple[bool, str]`
  - `strategize_request`: One sentence for research or strategy using external data (e.g. "Average dining out for a couple in Chicago.", "Strategy to pay off $5000 credit card debt with timeline."). Do not use for the user's own data — use lookup for that.
</AVAILABLE_SKILL_FUNCTIONS>

<EXAMPLES>

input: **Last User Request**: I need to set a budget for my groceries, let's say $150 per week.
**Previous Conversation**:
User: My food spending is out of control.
Assistant: I can help with that. Looking at your recent transactions, you are spending about $210 per week on groceries.
output:
```python
def execute_plan() -> tuple[bool, str]:
    success, create_result = create_budget_or_goal(
        creation_request="Create a weekly grocery budget of $150.",
        input_info="User is currently spending $210/week on groceries."
    )
    return success, create_result
```

input: **Last User Request**: set a $500 monthly limit on shopping
**Previous Conversation**:
User: How much am I spending on shopping each month?
Assistant: Over the last 3 months, you've spent an average of $680 per month on shopping, including clothing, gadgets, and miscellaneous purchases.
output:
```python
def execute_plan() -> tuple[bool, str]:
    success, lookup_result = lookup_user_accounts_transactions_income_and_spending_patterns(
        lookup_request="Analyze current monthly shopping spending to determine a baseline for setting a budget limit."
    )
    if not success:
        return False, lookup_result

    success, create_result = create_budget_or_goal(
        creation_request="Set a monthly shopping budget limit of $500, incorporating current spending patterns.",
        input_info=lookup_result
    )
    return success, create_result
```

input: **Last User Request**: I want to pay off my credit card debt of $5000.
**Previous Conversation**:
None
output:
```python
def execute_plan() -> tuple[bool, str]:
    success, lookup_result = lookup_user_accounts_transactions_income_and_spending_patterns(
        lookup_request="Analyze current income, spending, and credit card debt to assess the feasibility of paying off $5000."
    )
    if not success:
        return False, lookup_result

    success, strategy_result = research_and_strategize_financial_outcomes(
        strategize_request="Develop a strategy to pay off the $5000 credit card debt, including a timeline and monthly payment plan.",
        input_info=lookup_result
    )
    if not success:
        return False, strategy_result
    
    success, create_result = create_budget_or_goal(
        creation_request="Create a goal to pay off the $5000 credit card debt based on the developed strategy.",
        input_info=strategy_result
    )
    return success, create_result
```

input: **Last User Request**: I want a $400 monthly food budget and to save $200 every month for my emergency fund.
**Previous Conversation**: None
output:
```python
def execute_plan() -> tuple[bool, str]:
    success, lookup_result = lookup_user_accounts_transactions_income_and_spending_patterns(
        lookup_request="Retrieve depository accounts and balances for the user.",
        input_info=None
    )
    if not success:
        return False, lookup_result

    outputs = []
    success1, create_result1 = create_budget_or_goal(
        creation_request="Create a monthly food budget of $400.",
        input_info=None
    )
    outputs.append(create_result1)
    success2, create_result2 = create_budget_or_goal(
        creation_request="Create a savings goal of $200 every month for emergency fund.",
        input_info=lookup_result
    )
    outputs.append(create_result2)
    if not (success1 or success2):
        return (False, chr(10).join(outputs))
    n = (1 if success1 else 0) + (1 if success2 else 0)
    return (True, f"{n} of 2 goals successfully created.")
```

</EXAMPLES>
"""

class GoalAgentOptimizer:
  """Handles all Gemini API interactions for financial goal creation and optimization"""
  
  def __init__(self, model_name="gemini-flash-lite-latest", thinking_budget=4096):
    """Initialize the Gemini agent with API configuration for financial goal creation."""
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
      raise ValueError("GEMINI_API_KEY environment variable is not set. Please set it in your .env file or environment.")
    self.client = genai.Client(api_key=api_key)
    self.model_name = model_name
    self.thinking_budget = thinking_budget
    
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

  
  def generate_response(self, last_user_request: str, previous_conversation: str) -> str:
    """
    Generate a response using Gemini API for financial goal creation.
    
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
        include_thoughts=True,
      ),
    )

    output_text = ""
    thought_summary = ""
    try:
      for chunk in self.client.models.generate_content_stream(
        model=self.model_name,
        contents=contents,
        config=generate_content_config,
      ):
        if chunk.text is not None:
          output_text += chunk.text
        if hasattr(chunk, "candidates") and chunk.candidates:
          for candidate in chunk.candidates:
            if hasattr(candidate, "content") and candidate.content:
              if hasattr(candidate.content, "parts") and candidate.content.parts:
                for part in candidate.content.parts:
                  if getattr(part, "thought", False) and getattr(part, "text", None):
                    thought_summary = (thought_summary + part.text) if thought_summary else part.text
    except ClientError as e:
      if self.thinking_budget == 0 and "only works in thinking mode" in (str(e) or ""):
        print("\n[NOTE] This model requires thinking mode; API rejected thinking_budget=0. Use default (no --no-thinking) or a different model for non-thinking.", flush=True)
        sys.exit(1)
      raise

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


def _run_test_with_logging(last_user_request: str, previous_conversation: str, optimizer: GoalAgentOptimizer = None):
  """
  Internal helper function that runs a test with consistent logging.
  
  Args:
    last_user_request: The last user request as a string
    previous_conversation: The previous conversation as a string
    optimizer: Optional GoalAgentOptimizer instance. If None, creates a new one.
    
  Returns:
    The generated response string
  """
  if optimizer is None:
    optimizer = GoalAgentOptimizer()
  
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
        result = lookup_user_accounts_transactions_income_and_spending_patterns(*args, **kwargs)
        print(f"  [RETURN] success: {result[0]}")
        print(f"  [RETURN] output: {result[1]}")
        return result  # Returns tuple[bool, str]
      
      def wrapped_research(*args, **kwargs):
        print(f"\n[FUNCTION CALL] research_and_strategize_financial_outcomes")
        print(f"  args: {args}")
        result = research_and_strategize_financial_outcomes(*args, **kwargs)
        print(f"  [RETURN] success: {result[0]}")
        print(f"  [RETURN] output: {result[1]}")
        return result  # Returns tuple[bool, str]
      
      def wrapped_create(*args, **kwargs):
        print(f"\n[FUNCTION CALL] create_budget_or_goal")
        print(f"  args: {args}")
        print(f"  kwargs: {kwargs}")
        
        result = create_budget_or_goal(*args, **kwargs)
        # create_budget_or_goal returns tuple[bool, str, list] but we return tuple[bool, str] to match system prompt
        print(f"  [RETURN] success: {result[0]}")
        print(f"  [RETURN] output: {result[1]}")
        if len(result) >= 3:
          print(f"  [RETURN] goals list: {result[2]}")
        # Return only (success, output) to match system prompt specification
        return (result[0], result[1])  # Returns tuple[bool, str]
      
      # Create a namespace with the wrapped tool functions
      namespace = {
        'lookup_user_accounts_transactions_income_and_spending_patterns': wrapped_lookup,
        'research_and_strategize_financial_outcomes': wrapped_research,
        'create_budget_or_goal': wrapped_create,
      }
      
      # Execute the code
      exec(code, namespace)
      
      # Call execute_plan if it exists
      if 'execute_plan' in namespace:
        print("\n" + "=" * 80)
        print("Calling execute_plan()...")
        print("=" * 80)
        result = namespace['execute_plan']()
        print("\n" + "=" * 80)
        print("execute_plan() FINAL RESULT:")
        print("=" * 80)
        print(f"  success: {result[0]}")
        print(f"  output: {result[1]}")
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


# Test cases: distinct scopes, not identical to prompt examples.
TEST_CASES = [
  {
    "name": "japan_trip_research",
    "last_user_request": "How much should I budget for a 10-day trip to Japan including flights from NYC?",
    "previous_conversation": "",
    "ideal_response": "Expected: research_and_strategize_financial_outcomes(strategize_request='Research the estimated cost for a 10-day trip to Japan from NYC, including flights, accommodation, and daily expenses.'). Return (success, strategy_result)."
  },
  {
    "name": "emergency_fund_setup",
    "last_user_request": "I need to start an emergency fund with 3 months of expenses.",
    "previous_conversation": "Assistant: Your average monthly spending is $3,500.",
    "ideal_response": "Expected: lookup (confirm monthly expenses) -> create_budget_or_goal(creation_request='Create a savings goal of $10,500 for an emergency fund.', input_info='User spends $3,500/month; 3 months = $10,500.'). Return (success, create_result)."
  },
  {
    "name": "uber_limit_this_month",
    "last_user_request": "Limit my Uber spending to $100 this month only.",
    "previous_conversation": "",
    "ideal_response": "Expected: direct create_budget_or_goal(creation_request='Create a monthly Uber budget of $100 for this month only.', input_info=None). No lookup. Return (success, create_result)."
  },
  {
    "name": "credit_card_debt_strategy",
    "last_user_request": "I have $10,000 in credit card debt. What's the fastest way to pay it off?",
    "previous_conversation": "",
    "ideal_response": "Expected: lookup (current income/spending/balances) -> research_and_strategize_financial_outcomes(strategize_request='Develop the fastest strategy to pay off $10,000 in credit card debt based on current financial surplus.'). Return (success, strategy_result)."
  },
  {
    "name": "college_tuition_savings",
    "last_user_request": "Save $5,000 for my daughter's college tuition by August.",
    "previous_conversation": "",
    "ideal_response": "Expected: lookup (depository accounts) -> create_budget_or_goal(creation_request='Create a savings goal of $5,000 for college tuition by August 2026.', input_info=lookup_result). Return (success, create_result)."
  },
  {
    "name": "coffee_spending_limit",
    "last_user_request": "I'm spending too much on coffee. Set a $40 monthly limit.",
    "previous_conversation": "Assistant: You spent $85 on Starbucks and Dunkin last month.",
    "ideal_response": "Expected: direct create_budget_or_goal(creation_request='Create a monthly coffee budget of $40.', input_info='User spent $85 last month.'). Return (success, create_result)."
  },
  {
    "name": "hawaii_summer_goal",
    "last_user_request": "I want to go to Hawaii next summer. Can you help me figure out the cost and set a goal?",
    "previous_conversation": "",
    "ideal_response": "Expected: research (Hawaii trip cost) -> lookup (finances) -> create_budget_or_goal(creation_request='Create a savings goal for a Hawaii trip by July 2027 based on research.', input_info=strategy_result). Return (success, create_result)."
  },
  {
    "name": "car_down_payment_2years",
    "last_user_request": "I want to buy a new car in 2 years. How much should I save monthly for a $10,000 down payment?",
    "previous_conversation": "",
    "ideal_response": "Expected: research (monthly savings calculation) -> lookup (accounts) -> create_budget_or_goal(creation_request='Create a monthly savings goal for a $10,000 car down payment over 24 months.', input_info=strategy_result). Return (success, create_result)."
  },
  {
    "name": "london_budget_check",
    "last_user_request": "Is $2,000 enough for a week in London?",
    "previous_conversation": "User: I'm planning a solo trip to the UK.",
    "ideal_response": "Expected: research_and_strategize_financial_outcomes(strategize_request='Evaluate if $2,000 is sufficient for a one-week solo trip to London, including typical costs.'). Return (success, strategy_result)."
  },
  {
    "name": "total_spending_cap_march",
    "last_user_request": "Cap my total spending at $3,000 for March.",
    "previous_conversation": "",
    "ideal_response": "Expected: direct create_budget_or_goal(creation_request='Create a total spending budget of $3,000 for the month of March.', input_info=None). Return (success, create_result)."
  },
  {
    "name": "grocery_comparison_last_year",
    "last_user_request": "Create a $500 grocery budget for this month. How does this compare to my spending last year?",
    "previous_conversation": "",
    "ideal_response": "Expected: lookup (historical grocery spending) -> research (comparison/advice) -> create_budget_or_goal(creation_request='Create a $500 grocery budget for this month.', input_info=lookup_result). Return (success, create_result)."
  },
  {
    "name": "house_down_payment_surplus",
    "last_user_request": "I want to save for a house down payment. Based on my current surplus, what's a realistic goal for next year?",
    "previous_conversation": "Assistant: Your monthly income is $5,000 and expenses are $3,800.",
    "ideal_response": "Expected: lookup (confirm $1,200 surplus) -> research (realistic down payment goal for 12 months) -> create_budget_or_goal(creation_request='Create a savings goal for a house down payment by end of next year.', input_info=strategy_result). Return (success, create_result)."
  }
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


def run_test(test_name_or_index_or_dict, optimizer: GoalAgentOptimizer = None):
  """
  Run a single test by name, index, or by passing test data directly.
  
  Args:
    test_name_or_index_or_dict: One of:
      - Test case name (str): e.g., "save_10000_to_end_of_year"
      - Test case index (int): e.g., 0
      - Test data dict: {"last_user_request": "...", "previous_conversation": "..."}
    optimizer: Optional GoalAgentOptimizer instance. If None, creates a new one.
    
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
      result = _run_test_with_logging(
        test_name_or_index_or_dict["last_user_request"],
        test_name_or_index_or_dict.get("previous_conversation", ""),
        optimizer
      )
      if test_name_or_index_or_dict.get("ideal_response"):
        print("\n" + "=" * 80 + "\nIDEAL RESPONSE:\n" + "=" * 80 + "\n" + test_name_or_index_or_dict["ideal_response"] + "\n" + "=" * 80 + "\n")
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
  if test_case.get("ideal_response"):
    print("\n" + "=" * 80 + "\nIDEAL RESPONSE:\n" + "=" * 80 + "\n" + test_case["ideal_response"] + "\n" + "=" * 80 + "\n")
  return result


def run_tests(test_names_or_indices_or_dicts, optimizer: GoalAgentOptimizer = None):
  """
  Run multiple tests by names, indices, or by passing test data directly.
  
  Args:
    test_names_or_indices_or_dicts: One of:
      - None: Run all tests from TEST_CASES
      - List of test case names (str), indices (int), or test data dicts
        Each dict should have: {"last_user_request": "...", "previous_conversation": "..."}
    optimizer: Optional GoalAgentOptimizer instance. If None, creates a new one.
    
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


def test_with_inputs(last_user_request: str, previous_conversation: str, optimizer: GoalAgentOptimizer = None):
  """
  Convenient method to test the goal agent optimizer with custom inputs.
  
  Args:
    last_user_request: The last user request as a string
    previous_conversation: The previous conversation as a string
    optimizer: Optional GoalAgentOptimizer instance. If None, creates a new one.
    
  Returns:
    The generated response string
  """
  return _run_test_with_logging(last_user_request, previous_conversation, optimizer)


def main(test: str = None, no_thinking: bool = False):
  """Main: run single test (name or index), all tests, or show usage. no_thinking=True sets thinking_budget=0."""
  optimizer = GoalAgentOptimizer(thinking_budget=0 if no_thinking else 4096)

  if test is not None:
    if test.strip().lower() == "all":
      print(f"\n{'='*80}")
      print("Running ALL test cases")
      print(f"{'='*80}\n")
      for i in range(len(TEST_CASES)):
        run_test(i, optimizer)
        if i < len(TEST_CASES) - 1:
          print("\n" + "-" * 80 + "\n")
      return
    test_val = int(test) if test.isdigit() else test
    result = run_test(test_val, optimizer)
    if result is None:
      print("\nAvailable test cases:")
      for i, tc in enumerate(TEST_CASES):
        print(f"  {i}: {tc['name']}")
      print("  all: run all test cases")
    return

  print("Usage:")
  print("  Run a single test: --test <name_or_index>")
  print("  Run all tests: --test all")
  print("  Disable thinking: --no-thinking (thinking_budget=0)")
  print("\nAvailable test cases:")
  for i, tc in enumerate(TEST_CASES):
    print(f"  {i}: {tc['name']}")
  print("  all: run all test cases")


if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser(description="Run goal agent optimizer tests")
  parser.add_argument("--test", type=str, help='Test name or index (e.g. "set_food_budget_500_next_month" or "0")')
  parser.add_argument("--no-thinking", action="store_true", help="Set thinking_budget=0 (Thinking OFF) for comparison")
  args = parser.parse_args()
  main(test=args.test, no_thinking=args.no_thinking)


from google import genai
from google.genai import types
from google.genai.errors import ClientError
import os
import sys
from datetime import datetime
from dotenv import load_dotenv

# Add the parent directory to the path so we can import the tools
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from penny.tool_funcs.lookup_user_accounts_transactions_income_and_spending_patterns import (
    lookup_user_accounts_transactions_income_and_spending_patterns,
)
from penny.tool_funcs.create_budget_or_goal_or_reminder import (
    create_budget_or_goal_or_reminder,
    extract_python_code as extract_python_code_from_create,
)
from create_budget_or_goal_optimizer import CreateBudgetOrGoal
import sandbox
from penny.tool_funcs.research_and_strategize_financial_outcomes import (
    research_and_strategize_financial_outcomes,
)
from penny.tool_funcs.update_transaction_category_or_create_category_rules import (
    update_transaction_category_or_create_category_rules,
)

load_dotenv()

SYSTEM_PROMPT = """You are a financial planner agent very good at understanding conversation.

## Your Tasks

1. **Prioritize the Last User Request**: Your main goal is to create a plan that directly addresses the **Last User Request**.
2. **Analyze User Intent**: Analyze the **Last User Request** in the context of the other previous messages in **Previous Conversation**. Determine if the user is making a request, asking a question, or providing information. If **Last User Request** is unintelligible, return `(True, brief clarification)` without calling any skill functions. If the request is **not supported by any available skill** (see rule 5), do not call any skills; return `(True, brief polite message)` (one or two sentences) explaining what you can help with.
3. **Use Previous Conversation for Context ONLY**:
    - If the **Last User Request** is a follow-up (e.g., "yes, do that"), use the context.
    - If the **Last User Request** is vague (e.g., "what about the other thing?"), use the context.
    - **CRITICAL**: For all skill requests, thoroughly analyze the `Previous Conversation` to gain an accurate understanding of the user's intent, identify any unresolved issues, and ensure the request parameter of the skill function is comprehensive and contextually relevant.
    - **If the Last User Request is a new, general question (e.g., "how's my accounts doing?"), DO NOT use specific details from the Previous Conversation in your plan.**
4. **Create a Focused Plan**: The steps in your plan should only be for achieving the **Last User Request**. Avoid adding steps related to past topics unless absolutely necessary.
5. **Output Python Code**: The plan must be written as a Python function `execute_plan`.

Write a python function `execute_plan` that takes no arguments:
  - Express actionable steps as **calls to skill functions**, passing in a natural language request and optionally another input from another skill.
  - Do not use other python functions, just available skill functions, conditional operations and string concatenations.

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
  - Avoid calling `create_budget_or_goal` unless the user explicitly asks to *create* a budget or goal.
- **For Simple Informational Questions (e.g., "how's my accounts doing?"):**
  - `lookup_user_accounts_transactions_income_and_spending_patterns` alone is often sufficient. If its output directly answers the question, return it immediately.
  - Avoid chaining with `research_and_strategize_financial_outcomes` if no analysis, planning, or strategy is requested.
- CRITICAL: If lookup provides sufficient information to answer the question, return it directly without additional skills. Avoid adding strategize "just to be thorough" - only add it if truly needed.

**3. `create_budget_or_goal` — Budgets and Goals Only:**
- `create_budget_or_goal` only creates spending limits (category budgets), income goals, and savings goals. It does not handle requests that are only about being notified or alerted or a one-off task by date. For those, do not call it; return `(True, brief polite message)` that you cannot do that and what you can help with instead.
- If the Last User Request is a question (e.g., "what are the steps to save money?"), avoid using `create_budget_or_goal`.
- Only use `create_budget_or_goal` when the user explicitly asks you to *create*, *set up*, *establish*, or *track* a **budget** (spending limit) or **savings/income goal**.

**4. Acknowledgments / No-Action Messages:** If the Last User Request does not require any new financial data, analysis, or creation (e.g., pure gratitude, social niceties, or "ok"/"got it" just acknowledging information already shown), do not call any skills and instead return `(True, brief natural response)`. Exception: if the user is clearly confirming a prior question (e.g. "yes" or "ok" to "Should I create that?"), call the appropriate skill instead.

**5. Requests Not Supported by Available Skills:** The available skills only support: (1) looking up user accounts/transactions/income/spending/subscriptions, (2) creating spending limits or income/savings goals, (3) researching and strategizing financial outcomes, (4) updating transaction categories or creating category rules. If the Last User Request clearly falls outside these, do not call any skills. Return `(True, brief polite message)` (one or two sentences) that you cannot do that and what you can help with.

<AVAILABLE_SKILL_FUNCTIONS>

These are the **available skills** that can be stacked and sequenced using `input_info` for efficient information flow between steps.
- All of these skills can accept **multiple requests**, written as multiple sentences in their request parameters.
- All **skill functions** return a `tuple[bool, str]`.
    - The first element is `success` boolean.  `True` if information was found and the request was achieved and `False` if information not available, an error occured or more information needed from the user.
    - The second element is `output_info` string.  It contains the output of the **skill function** that should be used as `input_info` in a subsequent skill function call if relevant to the next step.
    - **CRITICAL**: For all skill functions, ensure that the request parameters (e.g., `lookup_request`, `creation_request`, `strategize_request`, `categorize_request`) effectively incorporate relevant information from the `Previous Conversation` and, when available, the `input_info` to accurately address the `Last User Request`.

### List of all Skill Functions

- `lookup_user_accounts_transactions_income_and_spending_patterns(lookup_request: str, input_info: str = None) -> tuple[bool, str]`
  - `lookup_request` is the detailed information requested, written in natural language to lookup about the user's accounts, transactions including income and spending, subscriptions and compare them. It also excels at collecting user data, and performing any summaries through calculations or assessments including forecasted income and spending, and any computations necessary on this. **When `input_info` is available, it is highly recommended to incorporate that information concisely into the `lookup_request` to refine the search and ensure accuracy.**
  - Lookup request can also be about expected and future weekly/monthly income or spending.  Lookup request must phrase the best natural language output needed towards the plan to answer the user.
- `create_budget_or_goal(creation_request: str, input_info: str = None) -> tuple[bool, str]`
  - Creates **spending limits** (category budgets), **income goals**, or **savings goals** only. Does not handle notification/alert or one-off-by-date requests.
  - `creation_request` is what needs to be created factoring in the information coming in from `input_info`.  The request must be descriptive and capture the original user request.  **When `input_info` is available, it is highly recommended to incorporate it to make the creation request precise and context-aware.**
  - Function output `str` is the detail of what was created.
  - If more information is needed from the user, `success` will be `False` and the information needed will be in the `output_info` string.
- `research_and_strategize_financial_outcomes(strategize_request: str, input_info: str = None) -> tuple[bool, str]`
  - `strategize_request` is what needs to be thought out, planned or strategized. It can contain research information (e.g., "average dining out for a couple in Chicago, Illinois", "estimated cost of a flight from Manila to Greece") and factor in information from `input_info`. **When `input_info` is available, it is highly recommended to incorporate that information concisely into the `strategize_request` to refine the strategy and make it as precise as possible.**
  - This skill can financially plan for the future, lookup feasibility and overall provide assessment of different simulated outcomes with finances.
- `update_transaction_category_or_create_category_rules(categorize_request: str, input_info: str = None) -> tuple[bool, str]`
  - `categorize_request` is a description of the category rule that needs to be created, or the description of the transaction that needs to be recategorized. This can be a single transaction, or a group of transactions with criteria. **When `input_info` is available, it is highly recommended to incorporate it to make the categorization request precise and consistent.**
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
    success, budget_result = create_budget_or_goal(
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
    """Handles Gemini API interactions for financial planning using P:AgentPlanner system prompt."""

    def __init__(self, model_name: str = "gemini-flash-lite-latest", thinking_budget: int = 4096):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY environment variable is not set. Set it in .env or environment."
            )
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name
        self.thinking_budget = thinking_budget
        self.temperature = 0.6
        self.top_p = 0.95
        self.max_output_tokens = 8192
        self.safety_settings = [
            types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"),
            types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"),
            types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"),
            types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="OFF"),
        ]
        self.system_prompt = SYSTEM_PROMPT

    def generate_response(
        self,
        last_user_request: str,
        previous_conversation: str,
        replacements: dict = None,
        include_thoughts: bool = True,
    ) -> str:
        system_prompt = self.system_prompt
        if replacements:
            for key, value in replacements.items():
                system_prompt = system_prompt.replace(f"|{key}|", str(value))
        today = datetime.now().strftime("%B %d, %Y")
        system_prompt = system_prompt.replace("|TODAY_DATE|", today)

        request_text = types.Part.from_text(
            text=f"""**Last User Request**: {last_user_request}

**Previous Conversation**:

{previous_conversation}

output:"""
        )
        contents = [types.Content(role="user", parts=[request_text])]
        generate_content_config = types.GenerateContentConfig(
            temperature=self.temperature,
            top_p=self.top_p,
            max_output_tokens=self.max_output_tokens,
            safety_settings=self.safety_settings,
            system_instruction=[types.Part.from_text(text=system_prompt)],
            thinking_config=types.ThinkingConfig(
                thinking_budget=self.thinking_budget,
                include_thoughts=include_thoughts,
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
                                        thought_summary = (
                                            (thought_summary + part.text) if thought_summary else part.text
                                        )
        except ClientError as e:
            if self.thinking_budget == 0 and "only works in thinking mode" in (str(e) or ""):
                print(
                    "\n[NOTE] This model requires thinking mode; API rejected thinking_budget=0. "
                    "Use default (no --no-thinking) or a different model.",
                    flush=True,
                )
                sys.exit(1)
            raise

        if thought_summary:
            print("\n" + "-" * 80)
            print("THOUGHT SUMMARY:")
            print("-" * 80)
            print(thought_summary.strip())
            print("-" * 80 + "\n")

        return output_text


def extract_python_code(text: str) -> str:
    """Extract Python code from generated response (```python blocks)."""
    code_start = text.find("```python")
    if code_start != -1:
        code_start += len("```python")
        code_end = text.find("```", code_start)
        if code_end != -1:
            return text[code_start:code_end].strip()
        return text[code_start:].strip()
    return text.strip()


def _run_test_with_logging(
    last_user_request: str,
    previous_conversation: str,
    planner: "PlannerOptimizer" = None,
    replacements: dict = None,
    include_thoughts: bool = True,
):
    if planner is None:
        planner = PlannerOptimizer()
    llm_input = f"""**Last User Request**: {last_user_request}

**Previous Conversation**:

{previous_conversation}

output:"""
    print("=" * 80)
    print("LLM INPUT:")
    print("=" * 80)
    print(llm_input)
    print("=" * 80 + "\n")

    result = planner.generate_response(
        last_user_request,
        previous_conversation,
        replacements=replacements,
        include_thoughts=include_thoughts,
    )

    print("=" * 80)
    print("LLM OUTPUT:")
    print("=" * 80)
    print(result)
    print("=" * 80 + "\n")

    code = extract_python_code(result)
    if code:
        print("=" * 80)
        print("EXECUTING GENERATED CODE:")
        print("=" * 80)
        try:
            def wrapped_lookup(*args, **kwargs):
                print("\n[FUNCTION CALL] lookup_user_accounts_transactions_income_and_spending_patterns")
                print(f"  args: {args}")
                out = lookup_user_accounts_transactions_income_and_spending_patterns(*args, **kwargs)
                print(f"  [RETURN] success: {out[0]}")
                print(f"  [RETURN] output: {out[1]}")
                return out

            def wrapped_create(*args, **kwargs):
                print("\n[FUNCTION CALL] create_budget_or_goal")
                print(f"  args: {args}")
                print(f"  kwargs: {kwargs}")
                creation_request = args[0] if len(args) >= 1 else kwargs.get("creation_request")
                input_info = args[1] if len(args) >= 2 else kwargs.get("input_info")
                user_id = kwargs.get("user_id", 1)
                generated_code = None
                if creation_request:
                    generator = CreateBudgetOrGoal()
                    raw_response = generator.generate_response(creation_request, input_info)
                    generated_code = extract_python_code_from_create(raw_response)
                    print("\n  [GENERATED PYTHON CODE]:")
                    print("  " + "-" * 76)
                    for line in generated_code.split("\n"):
                        print(f"  {line}")
                    print("  " + "-" * 76)
                if generated_code and generated_code.strip():
                    try:
                        success, output_string, _logs, _goals = sandbox.execute_agent_with_tools(
                            generated_code, user_id
                        )
                        out = (success, output_string)
                    except Exception as e:
                        import traceback
                        out = (False, f"**Execution Error**: `{str(e)}`\n{traceback.format_exc()}")
                else:
                    out = create_budget_or_goal_or_reminder(*args, **kwargs)
                print(f"  [RETURN] success: {out[0]}")
                print(f"  [RETURN] output: {out[1]}")
                return out

            def wrapped_research(*args, **kwargs):
                print("\n[FUNCTION CALL] research_and_strategize_financial_outcomes")
                print(f"  args: {args}")
                out = research_and_strategize_financial_outcomes(*args, **kwargs)
                print(f"  [RETURN] success: {out[0]}")
                print(f"  [RETURN] output: {out[1]}")
                return out

            def wrapped_update(*args, **kwargs):
                print("\n[FUNCTION CALL] update_transaction_category_or_create_category_rules")
                print(f"  args: {args}")
                out = update_transaction_category_or_create_category_rules(*args, **kwargs)
                print(f"  [RETURN] success: {out[0]}")
                print(f"  [RETURN] output: {out[1]}")
                return out

            namespace = {
                "lookup_user_accounts_transactions_income_and_spending_patterns": wrapped_lookup,
                "create_budget_or_goal": wrapped_create,
                "research_and_strategize_financial_outcomes": wrapped_research,
                "update_transaction_category_or_create_category_rules": wrapped_update,
            }
            exec(code, namespace)
            if "execute_plan" in namespace:
                print("\n" + "=" * 80)
                print("Calling execute_plan()...")
                print("=" * 80)
                res = namespace["execute_plan"]()
                print("\n" + "=" * 80)
                print("execute_plan() FINAL RESULT:")
                print("=" * 80)
                print(f"  success: {res[0]}")
                print(f"  output: {res[1]}")
                print("=" * 80)
            else:
                print("Warning: execute_plan() not found in generated code")
                print("=" * 80)
        except Exception as e:
            print(f"Error executing generated code: {str(e)}")
            import traceback
            print(traceback.format_exc())
            print("=" * 80)
    return result


# Applicable test cases from planner_optimizer_v2 (no add_to_memory / follow_up_conversation)
TEST_CASES = [
    {
        "name": "hows_my_accounts_doing",
        "last_user_request": "how's my accounts doing?",
        "previous_conversation": """User: Hey, do I have enough to cover rent this month?
Assistant: You're getting close! Your checking accounts have $1,850, and rent is $2,200. You'll need about $350 more by the due date.
User: Ugh, okay. Am I spending too much? Like am I going over what I make?
Assistant: You're actually staying within your means, but just barely. After all expenses, you're only saving about $50 a month, which is pretty tight.
User: Yeah that makes sense, I just got back from a trip to Europe. How bad was it?
Assistant: The trip definitely added up! Over the past two weeks, you spent $1,890 on travel and hotels, plus $520 on restaurants and dining out.""",
        "ideal_response": "Expected: lookup_user_accounts_transactions_income_and_spending_patterns only (general question). Return (success, lookup_result). No strategize.",
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
Assistant: I don't have access to your credit score, but I can help you track your spending patterns that affect it.""",
        "ideal_response": "Expected: lookup only for net worth summary. Return (success, lookup_result). No strategize (informational).",
    },
    {
        "name": "research_and_strategize_savings_plan",
        "last_user_request": "I want to save $10,000 for a down payment on a house. What's the best plan to get there?",
        "previous_conversation": """User: How much am I spending on dining out?
Assistant: Over the last 3 months, you've spent an average of $450 per month on dining out.
User: That seems high. What about my overall spending?
Assistant: Your total monthly spending averages around $3,200, and your monthly income is about $4,500.""",
        "ideal_response": "Expected: lookup first (income/spending), then research_and_strategize_financial_outcomes for savings plan. Return (success, strategy_result). No create_budget unless user asks to create.",
    },
    {
        "name": "research_and_strategize_vacation_affordability",
        "last_user_request": "Is it feasible for me to take a 2-week vacation to Japan next year? Research the costs and tell me if I can afford it.",
        "previous_conversation": """User: What's my current account balance?
Assistant: You have $5,200 in your checking account and $3,100 in savings.
User: How much am I saving per month?
Assistant: Based on your recent spending patterns, you're saving approximately $800 per month.""",
        "ideal_response": "Expected: lookup (current finances) then research (Japan trip costs / feasibility). Return (success, strategy_result or combined message).",
    },
    {
        "name": "save_5000_in_6_months",
        "last_user_request": "I want to save $5000 in the next 6 months.",
        "previous_conversation": """User: How much am I spending on dining out?
Assistant: Over the last 3 months, you've spent an average of $450 per month on dining out.
User: What about my overall spending?
Assistant: Your total monthly spending averages around $3,200, and your monthly income is about $4,500.""",
        "ideal_response": "Expected: lookup first, then research for plan. Optionally create_budget_or_goal only if user intent is to create a goal (savings goal is explicit).",
    },
    {
        "name": "set_food_budget_500_next_month",
        "last_user_request": "set a food budget of $500 for next month.",
        "previous_conversation": "",
        "ideal_response": "Expected: create_budget_or_goal(creation_request='...food budget $500 next month...'). Optionally lookup first for context.",
    },
    {
        "name": "save_10000_to_end_of_year",
        "last_user_request": "save $10000 up to end of year.",
        "previous_conversation": "",
        "ideal_response": "Expected: lookup (accounts) then create_budget_or_goal for savings goal. Optionally research for plan.",
    },
    {
        "name": "reminder_cancel_spotify_end_of_year",
        "last_user_request": "remind me to cancel Spotify subscription at the end of this year.",
        "previous_conversation": "",
        "ideal_response": "Expected: No create_budget_or_goal call (request not supported). execute_plan returns (True, brief message that this cannot be done; can help with budgets/goals instead).",
    },
    {
        "name": "reminder_cancel_netflix_november_30",
        "last_user_request": "remind me to cancel Netflix on November 30th.",
        "previous_conversation": "",
        "ideal_response": "Expected: No create_budget_or_goal call (request not supported). execute_plan returns (True, brief message that this cannot be done; can help with budgets/goals instead).",
    },
    {
        "name": "reminder_checking_account_balance_below_1000",
        "last_user_request": "notify me when my checking account balance drops below $1000.",
        "previous_conversation": "",
        "ideal_response": "Expected: No create_budget_or_goal call (request not supported). execute_plan returns (True, brief message that this cannot be done; can help with budgets/goals instead).",
    },
    {
        "name": "reminder_savings_account_balance_below_1000",
        "last_user_request": "notify me when my savings account balance drops below $1000.",
        "previous_conversation": "",
        "ideal_response": "Expected: No create_budget_or_goal call (request not supported). execute_plan returns (True, brief message that this cannot be done; can help with budgets/goals instead).",
    },
    {
        "name": "categorize_transaction_travel",
        "last_user_request": "that's for my disneyland trip for next month. categorize it as travel.",
        "previous_conversation": """Assistant: There's an uncategorized $525 transaction.""",
        "ideal_response": "Expected: update_transaction_category_or_create_category_rules(categorize_request='...$525 transaction...travel...Disneyland...'). Optionally lookup to find transaction. No add_to_memory (not in P:AgentPlanner skills).",
    },
    {
        "name": "thanks_acknowledgment",
        "last_user_request": "thanks",
        "previous_conversation": """User: how about next week ?
Penny: For next week, starting March 1st, you're looking at earning $1,462 and spending about $5,031.
😬 This includes $62 for Kids Shopping and $68 for Gadgets.
I see 24 more forecasts too!""",
        "ideal_response": "Expected: No skill calls (acknowledgment only). execute_plan returns (True, a brief polite acknowledgment). Plan should not call lookup/create/research/categorize.",
    },
    {
        "name": "ok_yes_to_penny_question",
        "last_user_request": "ok",
        "previous_conversation": """User: set a savings goal of $2000 by end of the month
Penny: I can create a savings goal of $2000 by end of the month. Should I go ahead and create that for you?""",
        "ideal_response": "Expected: 'ok' means yes to Penny's question. execute_plan should call create_budget_or_goal(creation_request='...savings goal $2000 by end of month...'). Do NOT return (True, \"You're welcome!\") or treat as acknowledgment.",
    },
    {
        "name": "ok_got_it_acknowledgment",
        "last_user_request": "ok",
        "previous_conversation": """User: how much did I spend on dining last month?
Penny: You spent $412 on dining out last month—about $85 over your usual.""",
        "ideal_response": "Expected: 'ok' means 'I see' or 'got it' (acknowledging the answer). No skill calls. execute_plan returns (True, brief natural acknowledgment). Do NOT return 'You're welcome!' (user did not thank). Do NOT call lookup/create/research/categorize.",
    },
    {
        "name": "unintelligible_request",
        "last_user_request": "asdfgh qwerty zxcv ???",
        "previous_conversation": "",
        "ideal_response": "Expected: Last User Request is unintelligible. No skill calls. execute_plan returns (True, brief clarification asking what the user meant). Do NOT call lookup/create/research/categorize.",
    },
]


def get_test_case(test_name_or_index):
    if isinstance(test_name_or_index, int):
        if 0 <= test_name_or_index < len(TEST_CASES):
            return TEST_CASES[test_name_or_index]
        return None
    if isinstance(test_name_or_index, str):
        for tc in TEST_CASES:
            if tc["name"] == test_name_or_index:
                return tc
        return None
    return None


def run_test(
    test_name_or_index_or_dict,
    planner: "PlannerOptimizer" = None,
    replacements: dict = None,
    include_thoughts: bool = True,
):
    if isinstance(test_name_or_index_or_dict, dict):
        if "last_user_request" not in test_name_or_index_or_dict:
            print("Invalid test dict: must contain 'last_user_request'.")
            return None
        name = test_name_or_index_or_dict.get("name", "custom_test")
        repl = test_name_or_index_or_dict.get("replacements", replacements)
        print(f"\n{'='*80}\nRunning test: {name}\n{'='*80}\n")
        result = _run_test_with_logging(
            test_name_or_index_or_dict["last_user_request"],
            test_name_or_index_or_dict.get("previous_conversation", ""),
            planner,
            replacements=repl,
            include_thoughts=include_thoughts,
        )
        if test_name_or_index_or_dict.get("ideal_response"):
            print(
                "\n" + "=" * 80 + "\nIDEAL RESPONSE:\n" + "=" * 80 + "\n"
                + test_name_or_index_or_dict["ideal_response"] + "\n" + "=" * 80 + "\n"
            )
        return result

    tc = get_test_case(test_name_or_index_or_dict)
    if tc is None:
        print(f"Test case '{test_name_or_index_or_dict}' not found.")
        return None
    print(f"\n{'='*80}\nRunning test: {tc['name']}\n{'='*80}\n")
    result = _run_test_with_logging(
        tc["last_user_request"],
        tc["previous_conversation"],
        planner,
        replacements=replacements,
        include_thoughts=include_thoughts,
    )
    if tc.get("ideal_response"):
        print(
            "\n" + "=" * 80 + "\nIDEAL RESPONSE:\n" + "=" * 80 + "\n"
            + tc["ideal_response"] + "\n" + "=" * 80 + "\n"
        )
    return result


def run_tests(
    test_names_or_indices_or_dicts=None,
    planner: "PlannerOptimizer" = None,
    replacements: dict = None,
    include_thoughts: bool = True,
):
    if test_names_or_indices_or_dicts is None:
        test_names_or_indices_or_dicts = list(range(len(TEST_CASES)))
    results = []
    for item in test_names_or_indices_or_dicts:
        results.append(
            run_test(item, planner, replacements=replacements, include_thoughts=include_thoughts)
        )
    return results


def test_with_inputs(
    last_user_request: str,
    previous_conversation: str,
    planner: "PlannerOptimizer" = None,
    replacements: dict = None,
):
    return _run_test_with_logging(
        last_user_request, previous_conversation, planner, replacements=replacements
    )


def main(test: str = None, no_thinking: bool = False):
    optimizer = PlannerOptimizer(thinking_budget=0 if no_thinking else 4096)
    if test is not None:
        if test.strip().lower() == "all":
            print(f"\n{'='*80}\nRunning ALL test cases\n{'='*80}\n")
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
    parser = argparse.ArgumentParser(description="Run planner optimizer v3 tests (P:AgentPlanner)")
    parser.add_argument("--test", type=str, help='Test name or index (e.g. "hows_my_accounts_doing" or "0")')
    parser.add_argument("--no-thinking", action="store_true", help="Set thinking_budget=0")
    args = parser.parse_args()
    main(test=args.test, no_thinking=args.no_thinking)

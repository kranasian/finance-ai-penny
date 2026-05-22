from google import genai
from google.genai import types
import os
import json
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

SYSTEM_PROMPT = """You are a checker for create budget or goal optimizer outputs.

## Input
- **EVAL_INPUT**: Creation Request + Input Info (date, accounts, prior conversation)
- **GENERATED_CODE**: Python from the optimizer
- **EXECUTION_RESULT**: Tool logs and final return string
- **PAST_REVIEW_OUTCOMES**: optional prior reviews

## Output
JSON only: `{"good_copy": boolean, "info_correct": boolean, "eval_text": string}`

### Split (judge independently)
- **info_correct**: `GENERATED_CODE` matches `EVAL_INPUT` intent.
- **good_copy**: `EXECUTION_RESULT` matches `GENERATED_CODE` execution and user-message rules below.
- Faithful execution of wrong intent → `info_correct: false`, `good_copy: true`.
- Never set `good_copy: false` only because `info_correct` is false.
- `message:` / `goal count:` → `good_copy` only. `category:` / `goal_type:` / `end_date:` / `duration:` / `account_ids:` / `intent:` / `amount:` / `granularity:` → `info_correct` only. Never both `category:` and `message:` for the same flaw.

### info_correct
- Missing info → clarification must ask the **right** fields; wrong field → false.
- Sufficient info → executable code, not clarification.
- Multiple goals: create all or ask for all missing before any create; partial create → false.
- Phased strategy: each phase needs its own goal/budget and dates.
- Prior conversation: confirmed category/amount settled; re-ask → false.

**goal_type**: `save_X_amount` = total by deadline (per-period amounts as *how* still `save_X_amount`). `save_0` = fixed per period, no total, no end date. Wrong type → `goal_type:`. Savings + spending limit in request → both tools.

**intent**: Amount + category with no save-vs-limit language → clarify before create; `intent:`.

**granularity**: Unspecified OK for `save_X_amount` — default to any period **shorter than** the goal duration (e.g. weekly/monthly for a multi-month target; not yearly for a 10-month goal). Required for `save_0`, spending limits, income goals. Do not flag granularity when amount/deadline is the real gap for total-by-date savings.

**category** (spending limits; slugs in Categories):
- User named parent → parent OK.
- **Category-level term** (restaurants, groceries, utilities, school payments): maps to a dedicated subcategory slug. Wrong slug or parent → `info_correct: false`, `category:` only. If execution matches code, `good_copy: true` even when message uses a broad word (e.g. "food budget" with `meals`). **No `message:` disclaimer** for this case.
- **Merchant or line-item narrower than subcategory** (McDonald's, water bills): best subcategory without confirmation → `info_correct: true`; scope disclaimer → `good_copy` / `message:` only.
- Synonym equals subcategory (school payments + `education_tuition`) → true.

**account_ids**: only when user names storage account; balances listed without choice → must be blank.

**dates/amounts**: fractional periods round **up** for `end_date`; implied amounts (net $0, percent of income) need no "how much" ask; year from Input Info date.

### good_copy
- Tool calls and return match `GENERATED_CODE`.
- Brief confirmations ("Budget created.") OK when no disclaimer or multi-goal ack needed.
- **Disclaimer required** (merchant/line-item only): McDonald's, named store, water bills, etc. mapped to subcategory → message must state budget covers the **whole subcategory**; generic "Budget created." → `good_copy: false`, `info_correct: true`, `message:`.
- **Disclaimer not required**: category-level request term; synonym (school payments + `education_tuition`); rent + `shelter_home`.
- **Wrong category slug/parent**: execution matches code → `good_copy: true`, `info_correct: false`, `category:` only — never `message:`.
- N goals/budgets created → message acknowledges all N or `goal count:`.
- Clarification lists every missing field still needed.

### eval_text
- Empty when both true. Short `parameter: issue` lines. Priority: `goal_type:` → `intent:` → `category:` → `message:` → `account_ids:` → `goal count:` → `amount:` / `granularity:` / `end_date:` / `duration:`.

### Categories
meals → meals_groceries, meals_dining_out, meals_delivered_food | leisure → leisure_entertainment, leisure_travel | bills → bills_connectivity, bills_insurance, bills_tax, bills_service_fees | shelter → shelter_home, shelter_utilities, shelter_upkeep | education → education_kids_activities, education_tuition | shopping → shopping_clothing, shopping_gadgets, shopping_kids, shopping_pets | transportation → transportation_public, transportation_car | health → health_medical_pharmacy, health_gym_wellness, health_personal_care | donations_gifts, uncategorized, miscellaneous
"""

class CheckCreateBudgetOrGoalOptimizer:
    """Handles all Gemini API interactions for checking create budget or goal optimizer outputs against rules"""
    
    def __init__(self, model_name="gemini-3-flash-preview"):
        """Initialize the Gemini agent with API configuration"""
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is not set.")
        self.client = genai.Client(api_key=api_key)
        
        # run_config parameters
        self.model_name = model_name
        self.top_k = 40
        self.top_p = 0.95
        self.temperature = 0.2
        self.thinking_budget = 2048
        self.max_output_tokens = 2176
        
        self.safety_settings = [
            types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"),
            types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"),
            types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"),
            types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="OFF")
        ]
        
        self.system_prompt = SYSTEM_PROMPT

    def generate_response(self, request_text) -> dict:
        contents = [types.Content(role="user", parts=[request_text])]
        
        generate_content_config = types.GenerateContentConfig(
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
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
                        if hasattr(candidate, "content") and candidate.content and getattr(candidate.content, "parts", None):
                            for part in candidate.content.parts:
                                if getattr(part, "thought", False) and getattr(part, "text", None):
                                    thought_summary = (thought_summary + part.text) if thought_summary else part.text
        except Exception as e:
            raise e

        if thought_summary:
            print("\n" + "-" * 80)
            print("CHECKER THOUGHT SUMMARY:")
            print("-" * 80)
            print(thought_summary.strip())
            print("-" * 80 + "\n")
        
        if not output_text or not output_text.strip():
            raise ValueError("Empty response from model.")
        
        try:
            if "```json" in output_text:
                json_start = output_text.find("```json") + 7
                json_end = output_text.find("```", json_start)
                if json_end != -1:
                    output_text = output_text[json_start:json_end].strip()
            elif "```" in output_text:
                json_start = output_text.find("```") + 3
                json_end = output_text.find("```", json_start)
                if json_end != -1:
                    output_text = output_text[json_start:json_end].strip()
            
            parsed = json.loads(output_text.strip())
            return parsed
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON response: {e}")

def _run_test_with_logging(creation_request: str, input_info: Optional[str] = None, generated_code: str = "", execution_result: str = "", past_review_outcomes: Optional[list] = None, checker: Optional[CheckCreateBudgetOrGoalOptimizer] = None):
    if checker is None:
        checker = CheckCreateBudgetOrGoalOptimizer()
    
    past_review_section = ""
    if past_review_outcomes:
        for index, past_review_outcome in enumerate(past_review_outcomes):
            past_review_section += f"""<PAST_REVIEW_OUTCOME_{index + 1}>
## Generated Code for #{index + 1}
```python
{past_review_outcome['generated_code']}
```
## Execution Result for #{index + 1}
{past_review_outcome['execution_result']}
## Evaluation Output for #{index + 1}
```json
{json.dumps(past_review_outcome['output'], indent=2)}
```
</PAST_REVIEW_OUTCOME_{index + 1}>
"""
    
    input_info_section = f"**Input Info from previous skill**: {input_info}" if input_info else "**Input Info from previous skill**: None"

    if not generated_code.startswith("def"):
        generated_code_block = f"```python\n{generated_code}\n```"
    else:
        generated_code_block = generated_code

    request_str = f"""<EVAL_INPUT>
**Last User Request**: {creation_request}
{input_info_section}
</EVAL_INPUT>

<GENERATED_CODE>
{generated_code_block}
</GENERATED_CODE>

<EXECUTION_RESULT>
{execution_result}
</EXECUTION_RESULT>

{past_review_section}

Output:"""
    
    print("=" * 80)
    print("LLM INPUT:")
    print("=" * 80)
    print(request_str)
    print("=" * 80)
    
    try:
        result = checker.generate_response(types.Part.from_text(text=request_str))
        print("LLM OUTPUT:")
        print(json.dumps(result, indent=2))
        return result
    except Exception as e:
        print(f"ERROR: {str(e)}")
        return None

def test_batch_1(checker: CheckCreateBudgetOrGoalOptimizer = None):
    # Topic: Merchant-specific budget maps to category; message must disclaim category scope.
    # Accuracy: generated_code sets meals_dining_out but output only confirms creation without dining-out disclaimer.
    creation_request = "Set a $75 monthly budget for my McDonald's purchases."
    input_info = "Today is 2026-05-22."

    generated_code = """def process_input():
    success, result = create_category_spending_limit(
        category="meals_dining_out",
        granularity="monthly",
        amount=75.0,
        title="McDonald's Budget 🍔"
    )
    return success, result"""

    execution_result = """create_category_spending_limit:
  category: "meals_dining_out", amount: 75.0
  return: (True, "Budget created.")
process_input:
True
Budget created."""

    ideal_output = {
        "good_copy": False,
        "info_correct": True,
        "eval_text": "message: should inform user budget is for dining out category, not McDonald's specifically",
    }
    print("IDEAL OUTPUT:")
    print(json.dumps(ideal_output, indent=2))
    return _run_test_with_logging(creation_request, input_info, generated_code, execution_result, None, checker)

def test_batch_2(checker: CheckCreateBudgetOrGoalOptimizer = None):
    # Topic: Utility sub-item maps to shelter_utilities; message must disclaim utilities scope.
    # Accuracy: generated_code sets shelter_utilities but output only confirms creation without utilities disclaimer.
    creation_request = "Set a $120 monthly budget for my water bills."
    input_info = "Today is 2026-05-22."

    generated_code = """def process_input():
    success, result = create_category_spending_limit(
        category="shelter_utilities",
        granularity="monthly",
        amount=120.0,
        title="Water Bills Budget 💧"
    )
    return success, result"""

    execution_result = """create_category_spending_limit:
  category: "shelter_utilities", amount: 120.0
  return: (True, "Budget created.")
process_input:
True
Budget created."""

    ideal_output = {
        "good_copy": False,
        "info_correct": True,
        "eval_text": "message: should inform user budget is for utilities category, not water specifically",
    }
    print("IDEAL OUTPUT:")
    print(json.dumps(ideal_output, indent=2))
    return _run_test_with_logging(creation_request, input_info, generated_code, execution_result, None, checker)

def test_batch_3(checker: CheckCreateBudgetOrGoalOptimizer = None):
    # Topic: Synonymous mapping (school payments → education_tuition).
    # Accuracy: generated_code and generic confirmation are acceptable for synonymous terms.
    creation_request = "Set a $300 monthly budget for school payments."
    input_info = "Today is 2026-05-22."

    generated_code = """def process_input():
    success, result = create_category_spending_limit(
        category="education_tuition",
        granularity="monthly",
        amount=300.0,
        title="School Payments Budget 🎓"
    )
    return success, result"""

    execution_result = """create_category_spending_limit:
  category: "education_tuition", amount: 300.0
  return: (True, "Budget created.")
process_input:
True
Budget created."""

    ideal_output = {
        "good_copy": True,
        "info_correct": True,
        "eval_text": "",
    }
    print("IDEAL OUTPUT:")
    print(json.dumps(ideal_output, indent=2))
    return _run_test_with_logging(creation_request, input_info, generated_code, execution_result, None, checker)

def test_batch_4(checker: CheckCreateBudgetOrGoalOptimizer = None):
    # Topic: Specific subcategory required (restaurants → dining out, not meals parent).
    # Accuracy: generated_code uses meals parent instead of meals_dining_out.
    creation_request = "Set a $150 monthly budget for restaurants."
    input_info = "Today is 2026-05-22."

    generated_code = """def process_input():
    success, result = create_category_spending_limit(
        category="meals",
        granularity="monthly",
        amount=150.0,
        title="Restaurant Budget 🍴"
    )
    return success, result"""

    execution_result = """create_category_spending_limit:
  category: "meals", amount: 150.0
  return: (True, "Your $150/month food budget is set.")
process_input:
True
Your $150/month food budget is set."""

    ideal_output = {
        "good_copy": True,
        "info_correct": False,
        "eval_text": "category: should use meals_dining_out for restaurants, not meals parent",
    }
    print("IDEAL OUTPUT:")
    print(json.dumps(ideal_output, indent=2))
    return _run_test_with_logging(creation_request, input_info, generated_code, execution_result, None, checker)

def test_batch_5(checker: CheckCreateBudgetOrGoalOptimizer = None):
    # Topic: Decimal month duration rounded up for savings timeline.
    # Accuracy: generated_code uses save_0 for 2 months instead of rounding 2.2 months up to 3.
    creation_request = "I want to save $60 over 2.2 months."
    input_info = "Today is 2026-05-22."

    generated_code = """def process_input():
    success, result = create_savings_goal(
        amount=60.0,
        end_date="",
        goal_type="save_0",
        granularity="monthly",
        start_date="2026-05-22",
        title="Short-Term Savings 💰"
    )
    return success, result"""

    execution_result = """create_savings_goal:
  amount: 60.0, goal_type: "save_0", granularity: "monthly", start_date: "2026-05-22"
  return: (True, "Savings goal created for 2 months.")
process_input:
True
Savings goal created for 2 months."""

    ideal_output = {
        "good_copy": True,
        "info_correct": False,
        "eval_text": "duration: 2.2 months should round up to 3 months, not 2",
    }
    print("IDEAL OUTPUT:")
    print(json.dumps(ideal_output, indent=2))
    return _run_test_with_logging(creation_request, input_info, generated_code, execution_result, None, checker)

def test_batch_6(checker: CheckCreateBudgetOrGoalOptimizer = None):
    # Topic: Net-zero amount inferable from refunds vs payments context.
    # Accuracy: generated_code asks for amount when net 0 is implied by refunds covering payments.
    creation_request = "I'm expecting tax refunds and tax payments this month. My refunds should be at least equal to my payments."
    input_info = "Today is 2026-05-22."

    generated_code = """How much would you like to set for your tax-related goal this month?"""

    execution_result = """(no tool calls — clarification text only)"""

    ideal_output = {
        "good_copy": False,
        "info_correct": False,
        "eval_text": "amount: should be net 0 when refunds are at least equal to payments",
    }
    print("IDEAL OUTPUT:")
    print(json.dumps(ideal_output, indent=2))
    return _run_test_with_logging(creation_request, input_info, generated_code, execution_result, None, checker)

def test_batch_7(checker: CheckCreateBudgetOrGoalOptimizer = None):
    # Topic: account_ids set without user specifying storage account.
    # Accuracy: generated_code assigns Savings account_ids though EVAL_INPUT does not name which account.
    creation_request = "Grow my savings to $10,000."
    input_info = """Today is 2026-05-22.
Account 'Checking' (id: 101) | Balance: $2,400
Account 'Savings' (id: 202) | Balance: $6,200
Account 'Current' (id: 303) | Balance: $800"""

    generated_code = """def process_input():
    success, result = create_savings_goal(
        amount=10000.0,
        end_date="",
        goal_type="save_X_amount",
        granularity="monthly",
        account_ids=[202],
        title="Savings Growth 🌱"
    )
    return success, result"""

    execution_result = """create_savings_goal:
  amount: 10000.0, goal_type: "save_X_amount", account_ids: [202]
  return: (True, "Savings goal created in your Savings account.")
process_input:
True
Savings goal created in your Savings account."""

    ideal_output = {
        "good_copy": True,
        "info_correct": False,
        "eval_text": "account_ids: EVAL_INPUT does not specify which account; should be blank",
    }
    print("IDEAL OUTPUT:")
    print(json.dumps(ideal_output, indent=2))
    return _run_test_with_logging(creation_request, input_info, generated_code, execution_result, None, checker)

def test_batch_8(checker: CheckCreateBudgetOrGoalOptimizer = None):
    # Topic: Goal count mismatch between code execution and user message.
    # Accuracy: generated_code creates two goals but final message reports only one.
    creation_request = "Set a $200 monthly budget for groceries and save $1,000 for a trip by December."
    input_info = "Today is 2026-05-22."

    generated_code = """def process_input():
    s1, r1 = create_category_spending_limit(
        category="meals_groceries",
        granularity="monthly",
        amount=200.0,
        title="Grocery Budget 🛒"
    )
    s2, r2 = create_savings_goal(
        amount=1000.0,
        end_date="2026-12-31",
        goal_type="save_X_amount",
        granularity="monthly",
        title="Trip Fund ✈️"
    )
    return True, r1"""

    execution_result = """create_category_spending_limit:
  category: "meals_groceries", amount: 200.0
  return: (True, "Grocery budget set.")
create_savings_goal:
  amount: 1000.0, end_date: "2026-12-31", goal_type: "save_X_amount"
  return: (True, "Trip savings goal created.")
process_input:
True
Grocery budget set."""

    ideal_output = {
        "good_copy": False,
        "info_correct": True,
        "eval_text": "goal count: message should mention both grocery budget and trip savings goal",
    }
    print("IDEAL OUTPUT:")
    print(json.dumps(ideal_output, indent=2))
    return _run_test_with_logging(creation_request, input_info, generated_code, execution_result, None, checker)

def test_batch_9(checker: CheckCreateBudgetOrGoalOptimizer = None):
    # Topic: Ambiguous intent (save vs spending limit).
    # Accuracy: generated_code assumes monthly spending limit without clarifying goal type.
    creation_request = "$500 for pets."
    input_info = "Today is 2026-05-22."

    generated_code = """def process_input():
    success, result = create_category_spending_limit(
        category="shopping_pets",
        granularity="monthly",
        amount=500.0,
        title="Pets Budget 🐾"
    )
    return success, result"""

    execution_result = """create_category_spending_limit:
  category: "shopping_pets", amount: 500.0, granularity: "monthly"
  return: (True, "Your $500/month pets budget is set.")
process_input:
True
Your $500/month pets budget is set."""

    ideal_output = {
        "good_copy": True,
        "info_correct": False,
        "eval_text": "intent: unclear whether to save $500 or limit spending to $500/month; should ask before creating",
    }
    print("IDEAL OUTPUT:")
    print(json.dumps(ideal_output, indent=2))
    return _run_test_with_logging(creation_request, input_info, generated_code, execution_result, None, checker)

def test_batch_10(checker: CheckCreateBudgetOrGoalOptimizer = None):
    # Topic: Parent category required for broad shopping budget.
    # Accuracy: generated_code uses shopping_clothing instead of shopping parent.
    creation_request = "Budget $200 for shopping."
    input_info = "Today is 2026-05-22."

    generated_code = """def process_input():
    success, result = create_category_spending_limit(
        category="shopping_clothing",
        granularity="monthly",
        amount=200.0,
        title="Shopping Budget 🛍️"
    )
    return success, result"""

    execution_result = """create_category_spending_limit:
  category: "shopping_clothing", amount: 200.0
  return: (True, "Your $200/month shopping budget is set.")
process_input:
True
Your $200/month shopping budget is set."""

    ideal_output = {
        "good_copy": True,
        "info_correct": False,
        "eval_text": "category: should use shopping parent, not shopping_clothing",
    }
    print("IDEAL OUTPUT:")
    print(json.dumps(ideal_output, indent=2))
    return _run_test_with_logging(creation_request, input_info, generated_code, execution_result, None, checker)

def test_batch_11(checker: CheckCreateBudgetOrGoalOptimizer = None):
    # Topic: Redundant category reconfirmation after prior conversation.
    # Accuracy: generated_code asks to confirm category already agreed in previous messages.
    creation_request = "$100 weekly."
    input_info = """Today is 2026-05-22.

**Previous Conversation**:
User: I want a monthly budget for dining out.
Assistant: I can set that under meals_dining_out. Does that category work for you?
User: Yes, dining out is correct.
Assistant: Got it — meals_dining_out confirmed."""

    generated_code = """Which category should this budget use — meals_dining_out, meals_groceries, or another meals subcategory?"""

    execution_result = """(no tool calls — clarification text only)"""

    ideal_output = {
        "good_copy": False,
        "info_correct": False,
        "eval_text": "category: already confirmed as meals_dining_out in previous conversation; should not re-ask",
    }
    print("IDEAL OUTPUT:")
    print(json.dumps(ideal_output, indent=2))
    return _run_test_with_logging(creation_request, input_info, generated_code, execution_result, None, checker)

def test_batch_12(checker: CheckCreateBudgetOrGoalOptimizer = None):
    # Topic: Account IDs storage account check.
    # Accuracy: generated_code uses wrong account ID.
    creation_request = "I want to save $2,000 for my emergency fund in my High Yield Savings account."
    input_info = """Account 'Main Checking' (id: 123) | Balance: $5,000
Account 'High Yield Savings' (id: 456) | Balance: $1,000"""

    generated_code = """def process_input():
    success, result = create_savings_goal(
        amount=2000.0,
        account_ids=[123],
        goal_type="save_X_amount",
        title="Emergency Fund 🛡️"
    )
    return success, result"""

    execution_result = """create_savings_goal:
  amount: 2000.0, account_ids: [123]
  return: (True, "Goal created.")
process_input:
True
Goal created."""

    ideal_output = {
        "good_copy": True,
        "info_correct": False,
        "eval_text": "account_ids: should use High Yield Savings (456), not Main Checking (123)",
    }
    print("IDEAL OUTPUT:")
    print(json.dumps(ideal_output, indent=2))
    return _run_test_with_logging(creation_request, input_info, generated_code, execution_result, None, checker)

def main(batch: int = 1):
    print(f"Testing CheckCreateBudgetOrGoalOptimizer - Batch {batch}\\n")
    checker = CheckCreateBudgetOrGoalOptimizer()
    
    if batch == 1:
        test_batch_1(checker)
    elif batch == 2:
        test_batch_2(checker)
    elif batch == 3:
        test_batch_3(checker)
    elif batch == 4:
        test_batch_4(checker)
    elif batch == 5:
        test_batch_5(checker)
    elif batch == 6:
        test_batch_6(checker)
    elif batch == 7:
        test_batch_7(checker)
    elif batch == 8:
        test_batch_8(checker)
    elif batch == 9:
        test_batch_9(checker)
    elif batch == 10:
        test_batch_10(checker)
    elif batch == 11:
        test_batch_11(checker)
    elif batch == 12:
        test_batch_12(checker)
    print("\\n")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Run tests in batches')
    parser.add_argument('--batch', type=int, default=1, choices=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                        help='Batch number to run (1–12)')
    args = parser.parse_args()
    main(batch=args.batch)

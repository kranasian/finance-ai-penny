from google import genai
from google.genai import types
import os
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

SYSTEM_PROMPT = """You are a checker verifying research and strategy outputs against rules.

## Input:
- **EVAL_INPUT**: Contains "**User request**:" followed by the user's financial goal/request, and "**Input Information from previous skill**:" followed by detailed financial data (savings balance, accounts, past transactions, forecasted patterns, savings rate)
- **PAST_REVIEW_OUTCOMES**: Array of past reviews, each with `output`, `good_copy`, `info_correct`, `eval_text`
- **REVIEW_NEEDED**: Research and strategy output to review (string)

## Output:
JSON: `{"good_copy": boolean, "info_correct": boolean, "eval_text": string}`
- `good_copy`: True if REVIEW_NEEDED addresses EVAL_INPUT and includes required elements
- `info_correct`: True if REVIEW_NEEDED follows all rules from the ResearchStrategize template
- `eval_text`: Required if either boolean is False; be specific and concise

## Critical Priority: Learn from PAST_REVIEW_OUTCOMES
**MANDATORY**: If PAST_REVIEW_OUTCOMES flags issues that still exist in REVIEW_NEEDED, mark as incorrect.
- Extract all issues from past `eval_text` fields
- Check if REVIEW_NEEDED repeats the same mistakes
- If past reviews flag a missing element and it's still missing → mark `info_correct: False`

## Rules

### Process Requirements
1. **Find the Goal**: Must pinpoint the user's primary financial aim
2. **List Key Facts**: Must take income, expenses, savings, capacity from "**Input Information from previous skill**", and research on market data necessary to answer the "**User request**"
3. **Analyze Savings Opportunities**: If the user's goal requires increased savings, MUST analyze all spending data from "Key Facts" and provide a prioritized, actionable list of recommendations for reducing spending or increasing income. These recommendations MUST be integrated directly into the strategy.
4. **Create a Strategy**: 
   - Design a complete, self-contained strategy of **no more than 3 steps**
   - **No open-ended tasks for the user**
   - Must provide the concrete insights an expert would
   - Must specify exact financial vehicles and researched targets
   - User should have liquidity (an emergency fund) at any point of time

### Output Format Requirements (~120 words)
1. **Summary**: A simple 1-2 sentence summary of the plan
2. **Key Facts**: 
   - Must include main result, monthly/timeline, feasibility/growth
   - Each fact must have format: `[Main result] (Input: "[quote relevant part]")`
   - Must quote relevant parts from Input Information
3. **Strategy**: 
   - Must have exactly 3 steps (no more, no less)
   - Each step must include calculation and Input reference
   - Format: `1. [Step from strategy with calculation and Input reference]`
4. **Risks**: A single, brief bullet point on a potential risk or consideration

### Content Quality Requirements
- **Concise**: Should be around 120 words total
- **Well-rationalized**: Strategy must be logical and well-reasoned
- **Easy-to-follow**: Clear structure and language
- **Concrete**: No vague or open-ended instructions
- **Expert-level**: Must provide specific financial vehicles and researched targets

## Verification Steps

1. **Check PAST_REVIEW_OUTCOMES first**: Extract all flagged issues. If REVIEW_NEEDED repeats them → mark False
2. **Verify good_copy**: Does REVIEW_NEEDED address EVAL_INPUT? Includes Summary, Key Facts, Strategy (3 steps), and Risks? Missing elements = False
3. **Verify info_correct**: Apply all rules:
   - Does it follow the Process Requirements (Goal, Key Facts, Savings Opportunities if needed, Strategy)?
   - Does it have exactly 3 strategy steps with calculations and Input references?
   - Are there any open-ended tasks for the user? (Should be False)
   - Does it specify exact financial vehicles and researched targets?
   - Does it address liquidity/emergency fund?
   - Does Key Facts format include Input quotes?
   - Is it concise (~120 words)?
   - Is it well-rationalized and easy-to-follow?
4. **Write eval_text**: If False, list specific issues. Reference unfixed PAST_REVIEW_OUTCOMES issues.
"""

class CheckResearchStrategize:
  """Handles all Gemini API interactions for checking ResearchStrategize outputs against rules"""
  
  def __init__(self, model_name="gemini-flash-lite-latest"):
    """Initialize the Gemini agent with API configuration for checking ResearchStrategize evaluations"""
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
    self.max_output_tokens = 6000
    
    # Safety Settings
    self.safety_settings = [
      types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"),
      types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"),
      types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"),
      types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="OFF")
    ]
    
    # System Prompt
    self.system_prompt = SYSTEM_PROMPT

  
  def generate_response(self, eval_input: str, past_review_outcomes: list, review_needed: str) -> dict:
    """
    Generate a response using Gemini API for checking P:Func:ResearchStrategize outputs.
    
    Args:
      eval_input: String containing "**User request**:" followed by the user's financial goal/request, and "**Input Information from previous skill**:" followed by detailed financial data (savings balance, accounts, past transactions, forecasted patterns, savings rate).
      past_review_outcomes: An array of past review outcomes, each containing `output`, `good_copy`, `info_correct`, and `eval_text`.
      review_needed: The P:Func:ResearchStrategize output that needs to be reviewed (string).
      
    Returns:
      Dictionary with good_copy, info_correct, and eval_text keys
    """
    # Create request text with the new input structure
    request_text_str = f"""<EVAL_INPUT>

{eval_input}

</EVAL_INPUT>

<PAST_REVIEW_OUTCOMES>

{json.dumps(past_review_outcomes, indent=2)}

</PAST_REVIEW_OUTCOMES>

<REVIEW_NEEDED>

{review_needed}

</REVIEW_NEEDED>

Output:"""
    
    print(request_text_str)
    print(f"\n{'='*80}\n")
    
    request_text = types.Part.from_text(text=request_text_str)
    
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
    
    # Check if response is empty
    if not output_text or not output_text.strip():
      raise ValueError(f"Empty response from model. Check API key and model availability.")
    
    # Parse JSON response
    try:
      # Remove markdown code blocks if present
      if "```json" in output_text:
        json_start = output_text.find("```json") + 7
        json_end = output_text.find("```", json_start)
        if json_end != -1:
          output_text = output_text[json_start:json_end].strip()
      elif "```" in output_text:
        # Try to find JSON in code blocks
        json_start = output_text.find("```") + 3
        json_end = output_text.find("```", json_start)
        if json_end != -1:
          output_text = output_text[json_start:json_end].strip()
      
      # Extract JSON object from the response
      json_start = output_text.find('{')
      json_end = output_text.rfind('}') + 1
      
      if json_start != -1 and json_end > json_start:
        json_str = output_text[json_start:json_end]
        return json.loads(json_str)
      else:
        # Try parsing the whole response
        return json.loads(output_text.strip())
    except json.JSONDecodeError as e:
      raise ValueError(f"Failed to parse JSON response: {e}\nResponse length: {len(output_text)}\nResponse preview: {output_text[:500]}")


def test_with_inputs(eval_input: str, past_review_outcomes: list, review_needed: str, checker: CheckResearchStrategize = None):
  """
  Convenient method to test the checker with custom inputs.
  
  Args:
    eval_input: String containing "**User request**:" followed by the user's financial goal/request, and "**Input Information from previous skill**:" followed by detailed financial data.
    past_review_outcomes: An array of past review outcomes, each containing `output`, `good_copy`, `info_correct`, and `eval_text`.
    review_needed: The P:Func:ResearchStrategize output that needs to be reviewed (string).
    checker: Optional CheckResearchStrategize instance. If None, creates a new one.
    
  Returns:
    Dictionary with good_copy, info_correct, and eval_text keys
  """
  if checker is None:
    checker = CheckResearchStrategize()
  
  return checker.generate_response(eval_input, past_review_outcomes, review_needed)


# Test cases covering different scenarios
TEST_CASES = [
  {
    "name": "correct_response",
    "eval_input": """**User request**: Based on the current savings balance and net monthly savings rate provided, calculate the projected time (in years and months) required to reach a savings goal of $100,000.

**Input Information from previous skill**:

Total Current Savings Balance: $24502

Savings Accounts:

Account 'Chase Savings' (account_id: 6854) has $24502.

--- Past Transactions ---

Total Past Income: earned $92194

Total Past Spending: spent $90195

--- Forecasted Patterns ---

Total Expected Income: earn $250000

Forecasted Income:

- income_salary on 2025-11-01: $20000

- income_salary on 2025-12-01: $20000

Total Expected Spending: spend $212555

Forecasted Spending:

- meals on 2025-11-01: $1861

- meals on 2025-12-01: $2170

--- Savings Rate ---

Past Savings Rate (Income - Spending): 14.98%""",
    "review_needed": """**Summary:** You'll reach $100,000 in approximately 3 years and 2 months by maintaining your current savings rate of 14.98% and applying it to your forecasted income.

**Key Facts:**
- Current savings balance of $24,502 (Input: "Total Current Savings Balance: $24502")
- Past savings rate of 14.98% (Input: "Past Savings Rate (Income - Spending): 14.98%")
- Need additional $75,498 to reach goal (Input: "savings goal of $100,000" and "Total Current Savings Balance: $24502")

**Strategy:**

1. Calculate monthly savings from forecasted income: $20,000/month income with 14.98% savings rate yields $2,996/month savings (Input: "income_salary on 2025-11-01: $20000" and "Past Savings Rate (Income - Spending): 14.98%")
2. Project timeline: $75,498 needed ÷ $2,996/month = 25.2 months (approximately 2 years 1 month) to reach goal (Input: "Need additional $75,498" and "$2,996/month savings")
3. Maintain emergency fund of $10,000 in Chase Savings account, invest excess beyond goal in index funds (Input: "Account 'Chase Savings' (account_id: 6854) has $24502")

**Risks:** Forecasted spending patterns may change, affecting actual savings rate.""",
    "past_review_outcomes": []
  },
  {
    "name": "missing_key_facts_input_quotes",
    "eval_input": """**User request**: Help me build an emergency fund

**Input Information from previous skill**:

Total Current Savings Balance: $2000

Savings Accounts:

Account 'Savings' (account_id: 1234) has $2000.

--- Past Transactions ---

Total Past Income: earned $60000

Total Past Spending: spent $50400

--- Forecasted Patterns ---

Total Expected Income: earn $60000

Total Expected Spending: spend $50400

--- Savings Rate ---

Past Savings Rate (Income - Spending): 16.00%""",
    "review_needed": """**Summary:** Build a 6-month emergency fund by saving $800/month.

**Key Facts:**
- Monthly savings capacity of $800
- Current savings of $2,000
- Target emergency fund of $25,200

**Strategy:**

1. Save $800/month for 29 months to reach $25,200 emergency fund
2. Keep funds in high-yield savings account
3. Automate transfers on payday

**Risks:** Unexpected expenses could derail savings plan.""",
    "past_review_outcomes": []
  },
  {
    "name": "missing_strategy_calculations",
    "eval_input": """**User request**: I want to retire early

**Input Information from previous skill**:

Total Current Savings Balance: $100000

--- Past Transactions ---

Total Past Income: earned $96000

Total Past Spending: spent $60000

--- Forecasted Patterns ---

Total Expected Income: earn $96000

Total Expected Spending: spend $60000

--- Savings Rate ---

Past Savings Rate (Income - Spending): 37.50%""",
    "review_needed": """**Summary:** Achieve early retirement by maximizing savings and investing aggressively.

**Key Facts:**
- Current savings of $100,000 (Input: "Current retirement savings: $100,000")
- Monthly savings capacity of $3,000 (Input: "Monthly income: $8,000" and "Monthly expenses: $5,000")
- 30 years until traditional retirement age (Input: "Age: 35")

**Strategy:**

1. Invest in diversified index funds
2. Maximize 401(k) contributions
3. Consider real estate investments

**Risks:** Market downturns could delay retirement timeline.""",
    "past_review_outcomes": []
  },
  {
    "name": "too_many_strategy_steps",
    "eval_input": """**User request**: Pay off credit card debt

**Input Information from previous skill**:

Total Current Savings Balance: $5000

--- Past Transactions ---

Total Past Income: earned $48000

Total Past Spending: spent $38400

--- Forecasted Patterns ---

Total Expected Income: earn $48000

Total Expected Spending: spend $38400

--- Savings Rate ---

Past Savings Rate (Income - Spending): 20.00%""",
    "review_needed": """**Summary:** Eliminate $15,000 credit card debt in 12 months using debt avalanche method.

**Key Facts:**
- Total debt of $15,000 at 18% APR (Input: "Credit card debt: $15,000 at 18% APR")
- Monthly surplus of $800 (Input: "Monthly income: $4,000" and "Monthly expenses: $3,200")
- High interest rate costing $225/month (Input: "18% APR")

**Strategy:**

1. Pay $1,250/month toward highest interest card, eliminating debt in 12 months (Input: "$15,000" and "$800 monthly surplus")
2. Transfer $5,000 to 0% balance transfer card to reduce interest costs (Input: "18% APR")
3. Cut $200/month from expenses to accelerate payoff (Input: "Monthly expenses: $3,200")
4. Build $1,000 emergency fund to avoid new debt

**Risks:** Balance transfer fees and potential for accumulating new debt.""",
    "past_review_outcomes": []
  },
  {
    "name": "open_ended_tasks",
    "eval_input": """**User request**: Invest for my child's college education

**Input Information from previous skill**:

Total Current Savings Balance: $5000

--- Past Transactions ---

Total Past Income: earned $60000

Total Past Spending: spent $54000

--- Forecasted Patterns ---

Total Expected Income: earn $60000

Total Expected Spending: spend $54000

--- Savings Rate ---

Past Savings Rate (Income - Spending): 10.00%""",
    "review_needed": """**Summary:** Build $100,000 college fund over 13 years through 529 plan contributions.

**Key Facts:**
- 13 years until college (Input: "Child age: 5")
- Target of $100,000 (Input: "Target: $100,000 in 13 years")
- Current savings of $5,000 (Input: "Current savings: $5,000")

**Strategy:**

1. Open 529 plan and contribute $500/month, researching best state plans
2. Invest in age-based target date funds, selecting appropriate risk level
3. Review and adjust contributions annually based on market performance

**Risks:** Market volatility could affect final balance.""",
    "past_review_outcomes": []
  },
  {
    "name": "missing_emergency_fund_consideration",
    "eval_input": """**User request**: Maximize my investment returns

**Input Information from previous skill**:

Total Current Savings Balance: $53000

Savings Accounts:

Account 'Investment Account' (account_id: 5678) has $50000.

Account 'Savings' (account_id: 5679) has $3000.

--- Past Transactions ---

Total Past Income: earned $84000

Total Past Spending: spent $54000

--- Forecasted Patterns ---

Total Expected Income: earn $84000

Total Expected Spending: spend $54000

--- Savings Rate ---

Past Savings Rate (Income - Spending): 35.71%""",
    "review_needed": """**Summary:** Optimize investment portfolio by rebalancing and increasing contributions.

**Key Facts:**
- Current investments of $50,000 (Input: "Current investments: $50,000")
- Monthly surplus of $2,500 (Input: "Monthly income: $7,000" and "Monthly expenses: $4,500")
- Low current savings buffer (Input: "Savings: $3,000")

**Strategy:**

1. Invest entire $2,500/month surplus in S&P 500 index fund for maximum growth
2. Rebalance existing $50,000 portfolio to 100% stocks for highest returns
3. Use all $3,000 savings to buy additional index fund shares

**Risks:** Market downturns could result in significant losses.""",
    "past_review_outcomes": []
  },
  {
    "name": "missing_savings_opportunities_analysis",
    "eval_input": """**User request**: Save $30,000 for a car in 2 years

**Input Information from previous skill**:

Total Current Savings Balance: $2000

--- Past Transactions ---

Total Past Income: earned $66000

Total Past Spending: spent $62400

Forecasted Spending:

- meals on 2025-11-01: $600

- entertainment on 2025-11-01: $400

- subscriptions on 2025-11-01: $150

--- Forecasted Patterns ---

Total Expected Income: earn $66000

Total Expected Spending: spend $62400

--- Savings Rate ---

Past Savings Rate (Income - Spending): 5.45%""",
    "review_needed": """**Summary:** Save $30,000 in 24 months by maintaining current $300/month savings rate.

**Key Facts:**
- Current monthly savings of $300 (Input: "Monthly income: $5,500" and "Monthly expenses: $5,200")
- 24 months to reach goal (Input: "2 years")
- Need additional $28,000 (Input: "Target: $30,000" and "Current savings: $2,000")

**Strategy:**

1. Save $300/month in high-yield savings account, reaching $7,200 in 24 months
2. Keep $2,000 emergency fund separate
3. Review expenses quarterly

**Risks:** Current savings rate insufficient to reach goal in timeframe.""",
    "past_review_outcomes": []
  },
  {
    "name": "correct_with_savings_opportunities",
    "eval_input": """**User request**: Save $20,000 for vacation in 18 months

**Input Information from previous skill**:

Total Current Savings Balance: $1000

--- Past Transactions ---

Total Past Income: earned $72000

Total Past Spending: spent $60000

Forecasted Spending:

- meals on 2025-11-01: $800

- shopping on 2025-11-01: $500

- subscriptions on 2025-11-01: $200

--- Forecasted Patterns ---

Total Expected Income: earn $72000

Total Expected Spending: spend $60000

--- Savings Rate ---

Past Savings Rate (Income - Spending): 16.67%""",
    "review_needed": """**Summary:** Save $20,000 in 18 months by reducing dining and shopping expenses, increasing savings to $1,200/month.

**Key Facts:**
- Current monthly savings of $1,000 (Input: "Monthly income: $6,000" and "Monthly expenses: $5,000")
- High dining expenses of $800/month (Input: "Dining $800/month")
- 18 months to reach goal (Input: "18 months")

**Strategy:**

1. Reduce dining out from $800 to $400/month and shopping from $500 to $200/month, increasing savings to $1,200/month, reaching $21,600 in 18 months (Input: "Dining $800/month" and "Shopping $500/month")
2. Keep $3,000 emergency fund in separate high-yield account (Input: "Current savings: $1,000")
3. Automate $1,200 transfer to vacation savings account on payday (Input: "Monthly income: $6,000")

**Risks:** Lifestyle changes may be difficult to maintain long-term.""",
    "past_review_outcomes": []
  },
  {
    "name": "past_review_outcomes_issue_persists",
    "eval_input": """**User request**: Build wealth through investing

**Input Information from previous skill**:

Total Current Savings Balance: $20000

--- Past Transactions ---

Total Past Income: earned $60000

Total Past Spending: spent $42000

--- Forecasted Patterns ---

Total Expected Income: earn $60000

Total Expected Spending: spend $42000

--- Savings Rate ---

Past Savings Rate (Income - Spending): 30.00%""",
    "review_needed": """**Summary:** Build wealth by investing $1,500/month in diversified index funds.

**Key Facts:**
- Current investments of $20,000 (Input: "Current investments: $20,000")
- Monthly surplus of $1,500 (Input: "Monthly income: $5,000" and "Monthly expenses: $3,500")
- Long investment horizon (Input: "Age: 28")

**Strategy:**

1. Invest $1,500/month in S&P 500 index fund
2. Keep existing $20,000 invested
3. Review portfolio annually

**Risks:** Market volatility could affect returns.""",
    "past_review_outcomes": [
      {
        "output": "Previous output with missing emergency fund",
        "good_copy": True,
        "info_correct": False,
        "eval_text": "Strategy does not address liquidity/emergency fund requirement. User should have emergency fund at any point."
      }
    ]
  }
]


def run_test(test_case: dict, checker: CheckResearchStrategize = None):
  """
  Run a single test case.
  
  Args:
    test_case: Test case dict with name, eval_input, review_needed, and optionally past_review_outcomes
    checker: Optional CheckResearchStrategize instance. If None, creates a new one.
    
  Returns:
    Dictionary with good_copy, info_correct, and eval_text keys, or None if error occurred
  """
  if checker is None:
    checker = CheckResearchStrategize()
  
  print(f"\n{'='*80}")
  print(f"Running test: {test_case['name']}")
  print(f"{'='*80}")
  
  try:
    eval_input = test_case["eval_input"]
    past_review_outcomes = test_case.get("past_review_outcomes", [])
    review_needed = test_case["review_needed"]
    
    result = checker.generate_response(eval_input, past_review_outcomes, review_needed)
    print(f"Result:")
    print(json.dumps(result, indent=2))
    print(f"{'='*80}")
    return result
  except Exception as e:
    print(f"ERROR: {type(e).__name__}: {str(e)}")
    print(f"{'='*80}")
    return None


def run_tests(test_names: list = None, checker: CheckResearchStrategize = None):
  """
  Run multiple test cases.
  
  Args:
    test_names: List of test case names to run. If None, runs all tests.
    checker: Optional CheckResearchStrategize instance. If None, creates a new one.
    
  Returns:
    List of results (None entries indicate failed tests)
  """
  if checker is None:
    checker = CheckResearchStrategize()
  
  if test_names is None:
    tests_to_run = TEST_CASES
  else:
    tests_to_run = [tc for tc in TEST_CASES if tc["name"] in test_names]
  
  results = []
  passed = 0
  failed = 0
  
  for test_case in tests_to_run:
    result = run_test(test_case, checker)
    results.append(result)
    if result is None:
      failed += 1
    else:
      passed += 1
  
  print(f"\n{'='*80}")
  print(f"Test Summary: {passed} passed, {failed} failed out of {len(tests_to_run)} tests")
  print(f"{'='*80}")
  
  return results


def main():
  """Main function to test the ResearchStrategize checker"""
  checker = CheckResearchStrategize()
  
  # Run all tests
  run_tests(checker=checker)
  
  # Or run specific tests:
  # run_tests(["correct_response", "missing_key_facts_input_quotes"], checker=checker)


if __name__ == "__main__":
  main()

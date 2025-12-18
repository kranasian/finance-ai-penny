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
- `info_correct`: True if REVIEW_NEEDED follows all rules from the VerbalizerTextWithMemory template
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

class CheckVerbalizerTextWithMemory:
  """Handles all Gemini API interactions for checking VerbalizerTextWithMemory outputs against rules"""
  
  def __init__(self, model_name="gemini-flash-lite-latest"):
    """Initialize the Gemini agent with API configuration for checking VerbalizerTextWithMemory evaluations"""
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
    Generate a response using Gemini API for checking P:Func:VerbalizerTextWithMemory outputs.
    
    Args:
      eval_input: String containing "**User request**:" followed by the user's financial goal/request, and "**Input Information from previous skill**:" followed by detailed financial data (savings balance, accounts, past transactions, forecasted patterns, savings rate).
      past_review_outcomes: An array of past review outcomes, each containing `output`, `good_copy`, `info_correct`, and `eval_text`.
      review_needed: The P:Func:VerbalizerTextWithMemory output that needs to be reviewed (string).
      
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


def run_test_case(test_name: str, eval_input: str, review_needed: str, past_review_outcomes: list = None, checker: 'CheckVerbalizerTextWithMemory' = None):
  """
  Utility function to run a test case with custom inputs, common error handling, and output formatting.
  
  Args:
    test_name: Name of the test case
    eval_input: String containing "**User request**:" followed by the user's financial goal/request, and "**Input Information from previous skill**:" followed by detailed financial data.
    review_needed: The P:Func:VerbalizerTextWithMemory output that needs to be reviewed (string).
    past_review_outcomes: An array of past review outcomes, each containing `output`, `good_copy`, `info_correct`, and `eval_text`. Defaults to empty list.
    checker: Optional CheckVerbalizerTextWithMemory instance. If None, creates a new one.
    
  Returns:
    Dictionary with good_copy, info_correct, and eval_text keys, or None if error occurred
  """
  if past_review_outcomes is None:
    past_review_outcomes = []
  if checker is None:
    checker = CheckVerbalizerTextWithMemory()

  print(f"\n{'='*80}")
  print(f"Running test: {test_name}")
  print(f"{'='*80}")

  try:
    # Directly call the checker's response with the provided inputs.
    result = checker.generate_response(eval_input, past_review_outcomes, review_needed)
    print(f"Result:")
    print(json.dumps(result, indent=2))
    print(f"{'='*80}")
    return result
  except Exception as e:
    print(f"ERROR: {type(e).__name__}: {str(e)}")
    print(f"{'='*80}")
    return None


def run_correct_response(checker: CheckVerbalizerTextWithMemory = None):
  """
  Run the test case for correct_response.
  """
  eval_input = """**User request**: Based on the current savings balance and net monthly savings rate provided, calculate the projected time (in years and months) required to reach a savings goal of $100,000.
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
Past Savings Rate (Income - Spending): 14.98%"""
  
  review_needed = """**Summary:** You'll reach $100,000 in approximately 3 years and 2 months by maintaining your current savings rate of 14.98% and applying it to your forecasted income.

**Key Facts:**
- Current savings balance of $24,502 (Input: "Total Current Savings Balance: $24502")
- Past savings rate of 14.98% (Input: "Past Savings Rate (Income - Spending): 14.98%")
- Need additional $75,498 to reach goal (Input: "savings goal of $100,000" and "Total Current Savings Balance: $24502")

**Strategy:**

1. Calculate monthly savings from forecasted income: $20,000/month income with 14.98% savings rate yields $2,996/month savings (Input: "income_salary on 2025-11-01: $20000" and "Past Savings Rate (Income - Spending): 14.98%")
2. Project timeline: $75,498 needed ÷ $2,996/month = 25.2 months (approximately 2 years 1 month) to reach goal (Input: "Need additional $75,498" and "$2,996/month savings")
3. Maintain emergency fund of $10,000 in Chase Savings account, invest excess beyond goal in index funds (Input: "Account 'Chase Savings' (account_id: 6854) has $24502")

**Risks:** Forecasted spending patterns may change, affecting actual savings rate."""
  
  return run_test_case("correct_response", eval_input, review_needed, [], checker)



def main():
  """Main function to test the VerbalizerTextWithMemory checker"""
  checker = CheckVerbalizerTextWithMemory()
  
  # Run all tests
  run_correct_response(checker)


if __name__ == "__main__":
  main()

from google import genai
from google.genai import types
import os
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

SYSTEM_PROMPT = """You are an exacting and meticulous checker of financial strategy outputs. Your sole purpose is to validate a given text (`REVIEW_NEEDED`) against a set of strict rules and provided data (`EVAL_INPUT`). You must be precise and unforgiving in your evaluation.

## Input:
- **EVAL_INPUT**: The user's request and all relevant financial data. This is the single source of truth.
- **PAST_REVIEW_OUTCOMES**: A history of previous evaluations.
- **REVIEW_NEEDED**: The financial strategy text to be validated.

## Output:
A single, clean JSON object: `{"good_copy": boolean, "info_correct": boolean, "eval_text": string}`
- `good_copy`: `true` only if all formatting and structural rules are met.
- `info_correct`: `true` only if all information is perfectly accurate and logically consistent.
- `eval_text`: If any check fails, provide a concise but comprehensive explanation of all errors as a bulleted list of phrases, ensuring newlines are escaped (e.g., '- Error 1\\n- Error 2'). If both `good_copy` and `info_correct` are true, `eval_text` must be an empty string.

## Core Directives

### 1. Zero Tolerance for Past Mistakes
**MANDATORY**: Before anything else, check `PAST_REVIEW_OUTCOMES`. If `REVIEW_NEEDED` repeats a mistake from a past `eval_text`, it is an automatic and immediate failure. Mention the repeated error in your `eval_text`.

### 2. Information Correctness (`info_correct`) Verification
- **Internal Consistency**: All numbers, timelines, and statements within `REVIEW_NEEDED` must be perfectly consistent. A mismatch between the `Summary` and the `Strategy` calculation is a critical failure.
- **External Consistency**: Every single claim, number, and calculation must be directly traceable to `EVAL_INPUT`.
- **No External Information or Advice**: The strategy must not introduce any concepts, numbers (like an arbitrary emergency fund amount), or financial advice (like investing in index funds) that were not explicitly part of the `EVAL_INPUT`. The response must only use the data provided.
- **Logical and Complete**: The strategy must be a sound financial plan that directly and fully answers the user's request in `EVAL_INPUT`.
- **No Account IDs**: Account IDs should never be mentioned in `REVIEW_NEEDED`.
- **No Transaction IDs**: Transaction IDs should never be mentioned in `REVIEW_NEEDED`.
- **No Underscores in Categories**: Category names in `REVIEW_NEEDED` must not contain underscores.

### 3. Copy Quality (`good_copy`) Verification
- **Absolutely No Markdown**: `REVIEW_NEEDED` must be 100% plain text. The presence of any markdown syntax (e.g., `**`, `##`, `*`, `- ` for lists) is an immediate failure.
- **Conversational Tone & Tone for Inability/Mistakes**: Avoid explicit greetings (e.g., "Hi", "Hey", "Good morning"). Other conversational openers are permitted. When communicating inability to fulfill a user's request or acknowledging mistakes in shared information, the tone MUST be apologetic AND actionable (e.g., provide other options to help, commit to share feedback for improvement of app). Simple refusals or dry apologies without next steps are immediate failures.
- **Emojis**: Emojis are allowed, but they must not be in unicode format.

## Verification and Output Generation Workflow

1.  **Initial Check**: Review `PAST_REVIEW_OUTCOMES` for repeated errors.
2.  **Full Audit**: Perform a full audit based on all `info_correct` and `good_copy` rules.
3.  **Construct `eval_text`**: If any rule is violated, construct a clear and exhaustive `eval_text` that precisely lists ALL detected faults as a bulleted list of phrases, ensuring newlines are escaped.
4.  **Final Review**: Before outputting, review your own analysis. Is the JSON valid? Is the `eval_text` clear, accurate, and covering all detected faults? Then, and only then, provide the final JSON output.

---
**Internal Thought Rule**:
- ONLY analyze `EVAL_INPUT`, `PAST_REVIEW_OUTCOMES`, and `REVIEW_NEEDED`.
- STRICTLY adhere to `Core Directives` for evaluation.
- DO NOT generate new content or financial advice.
- Focus exclusively on identifying and explaining errors concisely.
- Be extremely brief.
---
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
      thinking_config=types.ThinkingConfig(
        thinking_budget=self.thinking_budget,
        include_thoughts=True
      ),
    )

    # Generate response
    output_text = ""
    thought_summary = ""
    
    # According to Gemini API docs: iterate through chunks and check part.thought boolean
    print("Starting generation stream...")
    for chunk in self.client.models.generate_content_stream(
      model=self.model_name,
      contents=contents,
      config=generate_content_config,
    ):
      # print(".", end="", flush=True) # Debug progress
      # Extract text content (non-thought parts)
      if chunk.text is not None:
        output_text += chunk.text
      
      # Extract thought summary from chunk
      if hasattr(chunk, 'candidates') and chunk.candidates:
        for candidate in chunk.candidates:
          # Extract thought summary from parts (per Gemini API docs)
          # Check part.thought boolean to identify thought parts
          if hasattr(candidate, 'content') and candidate.content:
            if hasattr(candidate.content, 'parts') and candidate.content.parts:
              for part in candidate.content.parts:
                # Check if this part is a thought summary (per documentation)
                if hasattr(part, 'thought') and part.thought:
                  if hasattr(part, 'text') and part.text:
                    # Accumulate thought summary text (for streaming, it may come in chunks)
                    if thought_summary:
                      thought_summary += part.text
                    else:
                      thought_summary = part.text
    
    # Check if response is empty
    if not output_text or not output_text.strip():
      raise ValueError(f"Empty response from model. Check API key and model availability.")
    
    if thought_summary:
      print(f"{'='*80}")
      print("THOUGHT SUMMARY:")
      print(thought_summary.strip())
      print("="*80)
    
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


  eval_input = """**User request**: Create a budget for my monthly groceries, currently I spend around $500.
**Input Information from previous skill**:
Monthly Grocery Spending: $500
Average Income: $3000"""
  review_needed = """Here's a breakdown for your grocery budget:
- Allocate $450 for groceries this month. 🛒
- This is a slight reduction to help you save more. 💰
- Enjoy your delicious meals! 😋"""
  return run_test_case("Batch 1 - Test 1: All Correct", eval_input, review_needed, [], checker)

def run_test_batch(batch_number: int, checker: CheckVerbalizerTextWithMemory = None):
  """
  Run a specific batch of test cases.
  """
  if checker is None:
    checker = CheckVerbalizerTextWithMemory()

  print(f"\n{'='*80}")
  print(f"Running Batch {batch_number}")
  print(f"{'='*80}")

  if batch_number == 1:
    # Test 1: Extra Advice (Not in Input) -> Should Fail info_correct
    eval_input_1 = """**User request**: Create a budget for my monthly groceries.
**Input Information from previous skill**:
Monthly Grocery Spending: $500
Average Income: $3000"""
    review_needed_1 = """Based on your spending, I've set a grocery budget of $450. You should also consider opening a high-yield savings account for your emergency fund. 💰"""
    run_test_case(f"Batch {batch_number} - Test 1: Extra Advice", eval_input_1, review_needed_1, [], checker)

    # Test 2: Markdown List -> Should Fail good_copy
    eval_input_2 = """**User request**: List my recent transactions.
**Input Information from previous skill**:
- Uber: $15
- Kroger: $50"""
    review_needed_2 = """Here are your recent transactions:
- Uber: $15 🚗
- Kroger: $50 🛒"""
    run_test_case(f"Batch {batch_number} - Test 2: Markdown List", eval_input_2, review_needed_2, [], checker)

    # Test 3: Inability (Bad Tone) -> Should Fail good_copy
    eval_input_3 = """**User request**: Predict my credit score for next year.
**Input Information from previous skill**:
(No capability to predict credit score)"""
    review_needed_3 = """I cannot predict your credit score for next year. 🛑"""
    run_test_case(f"Batch {batch_number} - Test 3: Inability (Bad Tone)", eval_input_3, review_needed_3, [], checker)

  elif batch_number == 2:
    # Test 4: Inability (Good Tone) -> Should Pass
    eval_input_4 = """**User request**: Predict my credit score for next year.
**Input Information from previous skill**:
(No capability to predict credit score)"""
    review_needed_4 = """I apologize, but I can't predict your future credit score. However, I can help you track your current score or set up payment reminders to help improve it! I'll also share this request with the team. 🙏📈"""
    run_test_case(f"Batch {batch_number} - Test 4: Inability (Good Tone)", eval_input_4, review_needed_4, [], checker)

    # Test 5: Unicode Emoji -> Should Fail good_copy
    eval_input_5 = """**User request**: How much did I save?
**Input Information from previous skill**:
Savings: $100"""
    review_needed_5 = """You saved $100! &#127881;"""
    run_test_case(f"Batch {batch_number} - Test 5: Unicode Emoji", eval_input_5, review_needed_5, [], checker)

    # Test 6: Plain List (Valid) -> Should Pass
    eval_input_6 = """**User request**: List my recent transactions.
**Input Information from previous skill**:
- Uber: $15
- Kroger: $50"""
    review_needed_6 = """Here are your recent transactions:
🚗 Uber: $15
🛒 Kroger: $50"""
    run_test_case(f"Batch {batch_number} - Test 6: Plain List (Valid)", eval_input_6, review_needed_6, [], checker)

  elif batch_number == 3:
    # Test 7: Extra Info (Subtle) -> Should Fail info_correct
    eval_input_7 = """**User request**: How much did I spend on coffee?
**Input Information from previous skill**:
Coffee Spending: $50"""
    review_needed_7 = """You spent $50 on coffee. That's about 10% of your discretionary income. ☕"""
    run_test_case(f"Batch {batch_number} - Test 7: Extra Info (Subtle)", eval_input_7, review_needed_7, [], checker)

    # Test 8: Inability (Apologetic but not Actionable) -> Should Fail good_copy
    eval_input_8 = """**User request**: Buy me some stocks.
**Input Information from previous skill**:
(No trading capability)"""
    review_needed_8 = """I'm really sorry, but I can't buy stocks for you. 😔"""
    run_test_case(f"Batch {batch_number} - Test 8: Inability (Not Actionable)", eval_input_8, review_needed_8, [], checker)

    # Test 9: Inability (Actionable but not Apologetic) -> Should Fail good_copy
    eval_input_9 = """**User request**: Buy me some stocks.
**Input Information from previous skill**:
(No trading capability)"""
    review_needed_9 = """I can't do that. Instead, I can show you your current portfolio performance. I'll note this for the team. 📉"""
    run_test_case(f"Batch {batch_number} - Test 9: Inability (Not Apologetic)", eval_input_9, review_needed_9, [], checker)

  elif batch_number == 4:
    # Test 10: Complex Valid -> Should Pass
    eval_input_10 = """**User request**: What are my top expenses?
**Input Information from previous skill**:
1. Rent: $1200
2. Food: $400"""
    review_needed_10 = """Your top expenses are:
🏠 Rent: $1200
🍔 Food: $400"""
    run_test_case(f"Batch {batch_number} - Test 10: Complex Valid", eval_input_10, review_needed_10, [], checker)

    # Test 11: Repeated Error -> Should Fail (Repeated)
    past_outcome_11 = [{"output": "...", "good_copy": False, "info_correct": True, "eval_text": "- markdown present"}]
    eval_input_11 = """**User request**: Status?
**Input Information from previous skill**:
Status: Good"""
    review_needed_11 = """Status is **Good**! 👍"""
    run_test_case(f"Batch {batch_number} - Test 11: Repeated Error", eval_input_11, review_needed_11, past_outcome_11, checker)

    # Test 12: Internal Info -> Should Fail info_correct
    eval_input_12 = """**User request**: Transaction details?
**Input Information from previous skill**:
ID: 999
Amount: $10"""
    review_needed_12 = """Transaction 999 was for $10. 💸"""
    run_test_case(f"Batch {batch_number} - Test 12: Internal Info", eval_input_12, review_needed_12, [], checker)




def main():
  """Main function to test the VerbalizerTextWithMemory checker"""
  checker = CheckVerbalizerTextWithMemory()
  
  for batch_num in range(1, 5):
    run_test_batch(batch_num, checker)


if __name__ == "__main__":
  main()

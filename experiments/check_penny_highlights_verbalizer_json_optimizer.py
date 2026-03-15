from google import genai
from google.genai import types
import os
import json
from dotenv import load_dotenv
import sys

# Import the verbalizer to generate test outputs
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from penny_highlights_verbalizer_json_optimizer import PennyHighlightsVerbalizerOptimizer

# Load environment variables
load_dotenv()

SYSTEM_PROMPT = """You are a rigorous financial copy auditor.
Audit `REVIEW_NEEDED` (copy) against `EVAL_INPUT` (facts).

## Audit Flow:
1. **Fact Audit (info_correct)**:
   - **Check IDs**: Does the ID of the Nth insight in `EVAL_INPUT` match the ID of the Nth summary in `REVIEW_NEEDED`?
   - **Check Numbers**: Are all numbers in `REVIEW_NEEDED` (titles and summaries) identical to those in `EVAL_INPUT`? (Note: not all information in `REVIEW_NEEDED` have to be present in `EVAL_INPUT`.)
   - **Result**: `info_correct: true` ONLY if both above are 100% accurate.

2. **Style Audit (good_copy)**:
   - **Directions**: EVERY monetary value ($) MUST be accompanied by a direction relative to expectation (explicit or implied). See examples below.
     - **FAIL**: "Medical is $50 this week." or similar
     - **PASS (Explicit)**: "Medical is up at $50 this week." or similar
     - **PASS (Implied)**: "Oh no! Medical is $50 this week." or similar (The "Oh no!" implies a negative state/increase)
   - **No Greetings**: MUST NOT start with conversation openers such as "Hi", "Hello", "Hey", etc. Expressions are acceptable.
   - **Tone**: MUST be friendly and encouraging. NEVER condescending.
   - **Titles**: The title MUST encapsulate ALL discussion items shared in the summary.
     - **PASS**: "Increased Spending!", "On your Meals and Health".
     - **FAIL**: "Meals are Up", "Rising Health" (These are too specific/narrow if the summary covers multiple items).
   - **Result**: `good_copy: true` ONLY if all above are 100% compliant.

## Evaluation Rules:
- **Feedback Format**: If any boolean is false, provide `eval_text`: "Insight [N]: [concise_issue]".
  - [N] is the 1-based order in the list. Concatenate multiple issues with "; ".
  - **CRITICAL**: `eval_text` must be concise and self-explanatory. The reader should understand the error without referring to the audit rules.

## Output Format:
STRICT SINGLE JSON OBJECT: `{"good_copy": boolean, "info_correct": boolean, "eval_text": string}`
Do NOT return an array.
"""

class CheckHighlightsVerbalizerJsonOptimizer2:
  """Handles all Gemini API interactions for checking PennyHighlightsVerbalizerJsonOptimizer outputs against rules"""
  
  def __init__(self, model_name="gemini-flash-lite-latest"):
    """Initialize the Gemini agent with API configuration for checking PennyHighlightsVerbalizerJsonOptimizer evaluations"""
    # API Configuration
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
      raise ValueError("GEMINI_API_KEY environment variable is not set. Please set it in your .env file or environment.")
    self.client = genai.Client(api_key=api_key)
    
    # Model Configuration
    self.thinking_budget = 2048
    self.model_name = model_name
    
    # Generation Configuration Constants
    self.temperature = 0.6
    self.top_p = 0.95
    self.top_k = 40
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

  
  def generate_response(self, eval_input: list, past_review_outcomes: list, review_needed: list) -> dict:
    """
    Generate a response using Gemini API for checking PennyHighlightsVerbalizerJsonOptimizer outputs.
    
    Args:
      eval_input: List of insight dictionaries, where each dictionary contains an "id" and "combined_insight".
      past_review_outcomes: An array of past review outcomes, each containing `output`, `good_copy`, `info_correct`, and `eval_text`.
      review_needed: The PennyHighlightsVerbalizerJsonOptimizer output that needs to be reviewed (JSON array of objects with id, title, summary).
      
    Returns:
      Dictionary with good_copy, info_correct, and eval_text keys
    """
    # Create request text with the new input structure
    request_text_str = f"""<EVAL_INPUT>

{json.dumps(eval_input, indent=2)}

</EVAL_INPUT>

<PAST_REVIEW_OUTCOMES>

{json.dumps(past_review_outcomes, indent=2)}

</PAST_REVIEW_OUTCOMES>

<REVIEW_NEEDED>

{json.dumps(review_needed, indent=2)}

</REVIEW_NEEDED>

Output:"""
    
    print(f"\n{'='*80}")
    print("INPUT JSON:")
    print(request_text_str)
    print("="*80)
    
    request_text = types.Part.from_text(text=request_text_str)
    
    # Create content and configuration
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
        include_thoughts=True
      ),
    )

    # Generate response
    output_text = ""
    thought_summary = ""
    
    for chunk in self.client.models.generate_content_stream(
      model=self.model_name,
      contents=contents,
      config=generate_content_config,
    ):
      if chunk.text is not None:
        output_text += chunk.text
      
      # Extract thought summary from chunk
      if hasattr(chunk, 'candidates') and chunk.candidates:
        for candidate in chunk.candidates:
          if hasattr(candidate, 'content') and candidate.content:
            if hasattr(candidate.content, 'parts') and candidate.content.parts:
              for part in candidate.content.parts:
                if hasattr(part, 'thought') and part.thought:
                  if hasattr(part, 'text') and part.text:
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
    
    print(f"{'='*80}")
    print("RESPONSE OUTPUT:")
    print(output_text.strip())
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


def run_test_case(test_name: str, eval_input: list, review_needed: list, past_review_outcomes: list = None, checker: 'CheckHighlightsVerbalizerJsonOptimizer2' = None):
  """
  Utility function to run a test case with custom inputs, common error handling, and output formatting.
  
  Args:
    test_name: Name of the test case
    eval_input: List of insight dictionaries, where each dictionary contains an "id" and "combined_insight".
    review_needed: The PennyHighlightsVerbalizerJsonOptimizer output that needs to be reviewed (JSON array of objects with id, title, summary).
    past_review_outcomes: An array of past review outcomes, each containing `output`, `good_copy`, `info_correct`, and `eval_text`. Defaults to empty list.
    checker: Optional CheckHighlightsVerbalizerJsonOptimizer2 instance. If None, creates a new one.
    
  Returns:
    Dictionary with good_copy, info_correct, and eval_text keys, or None if error occurred
  """
  if past_review_outcomes is None:
    past_review_outcomes = []
  if checker is None:
    checker = CheckHighlightsVerbalizerJsonOptimizer2()

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


def test_batch_1(verbalizer: PennyHighlightsVerbalizerOptimizer = None, checker: CheckHighlightsVerbalizerJsonOptimizer2 = None):
  """
  Batch 1: Subscription and Savings
  """
  if verbalizer is None:
    verbalizer = PennyHighlightsVerbalizerOptimizer()
  if checker is None:
    checker = CheckHighlightsVerbalizerJsonOptimizer2()
  
  eval_input_1 = [
    {
      "id": 501,
      "combined_insight": "Your monthly gym subscription increased to $65 this month, which is $15 higher than last month. Consider if you're still using it! 🏋️‍♂️💸"
    },
    {
      "id": 502,
      "combined_insight": "You've successfully saved $2,500 for your emergency fund, hitting 50% of your $5,000 goal! Keep it up! 🛡️💰"
    }
  ]
  
  print("\n--- Generating verbalizer output for Test 1 ---")
  review_needed_1 = verbalizer.generate_response(eval_input_1)
  
  print("\n--- Checking Test 1 ---")
  result_1 = run_test_case("batch_1_test_1", eval_input_1, review_needed_1, [], checker)
  
  return [result_1]


def test_batch_2(verbalizer: PennyHighlightsVerbalizerOptimizer = None, checker: CheckHighlightsVerbalizerJsonOptimizer2 = None):
  """
  Batch 2: Food and Repairs
  """
  if verbalizer is None:
    verbalizer = PennyHighlightsVerbalizerOptimizer()
  if checker is None:
    checker = CheckHighlightsVerbalizerJsonOptimizer2()
  
  eval_input_2 = [
    {
      "id": 601,
      "combined_insight": "Great job! Your grocery spending was lower at $280 this week. However, you spent $150 on dining out, which is higher than usual. 🛒🍽️"
    },
    {
      "id": 602,
      "combined_insight": "Heads up! You had an unexpected car repair cost of $450. Your electricity bill was also higher at $180 this month. 🚗⚡"
    }
  ]
  
  print("\n--- Generating verbalizer output for Test 2 ---")
  review_needed_2 = verbalizer.generate_response(eval_input_2)
  
  print("\n--- Checking Test 2 ---")
  result_2 = run_test_case("batch_2_test_2", eval_input_2, review_needed_2, [], checker)
  
  return [result_2]


def test_batch_3(verbalizer: PennyHighlightsVerbalizerOptimizer = None, checker: CheckHighlightsVerbalizerJsonOptimizer2 = None):
  """
  Batch 3: Income and Budgeting
  """
  if verbalizer is None:
    verbalizer = PennyHighlightsVerbalizerOptimizer()
  if checker is None:
    checker = CheckHighlightsVerbalizerJsonOptimizer2()
  
  eval_input_3 = [
    {
      "id": 701,
      "combined_insight": "Awesome! Your side hustle brought in $1,200 extra this month. You also received a tax refund of $850! 🚀💸"
    },
    {
      "id": 702,
      "combined_insight": "You spent $300 on shopping, which is $100 over budget. On the plus side, your rent was lower at $1,100 this month. 🛍️🏠"
    }
  ]
  
  print("\n--- Generating verbalizer output for Test 3 ---")
  review_needed_3 = verbalizer.generate_response(eval_input_3)
  
  print("\n--- Checking Test 3 ---")
  result_3 = run_test_case("batch_3_test_3", eval_input_3, review_needed_3, [], checker)
  
  return [result_3]


def test_batch_4(verbalizer: PennyHighlightsVerbalizerOptimizer = None, checker: CheckHighlightsVerbalizerJsonOptimizer2 = None):
  """
  Batch 4: Investments and Medical
  """
  if verbalizer is None:
    verbalizer = PennyHighlightsVerbalizerOptimizer()
  if checker is None:
    checker = CheckHighlightsVerbalizerJsonOptimizer2()
  
  eval_input_4 = [
    {
      "id": 801,
      "combined_insight": "Your investment portfolio gained $500 this quarter! You also received $120 in dividends. 📈💰"
    },
    {
      "id": 802,
      "combined_insight": "You had a medical expense of $75. Your health insurance premium was slightly lower at $210 this month. 🏥🛡️"
    }
  ]
  
  print("\n--- Generating verbalizer output for Test 4 ---")
  review_needed_4 = verbalizer.generate_response(eval_input_4)
  
  print("\n--- Checking Test 4 ---")
  result_4 = run_test_case("batch_4_test_4", eval_input_4, review_needed_4, [], checker)
  
  return [result_4]


def test_batch_5(checker: CheckHighlightsVerbalizerJsonOptimizer2 = None):
  """
  Batch 5: Failure cases to test Checker's detection
  """
  if checker is None:
    checker = CheckHighlightsVerbalizerJsonOptimizer2()

  # Test Case 1: ID mismatch
  eval_input_1 = [{"id": 901, "combined_insight": "You saved $40 on coffee."}]
  review_needed_1 = [{"id": 1, "title": "Coffee Savings", "summary": "You saved $40 on coffee this week!"}]
  result_1 = run_test_case("ID mismatch", eval_input_1, review_needed_1, [], checker)

  # Test Case 2: Missing relative qualifier
  eval_input_2 = [{"id": 902, "combined_insight": "Gym is $50."}]
  review_needed_2 = [{"id": 902, "title": "Gym Cost", "summary": "Your gym cost is $50."}]
  result_2 = run_test_case("Missing qualifier", eval_input_2, review_needed_2, [], checker)

  # Test Case 3: Greeting present
  eval_input_3 = [{"id": 903, "combined_insight": "Rent is $1,200."}]
  review_needed_3 = [{"id": 903, "title": "Rent", "summary": "Hello! Your rent hit $1,200."}]
  result_3 = run_test_case("Greeting present", eval_input_3, review_needed_3, [], checker)

  # Test Case 4: Title doesn't encapsulate all info
  eval_input_4 = [{"id": 904, "combined_insight": "Food is $100 and Fun is $50."}]
  review_needed_4 = [{"id": 904, "title": "Food Spend", "summary": "Food hit $100 and fun hit $50."}]
  result_4 = run_test_case("Incomplete title", eval_input_4, review_needed_4, [], checker)

  # Test Case 5: Tone mismatch
  eval_input_5 = [{"id": 905, "combined_insight": "You lost $1,000 in stocks."}]
  review_needed_5 = [{"id": 905, "title": "Stock Loss", "summary": "Yay! You lost $1,000 in stocks! 🥳"}]
  result_5 = run_test_case("Tone mismatch", eval_input_5, review_needed_5, [], checker)

  # Test Case 6: Multiple issues
  eval_input_6 = [{"id": 906, "combined_insight": "Income is $5,000."}]
  review_needed_6 = [{"id": 1, "title": "Income", "summary": "Hi! Income is $5,000."}]
  result_6 = run_test_case("Multiple issues", eval_input_6, review_needed_6, [], checker)

  return [result_1, result_2, result_3, result_4, result_5, result_6]


def main(batch: int = 1, run_number: int = 1):
  """
  Main function to test the PennyHighlightsVerbalizerJsonOptimizer checker
  """
  print(f"\n{'='*80}")
  print(f"BATCH {batch} - RUN {run_number}")
  print(f"{'='*80}\n")
  
  try:
    verbalizer = PennyHighlightsVerbalizerOptimizer()
    checker = CheckHighlightsVerbalizerJsonOptimizer2()
    
    if batch == 1:
      results = test_batch_1(verbalizer, checker)
    elif batch == 2:
      results = test_batch_2(verbalizer, checker)
    elif batch == 3:
      results = test_batch_3(verbalizer, checker)
    elif batch == 4:
      results = test_batch_4(verbalizer, checker)
    elif batch == 5:
      results = test_batch_5(checker)
    else:
      print(f"Invalid batch number: {batch}. Must be 1-5.")
      return
    
    print(f"\n{'='*80}")
    print(f"BATCH {batch} - RUN {run_number} COMPLETE")
    print(f"{'='*80}\n")
    
    return results
  except Exception as e:
    print(f"An error occurred: {e}")
    import traceback
    traceback.print_exc()


if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser(description='Run checker tests in batches')
  parser.add_argument('--batch', type=int, default=1, choices=[1, 2, 3, 4, 5],
                      help='Batch number to run (1-5)')
  parser.add_argument('--run', type=int, default=1, choices=[1, 2, 3],
                      help='Run number for iterative optimization (1-3)')
  args = parser.parse_args()
  main(batch=args.batch, run_number=args.run)

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

SYSTEM_PROMPT = """You are a rigorous checker verifying PennyHighlightsVerbalizerJsonOptimizer outputs against strict rules.

## Evaluation Goals:
1. **ID Integrity**: Output IDs MUST match Input IDs exactly. Any mismatch must be flagged.
2. **Fact Accuracy**: All numbers used must match the source data. However, the output does NOT have to include all insights from the input (filtering is allowed).
3. **Implicit Performance Context**: Summaries can indicate if spending/income is over/under budget either **explicitly** (e.g., "over budget") or **implicitly** through tone/words (e.g., "saved", "hit", "oops"). Do not flag if the meaning is clear.
4. **Magnitude Optionality**: Magnitude of divergence (exactly how much over budget) is NOT required in the output.
5. **Tone Perfection**: 
   - **Encouraging always**: Never negative, angry, or judgmental. Friendly slang like "oops", "ouch", "woah", or "oemgee" is ENCOURAGED as long as the overall tone remains supportive and not shaming.
   - **Reality-aligned**: Tone must reflect the financial reality (celebratory for wins, supportive for risks).
6. **Holistic Coverage**: Titles MUST encompass ALL financial events mentioned in the accompanying summary.
7. **No Greetings**: No "Hi", "Hello", "Hey", or conversation openers.

## Output Format:
STRICT JSON ONLY. No conversational filler.
`{"good_copy": boolean, "info_correct": boolean, "eval_text": string}`
- `info_correct`: True ONLY if ALL numbers, IDs, and financial facts match the EVAL_INPUT.
- `good_copy`: True ONLY if ALL stylistic, tone, and formatting rules are met.
- `eval_text`: Required if either boolean is False. **FORMAT**: "Insight [N]: [issue description]; Insight [N]: [issue description]; etc." where [N] is the order in the list.
  - **MANDATORY**: List EVERY single issue found for each insight. Do not stop at the first issue.
  - **Self-Contained Feedback**: Describe the issue directly (e.g., "Judgmental language like 'sneaky'", "Mismatched ID", "Title omits savings"). NEVER reference rule names or numbers.
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
  Batch 1: Test cases 1-2
  """
  if verbalizer is None:
    verbalizer = PennyHighlightsVerbalizerOptimizer()
  if checker is None:
    checker = CheckHighlightsVerbalizerJsonOptimizer2()
  
  # Test 1
  eval_input_1 = [
    {
      "id": 1,
      "combined_insight": "Your shelter costs are way down this month to $1,248, mainly from less on home stuff, utilities, and upkeep. 🥳🏠 Oh em gee!  You got a huge surprise income boost of $8,800 this week, mostly from your business, and you're projected to spend only $68 by the end of the week!  Way to go, you savvy boss babe!"
    },
    {
      "id": 2,
      "combined_insight": "Looks like you spent less on food this month, down to $1,007, mostly from less eating out, deliveries, and groceries. 🍽️🚚🛒 Your transport costs are way down this month to just $46, mostly 'cause you took public transit less. 🚇"
    }
  ]
  
  print("\n--- Generating verbalizer output for Test 1 ---")
  review_needed_1 = verbalizer.generate_response(eval_input_1)
  
  print("\n--- Checking Test 1 ---")
  result_1 = run_test_case("batch_1_test_1", eval_input_1, review_needed_1, [], checker)
  
  # Test 2
  eval_input_2 = [
    {
      "id": 3,
      "combined_insight": "Warning! 🚨 You've spent $750 on shopping this month, which is $250 over your budget. Most of it went to online stores."
    },
    {
      "id": 4,
      "combined_insight": "Heads up! You have $500 in uncategorized expenses this week. Let's categorize them to keep your budget on track! 🧐"
    }
  ]
  
  print("\n--- Generating verbalizer output for Test 2 ---")
  review_needed_2 = verbalizer.generate_response(eval_input_2)
  
  print("\n--- Checking Test 2 ---")
  result_2 = run_test_case("batch_1_test_2", eval_input_2, review_needed_2, [], checker)
  
  return [result_1, result_2]


def test_batch_2(verbalizer: PennyHighlightsVerbalizerOptimizer = None, checker: CheckHighlightsVerbalizerJsonOptimizer2 = None):
  """
  Batch 2: Test cases 3-4
  """
  if verbalizer is None:
    verbalizer = PennyHighlightsVerbalizerOptimizer()
  if checker is None:
    checker = CheckHighlightsVerbalizerJsonOptimizer2()
  
  # Test 3
  eval_input_3 = [
    {
      "id": 5,
      "combined_insight": "You're so close! You've saved $9,500 for your vacation, that's 95% of your $10,000 goal! 🌴☀️"
    },
    {
      "id": 6,
      "combined_insight": "Just a heads-up, you had a small unexpected charge of $35 for a subscription service you might have forgotten about. 😬"
    }
  ]
  
  print("\n--- Generating verbalizer output for Test 3 ---")
  review_needed_3 = verbalizer.generate_response(eval_input_3)
  
  print("\n--- Checking Test 3 ---")
  result_3 = run_test_case("batch_2_test_3", eval_input_3, review_needed_3, [], checker)
  
  # Test 4
  eval_input_4 = [
    {
      "id": 7,
      "combined_insight": "Amazing! Your side hustle brought in an extra $1,200 this month! Keep up the great work! 🚀💰"
    },
    {
      "id": 8,
      "combined_insight": "Great job on cutting down costs! Your electricity bill was only $55 this month, down from $85 last month. 💡"
    }
  ]
  
  print("\n--- Generating verbalizer output for Test 4 ---")
  review_needed_4 = verbalizer.generate_response(eval_input_4)
  
  print("\n--- Checking Test 4 ---")
  result_4 = run_test_case("batch_2_test_4", eval_input_4, review_needed_4, [], checker)
  
  return [result_3, result_4]


def test_batch_3(verbalizer: PennyHighlightsVerbalizerOptimizer = None, checker: CheckHighlightsVerbalizerJsonOptimizer2 = None):
  """
  Batch 3: Test cases 5-6
  """
  if verbalizer is None:
    verbalizer = PennyHighlightsVerbalizerOptimizer()
  if checker is None:
    checker = CheckHighlightsVerbalizerJsonOptimizer2()
  
  # Test 5
  eval_input_5 = [
    {
      "id": 9,
      "combined_insight": "Just noting a large expense: you paid $2,500 for car repairs this week. Remember to budget for these things! 🔧🚗"
    },
    {
      "id": 10,
      "combined_insight": "Your credit card balance is at $3,200 this month. Let's make a plan to pay it down! 💳"
    }
  ]
  
  print("\n--- Generating verbalizer output for Test 5 ---")
  review_needed_5 = verbalizer.generate_response(eval_input_5)
  
  print("\n--- Checking Test 5 ---")
  result_5 = run_test_case("batch_3_test_5", eval_input_5, review_needed_5, [], checker)
  
  # Test 6
  eval_input_6 = [
    {
      "id": 11,
      "combined_insight": "To the moon! 🚀 Your investment portfolio is up 15% this quarter, adding a nice $4,500 to your net worth."
    },
    {
      "id": 12,
      "combined_insight": "Quick reminder: Your rent of $2,200 is due in 3 days. Don't be late! 🗓️"
    }
  ]
  
  print("\n--- Generating verbalizer output for Test 6 ---")
  review_needed_6 = verbalizer.generate_response(eval_input_6)
  
  print("\n--- Checking Test 6 ---")
  result_6 = run_test_case("batch_3_test_6", eval_input_6, review_needed_6, [], checker)
  
  return [result_5, result_6]


def test_batch_4(verbalizer: PennyHighlightsVerbalizerOptimizer = None, checker: CheckHighlightsVerbalizerJsonOptimizer2 = None):
  """
  Batch 4: Test cases 7-8 (additional edge cases)
  """
  if verbalizer is None:
    verbalizer = PennyHighlightsVerbalizerOptimizer()
  if checker is None:
    checker = CheckHighlightsVerbalizerJsonOptimizer2()
  
  # Test 7
  eval_input_7 = [
    {
      "id": 13,
      "combined_insight": "Hello! Your grocery spending increased to $450 this month, which is higher than your usual $300 average. 🛒"
    },
    {
      "id": 14,
      "combined_insight": "Good morning! Your income from freelancing was $2,000 this month, matching your target perfectly. Great job staying on track! 💼"
    }
  ]
  
  print("\n--- Generating verbalizer output for Test 7 ---")
  review_needed_7 = verbalizer.generate_response(eval_input_7)
  
  print("\n--- Checking Test 7 ---")
  result_7 = run_test_case("batch_4_test_7", eval_input_7, review_needed_7, [], checker)
  
  # Test 8
  eval_input_8 = [
    {
      "id": 15,
      "combined_insight": "Your dining out expenses were $180 this week, down significantly from last week's $320. You're making great progress! 🍽️"
    },
    {
      "id": 16,
      "combined_insight": "Heads up! Your subscription costs totaled $150 this month, which is $50 more than your budgeted $100. Consider reviewing your subscriptions. 📱"
    }
  ]
  
  print("\n--- Generating verbalizer output for Test 8 ---")
  review_needed_8 = verbalizer.generate_response(eval_input_8)
  
  print("\n--- Checking Test 8 ---")
  result_8 = run_test_case("batch_4_test_8", eval_input_8, review_needed_8, [], checker)
  
  return [result_7, result_8]


def test_batch_5(checker: CheckHighlightsVerbalizerJsonOptimizer2 = None):
  """
  Batch 5: Failure cases to test Checker's detection (ID mismatch, tone, missing info, etc.)
  """
  if checker is None:
    checker = CheckHighlightsVerbalizerJsonOptimizer2()

  # Test Case 1: ID mismatch
  eval_input_1 = [{"id": 101, "combined_insight": "You saved $50 on coffee this month."}]
  review_needed_1 = [{"id": 1, "title": "Coffee Savings!", "summary": "You saved $50 on coffee this month!"}]
  result_1 = run_test_case("ID mismatch (Case 1)", eval_input_1, review_needed_1, [], checker)

  # Test Case 2: Title contradiction
  eval_input_2 = [
    {"id": 10, "combined_insight": "Food spending hit $500, which is $200 over budget. Shelter dropped to $1,200, which is $100 under budget."}
  ]
  review_needed_2 = [{
    "id": 10,
    "title": "Food and Shelter Excellence!",
    "summary": "Food hit $500 (over budget). Shelter dropped to $1,200 (under budget)."
  }]
  result_2 = run_test_case("Title contradiction (Case 2)", eval_input_2, review_needed_2, [], checker)

  # Test Case 3: Negative tone
  eval_input_3 = [{"id": 11, "combined_insight": "You spent $500 on shoes this month, $300 over budget."}]
  review_needed_3 = [{
    "id": 11,
    "title": "Irresponsible Spending! 😡",
    "summary": "You spent $500 on shoes? That's irresponsible and you should be ashamed. You're failing your budget."
  }]
  result_3 = run_test_case("Negative tone (Case 3)", eval_input_3, review_needed_3, [], checker)

  # Test Case 4: Missing performance context
  eval_input_4 = [{"id": 12, "combined_insight": "Grocery spending is $100 this week."}]
  review_needed_4 = [{
    "id": 12,
    "title": "Grocery Spend",
    "summary": "Spending for groceries is $100."
  }]
  result_4 = run_test_case("Missing performance context (Case 4)", eval_input_4, review_needed_4, [], checker)

  # Test Case 5: Missing amount
  eval_input_5 = [{"id": 13, "combined_insight": "Grocery spending is $450 this month, which is higher than usual."}]
  review_needed_5 = [{
    "id": 13,
    "title": "High Grocery Spend",
    "summary": "Grocery spending is higher than usual this month."
  }]
  result_5 = run_test_case("Missing amount (Case 5)", eval_input_5, review_needed_5, [], checker)

  # Test Case 6: Multiple issues
  eval_input_6 = [{"id": 99, "combined_insight": "Income hit $5,000 this month."}]
  review_needed_6 = [{
    "id": 1,
    "title": "Income",
    "summary": "Hello! Your income is high."
  }]
  result_6 = run_test_case("Multiple issues (Case 6)", eval_input_6, review_needed_6, [], checker)

  return [result_1, result_2, result_3, result_4, result_5, result_6]


def main(batch: int = 1, run_number: int = 1):
  """
  Main function to test the PennyHighlightsVerbalizerJsonOptimizer checker
  
  Args:
    batch: Batch number (1-5) to determine which tests to run
    run_number: Run number (1-3) for iterative optimization
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

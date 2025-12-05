from google import genai
from google.genai import types
import os
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

SYSTEM_PROMPT = """You are a checker verifying verbalizer outputs against rules.

## Input:
- **EVAL_INPUT**: Action plan requested by user (string)
- **PAST_REVIEW_OUTCOMES**: Array of past reviews, each with `output`, `good_copy`, `info_correct`, `eval_text`
- **REVIEW_NEEDED**: Verbalizer response to review (string)

## Output:
JSON: `{"good_copy": boolean, "info_correct": boolean, "eval_text": string}`
- `good_copy`: True if REVIEW_NEEDED addresses EVAL_INPUT and includes required elements
- `info_correct`: True if REVIEW_NEEDED follows all rules
- `eval_text`: Required if either boolean is False; be specific and concise

## Critical Priority: Learn from PAST_REVIEW_OUTCOMES
**MANDATORY**: If PAST_REVIEW_OUTCOMES flags issues that still exist in REVIEW_NEEDED, mark as incorrect.
- Extract all issues from past `eval_text` fields
- Check if REVIEW_NEEDED repeats the same mistakes
- If past reviews say "contains 'Please'" and it's still there → mark `info_correct: False`
- If past reviews say "missing wait messaging" and it's still missing → mark `info_correct: False`

## Rules

### Required Content
1. Progressive tense: Actions in progressive tense (e.g., "Creating...", "Setting up...")
2. Creative wait messaging: Varied phrases like "Bear with me...", "Back in a flash... ⚡", "Hol' up. 🤔", "Let me think... 🧠⚙️"
3. No "Please": Exclude "Please" from wait instructions

### Guidelines
- **15 words max** (strict): Count all words
- **Sentence-case**: Not all caps or title case
- **Few emojis**: Use sparingly
- **Focus on "you"**: No "let's", "us", "we" - use "you" doing it
- **Tone**: Light, conversational, friendly
- **Urgency**: Convey importance/urgency if needed

## Verification Steps

1. **Check PAST_REVIEW_OUTCOMES first**: Extract all flagged issues. If REVIEW_NEEDED repeats them → mark False
2. **Verify good_copy**: Does REVIEW_NEEDED address EVAL_INPUT? Includes progressive tense + creative wait? Missing elements = False
3. **Verify info_correct**: Apply all rules:
   - 15 words max? Sentence-case? Few emojis?
   - No "Please"? No "let's"/"us"/"we"?
   - Progressive tense? Creative wait messaging?
   - Light, conversational tone?
4. **Write eval_text**: If False, list specific issues. Reference unfixed PAST_REVIEW_OUTCOMES issues.
"""

class CheckStrOptimizer:
  """Handles all Gemini API interactions for checking evaluation of verbalizer outputs against rules"""
  
  def __init__(self, model_name="gemini-flash-lite-latest"):
    """Initialize the Gemini agent with API configuration for checking evaluations"""
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
    self.max_output_tokens = 8192
    
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
    Generate a response using Gemini API for checking evaluations.
    
    Args:
      eval_input: The action plan requested by a user (string).
      past_review_outcomes: An array of past review outcomes, each containing `output`, `good_copy`, `info_correct`, and `eval_text`.
      review_needed: The verbalizer response that needs to be reviewed (string).
      
    Returns:
      Dictionary with good_copy, info_correct, and eval_text keys
    """
    # Create request text with the new input structure
    request_text = types.Part.from_text(text=f"""<EVAL_INPUT>

{eval_input}

</EVAL_INPUT>

<PAST_REVIEW_OUTCOMES>

{json.dumps(past_review_outcomes, indent=2)}

</PAST_REVIEW_OUTCOMES>

<REVIEW_NEEDED>

{review_needed}

</REVIEW_NEEDED>

Output:""")
    
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


def test_with_inputs(eval_input: str, past_review_outcomes: list, review_needed: str, checker: CheckStrOptimizer = None):
  """
  Convenient method to test the checker optimizer with custom inputs.
  
  Args:
    eval_input: The action plan requested by a user (string).
    past_review_outcomes: An array of past review outcomes, each containing `output`, `good_copy`, `info_correct`, and `eval_text`.
    review_needed: The verbalizer response that needs to be reviewed (string).
    checker: Optional CheckStrOptimizer instance. If None, creates a new one.
    
  Returns:
    Dictionary with good_copy, info_correct, and eval_text keys
  """
  if checker is None:
    checker = CheckStrOptimizer()
  
  return checker.generate_response(eval_input, past_review_outcomes, review_needed)


# Test cases covering different scenarios
TEST_CASES = [
  {
    "name": "correct_response",
    "eval_input": "Create a budget for groceries and dining out",
    "review_needed": "Creating your budget now. Bear with me while I set this up... ⚡",
    "past_review_outcomes": []
  },
  {
    "name": "contains_please",
    "eval_input": "Analyze spending patterns for the last month",
    "review_needed": "Please wait while I analyze your spending patterns. This will take a moment.",
    "past_review_outcomes": []
  },
  {
    "name": "uses_lets",
    "eval_input": "Update transaction categories",
    "review_needed": "Let's update those categories together! Hold on...",
    "past_review_outcomes": []
  },
  {
    "name": "exceeds_word_limit",
    "eval_input": "Generate financial report",
    "review_needed": "I am currently generating your comprehensive financial report with all the details and insights you requested. This will take just a moment.",
    "past_review_outcomes": []
  },
  {
    "name": "missing_wait_messaging",
    "eval_input": "Calculate monthly expenses",
    "review_needed": "Calculating your monthly expenses now.",
    "past_review_outcomes": []
  },
  {
    "name": "missing_progressive_tense",
    "eval_input": "Review transaction history",
    "review_needed": "I will review your transaction history. Wait for me.",
    "past_review_outcomes": []
  },
  {
    "name": "correct_with_creative_wait",
    "eval_input": "Set up savings goal",
    "review_needed": "Setting up your savings goal. Back in a flash... ⚡",
    "past_review_outcomes": []
  },
  {
    "name": "correct_with_hol_up",
    "eval_input": "Categorize recent transactions",
    "review_needed": "Categorizing your transactions. Hol' up. 🤔",
    "past_review_outcomes": []
  },
  {
    "name": "correct_with_allow_me",
    "eval_input": "Create spending forecast",
    "review_needed": "Creating your forecast. Allow me a moment...",
    "past_review_outcomes": []
  },
  {
    "name": "correct_with_thinking",
    "eval_input": "Optimize budget allocation",
    "review_needed": "Optimizing your budget. Let me think... 🧠⚙️",
    "past_review_outcomes": []
  },
  {
    "name": "uses_us_together",
    "eval_input": "Plan monthly budget",
    "review_needed": "Planning our monthly budget together. Just a sec...",
    "past_review_outcomes": []
  },
  {
    "name": "too_formal",
    "eval_input": "Generate insights report",
    "review_needed": "Please allow me to generate your insights report. I appreciate your patience.",
    "past_review_outcomes": []
  },
  {
    "name": "missing_next_steps",
    "eval_input": "Update account settings",
    "review_needed": "Wait for me to finish.",
    "past_review_outcomes": []
  },
  {
    "name": "correct_urgent_tone",
    "eval_input": "Fix transaction categorization error",
    "review_needed": "Fixing that error now. Bear with me... ⚡",
    "past_review_outcomes": []
  },
  {
    "name": "past_review_outcomes_issue_persists",
    "eval_input": "Research different ways to get rich.",
    "review_needed": "I'm crafting your rich-gettin' roadmap. 🗺️  Bear with me... your financial future awaits! 🚀",
    "past_review_outcomes": [
      {
        "output": "I'm crafting your rich-gettin' roadmap. 🗺️  Hold tight, a treasure awaits! 💎",
        "good_copy": True,
        "info_correct": True,
        "eval_text": ""
      }
    ]
  }
]


def run_test(test_case: dict, checker: CheckStrOptimizer = None):
  """
  Run a single test case.
  
  Args:
    test_case: Test case dict with name, eval_item (or eval_input), eval_output (or review_needed), and optionally past_review_outcomes
    checker: Optional CheckStrOptimizer instance. If None, creates a new one.
    
  Returns:
    Dictionary with good_copy, info_correct, and eval_text keys, or None if error occurred
  """
  if checker is None:
    checker = CheckStrOptimizer()
  
  print(f"\n{'='*80}")
  print(f"Running test: {test_case['name']}")
  print(f"{'='*80}")
  
  try:
    # Support both old format (eval_item/eval_output) and new format (eval_input/review_needed/past_review_outcomes)
    if "eval_input" in test_case:
      # New format
      eval_input = test_case["eval_input"]
      past_review_outcomes = test_case.get("past_review_outcomes", [])
      review_needed = test_case["review_needed"]
    else:
      # Old format - convert to new format
      eval_input = test_case["eval_item"]
      past_review_outcomes = test_case.get("past_review_outcomes", [])
      review_needed = test_case["eval_output"]
    
    result = checker.generate_response(eval_input, past_review_outcomes, review_needed)
    print(f"Result:")
    print(json.dumps(result, indent=2))
    print(f"{'='*80}")
    return result
  except Exception as e:
    print(f"ERROR: {type(e).__name__}: {str(e)}")
    print(f"{'='*80}")
    return None


def run_tests(test_names: list = None, checker: CheckStrOptimizer = None):
  """
  Run multiple test cases.
  
  Args:
    test_names: List of test case names to run. If None, runs all tests.
    checker: Optional CheckStrOptimizer instance. If None, creates a new one.
    
  Returns:
    List of results (None entries indicate failed tests)
  """
  if checker is None:
    checker = CheckStrOptimizer()
  
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
  """Main function to test the checker optimizer"""
  checker = CheckStrOptimizer()
  
  # Run all tests
  run_tests(checker=checker)
  
  # Or run specific tests:
  # run_tests(["correct_response", "contains_please"], checker=checker)


if __name__ == "__main__":
  main()

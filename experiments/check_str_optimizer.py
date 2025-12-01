from google import genai
from google.genai import types
import os
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

SYSTEM_PROMPT = """You are a checker verifying verbalizer outputs against rules.

## Input:
- **eval_item**: The action plan requested by a user (string)
- **eval_output**: The verbalizer response (string)

## Output:
JSON with keys: **good_copy** (boolean), **info_correct** (boolean), **eval_text** (string with notes if either is false)

## Rules:
**Required Content** (not formatted as list):
1. Your Next Steps: Progressive tense summarizing actions
2. User's Next Steps: Creative wait instruction (exclude "Please")
3. Wait messaging: Varied phrases like "Bear with me...", "Back in a flash... ⚡", "Hol' up. 🤔", "Let me think... 🧠⚙️"

**Guidelines**:
- Max 15 words, sentence-case, few emojis
- Focus on "you" doing it (no "let's", "us", "we")
- Light, conversational, friendly tone
- Convey importance/urgency if needed

## Verification:
1. **good_copy**: Addresses eval_item and includes all required elements
2. **info_correct**: Follows all rules (word limit, formatting, tone, no "Please"/"let's", progressive tense, creative wait)
3. **eval_text**: Specific, concise notes on violations

<EXAMPLES>

input:
```json
{"eval_item": "Create a budget for groceries", "eval_output": "Creating your budget now. Bear with me... ⚡"}
```
output:
```json
{"good_copy": true, "info_correct": true, "eval_text": ""}
```

input:
```json
{"eval_item": "Analyze spending patterns", "eval_output": "Please wait while I analyze. Let's do this together!"}
```
output:
```json
{"good_copy": false, "info_correct": false, "eval_text": "Contains 'Please' (exclude). Uses 'Let's' (use 'you' instead). Missing creative wait messaging. May exceed 15 words."}
```

</EXAMPLES>
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

  
  def generate_response(self, eval_item: str, eval_output: str) -> dict:
    """
    Generate a response using Gemini API for checking evaluations.
    
    Args:
      eval_item: The detailed action plan requested by a user (string).
      eval_output: The evaluation output. A string response from the verbalizer.
      
    Returns:
      Dictionary with good_copy, info_correct, and eval_text keys
    """
    # Create request text with eval_item and eval_output
    request_text = types.Part.from_text(text=f"""Input:
{json.dumps({"eval_item": eval_item, "eval_output": eval_output}, indent=2)}

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


def test_with_inputs(eval_item: str, eval_output: str, checker: CheckStrOptimizer = None):
  """
  Convenient method to test the checker optimizer with custom inputs.
  
  Args:
    eval_item: The detailed action plan requested by a user (string).
    eval_output: The evaluation output. A string response from the verbalizer.
    checker: Optional CheckStrOptimizer instance. If None, creates a new one.
    
  Returns:
    Dictionary with good_copy, info_correct, and eval_text keys
  """
  if checker is None:
    checker = CheckStrOptimizer()
  
  return checker.generate_response(eval_item, eval_output)


# Test cases covering different scenarios
TEST_CASES = [
  {
    "name": "correct_response",
    "eval_item": "Create a budget for groceries and dining out",
    "eval_output": "Creating your budget now. Bear with me while I set this up... ⚡"
  },
  {
    "name": "contains_please",
    "eval_item": "Analyze spending patterns for the last month",
    "eval_output": "Please wait while I analyze your spending patterns. This will take a moment."
  },
  {
    "name": "uses_lets",
    "eval_item": "Update transaction categories",
    "eval_output": "Let's update those categories together! Hold on..."
  },
  {
    "name": "exceeds_word_limit",
    "eval_item": "Generate financial report",
    "eval_output": "I am currently generating your comprehensive financial report with all the details and insights you requested. This will take just a moment."
  },
  {
    "name": "missing_wait_messaging",
    "eval_item": "Calculate monthly expenses",
    "eval_output": "Calculating your monthly expenses now."
  },
  {
    "name": "missing_progressive_tense",
    "eval_item": "Review transaction history",
    "eval_output": "I will review your transaction history. Wait for me."
  },
  {
    "name": "correct_with_creative_wait",
    "eval_item": "Set up savings goal",
    "eval_output": "Setting up your savings goal. Back in a flash... ⚡"
  },
  {
    "name": "correct_with_hol_up",
    "eval_item": "Categorize recent transactions",
    "eval_output": "Categorizing your transactions. Hol' up. 🤔"
  },
  {
    "name": "correct_with_allow_me",
    "eval_item": "Create spending forecast",
    "eval_output": "Creating your forecast. Allow me a moment..."
  },
  {
    "name": "correct_with_thinking",
    "eval_item": "Optimize budget allocation",
    "eval_output": "Optimizing your budget. Let me think... 🧠⚙️"
  },
  {
    "name": "uses_us_together",
    "eval_item": "Plan monthly budget",
    "eval_output": "Planning our monthly budget together. Just a sec..."
  },
  {
    "name": "too_formal",
    "eval_item": "Generate insights report",
    "eval_output": "Please allow me to generate your insights report. I appreciate your patience."
  },
  {
    "name": "missing_next_steps",
    "eval_item": "Update account settings",
    "eval_output": "Wait for me to finish."
  },
  {
    "name": "correct_urgent_tone",
    "eval_item": "Fix transaction categorization error",
    "eval_output": "Fixing that error now. Bear with me... ⚡"
  }
]


def run_test(test_case: dict, checker: CheckStrOptimizer = None):
  """
  Run a single test case.
  
  Args:
    test_case: Test case dict with name, eval_item, and eval_output
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
    result = checker.generate_response(test_case["eval_item"], test_case["eval_output"])
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

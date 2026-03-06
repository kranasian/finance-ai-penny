from google import genai
from google.genai import types
import os
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

SYSTEM_PROMPT = """You are an AI assistant that evaluates account renaming. Check the optimizer's output against strict rules. Identify errors rigorously and concisely.

## Core Task
- Evaluate `REVIEW_NEEDED` against `EVAL_INPUT`.
- Output a JSON object: `{"good_copy": boolean, "info_correct": boolean, "eval_text": string}`.

## Output Schema
- `good_copy`: `true` if the output follows ALL renaming rules. `false` if ANY rule is violated.
- `info_correct`: `true` if the output name conceptually matches the input account. `false` if wrong bank/account/hallucination.
- `eval_text`: Required if `good_copy` or `info_correct` is `false`.
  - **Must be a single string** (use `\\n` for newlines).
  - **Use single quotes** for quoted text inside the string to avoid JSON errors.
  - Bullet points. Concise.
  - **Start with the FIX action**.
  - Format: `- ID <id>: <Action 1>; <Action 2> (Expected: <Final Correct Value>)`

## Rules (Apply in Order)

1. **Identity**: Output must match input account concept. **Use correct masks from input.**
2. **Cleaning**:
   - **Remove**: "Free", "Visa", "Mastercard", "Signature Card", "Signature", "Account", "The", "By", "Card".
   - **Keep**: "Checking", "Savings", "Money Market", "Credit", "Debit".
   - Strip special symbols (except hyphens/slashes).
3. **Bank Prefix**:
   - **Single Bank Batch** (All accounts from same bank): REMOVE bank name.
   - **Mixed Bank Batch** (Accounts from different banks): MUST HAVE bank prefix.
4. **Deduplication**:
   - Check for EXACT string matches after Cleaning & Prefixing.
   - **Count occurrences** of each final name string in the entire batch.
   - **If Count == 1 (Unique)**: Output name AS IS. **ABSOLUTELY NO MASK**.
   - **If Count > 1 (Duplicate)**: Append " **" + `mask` (from input) to ALL duplicates.
   - **CRITICAL**: Do NOT add a mask unless there is an actual duplicate name in the final output list.
5. **Truncation** (CRITICAL):
   - Check length of the *Final Name*.
   - **If > 30 chars**: Truncate to exactly 30.
   - **If <= 30 chars**: DO NOT TRUNCATE.

## Examples of `eval_text`
- `- ID 101: Remove "Chase" prefix (Expected: Sapphire Reserve)`
- `- ID 102: Add "Citi" prefix (Expected: Citi Double Cash)`
- `- ID 103: Truncate to 30 chars (Expected: Capital One 360 Performance S)`
- `- ID 104: Add mask " **5678" (Expected: Checking **5678)`
- `- ID 105: Remove mask (Expected: Savings)`
"""

class CheckAccountRenamerOptimizer:
  """Handles all Gemini API interactions for checking AccountRenamerOptimizer outputs against rules"""
  
  def __init__(self, model_name="gemini-flash-lite-latest"):
    """Initialize the Gemini agent with API configuration for checking AccountRenamerOptimizer evaluations"""
    # API Configuration
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
      raise ValueError("GEMINI_API_KEY environment variable is not set. Please set it in your .env file or environment.")
    self.client = genai.Client(api_key=api_key)
    
    # Model Configuration
    self.thinking_budget = 2048
    self.model_name = model_name
    
    # Generation Configuration Constants
    self.temperature = 0.1
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
  
  def generate_response(self, eval_input: list, review_needed: list) -> dict:
    """
    Generate a response using Gemini API for checking AccountRenamerOptimizer outputs.
    
    Args:
      eval_input: A JSON array of account objects (the input to the optimizer).
      review_needed: The AccountRenamerOptimizer output that needs to be reviewed (JSON array).
      
    Returns:
      Dictionary with good_copy, info_correct, and eval_text keys
    """
    # Create request text with the new input structure
    request_text_str = f"""<EVAL_INPUT>

{json.dumps(eval_input, indent=2)}

</EVAL_INPUT>

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
    
    # Parse JSON response
    response_json = None
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
        response_json = json.loads(json_str)
      else:
        # Try parsing the whole response
        response_json = json.loads(output_text.strip())
    except json.JSONDecodeError as e:
      raise ValueError(f"Failed to parse JSON response: {e}\\nResponse length: {len(output_text)}\\nResponse preview: {output_text[:500]}")

    if thought_summary:
      print("-" * 80)
      print("THOUGHT SUMMARY:")
      print(thought_summary.strip())
      print("-" * 80)
      
    return {
      "response": response_json,
      "thought_summary": thought_summary.strip() if thought_summary else ""
    }


def run_test_case(test_name: str, eval_input: list, review_needed: list, checker: 'CheckAccountRenamerOptimizer' = None):
  """
  Utility function to run a test case with custom inputs, common error handling, and output formatting.
  
  Args:
    test_name: Name of the test case
    eval_input: A list of account objects.
    review_needed: The optimizer output that needs to be reviewed.
    checker: Optional CheckAccountRenamerOptimizer instance. If None, creates a new one.
    
  Returns:
    Dictionary with good_copy, info_correct, and eval_text keys, or None if error occurred
  """
  if checker is None:
    checker = CheckAccountRenamerOptimizer()

  print(f"\n{'='*80}")
  print(f"Running test: {test_name}")
  print(f"{'='*80}")

  try:
    # Directly call the checker's response with the provided inputs.
    result = checker.generate_response(eval_input, review_needed)
    print(f"Result:")
    print(json.dumps(result["response"], indent=2))
    print(f"{'='*80}")
    return result
  except Exception as e:
    print(f"ERROR: {type(e).__name__}: {str(e)}")
    print(f"{'='*80}")
    return None


def run_correct_renaming_test(checker: CheckAccountRenamerOptimizer = None):
  """
  Test a case where the optimizer's output is perfectly correct.
  """
  eval_input = [
      {"id": 1, "account_name": "PLATINUM CHK", "long_account_name": "Amex Platinum Checking", "bank_name": "American Express", "mask": "9012"},
      {"id": 2, "account_name": "SAVINGS", "long_account_name": "High Yield Savings", "bank_name": "Discover", "mask": "3456"}
  ]
  
  review_needed = [
      {"id": 1, "account_name": "American Express Platinum Checking"},
      {"id": 2, "account_name": "Discover High Yield Savings"}
  ]
  
  return run_test_case("correct_renaming_test", eval_input, review_needed, checker)

def run_missing_bank_prefix_test(checker: CheckAccountRenamerOptimizer = None):
  """
  Test for missing bank prefixes in a mixed batch.
  """
  eval_input = [
      {"id": 1, "account_name": "ADVANTAGE SAFE", "long_account_name": "Advantage SafeBalance", "bank_name": "Bank of America", "mask": "7890"},
      {"id": 2, "account_name": "360 CHECKING", "long_account_name": "360 Checking", "bank_name": "Capital One", "mask": "1122"}
  ]
  
  review_needed = [
      {"id": 1, "account_name": "Advantage SafeBalance"},
      {"id": 2, "account_name": "360 Checking"}
  ]
  
  return run_test_case("missing_bank_prefix_test", eval_input, review_needed, checker)

def run_failed_deduplication_test(checker: CheckAccountRenamerOptimizer = None):
  """
  Test for failed deduplication (missing masks on duplicates).
  """
  eval_input = [
      {"id": 1, "account_name": "Everyday Check", "long_account_name": "Everyday Checking", "bank_name": "Wells Fargo", "mask": "5555"},
      {"id": 2, "account_name": "Everyday Check", "long_account_name": "Everyday Checking", "bank_name": "Wells Fargo", "mask": "6666"}
  ]
  
  review_needed = [
      {"id": 1, "account_name": "Wells Fargo Everyday Checking"},
      {"id": 2, "account_name": "Wells Fargo Everyday Checking"}
  ]
  
  return run_test_case("failed_deduplication_test", eval_input, review_needed, checker)

def run_incomplete_cleaning_test(checker: CheckAccountRenamerOptimizer = None):
  """
  Test for incomplete cleaning (forbidden words left in).
  """
  eval_input = [
      {"id": 1, "account_name": "Sapphire Pref", "long_account_name": "Chase Sapphire Preferred Visa Signature", "bank_name": "Chase", "mask": "7777"}
  ]
  
  review_needed = [
      {"id": 1, "account_name": "Chase Sapphire Preferred Visa Signature"}
  ]
  
  return run_test_case("incomplete_cleaning_test", eval_input, review_needed, checker)

def run_truncation_fail_test(checker: CheckAccountRenamerOptimizer = None):
  """
  Test for failure to truncate long names.
  """
  eval_input = [
      {"id": 1, "account_name": "Super Long", "long_account_name": "Schwab Investor Checking Account with High Yield", "bank_name": "Charles Schwab", "mask": "8888"}
  ]
  
  review_needed = [
      {"id": 1, "account_name": "Charles Schwab Investor Checking Account with High Yield"}
  ]
  
  return run_test_case("truncation_fail_test", eval_input, review_needed, checker)


def main(batch: int = 0):
  """Main function to test the AccountRenamerOptimizer checker"""
  checker = CheckAccountRenamerOptimizer()
  
  if batch == 0:
    # Run all tests if no specific batch is selected
    run_correct_renaming_test(checker)
    run_missing_bank_prefix_test(checker)
    run_failed_deduplication_test(checker)
    run_incomplete_cleaning_test(checker)
    run_truncation_fail_test(checker)
  elif batch == 1:
    run_correct_renaming_test(checker)
  elif batch == 2:
    run_missing_bank_prefix_test(checker)
  elif batch == 3:
    run_failed_deduplication_test(checker)
  elif batch == 4:
    run_incomplete_cleaning_test(checker)
  elif batch == 5:
    run_truncation_fail_test(checker)
  else:
    print("Invalid batch number. Please choose from 0 to 5.")


if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser(description='Run checker tests in batches')
  parser.add_argument('--batch', type=int, default=0, choices=range(6),
                      help='Batch number to run (1-5). 0 runs all.')
  args = parser.parse_args()
  main(batch=args.batch)

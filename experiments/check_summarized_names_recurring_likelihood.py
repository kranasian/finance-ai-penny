from google import genai
from google.genai import types
import os
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

SYSTEM_PROMPT = """You are an AI assistant that evaluates the output of a transaction classifier. Your job is to check the classifier's output against a set of rules and flag any issues. You do not correct the output, you only identify and explain errors. Your evaluation must be rigorous and detailed.

## Core Task
- Evaluate the `REVIEW_NEEDED` JSON data based on the provided `EVAL_INPUT` and `PAST_REVIEW_OUTCOMES`.
- Identify all discrepancies between the classifier's output and the rules.
- Provide clear, concise, and non-circular feedback for every error in the `eval_text`.

## Key Definitions
- **Recurring**: Any transaction that is expected to occur at regular intervals (e.g., monthly, weekly, bi-weekly, annually, quarterly).
- **Bills**: Any recurring outflow of money (payment). This is a broad definition and not limited to traditional utility bills. For example, a monthly Netflix subscription is a bill.

## Input Schema
- `EVAL_INPUT`: A JSON array of transactions, each with `id`, `name`, and `description`.
- `PAST_REVIEW_OUTCOMES`: A history of previous evaluations. **You must learn from these**. If the classifier repeats a mistake found in here, it is an error.
- `REVIEW_NEEDED`: The JSON output from the classifier that you must review.

## Output Schema
You must output a single JSON object with the following structure:
`{"good_copy": boolean, "info_correct": boolean, "eval_text": string}`
- `good_copy`: **This is a structural check only.** `true` if `REVIEW_NEEDED` is a valid JSON array and every object within it contains the keys `id`, `is_bills`, `is_salary`, and `is_sidegig`. It does not check if the values are correct.
- `info_correct`: `true` if all classifications in `REVIEW_NEEDED` adhere to the "Classifier Rules for Verification".
- `eval_text`: **Required if `good_copy` or `info_correct` is `false`. Only explain mistakes in this field.**
  - **If the output for an ID is correct, do not include it in the `eval_text`.**
  - For each ID that has an error, list every single issue.
  - Each ID's feedback must be on its own line.
  - Format: `ID <id>: For <field_name>, the value should be <correct_value_or_range> because <explanation>.`
  - Your explanations must be clear, avoid circular reasoning, and be understandable on their own without referencing the rules list.
  - **Do not refer to rule names or numbers** (e.g., instead of saying "violates Golden Rule #1," say "the transaction name is ambiguous").
  - **Before finishing, double-check that you have not missed any errors.**

## Classifier Rules for Verification

### Golden Rules (Highest Priority)
1.  **One-Time Purchases**: If a transaction is clearly a one-time event (e.g., "Starbucks coffee", "Shopping Mart"), all likelihoods **MUST** be `IMPOSSIBLE`.

### Categorization Logic
1.  **Step 1: Determine if Recurring**: First, analyze the transaction's `name` and `description` to see if it's recurring. If not, Golden Rule #2 applies.
2.  **Step 2: Infer Direction (Inflow/Outflow)**: Determine if the money is coming in (income) or going out (expense).
    - **If the direction is ambiguous**: Assume the transaction type that is more common for the establishment. For example, "Netflix" is typically an outflow (payment), while a transaction from a known payroll company is an inflow.
3.  **Step 3: Apply Rules (only if recurring)**:
    - **If Outflow (Expense)**:
        - `is_salary` **MUST** be `IMPOSSIBLE`. Salary is always an inflow.
        - The transaction is likely a bill, so `is_bills` should be `LIKELY` or `UNLIKELY`.
        - `is_sidegig` could be `LIKELY` or `UNLIKELY` if it's a business expense for a side gig.
    - **If Inflow (Income)**:
        - `is_bills` **MUST** be `IMPOSSIBLE`. Bills are always outflows.
        - It could be a `salary` or `sidegig`.

### Likelihood Flexibility
- Sometimes, both `LIKELY` and `UNLIKELY` can be acceptable. For example, a subscription could be a personal bill (`UNLIKELY` for `is_sidegig`) or a business expense for a side gig (`LIKELY` for `is_sidegig`).
- **Only flag an issue if the output is `IMPOSSIBLE` when it should be `LIKELY` or `UNLIKELY`.**

### Likelihood Definitions
- `LIKELY`: High probability (e.g., >80% chance).
- `UNLIKELY`: Plausible but not highly probable (e.g., 20-80% chance).
- `IMPOSSIBLE`: Very low to no probability (e.g., <20% chance).
"""

class CheckSummarizedNamesRecurringLikelihood:
  """Handles all Gemini API interactions for checking SummarizedNamesRecurringLikelihood outputs against rules"""
  
  def __init__(self, model_name="gemini-flash-lite-latest"):
    """Initialize the Gemini agent with API configuration for checking SummarizedNamesRecurringLikelihood evaluations"""
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
    Generate a response using Gemini API for checking SummarizedNamesRecurringLikelihood outputs.
    
    Args:
      eval_input: A JSON array of transactions.
      past_review_outcomes: An array of past review outcomes, each containing `output`, `good_copy`, `info_correct`, and `eval_text`.
      review_needed: The SummarizedNamesRecurringLikelihood output that needs to be reviewed (JSON array).
      
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


def run_test_case(test_name: str, eval_input: list, review_needed: list, past_review_outcomes: list = None, checker: 'CheckSummarizedNamesRecurringLikelihood' = None):
  """
  Utility function to run a test case with custom inputs, common error handling, and output formatting.
  
  Args:
    test_name: Name of the test case
    eval_input: A list of transaction pairs to be evaluated.
    review_needed: The classifier output that needs to be reviewed (list of dicts).
    past_review_outcomes: An array of past review outcomes. Defaults to empty list.
    checker: Optional CheckSummarizedNamesRecurringLikelihood instance. If None, creates a new one.
    
  Returns:
    Dictionary with good_copy, info_correct, and eval_text keys, or None if error occurred
  """
  if past_review_outcomes is None:
    past_review_outcomes = []
  if checker is None:
    checker = CheckSummarizedNamesRecurringLikelihood()

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


def run_varied_correct_response_test(checker: CheckSummarizedNamesRecurringLikelihood = None):
  """
  Test a variety of transactions where the classifier's output is perfectly correct.
  """
  eval_input = [
      {"id": 1, "name": "Netflix", "description": "Monthly subscription for streaming service."},
      {"id": 2, "name": "PAYROLL-COMPANY-ABC", "description": "Direct deposit for salary."},
      {"id": 3, "name": "Upwork", "description": "Payment for freelance work."},
      {"id": 4, "name": "ConEd", "description": "Utility bill for electricity."},
      {"id": 5, "name": "Starbucks", "description": "One-time coffee purchase."}
  ]
  
  review_needed = [
      {"id": 1, "is_bills": "LIKELY", "is_salary": "IMPOSSIBLE", "is_sidegig": "UNLIKELY"},
      {"id": 2, "is_bills": "IMPOSSIBLE", "is_salary": "LIKELY", "is_sidegig": "UNLIKELY"},
      {"id": 3, "is_bills": "IMPOSSIBLE", "is_salary": "UNLIKELY", "is_sidegig": "LIKELY"},
      {"id": 4, "is_bills": "LIKELY", "is_salary": "IMPOSSIBLE", "is_sidegig": "IMPOSSIBLE"},
      {"id": 5, "is_bills": "IMPOSSIBLE", "is_salary": "IMPOSSIBLE", "is_sidegig": "IMPOSSIBLE"}
  ]
  
  return run_test_case("varied_correct_response_test", eval_input, review_needed, [], checker)

def run_partially_correct_recurring_test(checker: CheckSummarizedNamesRecurringLikelihood = None):
  """
  Test for a partially correct response where a recurring item is misclassified.
  """
  eval_input = [
      {"id": 1, "name": "Adobe Creative Cloud", "description": "Subscription for creative software, potentially a business expense."},
      {"id": 2, "name": "Starbucks", "description": "Coffee purchase, not a recurring bill."},
      {"id": 3, "name": "IRS", "description": "Tax payment, a recurring but not typically monthly bill."}
  ]
  
  review_needed = [
      {"id": 1, "is_bills": "LIKELY", "is_salary": "IMPOSSIBLE", "is_sidegig": "LIKELY"},
      {"id": 2, "is_bills": "LIKELY", "is_salary": "IMPOSSIBLE", "is_sidegig": "IMPOSSIBLE"},
      {"id": 3, "is_bills": "IMPOSSIBLE", "is_salary": "IMPOSSIBLE", "is_sidegig": "IMPOSSIBLE"}
  ]
  
  return run_test_case("partially_correct_recurring_test", eval_input, review_needed, [], checker)

def run_mixed_correctness_test(checker: CheckSummarizedNamesRecurringLikelihood = None):
  """
  Test a mix of correct, incorrect, and acceptable but not perfect classifications.
  """
  eval_input = [
      {"id": 1, "name": "Direct Deposit", "description": "Salary payment."},
      {"id": 2, "name": "Blue Apron", "description": "Meal kit delivery service."},
      {"id": 3, "name": "Amazon", "description": "Online retailer with optional subscriptions."}
  ]
  
  review_needed = [
      {"id": 1, "is_bills": "LIKELY", "is_salary": "IMPOSSIBLE", "is_sidegig": "IMPOSSIBLE"},
      {"id": 2, "is_bills": "LIKELY", "is_salary": "IMPOSSIBLE", "is_sidegig": "IMPOSSIBLE"},
      {"id": 3, "is_bills": "IMPOSSIBLE", "is_salary": "IMPOSSIBLE", "is_sidegig": "IMPOSSIBLE"}
  ]
  
  return run_test_case("mixed_correctness_test", eval_input, review_needed, [], checker)

def run_multiple_errors_test(checker: CheckSummarizedNamesRecurringLikelihood = None):
  """
  Test for multiple errors in the response.
  """
  eval_input = [
      {"id": 2, "name": "Starbucks", "description": "A global chain of coffeehouses."},
      {"id": 4, "name": "Stripe", "description": "An online payment processing platform for businesses."}
  ]
  
  review_needed = [
      {"id": 2, "is_bills": "LIKELY", "is_salary": "IMPOSSIBLE", "is_sidegig": "UNLIKELY"},
      {"id": 4, "is_bills": "IMPOSSIBLE", "is_salary": "IMPOSSIBLE", "is_sidegig": "LIKELY"}
  ]
  
  return run_test_case("multiple_errors_test", eval_input, review_needed, [], checker)

def run_good_copy_fail_test(checker: CheckSummarizedNamesRecurringLikelihood = None):
  """
  Test for a response that fails the good_copy check (e.g., missing fields).
  """
  eval_input = [
      {"id": 1, "name": "Netflix", "description": "An online streaming service for movies and TV shows."}
  ]
  
  review_needed = [
      {"id": 1, "is_bills": "LIKELY", "is_salary": "IMPOSSIBLE"} 
  ]
  
  return run_test_case("good_copy_fail_test", eval_input, review_needed, [], checker)


def run_ambiguous_transactions_test(checker: CheckSummarizedNamesRecurringLikelihood = None):
  """
  Test for ambiguous transactions where the classification is not clear-cut.
  """
  eval_input = [
    {"id": 1, "name": "Chase Bank", "description": "Financial institution with various services."},
    {"id": 2, "name": "Zelle to John", "description": "P2P transfer, could be for anything."},
    {"id": 3, "name": "Kaiser Permanente", "description": "Healthcare provider, could be a recurring bill or a one-off."}
  ]

  review_needed = [
    {"id": 1, "is_bills": "LIKELY", "is_salary": "IMPOSSIBLE", "is_sidegig": "UNLIKELY"},
    {"id": 2, "is_bills": "IMPOSSIBLE", "is_salary": "IMPOSSIBLE", "is_sidegig": "IMPOSSIBLE"},
    {"id": 3, "is_bills": "LIKELY", "is_salary": "IMPOSSIBLE", "is_sidegig": "IMPOSSIBLE"}
  ]

  return run_test_case("ambiguous_transactions_test", eval_input, review_needed, [], checker)



def main(batch: int = 0):
  """Main function to test the SummarizedNamesRecurringLikelihood checker"""
  checker = CheckSummarizedNamesRecurringLikelihood()
  
  if batch == 0:
    # Run all tests if no specific batch is selected
    run_varied_correct_response_test(checker)
    run_partially_correct_recurring_test(checker)
    run_mixed_correctness_test(checker)
    run_multiple_errors_test(checker)
    run_good_copy_fail_test(checker)
    run_ambiguous_transactions_test(checker)
  elif batch == 1:
    run_varied_correct_response_test(checker)
  elif batch == 2:
    run_partially_correct_recurring_test(checker)
  elif batch == 3:
    run_mixed_correctness_test(checker)
  elif batch == 4:
    run_multiple_errors_test(checker)
  elif batch == 5:
    run_good_copy_fail_test(checker)
  elif batch == 6:
    run_ambiguous_transactions_test(checker)
  else:
    print("Invalid batch number. Please choose from 0 to 6.")


if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser(description='Run checker tests in batches')
  parser.add_argument('--batch', type=int, default=0, choices=range(7),
                      help='Batch number to run (1-6). 0 runs all.')
  args = parser.parse_args()
  main(batch=args.batch)

from google import genai
from google.genai import types
import os
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Prompt optimization axes: (1) Output category = one of category_options or "unknown".
# (2) Debt, mortgage, or any other liability payment → transfer.
# (3) Category must be chosen rationally from establishment_name, establishment_description, transaction_text, and/or amount.
# (4) When only issue is wording → good_copy False, info_correct True. eval_text lines start with "Transaction <id>: ".
SYSTEM_PROMPT = """Audit RethinkTransactionCategorization output.

## Rules
- **Category Selection**: Must match `category_options` or be `unknown`. `unknown` is ALWAYS acceptable even if not listed.
- **P2P Transfers**: `unknown` if purpose is unspecified (e.g., "Zelle from Laura"). If purpose is specified (e.g., "Dinner"), refer to that.
- **Prohibited Categories**: NEVER use parent categories (food, leisure, bills, shelter, education, shopping, transport, health, income) as the output category.
- **Transfers**: Debt/Liability payments (Credit Card, Mortgage, Loan) are `Transfer`. Bank transfers are `Transfer` ONLY if to/from the same person; otherwise use `unknown`.
- **Business**: Business-related outflows (expenses) MUST be categorized as `income_business` if available. This is a MANDATORY mapping for this auditor, even if it sounds like an income category.
- **Acceptability**: Judge if the category is *acceptable/plausible* according to the provided options and rules, and not if it matches your own expectation or perception of the ideal. If a category is in `category_options` and is not a parent category, it is generally acceptable.

## Category Hierarchy (Reference)
- **Meals**: all sources of food from supermarkets to restaurants and food deliveries. Subcategories: `Dining Out`, `Delivered Food`, `Groceries`.
- **Leisure**: all relaxation or recreation activities. Subcategories: `Entertainment`, `Travel & Vacations`.
- **Bills**: essential practical services and recurring costs. Subcategories: `Connectivity`, `Insurance`, `Taxes`, `Service Fees`.
- **Shelter**: place of residence expenses. Subcategories: `Home` (rent/mortgage/debt), `Utilities`, `Upkeep`.
- **Education**: learning and development. Subcategories: `Kids Activities`, `Tuition`.
- **Shopping**: discretionary purchases. Subcategories: `Clothing`, `Gadgets`, `Kids`, `Pets`.
- **Transport**: moving from point A to B. Subcategories: `Public Transit`, `Car & Fuel` (gas/EV/repairs).
- **Health**: well-being. Subcategories: `Medical & Pharmacy`, `Gym & Wellness`, `Personal Care`.
- **Income**: money earned or returns. Subcategories: `Salary`, `Side-Gig`, `Business`, `Interest`.
- **Others**: `Donations & Gifts`, `Transfers`, `Miscellaneous`, `Uncategorized`.

## Verification
- **good_copy**: `true` if `transaction_id`, `reasoning`, `category`, and `confidence` are present for all input transactions.
- **info_correct**: `true` if the category choice is acceptable per the rules above.
- **eval_text**: "" if correct. Else, "Transaction <id>: <reason>" (max 25 words).

## Output (JSON Only)
{
  "good_copy": boolean,
  "info_correct": boolean,
  "eval_text": string
}

<EXAMPLES>
Input:
<EVAL_INPUT>
[{"establishment_name": "Zelle", "transactions": [{"transaction_id": 1, "transaction_text": "Zelle from James: Dinner"}], "category_options": ["food", "food_dining_out", "donations_gifts", "leisure"]}]
</EVAL_INPUT>
<PAST_REVIEW_OUTCOMES>[]</PAST_REVIEW_OUTCOMES>
<REVIEW_NEEDED>
[{"transaction_id": 1, "reasoning": "Dinner specified.", "category": "food_dining_out", "confidence": "high"}]
</REVIEW_NEEDED>
Output: {"good_copy": true, "info_correct": true, "eval_text": ""}

Input:
<EVAL_INPUT>
[{"establishment_name": "Zelle", "transactions": [{"transaction_id": 2, "transaction_text": "Zelle from Laura"}], "category_options": ["food", "food_dining_out", "donations_gifts", "leisure"]}]
</EVAL_INPUT>
<PAST_REVIEW_OUTCOMES>[]</PAST_REVIEW_OUTCOMES>
<REVIEW_NEEDED>
[{"transaction_id": 2, "reasoning": "P2P to Laura.", "category": "donations_gifts", "confidence": "medium"}]
</REVIEW_NEEDED>
Output: {"good_copy": true, "info_correct": false, "eval_text": "Transaction 2: P2P purpose unspecified; should be unknown."}
</EXAMPLES>
"""


class CheckRethinkTransactionCategorization:
  """Handles all Gemini API interactions for checking RethinkTransactionCategorization outputs against rules"""

  def __init__(self, model_name="gemini-flash-lite-latest"):
    """Initialize the Gemini agent with API configuration for checking categorization evaluations"""
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
      raise ValueError("GEMINI_API_KEY environment variable is not set. Please set it in your .env file or environment.")
    self.client = genai.Client(api_key=api_key)

    self.model_name = model_name
    self.top_k = 40
    self.top_p = 0.95
    self.temperature = 0.5
    self.thinking_budget = 1024
    self.max_output_tokens = 4096

    self.safety_settings = [
      types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"),
      types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"),
      types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"),
      types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="OFF")
    ]
    self.system_prompt = SYSTEM_PROMPT

  def generate_response(self, eval_input: list, past_review_outcomes: list, review_needed: list) -> dict:
    """
    Generate a response using Gemini API for checking RethinkTransactionCategorization outputs.

    Args:
      eval_input: A JSON array of transaction groups (establishment_name, establishment_description, transactions, category_options).
      past_review_outcomes: An array of past review outcomes, each containing `output`, `good_copy`, `info_correct`, and `eval_text`.
      review_needed: The categorizer output that needs to be reviewed (JSON array of result objects).

    Returns:
      Dictionary with good_copy, info_correct, and eval_text keys
    """
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
    contents = [types.Content(role="user", parts=[request_text])]

    generate_content_config = types.GenerateContentConfig(
      top_k=self.top_k,
      top_p=self.top_p,
      temperature=self.temperature,
      max_output_tokens=self.max_output_tokens,
      safety_settings=self.safety_settings,
      system_instruction=[types.Part.from_text(text=self.system_prompt)],
      thinking_config=types.ThinkingConfig(
        thinking_budget=self.thinking_budget,
        include_thoughts=True,
      ),
      response_mime_type="application/json",
    )

    output_text = ""
    thought_summary = ""

    for chunk in self.client.models.generate_content_stream(
      model=self.model_name,
      contents=contents,
      config=generate_content_config,
    ):
      if chunk.text is not None:
        output_text += chunk.text

      if hasattr(chunk, "candidates") and chunk.candidates:
        for candidate in chunk.candidates:
          if hasattr(candidate, "content") and candidate.content:
            if hasattr(candidate.content, "parts") and candidate.content.parts:
              for part in candidate.content.parts:
                if hasattr(part, "thought") and part.thought:
                  if hasattr(part, "text") and part.text:
                    if thought_summary:
                      thought_summary += part.text
                    else:
                      thought_summary = part.text

    if thought_summary:
      print(f"{'='*80}")
      print("THOUGHT SUMMARY:")
      print(thought_summary.strip())
      print("="*80)

    if not output_text or not output_text.strip():
      raise ValueError("Empty response from model. Check API key and model availability.")

    try:
      output_text = output_text.strip()
      if output_text.startswith("```"):
        lines = output_text.split("\n")
        json_lines = []
        in_code_block = False
        for line in lines:
          if line.strip().startswith("```"):
            in_code_block = not in_code_block
            continue
          if in_code_block or (not in_code_block and line.strip()):
            json_lines.append(line)
        output_text = "\n".join(json_lines).strip()
      elif "```json" in output_text:
        fence_start = output_text.find("```json") + 7
        fence_end = output_text.find("```", fence_start)
        if fence_end != -1:
          output_text = output_text[fence_start:fence_end].strip()
      elif "```" in output_text:
        fence_start = output_text.find("```") + 3
        fence_end = output_text.find("```", fence_start)
        if fence_end != -1:
          output_text = output_text[fence_start:fence_end].strip()

      return json.loads(output_text)
    except json.JSONDecodeError as e:
      # Without a response schema, the model may wrap the object in prose; try the outermost {...} slice.
      brace_start = output_text.find("{")
      brace_end = output_text.rfind("}")
      if brace_start != -1 and brace_end > brace_start:
        try:
          return json.loads(output_text[brace_start : brace_end + 1])
        except json.JSONDecodeError:
          pass
      raise ValueError(f"Failed to parse JSON response: {e}\nResponse length: {len(output_text)}\nResponse preview: {output_text[:500]}")


def run_test_case(test_name: str, eval_input: list, review_needed: list, past_review_outcomes: list = None, checker: 'CheckRethinkTransactionCategorization' = None):
  """
  Run a test case with custom inputs, common error handling, and output formatting.

  Args:
    test_name: Name of the test case
    eval_input: List of transaction groups (same format as categorizer input).
    review_needed: The categorizer output to be reviewed (list of dicts with transaction_id, reasoning, category, confidence).
    past_review_outcomes: Optional list of past review outcomes.
    checker: Optional CheckRethinkTransactionCategorization instance.

  Returns:
    Dictionary with good_copy, info_correct, and eval_text keys, or None on error.
  """
  if past_review_outcomes is None:
    past_review_outcomes = []
  if checker is None:
    checker = CheckRethinkTransactionCategorization()

  print(f"\n{'='*80}")
  print(f"Running test: {test_name}")
  print(f"{'='*80}")

  try:
    result = checker.generate_response(eval_input, past_review_outcomes, review_needed)
    print("Result:")
    print(json.dumps(result, indent=2))
    print(f"{'='*80}")
    return result
  except Exception as e:
    print(f"ERROR: {type(e).__name__}: {str(e)}")
    print(f"{'='*80}")
    return None


def run_correct_response(checker: CheckRethinkTransactionCategorization = None):
  """Run test for a correct categorization response."""
  eval_input = [
    {
      "establishment_name": "Macho's",
      "establishment_description": "sells Mexican food such as tacos, burritos, and quesadillas",
      "transactions": [
        {"transaction_id": 1, "transaction_text": "PY *MACHO SE REF# 509700022869 972-525-0686,TX", "amount": 40.00},
        {"transaction_id": 2, "transaction_text": "PY *MACHO SE REF# 506600014887 972-525-0686,TX", "amount": 40.00}
      ],
      "category_options": ["food", "food_dining_out", "leisure", "shopping", "food_groceries"]
    }
  ]
  review_needed = [
    {"transaction_id": 1, "reasoning": "Mexican restaurant purchase.", "category": "food_dining_out", "confidence": "high"},
    {"transaction_id": 2, "reasoning": "Mexican restaurant purchase.", "category": "food_dining_out", "confidence": "high"}
  ]
  return run_test_case("correct_response", eval_input, review_needed, [], checker)


def run_transfer_rule_correct(checker: CheckRethinkTransactionCategorization = None):
  """Run test for correct transfer categorization."""
  eval_input = [
    {
      "establishment_name": "Transfer",
      "establishment_description": "Internal transfer between accounts",
      "transactions": [
        {"transaction_id": 1, "transaction_text": "Transfer To Checking", "amount": 500.00}
      ],
      "category_options": ["transfer", "income", "bills", "bills_service_fees", "income_salary"]
    }
  ]
  review_needed = [
    {"transaction_id": 1, "reasoning": "Movement between same person's accounts.", "category": "transfer", "confidence": "high"}
  ]
  return run_test_case("transfer_rule_correct", eval_input, review_needed, [], checker)


def run_p2p_unknown_correct(checker: CheckRethinkTransactionCategorization = None):
  """Run test for P2P payment with unspecified purpose → unknown."""
  eval_input = [
    {
      "establishment_name": "Zelle",
      "establishment_description": "Peer-to-peer payment",
      "transactions": [
        {"transaction_id": 1, "transaction_text": "Zelle payment to John Smith", "amount": 50.00}
      ],
      "category_options": ["donations_gifts", "transfer", "bills", "bills_service_fees", "income"]
    }
  ]
  review_needed = [
    {"transaction_id": 1, "reasoning": "Peer-to-peer payment to another person, purpose unspecified.", "category": "unknown", "confidence": "high"}
  ]
  return run_test_case("p2p_unknown_correct", eval_input, review_needed, [], checker)


def run_subcategory_preference_correct(checker: CheckRethinkTransactionCategorization = None):
  """Run test for subcategory preference (must use leisure_entertainment not leisure)."""
  eval_input = [
    {
      "establishment_name": "City News Stand",
      "establishment_description": "Sells newspapers and magazines.",
      "transactions": [
        {"transaction_id": 7766, "transaction_text": "CITY NEWS NYC", "amount": 5.50}
      ],
      "category_options": ["leisure", "leisure_entertainment", "bills", "shopping", "shopping_clothing"]
    }
  ]
  review_needed = [
    {"transaction_id": 7766, "reasoning": "Newsstand purchase for magazines/newspapers.", "category": "leisure_entertainment", "confidence": "high"}
  ]
  return run_test_case("subcategory_preference_correct", eval_input, review_needed, [], checker)


def run_wrong_category_test(checker: CheckRethinkTransactionCategorization = None):
  """Run test for wrong category (P2P labeled as donations_gifts without evidence)."""
  eval_input = [
    {
      "establishment_name": "Venmo",
      "establishment_description": "P2P payment",
      "transactions": [
        {"transaction_id": 1, "transaction_text": "Venmo payment to Jane Doe", "amount": 25.00}
      ],
      "category_options": ["donations_gifts", "transfer", "income", "bills", "food"]
    }
  ]
  review_needed = [
    {"transaction_id": 1, "reasoning": "Payment to another person.", "category": "donations_gifts", "confidence": "medium"}
  ]
  return run_test_case("wrong_category_p2p", eval_input, review_needed, [], checker)


def run_forbidden_reasoning_test(checker: CheckRethinkTransactionCategorization = None):
  """Run test for forbidden reasoning phrases."""
  eval_input = [
    {
      "establishment_name": "Starbucks Coffee",
      "establishment_description": "Coffee shop and cafe.",
      "transactions": [
        {"transaction_id": 123456, "transaction_text": "POS DEBIT STARBUCKS STORE #1234 SEATTLE WA", "amount": 5.75}
      ],
      "category_options": ["food", "food_dining_out", "bills", "transport", "transport_car_fuel"]
    }
  ]
  review_needed = [
    {"transaction_id": 123456, "reasoning": "Dining out subcategory is the most specific match from the provided options.", "category": "food_dining_out", "confidence": "high"}
  ]
  return run_test_case("forbidden_reasoning", eval_input, review_needed, [], checker)


def run_chase_bank_mortgage_payment(checker: CheckRethinkTransactionCategorization = None):
  """Run test for Chase Bank mortgage payment (payment to own mortgage/debt at same bank → transfer)."""
  eval_input = [
    {
      "establishment_name": "Chase Bank Mortgage Payment",
      "establishment_description": "Payment to Chase for mortgage on own property; same-bank payment from checking to mortgage account.",
      "transactions": [
        {"transaction_id": 9001, "transaction_text": "CHASE BANK MORTGAGE PAYMENT", "amount": 1850.00}
      ],
      "category_options": ["transfer", "shelter", "shelter_home", "bills", "bills_service_fees"]
    }
  ]
  review_needed = [
    {"transaction_id": 9001, "reasoning": "Payment to own mortgage account at same bank; net worth unchanged.", "category": "transfer", "confidence": "high"}
  ]
  return run_test_case("chase_bank_mortgage_payment", eval_input, review_needed, [], checker)


def run_home_decor_brand(checker: CheckRethinkTransactionCategorization = None):
  """Run test for a home decor brand purchase (category_options exclude shelter_upkeep)."""
  eval_input = [
    {
      "establishment_name": "West Elm",
      "establishment_description": "Home decor and furniture brand selling furniture, rugs, and accessories.",
      "transactions": [
        {"transaction_id": 7001, "transaction_text": "WEST ELM ONLINE", "amount": 129.00}
      ],
      "category_options": ["shopping", "shopping_general", "leisure", "leisure_entertainment", "bills"]
    }
  ]
  review_needed = [
    {"transaction_id": 7001, "reasoning": "Home decor retailer purchase.", "category": "shopping_general", "confidence": "high"}
  ]
  return run_test_case("home_decor_brand", eval_input, review_needed, [], checker)


def run_qt_negative_small(checker: CheckRethinkTransactionCategorization = None):
  """Run test for QT transaction with negative small amount (e.g. refund/discount)."""
  eval_input = [
    {
      "establishment_name": "QT",
      "establishment_description": "QuikTrip gas station and convenience store.",
      "transactions": [
        {"transaction_id": 8001, "transaction_text": "QT 12345 DALLAS TX", "amount": -2.50}
      ],
      "category_options": ["income", "income_business", "transfer", "transport", "transport_car_fuel"]
    }
  ]
  review_needed = [
    {"transaction_id": 8001, "reasoning": "Small negative amount; likely fuel discount or refund.", "category": "income", "confidence": "medium"}
  ]
  return run_test_case("qt_negative_small", eval_input, review_needed, [], checker)


def run_shopping_grocery_wording(checker: CheckRethinkTransactionCategorization = None):
  """Run test: valid exact category string from category_options (shopping_groceries)."""
  eval_input = [
    {
      "establishment_name": "Kroger",
      "establishment_description": "Grocery store.",
      "transactions": [
        {"transaction_id": 6001, "transaction_text": "KROGER #123", "amount": 84.20}
      ],
      "category_options": ["shopping", "shopping_groceries", "food", "food_dining_out", "donations_gifts"]
    }
  ]
  review_needed = [
    {"transaction_id": 6001, "reasoning": "Grocery store purchase.", "category": "shopping_groceries", "confidence": "high"}
  ]
  return run_test_case("shopping_grocery_wording", eval_input, review_needed, [], checker)


def run_marketplace_unknown_correct(checker: CheckRethinkTransactionCategorization = None):
  """Run test for marketplace with unknown item → unknown."""
  eval_input = [
    {
      "establishment_name": "Shopee",
      "establishment_description": "Online marketplace for various products.",
      "transactions": [
        {"transaction_id": 1, "transaction_text": "SHOPEE PAYMENT", "amount": 35.00}
      ],
      "category_options": ["shopping", "shopping_clothing", "shopping_gadgets", "food", "food_delivered_food"]
    }
  ]
  review_needed = [
    {"transaction_id": 1, "reasoning": "Marketplace transaction; specific item unknown.", "category": "unknown", "confidence": "high"}
  ]
  return run_test_case("marketplace_unknown_correct", eval_input, review_needed, [], checker)


def run_p2p_with_purpose(checker: CheckRethinkTransactionCategorization = None):
  """Run test for P2P payment with purpose specified (Zelle from James: Dinner)."""
  eval_input = [
    {
      "establishment_name": "Zelle",
      "establishment_description": "Peer-to-peer payment",
      "transactions": [
        {"transaction_id": 101, "transaction_text": "Zelle from James: Dinner", "amount": 25.00}
      ],
      "category_options": ["food", "food_dining_out", "donations_gifts", "transfer", "income"]
    }
  ]
  review_needed = [
    {"transaction_id": 101, "reasoning": "P2P payment for dinner.", "category": "food_dining_out", "confidence": "high"}
  ]
  return run_test_case("p2p_with_purpose", eval_input, review_needed, [], checker)


def run_p2p_without_purpose(checker: CheckRethinkTransactionCategorization = None):
  """Run test for P2P payment without purpose specified (Zelle from Laura)."""
  eval_input = [
    {
      "establishment_name": "Zelle",
      "establishment_description": "Peer-to-peer payment",
      "transactions": [
        {"transaction_id": 102, "transaction_text": "Zelle from Laura", "amount": 50.00}
      ],
      "category_options": ["donations_gifts", "transfer", "income", "bills", "food"]
    }
  ]
  review_needed = [
    {"transaction_id": 102, "reasoning": "P2P payment to Laura.", "category": "donations_gifts", "confidence": "medium"}
  ]
  # This should be marked as incorrect (info_correct=False) because purpose is unspecified.
  return run_test_case("p2p_without_purpose", eval_input, review_needed, [], checker)


def run_parent_category_test(checker: CheckRethinkTransactionCategorization = None):
  """Run test for transaction with category options that are parent categories (food vs food_dining_out)."""
  eval_input = [
    {
      "establishment_name": "Starbucks",
      "establishment_description": "Coffee shop",
      "transactions": [
        {"transaction_id": 103, "transaction_text": "Starbucks Coffee", "amount": 5.00}
      ],
      "category_options": ["food", "food_dining_out", "bills", "shopping", "leisure"]
    }
  ]
  review_needed = [
    {"transaction_id": 103, "reasoning": "Coffee shop purchase.", "category": "food", "confidence": "high"}
  ]
  # This should be marked as incorrect (info_correct=False) because subcategory is available.
  return run_test_case("parent_category_test", eval_input, review_needed, [], checker)


def run_business_expense_test(checker: CheckRethinkTransactionCategorization = None):
  """Run test for business expense transactions (outflow categorized as income)."""
  eval_input = [
    {
      "establishment_name": "AWS",
      "establishment_description": "Cloud services",
      "transactions": [
        {"transaction_id": 104, "transaction_text": "AWS Service Fees", "amount": 100.00}
      ],
      "category_options": ["income", "income_business", "bills", "bills_service_fees", "shopping"]
    }
  ]
  review_needed = [
    {"transaction_id": 104, "reasoning": "Cloud service fees.", "category": "income_business", "confidence": "high"}
  ]
  # This should be marked as incorrect (info_correct=False) because outflow cannot be income.
  return run_test_case("business_expense_test", eval_input, review_needed, [], checker)


def run_category_not_in_options(checker: CheckRethinkTransactionCategorization = None):
  """Run test where output is a category not listed in category_options."""
  eval_input = [
    {
      "establishment_name": "Netflix",
      "establishment_description": "Streaming service",
      "transactions": [
        {"transaction_id": 105, "transaction_text": "Netflix Subscription", "amount": 15.00}
      ],
      "category_options": ["leisure", "leisure_entertainment", "bills", "bills_connectivity", "shopping"]
    }
  ]
  review_needed = [
    {"transaction_id": 105, "reasoning": "Streaming subscription.", "category": "subscriptions", "confidence": "high"}
  ]
  # This should be marked as incorrect (good_copy=False or info_correct=False) because 'subscriptions' is not in options.
  return run_test_case("category_not_in_options", eval_input, review_needed, [], checker)


def main(batch: int = 0):
  """Main function to test the RethinkTransactionCategorization checker."""
  checker = CheckRethinkTransactionCategorization()

  if batch == 0:
    run_correct_response(checker)
    run_transfer_rule_correct(checker)
    run_p2p_unknown_correct(checker)
    run_subcategory_preference_correct(checker)
    run_marketplace_unknown_correct(checker)
    run_wrong_category_test(checker)
    run_forbidden_reasoning_test(checker)
    run_chase_bank_mortgage_payment(checker)
    run_home_decor_brand(checker)
    run_qt_negative_small(checker)
    run_shopping_grocery_wording(checker)
    # New tests
    run_p2p_with_purpose(checker)
    run_p2p_without_purpose(checker)
    run_parent_category_test(checker)
    run_business_expense_test(checker)
    run_category_not_in_options(checker)
  elif batch == 1:
    run_p2p_with_purpose(checker)
  elif batch == 2:
    run_p2p_without_purpose(checker)
  elif batch == 3:
    run_parent_category_test(checker)
  elif batch == 4:
    run_business_expense_test(checker)
  elif batch == 5:
    run_category_not_in_options(checker)
  elif batch == 6:
    run_correct_response(checker)
  elif batch == 7:
    run_transfer_rule_correct(checker)
  elif batch == 8:
    run_p2p_unknown_correct(checker)
  elif batch == 9:
    run_subcategory_preference_correct(checker)
  elif batch == 10:
    run_marketplace_unknown_correct(checker)
  elif batch == 11:
    run_wrong_category_test(checker)
  else:
    print("Invalid batch number. Please choose from 0 to 11.")


if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser(description='Run RethinkTransactionCategorization checker tests in batches')
  parser.add_argument('--batch', type=int, default=0, choices=range(12),
                      help='Batch number to run (1-11). 0 runs all.')
  args = parser.parse_args()
  main(batch=args.batch)

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
SYSTEM_PROMPT = """You are a checker verifying the output of a transaction categorization model (RethinkTransactionCategorization).

## Input:
- **EVAL_INPUT**: A JSON array of transaction groups. Each group has `group_id`, `establishment_name`, `establishment_description`, `transactions` (array of `transaction_id`, `transaction_text`, `amount`), and `category_options`.
- **PAST_REVIEW_OUTCOMES**: An array of past review outcomes.
- **REVIEW_NEEDED**: The JSON output from the categorizer that needs to be reviewed (array of result objects). You judge accuracy of the **categories stated in REVIEW_NEEDED** only.

## Output:
Return valid JSON only. Put each top-level key on its own line (line break after each of good_copy, info_correct, eval_text). Example format:
```
{"good_copy": true,
"info_correct": true,
"eval_text": ""}
```

- `good_copy`: True if REVIEW_NEEDED is a valid JSON array and each item has the required fields: `group_id`, `transaction_id`, `reasoning`, `category`, `confidence`. Every `transaction_id` in REVIEW_NEEDED must exist in the corresponding group in EVAL_INPUT (matching by group_id). There must be exactly one output item per transaction in EVAL_INPUT.
- `info_correct`: True if the **`category`** (and adherence to critical rules) for each item in REVIEW_NEEDED is correct. **Ignore reasoning quality and confidence when judging**—only whether the chosen category and rule compliance are right. If the only issue is how the category is written (e.g. typo, casing, wording) but the semantic choice is correct, still set info_correct True.
- `eval_text`: **Empty string when good_copy and info_correct are both True.** Otherwise, eval_text must explain why REVIEW_NEEDED is incorrect. **Each line must start with "Transaction <transaction_id>: "** (e.g. "Transaction 387: P2P purpose unspecified; should be unknown."). One line per erroneous item (max 25 words per line), separate with newline (`\n`). Do not reference PAST_REVIEW_OUTCOMES.

## Critical Priority: Learn from PAST_REVIEW_OUTCOMES
Use PAST_REVIEW_OUTCOMES as a knowledge base. If REVIEW_NEEDED repeats mistakes flagged in past outcomes, mark incorrect. Do not mention past outcomes in `eval_text`.

## Verification Steps
1. Check PAST_REVIEW_OUTCOMES: if REVIEW_NEEDED repeats past mistakes → mark False.
2. Verify good_copy: valid JSON array; each item has group_id, transaction_id, reasoning, category, confidence; every transaction in EVAL_INPUT has exactly one matching output row (same group_id + transaction_id); category is one of the group's category_options or "unknown".
3. Verify info_correct: For each item, check **only** that the `category` in REVIEW_NEEDED is correct per the rules below. If the only flaw is superficial category wording (not wrong choice), info_correct stays True.
4. eval_text: **Only when REVIEW_NEEDED is incorrect.** Every line must start with "Transaction " then the exact transaction_id, then ": " (e.g. "Transaction 387: ...", "Transaction 123456: ..."). If REVIEW_NEEDED is correct, eval_text must be empty.

## Category and Rule Rules (apply to judge info_correct)
1. **Transfer rule**: `transfer` is for movements between the same person's accounts (net worth unchanged). This **includes**: moving money between checking/savings, **payments to own credit card or other debt accounts**, and **inflows** (e.g. transfer from savings to checking). "Transfer To Checking", "Payment to Credit Card", "ACH Transfer" between own accounts → transfer. "Zelle TO [Person]", "Venmo TO [Person]" → NOT transfer; purpose must be inferred or use `unknown`.
2. **P2P rule**: Zelle, Venmo, PayPal, Cash App to/from another person: if purpose is not specified or unclear → correct category is **unknown**. Do NOT mark correct if they used donations_gifts or another category without clear evidence.
3. **Marketplace rule**: Transactions from general-purpose marketplaces (Amazon, Shopee, Walmart, etc.) where the specific item is unknown → correct category is **unknown**, not a generic "shopping".
4. **Subcategory preference**: If both a parent (e.g. `leisure`) and a subcategory (e.g. `leisure_entertainment`) are in category_options, the output MUST use the subcategory. Using parent when subcategory is available is wrong.
5. **Category membership**: Judge the `category` value in REVIEW_NEEDED. It must be an exact match from that group's `category_options`, or `unknown`. **Closest-category rule**: If the ideal category is not in category_options, the model may choose the closest available option from the list instead of "unknown"; that is correct—do not mark info_correct False for picking a close category.

### What you verify in REVIEW_NEEDED
- **good_copy**: structure, required fields, one-to-one mapping of (group_id, transaction_id) to input transactions.
- **info_correct**: category and rule compliance for the categories in REVIEW_NEEDED only; ignore confidence and reasoning; do not mark false for category wording alone when choice is correct.
- `eval_text`: only when something is wrong; each line starts with "Transaction <id>: "; empty when correct.
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
    self.thinking_budget = 0
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
      eval_input: A JSON array of transaction groups (group_id, establishment_name, establishment_description, transactions, category_options).
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
      thinking_config=types.ThinkingConfig(thinking_budget=self.thinking_budget),
    )

    output_text = ""
    for chunk in self.client.models.generate_content_stream(
      model=self.model_name,
      contents=contents,
      config=generate_content_config,
    ):
      if chunk.text is not None:
        output_text += chunk.text

    if not output_text or not output_text.strip():
      raise ValueError("Empty response from model. Check API key and model availability.")

    try:
      if "```json" in output_text:
        json_start = output_text.find("```json") + 7
        json_end = output_text.find("```", json_start)
        if json_end != -1:
          output_text = output_text[json_start:json_end].strip()
      elif "```" in output_text:
        json_start = output_text.find("```") + 3
        json_end = output_text.find("```", json_start)
        if json_end != -1:
          output_text = output_text[json_start:json_end].strip()

      json_start = output_text.find('{')
      json_end = output_text.rfind('}') + 1

      if json_start != -1 and json_end > json_start:
        json_str = output_text[json_start:json_end]
        return json.loads(json_str)
      return json.loads(output_text.strip())
    except json.JSONDecodeError as e:
      raise ValueError(f"Failed to parse JSON response: {e}\nResponse length: {len(output_text)}\nResponse preview: {output_text[:500]}")


def run_test_case(test_name: str, eval_input: list, review_needed: list, past_review_outcomes: list = None, checker: 'CheckRethinkTransactionCategorization' = None):
  """
  Run a test case with custom inputs, common error handling, and output formatting.

  Args:
    test_name: Name of the test case
    eval_input: List of transaction groups (same format as categorizer input).
    review_needed: The categorizer output to be reviewed (list of dicts with group_id, transaction_id, reasoning, category, confidence).
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
      "group_id": "8809",
      "establishment_name": "Macho's",
      "establishment_description": "sells Mexican food such as tacos, burritos, and quesadillas",
      "transactions": [
        {"transaction_id": 1, "transaction_text": "PY *MACHO SE REF# 509700022869 972-525-0686,TX", "amount": 40.00},
        {"transaction_id": 2, "transaction_text": "PY *MACHO SE REF# 506600014887 972-525-0686,TX", "amount": 40.00}
      ],
      "category_options": ["food_dining_out", "food_delivered_food", "donations_gifts", "food", "income_business", "food_groceries", "leisure_travel_vacations", "leisure"]
    }
  ]
  review_needed = [
    {"group_id": "8809", "transaction_id": 1, "reasoning": "Mexican restaurant purchase.", "category": "food_dining_out", "confidence": "high"},
    {"group_id": "8809", "transaction_id": 2, "reasoning": "Mexican restaurant purchase.", "category": "food_dining_out", "confidence": "high"}
  ]
  return run_test_case("correct_response", eval_input, review_needed, [], checker)


def run_transfer_rule_correct(checker: CheckRethinkTransactionCategorization = None):
  """Run test for correct transfer categorization."""
  eval_input = [
    {
      "group_id": "8657",
      "establishment_name": "Transfer",
      "establishment_description": "Internal transfer between accounts",
      "transactions": [
        {"transaction_id": 1, "transaction_text": "Transfer To Checking", "amount": 500.00}
      ],
      "category_options": ["transfer", "donations_gifts", "bills_service_fees"]
    }
  ]
  review_needed = [
    {"group_id": "8657", "transaction_id": 1, "reasoning": "Movement between same person's accounts.", "category": "transfer", "confidence": "high"}
  ]
  return run_test_case("transfer_rule_correct", eval_input, review_needed, [], checker)


def run_p2p_unknown_correct(checker: CheckRethinkTransactionCategorization = None):
  """Run test for P2P payment with unspecified purpose → unknown."""
  eval_input = [
    {
      "group_id": "zelle-01",
      "establishment_name": "Zelle",
      "establishment_description": "Peer-to-peer payment",
      "transactions": [
        {"transaction_id": 1, "transaction_text": "Zelle payment to John Smith", "amount": 50.00}
      ],
      "category_options": ["donations_gifts", "transfer", "unknown", "bills_service_fees"]
    }
  ]
  review_needed = [
    {"group_id": "zelle-01", "transaction_id": 1, "reasoning": "Peer-to-peer payment to another person, purpose unspecified.", "category": "unknown", "confidence": "high"}
  ]
  return run_test_case("p2p_unknown_correct", eval_input, review_needed, [], checker)


def run_subcategory_preference_correct(checker: CheckRethinkTransactionCategorization = None):
  """Run test for subcategory preference (must use leisure_entertainment not leisure)."""
  eval_input = [
    {
      "group_id": "news-01",
      "establishment_name": "City News Stand",
      "establishment_description": "Sells newspapers and magazines.",
      "transactions": [
        {"transaction_id": 7766, "transaction_text": "CITY NEWS NYC", "amount": 5.50}
      ],
      "category_options": ["leisure", "leisure_entertainment", "bills"]
    }
  ]
  review_needed = [
    {"group_id": "news-01", "transaction_id": 7766, "reasoning": "Newsstand purchase for magazines/newspapers.", "category": "leisure_entertainment", "confidence": "high"}
  ]
  return run_test_case("subcategory_preference_correct", eval_input, review_needed, [], checker)


def run_wrong_category_test(checker: CheckRethinkTransactionCategorization = None):
  """Run test for wrong category (P2P labeled as donations_gifts without evidence)."""
  eval_input = [
    {
      "group_id": "venmo-01",
      "establishment_name": "Venmo",
      "establishment_description": "P2P payment",
      "transactions": [
        {"transaction_id": 1, "transaction_text": "Venmo payment to Jane Doe", "amount": 25.00}
      ],
      "category_options": ["donations_gifts", "unknown", "transfer"]
    }
  ]
  review_needed = [
    {"group_id": "venmo-01", "transaction_id": 1, "reasoning": "Payment to another person.", "category": "donations_gifts", "confidence": "medium"}
  ]
  return run_test_case("wrong_category_p2p", eval_input, review_needed, [], checker)


def run_forbidden_reasoning_test(checker: CheckRethinkTransactionCategorization = None):
  """Run test for forbidden reasoning phrases."""
  eval_input = [
    {
      "group_id": "starbucks-01",
      "establishment_name": "Starbucks Coffee",
      "establishment_description": "Coffee shop and cafe.",
      "transactions": [
        {"transaction_id": 123456, "transaction_text": "POS DEBIT STARBUCKS STORE #1234 SEATTLE WA", "amount": 5.75}
      ],
      "category_options": ["food_dining_out", "food_groceries", "bills", "transportation"]
    }
  ]
  review_needed = [
    {"group_id": "starbucks-01", "transaction_id": 123456, "reasoning": "Dining out subcategory is the most specific match from the provided options.", "category": "food_dining_out", "confidence": "high"}
  ]
  return run_test_case("forbidden_reasoning", eval_input, review_needed, [], checker)


def run_chase_bank_mortgage_payment(checker: CheckRethinkTransactionCategorization = None):
  """Run test for Chase Bank mortgage payment (payment to own mortgage/debt at same bank → transfer)."""
  eval_input = [
    {
      "group_id": "chase-mtg-01",
      "establishment_name": "Chase Bank Mortgage Payment",
      "establishment_description": "Payment to Chase for mortgage on own property; same-bank payment from checking to mortgage account.",
      "transactions": [
        {"transaction_id": 9001, "transaction_text": "CHASE BANK MORTGAGE PAYMENT", "amount": 1850.00}
      ],
      "category_options": ["transfer", "shelter_home", "bills_service_fees", "donations_gifts"]
    }
  ]
  review_needed = [
    {"group_id": "chase-mtg-01", "transaction_id": 9001, "reasoning": "Payment to own mortgage account at same bank; net worth unchanged.", "category": "transfer", "confidence": "high"}
  ]
  return run_test_case("chase_bank_mortgage_payment", eval_input, review_needed, [], checker)


def run_home_decor_brand(checker: CheckRethinkTransactionCategorization = None):
  """Run test for a home decor brand purchase (category_options exclude shelter_upkeep)."""
  eval_input = [
    {
      "group_id": "home-decor-01",
      "establishment_name": "West Elm",
      "establishment_description": "Home decor and furniture brand selling furniture, rugs, and accessories.",
      "transactions": [
        {"transaction_id": 7001, "transaction_text": "WEST ELM ONLINE", "amount": 129.00}
      ],
      "category_options": ["shopping_general", "shopping_groceries", "leisure_entertainment", "donations_gifts", "bills_service_fees"]
    }
  ]
  review_needed = [
    {"group_id": "home-decor-01", "transaction_id": 7001, "reasoning": "Home decor retailer purchase.", "category": "shopping_general", "confidence": "high"}
  ]
  return run_test_case("home_decor_brand", eval_input, review_needed, [], checker)


def run_qt_negative_small(checker: CheckRethinkTransactionCategorization = None):
  """Run test for QT transaction with negative small amount (e.g. refund/discount)."""
  eval_input = [
    {
      "group_id": "qt-01",
      "establishment_name": "QT",
      "establishment_description": "QuikTrip gas station and convenience store.",
      "transactions": [
        {"transaction_id": 8001, "transaction_text": "QT 12345 DALLAS TX", "amount": -2.50}
      ],
      "category_options": ["income_business", "income", "income_sidegig", "income_salary", "transfer", "transport_car_fuel", "meals_groceries"]
    }
  ]
  review_needed = [
    {"group_id": "qt-01", "transaction_id": 8001, "reasoning": "Small negative amount; likely fuel discount or refund.", "category": "income", "confidence": "medium"}
  ]
  return run_test_case("qt_negative_small", eval_input, review_needed, [], checker)


def run_shopping_grocery_wording(checker: CheckRethinkTransactionCategorization = None):
  """Run test: REVIEW_NEEDED has shopping_grocery; EVAL_INPUT has shopping_groceries in options. Wording difference only → info_correct true."""
  eval_input = [
    {
      "group_id": "grocery-01",
      "establishment_name": "Kroger",
      "establishment_description": "Grocery store.",
      "transactions": [
        {"transaction_id": 6001, "transaction_text": "KROGER #123", "amount": 84.20}
      ],
      "category_options": ["shopping_groceries", "shopping_general", "meals_dining_out", "donations_gifts"]
    }
  ]
  review_needed = [
    {"group_id": "grocery-01", "transaction_id": 6001, "reasoning": "Grocery store purchase.", "category": "shopping_grocery", "confidence": "high"}
  ]
  return run_test_case("shopping_grocery_wording", eval_input, review_needed, [], checker)


def run_marketplace_unknown_correct(checker: CheckRethinkTransactionCategorization = None):
  """Run test for marketplace with unknown item → unknown."""
  eval_input = [
    {
      "group_id": "shopee-01",
      "establishment_name": "Shopee",
      "establishment_description": "Online marketplace for various products.",
      "transactions": [
        {"transaction_id": 1, "transaction_text": "SHOPEE PAYMENT", "amount": 35.00}
      ],
      "category_options": ["shopping_clothing", "shopping_gadgets", "shopping", "unknown", "food_delivered_food"]
    }
  ]
  review_needed = [
    {"group_id": "shopee-01", "transaction_id": 1, "reasoning": "Marketplace transaction; specific item unknown.", "category": "unknown", "confidence": "high"}
  ]
  return run_test_case("marketplace_unknown_correct", eval_input, review_needed, [], checker)


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
  elif batch == 1:
    run_correct_response(checker)
  elif batch == 2:
    run_transfer_rule_correct(checker)
  elif batch == 3:
    run_p2p_unknown_correct(checker)
  elif batch == 4:
    run_subcategory_preference_correct(checker)
  elif batch == 5:
    run_marketplace_unknown_correct(checker)
  elif batch == 6:
    run_wrong_category_test(checker)
  elif batch == 7:
    run_forbidden_reasoning_test(checker)
  elif batch == 8:
    run_chase_bank_mortgage_payment(checker)
  elif batch == 9:
    run_home_decor_brand(checker)
  elif batch == 10:
    run_qt_negative_small(checker)
  elif batch == 11:
    run_shopping_grocery_wording(checker)
  else:
    print("Invalid batch number. Please choose from 0 to 11.")


if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser(description='Run RethinkTransactionCategorization checker tests in batches')
  parser.add_argument('--batch', type=int, default=0, choices=range(12),
                      help='Batch number to run (1-11). 0 runs all.')
  args = parser.parse_args()
  main(batch=args.batch)

from google import genai
from google.genai import types
import json
import os
from dotenv import load_dotenv


load_dotenv()


OUTPUT_SCHEMA = types.Schema(
  type=types.Type.OBJECT,
  required=["rules_satisfied", "notes"],
  properties={
    "rules_satisfied": types.Schema(
      type=types.Type.BOOLEAN,
      description="Boolean flag: true only if every filter can be evaluated and every transaction satisfies every filter."
    ),
    "notes": types.Schema(
      type=types.Type.STRING,
      description="Exactly one short sentence explaining why rules_satisfied is true or false.",
      min_length=8,
      max_length=140,
      pattern=r"^[^\n]+$",
    ),
  },
)


SYSTEM_PROMPT = """You are a deterministic verification agent for transaction categorization. Output JSON only.

Goal:
- `rules_satisfied` must exactly match your own evaluation result.
- `notes` must be one brief, direct, complete sentence.

Input:
- The user message has:
  1) **Categorize Request** (prose request text)
  2) **Transactions** (JSON array)
- Ignore requested category labels for verification. Extract only verifiable predicates:
  - `name_contains`, `name_eq`
  - `date_greater_than_or_equal_to`, `date_less_than_or_equal_to`, `date_eq`
  - `amount_greater_than_or_equal_to`, `amount_less_than_or_equal_to`, `amount_eq`
  - `account_id_eq`

Rule extraction:
1) If Categorize Request is empty/whitespace -> missing filters.
2) If request text is JSON and not an object (e.g. array) -> malformed.
3) If request clearly provides filters only as an embedded JSON array -> malformed.
4) Else infer one filter object `rules` from prose. If no checkable filters are extractable -> missing filters.

Decision algorithm (strict order):
1) Malformed input:
   - missing extractable filters OR malformed filter structure OR invalid transactions array.
   - Return `rules_satisfied: false`.
2) Cannot evaluate:
   - Any transaction is missing a field required by active rules.
   - Return `rules_satisfied: false`.
3) Violation:
   - Any transaction fails any active rule key.
   - Return `rules_satisfied: false`.
4) Satisfied:
   - Every transaction satisfies every active rule key.
   - Return `rules_satisfied: true`.

Matching semantics:
- `name_contains`: case-insensitive substring on `name`
- `name_eq`: case-insensitive exact match on `name`
- Date comparisons: lexicographic compare on ISO `YYYY-MM-DD`
- Amount comparisons: numeric compare; `amount_eq` is exact numeric equality
- If transactions array is empty: cannot verify -> `rules_satisfied: false`

Output:
- Exactly one JSON object with keys:
  - `rules_satisfied`: boolean
  - `notes`: one sentence, <= 140 chars, no newline
- No markdown, no code fences, no extra keys, no surrounding text.

Required notes for specific cases:
- Missing filters: "Malformed input: missing verifiable filters."
- Filters as array: "Malformed input: structured filters must be a JSON object, not an array."
- Missing required transaction field: "Cannot evaluate: a transaction is missing `<field>` required by the rules."
"""


def format_categorization_verify_input(categorize_request: str | None, transactions: list) -> str:
  """Build user message: **Categorize Request** prose + **Transactions** JSON array."""
  request_body = "" if categorize_request is None else categorize_request
  transactions_body = json.dumps(transactions, ensure_ascii=False, indent=2)
  return (
    "**Categorize Request**:\n\n"
    f"{request_body}\n\n"
    "**Transactions**:\n\n"
    f"{transactions_body}"
  )


TEST_CASES = [
  {
    "name": "all_rules_satisfied_simple",
    "input": format_categorization_verify_input(
      "Re-categorize all transactions from 'Starbucks' where each charge was $50 or less on or after 2025-11-01, setting their category to 'meals_dining_out'. Also, create a rule for future similar transactions.",
      [
        {
          "transaction_id": 1,
          "name": "Starbucks",
          "amount": 12.5,
          "date": "2025-11-10",
          "account_id": 20,
        },
        {
          "transaction_id": 2,
          "name": "Starbucks",
          "amount": 4.75,
          "date": "2025-11-12",
          "account_id": 20,
        },
      ],
    ),
    "output": {
      "rules_satisfied": True,
      "notes": "All transactions meet the Starbucks/≤$50/≥2025-11-01 rules.",
    },
  },
  {
    "name": "rule_violated_by_one_transaction",
    "input": format_categorization_verify_input(
      "Re-categorize all transactions from 'Costco' where each charge was at least $100, setting their category to 'groceries'. Also, create a rule for future similar transactions.",
      [
        {
          "transaction_id": 10,
          "name": "Costco",
          "amount": 90,
          "date": "2025-10-15",
          "account_id": 20,
        },
        {
          "transaction_id": 11,
          "name": "Costco",
          "amount": 220,
          "date": "2025-10-20",
          "account_id": 20,
        },
      ],
    ),
    "output": {
      "rules_satisfied": False,
      "notes": "Not all transactions satisfy the Costco/≥$100 rules.",
    },
  },
  {
    "name": "unevaluable_rule_due_to_missing_fields",
    "input": format_categorization_verify_input(
      "Re-categorize transactions as 'income_salary' where the merchant name includes 'payroll' and the amount is at least $1000 for every listed row (transaction_id 21 and 22).",
      [
        {
          "transaction_id": 21,
          "name": "CA State Payroll",
          "amount": 1440,
          "date": "2025-11-18",
          "account_id": 20,
        },
        {
          "transaction_id": 22,
          "name": "ACME Corp",
          "date": "2025-11-20",
          "account_id": 20,
        },
      ],
    ),
    "output": {
      "rules_satisfied": False,
      "notes": "Cannot evaluate: a transaction is missing `amount` required by the rules.",
    },
  },
  {
    "name": "malformed_input_missing_rules",
    "input": format_categorization_verify_input(
      None,
      [
        {
          "transaction_id": 31,
          "name": "Apartments LLC",
          "amount": 2000,
          "date": "2025-11-18",
          "account_id": 20,
        },
      ],
    ),
    "output": {
      "rules_satisfied": False,
      "notes": "Malformed input: missing verifiable filters.",
    },
  },
  {
    "name": "name_eq_requires_exact_match",
    "input": format_categorization_verify_input(
      "Categorize transaction_id 41 and transaction_id 42 as 'meals_dining_out'; for each row the merchant name must be exactly 'Starbucks' (not 'Starbucks Coffee' or any longer string).",
      [
        {"transaction_id": 41, "account_id": 20, "name": "Starbucks", "amount": 5.0, "date": "2025-11-05"},
        {"transaction_id": 42, "account_id": 20, "name": "Starbucks Coffee", "amount": 6.0, "date": "2025-11-06"},
      ],
    ),
    "output": {
      "rules_satisfied": False,
      "notes": "A transaction name is not an exact match for `name_eq`.",
    },
  },
  {
    "name": "date_eq_enforced",
    "input": format_categorization_verify_input(
      "Re-categorize all transactions from 'Whole Foods' as 'groceries' where the transaction date is exactly 2025-11-26 for each matching charge.",
      [
        {"transaction_id": 51, "account_id": 20, "name": "Whole Foods", "amount": 45.0, "date": "2025-11-26"},
        {"transaction_id": 52, "account_id": 20, "name": "Whole Foods", "amount": 12.0, "date": "2025-11-25"},
      ],
    ),
    "output": {
      "rules_satisfied": False,
      "notes": "A matching merchant transaction is not on `date_eq`.",
    },
  },
  {
    "name": "amount_eq_enforced",
    "input": format_categorization_verify_input(
      "Categorize the Netflix subscription charges (transaction_id 61 and 62) as 'subscriptions_entertainment' where each amount is exactly $15.99.",
      [
        {"transaction_id": 61, "account_id": 20, "name": "Netflix", "amount": 15.99, "date": "2025-11-03"},
        {"transaction_id": 62, "account_id": 20, "name": "Netflix", "amount": 16.99, "date": "2025-12-03"},
      ],
    ),
    "output": {
      "rules_satisfied": False,
      "notes": "A matching merchant transaction does not meet `amount_eq`.",
    },
  },
  {
    "name": "account_id_eq_enforced",
    "input": format_categorization_verify_input(
      "Re-categorize all transactions from 'Amazon' as 'shopping_online' where each charge posts to account_id 20 only. Also, create a rule for future similar transactions.",
      [
        {"transaction_id": 71, "name": "AMAZON", "amount": 25.0, "date": "2025-11-02", "account_id": 20},
        {"transaction_id": 72, "name": "Amazon", "amount": 30.0, "date": "2025-11-04", "account_id": 21},
      ],
    ),
    "output": {
      "rules_satisfied": False,
      "notes": "A matching merchant transaction is from a different account than `account_id_eq`.",
    },
  },
  {
    "name": "case_insensitive_name_contains",
    "input": format_categorization_verify_input(
      "Re-categorize the Amazon marketplace transactions (transaction_id 81 and 82) as 'shopping_online'; the merchant name should contain 'amazon' and matching must be case-insensitive.",
      [
        {"transaction_id": 81, "account_id": 20, "name": "AMAZON MARKETPLACE", "amount": 12.0, "date": "2025-11-01"},
        {"transaction_id": 82, "account_id": 20, "name": "Amazon.com", "amount": 9.0, "date": "2025-11-02"},
      ],
    ),
    "output": {
      "rules_satisfied": True,
      "notes": "`name_contains` is case-insensitive.",
    },
  },
  {
    "name": "date_range_lte_gte_combined",
    "input": format_categorization_verify_input(
      "Re-categorize all 'Shell' fuel transactions as 'transport_gas' where the charge date falls from 2025-11-01 through 2025-11-30 inclusive.",
      [
        {"transaction_id": 91, "account_id": 20, "name": "Shell", "amount": 40.0, "date": "2025-11-15"},
        {"transaction_id": 92, "account_id": 20, "name": "Shell", "amount": 35.0, "date": "2025-12-01"},
      ],
    ),
    "output": {
      "rules_satisfied": False,
      "notes": "A matching merchant transaction falls outside the date range.",
    },
  },
  {
    "name": "amount_range_lte_gte_combined",
    "input": format_categorization_verify_input(
      "Re-categorize all 'Costco' transactions as 'groceries' where each charge amount is between $50 and $200 inclusive.",
      [
        {"transaction_id": 101, "account_id": 20, "name": "Costco", "amount": 75.0, "date": "2025-11-08"},
        {"transaction_id": 102, "account_id": 20, "name": "Costco", "amount": 250.0, "date": "2025-11-09"},
      ],
    ),
    "output": {
      "rules_satisfied": False,
      "notes": "A matching merchant transaction falls outside the amount range.",
    },
  },
  {
    "name": "malformed_rules_not_dict",
    "input": format_categorization_verify_input(
      "Categorize the Starbucks transaction (transaction_id 111) as 'meals_dining_out'. Technical rules payload is a JSON array only: [\"name_contains: starbucks\"]",
      [{"transaction_id": 111, "account_id": 20, "name": "Starbucks", "amount": 5.0, "date": "2025-11-01"}],
    ),
    "output": {
      "rules_satisfied": False,
      "notes": "Malformed input: structured filters must be a JSON object, not an array.",
    },
  },
  {
    "name": "empty_transactions_array",
    "input": format_categorization_verify_input(
      "Re-categorize all 'Lyft' charges as 'transport_rideshare' where each amount is at least $10. Also, create a rule for future similar transactions.",
      [],
    ),
    "output": {
      "rules_satisfied": False,
      "notes": "Transactions array is empty; cannot verify filters.",
    },
  },
  {
    "name": "missing_name_required_by_rules",
    "input": format_categorization_verify_input(
      "Re-categorize these trips as 'transport_rideshare' where the merchant name contains 'Uber' and the date is on or after 2025-11-01.",
      [
        {
          "transaction_id": 201,
          "account_id": 20,
          "name": "Uber *TRIP",
          "amount": 18.0,
          "date": "2025-11-10",
        },
        {
          "transaction_id": 202,
          "account_id": 20,
          "amount": 22.0,
          "date": "2025-11-12",
        },
      ],
    ),
    "output": {
      "rules_satisfied": False,
      "notes": "Cannot evaluate: a transaction is missing `name` required by the rules.",
    },
  },
  {
    "name": "missing_account_id_required_by_rules",
    "input": format_categorization_verify_input(
      "Re-categorize these housing payments as 'shelter_home' where the merchant contains 'RentCo' and account_id is exactly 99.",
      [
        {
          "transaction_id": 301,
          "account_id": 99,
          "name": "RentCo Apartments",
          "amount": 2100.0,
          "date": "2025-11-01",
        },
        {
          "transaction_id": 302,
          "name": "RentCo Apartments",
          "amount": 2100.0,
          "date": "2025-12-01",
        },
      ],
    ),
    "output": {
      "rules_satisfied": False,
      "notes": "Cannot evaluate: a transaction is missing `account_id` required by the rules.",
    },
  },
]


BATCHES: dict[int, dict[str, object]] = {
  1: {
    "name": "Core success/failure + missing fields",
    "tests": [0, 1, 2, 3],
  },
  2: {
    "name": "Empty transactions + additional missing fields",
    "tests": [2, 12, 13, 14],
  },
  3: {
    "name": "Ranges + case-insensitivity + malformed type",
    "tests": [8, 9, 10, 11],
  },
  4: {
    "name": "Exact/equality keys",
    "tests": [4, 5, 6, 7],
  },
}


class UpdateTransactionCategoryVerifyOptimizer:
  """Handles Gemini API interactions for verifying rule matches."""

  def __init__(self, model_name: str = "gemini-flash-lite-latest"):
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
      raise ValueError("GEMINI_API_KEY environment variable is not set.")
    self.client = genai.Client(api_key=api_key)
    self.model_name = model_name or "gemini-flash-lite-latest"
    # Minimal-cost deterministic settings for short JSON verdicts.
    self.thinking_budget = 8
    self.temperature = 0.0
    self.top_p = 0.1
    self.max_output_tokens = 160
    self.safety_settings = [
      types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"),
      types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"),
      types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"),
      types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="OFF"),
    ]
    self.system_prompt = SYSTEM_PROMPT

  def generate_response(self, user_message: str) -> str:
    request_text = types.Part.from_text(text=user_message)
    contents = [types.Content(role="user", parts=[request_text])]
    config = types.GenerateContentConfig(
      temperature=self.temperature,
      top_p=self.top_p,
      max_output_tokens=self.max_output_tokens,
      safety_settings=self.safety_settings,
      system_instruction=[types.Part.from_text(text=self.system_prompt)],
      thinking_config=types.ThinkingConfig(thinking_budget=self.thinking_budget),
      response_mime_type="application/json",
      response_schema=OUTPUT_SCHEMA,
    )
    output_text = ""
    for chunk in self.client.models.generate_content_stream(
      model=self.model_name,
      contents=contents,
      config=config,
    ):
      if chunk.text is not None:
        output_text += chunk.text
    return output_text


def get_test_case(test_name_or_index):
  """
  Get a test case by name or index.
  
  Args:
    test_name_or_index: Test case name (str) or index (int)
    
  Returns:
    Test case dict or None if not found
  """
  if isinstance(test_name_or_index, int):
    if 0 <= test_name_or_index < len(TEST_CASES):
      return TEST_CASES[test_name_or_index]
    return None
  if isinstance(test_name_or_index, str):
    for test_case in TEST_CASES:
      if test_case["name"] == test_name_or_index:
        return test_case
  return None


def _run_test_with_logging(tc: dict, optimizer: UpdateTransactionCategoryVerifyOptimizer | None = None):
  if optimizer is None:
    optimizer = UpdateTransactionCategoryVerifyOptimizer()

  print("\n" + "=" * 80)
  print(f"Running verification test: {tc['name']}")
  print("=" * 80)
  print("\nLLM INPUT:")
  print(tc["input"])

  result = optimizer.generate_response(tc["input"])

  print("\nLLM OUTPUT:")
  print(result)
  print("\nEXPECTED OUTPUT (compact):")
  print(json.dumps(tc["output"], indent=2))
  print("\n" + "=" * 80 + "\n")
  return result


def run_test(test_name_or_index_or_dict, optimizer: UpdateTransactionCategoryVerifyOptimizer | None = None):
  if isinstance(test_name_or_index_or_dict, dict):
    tc = test_name_or_index_or_dict
  else:
    tc = get_test_case(test_name_or_index_or_dict)

  if tc is None:
    print(f"Test case '{test_name_or_index_or_dict}' not found.")
    return None

  return _run_test_with_logging(tc, optimizer)


def run_tests(test_names_or_indices=None, optimizer: UpdateTransactionCategoryVerifyOptimizer | None = None):
  """
  Run multiple tests by names or indices.
  
  Args:
    test_names_or_indices: list of names/indices, or None to run all tests
  """
  if test_names_or_indices is None:
    test_names_or_indices = list(range(len(TEST_CASES)))
  results = []
  for item in test_names_or_indices:
    results.append(run_test(item, optimizer=optimizer))
  return results


def main(test: str = None, batch: int | None = None, model: str | None = None):
  """
  CLI entrypoint to inspect verification test cases.
  
  Args:
    test: optional test name or index (as string); if provided, runs only that test.
    batch: optional batch number; runs 4 tests in that batch.
  """
  if batch is not None:
    if batch not in BATCHES:
      print(f"Invalid batch number: {batch}. Available batches: {sorted(BATCHES.keys())}")
      print("\nBatch descriptions:")
      for b, info in BATCHES.items():
        test_names = [TEST_CASES[idx]["name"] for idx in info["tests"]]
        print(f"  Batch {b}: {info['name']} — {', '.join(test_names)}")
      return
    info = BATCHES[batch]
    print(f"\nRunning batch {batch}: {info['name']}\n")
    optimizer = UpdateTransactionCategoryVerifyOptimizer(model_name=model or "gemini-flash-lite-latest")
    run_tests(test_names_or_indices=info["tests"], optimizer=optimizer)
    return
  
  if test is not None:
    optimizer = UpdateTransactionCategoryVerifyOptimizer(model_name=model or "gemini-flash-lite-latest")
    # Accept numeric index via string
    test_key: int | str
    if test.isdigit():
      test_key = int(test)
    else:
      test_key = test
    run_test(test_key, optimizer=optimizer)
    return
  
  # No args: list available tests
  print("Available verification test cases:")
  for i, tc in enumerate(TEST_CASES):
    print(f"  {i}: {tc['name']}")


if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser(
    description="Inspect update_transaction_category verification SYSTEM_PROMPT test cases."
  )
  parser.add_argument(
    "--test",
    type=str,
    help="Test name or index to run (e.g., 'all_rules_satisfied_simple' or '0').",
  )
  parser.add_argument(
    "--batch",
    type=int,
    nargs="?",
    const=1,
    default=None,
    metavar="N",
    help="Run batch N of tests (4 cases per batch).",
  )
  parser.add_argument(
    "--model",
    type=str,
    default=None,
    help="Gemini model name (e.g. gemini-2.0-flash).",
  )
  args = parser.parse_args()
  main(test=args.test, batch=args.batch, model=args.model)
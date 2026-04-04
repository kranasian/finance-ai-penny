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
      description=(
        "Set first; `notes` must agree: true only with an all-clear message; false only with a failure—same verdict, no mixed signals. "
        "True iff: ≥1 bullet; non-empty request with mappable rules (never infer from lines if request blank); "
        "structured rules use one {...} block not [...]-only; each line shows payee, date, amount or their sentinels; "
        "no transaction ids on lines; if request names account (account_id, account id, posts to account), enforce (Account…) and Account not given = missing; "
        "every line passes every mapped rule. False if unsure."
      ),
    ),
    "notes": types.Schema(
      type=types.Type.STRING,
      description=(
        "One line, 8–140 chars; must match boolean. Target ≤100 chars. "
        "Use $ and digits for money and dates like the request (e.g. $16.99, 2025-11-26)—never spell amounts or years in words. "
        "State the fact only; never say categorized, processed, or successfully unless quoting the request. "
        "Plain English; no snake_case or filter names; say at most, at least, exactly—not comparison symbols; no JSON/schema talk."
      ),
      min_length=8,
      max_length=140,
      pattern=r"^[^\n]+$",
    ),
  },
)


SYSTEM_PROMPT = """Return only the JSON object. Set `rules_satisfied` then `notes`; they must agree (true ↔ success wording, false ↔ one failure reason).

Ignore category labels when interpreting rules.

First failing gate → false with a matching note:
A) Blank request → false; never infer rules from `# Transactions` alone.
B) No bullets, or only “(+N more transactions.)” → false.
C) No mappable payee/amount/date rules (treat account as required only if text says account_id, account id, or posts to account) → false.
D) Structured rules only in [...] without a {...} block → false.
E) Each bullet (not the +more line): apply all mapped rules—contains vs exact name (case-insensitive; exact = full payee); each line’s date and amount satisfy the request’s rules. If account is in scope, (Account not given) is missing; if not in scope, ignore absent Account tails.
F) Sentinels such as Amount not given block any check that needs that value → false.

If every line passes → true + short success note.

Notes: one sentence, fact-only; use $ and digits as in the request (no spelled-out dollars or years). Aim ≤100 chars. Never say categorized, processed, or successfully. No snake_case; say at most, at least, exactly—not symbols."""


TEST_CASES = [
  {
    "name": "all_rules_satisfied_simple",
    "input": """# Categorize Request

Re-categorize all transactions from 'Starbucks' where each charge was $50 or less on or after 2025-11-01, setting their category to 'meals_dining_out'.

# Transactions

- $12.50 Starbucks on 2025-11-10.
- $4.75 Starbucks on 2025-11-12.""",
    "output": {
      "rules_satisfied": True,
      "notes": "Every Starbucks transaction and charge shown meets the $50 cap and 2025-11-01 date cutoff.",
    },
  },
  {
    "name": "rule_violated_by_one_transaction",
    "input": """# Categorize Request

Re-categorize all transactions from 'Costco' where each charge was at least $100, setting their category to 'groceries'.

# Transactions

- $25.00 AMAZON on 2025-10-25.""",
    "output": {
      "rules_satisfied": False,
      "notes": "Not every Costco transaction has a charge of at least $100 as the groceries rule requires.",
    },
  },
  {
    "name": "unevaluable_rule_due_to_missing_fields",
    "input": """# Categorize Request

Re-categorize transactions as 'income_salary' where the name includes 'payroll' and the amount is at least $1000 for every listed row.

# Transactions

- $1440.00 CA State Payroll transaction on 2025-11-18.
- Amount not given, ACME Corp transaction on November 20, 2025.""",
    "output": {
      "rules_satisfied": False,
      "notes": "Cannot evaluate: one transaction has no amount for the payroll and at-least-$1000 charge rules.",
    },
  },
  {
    "name": "name_eq_requires_exact_match",
    "input": """# Categorize Request

Categorize these rows as 'meals_dining_out'; for each row the name must be exactly 'Starbucks' (not 'Starbucks Coffee' or any longer string).

# Transactions

- $5.00 Starbucks on 2025-11-05.
- $6.00 Starbucks Coffee transaction on November 6, 2025.""",
    "output": {
      "rules_satisfied": False,
      "notes": "One row fails: name is Starbucks Coffee, not exactly Starbucks (no longer strings).",
    },
  },
  {
    "name": "date_eq_enforced",
    "input": """# Categorize Request

Re-categorize all transactions from 'Whole Foods' as 'groceries' where the transaction date is exactly 2025-11-26 for each matching charge.

# Transactions

- $45.00 Whole Foods on 2025-11-26.
- $12.00 Whole Foods on 2025-11-25.""",
    "output": {
      "rules_satisfied": False,
      "notes": "A matching Whole Foods charge is not on transaction date 2025-11-26 as required.",
    },
  },
  {
    "name": "amount_eq_enforced",
    "input": """# Categorize Request

Categorize the Netflix subscription charges as 'subscriptions_entertainment' where each amount is exactly $15.99.

# Transactions

- $15.99 Netflix on 2025-11-03.
- $16.99 Netflix on 2025-12-03.""",
    "output": {
      "rules_satisfied": False,
      "notes": "One Netflix charge is $16.99, not exactly $15.99 as required for every row.",
    },
  },
  {
    "name": "account_id_eq_enforced",
    "input": """# Categorize Request

Re-categorize all transactions from 'Amazon' as 'shopping_clothing' where each charge posts to account_id 20 only. Also, create a rule for future similar transactions.

# Transactions

- $25.00 AMAZON transaction on November 2, 2025 (Account 20).
- $30.00 Amazon transaction on November 4, 2025 (Account 21).""",
    "output": {
      "rules_satisfied": False,
      "notes": "One Amazon charge posts to an account other than 20, conflicting with account_id equal to 20.",
    },
  },
  {
    "name": "case_insensitive_name_contains",
    "input": """# Categorize Request

Re-categorize the Amazon marketplace transactions as 'shopping_clothing'; the name should contain 'amazon' and matching must be case-insensitive.

# Transactions

- $12.00 AMAZON MARKETPLACE on 2025-11-01.
- $9.00 Amazon.com on 2025-11-02.""",
    "output": {
      "rules_satisfied": True,
      "notes": "Both marketplace transactions meet case-insensitive name matching and the other rules.",
    },
  },
  {
    "name": "date_range_lte_gte_combined",
    "input": """# Categorize Request

Re-categorize all 'Shell' fuel transactions as 'transport_car_fuel' where the charge date falls from 2025-11-01 through 2025-11-30 inclusive.

# Transactions

- $40.00 Shell on 2025-11-15.
- $35.00 Shell on 2025-12-01.""",
    "output": {
      "rules_satisfied": False,
      "notes": "One Shell fuel charge has a charge date outside 2025-11-01 through 2025-11-30.",
    },
  },
  {
    "name": "amount_range_lte_gte_combined",
    "input": """# Categorize Request

Re-categorize all 'Costco' transactions as 'groceries' where each charge amount is between $50 and $200 inclusive.

# Transactions

- $75.00 Costco on 2025-11-08.
- $250.00 Costco on 2025-11-09.""",
    "output": {
      "rules_satisfied": False,
      "notes": "One Costco charge amount is outside the $50–$200 inclusive groceries range.",
    },
  },
  {
    "name": "empty_transactions_array",
    "input": """# Categorize Request

Re-categorize all 'Lyft' charges as 'transport_public_transit' where each amount is at least $10. Also, create a rule for future similar transactions.

# Transactions

None""",
    "output": {
      "rules_satisfied": False,
      "notes": "No Lyft charges were listed, so the at-least-$10 transport rules cannot be verified.",
    },
  },
  {
    "name": "missing_account_id_required_by_rules",
    "input": """# Categorize Request

Re-categorize these housing payments as 'shelter_home' where the name contains 'RentCo' and account_id is exactly 99.

# Transactions

- $2100.00 RentCo Apartments transaction on 2025-11-01.""",
    "output": {
      "rules_satisfied": False,
      "notes": "Cannot evaluate: housing payment has no account though the rules require account 99.",
    },
  },
  {
    "name": "truncated_transactions_more_than_three",
    "input": """# Categorize Request

Re-categorize all transactions from 'Starbucks' where each charge was $50 or less on or after 2025-11-01, setting their category to 'meals_dining_out'.

# Transactions

- $10.00 Starbucks on 2025-11-05.
- $20.00 Starbucks on 2025-11-06.
- $15.00 Starbucks on 2025-11-07.
(+2 more transactions.)""",
    "output": {
      "rules_satisfied": True,
      "notes": "Each listed Starbucks transaction and charge meets the $50 cap and 2025-11-01 cutoff.",
    },
  },
  {
    "name": "name_may_contain_pipe_and_tab",
    "input": """# Categorize Request

Re-categorize transactions where the name contains 'ACME' and each charge is at most $100.

# Transactions

- $55.00 ACME|UK Retail transaction on 2025-08-01.""",
    "output": {
      "rules_satisfied": True,
      "notes": "This transaction satisfies name contains ACME and each charge at most $100.",
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
    "tests": [2, 10, 11],
  },
  3: {
    "name": "Ranges + case-insensitivity",
    "tests": [7, 8, 9],
  },
  4: {
    "name": "Exact/equality keys",
    "tests": [4, 5, 6, 7],
  },
  5: {
    "name": "Truncated transactions (>3 shown as compact + more note)",
    "tests": [12, 13],
  },
}


class UpdateTransactionCategoryVerifyOptimizer:
  """Gemini verify client.

  Last live check (14 `TEST_CASES`, all batch indices): `gemini-flash-lite-latest`, tokens 128, thinking 0 — 0 golden
  mismatches on `rules_satisfied`; notes stayed ≤100 chars without spelled-out money/years or banned filler words.

  Defaults: `gemini-flash-lite-latest`, `thinking_budget=0`, `temperature=0`, `top_p=1`, `max_output_tokens=128`.
  If JSON truncates, try `max_output_tokens=160`; if verdicts drift, try `gemini-2.0-flash`.
  """

  def __init__(self, model_name: str = "gemini-flash-lite-latest"):
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
      raise ValueError("GEMINI_API_KEY environment variable is not set.")
    self.client = genai.Client(api_key=api_key)
    self.model_name = model_name or "gemini-flash-lite-latest"
    self.thinking_budget = 0
    self.temperature = 0.0
    self.top_p = 1.0
    self.max_output_tokens = 128
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
    batch: optional batch number; runs the tests listed for that batch in BATCHES.
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


# python experiments/check_before_recategorize_optimizer.py --test 0
# python experiments/check_before_recategorize_optimizer.py --batch 1
# python experiments/check_before_recategorize_optimizer.py --test all_rules_satisfied_simple
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
    help="Run batch N of tests (see BATCHES in script).",
  )
  parser.add_argument(
    "--model",
    type=str,
    default=None,
    help="Gemini model name (e.g. gemini-2.0-flash).",
  )
  args = parser.parse_args()
  main(test=args.test, batch=args.batch, model=args.model)
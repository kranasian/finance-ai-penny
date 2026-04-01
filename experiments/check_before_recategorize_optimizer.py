from datetime import datetime

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
        "true iff: ≥1 parseable line; # Categorize Request has non-blank prose with constraints you can map (never infer rules from transactions if request text is empty); "
        "if the request shows structured rules, they must be one {...} block—not rules given only as a [...] list; "
        "each line supplies fields active checks need (sentinels Amount not given, (payee not given), date not given, Account/ID not given count as missing); "
        "every line passes every check. Else false. Unsure → false. Set boolean before notes."
      ),
    ),
    "notes": types.Schema(
      type=types.Type.STRING,
      description=(
        "Single line 8–140 chars, matches rules_satisfied tone. Plain English like # Categorize Request—no snake_case, no filter API names, no code operators (≤ ≥). "
        "True: short success. False: one blocker (which row, what failed); only cite fields actually missing. Bad request shape: say invalid/malformed categorize request in plain words, not “JSON/array/object.”"
      ),
      min_length=8,
      max_length=140,
      pattern=r"^[^\n]+$",
    ),
  },
)


SYSTEM_PROMPT = """Verify `# Transactions` against `# Categorize Request`. JSON only.

Order: verdict → `rules_satisfied` → one-line `notes` (same outcome); re-read both.

Request: Ignore category slugs. If the categorize-request prose is empty or whitespace, false—never invent rules from transaction lines alone. Otherwise need mappable constraints on payee, amount, date, account, or transaction id; if nothing mappable → false. Rules only as `[...]` not one `{...}` block → false (invalid categorize request).

`# Transactions` is `None` or has zero parseable lines → false.

Row shape: `- $5.00 Starbucks transaction on November 5, 2025 (ID 41, Account 20).` (`$` or Amount not given; payee or (payee not given); on Month D, YYYY or date not given; ID/Account or not given). `(+N more transactions.)` is not a row.

Every row passes every check (not just some rows). Amounts/dates numeric; spelled dates = ISO day; name substring/exact: case-insensitive, exact = full payee. Reason with internal filter names if needed—never put snake_case API names into `notes`.

Notes: Plain English like the request, ~120 chars. Good: “Name contains ACME and the charge is at most $100.” Bad: name_contains or “amount <= $100.” Bad shape: invalid categorize request, not “JSON/array.”
"""


def _tx_field_str(tx: dict, key: str) -> str:
  if key not in tx or tx[key] is None:
    return ""
  return str(tx[key])


def _format_currency(amount_raw: str) -> str:
  s = amount_raw.strip().replace("$", "").replace(",", "")
  try:
    v = float(s)
  except ValueError:
    return f"${amount_raw.strip()}"
  sign = "-" if v < 0 else ""
  return f"{sign}${abs(v):.2f}"


def _format_date_long(date_raw: str) -> str:
  if not date_raw or not date_raw.strip():
    return "date not given"
  ds = date_raw.strip()
  try:
    d = datetime.strptime(ds, "%Y-%m-%d")
  except ValueError:
    return ds
  return f"{d.strftime('%B')} {d.day}, {d.year}"


def _format_transaction_sentence(tx: dict) -> str:
  tid_raw = _tx_field_str(tx, "transaction_id").replace("\n", " ").replace("\r", " ").strip()
  tid_slot = tid_raw if tid_raw else "not given"

  name_raw = _tx_field_str(tx, "name").replace("\n", " ").replace("\r", " ").replace("\t", " ")
  payee = name_raw.strip()
  payee_words = f"{payee} " if payee else "(payee not given) "

  amt_raw = _tx_field_str(tx, "amount").strip()
  if amt_raw:
    lead = f"{_format_currency(amt_raw)} {payee_words}transaction on "
  else:
    lead = f"Amount not given, {payee_words}transaction on "

  date_slot = _format_date_long(_tx_field_str(tx, "date"))
  acct_raw = _tx_field_str(tx, "account_id").strip()
  acct_slot = acct_raw if acct_raw else "not given"

  return f"- {lead}{date_slot} (ID {tid_slot}, Account {acct_slot})."


def format_transactions_compact(transactions: list, *, max_visible: int = 3) -> str:
  """
  One markdown bullet per transaction: `$5.00 Payee transaction on Month D, YYYY (ID x, Account y).`
  If len(transactions) > max_visible, append (+N more transactions.)
  """
  if not transactions:
    return "None"
  n = len(transactions)
  chunk = transactions[:max_visible]
  parts = [_format_transaction_sentence(tx) for tx in chunk]
  body = "\n".join(parts)
  if n > max_visible:
    return f"{body}\n(+{n - max_visible} more transactions.)"
  return body


def format_categorization_verify_input(categorize_request: str | None, transactions: list) -> str:
  """Build user message: # Categorize Request prose + # Transactions compact string."""
  request_body = "" if categorize_request is None else categorize_request
  transactions_body = format_transactions_compact(transactions)
  return (
    "# Categorize Request\n\n"
    f"{request_body}\n\n"
    "# Transactions\n\n"
    f"{transactions_body}"
  )


TEST_CASES = [
  {
    "name": "all_rules_satisfied_simple",
    "input": """# Categorize Request

Re-categorize all transactions from 'Starbucks' where each charge was $50 or less on or after 2025-11-01, setting their category to 'meals_dining_out'.

# Transactions

- $12.50 Starbucks transaction on November 10, 2025 (ID 1, Account 20).
- $4.75 Starbucks transaction on November 12, 2025 (ID 2, Account 20).""",
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

- $90.00 Costco transaction on October 15, 2025 (ID 10, Account 20).
- $220.00 Costco transaction on October 20, 2025 (ID 11, Account 20).""",
    "output": {
      "rules_satisfied": False,
      "notes": "Not every Costco transaction has a charge of at least $100 as the groceries rule requires.",
    },
  },
  {
    "name": "unevaluable_rule_due_to_missing_fields",
    "input": """# Categorize Request

Re-categorize transactions as 'income_salary' where the name includes 'payroll' and the amount is at least $1000 for every listed row (transaction_id 21 and 22).

# Transactions

- $1440.00 CA State Payroll transaction on November 18, 2025 (ID 21, Account 20).
- Amount not given, ACME Corp transaction on November 20, 2025 (ID 22, Account 20).""",
    "output": {
      "rules_satisfied": False,
      "notes": "Cannot evaluate: one transaction has no amount for the payroll and at-least-$1000 charge rules.",
    },
  },
  {
    "name": "malformed_input_missing_rules",
    "input": """# Categorize Request


# Transactions

- $2000.00 Apartments LLC transaction on November 18, 2025 (ID 31, Account 20).""",
    "output": {
      "rules_satisfied": False,
      "notes": "Malformed input: missing verifiable filters.",
    },
  },
  {
    "name": "name_eq_requires_exact_match",
    "input": """# Categorize Request

Categorize transaction_id 41 and transaction_id 42 as 'meals_dining_out'; for each row the name must be exactly 'Starbucks' (not 'Starbucks Coffee' or any longer string).

# Transactions

- $5.00 Starbucks transaction on November 5, 2025 (ID 41, Account 20).
- $6.00 Starbucks Coffee transaction on November 6, 2025 (ID 42, Account 20).""",
    "output": {
      "rules_satisfied": False,
      "notes": "transaction_id 42 fails: name is Starbucks Coffee, not exactly Starbucks (no longer strings).",
    },
  },
  {
    "name": "date_eq_enforced",
    "input": """# Categorize Request

Re-categorize all transactions from 'Whole Foods' as 'groceries' where the transaction date is exactly 2025-11-26 for each matching charge.

# Transactions

- $45.00 Whole Foods transaction on November 26, 2025 (ID 51, Account 20).
- $12.00 Whole Foods transaction on November 25, 2025 (ID 52, Account 20).""",
    "output": {
      "rules_satisfied": False,
      "notes": "A matching Whole Foods charge is not on transaction date 2025-11-26 as required.",
    },
  },
  {
    "name": "amount_eq_enforced",
    "input": """# Categorize Request

Categorize the Netflix subscription charges (transaction_id 61 and 62) as 'subscriptions_entertainment' where each amount is exactly $15.99.

# Transactions

- $15.99 Netflix transaction on November 3, 2025 (ID 61, Account 20).
- $16.99 Netflix transaction on December 3, 2025 (ID 62, Account 20).""",
    "output": {
      "rules_satisfied": False,
      "notes": "Netflix charge ID 62 is $16.99, not exactly $15.99 as required for every row.",
    },
  },
  {
    "name": "account_id_eq_enforced",
    "input": """# Categorize Request

Re-categorize all transactions from 'Amazon' as 'shopping_clothing' where each charge posts to account_id 20 only. Also, create a rule for future similar transactions.

# Transactions

- $25.00 AMAZON transaction on November 2, 2025 (ID 71, Account 20).
- $30.00 Amazon transaction on November 4, 2025 (ID 72, Account 21).""",
    "output": {
      "rules_satisfied": False,
      "notes": "One Amazon charge posts to an account other than 20, conflicting with account_id equal to 20.",
    },
  },
  {
    "name": "case_insensitive_name_contains",
    "input": """# Categorize Request

Re-categorize the Amazon marketplace transactions (transaction_id 81 and 82) as 'shopping_clothing'; the name should contain 'amazon' and matching must be case-insensitive.

# Transactions

- $12.00 AMAZON MARKETPLACE transaction on November 1, 2025 (ID 81, Account 20).
- $9.00 Amazon.com transaction on November 2, 2025 (ID 82, Account 20).""",
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

- $40.00 Shell transaction on November 15, 2025 (ID 91, Account 20).
- $35.00 Shell transaction on December 1, 2025 (ID 92, Account 20).""",
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

- $75.00 Costco transaction on November 8, 2025 (ID 101, Account 20).
- $250.00 Costco transaction on November 9, 2025 (ID 102, Account 20).""",
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
    "name": "missing_name_required_by_rules",
    "input": """# Categorize Request

Re-categorize these trips as 'transport_public_transit' where the name contains 'Uber' and the date is on or after 2025-11-01.

# Transactions

- $18.00 Uber *TRIP transaction on November 10, 2025 (ID 201, Account 20).
- $22.00 (payee not given) transaction on November 12, 2025 (ID 202, Account 20).""",
    "output": {
      "rules_satisfied": False,
      "notes": "Cannot evaluate: one trip has no name for the Uber substring and date rules.",
    },
  },
  {
    "name": "missing_account_id_required_by_rules",
    "input": """# Categorize Request

Re-categorize these housing payments as 'shelter_home' where the name contains 'RentCo' and account_id is exactly 99.

# Transactions

- $2100.00 RentCo Apartments transaction on November 1, 2025 (ID 301, Account 99).
- $2100.00 RentCo Apartments transaction on December 1, 2025 (ID 302, Account not given).""",
    "output": {
      "rules_satisfied": False,
      "notes": "Cannot evaluate: one housing payment has no account though the rules require account 99.",
    },
  },
  {
    "name": "truncated_transactions_more_than_three",
    "input": """# Categorize Request

Re-categorize all transactions from 'Starbucks' where each charge was $50 or less on or after 2025-11-01, setting their category to 'meals_dining_out'.

# Transactions

- $10.00 Starbucks transaction on November 5, 2025 (ID 401, Account 1).
- $20.00 Starbucks transaction on November 6, 2025 (ID 402, Account 1).
- $15.00 Starbucks transaction on November 7, 2025 (ID 403, Account 1).
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

- $55.00 ACME|UK Retail transaction on August 1, 2025 (ID 601, Account 20).""",
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
    "tests": [2, 11, 12, 13],
  },
  3: {
    "name": "Ranges + case-insensitivity",
    "tests": [8, 9, 10],
  },
  4: {
    "name": "Exact/equality keys",
    "tests": [4, 5, 6, 7],
  },
  5: {
    "name": "Truncated transactions (>3 shown as compact + more note)",
    "tests": [14, 15],
  },
}


class UpdateTransactionCategoryVerifyOptimizer:
  """Gemini client for categorization rule verification.

  Recommended defaults (batches 1–3 in this repo were last validated with these; re-run if you change SCHEMA/SYSTEM_PROMPT):
  - `gemini-flash-lite-latest`: minimum model/cost; step up to `gemini-2.0-flash` if booleans or notes drift.
  - `thinking_budget=0`: structured JSON; extra thinking rarely helps alignment.
  - `temperature=0`, `top_p=1`: stable `rules_satisfied` and matching `notes`.
  - `max_output_tokens=128`; raise to 160–256 only on truncated JSON.
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
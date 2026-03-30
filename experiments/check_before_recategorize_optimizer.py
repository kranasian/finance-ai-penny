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
        "true iff: ≥1 transaction line parses; inferred filters are one valid object (not missing, not array-as-rules); "
        "each line has fields required by active filters (placeholders like (amount not given) count as missing); "
        "each line passes every active filter. Otherwise false."
      ),
    ),
    "notes": types.Schema(
      type=types.Type.STRING,
      description=(
        "One line, 8–140 chars, no newlines. The decisive reason for rules_satisfied—must agree (success wording ↔ true, "
        "failure/cannot-evaluate ↔ false). Match # Categorize Request vocabulary (transaction, charge, name); avoid "
        "wire labels like **Name**. Backticks only for filter keys if needed. No list bullets or meta-commentary."
      ),
      min_length=8,
      max_length=140,
      pattern=r"^[^\n]+$",
    ),
  },
)


SYSTEM_PROMPT = """You are a verifier. From # Categorize Request and # Transactions, decide if every listed transaction satisfies the inferred rules. Reply with JSON only (schema enforced).

## Lockstep output (highest priority)
- Pick one boolean verdict. Set `rules_satisfied` to that verdict.
- `notes` must describe **the same** verdict in one short sentence: if the verdict is failure or cannot evaluate, `notes` must sound like failure; if success, `notes` must sound like success. Before sending, mentally check they agree.

## Procedure
1. Read `# Categorize Request` prose. Ignore target category names (e.g. meals_dining_out); extract only constraints that map to filter keys below (name, amount, date, account_id, transaction_id).
2. If the request body is empty/whitespace, or you cannot map any constraint to those keys, or the only structured rules are a JSON array instead of one object → filters missing/malformed → `rules_satisfied` false.
3. Read `# Transactions`. If body is exactly `None`, or after skipping blank lines there are zero data lines → `rules_satisfied` false.
4. Each data line = one markdown bullet, one physical line: `- **Transaction id**: … **Name**: … **Amount**: … **Date**: … **Account id**: …` (colon after closing `**`; single spaces between clauses). Missing values use exactly `(name not given)`, `(amount not given)`, `(date not given)`, `(account not given)`. **Name** text is between `**Name**: ` and ` **Amount**:`; it may include `|`, tab, commas. Wrong shape → line unparsable → `rules_satisfied` false. Optional last line `(+N more transactions.)` is not data; do not invent hidden rows.
5. For each parsed line, AND every active filter. Compare amounts and dates numerically / ISO string order as specified. `name_contains` / `name_eq` are case-insensitive; `name_eq` is full string equality of the **Name** value, not a prefix of a longer name.
6. If any line lacks a field required by an active filter, or any line violates a filter → `rules_satisfied` false. Else true.

## Filter keys (all active keys must pass on every line)
- `name_contains`, `name_eq`
- `date_greater_than_or_equal_to`, `date_less_than_or_equal_to`, `date_eq`
- `amount_greater_than_or_equal_to`, `amount_less_than_or_equal_to`, `amount_eq`
- `account_id_eq`

## `notes` style
- Same natural language as the categorize request; reuse their words (charge, transaction, name, account) where natural.
- Be brief: one reason, no preamble, under 140 characters.
- For malformed-input cases only, you may use these compact shapes (adapt wording to the request if needed): missing filters → mention missing/unverifiable filters; rules as array → object required not array; missing field → name the missing `amount`/`name`/etc. in backticks when clear.
"""


_TRANSACTION_ROW_KEYS = ("transaction_id", "name", "amount", "date", "account_id")

_MISSING_NAME = "(name not given)"
_MISSING_AMOUNT = "(amount not given)"
_MISSING_DATE = "(date not given)"
_MISSING_ACCOUNT = "(account not given)"


def _tx_field_str(tx: dict, key: str) -> str:
  if key not in tx or tx[key] is None:
    return ""
  return str(tx[key])


def _format_transaction_markdown(tx: dict) -> str:
  tid = _tx_field_str(tx, "transaction_id").replace("\n", " ").replace("\r", " ").strip()
  if not tid:
    tid = "(transaction id not given)"
  name_raw = _tx_field_str(tx, "name").replace("\n", " ").replace("\r", " ")
  name_part = name_raw.strip() if name_raw.strip() else _MISSING_NAME
  amt_raw = _tx_field_str(tx, "amount").strip()
  amt_part = amt_raw if amt_raw else _MISSING_AMOUNT
  date_raw = _tx_field_str(tx, "date").strip()
  date_part = date_raw if date_raw else _MISSING_DATE
  acct_raw = _tx_field_str(tx, "account_id").strip()
  acct_part = acct_raw if acct_raw else _MISSING_ACCOUNT
  return (
    f"- **Transaction id**: {tid} **Name**: {name_part} **Amount**: {amt_part} "
    f"**Date**: {date_part} **Account id**: {acct_part}"
  )


def format_transactions_compact(transactions: list, *, max_visible: int = 3) -> str:
  """
  One markdown list line per transaction (bold labels, single-space-separated clauses). | and tab may appear in the name.
  If len(transactions) > max_visible, append (+N more transactions.)
  """
  if not transactions:
    return "None"
  n = len(transactions)
  chunk = transactions[:max_visible]
  parts = [_format_transaction_markdown(tx) for tx in chunk]
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

- **Transaction id**: 1 **Name**: Starbucks **Amount**: 12.5 **Date**: 2025-11-10 **Account id**: 20
- **Transaction id**: 2 **Name**: Starbucks **Amount**: 4.75 **Date**: 2025-11-12 **Account id**: 20""",
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

- **Transaction id**: 10 **Name**: Costco **Amount**: 90 **Date**: 2025-10-15 **Account id**: 20
- **Transaction id**: 11 **Name**: Costco **Amount**: 220 **Date**: 2025-10-20 **Account id**: 20""",
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

- **Transaction id**: 21 **Name**: CA State Payroll **Amount**: 1440 **Date**: 2025-11-18 **Account id**: 20
- **Transaction id**: 22 **Name**: ACME Corp **Amount**: (amount not given) **Date**: 2025-11-20 **Account id**: 20""",
    "output": {
      "rules_satisfied": False,
      "notes": "Cannot evaluate: one transaction has no amount for the payroll and at-least-$1000 charge rules.",
    },
  },
  {
    "name": "malformed_input_missing_rules",
    "input": """# Categorize Request


# Transactions

- **Transaction id**: 31 **Name**: Apartments LLC **Amount**: 2000 **Date**: 2025-11-18 **Account id**: 20""",
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

- **Transaction id**: 41 **Name**: Starbucks **Amount**: 5.0 **Date**: 2025-11-05 **Account id**: 20
- **Transaction id**: 42 **Name**: Starbucks Coffee **Amount**: 6.0 **Date**: 2025-11-06 **Account id**: 20""",
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

- **Transaction id**: 51 **Name**: Whole Foods **Amount**: 45.0 **Date**: 2025-11-26 **Account id**: 20
- **Transaction id**: 52 **Name**: Whole Foods **Amount**: 12.0 **Date**: 2025-11-25 **Account id**: 20""",
    "output": {
      "rules_satisfied": False,
      "notes": "A matching Whole Foods charge is not on transaction date 2025-11-26 (`date_eq`).",
    },
  },
  {
    "name": "amount_eq_enforced",
    "input": """# Categorize Request

Categorize the Netflix subscription charges (transaction_id 61 and 62) as 'subscriptions_entertainment' where each amount is exactly $15.99.

# Transactions

- **Transaction id**: 61 **Name**: Netflix **Amount**: 15.99 **Date**: 2025-11-03 **Account id**: 20
- **Transaction id**: 62 **Name**: Netflix **Amount**: 16.99 **Date**: 2025-12-03 **Account id**: 20""",
    "output": {
      "rules_satisfied": False,
      "notes": "A Netflix subscription charge is not exactly $15.99 as required (`amount_eq`).",
    },
  },
  {
    "name": "account_id_eq_enforced",
    "input": """# Categorize Request

Re-categorize all transactions from 'Amazon' as 'shopping_online' where each charge posts to account_id 20 only. Also, create a rule for future similar transactions.

# Transactions

- **Transaction id**: 71 **Name**: AMAZON **Amount**: 25.0 **Date**: 2025-11-02 **Account id**: 20
- **Transaction id**: 72 **Name**: Amazon **Amount**: 30.0 **Date**: 2025-11-04 **Account id**: 21""",
    "output": {
      "rules_satisfied": False,
      "notes": "One Amazon charge posts to an account other than 20, conflicting with `account_id_eq`.",
    },
  },
  {
    "name": "case_insensitive_name_contains",
    "input": """# Categorize Request

Re-categorize the Amazon marketplace transactions (transaction_id 81 and 82) as 'shopping_online'; the name should contain 'amazon' and matching must be case-insensitive.

# Transactions

- **Transaction id**: 81 **Name**: AMAZON MARKETPLACE **Amount**: 12.0 **Date**: 2025-11-01 **Account id**: 20
- **Transaction id**: 82 **Name**: Amazon.com **Amount**: 9.0 **Date**: 2025-11-02 **Account id**: 20""",
    "output": {
      "rules_satisfied": True,
      "notes": "Both marketplace transactions meet case-insensitive name matching and the other rules.",
    },
  },
  {
    "name": "date_range_lte_gte_combined",
    "input": """# Categorize Request

Re-categorize all 'Shell' fuel transactions as 'transport_gas' where the charge date falls from 2025-11-01 through 2025-11-30 inclusive.

# Transactions

- **Transaction id**: 91 **Name**: Shell **Amount**: 40.0 **Date**: 2025-11-15 **Account id**: 20
- **Transaction id**: 92 **Name**: Shell **Amount**: 35.0 **Date**: 2025-12-01 **Account id**: 20""",
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

- **Transaction id**: 101 **Name**: Costco **Amount**: 75.0 **Date**: 2025-11-08 **Account id**: 20
- **Transaction id**: 102 **Name**: Costco **Amount**: 250.0 **Date**: 2025-11-09 **Account id**: 20""",
    "output": {
      "rules_satisfied": False,
      "notes": "One Costco charge amount is outside the $50–$200 inclusive groceries range.",
    },
  },
  {
    "name": "malformed_rules_not_dict",
    "input": """# Categorize Request

Categorize the Starbucks transaction (transaction_id 111) as 'meals_dining_out'. Technical rules payload is a JSON array only: ["name_contains: starbucks"]

# Transactions

- **Transaction id**: 111 **Name**: Starbucks **Amount**: 5.0 **Date**: 2025-11-01 **Account id**: 20""",
    "output": {
      "rules_satisfied": False,
      "notes": "Malformed input: structured filters must be a JSON object, not an array.",
    },
  },
  {
    "name": "empty_transactions_array",
    "input": """# Categorize Request

Re-categorize all 'Lyft' charges as 'transport_rideshare' where each amount is at least $10. Also, create a rule for future similar transactions.

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

Re-categorize these trips as 'transport_rideshare' where the name contains 'Uber' and the date is on or after 2025-11-01.

# Transactions

- **Transaction id**: 201 **Name**: Uber *TRIP **Amount**: 18.0 **Date**: 2025-11-10 **Account id**: 20
- **Transaction id**: 202 **Name**: (name not given) **Amount**: 22.0 **Date**: 2025-11-12 **Account id**: 20""",
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

- **Transaction id**: 301 **Name**: RentCo Apartments **Amount**: 2100.0 **Date**: 2025-11-01 **Account id**: 99
- **Transaction id**: 302 **Name**: RentCo Apartments **Amount**: 2100.0 **Date**: 2025-12-01 **Account id**: (account not given)""",
    "output": {
      "rules_satisfied": False,
      "notes": "Cannot evaluate: one housing payment lacks `account_id` though the rules require account 99.",
    },
  },
  {
    "name": "truncated_transactions_more_than_three",
    "input": """# Categorize Request

Re-categorize all transactions from 'Starbucks' where each charge was $50 or less on or after 2025-11-01, setting their category to 'meals_dining_out'.

# Transactions

- **Transaction id**: 401 **Name**: Starbucks **Amount**: 10.0 **Date**: 2025-11-05 **Account id**: 1
- **Transaction id**: 402 **Name**: Starbucks **Amount**: 20.0 **Date**: 2025-11-06 **Account id**: 1
- **Transaction id**: 403 **Name**: Starbucks **Amount**: 15.0 **Date**: 2025-11-07 **Account id**: 1
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

- **Transaction id**: 601 **Name**: ACME|UK\tRetail **Amount**: 55.0 **Date**: 2025-08-01 **Account id**: 20""",
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
  5: {
    "name": "Truncated transactions (>3 shown as compact + more note)",
    "tests": [15, 16],
  },
}


class UpdateTransactionCategoryVerifyOptimizer:
  """Gemini client for categorization rule verification.

  Generation defaults (cost/quality tradeoff for tiny JSON + short `notes`):
  - Model `gemini-flash-lite-latest`: smallest Flash-Lite alias; use `gemini-2.0-flash` if you need more accuracy.
  - `thinking_budget=0`: no thinking block—faster, cheaper; schema-grounded JSON does not benefit much from extra reasoning.
  - `temperature=0`, `top_p=1`: deterministic sampling, best alignment between `rules_satisfied` and `notes`.
  - `max_output_tokens=128`: enough for this schema (~40 tokens typical); raise only if you see truncation.
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